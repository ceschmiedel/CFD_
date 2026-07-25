/* ── src/core/emit/wgsl.js ───────────────────────────────────────────────────
 *
 * O dialeto WGSL: traduz os blocos de ir.js para compute shaders do WebGPU.
 *
 * Este arquivo não tem física. Ele sabe declarar uma variável, endereçar uma
 * população e escrever uma população em WGSL — e mais nada. Se você está
 * procurando a colisão TRT ou o modelo de Smagorinsky, eles estão em ir.js,
 * uma vez só, compartilhados com o backend WebGL2.
 *
 *
 * LAYOUT DE MEMÓRIA
 * -----------------
 * Structure-of-arrays, direção-maior:
 *
 *     pop[i][cell]   com cell = z*nx*ny + y*nx + x
 *
 * A célula varia com x na dimensão mais rápida, e o workgroup é despachado ao
 * longo de x. Assim as 64 invocações de um workgroup leem 64 floats
 * contíguos de cada uma das dezenove direções: dezenove leituras perfeitamente
 * coalescidas por passo, que é o melhor que um kernel limitado por banda pode
 * fazer.
 *
 * A alternativa óbvia — array-of-structures, `pop[cell*19 + i]` — parece mais
 * natural e é bem mais lenta: cada invocação passaria a puxar 19 floats
 * espalhados com passo 76 bytes, e a GPU buscaria uma linha de cache inteira
 * para usar 4 bytes dela.
 *
 * As Q direções são fatiadas entre `nbuf` storage buffers porque um adaptador
 * limita maxStorageBufferBindingSize (tipicamente 2 GiB) e um lattice grande
 * passa disso. Como o índice da direção é constante em tempo de GERAÇÃO, a
 * escolha do buffer e o deslocamento dentro dele são resolvidos aqui e viram
 * literais: o shader nunca faz uma indireção para descobrir onde uma população
 * mora.
 */

import {
  Q, C, W, OPP, CELULA, num,
  emitirPasso, blocoEquilibrio,
} from './ir.js';

export const TIPO = { ...CELULA };

/** Como as Q direções se distribuem entre `nbuf` buffers. */
export function planoDeBuffers(nbuf) {
  const porBuffer = Math.ceil(Q / nbuf);
  return Array.from({ length: Q }, (_, i) => ({
    dir: i,
    buffer: Math.floor(i / porBuffer),
    offset: (i % porBuffer),
  }));
}

/**
 * Constrói o dialeto WGSL para um plano de buffers.
 *
 * @param {object[]} plano  saída de planoDeBuffers
 * @param {string} prefSrc  prefixo dos buffers de leitura
 * @param {string} prefDst  prefixo dos buffers de escrita
 */
function dialeto(plano, prefSrc = 'src', prefDst = 'dst') {
  const end = (pref, i, idx) => {
    const { buffer, offset } = plano[i];
    /* offset*N é o início da fatia daquela direção dentro do buffer */
    return offset === 0
      ? `${pref}${buffer}[${idx}]`
      : `${pref}${buffer}[${offset}u * N + ${idx}]`;
  };

  return {
    comentario: (s) => `// ${s}`,
    letF32: (n, e) => `let ${n} = ${e};`,
    letVec3: (n, e) => `let ${n} = ${e};`,
    vec3: (a, b, c) => `vec3<f32>(${a}, ${b}, ${c})`,
    indentar: (s) => (s ? '  ' + s : s),

    lerPop: (i, idx) => end(prefSrc, i, idx),
    lerPopDst: (i) => end(prefDst, i, 'cell'),
    setPop: (i, e) => `${end(prefDst, i, 'cell')} = ${e};`,

    /*
     * Índice do vizinho, montado a partir de coordenadas JÁ envolvidas.
     *
     * A versão anterior chamava uma função `viz(pos, d)` que fazia
     * `(p + d + n) % n`. Correto e devastador: são três módulos inteiros por
     * direção, dezenove direções, cinquenta e sete divisões inteiras por
     * thread. Divisão inteira custa dezenas de ciclos numa GPU, e num kernel
     * que deveria estar limitado por banda de memória ela passou a ser o
     * gargalo — o solver rodava a uma fração da banda disponível.
     *
     * Como |c_i| <= 1, o envolvimento tem só três casos por eixo, e os três
     * são calculados uma vez no cabeçalho (xm, x0, xp e análogos). Cada
     * direção vira aritmética de índice pura. Seis `select` no total,
     * nenhuma divisão.
     */
    vizinho: (dx, dy, dz) => {
      if (dx === 0 && dy === 0 && dz === 0) return 'cell';
      const eixo = (d, n) => (d < 0 ? `${n}m` : d > 0 ? `${n}p` : `${n}0`);
      return `(${eixo(dz, 'z')} * nxny + ${eixo(dy, 'y')} * P.dim.x + ${eixo(dx, 'x')})`;
    },

    letU32: (n, e) => `let ${n} = ${e};`,
    /* O endereço do vizinho. Aqui é o mesmo `let` de sempre — o tipo é
     * inferido; o dialeto GLSL precisa escrevê-lo, e é por isso que o IR pede
     * este método em vez de reaproveitar letF32. */
    letIdx: (n, e) => `let ${n} = ${e};`,
    ehTipo: (nome) => `ct == ${CELULA[nome]}u`,
    ehFluidoEm: (idx) => `tipo[${idx}] == ${CELULA.FLUIDO}u`,
    tipoEm: (idx) => `tipo[${idx}]`,
    ehSolidoTipo: (t) =>
      `(${t} == ${CELULA.SOLIDO}u || ${t} == ${CELULA.SOLIDO_MOVEL}u ` +
      `|| ${t} == ${CELULA.PAREDE}u)`,
    velParedeDe: (t) =>
      `select(vec3<f32>(0.0), P.beltU.xyz, ${t} == ${CELULA.SOLIDO_MOVEL}u)`,
    se: (cond) => [`if (${cond}) {`],
    senaoSe: (cond) => [`} else if (${cond}) {`],
    senao: () => ['} else {'],
    fimSe: () => ['}'],
  };
}

/* ──────────────────────────────────────────────────────────────── preâmbulo */

function declaracoes(nbuf, comMacros) {
  const L = [];
  L.push('struct Params {');
  L.push('  dim: vec4<u32>,        // nx, ny, nz, _');
  L.push('  omegaPlus: f32,        // taxa de relaxação par (viscosidade molecular)');
  L.push('  magic: f32,            // Lambda TRT = (1/w+ - 1/2)(1/w- - 1/2)');
  L.push('  lesCs: f32,            // constante de Smagorinsky; 0.0 desliga');
  L.push('  _pad: f32,');
  L.push('  beltU: vec4<f32>,      // velocidade tangencial do piso (esteira)');
  L.push('  inletU: vec4<f32>,     // corrente livre, unidades de lattice');
  L.push('  sponge: vec4<f32>,     // início, comprimento, espessura da CL da entrada, _');
  L.push('};');
  L.push('');
  L.push('@group(0) @binding(0) var<uniform> P: Params;');

  let b = 1;
  for (let i = 0; i < nbuf; i++) {
    L.push(`@group(0) @binding(${b++}) var<storage, read> src${i}: array<f32>;`);
  }
  for (let i = 0; i < nbuf; i++) {
    L.push(`@group(0) @binding(${b++}) var<storage, read_write> dst${i}: array<f32>;`);
  }
  L.push(`@group(0) @binding(${b++}) var<storage, read> tipo: array<u32>;`);
  if (comMacros) {
    L.push(`@group(0) @binding(${b++}) var<storage, read_write> macros: array<vec4<f32>>;`);
  }

  return L;
}

/**
 * Coordenadas envolvidas dos três vizinhos possíveis em cada eixo.
 *
 * Emitidas uma vez por invocação e reaproveitadas pelas dezenove direções.
 * Ver o comentário em `vizinho` no dialeto para por que isto não é um módulo.
 */
function envolvimento() {
  const L = [];
  L.push('  let nxny = P.dim.x * P.dim.y;');
  L.push('  // Vizinhos por eixo, com envolvimento PERIÓDICO.');
  L.push('  //');
  L.push('  // O envolvimento periódico é a escolha certa porque a condição de');
  L.push('  // contorno de parede não depende mais dele: com bounce-back no nó de');
  L.push('  // fluido (ver blocoPull em ir.js), uma célula ao lado do piso jamais');
  L.push('  // lê o piso — ela devolve a si mesma o que mandou. O que houver do');
  L.push('  // outro lado do domínio é irrelevante para ela.');
  L.push('  //');
  L.push('  // Houve uma tentativa de grampear em vez de envolver, para impedir que');
  L.push('  // o piso puxasse do teto. Ela custou o teste de viscosidade: a onda de');
  L.push('  // cisalhamento é medida num domínio PERIÓDICO, e grampear quebra a');
  L.push('  // periodicidade em y — a viscosidade medida caiu de 0,3% de erro para');
  L.push('  // 15 a 53%. O problema do piso era real, mas o grampeamento tratava o');
  L.push('  // sintoma; a causa era a reflexão acontecer no nó errado.');
  for (const [n, dim] of [['x', 'P.dim.x'], ['y', 'P.dim.y'], ['z', 'P.dim.z']]) {
    L.push(`  let ${n}0 = gid.${n};`);
    L.push(`  let ${n}m = select(gid.${n} - 1u, ${dim} - 1u, gid.${n} == 0u);`);
    L.push(`  let ${n}p = select(gid.${n} + 1u, 0u, gid.${n} == ${dim} - 1u);`);
  }
  return L;
}

function cabecalhoKernel(comMacros) {
  const L = [];
  L.push('@compute @workgroup_size(64, 1, 1)');
  L.push('fn main(@builtin(global_invocation_id) gid: vec3<u32>) {');
  L.push('  if (gid.x >= P.dim.x || gid.y >= P.dim.y || gid.z >= P.dim.z) { return; }');
  L.push('');
  L.push('  let N = P.dim.x * P.dim.y * P.dim.z;');
  L.push(...envolvimento());
  L.push('  let cell = z0 * nxny + y0 * P.dim.x + x0;');
  L.push('');
  L.push('  let ct = tipo[cell];');
  L.push('');
  L.push('  // Célula sólida não é simulada. Com o bounce-back acontecendo no nó');
  L.push('  // de fluido, nada nunca lê as populações de dentro de um sólido — não');
  L.push('  // há estado ali para manter, nem interior de corpo para divergir, nem');
  L.push('  // laço para se realimentar. Sair cedo economiza a colisão inteira em');
  L.push('  // cada célula de casca, piso e parede.');
  L.push(`  if (ct == ${TIPO.SOLIDO}u || ct == ${TIPO.SOLIDO_MOVEL}u || ct == ${TIPO.PAREDE}u) {`);
  if (comMacros) L.push('    macros[cell] = vec4<f32>(0.0);');
  L.push('    return;');
  L.push('  }');
  L.push('');
  L.push('  let uOmegaPlus = P.omegaPlus;');
  L.push('  let uMagic = P.magic;');
  L.push('  let uLesCs = P.lesCs;');
  L.push('  let uSpongeStart = P.sponge.x;');
  L.push('  let uSpongeLen = P.sponge.y;');
  L.push('  let uInf = P.inletU.xyz;');
  L.push('');
  L.push('  // Perfil de camada limite na entrada, lei de potência 1/7.');
  L.push('  // Com esteira rolante uInletBL é 0 e isto vira escoamento uniforme,');
  L.push('  // que é solução EXATA do lattice — a única condição em que');
  L.push('  // omega = 1.98 é estável (ver units.js).');
  L.push('  let uInletBL = max(P.sponge.z, 1.0);');
  L.push('  let zc = f32(gid.z) + 0.5;');
  L.push('  let blf = select(1.0, pow(zc / uInletBL, 0.1428571429), zc < P.sponge.z);');
  L.push('  let uEntrada = P.inletU.xyz * blf;');
  return L;
}

/* ──────────────────────────────────────────────────────────────── os kernels */

/**
 * Kernel do passo: streaming + colisão, ou reflexão se a célula é sólida.
 * Este é o único kernel do laço quente.
 */
export function shaderPasso({ nbuf = 1, escreverMacros = true } = {}) {
  const plano = planoDeBuffers(nbuf);
  const d = dialeto(plano);

  const L = [];
  L.push('/* GERADO por src/core/emit/wgsl.js a partir de src/core/emit/ir.js.');
  L.push(' * Não edite: edite a física em ir.js e os dois backends acompanham. */');
  L.push('');
  L.push(...declaracoes(nbuf, escreverMacros));
  L.push('');
  L.push(...cabecalhoKernel(escreverMacros));
  L.push('');
  L.push(...emitirPasso(d).map(s => (s ? '  ' + s : s)));

  if (escreverMacros) {
    L.push('');
    L.push('  // Campo macroscópico para a renderização ler sem refazer a soma.');
    L.push('  //');
    L.push('  // As células sólidas já saíram no topo do kernel escrevendo zero,');
    L.push('  // então o que chega aqui é fluido. Zerar o sólido importa: escrever');
    L.push('  // a densidade de dentro de um corpo envenena tudo que lê este');
    L.push('  // buffer — as partículas ganham posições absurdas, e um diagnóstico');
    L.push('  // que varra o domínio acusa divergência olhando para células que');
    L.push('  // nunca foram escoamento. (|u| = 3,02 dentro do carro enquanto o');
    L.push('  // fluido estava em 0,0707: um dia inteiro de depuração no rumo errado.)');
    L.push('  macros[cell] = vec4<f32>(u, delta);');
  }
  L.push('}');
  return L.join('\n');
}

/**
 * Kernel de inicialização: preenche com o equilíbrio da corrente livre.
 *
 * Um buffer WebGPU nasce zerado, e g_i = 0 já é um estado válido — fluido
 * parado com rho = 1 exatamente. Mas partir do repouso e esperar o túnel
 * encher custa uma travessia inteira do domínio; partir do escoamento
 * uniforme corta esse transiente quase todo.
 */
export function shaderInit({ nbuf = 1 } = {}) {
  const plano = planoDeBuffers(nbuf);
  const d = dialeto(plano, 'src', 'dst');

  const L = [];
  L.push('/* GERADO por src/core/emit/wgsl.js */');
  L.push('');
  L.push(...declaracoes(nbuf, false));
  L.push('');
  L.push('@compute @workgroup_size(64, 1, 1)');
  L.push('fn main(@builtin(global_invocation_id) gid: vec3<u32>) {');
  L.push('  if (gid.x >= P.dim.x || gid.y >= P.dim.y || gid.z >= P.dim.z) { return; }');
  L.push('  let N = P.dim.x * P.dim.y * P.dim.z;');
  L.push('  let cell = gid.z * P.dim.x * P.dim.y + gid.y * P.dim.x + gid.x;');
  L.push('');
  L.push('  let ct = tipo[cell];');
  L.push(`  let parado = (ct == ${TIPO.SOLIDO}u);`);
  L.push('  // O piso da esteira parte com a velocidade da esteira, não parado:');
  L.push('  // partir com ele parado cria um degrau de cisalhamento no primeiro');
  L.push('  // passo, exatamente onde o bounce-back é menos estável.');
  L.push(`  let esteira = (ct == ${TIPO.SOLIDO_MOVEL}u);`);
  L.push('  var u = P.inletU.xyz;');
  L.push('  if (parado) { u = vec3<f32>(0.0); }');
  L.push('  if (esteira) { u = P.beltU.xyz; }');
  L.push('  let delta = 0.0;');
  L.push('  let rho = 1.0;');
  L.push('  let uu = dot(u, u);');
  L.push('');
  L.push(...blocoEquilibrio(d).map(s => '  ' + s));
  L.push('');
  for (let i = 0; i < Q; i++) L.push('  ' + d.setPop(i, `e${i}`));
  L.push('}');
  return L.join('\n');
}

/**
 * Kernel de forças por TROCA DE MOMENTO.
 *
 * Para cada link que cruza a superfície — nó de fluido x_f com vizinho sólido
 * x_s = x_f + c_i — a quantidade de movimento entregue à parede é
 *
 *     dF = c_i [ f_i(x_f) + f_ī(x_s) ]
 *
 * A população parte com momento c_i f_i e volta com c_ī f_ī = -c_i f_ī; a
 * diferença ficou com o corpo. Somando sobre todos os links sai a força total.
 *
 * ISTO INCLUI O ATRITO VISCOSO. É a razão de o método ser este e não uma
 * integração de pressão sobre a superfície: a integração de pressão precisa de
 * normais reconstruídas a partir de voxels — que numa escada de voxels são
 * ruído — e ignora o cisalhamento, que responde por 10 a 25% do arrasto de um
 * carro. A troca de momento não usa normal nenhuma e entrega as duas parcelas
 * juntas, porque elas são a mesma coisa vista do lattice.
 *
 * O DETALHE QUE SE ERRA: com populações deslocadas, f = g + w, e
 *
 *     c_i [ (g_i + w_i) + (g_ī + w_ī) ] = c_i [ g_i + g_ī + 2 w_i ]
 *
 * O termo 2 w_i c_i NÃO cancela. Ele só cancelaria somado sobre o par oposto
 * inteiro, e um link é um só — o link oposto atravessa a superfície do outro
 * lado do corpo e é uma contribuição legítima e diferente. Esquecer o termo dá
 * um Cd plausível e errado por um fator constante, que é o pior tipo de erro
 * porque sobrevive a toda inspeção visual.
 */
export function shaderForcas({ nbuf = 1 } = {}) {
  const plano = planoDeBuffers(nbuf);
  const d = dialeto(plano);

  const L = [];
  L.push('/* GERADO por src/core/emit/wgsl.js */');
  L.push('');
  L.push(...declaracoes(nbuf, true));
  L.push('@group(0) @binding(' + (3 + 2 * nbuf) + ') var<storage, read_write> parciais: array<vec4<f32>>;');
  L.push('');
  L.push('var<workgroup> sh: array<vec4<f32>, 64>;');
  L.push('');
  L.push('@compute @workgroup_size(64, 1, 1)');
  L.push('fn main(@builtin(global_invocation_id) gid: vec3<u32>,');
  L.push('        @builtin(local_invocation_index) lid: u32,');
  L.push('        @builtin(workgroup_id) wid: vec3<u32>,');
  L.push('        @builtin(num_workgroups) nwg: vec3<u32>) {');
  L.push('  var f = vec3<f32>(0.0, 0.0, 0.0);');
  L.push('  let dentro = gid.x < P.dim.x && gid.y < P.dim.y && gid.z < P.dim.z;');
  L.push('');
  L.push('  if (dentro) {');
  L.push('    let N = P.dim.x * P.dim.y * P.dim.z;');
  L.push(...envolvimento().map(s => '  ' + s));
  L.push('    let cell = z0 * nxny + y0 * P.dim.x + x0;');
  L.push(`    if (tipo[cell] == ${TIPO.FLUIDO}u) {`);

  for (let i = 1; i < Q; i++) {          // i=0 não cruza superfície nenhuma
    const c = C[i];
    const j = OPP[i];
    L.push(`      {`);
    L.push(`        let nb = ${d.vizinho(c[0], c[1], c[2])};`);
    L.push(`        let tb = tipo[nb];`);
    /* SOMENTE o corpo. O piso e as paredes do túnel refletem igual mas não
     * entram na conta — ver o cabeçalho de CELULA em ir.js. */
    L.push(`        if (tb == ${TIPO.SOLIDO}u) {`);
    /* Com bounce-back no nó de fluido, o que volta é exatamente o que saiu
     * (a parede do corpo é estática), então a soma dos dois é o dobro do que
     * saiu — e o kernel não lê nenhuma população dentro do sólido. */
    L.push(`          let q = 2.0 * (${d.lerPop(i, 'cell')} + ${num(W[i])});`);
    const termos = [];
    for (let a = 0; a < 3; a++) {
      if (c[a] === 0) { termos.push('0.0'); continue; }
      termos.push(c[a] > 0 ? 'q' : '-q');
    }
    L.push(`          f += vec3<f32>(${termos.join(', ')});`);
    L.push(`        }`);
    L.push(`      }`);
  }

  L.push('    }');
  L.push('  }');
  L.push('');
  L.push('  // redução em árvore dentro do workgroup: 64 valores viram 1');
  L.push('  sh[lid] = vec4<f32>(f, 0.0);');
  L.push('  workgroupBarrier();');
  for (let s = 32; s > 0; s >>= 1) {
    L.push(`  if (lid < ${s}u) { sh[lid] = sh[lid] + sh[lid + ${s}u]; }`);
    L.push('  workgroupBarrier();');
  }
  L.push('  if (lid == 0u) {');
  L.push('    parciais[wid.x + nwg.x * (wid.y + nwg.y * wid.z)] = sh[0];');
  L.push('  }');
  L.push('}');
  return L.join('\n');
}

/**
 * Segunda passada: soma o vetor de parciais num único vec4.
 *
 * Um workgroup só. O vetor de parciais tem da ordem de 10^5 entradas e cada
 * invocação percorre uma fatia com passo 256, o que mantém a leitura
 * coalescida. A alternativa — atomicAdd em ponto flutuante — não existe em
 * WGSL sem emular com inteiros de ponto fixo, e ponto fixo aqui trocaria um
 * problema resolvido por uma escolha de escala que ninguém saberia justificar.
 */
export function shaderReduzir() {
  const L = [];
  L.push('/* GERADO por src/core/emit/wgsl.js */');
  L.push('');
  L.push('@group(0) @binding(0) var<uniform> nPart: vec4<u32>;');
  L.push('@group(0) @binding(1) var<storage, read> parciais: array<vec4<f32>>;');
  L.push('@group(0) @binding(2) var<storage, read_write> total: array<vec4<f32>>;');
  L.push('');
  L.push('var<workgroup> sh: array<vec4<f32>, 256>;');
  L.push('');
  L.push('@compute @workgroup_size(256, 1, 1)');
  L.push('fn main(@builtin(local_invocation_index) lid: u32) {');
  L.push('  var acc = vec4<f32>(0.0);');
  L.push('  var i = lid;');
  L.push('  loop {');
  L.push('    if (i >= nPart.x) { break; }');
  L.push('    acc = acc + parciais[i];');
  L.push('    i = i + 256u;');
  L.push('  }');
  L.push('  sh[lid] = acc;');
  L.push('  workgroupBarrier();');
  for (let s = 128; s > 0; s >>= 1) {
    L.push(`  if (lid < ${s}u) { sh[lid] = sh[lid] + sh[lid + ${s}u]; }`);
    L.push('  workgroupBarrier();');
  }
  L.push('  if (lid == 0u) { total[0] = sh[0]; }');
  L.push('}');
  return L.join('\n');
}

/**
 * Kernel de macros isolado, para quando se quer ler o campo sem avançar.
 * O laço quente NÃO usa isto — shaderPasso já escreve macros de graça, porque
 * ele acabou de somar as dezenove populações de qualquer jeito.
 */
export function shaderMacros({ nbuf = 1 } = {}) {
  const plano = planoDeBuffers(nbuf);
  const d = dialeto(plano);

  const L = [];
  L.push('/* GERADO por src/core/emit/wgsl.js */');
  L.push('');
  L.push(...declaracoes(nbuf, true));
  L.push('');
  L.push('@compute @workgroup_size(64, 1, 1)');
  L.push('fn main(@builtin(global_invocation_id) gid: vec3<u32>) {');
  L.push('  if (gid.x >= P.dim.x || gid.y >= P.dim.y || gid.z >= P.dim.z) { return; }');
  L.push('  let N = P.dim.x * P.dim.y * P.dim.z;');
  L.push('  let cell = gid.z * P.dim.x * P.dim.y + gid.y * P.dim.x + gid.x;');
  L.push('');
  /* aqui as populações são lidas na própria célula, sem streaming */
  for (let i = 0; i < Q; i++) {
    L.push('  ' + d.letF32(`g${i}`, d.lerPop(i, 'cell')));
  }
  L.push('');
  const soma = Array.from({ length: Q }, (_, i) => `g${i}`).join(' + ');
  L.push(`  let delta = ${soma};`);
  L.push('  let rho = 1.0 + delta;');
  for (let a = 0; a < 3; a++) {
    const termos = [];
    for (let i = 0; i < Q; i++) {
      if (C[i][a] === 0) continue;
      termos.push(C[i][a] > 0 ? `g${i}` : `-g${i}`);
    }
    L.push(`  let m${'xyz'[a]} = ${termos.join(' + ').replace(/\+ -/g, '- ')};`);
  }
  L.push('  macros[cell] = vec4<f32>(vec3<f32>(mx, my, mz) / rho, delta);');
  L.push('}');
  return L.join('\n');
}
