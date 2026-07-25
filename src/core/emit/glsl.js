/* ── src/core/emit/glsl.js ───────────────────────────────────────────────────
 *
 * O dialeto GLSL ES 3.00: traduz os blocos de ir.js para fragment shaders do
 * WebGL2.
 *
 * Como wgsl.js, este arquivo NÃO tem física. Ele sabe endereçar uma célula,
 * ler uma população e escrever uma população em GLSL — e mais nada. A colisão
 * TRT, o Smagorinsky e o bounce-back estão em ir.js, uma vez só, e é de lá que
 * os dois backends saem. Uma discordância entre WebGPU e WebGL2 num caso de
 * validação é, por construção, bug de memória ou de despacho — nunca de
 * modelagem.
 *
 *
 * SEM COMPUTE: O LATTICE VIRA UM ATLAS
 * ------------------------------------
 * WebGL2 não tem compute shader nem storage buffer. O que ele tem é render
 * target de float e MRT, então o kernel do passo vira um FRAGMENT SHADER que
 * desenha um quad cobrindo uma textura, e cada fragmento é uma célula.
 *
 * Um fragment shader escreve em textura 2D, e o lattice é 3D. O domínio é
 * ladrilhado: a fatia z ocupa o retângulo (nx, ny) na posição (z % tx, z / tx)
 * do atlas. `atlasLayout` em caps.js escolhe tx para o atlas ficar o mais
 * quadrado possível, porque o driver aloca o retângulo envolvente de qualquer
 * jeito e perder para padding é pior que perder para forma.
 *
 * As 19 populações vão em CINCO texturas RGBA32F (4 + 4 + 4 + 4 + 3, uma
 * componente sobrando), escritas de uma vez por MRT. Duas coleções dessas,
 * alternadas a cada passo: é o mesmo ping-pong do WebGPU, com framebuffer no
 * lugar do par de storage buffers.
 *
 *
 * OS TRÊS REMENDOS DE LINGUAGEM
 * -----------------------------
 * O IR foi escrito primeiro contra WGSL e três construções vazaram para ele.
 * Em vez de encher ir.js de condicionais por backend — que é como uma física
 * escrita uma vez volta a ser duas —, o preâmbulo daqui as define em GLSL:
 *
 *   select(a, b, c)  existe em WGSL, não em GLSL. Sai como função, com uma
 *                    sobrecarga por tipo usado. (`mix` com bool serviria, mas
 *                    só para float e vetor de float, e o IR também seleciona
 *                    entre endereços.)
 *   f32(x)           o cast de WGSL, aqui uma macro para `float(x)`.
 *   gid              o índice global de invocação, que no fragment shader é a
 *                    célula decodificada do atlas.
 *
 * O que NÃO é remendo: `mix`, `dot`, `clamp`, `sqrt`, `max` e `pow` têm o
 * mesmo nome e a mesma semântica nas duas linguagens, e os blocos do IR usam
 * só esses.
 *
 *
 * SAÍDAS COMO LOCAIS
 * ------------------
 * As populações escritas vão para `vec4 P0..P4` locais, copiadas para os
 * `out` no fim do main. Dois motivos: a esponja LÊ o que a colisão acabou de
 * escrever (`lerPopDst`), e uma variável de saída lida é terreno pantanoso; e
 * o ramo de célula sólida precisa escrever ALGUMA coisa — em GLSL não há como
 * sair do fragment sem deixar as saídas definidas, e uma saída não escrita é
 * lixo que o próximo passo vai ler como população.
 */

import {
  Q, C, W, OPP, CELULA, num,
  emitirPasso, blocoEquilibrio,
} from './ir.js';

export const TIPO = { ...CELULA };

/** Quantas texturas RGBA32F as Q populações ocupam. */
export const N_ALVOS = Math.ceil(Q / 4);

/** Em que textura e em que componente mora cada direção. */
export function planoDeTexturas() {
  return Array.from({ length: Q }, (_, i) => ({
    dir: i,
    textura: Math.floor(i / 4),
    componente: i % 4,
  }));
}

/* ──────────────────────────────────────────────────────────────── o dialeto */

function dialeto(plano, { destino = 'P' } = {}) {
  const src = (i, idx) => {
    const { textura, componente } = plano[i];
    return `texelFetch(uSrc${textura}, ${idx}, 0)[${componente}]`;
  };
  const dst = (i) => {
    const { textura, componente } = plano[i];
    return `${destino}${textura}[${componente}]`;
  };

  return {
    comentario: (s) => `// ${s}`,
    letF32: (n, e) => `float ${n} = ${e};`,
    letVec3: (n, e) => `vec3 ${n} = ${e};`,
    letU32: (n, e) => `uint ${n} = ${e};`,
    /* O endereço de uma célula é um texel do atlas. */
    letIdx: (n, e) => `ivec2 ${n} = ${e};`,
    vec3: (a, b, c) => `vec3(${a}, ${b}, ${c})`,
    indentar: (s) => (s ? '  ' + s : s),

    lerPop: (i, idx) => src(i, idx),
    lerPopDst: (i) => dst(i),
    setPop: (i, e) => `${dst(i)} = ${e};`,

    /*
     * Vizinho, em coordenadas de atlas.
     *
     * Mesma estrutura do dialeto WGSL: as coordenadas envolvidas dos três
     * vizinhos possíveis em cada eixo são calculadas UMA vez no cabeçalho
     * (xm/x0/xp e análogos) e cada direção vira só uma consulta. A diferença é
     * que aqui o resultado passa por `emAtlas`, que dobra o z na grade de
     * ladrilhos — a única aritmética que o backend WebGL2 tem a mais.
     */
    vizinho: (dx, dy, dz) => {
      if (dx === 0 && dy === 0 && dz === 0) return 'cell';
      const eixo = (d, n) => (d < 0 ? `${n}m` : d > 0 ? `${n}p` : `${n}0`);
      return `emAtlas(${eixo(dx, 'x')}, ${eixo(dy, 'y')}, ${eixo(dz, 'z')})`;
    },

    ehTipo: (nome) => `ct == ${CELULA[nome]}u`,
    ehFluidoEm: (idx) => `tipoEm(${idx}) == ${CELULA.FLUIDO}u`,
    tipoEm: (idx) => `tipoEm(${idx})`,
    ehSolidoTipo: (t) =>
      `(${t} == ${CELULA.SOLIDO}u || ${t} == ${CELULA.SOLIDO_MOVEL}u ` +
      `|| ${t} == ${CELULA.PAREDE}u)`,
    velParedeDe: (t) =>
      `select(vec3(0.0), uBeltU, ${t} == ${CELULA.SOLIDO_MOVEL}u)`,
    se: (cond) => [`if (${cond}) {`],
    senaoSe: (cond) => [`} else if (${cond}) {`],
    senao: () => ['} else {'],
    fimSe: () => ['}'],
  };
}

/* ──────────────────────────────────────────────────────────────── preâmbulo */

function preambulo({ comTipo = true } = {}) {
  const L = [];
  L.push('#version 300 es');
  /*
   * highp em tudo, e int junto.
   *
   * As populações são fp32 e a soma delas é a densidade — em mediump (fp16 em
   * muito celular) o delta de 1e-4 que carrega toda a informação de pressão
   * simplesmente não existe. E os índices de atlas passam de 32 mil num preset
   * grande, o que estoura o mínimo garantido de mediump int.
   */
  L.push('precision highp float;');
  L.push('precision highp int;');
  L.push('precision highp sampler2D;');
  if (comTipo) L.push('precision highp usampler2D;');
  L.push('');
  L.push('// ─── remendos de linguagem: ver o cabeçalho de glsl.js');
  L.push('#define f32(v) float(v)');
  L.push('float select(float a, float b, bool c) { return c ? b : a; }');
  L.push('vec3 select(vec3 a, vec3 b, bool c) { return c ? b : a; }');
  L.push('');
  L.push('uniform ivec3 uDim;      // nx, ny, nz');
  L.push('uniform ivec2 uTiles;    // ladrilhos em x e y do atlas');
  L.push('uniform float uOmega;    // taxa de relaxação par');
  L.push('uniform float uMagicU;   // Lambda TRT');
  L.push('uniform float uLesCsU;   // constante de Smagorinsky; 0 desliga');
  L.push('uniform vec3 uBeltU;     // velocidade do piso (esteira rolante)');
  L.push('uniform vec3 uInletU;    // corrente livre, unidades de lattice');
  L.push('uniform vec3 uSponge;    // início, comprimento, espessura da CL da entrada');
  for (let t = 0; t < N_ALVOS; t++) L.push(`uniform sampler2D uSrc${t};`);
  if (comTipo) L.push('uniform usampler2D uTipo;');
  L.push('');
  L.push('/* Célula (x, y, z) -> texel do atlas. A fatia z mora no ladrilho');
  L.push(' * (z % tx, z / tx), e o ladrilho tem o tamanho de uma fatia inteira. */');
  L.push('ivec2 emAtlas(int x, int y, int z) {');
  L.push('  return ivec2(x + (z % uTiles.x) * uDim.x,');
  L.push('               y + (z / uTiles.x) * uDim.y);');
  L.push('}');
  if (comTipo) {
    L.push('uint tipoEm(ivec2 t) { return texelFetch(uTipo, t, 0).r; }');
  }
  return L;
}

/**
 * Decodifica a célula a partir do fragmento e monta os vizinhos envolvidos.
 *
 * O ENVOLVIMENTO É PERIÓDICO, como no WGSL, e pelo mesmo motivo: com o
 * bounce-back acontecendo no nó de FLUIDO, uma célula ao lado do piso nunca lê
 * o piso — ela devolve a si mesma o que mandou —, então o que existe do outro
 * lado do domínio não a alcança. Grampear em vez de envolver quebraria a
 * periodicidade em y, e é justamente ela que o teste de viscosidade por
 * decaimento de onda de cisalhamento mede.
 */
function decodificarCelula() {
  const L = [];
  L.push('  ivec2 frag = ivec2(gl_FragCoord.xy);');
  L.push('  int tx = frag.x / uDim.x;');
  L.push('  int ty = frag.y / uDim.y;');
  L.push('  int z = ty * uTiles.x + tx;');
  L.push('  ivec3 c = ivec3(frag.x - tx * uDim.x, frag.y - ty * uDim.y, z);');
  L.push('  // Ladrilho além de nz: o atlas é retangular e a última linha sobra.');
  L.push('  if (z >= uDim.z) { discard; }');
  L.push('  uvec3 gid = uvec3(c);');
  L.push('  ivec2 cell = frag;');
  L.push('');
  L.push('  int x0 = c.x, y0 = c.y, z0 = c.z;');
  L.push('  int xm = c.x == 0 ? uDim.x - 1 : c.x - 1;');
  L.push('  int xp = c.x == uDim.x - 1 ? 0 : c.x + 1;');
  L.push('  int ym = c.y == 0 ? uDim.y - 1 : c.y - 1;');
  L.push('  int yp = c.y == uDim.y - 1 ? 0 : c.y + 1;');
  L.push('  int zm = c.z == 0 ? uDim.z - 1 : c.z - 1;');
  L.push('  int zp = c.z == uDim.z - 1 ? 0 : c.z + 1;');
  return L;
}

function saidas() {
  const L = [];
  for (let t = 0; t < N_ALVOS; t++) {
    L.push(`layout(location = ${t}) out vec4 oPop${t};`);
  }
  return L;
}

function locaisDeSaida() {
  return [`  vec4 ${Array.from({ length: N_ALVOS }, (_, t) => `P${t}`).join(', ')};`];
}

function copiarSaidas() {
  return Array.from({ length: N_ALVOS }, (_, t) => `  oPop${t} = P${t};`);
}

/** Lê as Q populações da célula própria para P0..P4 — o passa-adiante. */
function lerTudoParaLocais(plano) {
  const L = [];
  for (let t = 0; t < N_ALVOS; t++) {
    L.push(`  P${t} = texelFetch(uSrc${t}, cell, 0);`);
  }
  return L;
}

/* ──────────────────────────────────────────────────────────────── os shaders */

/**
 * Kernel do passo: streaming + colisão, ou reflexão se a célula é de contorno.
 *
 * Um fragmento por célula, cinco alvos de cor. É o único shader do laço
 * quente, exatamente como o kernel de compute do WebGPU.
 */
export function shaderPasso() {
  const plano = planoDeTexturas();
  const d = dialeto(plano);

  const L = [];
  L.push(...preambulo());
  L.push('');
  L.push('/* GERADO por src/core/emit/glsl.js a partir de src/core/emit/ir.js.');
  L.push(' * Não edite: edite a física em ir.js e os dois backends acompanham. */');
  L.push(...saidas());
  L.push('');
  L.push('void main() {');
  L.push(...decodificarCelula());
  L.push('');
  L.push(...locaisDeSaida());
  L.push('  uint ct = tipoEm(cell);');
  L.push('');
  L.push('  // Célula sólida não é simulada: com o bounce-back no nó de fluido,');
  L.push('  // ninguém nunca lê as populações de dentro de um sólido. Mas em GLSL');
  L.push('  // não dá para sair sem escrever — saída não escrita é lixo que o');
  L.push('  // passo seguinte leria como população —, então ela passa adiante o');
  L.push('  // que já estava lá, que é finito e é ignorado.');
  L.push(`  if (ct == ${TIPO.SOLIDO}u || ct == ${TIPO.SOLIDO_MOVEL}u || ct == ${TIPO.PAREDE}u) {`);
  L.push(...lerTudoParaLocais(plano).map(s => '  ' + s));
  L.push(...copiarSaidas().map(s => '  ' + s));
  L.push('    return;');
  L.push('  }');
  L.push('');
  L.push('  float uOmegaPlus = uOmega;');
  L.push('  float uMagic = uMagicU;');
  L.push('  float uLesCs = uLesCsU;');
  L.push('  float uSpongeStart = uSponge.x;');
  L.push('  float uSpongeLen = uSponge.y;');
  L.push('  vec3 uInf = uInletU;');
  L.push('');
  L.push('  // Perfil de camada limite na entrada, lei de potência 1/7. Com');
  L.push('  // esteira rolante uSponge.z é 0 e isto vira escoamento uniforme.');
  L.push('  float uInletBL = max(uSponge.z, 1.0);');
  L.push('  float zc = float(c.z) + 0.5;');
  L.push('  float blf = select(1.0, pow(zc / uInletBL, 0.1428571429), zc < uSponge.z);');
  L.push('  vec3 uEntrada = uInletU * blf;');
  L.push('');
  L.push(...emitirPasso(d).map(s => (s ? '  ' + s : s)));
  L.push('');
  L.push(...copiarSaidas());
  L.push('}');
  return L.join('\n');
}

/**
 * Inicialização: equilíbrio da corrente livre em todo o domínio.
 *
 * Uma textura recém-alocada vem zerada e g_i = 0 já é estado válido (fluido
 * parado, rho = 1 exatamente). Partir do repouso, porém, custa uma travessia
 * inteira de domínio de transiente; partir do escoamento uniforme corta quase
 * tudo isso.
 */
export function shaderInit() {
  const plano = planoDeTexturas();
  const d = dialeto(plano);

  const L = [];
  L.push(...preambulo());
  L.push('');
  L.push('/* GERADO por src/core/emit/glsl.js */');
  L.push(...saidas());
  L.push('');
  L.push('void main() {');
  L.push(...decodificarCelula());
  L.push('');
  L.push(...locaisDeSaida());
  L.push('  uint ct = tipoEm(cell);');
  L.push(`  bool parado = (ct == ${TIPO.SOLIDO}u);`);
  L.push('  // O piso da esteira parte COM a velocidade da esteira: parti-lo');
  L.push('  // parado cria um degrau de cisalhamento no primeiro passo,');
  L.push('  // exatamente onde o bounce-back é menos estável.');
  L.push(`  bool esteira = (ct == ${TIPO.SOLIDO_MOVEL}u);`);
  L.push('  vec3 u = uInletU;');
  L.push('  if (parado) { u = vec3(0.0); }');
  L.push('  if (esteira) { u = uBeltU; }');
  L.push('  float delta = 0.0;');
  L.push('  float rho = 1.0;');
  L.push('  float uu = dot(u, u);');
  L.push('');
  L.push(...blocoEquilibrio(d).map(s => '  ' + s));
  L.push('');
  for (let i = 0; i < Q; i++) L.push('  ' + d.setPop(i, `e${i}`));
  L.push('');
  L.push(...copiarSaidas());
  L.push('}');
  return L.join('\n');
}

/**
 * Campo macroscópico (u, delta) para a renderização e para o diagnóstico.
 *
 * No WebGPU isto sai de graça no fim do kernel do passo, que já tem os
 * momentos na mão. Aqui não: escrevê-lo junto exigiria um sexto alvo de cor, e
 * `caps.js` só garante os cinco que as populações ocupam. Um passe separado
 * custa uma leitura a mais do domínio e não arrisca o backend inteiro num
 * limite que a máquina pode não ter.
 *
 * A CONTA É A MESMA. Momentos lidos das populações PÓS-colisão da própria
 * célula, e não pós-streaming: massa e momento são invariantes da colisão —
 * sum(g - e) = 0 e sum(c (g - e)) = 0 por construção do equilíbrio —, então
 * delta e u saem idênticos, a menos do erro de arredondamento.
 */
export function shaderMacros() {
  const plano = planoDeTexturas();
  const L = [];
  L.push(...preambulo({ comTipo: true }));
  L.push('');
  L.push('/* GERADO por src/core/emit/glsl.js */');
  L.push('layout(location = 0) out vec4 oMacro;');
  L.push('');
  L.push('void main() {');
  L.push(...decodificarCelula());
  L.push('');
  L.push('  uint ct = tipoEm(cell);');
  L.push('  // Sólido zerado: escrever a densidade de dentro de um corpo envenena');
  L.push('  // tudo que lê este campo — as partículas ganham posições absurdas e');
  L.push('  // um diagnóstico de divergência acusa células que nunca foram');
  L.push('  // escoamento.');
  L.push(`  if (ct == ${TIPO.SOLIDO}u || ct == ${TIPO.SOLIDO_MOVEL}u || ct == ${TIPO.PAREDE}u) {`);
  L.push('    oMacro = vec4(0.0);');
  L.push('    return;');
  L.push('  }');
  L.push('');
  for (let i = 0; i < Q; i++) {
    const { textura, componente } = plano[i];
    L.push(`  float g${i} = texelFetch(uSrc${textura}, cell, 0)[${componente}];`);
  }
  L.push('');
  L.push(`  float delta = ${Array.from({ length: Q }, (_, i) => `g${i}`).join(' + ')};`);
  L.push('  float rho = 1.0 + delta;');
  for (let a = 0; a < 3; a++) {
    const termos = [];
    for (let i = 0; i < Q; i++) {
      if (C[i][a] === 0) continue;
      termos.push(C[i][a] > 0 ? `g${i}` : `-g${i}`);
    }
    L.push(`  float m${'xyz'[a]} = ${termos.join(' + ').replace(/\+ -/g, '- ')};`);
  }
  L.push('  oMacro = vec4(vec3(mx, my, mz) / rho, delta);');
  L.push('}');
  return L.join('\n');
}

/**
 * Forças por TROCA DE MOMENTO, uma célula por fragmento.
 *
 * Para cada link que cruza a superfície — nó de fluido x_f com vizinho sólido
 * x_f + c_i — a quantidade de movimento entregue à parede é
 *
 *     dF = c_i [ f_i(x_f) + f_ī(x_s) ]
 *
 * e com bounce-back no nó de fluido o que volta é o que saiu, então a soma é o
 * dobro. ISTO INCLUI O ATRITO VISCOSO, que é a razão de o método ser este e
 * não integração de pressão: não precisa de normal reconstruída de uma escada
 * de voxels e entrega pressão e cisalhamento juntos.
 *
 * O termo `2 w_i c_i` das populações deslocadas NÃO cancela — ele só
 * cancelaria somado sobre o par oposto inteiro, e um link é um só. Esquecê-lo
 * dá um Cd plausível e errado por um fator constante, que sobrevive a qualquer
 * inspeção visual.
 *
 * A soma dos fragmentos é feita depois, pela pirâmide de `shaderReduzir` — o
 * WebGL2 não tem memória compartilhada de workgroup nem atômico de float.
 */
export function shaderForcas() {
  const plano = planoDeTexturas();
  const L = [];
  L.push(...preambulo());
  L.push('');
  L.push('/* GERADO por src/core/emit/glsl.js */');
  L.push('layout(location = 0) out vec4 oForca;');
  L.push('');
  L.push('void main() {');
  L.push(...decodificarCelula());
  L.push('');
  L.push('  vec3 f = vec3(0.0);');
  L.push(`  if (tipoEm(cell) == ${TIPO.FLUIDO}u) {`);
  for (let i = 1; i < Q; i++) {          // i = 0 não cruza superfície nenhuma
    const c = C[i];
    const eixo = (d, n) => (d < 0 ? `${n}m` : d > 0 ? `${n}p` : `${n}0`);
    const nb = `emAtlas(${eixo(c[0], 'x')}, ${eixo(c[1], 'y')}, ${eixo(c[2], 'z')})`;
    const { textura, componente } = plano[i];
    L.push('    {');
    L.push(`      ivec2 nb = ${nb};`);
    /* SÓ o corpo. O piso e as paredes do túnel refletem igual e não entram na
     * conta — um piso contabilizado tem nx*ny células de superfície contra
     * alguns milhares do corpo, e o que sai é o arrasto do CHÃO. */
    L.push(`      if (tipoEm(nb) == ${TIPO.SOLIDO}u) {`);
    L.push(`        float q = 2.0 * (texelFetch(uSrc${textura}, cell, 0)[${componente}]` +
      ` + ${num(W[i])});`);
    const termos = [];
    for (let a = 0; a < 3; a++) {
      if (c[a] === 0) { termos.push('0.0'); continue; }
      termos.push(c[a] > 0 ? 'q' : '-q');
    }
    L.push(`        f += vec3(${termos.join(', ')});`);
    L.push('      }');
    L.push('    }');
  }
  L.push('  }');
  L.push('  oForca = vec4(f, 0.0);');
  L.push('}');
  return L.join('\n');
}

/**
 * Redução por soma: cada fragmento soma um bloco 2×2 da textura anterior.
 *
 * Aplicada repetidamente até sobrar um texel, que é lido com readPixels. São
 * uns doze passes num atlas de 2560² — cada um com um quarto do trabalho do
 * anterior, o que soma um terço a mais que o primeiro.
 *
 * Somar em ÁRVORE não é só o que a falta de atômico obriga: é também a melhor
 * ordem numérica. Um acumulador único percorrendo um milhão de parcelas em
 * fp32 acumula erro proporcional ao número de termos; a soma emparelhada
 * acumula proporcional ao log dele.
 *
 * Os limites são testados explicitamente. Dimensão ímpar faz o último bloco
 * cair meio fora, e `texelFetch` fora da textura é comportamento indefinido em
 * GLSL ES — em alguns drivers devolve zero, em outros repete a borda, e aí a
 * força ganha um viés que depende de quem fabricou a placa.
 */
export function shaderReduzir() {
  return `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D uFonte;
uniform ivec2 uTam;        // tamanho da textura de origem
layout(location = 0) out vec4 oSoma;

void main() {
  ivec2 d = ivec2(gl_FragCoord.xy);
  ivec2 b = d * 2;
  vec4 s = texelFetch(uFonte, b, 0);
  bool temX = b.x + 1 < uTam.x;
  bool temY = b.y + 1 < uTam.y;
  if (temX) { s += texelFetch(uFonte, ivec2(b.x + 1, b.y), 0); }
  if (temY) { s += texelFetch(uFonte, ivec2(b.x, b.y + 1), 0); }
  if (temX && temY) { s += texelFetch(uFonte, b + ivec2(1, 1), 0); }
  oSoma = s;
}`;
}

/**
 * Vertex shader comum: um triângulo que cobre o alvo inteiro.
 *
 * Triângulo e não quad — dois vértices ficam fora do viewport e não há costura
 * diagonal onde as metades se encontram. Não há atributo nenhum: a posição sai
 * do gl_VertexID, então não existe buffer de vértices para ligar.
 */
export const VERTEX_COBERTURA = `#version 300 es
void main() {
  vec2 v = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(v * 2.0 - 1.0, 0.0, 1.0);
}`;

/** Nomes dos uniformes, para o runtime não sair adivinhando. */
export const UNIFORMES_GLSL = [
  'uDim', 'uTiles', 'uOmega', 'uMagicU', 'uLesCsU',
  'uBeltU', 'uInletU', 'uSponge', 'uTipo',
  ...Array.from({ length: N_ALVOS }, (_, t) => `uSrc${t}`),
];

export { Q, C, W, OPP };
