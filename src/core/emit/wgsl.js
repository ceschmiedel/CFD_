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

/* Tipos de célula que o shader distingue. SOLIDO_MOVEL existe separado para a
 * esteira rolante e as rodas: mesma reflexão, mais o termo de parede móvel. */
export const TIPO = {
  ...CELULA,
  SOLIDO_MOVEL: 4,
};

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
    setPop: (i, e) => `${end(prefDst, i, 'cell')} = ${e};`,

    vizinho: (dx, dy, dz) =>
      (dx === 0 && dy === 0 && dz === 0)
        ? 'cell'
        : `viz(pos, vec3<i32>(${dx}, ${dy}, ${dz}))`,

    seSolido: () => ['if (solido) {'],
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

  L.push('');
  L.push('// índice linear do vizinho, com envolvimento periódico.');
  L.push('// |c_i| <= 1 em toda direção, então somar n antes do módulo basta.');
  L.push('fn viz(p: vec3<i32>, d: vec3<i32>) -> u32 {');
  L.push('  let n = vec3<i32>(i32(P.dim.x), i32(P.dim.y), i32(P.dim.z));');
  L.push('  let q = (p + d + n) % n;');
  L.push('  return u32(q.z) * P.dim.x * P.dim.y + u32(q.y) * P.dim.x + u32(q.x);');
  L.push('}');
  return L;
}

function cabecalhoKernel(comMacros) {
  const L = [];
  L.push('@compute @workgroup_size(64, 1, 1)');
  L.push('fn main(@builtin(global_invocation_id) gid: vec3<u32>) {');
  L.push('  if (gid.x >= P.dim.x || gid.y >= P.dim.y || gid.z >= P.dim.z) { return; }');
  L.push('');
  L.push('  let pos = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));');
  L.push('  let N = P.dim.x * P.dim.y * P.dim.z;');
  L.push('  let cell = gid.z * P.dim.x * P.dim.y + gid.y * P.dim.x + gid.x;');
  L.push('');
  L.push('  let ct = tipo[cell];');
  L.push(`  let solido = (ct == ${TIPO.SOLIDO}u) || (ct == ${TIPO.SOLIDO_MOVEL}u);`);
  L.push('  // só a parede marcada como móvel arrasta o fluido junto');
  L.push(`  let uWall = select(vec3<f32>(0.0), P.beltU.xyz, ct == ${TIPO.SOLIDO_MOVEL}u);`);
  L.push('');
  L.push('  let uOmegaPlus = P.omegaPlus;');
  L.push('  let uMagic = P.magic;');
  L.push('  let uLesCs = P.lesCs;');
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
    L.push('  // campo macroscópico para a renderização ler sem refazer a soma');
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
  L.push(`  let parado = (ct == ${TIPO.SOLIDO}u) || (ct == ${TIPO.SOLIDO_MOVEL}u);`);
  L.push('  let u = select(P.inletU.xyz, vec3<f32>(0.0), parado);');
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
