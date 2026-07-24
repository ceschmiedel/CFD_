/* ── src/core/emit/ir.js ─────────────────────────────────────────────────────
 *
 * A física do solver, escrita uma vez.
 *
 * Este arquivo não contém WGSL nem GLSL. Ele contém o passo LBM expresso em
 * termos de um `dialeto` — um objeto pequeno que sabe declarar uma variável,
 * ler uma população e escrever uma população na linguagem alvo. wgsl.js e
 * glsl.js fornecem esses dialetos; tudo o mais aqui é comum aos dois.
 *
 * O motivo de existir é simples: um solver com dois backends e duas cópias da
 * colisão TRT tem, mais cedo ou mais tarde, duas físicas diferentes. E o modo
 * como isso se manifesta é cruel — não um erro de compilação, mas um Cd que
 * difere em 4% entre os backends e ninguém sabe qual está certo. Gerando os
 * dois a partir daqui, uma discordância entre backends é necessariamente bug
 * de memória ou de despacho, nunca de modelagem, e a suíte de validação sabe
 * onde procurar.
 *
 * O gerador também ganha coisas que ninguém escreveria à mão: as dezenove
 * direções viram código reto sem laço nem indireção, as constantes c_i entram
 * como literais (o compilador do driver dobra `1.0 * x` sozinho e some com os
 * zeros), e a colisão TRT é emitida por PAR OPOSTO, que corta metade das
 * multiplicações porque g^+ é simétrico e g^- é antissimétrico no par.
 *
 *
 * O ALGORITMO, PRECISAMENTE
 * -------------------------
 * O buffer guarda as distribuições PÓS-COLISÃO do passo anterior. Num passo,
 * para cada célula x:
 *
 *   1. PULL   gin[i] = src[i][x - c_i]        (isto é o streaming)
 *   2. se x é SÓLIDO:  dst[i][x] = gin[opp i] + termo de parede móvel
 *                      e acabou — sólido não coride.
 *   3. senão: momentos -> equilíbrio -> LES -> colisão TRT -> dst[i][x]
 *
 * O passo 2 é bounce-back de meio-caminho: a parede fica exatamente no meio
 * entre o último nó de fluido e o primeiro nó sólido. Com Lambda = 3/16 essa
 * posição é independente da viscosidade, que é a razão de o TRT existir aqui.
 *
 * Confira o passo 2 você mesmo: o nó de fluido y = x - c_i, no passo seguinte,
 * puxa a direção opp(i) de y - c_opp(i) = y + c_i = x, e encontra
 * dst[opp i][x] = gin[i] — exatamente o que ele mandou, de volta.
 */

import { Q, C, W, OPP, CS2, INV_CS2, INV_CS4, MAGIC } from '../lattice.js';

/* Tipos de célula. Cabem num u32; o solver lê um por célula. */
export const CELULA = {
  FLUIDO: 0,
  SOLIDO: 1,
  ENTRADA: 2,
  SAIDA: 3,
};

/**
 * Formata um float de modo que float64 -> texto -> float32 seja exato, e que
 * ainda dê para ler o número no shader gerado quando algo der errado.
 */
export function num(x) {
  if (Number.isInteger(x)) return x.toFixed(1);
  const s = x.toPrecision(9);
  /* aparar zeros à direita sem comer o ponto decimal */
  return s.includes('.') && !s.includes('e')
    ? s.replace(/0+$/, '').replace(/\.$/, '.0')
    : s;
}

/**
 * Produto escalar c_i . v emitido como aritmética literal.
 *
 * c_i só tem componentes -1, 0 e +1, então isto vira `v.x + v.y`, `-v.z` ou
 * literalmente `0.0`. Emitir `1.0*v.x + 0.0*v.y + 0.0*v.z` também funcionaria
 * e o driver otimizaria — mas o shader gerado é para ser lido por gente
 * quando um caso de validação falhar, e 19 linhas de zeros o tornam ilegível.
 */
export function dotC(i, v) {
  const c = C[i];
  const termos = [];
  for (let a = 0; a < 3; a++) {
    if (c[a] === 0) continue;
    const comp = `${v}.${'xyz'[a]}`;
    termos.push(c[a] > 0 ? comp : `-${comp}`);
  }
  if (!termos.length) return '0.0';
  return termos.join(' + ').replace(/\+ -/g, '- ');
}

/* ────────────────────────────────────────────────────── blocos reutilizáveis */

/** Puxa as Q populações dos vizinhos. Este é o streaming. */
export function blocoPull(d) {
  const L = [];
  L.push(d.comentario('streaming: cada população vem do vizinho a montante'));
  for (let i = 0; i < Q; i++) {
    const c = C[i];
    L.push(d.letF32(`g${i}`, d.lerPop(i, d.vizinho(-c[0], -c[1], -c[2]))));
  }
  return L;
}

/** delta = sum g_i, rho*u = sum g_i c_i, u = (rho u)/rho. */
export function blocoMomentos(d) {
  const L = [];
  L.push(d.comentario('momentos hidrodinâmicos (exatos nas populações deslocadas)'));

  const soma = Array.from({ length: Q }, (_, i) => `g${i}`).join(' + ');
  L.push(d.letF32('delta', soma));
  L.push(d.letF32('rho', '1.0 + delta'));

  /* rho*u = sum g_i c_i. Só as direções com c_a != 0 entram em cada eixo. */
  for (let a = 0; a < 3; a++) {
    const termos = [];
    for (let i = 0; i < Q; i++) {
      if (C[i][a] === 0) continue;
      termos.push(C[i][a] > 0 ? `g${i}` : `-g${i}`);
    }
    L.push(d.letF32(`m${'xyz'[a]}`, termos.join(' + ').replace(/\+ -/g, '- ')));
  }

  L.push(d.letVec3('u', d.vec3('mx', 'my', 'mz') + ' / rho'));
  L.push(d.letF32('uu', 'dot(u, u)'));
  return L;
}

/** g_i^eq = w_i (delta + rho (3 c.u + 4.5 (c.u)^2 - 1.5 u.u)) */
export function blocoEquilibrio(d) {
  const L = [];
  L.push(d.comentario('equilíbrio deslocado — ver o cabeçalho de lattice.js'));
  for (let i = 0; i < Q; i++) {
    if (i === 0) {
      /* c_0 = 0 anula os termos de c.u; sobra delta e o -1.5 u.u */
      L.push(d.letF32('e0', `${num(W[0])} * (delta - 1.5 * rho * uu)`));
      continue;
    }
    L.push(d.letF32(`cu${i}`, dotC(i, 'u')));
    L.push(d.letF32(
      `e${i}`,
      `${num(W[i])} * (delta + rho * (3.0 * cu${i} + 4.5 * cu${i} * cu${i} - 1.5 * uu))`,
    ));
  }
  return L;
}

/**
 * Viscosidade turbulenta de Smagorinsky, resolvida em forma fechada.
 *
 * O tensor de fluxo de momento fora do equilíbrio é
 *
 *     Pi_ab = sum_i c_ia c_ib (g_i - g_i^eq)
 *
 * (o deslocamento w_i cancela aqui, porque sai igual dos dois lados da
 * subtração). Dele saem a taxa de deformação e a viscosidade sub-grid:
 *
 *     S_ab = -Pi_ab / (2 rho c_s^2 tau)
 *     nu_t = (C_s Delta)^2 |S|,   |S| = sqrt(2 S:S)
 *
 * Substituindo nu_total = c_s^2 (tau - 1/2) = nu_0 + nu_t e resolvendo a
 * quadrática em tau (com Delta = 1 célula):
 *
 *     tau = 1/2 [ tau_0 + sqrt(tau_0^2 + 2 C_s^2 Qs / (rho c_s^4)) ]
 *
 * com Qs = sqrt(2 Pi:Pi). Fechada, sem iteração, um sqrt por célula.
 *
 * A taxa PAR passa a ser 1/tau. A ÍMPAR é recalculada a partir do Lambda, que
 * se mantém fixo — e essa é a parte que se erra por descuido. Se omega_minus
 * ficar congelado enquanto omega_plus varia com a turbulência, Lambda passa a
 * variar de célula para célula, a parede sai do meio-caminho onde o LES está
 * ativo, e o arrasto passa a depender da intensidade turbulenta local. O bug
 * não aparece em Poiseuille nem em nenhum caso laminar: só aparece como um Cd
 * que não bate, no único regime em que você não tem gabarito.
 */
export function blocoLES(d) {
  const L = [];
  L.push(d.comentario('LES Smagorinsky: eddy viscosity a partir de Pi^neq'));

  for (let i = 0; i < Q; i++) L.push(d.letF32(`n${i}`, `g${i} - e${i}`));

  /* As seis componentes independentes de Pi (simétrico). */
  const pares = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]];
  const nomes = ['pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz'];

  pares.forEach(([a, b], k) => {
    const termos = [];
    for (let i = 0; i < Q; i++) {
      const coef = C[i][a] * C[i][b];
      if (coef === 0) continue;
      termos.push(coef > 0 ? `n${i}` : `-n${i}`);
    }
    L.push(d.letF32(nomes[k], termos.length
      ? termos.join(' + ').replace(/\+ -/g, '- ')
      : '0.0'));
  });

  /* Pi:Pi conta as fora-da-diagonal duas vezes — o tensor é simétrico. */
  L.push(d.letF32('pp',
    'pxx * pxx + pyy * pyy + pzz * pzz + ' +
    '2.0 * (pxy * pxy + pxz * pxz + pyz * pyz)'));
  L.push(d.letF32('qs', 'sqrt(2.0 * pp)'));

  L.push(d.letF32('tau0', '1.0 / uOmegaPlus'));
  L.push(d.letF32('tauT',
    `0.5 * (tau0 + sqrt(tau0 * tau0 + ` +
    `${num(2 * INV_CS4)} * uLesCs * uLesCs * qs / rho))`));

  /* uLesCs = 0 zera o acréscimo e tauT volta a ser exatamente tau0 — o
   * caminho sem LES não é um `if`, é o mesmo código com a constante nula. */
  L.push(d.letF32('omegaP', '1.0 / tauT'));
  L.push(d.comentario('Lambda fixo: a parede não pode se mexer com a turbulência'));
  L.push(d.letF32('omegaM', '1.0 / (uMagic / (1.0 / omegaP - 0.5) + 0.5)'));
  return L;
}

/** Colisão TRT, emitida por par oposto. */
export function blocoColisao(d) {
  const L = [];
  L.push(d.comentario('colisão TRT: par relaxa em omegaP, ímpar em omegaM'));

  /* i = 0 é seu próprio oposto: a parte ímpar é identicamente nula. */
  L.push(d.setPop(0, 'g0 - omegaP * (g0 - e0)'));

  const feitos = new Set([0]);
  for (let i = 1; i < Q; i++) {
    if (feitos.has(i)) continue;
    const j = OPP[i];
    feitos.add(i); feitos.add(j);

    L.push(d.letF32(`p${i}`, `0.5 * ((g${i} + g${j}) - (e${i} + e${j}))`));
    L.push(d.letF32(`m${i}`, `0.5 * ((g${i} - g${j}) - (e${i} - e${j}))`));
    L.push(d.setPop(i, `g${i} - omegaP * p${i} - omegaM * m${i}`));
    L.push(d.setPop(j, `g${j} - omegaP * p${i} + omegaM * m${i}`));
  }
  return L;
}

/**
 * Bounce-back de meio-caminho, com correção de parede móvel.
 *
 *     dst[i] = gin[opp i] + 2 w_i rho_w (c_i . u_w) / c_s^2
 *
 * O sinal: parede indo em +x deposita população extra na direção +x, ou seja
 * empurra o fluido a favor dela. Se o sinal estiver trocado a esteira rolante
 * FREIA o escoamento em vez de acompanhá-lo, e o resultado é uma camada
 * limite grossa demais que parece só "um pouco de arrasto a mais".
 *
 * rho_w é a densidade local; usar 1.0 é a aproximação usual e vale enquanto
 * delta ~ 1e-4, mas custa nada usar rho e some com um viés de ordem Ma^2.
 */
export function blocoBounceBack(d) {
  const L = [];
  L.push(d.comentario('sólido: reflete, não coride'));
  for (let i = 0; i < Q; i++) {
    const j = OPP[i];
    if (i === 0) { L.push(d.setPop(0, 'g0')); continue; }
    const cu = dotC(i, 'uWall');
    L.push(d.setPop(i, `g${j} + ${num(2 * W[i] * INV_CS2)} * rho * (${cu})`));
  }
  return L;
}

/* ──────────────────────────────────────────────────────────── passo inteiro */

/**
 * Corpo completo do kernel de passo, em linhas.
 *
 * O dialeto fornece: comentario, letF32, letVec3, vec3, lerPop, setPop,
 * vizinho, seSolido/senao/fim. Nada mais deste arquivo conhece a linguagem
 * alvo.
 */
export function emitirPasso(d) {
  const L = [];
  L.push(...blocoPull(d));
  L.push('');
  L.push(...blocoMomentos(d));
  L.push('');

  L.push(...d.seSolido());
  L.push(...blocoBounceBack(d).map(s => d.indentar(s)));
  L.push(...d.senao());

  const corpo = [];
  corpo.push(...blocoEquilibrio(d));
  corpo.push('');
  corpo.push(...blocoLES(d));
  corpo.push('');
  corpo.push(...blocoColisao(d));
  L.push(...corpo.map(s => d.indentar(s)));

  L.push(...d.fimSe());
  return L;
}

/**
 * Constantes que o shader gerado precisa ver como uniformes, com os nomes que
 * os blocos acima usam. Manter esta lista junto do código que a consome evita
 * o modo de falha clássico: renomear um uniforme e descobrir três backends
 * depois que um deles ainda liga o antigo.
 */
export const UNIFORMES = [
  { nome: 'uOmegaPlus', tipo: 'f32', doc: 'taxa de relaxação par, 1/tau da viscosidade molecular' },
  { nome: 'uMagic', tipo: 'f32', doc: 'Lambda TRT = (1/w+ - 1/2)(1/w- - 1/2)' },
  { nome: 'uLesCs', tipo: 'f32', doc: 'constante de Smagorinsky; 0.0 desliga o LES' },
];

export const PADROES = {
  magic: MAGIC.WALL,
  lesCs: 0.1,
};

/* Reexportados para os dialetos não precisarem importar lattice.js de novo. */
export { Q, C, W, OPP, CS2, INV_CS2, INV_CS4 };
