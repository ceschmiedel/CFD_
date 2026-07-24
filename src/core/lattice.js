/* ── src/core/lattice.js ─────────────────────────────────────────────────────
 *
 * O conjunto de velocidades D3Q19 e as distribuições deslocadas.
 *
 *
 * CONVENÇÃO DE ARMAZENAMENTO
 * --------------------------
 * Este solver NÃO armazena f_i. Armazena o desvio do repouso:
 *
 *     g_i = f_i - w_i
 *
 * O motivo é aritmética de ponto flutuante, e ele decide sozinho se o mapa de
 * pressão sai limpo ou sai granulado.
 *
 * A densidade é rho = 1 + delta, e nos números de Mach em que rodamos delta é
 * da ordem de 1e-4. A pressão que pintamos na superfície do corpo é
 * p = c_s^2 * delta. Se armazenamos f_i — cada um por volta de 0.05, o de
 * repouso por volta de 0.33 — e somamos dezenove deles em fp32, o erro
 * absoluto da soma fica em torno de 2e-7. Contra um delta de 1e-4 isso é 0,2%
 * de erro, que aparece como ruído visível no campo de Cp. Armazenando g_i,
 * cujos valores já são da ordem do desvio, a mesma soma erra por volta de 1e-9.
 *
 * Todas as identidades sobrevivem ao deslocamento porque sum(w_i) = 1 e
 * sum(w_i c_i) = 0:
 *
 *     delta   = sum_i g_i                    (rho = 1 + delta)
 *     rho * u = sum_i g_i c_i                (exatamente; os w_i c_i cancelam)
 *     g_i^eq  = w_i (delta + rho (3(c.u) + 4.5(c.u)^2 - 1.5 u.u))
 *
 * E o bounce-back continua sendo g_opp(i) <- g_i, porque w_opp(i) = w_i.
 *
 * Efeito colateral agradável: o repouso é g_i = 0 em toda parte, então zerar um
 * buffer inicializa o fluido parado com rho = 1 exatamente — e no WebGPU um
 * buffer nasce zerado, então não existe kernel de inicialização a escrever.
 *
 *
 * CUIDADO — ONDE O DESLOCAMENTO NÃO CANCELA
 * -----------------------------------------
 * Na troca de momento (forces.js) a contribuição de UM link é
 *
 *     c_i [ f_i + f_opp ] = c_i [ g_i + g_opp + 2 w_i ]
 *
 * O termo 2 w_i c_i só cancelaria somado sobre o par oposto inteiro, e um link
 * é um só. Quem calcula força tem de recolocá-lo à mão. Esquecer isso dá um Cd
 * que parece plausível e está errado por uma constante — o pior tipo de erro.
 */

export const Q = 19;

export const CS2 = 1 / 3;
export const INV_CS2 = 3;
export const INV_CS4 = 9;

/* c_i em pares opostos adjacentes: (0), (1,2), (3,4), ...
 * Isso torna o índice oposto uma expressão fechada em vez de uma tabela de
 * busca — importante no shader, onde uma indireção custa. */
export const C = [
  [0, 0, 0],      //  0   repouso

  [1, 0, 0],      //  1   +x
  [-1, 0, 0],     //  2   -x
  [0, 1, 0],      //  3   +y
  [0, -1, 0],     //  4   -y
  [0, 0, 1],      //  5   +z
  [0, 0, -1],     //  6   -z

  [1, 1, 0],      //  7
  [-1, -1, 0],    //  8
  [1, -1, 0],     //  9
  [-1, 1, 0],     // 10
  [1, 0, 1],      // 11
  [-1, 0, -1],    // 12
  [1, 0, -1],     // 13
  [-1, 0, 1],     // 14
  [0, 1, 1],      // 15
  [0, -1, -1],    // 16
  [0, 1, -1],     // 17
  [0, -1, 1],     // 18
];

/* 1/3 no repouso, 1/18 nos eixos, 1/36 nas diagonais de face. */
export const W = [
  1 / 3,
  ...Array(6).fill(1 / 18),
  ...Array(12).fill(1 / 36),
];

/* Índice oposto. Com os pares adjacentes, opp(0)=0 e opp(i)=((i-1) XOR 1)+1. */
export const OPP = [0, ...Array.from({ length: Q - 1 }, (_, k) => (((k) ^ 1) + 1))];

/* Direções que apontam para +x, usadas pelo inlet, e as que apontam para -x,
 * usadas pela saída. Pré-computar evita um `if` por direção no shader. */
export const DIRS_PLUS_X = C.map((c, i) => (c[0] > 0 ? i : -1)).filter(i => i >= 0);
export const DIRS_MINUS_X = C.map((c, i) => (c[0] < 0 ? i : -1)).filter(i => i >= 0);

/*
 * Valores mágicos de Lambda para o operador TRT.
 *
 *     Lambda = (1/omega_plus - 1/2) (1/omega_minus - 1/2)
 *
 * WALL         3/16  põe uma parede reta de bounce-back exatamente no meio do
 *                    caminho entre nós, INDEPENDENTE da viscosidade
 *                    (Ginzburg & Adler 1994). O arrasto depende inteiramente
 *                    de onde a parede pensa que está, então é este que importa
 *                    aqui e é o padrão.
 * STABLE       1/4   o mais estável em tau = 1. Não em geral — e nós quase
 *                    nunca rodamos em tau = 1.
 * THIRD_ORDER  1/12  cancela o erro espacial de terceira ordem dominante.
 * FOURTH_ORDER 1/6   cancela o de quarta ordem (melhor advecção pura).
 */
export const MAGIC = {
  WALL: 3 / 16,
  STABLE: 1 / 4,
  THIRD_ORDER: 1 / 12,
  FOURTH_ORDER: 1 / 6,
};

/* ─────────────────────────────────────────────────────── relações viscosidade */

/** nu = c_s^2 (1/omega - 1/2)  =>  omega */
export function omegaFromNu(nu) {
  return 1 / (nu / CS2 + 0.5);
}

/** o inverso */
export function nuFromOmega(omega) {
  return CS2 * (1 / omega - 0.5);
}

/** Taxa de relaxação ímpar que produz o Lambda pedido. */
export function omegaMinusFromLambda(omegaPlus, lambda) {
  const tauMinusHalf = lambda / (1 / omegaPlus - 0.5);
  return 1 / (tauMinusHalf + 0.5);
}

/** O inverso: o parâmetro mágico implicado por um par de taxas. */
export function lambdaFromOmegas(omegaPlus, omegaMinus) {
  return (1 / omegaPlus - 0.5) * (1 / omegaMinus - 0.5);
}

/* ──────────────────────────────────────────── referência em JS (só validação) */

/*
 * O caminho quente é o shader gerado por emit/ir.js. As duas funções abaixo
 * existem para a suíte de validação poder checar as identidades de momento e
 * comparar contra o que a GPU devolve, sem envolver a GPU na verificação.
 */

/** g_i^eq deslocado, dado delta = rho-1 e u (unidades de lattice). */
export function equilibrium(delta, u, out) {
  const g = out || new Float64Array(Q);
  const rho = 1 + delta;
  const uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
  for (let i = 0; i < Q; i++) {
    const c = C[i];
    const cu = c[0] * u[0] + c[1] * u[1] + c[2] * u[2];
    g[i] = W[i] * (delta + rho * (3 * cu + 4.5 * cu * cu - 1.5 * uu));
  }
  return g;
}

/** Momentos hidrodinâmicos: devolve { delta, u }. */
export function moments(g) {
  let delta = 0, mx = 0, my = 0, mz = 0;
  for (let i = 0; i < Q; i++) {
    const gi = g[i], c = C[i];
    delta += gi;
    mx += gi * c[0];
    my += gi * c[1];
    mz += gi * c[2];
  }
  const inv = 1 / (1 + delta);
  return { delta, u: [mx * inv, my * inv, mz * inv] };
}

/* ─────────────────────────────────────────────────────────── auto-verificação */

/*
 * Invariantes do conjunto de velocidades. Rodam na importação, custam
 * microssegundos, e transformam um erro de digitação numa tabela de 19 linhas
 * — que de outro modo apareceria como um arrasto sutilmente errado três dias
 * depois — em uma exceção imediata.
 */
export function selfCheck() {
  const fail = (m) => { throw new Error('lattice.js: ' + m); };

  if (C.length !== Q || W.length !== Q || OPP.length !== Q) fail('tamanhos');

  const sw = W.reduce((a, b) => a + b, 0);
  if (Math.abs(sw - 1) > 1e-15) fail(`sum(w_i) = ${sw}, deveria ser 1`);

  for (let a = 0; a < 3; a++) {
    let s = 0;
    for (let i = 0; i < Q; i++) s += W[i] * C[i][a];
    if (Math.abs(s) > 1e-15) fail(`sum(w_i c_i[${a}]) = ${s}, deveria ser 0`);
  }

  /* segundo momento: sum(w_i c_ia c_ib) = c_s^2 delta_ab */
  for (let a = 0; a < 3; a++) {
    for (let b = 0; b < 3; b++) {
      let s = 0;
      for (let i = 0; i < Q; i++) s += W[i] * C[i][a] * C[i][b];
      const want = a === b ? CS2 : 0;
      if (Math.abs(s - want) > 1e-15) fail(`sum(w c c)[${a}][${b}] = ${s}`);
    }
  }

  /* oposição é involução, nega a velocidade e preserva o peso */
  for (let i = 0; i < Q; i++) {
    if (OPP[OPP[i]] !== i) fail(`OPP não é involução em ${i}`);
    for (let a = 0; a < 3; a++) {
      if (C[OPP[i]][a] !== -C[i][a]) fail(`c[opp(${i})] != -c[${i}]`);
    }
    if (W[OPP[i]] !== W[i]) fail(`w[opp(${i})] != w[${i}]`);
  }

  /* o equilíbrio em repouso é exatamente zero — a propriedade que deixa o
   * buffer zerado do WebGPU servir de estado inicial */
  const g0 = equilibrium(0, [0, 0, 0]);
  for (let i = 0; i < Q; i++) if (g0[i] !== 0) fail('equilíbrio de repouso != 0');

  /* e o equilíbrio reproduz os momentos que o geraram */
  const d = 1e-4, u = [0.05, -0.02, 0.01];
  const m = moments(equilibrium(d, u));
  if (Math.abs(m.delta - d) > 1e-15) fail(`delta round-trip: ${m.delta}`);
  for (let a = 0; a < 3; a++) {
    if (Math.abs(m.u[a] - u[a]) > 1e-15) fail(`u round-trip[${a}]: ${m.u[a]}`);
  }

  return true;
}

selfCheck();
