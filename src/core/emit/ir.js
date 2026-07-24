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

/*
 * Tipos de célula. Cabem num u32; o solver lê um por célula.
 *
 * ESPELHO_Y e ESPELHO_Z são paredes de DESLIZAMENTO LIVRE, e não paredes de
 * verdade. As laterais e o teto de um túnel numérico não devem crescer camada
 * limite: elas não existem no escoamento que estamos modelando (um carro na
 * estrada não tem paredes a três metros de cada lado), e a camada limite que
 * elas criariam estreitaria a seção de teste ao longo do domínio, aumentando o
 * bloqueio progressivamente e inflando o arrasto. Reflexão especular deixa o
 * ar deslizar: a componente normal inverte, as tangenciais passam.
 *
 * O PISO é o oposto — ele existe, e é SOLIDO_MOVEL quando a esteira está
 * ligada. Essa assimetria entre piso e teto é a geometria real do problema.
 */
export const CELULA = {
  FLUIDO: 0,
  SOLIDO: 1,
  ENTRADA: 2,
  SAIDA: 3,
  SOLIDO_MOVEL: 4,
  ESPELHO_Y: 5,
  ESPELHO_Z: 6,
};

/**
 * Índice da direção com a componente `eixo` negada e as outras preservadas —
 * a reflexão especular. Existe porque D3Q19 é fechado sob essa operação.
 */
export function espelharIdx(i, eixo) {
  const c = C[i].slice();
  c[eixo] = -c[eixo];
  for (let j = 0; j < Q; j++) {
    if (C[j][0] === c[0] && C[j][1] === c[1] && C[j][2] === c[2]) return j;
  }
  throw new Error(`espelharIdx: sem par para ${i} no eixo ${eixo}`);
}

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

/**
 * Reflexão especular numa parede de deslizamento livre.
 *
 * Idêntico em forma ao bounce-back, trocando `opp` por `espelhar`: o nó de
 * parede devolve o que chegou, mas só com a componente normal invertida.
 */
export function blocoEspelho(d, eixo) {
  const L = [];
  L.push(d.comentario(`parede de deslizamento livre: inverte só ${'xyz'[eixo]}`));
  for (let i = 0; i < Q; i++) {
    L.push(d.setPop(i, `g${espelharIdx(i, eixo)}`));
  }
  return L;
}

/**
 * Entrada: equilíbrio na velocidade prescrita.
 *
 * A alternativa mais precisa é extrapolação de não-equilíbrio (Guo), que
 * preserva o tensor de tensões vindo de dentro do domínio e custa ler as
 * dezenove populações do vizinho. Não vale aqui: a entrada fica a três
 * comprimentos de corpo do nariz, o escoamento que chega nela é uniforme por
 * construção, e não há tensão de não-equilíbrio para preservar. Onde o
 * equilíbrio puro machuca é encostado numa parede que cisalha — e é
 * exatamente por isso que o piso da entrada acompanha a esteira.
 *
 * `uEntrada` já vem com o perfil de camada limite aplicado pelo chamador.
 */
export function blocoEntrada(d) {
  const L = [];
  L.push(d.comentario('entrada: equilíbrio na velocidade prescrita, rho = 1'));
  L.push(d.letF32('deltaIn', '0.0'));
  L.push(d.letF32('rhoIn', '1.0'));
  L.push(d.letF32('uuIn', 'dot(uEntrada, uEntrada)'));
  for (let i = 0; i < Q; i++) {
    if (i === 0) {
      L.push(d.setPop(0, `${num(W[0])} * (deltaIn - 1.5 * rhoIn * uuIn)`));
      continue;
    }
    const cu = dotC(i, 'uEntrada');
    L.push(d.letF32(`ci${i}`, cu));
    L.push(d.setPop(i, `${num(W[i])} * (deltaIn + rhoIn * ` +
      `(3.0 * ci${i} + 4.5 * ci${i} * ci${i} - 1.5 * uuIn))`));
  }
  return L;
}

/**
 * Saída: equilíbrio na velocidade local, densidade de referência.
 *
 * Sozinha isto refletiria — e a onda refletida sobe o domínio, alcança o corpo
 * e contamina o arrasto com uma oscilação que parece desprendimento de vórtice
 * e não é. Quem faz o trabalho de verdade é a camada esponja (ver
 * blocoEsponja): quando a esteira chega neste plano ela já foi absorvida.
 */
export function blocoSaida(d) {
  const L = [];
  L.push(d.comentario('saída: gradiente nulo em u, pressão de referência'));
  for (let i = 0; i < Q; i++) {
    if (i === 0) {
      L.push(d.setPop(0, `${num(W[0])} * (-1.5 * uu)`));
      continue;
    }
    L.push(d.setPop(i, `${num(W[i])} * ` +
      `(3.0 * cu${i} + 4.5 * cu${i} * cu${i} - 1.5 * uu)`));
  }
  return L;
}

/**
 * Camada esponja: mistura o estado pós-colisão com o equilíbrio da corrente
 * livre, com peso crescendo até 1 na saída.
 *
 * É o absorvedor. Sem ele o domínio é uma caixa com paredes acústicas: cada
 * vórtice que atinge a saída volta, e o Cd ganha uma oscilação espúria com o
 * período da travessia do domínio — que é fácil de confundir com Strouhal,
 * porque tem a mesma cara num gráfico.
 *
 * A rampa é quadrática de propósito. Linear introduz um degrau de derivada no
 * começo da esponja, e um degrau também reflete, só que menos.
 */
export function blocoEsponja(d) {
  const L = [];
  L.push(d.comentario('esponja: absorve a esteira antes que ela reflita na saída'));
  L.push(d.letF32('sx', '(f32(gid.x) - uSpongeStart) / max(uSpongeLen, 1.0)'));
  L.push(d.letF32('sigma', 'clamp(sx, 0.0, 1.0)'));
  L.push(d.letF32('sig2', 'sigma * sigma'));
  L.push(d.letF32('uuInf', 'dot(uInf, uInf)'));
  for (let i = 0; i < Q; i++) {
    if (i === 0) {
      L.push(d.letF32('t0', `${num(W[0])} * (-1.5 * uuInf)`));
      L.push(d.setPop(0, `mix(${d.lerPopDst(0)}, t0, sig2)`));
      continue;
    }
    const cu = dotC(i, 'uInf');
    L.push(d.letF32(`si${i}`, cu));
    L.push(d.letF32(`t${i}`, `${num(W[i])} * ` +
      `(3.0 * si${i} + 4.5 * si${i} * si${i} - 1.5 * uuInf)`));
    L.push(d.setPop(i, `mix(${d.lerPopDst(i)}, t${i}, sig2)`));
  }
  return L;
}

/* ──────────────────────────────────────────────────────────── passo inteiro */

/**
 * Corpo completo do kernel de passo, em linhas.
 *
 * O dialeto fornece: comentario, letF32, letVec3, vec3, lerPop, lerPopDst,
 * setPop, vizinho, se/senaoSe/senao/fimSe e indentar. Nada mais deste arquivo
 * conhece a linguagem alvo.
 *
 * A ordem dos ramos importa para o desempenho, não para a correção: o caso
 * FLUIDO é a esmagadora maioria das células e fica por último como `else`,
 * onde não paga nenhum teste.
 */
export function emitirPasso(d, { comEsponja = true } = {}) {
  const L = [];
  L.push(...blocoPull(d));
  L.push('');
  L.push(...blocoMomentos(d));
  L.push('');

  const ramo = (cond, corpo) => {
    L.push(...(L._aberto ? d.senaoSe(cond) : d.se(cond)));
    L._aberto = true;
    L.push(...corpo.map(s => d.indentar(s)));
  };

  ramo(d.ehTipo('SOLIDO') + ' || ' + d.ehTipo('SOLIDO_MOVEL'), blocoBounceBack(d));
  ramo(d.ehTipo('ESPELHO_Y'), blocoEspelho(d, 1));
  ramo(d.ehTipo('ESPELHO_Z'), blocoEspelho(d, 2));
  ramo(d.ehTipo('ENTRADA'), blocoEntrada(d));

  /* SAIDA usa cu{i} e uu, que blocoEquilibrio ainda não declarou neste ramo;
   * por isso ela reaproveita os momentos e emite os c.u que precisa. */
  const saida = [];
  for (let i = 1; i < Q; i++) saida.push(d.letF32(`cu${i}`, dotC(i, 'u')));
  saida.push(...blocoSaida(d));
  ramo(d.ehTipo('SAIDA'), saida);

  const fluido = [];
  fluido.push(...blocoEquilibrio(d));
  fluido.push('');
  fluido.push(...blocoLES(d));
  fluido.push('');
  fluido.push(...blocoColisao(d));
  if (comEsponja) {
    fluido.push('');
    fluido.push(...blocoEsponja(d));
  }
  L.push(...d.senao());
  L.push(...fluido.map(s => d.indentar(s)));
  L.push(...d.fimSe());

  delete L._aberto;
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
