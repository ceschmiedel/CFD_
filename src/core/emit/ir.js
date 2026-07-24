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
 *
 * SOLIDO é o CORPO, e só ele. PAREDE é uma parede de túnel parada, com
 * exatamente o mesmo bounce-back. A distinção não é numérica — é contábil, e é
 * a diferença entre um coeficiente e um número sem sentido.
 *
 * O cálculo de força soma sobre os links que cruzam uma superfície sólida. Se
 * o piso do túnel for do mesmo tipo que o carro, ele entra na soma: o piso tem
 * nx*ny células de superfície contra alguns milhares do corpo, e o resultado é
 * o arrasto do CHÃO, com o carro como ruído. Medimos exatamente isso antes da
 * separação existir — o coeficiente de sustentação saiu em -81 112, e a única
 * pista de que estava errado era a magnitude.
 */
export const CELULA = {
  FLUIDO: 0,
  SOLIDO: 1,          // o corpo: bounce-back E contabilizado nas forças
  ENTRADA: 2,
  SAIDA: 3,
  SOLIDO_MOVEL: 4,    // esteira rolante: bounce-back com parede móvel, sem força
  ESPELHO_Y: 5,
  ESPELHO_Z: 6,
  PAREDE: 7,          // parede de túnel parada: bounce-back, sem força
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

/**
 * Streaming com BOUNCE-BACK NO NÓ DE FLUIDO.
 *
 * Cada população vem do vizinho a montante — a menos que esse vizinho seja
 * sólido, e nesse caso ela é o que esta própria célula mandou na direção
 * oposta no passo anterior, voltando refletida.
 *
 *     g_i(x) = g_opp(i)^pos-colisão(x)  +  2 w_i rho (c_i . u_parede) / c_s^2
 *
 * POR QUE ASSIM, E NÃO REFLETINDO NO NÓ SÓLIDO
 * -------------------------------------------
 * A formulação anterior tratava o sólido como um nó que participa do passo:
 * ele puxava dos vizinhos, invertia os pares e escrevia de volta. É a versão
 * que aparece em todo tutorial, e funciona — com parede PARADA.
 *
 * Com parede móvel ela vaza. O nó do piso está na borda do domínio, e as
 * direções que apontam para fora dele não têm de onde puxar: com o pull
 * grampeado, a célula lê a si mesma. Então os pares para cima e para baixo
 * simplesmente trocam de lugar a cada passo, e o termo de esteira soma momento
 * nesse par toda vez. Um laço fechado recebendo energia sem nada que dissipe.
 * Com o piso parado o termo é zero e o laço é inofensivo — foi exatamente essa
 * a variável que separou o caso estável do instável na medição.
 *
 * Restringir a injeção aos links que terminam em fluido adiou o problema de
 * 1500 para 4500 passos, e não o resolveu: o par continua se auto-alimentando,
 * só que por um caminho a mais.
 *
 * Puxando com bounce-back, o nó sólido deixa de ter estado. Nenhuma população
 * dentro do corpo é lida por ninguém, não há interior para divergir, não há
 * laço para realimentar, e a condição de contorno vira o que ela é de fato:
 * uma regra sobre o que o FLUIDO recebe da parede.
 *
 * rho na parede é aproximado por 1. O erro é da ordem de delta, ou seja 1e-4,
 * contra a alternativa de um segundo passe só para conhecer a densidade antes
 * de montar o streaming.
 */
export function blocoPull(d) {
  const L = [];
  L.push(d.comentario('streaming, com bounce-back onde o vizinho a montante é sólido'));
  for (let i = 0; i < Q; i++) {
    const c = C[i];
    if (i === 0) {
      L.push(d.letF32('g0', d.lerPop(0, 'cell')));
      continue;
    }
    const j = OPP[i];
    const fonte = d.vizinho(-c[0], -c[1], -c[2]);
    L.push(d.letF32(`s${i}`, fonte));
    L.push(d.letU32(`t${i}`, d.tipoEm(`s${i}`)));
    L.push(d.letVec3(`uw${i}`, d.velParedeDe(`t${i}`)));
    L.push(d.letF32(`g${i}`,
      `select(${d.lerPop(i, `s${i}`)}, ` +
      `${d.lerPop(j, 'cell')} + ${num(2 * W[i] * INV_CS2)} * (${dotC(i, `uw${i}`)}), ` +
      `${d.ehSolidoTipo(`t${i}`)})`));
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
  L.push(d.setPop(0, 'g0'));

  for (let i = 1; i < Q; i++) {
    const j = OPP[i];
    const c = C[i];
    const cu = dotC(i, 'uWall');
    /*
     * O termo de parede móvel só entra se a população que sai daqui na direção
     * i for de fato chegar em FLUIDO.
     *
     * POR QUE O PORTÃO EXISTE
     * -----------------------
     * Sem ele, a esteira injeta momento em TODA direção de todo nó de parede,
     * inclusive nas que apontam para dentro de outro sólido ou para fora do
     * domínio. Essas populações nunca alcançam fluido: elas voltam ao próprio
     * nó no passo seguinte — o pull é grampeado na borda, então a célula do
     * piso lê a si mesma — e recebem a injeção de novo. É um laço fechado
     * ganhando momento todo passo, sem nada que dissipe.
     *
     * Com parede PARADA o termo é nulo e o laço é inofensivo, o que explica
     * por que isto passou despercebido: o solver era estável com o piso
     * parado a ω = 1,90 e divergia com a esteira ligada no mesmo ω. Foi a
     * única variável que separou os dois casos.
     *
     * Com o portão, a injeção acontece só onde ela significa alguma coisa: no
     * link que entrega momento ao escoamento.
     */
    const viz = d.vizinho(c[0], c[1], c[2]);
    L.push(d.letF32(`w${i}`,
      `select(0.0, ${num(2 * W[i] * INV_CS2)} * rho * (${cu}), ` +
      `${d.ehFluidoEm(viz)})`));
    L.push(d.setPop(i, `g${j} + w${i}`));
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
