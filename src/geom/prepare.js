/* ── src/geom/prepare.js ─────────────────────────────────────────────────────
 *
 * Da malha como veio do exportador para a malha posicionada no túnel.
 *
 * Três perguntas, nenhuma respondida pelo arquivo:
 *
 *   1. Qual eixo é a direção do escoamento?
 *   2. Qual eixo é para cima?
 *   3. Um metro no arquivo é um metro?
 *
 * Nenhum formato de malha carrega essa informação de modo confiável. glTF diz
 * que é Y-para-cima e metros, e uma parte dos arquivos por aí não é nem uma
 * coisa nem outra. STL não diz nada — nem unidade, nem orientação, nem sequer
 * se as normais apontam para fora. Então adivinhamos, e mostramos o palpite
 * para o usuário poder corrigi-lo em vez de descobrir depois que simulou o
 * carro de lado.
 *
 *
 * COMO ADIVINHAR O EIXO VERTICAL
 * ------------------------------
 * O caminho óbvio — "maior extensão é o comprimento, menor é a altura" — é o
 * que o solver antigo deste repositório usava, e ele erra sempre que a segunda
 * e a terceira dimensão são parecidas. Um carro largo e baixo funciona; uma
 * asa, um caminhão ou uma moto, não.
 *
 * O sinal que funciona é SIMETRIA. Quase todo veículo é espelhado em torno do
 * plano longitudinal e não é espelhado em torno do horizontal — um carro tem
 * teto em cima e rodas embaixo. Então: das duas dimensões que sobram depois de
 * escolher o comprimento, a LATERAL é aquela em torno da qual o modelo é mais
 * simétrico, e a outra é a VERTICAL.
 *
 * Medimos isso numa grade de ocupação grosseira, não vértice a vértice: é
 * imune a densidade irregular de malha (um carro tem dez mil triângulos na
 * grade do radiador e doze no capô) e custa um passe.
 */

import { extentos } from './parse.js';

/* ────────────────────────────────────────────────────── palpites de eixo */

/**
 * Ocupação numa grade K³ sobre a caixa envolvente. Grosseira de propósito.
 */
function ocupacao(pos, ext, K = 24) {
  const oc = new Uint8Array(K * K * K);
  const inv = ext.tamanho.map(t => (t > 0 ? K / t : 0));
  for (let i = 0; i < pos.length; i += 3) {
    const x = pos[i], y = pos[i + 1], z = pos[i + 2];
    if (!Number.isFinite(x + y + z)) continue;
    const a = Math.min(K - 1, Math.max(0, ((x - ext.min[0]) * inv[0]) | 0));
    const b = Math.min(K - 1, Math.max(0, ((y - ext.min[1]) * inv[1]) | 0));
    const c = Math.min(K - 1, Math.max(0, ((z - ext.min[2]) * inv[2]) | 0));
    oc[(c * K + b) * K + a] = 1;
  }
  return { oc, K };
}

/** Fração de voxels ocupados que têm par espelhado no eixo `eixo`. */
function simetria({ oc, K }, eixo) {
  let ocupados = 0, casados = 0;
  for (let c = 0; c < K; c++) {
    for (let b = 0; b < K; b++) {
      for (let a = 0; a < K; a++) {
        if (!oc[(c * K + b) * K + a]) continue;
        ocupados++;
        const A = eixo === 0 ? K - 1 - a : a;
        const B = eixo === 1 ? K - 1 - b : b;
        const Cc = eixo === 2 ? K - 1 - c : c;
        if (oc[(Cc * K + B) * K + A]) casados++;
      }
    }
  }
  return ocupados ? casados / ocupados : 0;
}

/**
 * Decide a permutação de eixos que leva a malha à convenção do túnel:
 * X = escoamento, Y = lateral, Z = vertical.
 *
 * @returns {{ordem: number[], flip: number[], simetrias: number[], confianca: number}}
 */
export function palpitarEixos(pos, ext) {
  /* comprimento = maior extensão. Este palpite raramente erra: se o corpo é
   * mais alto ou mais largo do que comprido, ele não é um veículo. */
  const ordemPorTamanho = [0, 1, 2].sort((a, b) => ext.tamanho[b] - ext.tamanho[a]);
  const eixoFluxo = ordemPorTamanho[0];
  const restantes = [0, 1, 2].filter(a => a !== eixoFluxo);

  const grade = ocupacao(pos, ext);
  const sim = [0, 1, 2].map(a => simetria(grade, a));

  /* dos dois que sobram, o mais simétrico é o lateral */
  const [r0, r1] = restantes;
  const eixoLateral = sim[r0] >= sim[r1] ? r0 : r1;
  const eixoVertical = eixoLateral === r0 ? r1 : r0;

  /* A confiança é a separação entre as duas simetrias. Perto de zero, o
   * modelo é simétrico (ou assimétrico) nos dois eixos por igual e o palpite
   * é um cara-ou-coroa — a interface mostra isso e oferece o giro manual. */
  const confianca = Math.abs(sim[r0] - sim[r1]);

  return {
    ordem: [eixoFluxo, eixoLateral, eixoVertical],
    simetrias: sim,
    confianca,
    /* Um veículo tem mais massa embaixo (rodas, assoalho) que em cima. Se o
     * centroide estiver acima do meio da caixa, o modelo provavelmente está
     * de cabeça para baixo. */
    talvezInvertido: null,
  };
}

/* ────────────────────────────────────────────────────── palpite de unidade */

/**
 * Adivinha a escala pelo tamanho absoluto, comparando com o comprimento
 * plausível de um veículo (~4,5 m).
 *
 * Um número aqui não é "a unidade do arquivo": é o fator que leva o arquivo a
 * metros. Modelos normalizados para 1.0 são comuníssimos (todo pipeline de
 * jogo faz isso) e ficam registrados como tal, porque nesse caso o usuário
 * PRECISA informar o comprimento real — não há como deduzi-lo.
 */
export function palpitarUnidade(comprimento) {
  const candidatos = [
    { fator: 1, nome: 'metros', tipico: [0.5, 30] },
    { fator: 0.01, nome: 'centímetros', tipico: [50, 3000] },
    { fator: 0.001, nome: 'milímetros', tipico: [500, 30000] },
    { fator: 0.0254, nome: 'polegadas', tipico: [20, 1200] },
  ];
  for (const c of candidatos) {
    if (comprimento >= c.tipico[0] && comprimento <= c.tipico[1]) {
      return { ...c, normalizado: false, confianca: 'plausível' };
    }
  }
  if (comprimento > 0.2 && comprimento < 2.5) {
    return {
      fator: 1, nome: 'normalizado', normalizado: true, confianca: 'nenhuma',
      aviso: `maior dimensão = ${comprimento.toFixed(3)}; o modelo parece ` +
        'normalizado. Informe o comprimento real do corpo — Re e Cd dependem dele.',
    };
  }
  return {
    fator: 1, nome: 'desconhecida', normalizado: true, confianca: 'nenhuma',
    aviso: `maior dimensão = ${comprimento.toPrecision(4)}, fora de qualquer ` +
      'faixa típica. Informe o comprimento real do corpo.',
  };
}

/* ───────────────────────────────────────────────── colocação no domínio */

/**
 * Reorienta, escala e posiciona a malha no espaço do lattice (uma célula = 1).
 *
 * Devolve uma CÓPIA das posições. O original fica intacto para poder ser
 * reposicionado quando o usuário mudar a resolução ou o ângulo de guinada sem
 * reler o arquivo.
 *
 * @param {object} o
 * @param {Float32Array} o.positions
 * @param {number[]} o.ordem            permutação de eixos (de palpitarEixos)
 * @param {boolean[]} [o.espelhar]      inverter cada eixo após permutar
 * @param {object} o.grade              { nx, ny, nz }
 * @param {number} o.celulasPorComprimento
 * @param {number} [o.fracaoEntrada]    onde o nariz fica, em frações de nx
 * @param {boolean} [o.assentarNoPiso]  encostar o corpo em z = piso
 * @param {number} [o.folgaPiso]        células entre o corpo e o piso
 * @param {number} [o.guinadaGraus]     rotação em torno do eixo vertical
 */
export function colocar({
  positions, ordem, espelhar = [false, false, false],
  grade, celulasPorComprimento,
  fracaoEntrada = 0.28, assentarNoPiso = true, folgaPiso = 1,
  guinadaGraus = 0,
}) {
  const n = positions.length / 3;
  const saida = new Float32Array(positions.length);

  /* 1. permutar e espelhar */
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < 3; a++) {
      const v = positions[i * 3 + ordem[a]];
      saida[i * 3 + a] = espelhar[a] ? -v : v;
    }
  }

  /* 2. guinada em torno do vertical (z), antes de escalar */
  if (guinadaGraus) {
    const t = guinadaGraus * Math.PI / 180;
    const cs = Math.cos(t), sn = Math.sin(t);
    const ext0 = extentos(saida);
    const cx = ext0.centro[0], cy = ext0.centro[1];
    for (let i = 0; i < n; i++) {
      const x = saida[i * 3] - cx, y = saida[i * 3 + 1] - cy;
      saida[i * 3] = cx + x * cs - y * sn;
      saida[i * 3 + 1] = cy + x * sn + y * cs;
    }
  }

  /* 3. escalar para que o comprimento ocupe celulasPorComprimento células */
  const ext = extentos(saida);
  const comprimento = Math.max(ext.tamanho[0], 1e-12);
  const s = celulasPorComprimento / comprimento;

  /* 4. transladar: nariz em fracaoEntrada*nx, centrado em y,
   *    assentado no piso (ou centrado em z) */
  const alvoX = grade.nx * fracaoEntrada - ext.min[0] * s;
  const alvoY = grade.ny * 0.5 - ext.centro[1] * s;
  const alvoZ = assentarNoPiso
    ? folgaPiso - ext.min[2] * s
    : grade.nz * 0.5 - ext.centro[2] * s;

  for (let i = 0; i < n; i++) {
    saida[i * 3] = saida[i * 3] * s + alvoX;
    saida[i * 3 + 1] = saida[i * 3 + 1] * s + alvoY;
    saida[i * 3 + 2] = saida[i * 3 + 2] * s + alvoZ;
  }

  const extFinal = extentos(saida);

  /* Bloqueio: a razão entre a área frontal e a da seção de teste. Todo túnel
   * real tem esse número, e acima de ~5% ele muda o Cd o bastante para exigir
   * correção (que forces.js aplica). Aqui só reportamos a estimativa pela
   * caixa; a área frontal exata sai da voxelização. */
  const bloqueioCaixa = (extFinal.tamanho[1] * extFinal.tamanho[2]) /
    (grade.ny * grade.nz);

  const avisos = [];
  for (const [a, nome] of [[0, 'x'], [1, 'y'], [2, 'z']]) {
    const lim = [grade.nx, grade.ny, grade.nz][a];
    if (extFinal.min[a] < 0 || extFinal.max[a] > lim) {
      avisos.push(`o corpo sai do domínio em ${nome} ` +
        `(${extFinal.min[a].toFixed(1)}…${extFinal.max[a].toFixed(1)} de 0…${lim})`);
    }
  }
  if (bloqueioCaixa > 0.10) {
    avisos.push(`bloqueio de ${(bloqueioCaixa * 100).toFixed(1)}% — acima de ` +
      '10% a correção de Maskell deixa de ser confiável; use um domínio maior');
  }

  return { positions: saida, escala: s, extentos: extFinal, bloqueioCaixa, avisos };
}

/**
 * Pipeline completo: das posições cruas ao que a voxelização consome, com
 * todos os palpites explícitos e sobrescrevíveis.
 */
export function preparar(malha, grade, opcoes = {}) {
  const ext = extentos(malha.positions);
  const eixos = opcoes.ordem
    ? { ordem: opcoes.ordem, confianca: 1, simetrias: null, manual: true }
    : palpitarEixos(malha.positions, ext);

  const comprimentoArquivo = ext.tamanho[eixos.ordem[0]];
  const unidade = opcoes.unidade ?? palpitarUnidade(comprimentoArquivo);
  const comprimentoM = opcoes.comprimentoM ?? (comprimentoArquivo * unidade.fator);

  const colocada = colocar({
    positions: malha.positions,
    ordem: eixos.ordem,
    espelhar: opcoes.espelhar,
    grade,
    celulasPorComprimento: opcoes.celulasPorComprimento ?? Math.round(grade.nx * 0.34),
    fracaoEntrada: opcoes.fracaoEntrada,
    assentarNoPiso: opcoes.assentarNoPiso,
    folgaPiso: opcoes.folgaPiso,
    guinadaGraus: opcoes.guinadaGraus,
  });

  return {
    ...colocada,
    indices: malha.indices,
    eixos,
    unidade,
    comprimentoM,
    extentosOriginais: ext,
    avisos: [...(malha.avisos ?? []), ...colocada.avisos,
      ...(unidade.aviso ? [unidade.aviso] : [])],
  };
}
