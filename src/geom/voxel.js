/* ── src/geom/voxel.js ───────────────────────────────────────────────────────
 *
 * Da malha triangular ao campo de sólido do lattice.
 *
 *
 * POR QUE NÃO WINDING NUMBER, NEM PSEUDO-NORMAIS
 * ----------------------------------------------
 * A pergunta "este voxel está dentro do corpo?" tem duas respostas clássicas,
 * e as duas assumem coisas que os modelos reais não cumprem.
 *
 * As PSEUDO-NORMAIS ponderadas por ângulo (Bærentzen & Aanæs) dão o sinal a
 * partir da normal do elemento mais próximo. Exigem uma malha fechada e com
 * orientação consistente. Um GLB de carro de F1 é sopa de triângulos: milhares
 * de peças finas desconectadas, normais apontando para os dois lados, asas que
 * são superfícies abertas sem espessura nenhuma. A pseudo-normal ali devolve
 * sinal aleatório, e o resultado é um corpo com buracos e ilhas de "dentro"
 * espalhadas pelo ar.
 *
 * O WINDING NUMBER GENERALIZADO (Jacobson et al. 2013) tolera malha aberta e
 * é a resposta certa para muita coisa — mas para uma asa de espessura zero ele
 * devolve ~0 dos dois lados, porque não há volume envolvido. E uma asa de
 * espessura zero é precisamente o que o exportador produziu.
 *
 * O QUE FAZEMOS
 * -------------
 * O escoamento não pergunta se um ponto está "dentro" no sentido topológico.
 * Ele pergunta se pode chegar lá. Então:
 *
 *   1. Marcar todo voxel que a SUPERFÍCIE ATRAVESSA, com o teste exato de
 *      sobreposição triângulo-caixa (eixo separador). Conservador por
 *      construção: se o triângulo toca a caixa, o voxel é casca. Sem
 *      espessamento, sem vazamento.
 *   2. Inundar a partir da borda do domínio pelos voxels que NÃO são casca.
 *      O que a inundação alcança é EXTERIOR — o ar que o túnel pode ocupar.
 *   3. Sólido = casca + tudo que a inundação não alcançou.
 *
 * Isto é indiferente a orientação de normal, a malha aberta, a
 * auto-interseção e a componentes desconectadas. Uma asa de espessura zero
 * vira uma parede de um voxel de espessura, que é exatamente o que ela deve
 * ser no lattice. E a única falha possível — um furo na casca maior que um
 * voxel, pelo qual a inundação entra e oca o corpo — é detectável: comparamos
 * o volume preenchido com o da casca e avisamos quando o corpo sai oco.
 *
 * A distância assinada de verdade é calculada em BANDA ESTREITA em volta da
 * superfície, por espalhamento a partir dos triângulos. Um BVH daria o mesmo
 * resultado e seria mais lento aqui: espalhar custa O(n_triângulos * r³), com
 * r em torno de 2 células, enquanto consultar o BVH custa O(n_voxels * log n)
 * e n_voxels é a ordem de dez milhões.
 */

/* ─────────────────────────────────────────── geometria elementar exata */

/**
 * Sobreposição triângulo × caixa alinhada aos eixos (Akenine-Möller 2001).
 *
 * Treze eixos separadores: 3 da caixa, 1 da normal do triângulo, 9 dos
 * produtos vetoriais aresta×eixo. Se nenhum separa, há interseção.
 *
 * `v0,v1,v2` já vêm com o centro da caixa na origem, e `h` é o meio-lado.
 */
function trianguloTocaCaixa(v0, v1, v2, h) {
  const e0 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
  const e1 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];
  const e2 = [v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2]];

  /* 9 eixos aresta × eixo-da-caixa */
  const eixo = (a, b, c, fa, fb, i, j) => {
    const p0 = a * v0[i] + b * v0[j];
    const p1 = a * v1[i] + b * v1[j];
    const p2 = a * v2[i] + b * v2[j];
    const min = Math.min(p0, p1, p2), max = Math.max(p0, p1, p2);
    const r = fa * h[i] + fb * h[j];
    return !(min > r || max < -r);
  };

  for (const e of [e0, e1, e2]) {
    const fx = Math.abs(e[0]), fy = Math.abs(e[1]), fz = Math.abs(e[2]);
    if (!eixo(e[2], -e[1], 0, fz, fy, 1, 2)) return false;   // a00..a02
    if (!eixo(-e[2], e[0], 0, fz, fx, 0, 2)) return false;   // a10..a12
    if (!eixo(e[1], -e[0], 0, fy, fx, 0, 1)) return false;   // a20..a22
  }

  /* 3 eixos da própria caixa */
  for (let k = 0; k < 3; k++) {
    if (Math.min(v0[k], v1[k], v2[k]) > h[k]) return false;
    if (Math.max(v0[k], v1[k], v2[k]) < -h[k]) return false;
  }

  /* o eixo da normal do triângulo */
  const n = [
    e0[1] * e1[2] - e0[2] * e1[1],
    e0[2] * e1[0] - e0[0] * e1[2],
    e0[0] * e1[1] - e0[1] * e1[0],
  ];
  const d = n[0] * v0[0] + n[1] * v0[1] + n[2] * v0[2];
  const r = Math.abs(n[0]) * h[0] + Math.abs(n[1]) * h[1] + Math.abs(n[2]) * h[2];
  return Math.abs(d) <= r;
}

/**
 * Distância ao quadrado de um ponto ao triângulo (Ericson, "Real-Time
 * Collision Detection", §5.1.5). Percorre as sete regiões de Voronoi do
 * triângulo — três vértices, três arestas e a face — sem raiz quadrada.
 */
function distQuadPontoTriangulo(p, a, b, c) {
  const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
  const acx = c[0] - a[0], acy = c[1] - a[1], acz = c[2] - a[2];
  const apx = p[0] - a[0], apy = p[1] - a[1], apz = p[2] - a[2];

  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) return apx * apx + apy * apy + apz * apz;

  const bpx = p[0] - b[0], bpy = p[1] - b[1], bpz = p[2] - b[2];
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) return bpx * bpx + bpy * bpy + bpz * bpz;

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    const qx = apx - v * abx, qy = apy - v * aby, qz = apz - v * abz;
    return qx * qx + qy * qy + qz * qz;
  }

  const cpx = p[0] - c[0], cpy = p[1] - c[1], cpz = p[2] - c[2];
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) return cpx * cpx + cpy * cpy + cpz * cpz;

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    const qx = apx - w * acx, qy = apy - w * acy, qz = apz - w * acz;
    return qx * qx + qy * qy + qz * qz;
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    const qx = bpx + w * (cpx - bpx), qy = bpy + w * (cpy - bpy), qz = bpz + w * (cpz - bpz);
    return qx * qx + qy * qy + qz * qz;
  }

  const den = 1 / (va + vb + vc);
  const v = vb * den, w = vc * den;
  const qx = apx - (v * abx + w * acx);
  const qy = apy - (v * aby + w * acy);
  const qz = apz - (v * abz + w * acz);
  return qx * qx + qy * qy + qz * qz;
}

/* ─────────────────────────────────────────────────────────── voxelização */

export const VOXEL = {
  FLUIDO: 0,
  CASCA: 1,     // a superfície atravessa este voxel
  DENTRO: 2,    // a inundação não alcançou
};

/**
 * Marca a casca e calcula a distância não-assinada em banda estreita.
 *
 * @param {Float32Array} pos  posições já no espaço do lattice (célula = 1)
 * @param {Uint32Array} idx
 * @param {object} g  { nx, ny, nz }
 * @param {number} banda  raio da banda de distância, em células
 */
function rasterizar(pos, idx, g, banda) {
  const { nx, ny, nz } = g;
  const N = nx * ny * nz;
  const flags = new Uint8Array(N);
  const dist2 = new Float32Array(N).fill(Infinity);

  const h = [0.5, 0.5, 0.5];
  const a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0];
  const la = [0, 0, 0], lb = [0, 0, 0], lc = [0, 0, 0];
  const p = [0, 0, 0];
  const banda2 = banda * banda;

  let fora = 0;
  const nTri = idx.length / 3;

  for (let t = 0; t < nTri; t++) {
    const i0 = idx[t * 3] * 3, i1 = idx[t * 3 + 1] * 3, i2 = idx[t * 3 + 2] * 3;
    a[0] = pos[i0]; a[1] = pos[i0 + 1]; a[2] = pos[i0 + 2];
    b[0] = pos[i1]; b[1] = pos[i1 + 1]; b[2] = pos[i1 + 2];
    c[0] = pos[i2]; c[1] = pos[i2 + 1]; c[2] = pos[i2 + 2];

    if (!Number.isFinite(a[0] + a[1] + a[2] + b[0] + b[1] + b[2] + c[0] + c[1] + c[2])) {
      continue;
    }

    /* caixa do triângulo, dilatada pela banda, recortada no domínio */
    const x0 = Math.max(0, Math.floor(Math.min(a[0], b[0], c[0]) - banda));
    const x1 = Math.min(nx - 1, Math.ceil(Math.max(a[0], b[0], c[0]) + banda));
    const y0 = Math.max(0, Math.floor(Math.min(a[1], b[1], c[1]) - banda));
    const y1 = Math.min(ny - 1, Math.ceil(Math.max(a[1], b[1], c[1]) + banda));
    const z0 = Math.max(0, Math.floor(Math.min(a[2], b[2], c[2]) - banda));
    const z1 = Math.min(nz - 1, Math.ceil(Math.max(a[2], b[2], c[2]) + banda));
    if (x1 < x0 || y1 < y0 || z1 < z0) { fora++; continue; }

    for (let z = z0; z <= z1; z++) {
      for (let y = y0; y <= y1; y++) {
        const base = (z * ny + y) * nx;
        for (let x = x0; x <= x1; x++) {
          p[0] = x + 0.5; p[1] = y + 0.5; p[2] = z + 0.5;
          const cell = base + x;

          const d2 = distQuadPontoTriangulo(p, a, b, c);
          if (d2 < dist2[cell]) dist2[cell] = d2;

          /* Casca: teste exato de sobreposição, não limiar de distância.
           * Um limiar erra dos dois lados — engrossa paredes finas e deixa
           * passar triângulos que cortam o canto do voxel. */
          if (d2 <= 0.75 && flags[cell] !== VOXEL.CASCA) {
            la[0] = a[0] - p[0]; la[1] = a[1] - p[1]; la[2] = a[2] - p[2];
            lb[0] = b[0] - p[0]; lb[1] = b[1] - p[1]; lb[2] = b[2] - p[2];
            lc[0] = c[0] - p[0]; lc[1] = c[1] - p[1]; lc[2] = c[2] - p[2];
            if (trianguloTocaCaixa(la, lb, lc, h)) flags[cell] = VOXEL.CASCA;
          }
        }
      }
    }
  }

  return { flags, dist2, trianguloFora: fora, banda2 };
}

/**
 * Dilata a casca por uma célula em vizinhança de 26.
 *
 * Serve para SELAR o corpo antes da inundação, e é a diferença entre um carro
 * e um carro furado. Um GLB de veículo é montado em painéis — capô, portas,
 * para-brisa — que se encontram com folgas de fração de milímetro no modelo e
 * que, na resolução do lattice, viram fendas por onde a inundação entra e oca
 * o corpo inteiro. Um corpo oco não é um erro cosmético: o ar passa POR DENTRO
 * dele, e o arrasto sai errado para menos.
 *
 * A dilatação é usada só como barreira para a inundação. A casca final volta a
 * ser a original, então a geometria não engorda — o corpo é selado sem ficar
 * uma célula maior em cada direção.
 */
function dilatar(flags, g) {
  const { nx, ny, nz } = g;
  const nxny = nx * ny;
  const d = new Uint8Array(flags.length);

  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      const base = (z * ny + y) * nx;
      for (let x = 0; x < nx; x++) {
        if (flags[base + x] !== VOXEL.CASCA) continue;
        const z0 = Math.max(0, z - 1), z1 = Math.min(nz - 1, z + 1);
        const y0 = Math.max(0, y - 1), y1 = Math.min(ny - 1, y + 1);
        const x0 = Math.max(0, x - 1), x1 = Math.min(nx - 1, x + 1);
        for (let k = z0; k <= z1; k++) {
          for (let j = y0; j <= y1; j++) {
            const b = (k * ny + j) * nx;
            for (let i = x0; i <= x1; i++) d[b + i] = 1;
          }
        }
      }
    }
  }
  return d;
}

/**
 * Inunda o exterior a partir da borda do domínio.
 *
 * BFS com pilha explícita e um Int32Array — a recursão estoura a pilha do JS
 * em qualquer domínio de tamanho real, e a fila como Array de JS aloca demais.
 *
 * `barreira[i]` verdadeiro impede a passagem. Passamos a casca dilatada, não a
 * casca crua, para as fendas de subvoxel não vazarem.
 */
function inundarExterior(barreira, g) {
  const { nx, ny, nz } = g;
  const N = nx * ny * nz;
  const visitado = new Uint8Array(N);
  const pilha = new Int32Array(N);
  let topo = 0;

  const empilhar = (cell) => {
    if (visitado[cell] || barreira[cell]) return;
    visitado[cell] = 1;
    pilha[topo++] = cell;
  };

  /* as seis faces do domínio */
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      empilhar((z * ny + y) * nx);
      empilhar((z * ny + y) * nx + nx - 1);
    }
  }
  for (let z = 0; z < nz; z++) {
    for (let x = 0; x < nx; x++) {
      empilhar((z * ny) * nx + x);
      empilhar((z * ny + ny - 1) * nx + x);
    }
  }
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      empilhar(y * nx + x);
      empilhar(((nz - 1) * ny + y) * nx + x);
    }
  }

  const nxny = nx * ny;
  while (topo > 0) {
    const cell = pilha[--topo];
    const x = cell % nx;
    const y = ((cell - x) / nx) % ny;
    const z = (cell - x - y * nx) / nxny;

    if (x > 0) empilhar(cell - 1);
    if (x < nx - 1) empilhar(cell + 1);
    if (y > 0) empilhar(cell - nx);
    if (y < ny - 1) empilhar(cell + nx);
    if (z > 0) empilhar(cell - nxny);
    if (z < nz - 1) empilhar(cell + nxny);
  }

  return visitado;   // 1 = exterior alcançável
}

/**
 * Cresce o exterior uma célula, atravessando apenas células que não são casca.
 * Modifica `exterior` no lugar. Ver a chamada em voxelizar() para o porquê.
 */
function recuperarPele(exterior, flags, g) {
  const { nx, ny, nz } = g;
  const nxny = nx * ny;
  /* Cópia: sem ela o crescimento se propagaria em cascata dentro do mesmo
   * passe, atravessando fendas de várias células e desfazendo a selagem. */
  const antes = exterior.slice();

  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      const base = (z * ny + y) * nx;
      for (let x = 0; x < nx; x++) {
        const c = base + x;
        if (antes[c] || flags[c] === VOXEL.CASCA) continue;
        const viz = (
          (x > 0 && antes[c - 1]) || (x < nx - 1 && antes[c + 1]) ||
          (y > 0 && antes[c - nx]) || (y < ny - 1 && antes[c + nx]) ||
          (z > 0 && antes[c - nxny]) || (z < nz - 1 && antes[c + nxny])
        );
        if (viz) exterior[c] = 1;
      }
    }
  }
}

/**
 * Voxeliza uma malha no lattice.
 *
 * @param {Float32Array} positions  já no espaço do lattice (uma célula = 1)
 * @param {Uint32Array} indices
 * @param {object} g  { nx, ny, nz }
 * @param {object} [opt]
 * @param {number} [opt.banda]  raio da banda de SDF, em células
 * @returns {{ tipo: Uint8Array, sdf: Float32Array, stats: object }}
 *
 * O padrão de `banda` é 1.5 e não é uma escolha de precisão — é de custo. A
 * classificação sólido/fluido precisa de banda ~1; a banda mais larga só
 * serve à renderização. E o custo é cúbico nela: com 1,5 milhão de triângulos
 * (um GLB de F1), banda 3 dilata a caixa de cada triângulo para ~343 voxels e
 * dá meio bilhão de avaliações de distância ponto-triângulo; banda 1.5 dá 64,
 * e umas noventa milhões. Quando a renderização quiser SDF longe da
 * superfície, o caminho barato é varrer para fora a partir desta banda
 * (transformada de distância), não voltar aos triângulos.
 */
export function voxelizar(positions, indices, g, { banda = 1.5, selarFrestas = true } = {}) {
  const t0 = performance.now();
  const { nx, ny, nz } = g;
  const N = nx * ny * nz;

  const { flags, dist2, trianguloFora } = rasterizar(positions, indices, g, banda);
  const tRaster = performance.now();

  const barreira = selarFrestas ? dilatar(flags, g) : flags;
  const exterior = inundarExterior(barreira, g);

  /*
   * A barreira dilatada para a inundação uma célula antes da superfície, então
   * o que ela alcançou está faltando uma pele de uma célula em volta do corpo
   * inteiro — e essa pele, classificada como "não alcançada", viraria sólido.
   * O corpo engordaria uma célula em todas as direções e o Cd subiria junto.
   *
   * Uma única dilatação do EXTERIOR, restrita a células que não são casca
   * original, devolve exatamente essa pele. E não reabre as fendas: para
   * atravessar uma fenda o exterior teria de passar por uma célula de casca,
   * que esta dilatação não atravessa.
   */
  if (selarFrestas) recuperarPele(exterior, flags, g);
  const tInundar = performance.now();

  /* Sólido = casca + o que a inundação não alcançou. */
  const tipo = new Uint8Array(N);
  let nCasca = 0, nDentro = 0;
  for (let i = 0; i < N; i++) {
    if (flags[i] === VOXEL.CASCA) { tipo[i] = VOXEL.CASCA; nCasca++; }
    else if (!exterior[i]) { tipo[i] = VOXEL.DENTRO; nDentro++; }
  }

  /* SDF assinada, negativa dentro, só válida dentro da banda. */
  const sdf = new Float32Array(N);
  const foraDaBanda = banda + 1;
  for (let i = 0; i < N; i++) {
    const d = dist2[i] === Infinity ? foraDaBanda : Math.sqrt(dist2[i]);
    sdf[i] = tipo[i] === VOXEL.FLUIDO ? d : -d;
  }

  const nSolido = nCasca + nDentro;
  const stats = {
    nCelulas: N,
    nCasca, nDentro, nSolido,
    fracaoSolida: nSolido / N,
    trianguloFora,
    /* Um corpo fechado tem muito mais interior que casca. Se a casca domina,
     * ou o corpo é genuinamente uma superfície fina (uma asa, uma placa) ou a
     * inundação vazou por um furo — e a interface precisa saber qual. */
    ocoSuspeito: nDentro < nCasca * 0.25 && nCasca > 500,
    msRasterizacao: tRaster - t0,
    msInundacao: tInundar - tRaster,
    msTotal: performance.now() - t0,
  };

  return { tipo, sdf, stats };
}

/**
 * Área frontal projetada, em células², contando a sombra do sólido no plano
 * perpendicular ao escoamento.
 *
 * Este é o denominador do Cd, e é onde o solver 2D antigo errava: ele usava a
 * ALTURA do corpo (número de linhas ocupadas) como se fosse área. A área
 * frontal de verdade é a projeção em YZ, que só existe em três dimensões.
 */
export function areaFrontal(tipo, g) {
  const { nx, ny, nz } = g;
  const sombra = new Uint8Array(ny * nz);
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      const base = (z * ny + y) * nx;
      for (let x = 0; x < nx; x++) {
        if (tipo[base + x] !== VOXEL.FLUIDO) { sombra[z * ny + y] = 1; break; }
      }
    }
  }
  let n = 0;
  for (let i = 0; i < sombra.length; i++) n += sombra[i];
  return n;
}
