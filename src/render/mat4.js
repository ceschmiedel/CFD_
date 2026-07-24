/* ── src/render/mat4.js ──────────────────────────────────────────────────────
 *
 * O mínimo de álgebra de matrizes para a câmera. Coluna-maior, como a GPU
 * espera, e sem dependência externa: são setenta linhas contra um pacote
 * inteiro, e um arquivo único que se hospeda em qualquer lugar é uma decisão
 * de distribuição deste projeto, não um detalhe.
 *
 * Profundidade em [0,1] e não [-1,1] — é a convenção do WebGPU (e do Vulkan e
 * do D3D). Usar a matriz de perspectiva do OpenGL aqui não dá erro: dá metade
 * do frustum descartada, com a cena aparecendo cortada de um jeito que parece
 * problema de câmera.
 */

export function identidade() {
  return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
}

export function multiplicar(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c*4+r] = a[r]*b[c*4] + a[4+r]*b[c*4+1] + a[8+r]*b[c*4+2] + a[12+r]*b[c*4+3];
    }
  }
  return o;
}

/** Perspectiva com profundidade em [0,1]. fovY em radianos. */
export function perspectiva(fovY, aspecto, perto, longe) {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (perto - longe);
  return new Float32Array([
    f / aspecto, 0, 0, 0,
    0, f, 0, 0,
    0, 0, longe * nf, -1,
    0, 0, longe * perto * nf, 0,
  ]);
}

export function olharPara(olho, alvo, cima) {
  const z = normalizar(subtrair(olho, alvo));
  const x = normalizar(produtoVetorial(cima, z));
  const y = produtoVetorial(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -ponto(x, olho), -ponto(y, olho), -ponto(z, olho), 1,
  ]);
}

export const subtrair = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
export const ponto = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
export const produtoVetorial = (a, b) => [
  a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0],
];
export function normalizar(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0]/n, v[1]/n, v[2]/n];
}

/**
 * Câmera em órbita.
 *
 * O ângulo polar é limitado para não passar pelos polos: exatamente no polo o
 * vetor "para cima" fica paralelo ao eixo de visão, o produto vetorial que
 * monta a base zera, e a matriz vira NaN — a cena some e volta quando o mouse
 * passa. Limitar a 0,02 rad das extremidades custa nada.
 */
export class Orbita {
  constructor({ alvo = [0,0,0], distancia = 3, azimute = -0.6, polar = 1.15 } = {}) {
    this.alvo = alvo; this.distancia = distancia;
    this.azimute = azimute; this.polar = polar;
  }

  get olho() {
    const sp = Math.sin(this.polar), cp = Math.cos(this.polar);
    return [
      this.alvo[0] + this.distancia * sp * Math.cos(this.azimute),
      this.alvo[1] + this.distancia * sp * Math.sin(this.azimute),
      this.alvo[2] + this.distancia * cp,
    ];
  }

  girar(dAz, dPol) {
    this.azimute += dAz;
    this.polar = Math.min(Math.PI - 0.02, Math.max(0.02, this.polar + dPol));
  }

  aproximar(fator) {
    this.distancia = Math.min(60, Math.max(0.4, this.distancia * fator));
  }

  /** Z é para cima: é a convenção do lattice (ver geom/prepare.js). */
  matriz(aspecto) {
    return multiplicar(
      perspectiva(0.85, aspecto, 0.05, 200),
      olharPara(this.olho, this.alvo, [0, 0, 1]));
  }
}
