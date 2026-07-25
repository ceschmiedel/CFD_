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

/**
 * Inversa de uma 4x4 geral, por cofatores.
 *
 * Serve para reconstruir o raio de cada pixel a partir do NDC — é assim que o
 * ray-march da fumaça sabe para onde olhar. Uma inversa genérica e não a
 * decomposição "transposta da rotação, menos a translação": a matriz aqui é
 * viewProj, que inclui a projeção, e a versão barata só vale para matrizes de
 * corpo rígido.
 */
export function inversa(m) {
  const a00=m[0],a01=m[1],a02=m[2],a03=m[3];
  const a10=m[4],a11=m[5],a12=m[6],a13=m[7];
  const a20=m[8],a21=m[9],a22=m[10],a23=m[11];
  const a30=m[12],a31=m[13],a32=m[14],a33=m[15];

  const b00=a00*a11-a01*a10, b01=a00*a12-a02*a10, b02=a00*a13-a03*a10;
  const b03=a01*a12-a02*a11, b04=a01*a13-a03*a11, b05=a02*a13-a03*a12;
  const b06=a20*a31-a21*a30, b07=a20*a32-a22*a30, b08=a20*a33-a23*a30;
  const b09=a21*a32-a22*a31, b10=a21*a33-a23*a31, b11=a22*a33-a23*a32;

  let det = b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06;
  if (!det) return identidade();
  det = 1 / det;

  return new Float32Array([
    (a11*b11 - a12*b10 + a13*b09)*det, (a02*b10 - a01*b11 - a03*b09)*det,
    (a31*b05 - a32*b04 + a33*b03)*det, (a22*b04 - a21*b05 - a23*b03)*det,
    (a12*b08 - a10*b11 - a13*b07)*det, (a00*b11 - a02*b08 + a03*b07)*det,
    (a32*b02 - a30*b05 - a33*b01)*det, (a20*b05 - a22*b02 + a23*b01)*det,
    (a10*b10 - a11*b08 + a13*b06)*det, (a01*b08 - a00*b10 - a03*b06)*det,
    (a30*b04 - a31*b02 + a33*b00)*det, (a21*b02 - a20*b04 - a23*b00)*det,
    (a11*b07 - a10*b09 - a12*b06)*det, (a00*b09 - a01*b07 + a02*b06)*det,
    (a31*b01 - a30*b03 - a32*b00)*det, (a20*b03 - a21*b01 + a22*b00)*det,
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
