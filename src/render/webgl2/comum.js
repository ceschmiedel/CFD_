/* ── src/render/webgl2/comum.js ──────────────────────────────────────────────
 *
 * Os trechos de GLSL compartilhados por todos os passes de desenho do caminho
 * WebGL2: os uniformes da cena, a transformação para o mundo, a amostragem do
 * campo e a paleta.
 *
 * É o par de src/render/comum.js, que faz o mesmo em WGSL. Duas cópias da
 * projeção é como se descobre, três semanas depois, que a fumaça e o carro
 * estão em espaços ligeiramente diferentes — então as duas versões existem
 * lado a lado, curtas o bastante para serem lidas juntas, e qualquer mudança
 * numa pede a outra.
 *
 *
 * A DIFERENÇA QUE ATRAVESSA TUDO
 * ------------------------------
 * No WebGPU o campo macroscópico é um storage buffer indexado por célula. Aqui
 * ele é o mesmo ATLAS 2D do solver: a fatia z mora no ladrilho (z % tx, z / tx).
 * `amostrar` esconde isso — quem desenha continua pedindo a velocidade num
 * ponto de lattice e recebendo um vec4, sem saber onde ele mora.
 */

/** Turbo (Mikhailov 2019). Mesma aproximação polinomial do caminho WGSL. */
export const TURBO = `
vec3 turbo(float t0) {
  float t = clamp(t0, 0.0, 1.0);
  float r = 0.13572138 + t*(4.61539260 + t*(-42.66032258 + t*(132.13108234 + t*(-152.94239396 + t*59.28637943))));
  float g = 0.09140261 + t*(2.19418839 + t*(4.84296658 + t*(-14.18503333 + t*(4.27729857 + t*2.82956604))));
  float b = 0.10667330 + t*(12.64194608 + t*(-60.58204836 + t*(110.36276771 + t*(-89.90310912 + t*27.34824973))));
  return clamp(vec3(r, g, b), vec3(0.0), vec3(1.0));
}`;

/**
 * Uniformes da cena e as funções de espaço.
 *
 * Uniformes soltos e não um bloco uniforme: são uma dúzia de valores escritos
 * uma vez por quadro, o custo é irrelevante, e um UBO acrescentaria uma regra
 * de alinhamento por membro que é justamente a classe de erro que ninguém vê
 * (um vec3 alinhado como vec4 desloca todo o resto e a cena aparece torta).
 */
export const CENA = `
uniform mat4 uViewProj;
uniform mat4 uInvViewProj;
uniform ivec3 uDim;        // nx, ny, nz
uniform ivec2 uTiles;      // ladrilhos do atlas
uniform vec4 uEscala;      // 1/nx, 1/ny, 1/nz, uRef
uniform vec4 uOpcoes;      // modoCor, ganhoCp, quadro, deslocamento do piso
uniform vec3 uOlho;        // posição da câmera, em mundo
uniform vec4 uFumo;        // densidade, anisotropia g, passos, passosLuz
uniform vec4 uRelogio;     // dt do quadro (passos), subpassos, _, _
uniform sampler2D uMacros; // (ux, uy, uz, delta) por célula, no atlas
uniform highp usampler2D uTipoR;

ivec2 emAtlas(ivec3 c) {
  return ivec2(c.x + (c.z % uTiles.x) * uDim.x,
               c.y + (c.z / uTiles.x) * uDim.y);
}

/* Do espaço do lattice para o mundo, com a origem no centro do piso.
 *
 * UM ÚNICO FATOR PARA OS TRÊS EIXOS — dividir cada eixo pela sua dimensão
 * mapeia o domínio para um cubo, e o domínio não é cúbico: num túnel
 * 320x160x128 um carro de proporção 1 : 0,47 : 0,31 sairia desenhado em
 * 1 : 0,94 : 0,78. É o mesmo fator do caminho WGSL, e tem de continuar sendo. */
vec3 paraMundo(vec3 p) {
  float k = uEscala.x * 2.0;
  return vec3((p.x - float(uDim.x) * 0.5) * k,
              (p.y - float(uDim.y) * 0.5) * k,
              p.z * k);
}

vec3 paraLattice(vec3 w) {
  float k = uEscala.x * 2.0;
  return vec3(w.x / k + float(uDim.x) * 0.5,
              w.y / k + float(uDim.y) * 0.5,
              w.z / k);
}

vec4 amostrar(vec3 p) {
  ivec3 q = clamp(ivec3(p), ivec3(0), uDim - ivec3(1));
  return texelFetch(uMacros, emAtlas(q), 0);
}

uint tipoLattice(vec3 p) {
  ivec3 q = clamp(ivec3(p), ivec3(0), uDim - ivec3(1));
  return texelFetch(uTipoR, emAtlas(q), 0).r;
}

/** Interseção raio-caixa (slab). Saída < entrada quando não há interseção. */
vec2 raioCaixa(vec3 orig, vec3 dir, vec3 lo, vec3 hi) {
  vec3 inv = 1.0 / dir;
  vec3 t0 = (lo - orig) * inv;
  vec3 t1 = (hi - orig) * inv;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  return vec2(max(max(tmin.x, tmin.y), tmin.z),
              min(min(tmax.x, tmax.y), tmax.z));
}

/** Ruído por pixel para deslocar o início do passo do ray-march. Sem ele o
 *  volume vira uma pilha de cascas concêntricas. */
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}`;

/** Cabeçalho de todo shader deste caminho. */
export const PRECISAO = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler3D;
precision highp usampler2D;`;

/**
 * Vertex shader de cobertura: um triângulo que cobre o alvo inteiro.
 *
 * Triângulo e não quad — dois vértices ficam fora do viewport e não há costura
 * diagonal onde as metades se encontram, que aparece em qualquer efeito que
 * dependa de derivadas. Sem atributo: a posição sai do gl_VertexID.
 */
export const VS_COBERTURA = `${PRECISAO}
out vec2 vNdc;
void main() {
  vec2 v = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;
  vNdc = v;
  gl_Position = vec4(v, 0.0, 1.0);
}`;

/* ────────────────────────────────────────────────────────── utilitários GL */

/** Compila, liga e devolve { p, u } com as localizações resolvidas sob demanda. */
export function programa(gl, vsFonte, fsFonte, rotulo) {
  const compilar = (tipo, fonte) => {
    const s = gl.createShader(tipo);
    gl.shaderSource(s, fonte);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(s);
      const n = /(\d+):(\d+)/.exec(log ?? '');
      const trecho = n
        ? fonte.split('\n').slice(Math.max(0, +n[2] - 4), +n[2] + 2)
          .map((l, k) => `${+n[2] - 3 + k} | ${l}`).join('\n')
        : '';
      throw new Error(`shader "${rotulo}" não compilou:\n${log}\n${trecho}`);
    }
    return s;
  };
  const vs = compilar(gl.VERTEX_SHADER, vsFonte);
  const fs = compilar(gl.FRAGMENT_SHADER, fsFonte);
  const p = gl.createProgram();
  gl.attachShader(p, vs); gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(`programa "${rotulo}" não ligou: ${gl.getProgramInfoLog(p)}`);
  }
  gl.deleteShader(vs); gl.deleteShader(fs);
  const u = new Proxy({}, {
    get: (cache, nome) => {
      if (!(nome in cache)) cache[nome] = gl.getUniformLocation(p, nome);
      return cache[nome];
    },
  });
  return { p, u };
}

/** Textura 2D com os parâmetros que este projeto sempre quer. */
export function textura2D(gl, w, h, formatoInterno, { filtro = 'nearest' } = {}) {
  const t = gl.createTexture();
  const f = filtro === 'linear' ? gl.LINEAR : gl.NEAREST;
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texStorage2D(gl.TEXTURE_2D, 1, formatoInterno, w, h);
  return t;
}
