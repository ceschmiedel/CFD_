/* ── src/render/comum.js ─────────────────────────────────────────────────────
 *
 * Trechos de WGSL compartilhados por todos os passes de desenho: o uniforme da
 * cena, a transformação para o mundo, a amostragem do campo e a paleta.
 *
 * Existem num arquivo próprio porque o volume de fumaça e o renderizador de
 * superfície precisam dos mesmos, e duas cópias de uma função de projeção é
 * como se descobre, três semanas depois, que a fumaça e o carro estão em
 * espaços ligeiramente diferentes.
 */

/*
 * Turbo (Mikhailov 2019), aproximação polinomial. Escolhido em vez de jet
 * porque jet tem bandas falsas — o olho vê contornos onde os dados são lisos —
 * e em vez de viridis porque num fundo escuro turbo tem mais faixa dinâmica
 * percebida, que é o que um campo de velocidade precisa.
 */
export const TURBO = `
fn turbo(t0: f32) -> vec3<f32> {
  let t = clamp(t0, 0.0, 1.0);
  let r = 0.13572138 + t*(4.61539260 + t*(-42.66032258 + t*(132.13108234 + t*(-152.94239396 + t*59.28637943))));
  let g = 0.09140261 + t*(2.19418839 + t*(4.84296658 + t*(-14.18503333 + t*(4.27729857 + t*2.82956604))));
  let b = 0.10667330 + t*(12.64194608 + t*(-60.58204836 + t*(110.36276771 + t*(-89.90310912 + t*27.34824973))));
  return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}`;

/** Bytes do uniforme Cena. Mantenha em sincronia com o struct abaixo. */
export const CENA_BYTES = 64 + 64 + 16 * 5;

export const CENA = `
struct Cena {
  viewProj: mat4x4<f32>,
  invViewProj: mat4x4<f32>,
  dim: vec4<u32>,          // nx, ny, nz, _
  escala: vec4<f32>,       // 1/nx, 1/ny, 1/nz, uRef
  opcoes: vec4<f32>,       // modoCor, ganhoCp, quadro, _
  olho: vec4<f32>,         // posição da câmera em mundo
  fumaca: vec4<f32>,       // densidade, anisotropia g, passos, passosLuz
};
@group(0) @binding(0) var<uniform> C: Cena;
@group(0) @binding(1) var<storage, read> macros: array<vec4<f32>>;

// Do espaço do lattice (célula) para o espaço de mundo, com a origem no centro
// do piso — assim a órbita gira em torno do corpo e não do canto do domínio.
//
// UM ÚNICO FATOR PARA OS TRÊS EIXOS. Dividir cada eixo pela SUA dimensão
// mapeia o domínio para um cubo, e o domínio não é cúbico: num túnel
// 320x160x128 um carro de proporção 1 : 0,47 : 0,31 sai desenhado em
// 1 : 0,94 : 0,78. Foi assim que o modelo aparecia atarracado.
fn paraMundo(p: vec3<f32>) -> vec3<f32> {
  let k = C.escala.x * 2.0;
  return vec3<f32>(
    (p.x - f32(C.dim.x) * 0.5) * k,
    (p.y - f32(C.dim.y) * 0.5) * k,
    p.z * k);
}

/** A inversa: do mundo de volta para coordenadas de célula. */
fn paraLattice(w: vec3<f32>) -> vec3<f32> {
  let k = C.escala.x * 2.0;
  return vec3<f32>(
    w.x / k + f32(C.dim.x) * 0.5,
    w.y / k + f32(C.dim.y) * 0.5,
    w.z / k);
}

fn amostrar(p: vec3<f32>) -> vec4<f32> {
  let n = vec3<i32>(i32(C.dim.x), i32(C.dim.y), i32(C.dim.z));
  let q = clamp(vec3<i32>(p), vec3<i32>(0), n - vec3<i32>(1));
  return macros[u32(q.z) * C.dim.x * C.dim.y + u32(q.y) * C.dim.x + u32(q.x)];
}

/** Interseção raio-caixa (slab). Devolve (t_entrada, t_saida); saída < entrada
 *  quando não há interseção. */
fn raioCaixa(orig: vec3<f32>, dir: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec2<f32> {
  let inv = 1.0 / dir;
  let t0 = (lo - orig) * inv;
  let t1 = (hi - orig) * inv;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  return vec2<f32>(max(max(tmin.x, tmin.y), tmin.z),
                   min(min(tmax.x, tmax.y), tmax.z));
}

/** Ruído por pixel para deslocar o início do passo. Sem ele o ray-march produz
 *  anéis concêntricos visíveis — a assinatura de volume mal amostrado. */
fn hash12(p: vec2<f32>) -> f32 {
  var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
  p3 = p3 + dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}`;
