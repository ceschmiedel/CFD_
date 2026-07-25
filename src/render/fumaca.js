/* ── src/render/fumaca.js ────────────────────────────────────────────────────
 *
 * Fumaça volumétrica: um campo de densidade advectado pelo escoamento e
 * renderizado por ray-marching com espalhamento.
 *
 * Isto é uma camada de VISUALIZAÇÃO, não de física. A fumaça é um traçador
 * passivo: ela não tem massa, não empurra o ar, não entra na conta do arrasto.
 * O escoamento decide para onde ela vai e ela não decide nada — que é
 * exatamente o papel da fumaça num túnel de verdade.
 *
 *
 * POR QUE UM RAKE, E NÃO NÉVOA
 * ----------------------------
 * A tentação é encher o domínio de fumaça. Fica bonito por dois segundos e não
 * mostra nada: um volume homogêneo não tem contraste, e sem contraste não há
 * como o olho ver para onde o ar foi.
 *
 * Todo túnel de vento físico usa um RAKE — um pente de tubos finos a montante
 * do modelo, soltando filamentos paralelos. O que se enxerga é o que acontece
 * COM os filamentos: eles se apertam onde o ar acelera, se abrem onde ele
 * desacelera, se enrolam onde há vórtice e se despedaçam onde a camada limite
 * descola. A separação no teto de um carro é invisível numa névoa e óbvia num
 * filamento que se desfaz.
 *
 *
 * ADVECÇÃO SEMI-LAGRANGIANA
 * -------------------------
 * Para cada voxel, recuamos no tempo pela velocidade local e lemos a densidade
 * de onde a parcela veio. Incondicionalmente estável para qualquer passo, que
 * é o que permite avançar a fumaça MUITO mais rápido que o lattice — a
 * velocidade de lattice é ~0,05 célula por passo, e à taxa real a fumaça
 * levaria minutos para atravessar a tela.
 *
 * O custo é difusão numérica: a interpolação trilinear borra um pouco a cada
 * passo, e filamentos finos engordam. É o defeito conhecido do método e é
 * aceitável aqui — a alternativa (MacCormack, BFECC) dobra o custo para
 * recuperar nitidez que a resolução do volume já não sustenta.
 *
 *
 * ESPALHAMENTO
 * ------------
 * Marchamos o raio primário acumulando transmitância de Beer-Lambert, e em
 * cada amostra marchamos um segundo raio curto na direção da luz para saber o
 * quanto dela chegou ali. É essa segunda marcha que dá VOLUME: sem ela a
 * fumaça é uma mancha chapada de cinza uniforme, com ela ela ganha o lado
 * iluminado e o lado na própria sombra que o olho lê como "isto tem forma".
 *
 * A fase é Henyey-Greenstein com g positivo: fumaça espalha para frente, então
 * ela acende quando está entre o observador e a luz. É um parâmetro de uma
 * linha que separa "nuvem de pixels cinza" de "fumaça".
 */

import { CENA } from './comum.js';

/* Um voxel de rgba16float. Escolhido em vez de r32float porque rgba16float é
 * filtrável em todo lugar, sem depender da feature `float32-filterable` — e a
 * amostragem trilinear é o coração da advecção semi-lagrangiana. */
const BYTES_POR_VOXEL = 8;

const FUMO_STRUCT = `
struct Fumo {
  rake:    vec4<f32>,   // x, passoY, passoZ, raio do filamento
  extensao: vec4<f32>,  // y0, y1, z0, z1 do pente, em células
  params:  vec4<f32>,   // dt, dissipação, taxa, _
  luz:     vec4<f32>,   // direção (xyz normalizada), intensidade
};
@group(0) @binding(2) var<uniform> F: Fumo;`;

const ADVECCAO = `
${CENA}
${FUMO_STRUCT}
@group(0) @binding(3) var densAnterior: texture_3d<f32>;
@group(0) @binding(4) var amostrador: sampler;
@group(0) @binding(5) var densNova: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var<storage, read> tipo: array<u32>;
@group(0) @binding(7) var intermediaria: texture_3d<f32>;

fn ehSolido(c: vec3<i32>) -> bool {
  let n = vec3<i32>(i32(C.dim.x), i32(C.dim.y), i32(C.dim.z));
  let q = clamp(c, vec3<i32>(0), n - vec3<i32>(1));
  let t = tipo[u32(q.z) * C.dim.x * C.dim.y + u32(q.y) * C.dim.x + u32(q.x)];
  return t == 1u || t == 4u || t == 7u;
}

/* O pente: filamentos alinhados numa grade (y, z) num plano a montante. */
fn rake(p: vec3<f32>) -> f32 {
  let dx = abs(p.x - F.rake.x);
  if (dx > 1.5) { return 0.0; }
  if (p.y < F.extensao.x || p.y > F.extensao.y) { return 0.0; }
  if (p.z < F.extensao.z || p.z > F.extensao.w) { return 0.0; }

  /* distância ao filamento mais próximo em cada eixo */
  let fy = abs(fract(p.y / F.rake.y) - 0.5) * F.rake.y;
  let fz = abs(fract(p.z / F.rake.z) - 0.5) * F.rake.z;
  let r = length(vec2<f32>(fy, fz)) / max(F.rake.w, 1e-3);
  return F.params.z * exp(-r * r * 5.5) * (1.0 - dx / 1.5);
}

fn dim3() -> vec3<f32> {
  return vec3<f32>(f32(C.dim.x), f32(C.dim.y), f32(C.dim.z));
}

fn dentroDoDominio(p: vec3<f32>) -> bool {
  return all(p >= vec3<f32>(0.5)) && all(p <= dim3() - 0.5);
}

fn amostraDens(t: texture_3d<f32>, p: vec3<f32>) -> f32 {
  return textureSampleLevel(t, amostrador, p / dim3(), 0.0).r;
}

/* ─── passe 1: advecção semi-lagrangiana pura, para a textura intermediária */

@compute @workgroup_size(4, 4, 4)
fn recuar(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= C.dim.x || gid.y >= C.dim.y || gid.z >= C.dim.z) { return; }
  let ci = vec3<i32>(gid);
  let p = vec3<f32>(gid) + 0.5;
  if (ehSolido(ci)) { textureStore(densNova, ci, vec4<f32>(0.0)); return; }

  let anterior = p - amostrar(p).xyz * F.params.x;
  var d = amostraDens(densAnterior, anterior);
  /* Fora do domínio a fumaça simplesmente acaba — sem isto, o clamp do
   * amostrador copia a borda para dentro e o domínio se enche por trás. */
  if (!dentroDoDominio(anterior)) { d = 0.0; }
  textureStore(densNova, ci, vec4<f32>(d, 0.0, 0.0, 1.0));
}

/*
 * ─── passe 2: correção de MacCormack, com limitador
 *
 * POR QUE ISTO PRECISA EXISTIR
 * ----------------------------
 * A advecção semi-lagrangiana é uma interpolação trilinear por passo, e cada
 * interpolação borra. Um filamento de 0,85 célula de raio sobrevive a poucas:
 * medido aqui, os filamentos saíam do pente e se dissolviam ANTES de alcançar
 * o carro — a única coisa visível era uma névoa cinza a montante do corpo, ou
 * seja, precisamente nada do que se quer ver.
 *
 * MacCormack estima o erro e o desconta. Advectamos para trás (passe 1), depois
 * advectamos o RESULTADO para frente: se o método fosse exato, voltaríamos ao
 * campo original. O que sobra da diferença é o erro de difusão, e metade dele
 * subtraída do resultado recupera a nitidez.
 *
 * O LIMITADOR NÃO É OPCIONAL. A correção é uma extrapolação e pode passar do
 * ponto, criando densidade negativa e picos que não existiam — o que num campo
 * realimentado explode em poucos passos. Grampeamos o resultado ao intervalo
 * dos oito vizinhos de onde a parcela veio: dentro dele, a correção é ganho
 * puro; fora, ela é invenção.
 */
@compute @workgroup_size(4, 4, 4)
fn corrigir(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= C.dim.x || gid.y >= C.dim.y || gid.z >= C.dim.z) { return; }
  let ci = vec3<i32>(gid);
  let p = vec3<f32>(gid) + 0.5;
  if (ehSolido(ci)) { textureStore(densNova, ci, vec4<f32>(0.0)); return; }

  let u = amostrar(p).xyz;
  let anterior = p - u * F.params.x;

  let chapeu = textureLoad(intermediaria, ci, 0).r;      // resultado do passe 1
  let voltando = amostraDens(intermediaria, p + u * F.params.x);
  let original = textureLoad(densAnterior, ci, 0).r;

  var d = chapeu + 0.5 * (original - voltando);

  /* Limitador: o intervalo dos oito cantos da célula de onde a parcela veio. */
  let b = floor(anterior - 0.5);
  var lo = 1e30;
  var hi = -1e30;
  for (var k = 0; k < 8; k = k + 1) {
    let o = vec3<f32>(f32(k & 1), f32((k >> 1) & 1), f32((k >> 2) & 1));
    let q = clamp(vec3<i32>(b + o), vec3<i32>(0), vec3<i32>(C.dim.xyz) - vec3<i32>(1));
    let v = textureLoad(densAnterior, q, 0).r;
    lo = min(lo, v);
    hi = max(hi, v);
  }
  d = clamp(d, lo, hi) * F.params.y;

  if (!dentroDoDominio(anterior)) { d = 0.0; }
  d = max(d, rake(p));
  textureStore(densNova, ci, vec4<f32>(clamp(d, 0.0, 4.0), 0.0, 0.0, 1.0));
}`;

const RENDER = `
${CENA}
${FUMO_STRUCT}
@group(0) @binding(3) var dens: texture_3d<f32>;
@group(0) @binding(4) var amostrador: sampler;
@group(0) @binding(5) var profundidade: texture_depth_2d;

struct Saida { @builtin(position) pos: vec4<f32>, @location(0) ndc: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> Saida {
  /* Triângulo único que cobre a tela: dois vértices fora do viewport. Um
   * triângulo em vez de um quad evita a costura diagonal onde as duas metades
   * se encontram, que aparece em qualquer efeito que dependa de derivadas. */
  var v = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o: Saida;
  o.ndc = v[vi];
  o.pos = vec4<f32>(v[vi], 0.0, 1.0);
  return o;
}

fn mundoDe(ndc: vec2<f32>, z: f32) -> vec3<f32> {
  let h = C.invViewProj * vec4<f32>(ndc.x, -ndc.y, z, 1.0);
  return h.xyz / h.w;
}

/* Henyey-Greenstein. g > 0 espalha para frente, que é o que fumaça faz — e é
 * por isso que ela ACENDE quando está entre você e a luz. */
fn hg(cosTheta: f32, g: f32) -> f32 {
  let g2 = g * g;
  let d = 1.0 + g2 - 2.0 * g * cosTheta;
  return (1.0 - g2) / (12.566370614 * max(d * sqrt(d), 1e-4));
}

fn densidadeEm(w: vec3<f32>) -> f32 {
  let p = paraLattice(w);
  let uvw = p / vec3<f32>(f32(C.dim.x), f32(C.dim.y), f32(C.dim.z));
  if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { return 0.0; }
  return textureSampleLevel(dens, amostrador, uvw, 0.0).r;
}

@fragment
fn fs(e: Saida) -> @location(0) vec4<f32> {
  let origem = C.olho.xyz;
  let alvo = mundoDe(e.ndc, 1.0);
  let dir = normalize(alvo - origem);

  /* A caixa do domínio em coordenadas de mundo. */
  let k = C.escala.x * 2.0;
  let lo = vec3<f32>(-f32(C.dim.x) * 0.5 * k, -f32(C.dim.y) * 0.5 * k, 0.0);
  let hi = vec3<f32>( f32(C.dim.x) * 0.5 * k,  f32(C.dim.y) * 0.5 * k,
                      f32(C.dim.z) * k);
  let t = raioCaixa(origem, dir, lo, hi);
  var t0 = max(t.x, 0.0);
  var t1 = t.y;
  if (t1 <= t0) { discard; }

  /* Para na geometria opaca já desenhada: sem isto a fumaça atravessa o carro
   * e o piso, e a cena perde toda a noção de profundidade. */
  let zbuf = textureLoad(profundidade, vec2<i32>(e.pos.xy), 0);
  if (zbuf < 1.0) {
    let wOpaco = mundoDe(e.ndc, zbuf);
    t1 = min(t1, length(wOpaco - origem));
  }
  if (t1 <= t0) { discard; }

  let nPassos = i32(C.fumaca.z);
  let nLuz = i32(C.fumaca.w);
  let ds = (t1 - t0) / f32(nPassos);
  let sigma = C.fumaca.x;
  let g = C.fumaca.y;

  let luzDir = normalize(F.luz.xyz);
  let fase = hg(dot(dir, luzDir), g);

  /* Desloca o início do passo por pixel. Sem isso o volume vira uma pilha de
   * cascas concêntricas — o artefato mais reconhecível de ray-march. */
  let jitter = hash12(e.pos.xy + C.opcoes.z * 0.618) * ds;

  var T = 1.0;                       // transmitância acumulada
  var L = vec3<f32>(0.0);            // radiância acumulada
  var s = t0 + jitter;

  for (var i = 0; i < nPassos; i = i + 1) {
    if (T < 0.01) { break; }
    let w = origem + dir * s;
    let d = densidadeEm(w);
    s = s + ds;
    if (d <= 0.001) { continue; }

    let st = d * sigma;
    let atenua = 1.0 - exp(-st * ds);

    /* Marcha secundária: quanta luz sobreviveu até aqui. Passos longos e
     * poucos — a sombra da fumaça é suave por natureza e não pede precisão. */
    var tau = 0.0;
    let dl = (hi.z - lo.z) / f32(nLuz) * 0.9;
    for (var j = 1; j <= nLuz; j = j + 1) {
      tau = tau + densidadeEm(w + luzDir * (f32(j) * dl)) * sigma * dl;
    }
    let luz = exp(-tau);

    /* Um piso de espalhamento múltiplo. Fumaça de verdade nunca é preta no
     * lado escuro: a luz que entra ricocheteia e sai por todo lado, e sem
     * esse termo o volume ganha um núcleo morto que denuncia o modelo. */
    let ambiente = 0.16 + 0.10 * exp(-tau * 0.35);

    L = L + T * atenua * (luz * fase * 3.6 + ambiente) * F.luz.w * vec3<f32>(0.96, 0.97, 1.0);
    T = T * exp(-st * ds);
  }

  return vec4<f32>(L, 1.0 - T);
}`;

export class VolumeFumaca {
  /**
   * @param {GPUDevice} device
   * @param {object} solver   fornece nx/ny/nz, macros e tipo
   * @param {object} [opt]
   * @param {number} [opt.escala]  fração da resolução do lattice (1 = igual)
   */
  constructor(device, solver, { escala = 1 } = {}) {
    this.device = device;
    this.solver = solver;
    this.escala = escala;

    /* O volume acompanha o lattice. Ele PODE ser menor — a fumaça não precisa
     * de tanta resolução quanto o escoamento — mas então a advecção precisa
     * converter coordenadas entre os dois espaços, e essa conversão é o tipo
     * de coisa que se erra por meia célula e ninguém descobre. */
    this.nx = solver.nx; this.ny = solver.ny; this.nz = solver.nz;

    this.params = {
      /* Velocidade da fumaça em "passos de lattice por quadro". Ela é MUITO
       * maior que o avanço real do solver de propósito: a 0,05 célula por
       * passo, a fumaça levaria minutos para atravessar a tela. O que se
       * preserva é a forma do campo, não a escala do tempo — e o painel mostra
       * o tempo físico separado, para os dois não se confundirem. */
      dt: 17,
      /* Deslocamento acumulado, em células, que dispara um passo de advecção.
       * Ver o comentário em avancar(): é o parâmetro que decide se os
       * filamentos sobrevivem ou viram névoa. */
      saltoCelulas: 2.2,
      dissipacao: 0.997,
      taxa: 1.3,
      densidade: 48,
      g: 0.45,
      passos: 96,
      passosLuz: 5,
      luz: [-0.35, 0.55, 0.76],
      intensidade: 1.0,
    };
  }

  get bytes() { return this.nx * this.ny * this.nz * BYTES_POR_VOXEL * 3; }

  async preparar(uCena, formatoAlvo) {
    const d = this.device;

    const tex = (rotulo) => d.createTexture({
      size: [this.nx, this.ny, this.nz],
      dimension: '3d',
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
      label: rotulo,
    });
    this.texA?.destroy(); this.texB?.destroy(); this.texC?.destroy();
    this.texA = tex('fumacaA');
    this.texB = tex('fumacaB');
    this.texC = tex('fumacaTmp');   // resultado do passe 1 do MacCormack
    this.frente = 'A';

    this.amostrador = d.createSampler({
      magFilter: 'linear', minFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });

    this.uFumo?.destroy();
    this.uFumo = d.createBuffer({
      size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.uCena = uCena;

    /* ─── advecção ─── */
    this.layoutAdv = d.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { viewDimension: '3d' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: {} },
      { binding: 5, visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { viewDimension: '3d' } },
    ] });
    const layoutPipe = d.createPipelineLayout({ bindGroupLayouts: [this.layoutAdv] });
    const modAdv = d.createShaderModule({ code: ADVECCAO, label: 'fumaca-adv' });
    this.pipeRecuar = d.createComputePipeline({
      layout: layoutPipe, compute: { module: modAdv, entryPoint: 'recuar' },
    });
    this.pipeCorrigir = d.createComputePipeline({
      layout: layoutPipe, compute: { module: modAdv, entryPoint: 'corrigir' },
    });

    const grupoAdv = (src, dst, inter) => d.createBindGroup({
      layout: this.layoutAdv, entries: [
        { binding: 0, resource: { buffer: uCena } },
        { binding: 1, resource: { buffer: this.solver.macros } },
        { binding: 2, resource: { buffer: this.uFumo } },
        { binding: 3, resource: src.createView({ dimension: '3d' }) },
        { binding: 4, resource: this.amostrador },
        { binding: 5, resource: dst.createView({ dimension: '3d' }) },
        { binding: 6, resource: { buffer: this.solver.tipo } },
        { binding: 7, resource: inter.createView({ dimension: '3d' }) },
      ],
    });

    /*
     * Passe 1 escreve na intermediária; passe 2 lê a intermediária e escreve
     * no destino final.
     *
     * O binding 7 do passe 1 aponta para uma textura QUE ELE NÃO USA, e isso é
     * deliberado: uma textura não pode estar ligada ao mesmo tempo como
     * storage de escrita e como textura amostrada. Apontar o binding 7 do
     * passe 1 para a intermediária — que é justamente o alvo de escrita dele —
     * invalida o pipeline inteiro, e como o erro do WebGPU é assíncrono o
     * despacho simplesmente não acontece: a fumaça desaparece sem nada no
     * console. Foi o que aconteceu aqui.
     */
    this.recuarDeA = grupoAdv(this.texA, this.texC, this.texB);
    this.recuarDeB = grupoAdv(this.texB, this.texC, this.texA);
    this.grupoAB = grupoAdv(this.texA, this.texB, this.texC);
    this.grupoBA = grupoAdv(this.texB, this.texA, this.texC);

    /* ─── render ─── */
    this.layoutRender = d.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: '3d' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'depth' } },
    ] });
    const mod = d.createShaderModule({ code: RENDER, label: 'fumaca-render' });
    this.pipeRender = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.layoutRender] }),
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [{
        format: formatoAlvo,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
        },
      }] },
      primitive: { topology: 'triangle-list' },
    });
    this.grupoRender = null;   // depende da textura de profundidade
    this.limpar();
  }

  /** Religa o grupo de render quando a textura de profundidade é recriada. */
  ligarProfundidade(vistaProfundidade) {
    this.grupoRender = this.device.createBindGroup({
      layout: this.layoutRender, entries: [
        { binding: 0, resource: { buffer: this.uCena } },
        { binding: 1, resource: { buffer: this.solver.macros } },
        { binding: 2, resource: { buffer: this.uFumo } },
        { binding: 3, resource: this.atual.createView({ dimension: '3d' }) },
        { binding: 4, resource: this.amostrador },
        { binding: 5, resource: vistaProfundidade },
      ],
    });
    this._vistaProf = vistaProfundidade;
  }

  get atual() { return this.frente === 'A' ? this.texA : this.texB; }

  /**
   * Posiciona o pente em relação ao corpo.
   *
   * Perto o suficiente para os filamentos chegarem organizados — um rake muito
   * a montante deixa a difusão numérica comê-los antes de encontrarem o carro
   * — e largo o suficiente para cobrir o corpo com folga, senão só se vê o
   * escoamento que passa pelo meio.
   */
  posicionarRake(extentosCorpo) {
    const e = extentosCorpo;
    const largura = Math.max(e.tamanho[1], 4);
    const altura = Math.max(e.tamanho[2], 4);
    this.rake = {
      x: Math.max(3, e.min[0] - Math.max(10, e.tamanho[0] * 0.55)),
      y0: Math.max(1, e.centro[1] - largura * 1.15),
      y1: Math.min(this.ny - 2, e.centro[1] + largura * 1.15),
      z0: 1,
      z1: Math.min(this.nz - 2, altura * 2.3),
      /* O espaçamento tem de ser VÁRIAS vezes o raio do filamento, senão as
       * caudas das gaussianas se tocam já na saída do pente e o que se vê é um
       * cobertor cinza — que é a primeira coisa que apareceu aqui. */
      passoY: Math.max(5, largura / 5.5),
      passoZ: Math.max(4.5, altura / 4.5),
      raio: 0.85,
    };
  }

  limpar() {
    /* Um textureStore num kernel dedicado seria mais rápido; escrever zeros da
     * CPU acontece uma vez por montagem e não vale um pipeline a mais. */
    const zeros = new Uint16Array(this.nx * this.ny * this.nz * 4);
    for (const t of [this.texA, this.texB, this.texC]) {
      this.device.queue.writeTexture(
        { texture: t },
        zeros,
        { bytesPerRow: this.nx * 8, rowsPerImage: this.ny },
        [this.nx, this.ny, this.nz]);
    }
  }

  _escreverUniforme(dtEfetivo) {
    const r = this.rake ?? { x: 10, y0: 1, y1: this.ny - 2, z0: 1, z1: this.nz - 2,
      passoY: 8, passoZ: 8, raio: 1.1 };
    const p = this.params;
    const buf = new ArrayBuffer(64);
    const f = new Float32Array(buf);
    f.set([r.x, r.passoY, r.passoZ, r.raio], 0);
    f.set([r.y0, r.y1, r.z0, r.z1], 4);
    f.set([dtEfetivo ?? p.dt, p.dissipacao, p.taxa, 0], 8);
    const l = p.luz, n = Math.hypot(...l) || 1;
    f.set([l[0] / n, l[1] / n, l[2] / n, p.intensidade], 12);
    this.device.queue.writeBuffer(this.uFumo, 0, buf);
  }

  /**
   * Avança a fumaça — mas não necessariamente a cada quadro.
   *
   * O QUE ESTA CADÊNCIA COMPRA
   * --------------------------
   * A difusão numérica da advecção semi-lagrangiana vem da interpolação
   * trilinear, e ela cobra POR PASSO, não por unidade de tempo simulado.
   * Avançar a fumaça todo quadro com um passo pequeno significa uma
   * interpolação por quadro: um filamento de uma célula de raio é borrado por
   * volta de meia célula a cada vez, e a 140 quadros por segundo ele vira
   * névoa antes de percorrer o comprimento do carro. Foi exatamente o que
   * aconteceu na primeira versão — os filamentos saíam do pente e se
   * dissolviam antes de encontrar o corpo.
   *
   * Acumular o deslocamento e disparar um passo GRANDE quando ele passa de um
   * par de células dá o mesmo movimento com três a quatro vezes menos
   * interpolações. O filamento sobrevive à travessia inteira.
   *
   * O preço é o movimento andar aos saltos de ~2 células. Invisível: dois
   * voxels num domínio de 320 projetado em mil pixels são menos de um pixel de
   * salto, e a interpolação da própria amostragem do volume o esconde.
   */
  avancar(enc) {
    this._acumulado = (this._acumulado ?? 0) + this.params.dt;

    /* Deslocamento em células que o acumulado representa, na corrente livre. */
    const uRef = 0.05;
    if (this._acumulado * uRef < this.params.saltoCelulas) return;

    const dtEfetivo = this._acumulado;
    this._acumulado = 0;

    this._escreverUniforme(dtEfetivo);
    const wg = [Math.ceil(this.nx / 4), Math.ceil(this.ny / 4), Math.ceil(this.nz / 4)];
    const p = enc.beginComputePass();

    p.setPipeline(this.pipeRecuar);
    p.setBindGroup(0, this.frente === 'A' ? this.recuarDeA : this.recuarDeB);
    p.dispatchWorkgroups(...wg);

    p.setPipeline(this.pipeCorrigir);
    p.setBindGroup(0, this.frente === 'A' ? this.grupoAB : this.grupoBA);
    p.dispatchWorkgroups(...wg);
    p.end();
    this.frente = this.frente === 'A' ? 'B' : 'A';
    /* O grupo de render aponta para a textura que acabou de ser escrita. */
    if (this._vistaProf) this.ligarProfundidade(this._vistaProf);
  }

  desenhar(rp) {
    if (!this.grupoRender) return;
    rp.setPipeline(this.pipeRender);
    rp.setBindGroup(0, this.grupoRender);
    rp.draw(3);
  }

  destruir() {
    this.texA?.destroy(); this.texB?.destroy(); this.texC?.destroy(); this.uFumo?.destroy();
  }
}
