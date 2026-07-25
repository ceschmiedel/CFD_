/* ── src/render/renderer.js ──────────────────────────────────────────────────
 *
 * A visualização, desenhada a partir do estado do lattice.
 *
 * A regra que governa este arquivo: TUDO O QUE APARECE NA TELA É O CAMPO
 * RESOLVIDO. Nenhuma partícula segue uma curva analítica, nenhuma linha de
 * corrente é um decalque desenhado em volta do corpo, nenhuma cor vem de uma
 * função do espaço. As partículas leem a velocidade das células em que estão;
 * o Cp na carroceria lê a densidade da célula adjacente. Se o solver estiver
 * errado, a imagem fica errada junto — e é assim que tem de ser, porque uma
 * imagem bonita que não depende do solver é um enfeite que mente.
 *
 * (O app anterior deste repositório desenhava exatamente esse enfeite: as
 * "streamlines 3D" eram um bump gaussiano analítico e a "animação do túnel de
 * vento" eram cem marcadores transladando em x. Nenhum dos dois tocava no
 * campo resolvido.)
 *
 *
 * AS TRÊS CAMADAS
 * ---------------
 * ESTEIRAS. Partículas advectadas pelo campo, desenhadas como segmentos entre
 * a posição anterior e a atual. Segmentos e não pontos porque um ponto de um
 * pixel some numa tela grande e não mostra direção; um traço mostra as duas
 * coisas de graça, e o comprimento dele já é a velocidade.
 *
 * CARROCERIA COM Cp. O corpo é desenhado com os triângulos originais — não
 * com os voxels — e colorido pela pressão do fluido ao lado. Ver o Cp na
 * superfície é o que transforma "tem ar passando" em "é AQUI que o arrasto
 * nasce".
 *
 * PISO. Uma grade, para a esteira rolante ter contra o que ser lida. Sem
 * referência de chão a cena flutua e o olho perde a escala.
 */

import { Orbita, inversa } from './mat4.js';
import { CENA, TURBO, CENA_BYTES } from './comum.js';
import { VolumeFumaca } from './fumaca.js';

/* ───────────────────────────────────────────────────────────────── esteiras */

const SHADER_ADVECCAO = `
${CENA}
struct Part { pos: vec4<f32>, ant: vec4<f32> };
@group(0) @binding(2) var<storage, read_write> parts: array<Part>;
@group(0) @binding(3) var<storage, read> tipo: array<u32>;

fn hash(n: u32) -> f32 {
  var x = n * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return f32((x >> 22u) ^ x) / 4294967296.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&parts)) { return; }

  var p = parts[i].pos;
  let m = amostrar(p.xyz);

  // As partículas andam MUITO mais rápido que o fluido, de propósito. A
  // velocidade de lattice é ~0,05 célula por passo: com dt pequeno, o traço
  // entre a posição anterior e a atual mede fração de célula, o que num
  // domínio de 320 células projetado numa tela de 800 pixels dá MENOS DE UM
  // PIXEL. As esteiras existiam e eram invisíveis.
  //
  // O que se preserva é a FORMA do campo, não a escala do tempo — e a tela
  // mostra o tempo físico separadamente, para ninguém confundir os dois.
  //
  // Euler com passo grande erra a trajetória em curva fechada, então o passo é
  // subdividido: mesmo comprimento de traço, caminho muito mais fiel em volta
  // dos cantos, que é justamente onde o olho procura a separação.
  let dt = 15.0;
  var np = p.xyz;
  for (var k = 0; k < 3; k = k + 1) {
    np = np + amostrar(np).xyz * dt;
  }

  p.w = p.w - 1.0;
  let n = vec3<f32>(f32(C.dim.x), f32(C.dim.y), f32(C.dim.z));
  let fora = np.x >= n.x - 1.0 || np.x < 1.0 || np.y < 1.0 || np.y >= n.y - 1.0
          || np.z < 0.5 || np.z >= n.z - 1.0;
  // Uma partícula que entrou no corpo (o campo lá é zero) fica parada para
  // sempre e vira um ponto morto na tela; renascer é mais barato que evitar.
  let presa = length(m.xyz) < 1e-5;

  if (fora || presa || p.w <= 0.0) {
    let s = i * 3u + u32(C.opcoes.z) * 7919u;
    np = vec3<f32>(2.0 + hash(s) * 3.0,
                   2.0 + hash(s + 1u) * (n.y - 4.0),
                   0.5 + hash(s + 2u) * (n.z - 2.0));
    p.w = 220.0 + hash(s + 5u) * 260.0;
    parts[i].ant = vec4<f32>(np, 0.0);   // sem rastro no quadro do renascimento
  } else {
    parts[i].ant = vec4<f32>(p.xyz, 1.0);
  }
  parts[i].pos = vec4<f32>(np, p.w);
}`;

const SHADER_ESTEIRAS = `
${CENA}
${TURBO}
struct Part { pos: vec4<f32>, ant: vec4<f32> };
@group(0) @binding(2) var<storage, read> parts: array<Part>;

struct Saida { @builtin(position) pos: vec4<f32>, @location(0) cor: vec4<f32> };

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> Saida {
  let p = parts[ii];
  let a = select(p.pos.xyz, p.ant.xyz, vi == 0u);
  var o: Saida;
  o.pos = C.viewProj * vec4<f32>(paraMundo(a), 1.0);
  let v = length(amostrar(p.pos.xyz).xyz) / max(C.escala.w, 1e-6);
  // desvanece nas pontas da vida para o renascimento não piscar
  let vida = clamp(p.pos.w / 90.0, 0.0, 1.0);
  o.cor = vec4<f32>(turbo(v * 0.55), p.ant.w * vida * 0.85);
  return o;
}

@fragment
fn fs(e: Saida) -> @location(0) vec4<f32> { return e.cor; }`;

/* ─────────────────────────────────────────────────────── carroceria com Cp */

const SHADER_CORPO = `
${CENA}
${TURBO}
@group(0) @binding(2) var<storage, read> tipo: array<u32>;

struct Saida {
  @builtin(position) pos: vec4<f32>,
  @location(0) mundo: vec3<f32>,
  @location(1) lattice: vec3<f32>,
};

@vertex
fn vs(@location(0) p: vec3<f32>) -> Saida {
  var o: Saida;
  o.lattice = p;
  o.mundo = paraMundo(p);
  o.pos = C.viewProj * vec4<f32>(o.mundo, 1.0);
  return o;
}

// A normal sai das derivadas de tela da posição de mundo. Evita subir um
// buffer de normais e, mais importante, dá a normal da FACE — que é a certa
// para uma malha importada, onde os vértices podem ser compartilhados entre
// painéis que não deveriam ser suavizados juntos.
fn ehSolido(p: vec3<f32>) -> bool {
  let n = vec3<i32>(i32(C.dim.x), i32(C.dim.y), i32(C.dim.z));
  let q = clamp(vec3<i32>(p), vec3<i32>(0), n - vec3<i32>(1));
  let t = tipo[u32(q.z) * C.dim.x * C.dim.y + u32(q.y) * C.dim.x + u32(q.x)];
  return t == 1u || t == 4u || t == 7u;
}

@fragment
fn fs(e: Saida) -> @location(0) vec4<f32> {
  // A normal sai das derivadas de tela, então ela aponta para a câmera e não
  // necessariamente para FORA do corpo. Para iluminar tanto faz; para amostrar
  // a pressão não: metade das vezes o ponto cairia dentro do corpo, onde a
  // densidade não significa nada. Tentamos os dois lados e ficamos com o que
  // está no fluido.
  var n = normalize(cross(dpdx(e.mundo), dpdy(e.mundo)));

  var fora = e.lattice + n * 1.8;
  if (ehSolido(fora)) { fora = e.lattice - n * 1.8; }
  let delta = amostrar(fora).w;

  // Cp = (p - p_inf) / (1/2 rho U^2). No lattice p = c_s^2 * delta com
  // c_s^2 = 1/3 e rho = 1, então Cp = delta / (1.5 u_lb^2).
  let cp = delta / max(1.5 * C.escala.w * C.escala.w, 1e-12);

  let luz = normalize(vec3<f32>(0.4, 0.7, 0.9));
  let dif = 0.35 + 0.65 * max(dot(n, luz), 0.0);
  let esp = pow(max(dot(reflect(-luz, n), normalize(vec3<f32>(0.0,0.0,1.0))), 0.0), 24.0);

  var base = vec3<f32>(0.72, 0.74, 0.78);
  // Cp vai de +1 no ponto de estagnacao a valores bem negativos na succao.
  // Mapear [-2, +1] para [0, 1] poe o azul na succao e o vermelho na
  // estagnacao, que e a convencao de todo relatorio de tunel.
  if (C.opcoes.x > 0.5) {
    base = turbo(clamp(1.0 - (cp * C.opcoes.y + 2.0) / 3.0, 0.0, 1.0));
  }
  return vec4<f32>(base * dif + vec3<f32>(esp * 0.25), 1.0);
}`;

/* ────────────────────────────────────────────────────────────────────  piso */

const SHADER_PISO = `
${CENA}
struct Saida { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> Saida {
  var q = array<vec2<f32>, 6>(
    vec2<f32>(0.0,0.0), vec2<f32>(1.0,0.0), vec2<f32>(1.0,1.0),
    vec2<f32>(0.0,0.0), vec2<f32>(1.0,1.0), vec2<f32>(0.0,1.0));
  let t = q[vi];
  var o: Saida;
  o.uv = t;
  // O piso é o chão do domínio inteiro, então ele passa por paraMundo como
  // qualquer outra coisa — senão a grade deixa de casar com o lattice.
  let p = vec3<f32>(t.x * f32(C.dim.x), t.y * f32(C.dim.y), 0.0);
  o.pos = C.viewProj * vec4<f32>(paraMundo(p), 1.0);
  return o;
}

@fragment
fn fs(e: Saida) -> @location(0) vec4<f32> {
  let g = vec2<f32>(f32(C.dim.x), f32(C.dim.y)) / 16.0;
  let l = abs(fract(e.uv * g) - 0.5) / fwidth(e.uv * g);
  let linha = 1.0 - min(min(l.x, l.y), 1.0);
  return vec4<f32>(vec3<f32>(0.16, 0.19, 0.24) + linha * 0.10, 0.55 + linha * 0.25);
}`;

/* ──────────────────────────────────────────────────────────────────── classe */


export class Renderer {
  constructor(device, canvas) {
    this.device = device;
    this.canvas = canvas;
    this.ctx = canvas.getContext('webgpu');
    this.formato = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device, format: this.formato, alphaMode: 'opaque' });
    this.camera = new Orbita({ alvo: [0, 0, 0.35], distancia: 2.6 });
    this.opcoes = { cp: true, ganhoCp: 1.0, esteiras: false, corpo: true, fumaca: true };
    this.nParticulas = 0;
    this.quadro = 0;
    this._ligarInteracao();
  }

  _ligarInteracao() {
    const c = this.canvas;
    let arrastando = false, lx = 0, ly = 0;
    c.addEventListener('pointerdown', e => {
      arrastando = true; lx = e.clientX; ly = e.clientY; c.setPointerCapture(e.pointerId);
    });
    c.addEventListener('pointerup', e => {
      arrastando = false; c.releasePointerCapture(e.pointerId);
    });
    c.addEventListener('pointermove', e => {
      if (!arrastando) return;
      this.camera.girar((e.clientX - lx) * 0.008, (e.clientY - ly) * 0.008);
      lx = e.clientX; ly = e.clientY;
    });
    c.addEventListener('wheel', e => {
      e.preventDefault();
      this.camera.aproximar(Math.exp(e.deltaY * 0.0011));
    }, { passive: false });
  }

  /** Prepara pipelines para um solver — refeito quando a resolução muda. */
  async preparar(solver, { nParticulas = 60000 } = {}) {
    const d = this.device;
    this.solver = solver;
    this.nParticulas = nParticulas;

    this.uCena?.destroy();
    this.uCena = d.createBuffer({
      size: CENA_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.bufParts?.destroy();
    this.bufParts = d.createBuffer({
      size: nParticulas * 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    /* Semeadas fora do domínio com vida zero: o primeiro passo de advecção as
     * renasce em posições válidas, então não há um quadro inicial com todas as
     * partículas empilhadas na origem. */
    const semente = new Float32Array(nParticulas * 8);
    for (let i = 0; i < nParticulas; i++) semente[i * 8 + 3] = -1;
    d.queue.writeBuffer(this.bufParts, 0, semente);

    const vis = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE;
    this.layout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: vis, buffer: { type: 'uniform' } },
        { binding: 1, visibility: vis, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: vis, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: vis, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.layoutAdv = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });

    const plr = d.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const pla = d.createPipelineLayout({ bindGroupLayouts: [this.layoutAdv] });

    const mod = (code) => d.createShaderModule({ code });
    const alvo = [{
      format: this.formato,
      blend: {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      },
    }];
    const prof = {
      depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus',
    };

    this.pAdv = d.createComputePipeline({
      layout: pla, compute: { module: mod(SHADER_ADVECCAO), entryPoint: 'main' },
    });

    const mEst = mod(SHADER_ESTEIRAS);
    this.pEsteiras = d.createRenderPipeline({
      layout: plr,
      vertex: { module: mEst, entryPoint: 'vs' },
      fragment: { module: mEst, entryPoint: 'fs', targets: alvo },
      primitive: { topology: 'line-list' },
      /* Escreve profundidade? Não: um traço de um pixel que ocultasse a
       * carroceria atrás dele deixaria a superfície salpicada. As esteiras
       * TESTAM profundidade (somem atrás do corpo) mas não a escrevem. */
      depthStencil: { ...prof, depthWriteEnabled: false },
    });

    const mCor = mod(SHADER_CORPO);
    this.pCorpo = d.createRenderPipeline({
      layout: plr,
      vertex: {
        module: mCor, entryPoint: 'vs',
        buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }],
      },
      fragment: { module: mCor, entryPoint: 'fs', targets: [{ format: this.formato }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: prof,
    });

    const mPiso = mod(SHADER_PISO);
    this.pPiso = d.createRenderPipeline({
      layout: plr,
      vertex: { module: mPiso, entryPoint: 'vs' },
      fragment: { module: mPiso, entryPoint: 'fs', targets: alvo },
      primitive: { topology: 'triangle-list' },
      depthStencil: { ...prof, depthWriteEnabled: false },
    });

    const entradas = (parts) => [
      { binding: 0, resource: { buffer: this.uCena } },
      { binding: 1, resource: { buffer: solver.macros } },
      { binding: 2, resource: { buffer: parts } },
      { binding: 3, resource: { buffer: solver.tipo } },
    ];
    this.grupo = d.createBindGroup({ layout: this.layout, entries: entradas(this.bufParts) });
    this.grupoAdv = d.createBindGroup({ layout: this.layoutAdv, entries: entradas(this.bufParts) });
    this.fumaca?.destruir();
    this.fumaca = new VolumeFumaca(d, solver);
    await this.fumaca.preparar(this.uCena, this.formato);
    this.depth = null;            // força recriação e religação da profundidade

    this.grupoCorpo = d.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.uCena } },
        { binding: 1, resource: { buffer: solver.macros } },
        { binding: 2, resource: { buffer: solver.tipo } },
        { binding: 3, resource: { buffer: solver.tipo } },
      ],
    });
  }

  /**
   * Aponta a câmera para o corpo, e não para o domínio.
   *
   * Com a escala uniforme corrigida o domínio tem 2 unidades de comprimento e
   * um carro tem 0,2 — enquadrar o domínio deixa o objeto de interesse com um
   * décimo da tela. `extentos` vem em células, do prepare.
   */
  enquadrar(extentos, solver) {
    const k = 2 / solver.nx;
    const c = extentos.centro;
    this.camera.alvo = [
      (c[0] - solver.nx / 2) * k,
      (c[1] - solver.ny / 2) * k,
      c[2] * k,
    ];
    const diag = Math.hypot(...extentos.tamanho) * k;
    this.camera.distancia = Math.max(0.5, diag * 1.9);
    this.fumaca?.posicionarRake(extentos);
  }

  /** Sobe a malha do corpo (posições no espaço do lattice). */
  definirMalha(positions, indices) {
    const d = this.device;
    this.vbo?.destroy(); this.ibo?.destroy();
    if (!positions || !indices?.length) { this.vbo = null; this.nIndices = 0; return; }

    this.vbo = d.createBuffer({
      size: positions.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(this.vbo, 0, positions);

    /* Uint32 sempre: uma malha de F1 passa dos 65 536 vértices com folga, e
     * cair para Uint16 silenciosamente embaralharia os triângulos. */
    const ind = indices instanceof Uint32Array ? indices : new Uint32Array(indices);
    this.ibo = d.createBuffer({
      size: ind.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(this.ibo, 0, ind);
    this.nIndices = ind.length;
  }

  _redimensionar() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width === w && this.canvas.height === h && this.depth) return;
    this.canvas.width = w; this.canvas.height = h;
    this.depth?.destroy();
    this.depth = this.device.createTexture({
      size: [w, h], format: 'depth24plus',
      /* TEXTURE_BINDING além de RENDER_ATTACHMENT: o ray-march da fumaça lê
         esta profundidade para parar na geometria opaca. Sem isso a fumaça
         atravessa o carro. */
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.vistaDepth = this.depth.createView();
    this.fumaca?.ligarProfundidade(this.vistaDepth);
  }

  desenhar({ uRef }) {
    if (!this.solver || !this.uCena) return;
    this._redimensionar();
    const s = this.solver;
    const d = this.device;
    this.quadro++;

    const vp = this.camera.matriz(this.canvas.width / this.canvas.height);
    const f = this.fumaca;
    const buf = new ArrayBuffer(CENA_BYTES);
    new Float32Array(buf, 0, 16).set(vp);
    new Float32Array(buf, 64, 16).set(inversa(vp));
    new Uint32Array(buf, 128, 4).set([s.nx, s.ny, s.nz, 0]);
    new Float32Array(buf, 144, 4).set([1/s.nx, 1/s.ny, 1/s.nz, uRef]);
    new Float32Array(buf, 160, 4).set([
      this.opcoes.cp ? 1 : 0, this.opcoes.ganhoCp, this.quadro % 4096, 0]);
    new Float32Array(buf, 176, 4).set([...this.camera.olho, 1]);
    new Float32Array(buf, 192, 4).set(f
      ? [f.params.densidade, f.params.g, f.params.passos, f.params.passosLuz]
      : [0, 0, 1, 1]);
    d.queue.writeBuffer(this.uCena, 0, buf);

    const enc = d.createCommandEncoder();

    if (this.opcoes.esteiras) {
      const cp = enc.beginComputePass();
      cp.setPipeline(this.pAdv);
      cp.setBindGroup(0, this.grupoAdv);
      cp.dispatchWorkgroups(Math.ceil(this.nParticulas / 64));
      cp.end();
    }
    if (this.opcoes.fumaca && f) f.avancar(enc);

    /*
     * DOIS PASSES. O opaco primeiro, escrevendo profundidade; a fumaça depois,
     * LENDO essa profundidade para saber onde parar.
     *
     * Não dá para fazer num passe só: um attachment de profundidade não pode
     * ser alvo de render e textura amostrada ao mesmo tempo, e a fumaça precisa
     * justamente da profundidade da geometria que acabou de ser desenhada para
     * não atravessá-la.
     */
    const vista = this.ctx.getCurrentTexture().createView();

    const rp = enc.beginRenderPass({
      colorAttachments: [{
        view: vista,
        clearValue: { r: 0.043, g: 0.051, b: 0.067, a: 1 },
        loadOp: 'clear', storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.vistaDepth,
        depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store',
      },
    });

    rp.setPipeline(this.pPiso);
    rp.setBindGroup(0, this.grupo);
    rp.draw(6);

    if (this.opcoes.corpo && this.vbo && this.nIndices) {
      rp.setPipeline(this.pCorpo);
      rp.setBindGroup(0, this.grupoCorpo);
      rp.setVertexBuffer(0, this.vbo);
      rp.setIndexBuffer(this.ibo, 'uint32');
      rp.drawIndexed(this.nIndices);
    }

    if (this.opcoes.esteiras) {
      rp.setPipeline(this.pEsteiras);
      rp.setBindGroup(0, this.grupo);
      rp.draw(2, this.nParticulas);
    }
    rp.end();

    if (this.opcoes.fumaca && f) {
      const rf = enc.beginRenderPass({
        colorAttachments: [{
          view: vista, loadOp: 'load', storeOp: 'store',
        }],
      });
      f.desenhar(rf);
      rf.end();
    }

    d.queue.submit([enc.finish()]);
  }
}
