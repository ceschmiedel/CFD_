/* ── src/render/filete.js ────────────────────────────────────────────────────
 *
 * Filetes de fumaça como GEOMETRIA: cordões de partículas por tubo do pente,
 * desenhados como fitas finas.
 *
 *
 * POR QUE NÃO O VOLUME
 * --------------------
 * O volume de fumaça (fumaca.js) vive na resolução do lattice. No preset Média
 * uma célula tem 15 cm, e o filamento mais fino que sobrevive à advecção
 * trilinear tem raio de uma célula e pouco — 40 cm de diâmetro. Num túnel de
 * verdade o filete tem 1 a 3 cm. Trinta e cinco filetes de 40 cm formam um
 * cobertor, e um filete de 10 cm simplesmente não existe no volume: medido
 * aqui, com raio de 0,7 célula a fumaça some por completo, mesmo com a
 * densidade no máximo. O limitador do MacCormack e a interpolação destroem
 * qualquer feição de um voxel. Na resolução do lattice o volume só dá duas
 * saídas: grosso e leitoso, ou nada.
 *
 * Um cordão de partículas não tem essa limitação. Cada tubo do pente solta
 * partículas continuamente, cada partícula vai para onde o campo manda, e a
 * linha que liga as partículas na ordem em que saíram é a STREAKLINE — que em
 * regime permanente coincide com a linha de corrente, e é exatamente o que a
 * fumaça de um túnel desenha. A espessura é medida em pixels, independente da
 * resolução do volume, então a linha é fina e nítida a qualquer distância.
 *
 *
 * O QUE É MEDIDO E O QUE NÃO É
 * ----------------------------
 * A posição de cada ponto é o campo resolvido, integrado no tempo — nada aqui
 * inventa direção. O que a linha NÃO mostra é a difusão da fumaça real, que
 * engorda o filete ao longo do caminho, e a camada limite que o lattice não
 * resolve: a Re de centenas o descolamento vem cedo e o filete contorna o
 * corpo por longe. Isso é física, e o painel diz o Reynolds resolvido.
 *
 *
 * O ANEL
 * ------
 * Cada tubo tem M posições num anel. `cabecas[t]` aponta para a mais nova; a
 * mais velha é a seguinte no anel. Emitir é avançar a cabeça e escrever a
 * origem nela — a mais velha é sobrescrita, que é a fumaça saindo pelo fundo.
 * O segmento entre a mais nova e a mais velha (a costura do anel) não é
 * desenhado.
 *
 * A emissão é por DISTÂNCIA, não por quadro. O ponto mais novo é comparado
 * com a origem do tubo; a cada `espacamento` de células percorridas nasce um
 * ponto, interpolado na corda — a 90 m/s um quadro anda seis células no pente
 * e um ponto só por quadro deixaria a linha grosseira perto do carro. A 5 m/s
 * um ponto nasce a cada três quadros. O cordão fica com a mesma densidade de
 * pontos em qualquer velocidade, e o movimento da cena vem do relógio comum
 * (C.relogio): os pontos correm ao longo da linha, sem picá-la.
 */

import { CENA } from './comum.js';

const FIL_STRUCT = `
struct Fil {
  a: vec4<f32>,   // espaçamento (céls), meia-largura (px), intensidade, M
  b: vec4<f32>,   // mundo por pixel a 1 de distância, nTubos, _, _
};
@group(0) @binding(3) var<uniform> F: Fil;`;

/* Amostragem trilinear do campo. `amostrar` de comum.js pega a célula inteira,
 * o que basta para uma partícula que dura um quadro (rasantes) mas desenha
 * escadinha num cordão que atravessa o domínio inteiro. Centro da célula i em
 * i + 0,5. */
const TRILINEAR = `
fn amostrarTri(p: vec3<f32>) -> vec3<f32> {
  let n = vec3<f32>(f32(C.dim.x), f32(C.dim.y), f32(C.dim.z));
  let q = clamp(p - 0.5, vec3<f32>(0.0), n - 1.001);
  let i0 = floor(q);
  let f = q - i0;
  var acc = vec3<f32>(0.0);
  for (var k = 0u; k < 8u; k = k + 1u) {
    let o = vec3<f32>(f32(k & 1u), f32((k >> 1u) & 1u), f32((k >> 2u) & 1u));
    let w = mix(1.0 - f, f, o);
    acc = acc + amostrar(i0 + o + 0.5).xyz * (w.x * w.y * w.z);
  }
  return acc;
}`;

const COMPUTE = `
${CENA}
${FIL_STRUCT}
${TRILINEAR}
@group(0) @binding(2) var<storage, read_write> parts: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> cabecas: array<u32>;
@group(0) @binding(5) var<storage, read> origens: array<vec4<f32>>;

fn dentro(p: vec3<f32>) -> bool {
  let n = vec3<f32>(f32(C.dim.x), f32(C.dim.y), f32(C.dim.z));
  return p.x >= 1.0 && p.x < n.x - 1.0 && p.y >= 1.0 && p.y < n.y - 1.0
      && p.z >= 0.5 && p.z < n.z - 1.0;
}

/* Um thread por ponto: anda com o campo. */
@compute @workgroup_size(64)
fn advectar(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let M = u32(F.a.w);
  if (i >= u32(F.b.y) * M) { return; }
  let p = parts[i];
  if (p.w < 0.5) { return; }

  /* Euler subdividido, como as rasantes: o renderizador escolhe os subpassos
   * para cada pedaço ficar perto de uma célula e meia. */
  var np = p.xyz;
  let nSub = max(u32(C.relogio.y), 1u);
  let dt = C.relogio.x / f32(nSub);
  var morta = false;
  for (var k = 0u; k < nSub; k = k + 1u) {
    let u = amostrarTri(np);
    /* Campo nulo é o interior do corpo (ou o piso): a fumaça para na
     * superfície, e a linha termina ali. */
    if (length(u) < 1e-5) { morta = true; break; }
    np = np + u * dt;
  }
  if (morta || !dentro(np)) { parts[i] = vec4<f32>(np, 0.0); return; }
  parts[i] = vec4<f32>(np, 1.0);
}

/* Um thread por tubo: emite pela distância percorrida desde a origem. */
@compute @workgroup_size(64)
fn emitir(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= u32(F.b.y)) { return; }
  let M = u32(F.a.w);
  let base = t * M;
  let o = origens[t].xyz;
  var h = cabecas[t];
  let nova = parts[base + h];
  let esp = F.a.x;

  /* Sem ponto vivo na cabeça — começo, ou o ponto morreu num campo parado —
   * começa um cordão novo. */
  if (nova.w < 0.5) {
    h = (h + 1u) % M;
    parts[base + h] = vec4<f32>(o, 1.0);
    cabecas[t] = h;
    return;
  }

  let d = distance(nova.xyz, o);
  let n = min(u32(d / esp), 8u);
  /* k = 1 é o ponto mais longe da origem (nasce primeiro, fica mais velho);
   * k = n é o mais perto, o mais novo. Interpolado na corda origem→nova, que
   * é o caminho de um quadro num trecho de corrente livre. */
  for (var k = 1u; k <= n; k = k + 1u) {
    let f = 1.0 - f32(k) * esp / d;
    h = (h + 1u) % M;
    parts[base + h] = vec4<f32>(mix(o, nova.xyz, f), 1.0);
  }
  cabecas[t] = h;
}`;

const RENDER = `
${CENA}
${FIL_STRUCT}
@group(0) @binding(2) var<storage, read> parts: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> cabecas: array<u32>;
@group(0) @binding(5) var<storage, read> origens: array<vec4<f32>>;

struct Saida {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) cor: vec4<f32>,
};

/* Uma instância por segmento (ponto k → ponto k+1 do mesmo tubo), seis
 * vértices por fita. A fita tem largura em PIXELS, pelas mesmas razões das
 * rasantes: um filete é fino de perto e de longe. */
@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> Saida {
  let M = u32(F.a.w);
  let t = ii / M;
  let k = ii % M;
  let base = t * M;
  let h = cabecas[t];
  let a = parts[base + k];
  let b = parts[base + (k + 1u) % M];

  var o: Saida;
  /* A costura do anel (mais nova → mais velha) e qualquer ponta morta: fora
   * do volume de recorte, e o rasterizador descarta. */
  if (k == h || a.w < 0.5 || b.w < 0.5) {
    o.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    o.uv = vec2<f32>(0.0);
    o.cor = vec4<f32>(0.0);
    return o;
  }

  let pa = paraMundo(a.xyz);
  let pb = paraMundo(b.xyz);
  var q = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0,  1.0), vec2<f32>(0.0, 1.0));
  let tq = q[vi];
  let pos = mix(pa, pb, tq.x);

  let eixo = pb - pa;
  let comp = length(eixo);
  let dir = select(vec3<f32>(1.0, 0.0, 0.0), eixo / max(comp, 1e-9), comp > 1e-9);
  let paraOlho = normalize(C.olho.xyz - pos);
  var lat = cross(dir, paraOlho);
  if (length(lat) < 1e-4) { lat = cross(dir, vec3<f32>(0.0, 0.0, 1.0)); }
  if (length(lat) < 1e-4) { lat = vec3<f32>(0.0, 1.0, 0.0); }
  lat = normalize(lat);

  let dist = length(C.olho.xyz - pos);
  let esp = dist * F.b.x * F.a.y;
  o.pos = C.viewProj * vec4<f32>(pos + lat * (esp * tq.y), 1.0);
  o.uv = tq;

  /* Idade: 0 na cabeça, 1 na cauda. Só o fim do cordão desvanece — fumaça de
   * verdade some por difusão e por sair do túnel, não por envelhecer. */
  let idade = f32((h + M - k) % M) / f32(M);
  let fade = 1.0 - smoothstep(0.8, 1.0, idade);
  /* A fumaça ACUMULA onde o ar desacelera: mesma vazão, menos velocidade,
   * mais fumaça por metro. É a leitura que faz a esteira ficar mais densa que
   * a corrente livre, e sai direto do campo. */
  let v = length(amostrar(a.xyz).xyz) / max(C.escala.w, 1e-6);
  let acumulo = clamp(0.55 / max(v, 0.25), 0.7, 2.2);
  /* Branco levemente quente: é a cor de fumaça de glicol sob luz de halogênio,
   * e fica fora da paleta turbo do Cp de propósito. */
  o.cor = vec4<f32>(vec3<f32>(1.0, 0.97, 0.92), F.a.z * fade * acumulo);
  return o;
}

@fragment
fn fs(e: Saida) -> @location(0) vec4<f32> {
  let perfil = 1.0 - smoothstep(0.45, 1.0, abs(e.uv.y));
  let a = e.cor.a * perfil;
  return vec4<f32>(e.cor.rgb * a, a);
}`;

export class Filetes {
  /**
   * @param {GPUDevice} device
   * @param {object} solver  fornece nx/ny/nz e macros
   * @param {object} [opt]
   * @param {number} [opt.pontosPorTubo]  tamanho do anel de cada tubo
   * @param {number} [opt.maxTubos]       teto de tubos (os buffers são fixos)
   */
  constructor(device, solver, { pontosPorTubo = 640, maxTubos = 300 } = {}) {
    this.device = device;
    this.solver = solver;
    this.M = pontosPorTubo;
    this.maxTubos = maxTubos;
    this.nTubos = 0;
    this.params = {
      colunas: 9,
      linhas: 6,
      /* Um ponto por célula: a linha acompanha qualquer curva que o lattice
       * resolve, e mais fino que isso só custa. */
      espacamento: 1.0,
      /* Meia-largura em pixels de dispositivo. Pouco menos de dois pixels de
       * largura total: um fio, não uma fita. */
      larguraPixels: 0.9,
      intensidade: 0.9,
    };
  }

  async preparar(uCena, formatoAlvo) {
    const d = this.device;
    this.uCena = uCena;

    this.bufParts?.destroy();
    this.bufParts = d.createBuffer({
      size: this.maxTubos * this.M * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'filetes-pontos',
    });
    this.bufCabecas?.destroy();
    this.bufCabecas = d.createBuffer({
      size: this.maxTubos * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'filetes-cabecas',
    });
    this.bufOrigens?.destroy();
    this.bufOrigens = d.createBuffer({
      size: this.maxTubos * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'filetes-origens',
    });
    this.uFil?.destroy();
    this.uFil = d.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const entradas = (vis, rw) => [
      { binding: 0, visibility: vis, buffer: { type: 'uniform' } },
      { binding: 1, visibility: vis, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: vis, buffer: { type: rw ? 'storage' : 'read-only-storage' } },
      { binding: 3, visibility: vis, buffer: { type: 'uniform' } },
      { binding: 4, visibility: vis, buffer: { type: rw ? 'storage' : 'read-only-storage' } },
      { binding: 5, visibility: vis, buffer: { type: 'read-only-storage' } },
    ];
    this.layoutCompute = d.createBindGroupLayout({
      entries: entradas(GPUShaderStage.COMPUTE, true),
    });
    this.layoutRender = d.createBindGroupLayout({
      entries: entradas(GPUShaderStage.VERTEX, false),
    });
    const recursos = (layout) => d.createBindGroup({
      layout, entries: [
        { binding: 0, resource: { buffer: this.uCena } },
        { binding: 1, resource: { buffer: this.solver.macros } },
        { binding: 2, resource: { buffer: this.bufParts } },
        { binding: 3, resource: { buffer: this.uFil } },
        { binding: 4, resource: { buffer: this.bufCabecas } },
        { binding: 5, resource: { buffer: this.bufOrigens } },
      ],
    });
    this.grupoCompute = recursos(this.layoutCompute);
    this.grupoRender = recursos(this.layoutRender);

    const modC = d.createShaderModule({ code: COMPUTE, label: 'filete-compute' });
    const plc = d.createPipelineLayout({ bindGroupLayouts: [this.layoutCompute] });
    this.pipeAdvectar = d.createComputePipeline({
      layout: plc, compute: { module: modC, entryPoint: 'advectar' },
    });
    this.pipeEmitir = d.createComputePipeline({
      layout: plc, compute: { module: modC, entryPoint: 'emitir' },
    });

    const modR = d.createShaderModule({ code: RENDER, label: 'filete-render' });
    this.pipeRender = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.layoutRender] }),
      vertex: { module: modR, entryPoint: 'vs' },
      fragment: { module: modR, entryPoint: 'fs', targets: [{
        format: formatoAlvo,
        /* Aditivo: dois filetes cruzando somam, como fumaça vista através de
         * fumaça. */
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one' },
          alpha: { srcFactor: 'one', dstFactor: 'one' },
        },
      }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less',
      },
    });

    if (this.rake) this._recalcularGrade();
  }

  /**
   * O pente NA ESCALA DO CORPO.
   *
   * O pente do volume cobre 2,3 vezes a altura e a largura do carro, e a
   * maioria dos filamentos passa a mais de um metro do teto sem sentir nada.
   * Um pente de túnel cobre o modelo com pouca folga: aqui, 1,25 da altura e
   * 1,1 da largura, com a primeira linha rente ao chão — onde estão o
   * assoalho, as rodas e o difusor.
   */
  posicionarRake(extentos) {
    const e = extentos;
    const largura = Math.max(e.tamanho[1], 4);
    const altura = Math.max(e.tamanho[2], 4);
    this.rake = {
      x: Math.max(3, e.min[0] - Math.max(8, e.tamanho[0] * 0.4)),
      y0: Math.max(1.5, e.centro[1] - largura * 0.55),
      y1: Math.min(this.solver.ny - 2.5, e.centro[1] + largura * 0.55),
      z0: 1.0,
      z1: Math.min(this.solver.nz - 2.5, altura * 1.25),
    };
    this._recalcularGrade();
  }

  definirGrade({ colunas, linhas } = {}) {
    if (colunas > 0) this.params.colunas = Math.round(colunas);
    if (linhas > 0) this.params.linhas = Math.round(linhas);
    this._recalcularGrade();
  }

  _recalcularGrade() {
    const r = this.rake;
    if (!r || !this.bufOrigens) return;
    const nc = Math.max(1, this.params.colunas);
    const nl = Math.max(1, this.params.linhas);
    const n = Math.min(nc * nl, this.maxTubos);
    const o = new Float32Array(n * 4);
    let i = 0;
    for (let l = 0; l < nl && i < n; l++) {
      for (let c = 0; c < nc && i < n; c++, i++) {
        o[i * 4] = r.x;
        o[i * 4 + 1] = r.y0 + (c + 0.5) * (r.y1 - r.y0) / nc;
        o[i * 4 + 2] = r.z0 + (l + 0.5) * (r.z1 - r.z0) / nl;
        o[i * 4 + 3] = 0;
      }
    }
    this.nTubos = n;
    this.device.queue.writeBuffer(this.bufOrigens, 0, o);
    this.limpar();
  }

  /** Apaga os cordões; eles renascem do pente nos quadros seguintes. */
  limpar() {
    if (!this.bufParts) return;
    const n = this.nTubos * this.M;
    if (n > 0) this.device.queue.writeBuffer(this.bufParts, 0, new Float32Array(n * 4));
    this.device.queue.writeBuffer(this.bufCabecas, 0, new Uint32Array(this.nTubos || 1));
  }

  _escreverUniforme({ mundoPorPixel }) {
    const p = this.params;
    const f = new Float32Array(8);
    f.set([p.espacamento, p.larguraPixels, p.intensidade, this.M], 0);
    f.set([mundoPorPixel, this.nTubos, 0, 0], 4);
    this.device.queue.writeBuffer(this.uFil, 0, f);
  }

  /**
   * @param {GPUCommandEncoder} enc
   * @param {object} q
   * @param {number} q.mundoPorPixel  ver rasante.js
   */
  avancar(enc, q) {
    if (!this.nTubos) return;
    this._escreverUniforme(q);
    const p = enc.beginComputePass();
    p.setPipeline(this.pipeAdvectar);
    p.setBindGroup(0, this.grupoCompute);
    p.dispatchWorkgroups(Math.ceil(this.nTubos * this.M / 64));
    p.setPipeline(this.pipeEmitir);
    p.dispatchWorkgroups(Math.ceil(this.nTubos / 64));
    p.end();
  }

  desenhar(rp) {
    if (!this.pipeRender || !this.nTubos) return;
    rp.setPipeline(this.pipeRender);
    rp.setBindGroup(0, this.grupoRender);
    rp.draw(6, this.nTubos * this.M);
  }

  destruir() {
    this.bufParts?.destroy();
    this.bufCabecas?.destroy();
    this.bufOrigens?.destroy();
    this.uFil?.destroy();
  }
}
