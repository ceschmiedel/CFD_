/* ── src/render/rasante.js ───────────────────────────────────────────────────
 *
 * Rasantes: traços de ar passando rente ao chão, para a cena ter velocidade.
 *
 *
 * O PROBLEMA QUE ISTO RESOLVE
 * ---------------------------
 * Um túnel de vento bem resolvido, visto de fora, é uma cena PARADA. Não é
 * defeito de renderização: em regime estacionário o campo não muda, os
 * filamentos de fumaça são streaklines fixas no espaço, e um observador que
 * chega sem contexto vê um carro de enfeite dentro de uma caixa. A informação
 * de que ali passam 30 m/s não está em lugar nenhum da imagem.
 *
 * O que falta é o que se vê pela janela de um carro em movimento: o chão
 * borrando. Perto do solo o ar vai à velocidade da corrente livre (com esteira
 * rolante, o próprio piso vai junto), e é onde o olho humano lê velocidade —
 * nós julgamos deslocamento por paralaxe de coisas PERTO, não pelo horizonte.
 * Por isso a faixa é enviesada para o chão e não distribuída pelo domínio.
 *
 *
 * O QUE É MEDIDO E O QUE É EXAGERADO
 * ----------------------------------
 * Vale aqui a mesma regra do resto do renderizador: a partícula lê a velocidade
 * da célula em que está e vai para onde o campo mandar. Direção, curvatura,
 * onde ela acelera, onde ela para na esteira — tudo isso é o campo resolvido.
 *
 * Duas coisas são exageradas, e é melhor dizer quais:
 *
 *   1. A ESCALA DE TEMPO — mas de um jeito específico, e é o que faz a
 *      velocidade do túnel ser SENTIDA. A cena roda a uma fração fixa do tempo
 *      real (ver RELOGIO em renderer.js). Não dá para andar na taxa do solver:
 *      um quadro do lattice avança 0,05 célula e o traço mediria menos de um
 *      pixel. E não dá para andar numa taxa fixa em células por quadro, que foi
 *      a primeira tentativa aqui: u_lb é 0,05 a 5 m/s e a 90 m/s — o que muda
 *      com a velocidade é o valor em SEGUNDOS de um passo, não o campo — então
 *      uma taxa fixa em células dá exatamente a mesma imagem nas duas pontas do
 *      controle. Reproduzir tempo FÍSICO a taxa constante põe o deslocamento na
 *      tela proporcional aos metros por segundo, que é o que se quer sentir.
 *
 *   2. O COMPRIMENTO DO TRAÇO. O traço desenhado é o caminho REAL percorrido no
 *      passo, multiplicado por um fator. Ele estica ao longo da trajetória que
 *      a partícula de fato fez — não inventa direção, e como o caminho já é
 *      proporcional à velocidade local, o traço continua sendo uma leitura de
 *      velocidade: comprido onde o ar corre, curto e embolado na esteira.
 *
 * O que NÃO se faz é o truque óbvio: linhas de velocidade desenhadas em x a uma
 * taxa constante. Ficariam idênticas com ou sem carro, e é exatamente o tipo de
 * enfeite que este projeto existe para não ter.
 *
 *
 * POR QUE QUADS E NÃO LINHAS
 * --------------------------
 * WebGPU desenha linha de um pixel e ponto final — sem controle de espessura e
 * sem suavização, então numa tela HiDPI ela some e cintila conforme cai dentro
 * ou fora do centro do pixel. O quad expandido em volta do eixo do traço custa
 * quatro vértices a mais por partícula (nada) e dá em troca largura escolhida
 * e borda antisserrilhada.
 *
 * A largura é medida EM PIXELS, não em células do lattice. Espessura de mundo
 * engorda com o zoom e transforma o traço numa gota alongada — um cometa com
 * cabeça acesa e rabo desvanecendo, que foi o primeiro resultado aqui e é
 * exatamente o que não se quer. Risco tem a mesma largura de perto e de longe;
 * o que varia de um para o outro é só o comprimento, que é a velocidade.
 */

import { CENA } from './comum.js';

/* O que é só desta camada. O passo de tempo NÃO está aqui: ele vem de
 * C.relogio, junto com o das outras camadas de movimento, porque duas fontes
 * para a mesma taxa é como se acaba com o ar perto do chão andando numa
 * velocidade e a fumaça em outra. */
const RASA_STRUCT = `
struct Rasa {
  a: vec4<f32>,   // alongamento, altura da faixa (céls), meia-largura (px), intensidade
  b: vec4<f32>,   // mundo por pixel a 1 de distância, traço máximo (céls), distância da câmera, _
};
@group(0) @binding(3) var<uniform> R: Rasa;`;

const ADVECCAO = `
${CENA}
${RASA_STRUCT}
struct Part { pos: vec4<f32>, ant: vec4<f32> };
@group(0) @binding(2) var<storage, read_write> parts: array<Part>;

fn hash(n: u32) -> f32 {
  var x = n * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return f32((x >> 22u) ^ x) / 4294967296.0;
}

/* Nasce logo depois da entrada, espalhada na largura toda e ENVIESADA PARA O
 * CHÃO. O expoente cúbico põe a maioria nas primeiras células da faixa e ainda
 * deixa algumas lá em cima: uma distribuição uniforme dentro da faixa daria uma
 * borda superior reta, que o olho lê como uma parede de vidro flutuando. */
fn nascer(i: u32) -> vec4<f32> {
  let s = i * 3u + u32(C.opcoes.z) * 7919u;
  let n = vec3<f32>(f32(C.dim.x), f32(C.dim.y), f32(C.dim.z));
  let h = hash(s + 2u);
  return vec4<f32>(
    1.5 + hash(s) * 4.0,
    1.0 + hash(s + 1u) * (n.y - 2.0),
    0.6 + h * h * h * R.a.y,
    /* Vida em quadros, sorteada larga. Quase toda partícula sai pelo fundo
     * antes de expirar — a vida é rede de segurança para as que ficam presas
     * numa recirculação e para quebrar a sincronia entre renascimentos. */
    150.0 + hash(s + 5u) * 320.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&parts)) { return; }

  var p = parts[i].pos;
  let m = amostrar(p.xyz);

  /* Euler subdividido: mesmo comprimento de caminho, trajetória muito mais
   * fiel em volta dos cantos — que é justamente onde o olho procura a
   * separação, e onde um passo único e grande cortaria a curva pela corda.
   *
   * A CONTAGEM DE SUBPASSOS VEM DE FORA e cresce com a velocidade: o passo de
   * um quadro mede meia célula a 5 m/s e mais de dez a 90, e um número fixo de
   * subdivisões que basta num extremo é grosseiro demais no outro. O
   * renderizador o escolhe para o subpasso ficar sempre perto de uma célula e
   * meia. */
  var np = p.xyz;
  let nSub = max(u32(C.relogio.y), 1u);
  let dt = C.relogio.x / f32(nSub);
  for (var k = 0u; k < nSub; k = k + 1u) {
    np = np + amostrar(np).xyz * dt;
  }

  p.w = p.w - 1.0;
  let n = vec3<f32>(f32(C.dim.x), f32(C.dim.y), f32(C.dim.z));
  let fora = np.x >= n.x - 1.0 || np.x < 1.0 || np.y < 1.0 || np.y >= n.y - 1.0
          || np.z < 0.4 || np.z >= n.z - 1.0;
  /* Campo nulo = a partícula entrou no corpo (ou no piso). Ela ficaria parada
   * para sempre e viraria um ponto morto brilhando na carroceria. */
  let presa = length(m.xyz) < 1e-5;

  if (fora || presa || p.w <= 0.0) {
    let nova = nascer(i);
    np = nova.xyz;
    p.w = nova.w;
    parts[i].ant = vec4<f32>(np, 0.0);   // sem traço no quadro do renascimento
  } else {
    parts[i].ant = vec4<f32>(p.xyz, 1.0);
  }
  parts[i].pos = vec4<f32>(np, p.w);
}`;

const RENDER = `
${CENA}
${RASA_STRUCT}
struct Part { pos: vec4<f32>, ant: vec4<f32> };
@group(0) @binding(2) var<storage, read> parts: array<Part>;

struct Saida {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,      // ao longo (0 cauda → 1 cabeça), através (-1..1)
  @location(1) cor: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> Saida {
  let p = parts[ii];

  /* O traço: o que a partícula percorreu neste passo, esticado ao longo do
   * próprio caminho. Ver o cabeçalho — estica comprimento, nunca direção.
   *
   * Com teto. O caminho por quadro cresce com a velocidade do túnel, e a 90 m/s
   * o traço alongado passaria de meio carro: a essa altura ele deixa de ser
   * borrão de movimento e vira um risco atravessando a cena, que além de feio
   * corta pela corda qualquer curva do escoamento. */
  let caminho = p.ant.xyz - p.pos.xyz;
  let lc = length(caminho);
  let esticado = caminho * min(R.a.x, R.b.y / max(lc, 1e-6));

  let cabeca = paraMundo(p.pos.xyz);
  let cauda  = paraMundo(p.pos.xyz + esticado);

  var q = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0,  1.0), vec2<f32>(0.0, 1.0));
  let t = q[vi];
  let base = mix(cauda, cabeca, t.x);

  let eixo = cabeca - cauda;
  let comp = length(eixo);
  let dir = select(vec3<f32>(1.0, 0.0, 0.0), eixo / max(comp, 1e-9), comp > 1e-9);

  /* Fita sempre de frente para a câmera. Se a linha de visão for paralela ao
   * traço o produto vetorial degenera — aí qualquer perpendicular serve, porque
   * nesse ângulo o traço tem tamanho de ponto na tela. */
  let paraOlho = normalize(C.olho.xyz - base);
  var lat = cross(dir, paraOlho);
  if (length(lat) < 1e-4) { lat = cross(dir, vec3<f32>(0.0, 0.0, 1.0)); }
  if (length(lat) < 1e-4) { lat = vec3<f32>(0.0, 1.0, 0.0); }
  lat = normalize(lat);

  /*
   * LARGURA EM PIXEL, NÃO NO MUNDO.
   *
   * Uma espessura medida em células cresce na tela quando a câmera aproxima, e
   * um traço de vários pixels de largura com borda suave não é um risco: é uma
   * gota alongada, um cometa. Aproximando, vira uma mancha.
   *
   * Um risco tem a mesma largura de perto e de longe — é o que um arranhão no
   * vidro faz, e é o que a fotografia de longa exposição faz com uma luz em
   * movimento. Fixando a meia-largura em pixels a fita fica com um pixel e
   * pouco de espessura em qualquer distância e qualquer zoom, e o que muda de
   * um traço para o outro passa a ser só o COMPRIMENTO — que é a velocidade.
   */
  let dist = length(C.olho.xyz - base);
  let esp = dist * R.b.x * R.a.z;

  var o: Saida;
  o.pos = C.viewProj * vec4<f32>(base + lat * (esp * t.y), 1.0);
  o.uv = t;

  /* Brilho pela velocidade local sobre a corrente livre: a esteira do carro
   * apaga sozinha e o contraste entre o chão rasgando e o ar morto atrás do
   * corpo passa a ser a leitura principal desta camada. */
  let v = length(amostrar(p.pos.xyz).xyz) / max(C.escala.w, 1e-6);
  /* Decaimento com a altura. Sobre a faixa de renascimento não há corte duro:
   * a partícula que sobe segue viva, só vai sumindo. */
  let alto = exp(-p.pos.z / max(R.a.y * 0.4, 1.0));
  let vida = clamp(p.pos.w / 60.0, 0.0, 1.0);
  /* E não pisca ao nascer: as primeiras células depois da entrada desvanecem. */
  let entrada = smoothstep(0.0, 10.0, p.pos.x);

  /* Atenuação com a distância, normalizada pela distância da câmera ao alvo.
   *
   * Sem ela o fundo entope: o piso tem centenas de células de profundidade e a
   * perspectiva empilha traços cada vez mais numerosos em cada vez menos
   * pixels, até o horizonte virar uma barra branca sólida. Normalizar pela
   * distância da órbita é o que mantém o efeito igual em qualquer zoom — o que
   * está perto do carro fica cheio, o que está longe se dissolve. É a mesma
   * pista de profundidade que a atmosfera dá de graça lá fora. */
  let longe = length(C.olho.xyz - cabeca) / max(R.b.z, 1e-3);
  let bruma = 1.0 / (1.0 + pow(max(longe - 1.0, 0.0) * 1.6, 2.0));

  /* POUCOS TRAÇOS, cada um legível sozinho. A versão anterior tinha milhares
   * deles com brilho baixo, contando com a soma aditiva; o resultado foi um
   * tapete fechado que mostrava o chão inteiro em movimento e não deixava ver
   * mais nada — nem a grade, nem a esteira do carro. Um punhado de riscos
   * passando diz a mesma coisa e deixa a cena visível. */
  let a = 0.8 * R.a.w * clamp(v * 0.9, 0.0, 1.6) * alto * vida * entrada
        * bruma * p.ant.w;
  /* Azul frio no ar lento, quase branco no rápido — um risco de um pixel perde
   * cor se for saturado demais. Deliberadamente FORA da paleta turbo: turbo é
   * a escala quantitativa do Cp e das esteiras, e uma segunda camada usando as
   * mesmas cores convidaria a ler número onde não há. */
  let cor = mix(vec3<f32>(0.50, 0.68, 1.0), vec3<f32>(0.92, 0.96, 1.0),
                clamp(v * 0.45, 0.0, 1.0));
  o.cor = vec4<f32>(cor, a);
  return o;
}

@fragment
fn fs(e: Saida) -> @location(0) vec4<f32> {
  /* Perfil transversal: NÚCLEO CHEIO com uma borda de antialias, e não uma
   * gaussiana. A queda suave ao longo de toda a largura é o que dava o aspecto
   * de borrão arredondado — o traço não tinha lado, tinha um meio brilhante
   * indo a zero em todas as direções. Um risco é chapado no miolo e acaba de
   * repente; o que sobra de suavidade serve só para a borda não serrilhar. */
  let perfil = 1.0 - smoothstep(0.35, 1.0, abs(e.uv.y));
  /* Cauda: some no último quarto e só. Desvanecer ao longo de todo o traço
   * transforma um risco num cometa — cabeça acesa e rabo de fumaça —, que é
   * justamente a forma arredondada que se quer evitar. */
  let cauda = smoothstep(0.0, 0.22, e.uv.x);
  let a = e.cor.a * perfil * cauda;
  /* Aditivo, então a cor sai pré-multiplicada. */
  return vec4<f32>(e.cor.rgb * a, a);
}`;

export class Rasantes {
  /**
   * @param {GPUDevice} device
   * @param {object} solver  fornece nx/ny/nz e macros
   * @param {object} [opt]
   * @param {number} [opt.n] quantidade de traços
   */
  constructor(device, solver, { n = 60000 } = {}) {
    this.device = device;
    this.solver = solver;
    this.n = n;

    this.params = {
      /* O traço tem de ser COMPRIDO. Curto, ele é um ponto que pisca, e pontos
       * piscando são estática de televisão; o risco alongado é o que o olho lê
       * como uma coisa passando depressa. */
      alongamento: 5.0,
      /* Teto do traço, em células — ver o comentário no shader. Dezesseis é
       * cerca de um quarto de carro nos presets usuais. */
      tracoMaximo: 16,
      /* Altura da faixa em células — reposicionada pelo tamanho do corpo. */
      altura: Math.max(6, solver.nz * 0.22),
      /* Meia-largura do risco, EM PIXELS de dispositivo — ver o shader. Pouco
       * mais de um pixel de largura total: fino o bastante para ser um risco e
       * não uma fita, largo o bastante para o rasterizador não comer metade
       * dos traços por falta de cobertura. */
      larguraPixels: 1.1,
      intensidade: 1.0,
    };
  }

  async preparar(uCena, formatoAlvo) {
    const d = this.device;
    this.uCena = uCena;

    this.bufParts?.destroy();
    this.bufParts = d.createBuffer({
      size: this.n * 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'rasantes',
    });
    this.semear();

    this.uRasa?.destroy();
    this.uRasa = d.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const entradas = (tipoParts) => [
      { binding: 0, visibility: tipoParts.vis, buffer: { type: 'uniform' } },
      { binding: 1, visibility: tipoParts.vis, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: tipoParts.vis, buffer: { type: tipoParts.parts } },
      { binding: 3, visibility: tipoParts.vis, buffer: { type: 'uniform' } },
    ];
    this.layoutAdv = d.createBindGroupLayout({
      entries: entradas({ vis: GPUShaderStage.COMPUTE, parts: 'storage' }),
    });
    this.layoutRender = d.createBindGroupLayout({
      entries: entradas({ vis: GPUShaderStage.VERTEX, parts: 'read-only-storage' }),
    });

    const recursos = (layout) => d.createBindGroup({
      layout, entries: [
        { binding: 0, resource: { buffer: this.uCena } },
        { binding: 1, resource: { buffer: this.solver.macros } },
        { binding: 2, resource: { buffer: this.bufParts } },
        { binding: 3, resource: { buffer: this.uRasa } },
      ],
    });
    this.grupoAdv = recursos(this.layoutAdv);
    this.grupoRender = recursos(this.layoutRender);

    this.pipeAdv = d.createComputePipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.layoutAdv] }),
      compute: {
        module: d.createShaderModule({ code: ADVECCAO, label: 'rasante-adv' }),
        entryPoint: 'main',
      },
    });

    const mod = d.createShaderModule({ code: RENDER, label: 'rasante-render' });
    this.pipeRender = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.layoutRender] }),
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [{
        format: formatoAlvo,
        /* Aditivo. O piso é escuro e o que se quer é brilho somando onde muitos
         * traços se cruzam — com alpha comum, o traço da frente TAPA o de trás
         * e a faixa perde a densidade que é justamente a sensação de velocidade. */
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one' },
          alpha: { srcFactor: 'one', dstFactor: 'one' },
        },
      }] },
      primitive: { topology: 'triangle-list' },
      /* Testa profundidade (some atrás do carro), não escreve: um traço
       * translúcido escrevendo profundidade recortaria buracos na fumaça. */
      depthStencil: {
        format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less',
      },
    });
  }

  /**
   * Semeia espalhando pelo domínio inteiro em x.
   *
   * O shader renasce as partículas na entrada, o que em regime dá uma faixa
   * uniforme — mas só depois de uma travessia inteira, e o primeiro par de
   * segundos ficaria com uma frente de onda subindo a tela. Semear já espalhado
   * é uma linha e faz o primeiro quadro parecer com o milésimo.
   */
  semear() {
    const { nx, ny } = this.solver;
    const h = this.params.altura;
    const a = new Float32Array(this.n * 8);
    for (let i = 0; i < this.n; i++) {
      const o = i * 8;
      const z = 0.6 + Math.pow(Math.random(), 3) * h;
      a[o] = 1.5 + Math.random() * (nx - 3);
      a[o + 1] = 1 + Math.random() * (ny - 2);
      a[o + 2] = z;
      a[o + 3] = 150 + Math.random() * 320;   // vida
      a[o + 7] = 0;                           // sem traço no primeiro quadro
    }
    this.device.queue.writeBuffer(this.bufParts, 0, a);
  }

  /**
   * Amarra a faixa à altura do corpo.
   *
   * Uma faixa fixa em fração do domínio erra nos dois sentidos: num carro baixo
   * ela cobre o teto e vira névoa sobre a cena; numa fórmula com asa traseira
   * alta ela mal chega ao difusor. Uma vez e meia a altura do corpo cobre o
   * assoalho, as rodas e o começo da lateral — que é a região onde o chão
   * passando tem o que fazer.
   */
  posicionarFaixa(extentos) {
    const alturaCorpo = Math.max(extentos.tamanho[2], 4);
    this.params.altura = Math.min(this.solver.nz * 0.5, alturaCorpo * 1.5);
    this.semear();
  }

  _escreverUniforme({ mundoPorPixel, distanciaCamera }) {
    const p = this.params;
    const f = new Float32Array(8);
    f.set([p.alongamento, p.altura, p.larguraPixels, p.intensidade], 0);
    f.set([mundoPorPixel, p.tracoMaximo, distanciaCamera, 0], 4);
    this.device.queue.writeBuffer(this.uRasa, 0, f);
  }

  /**
   * @param {GPUCommandEncoder} enc
   * @param {object} q  o que muda a cada quadro
   * @param {number} q.mundoPorPixel  tamanho de um pixel, em unidades de mundo,
   *   a uma unidade de distância. Vem do renderizador porque só ele sabe a
   *   altura do canvas e o campo de visão.
   * @param {number} q.distanciaCamera  distância da órbita ao alvo, que é a
   *   escala em que a bruma de profundidade é medida.
   */
  avancar(enc, q) {
    this._escreverUniforme(q);
    const p = enc.beginComputePass();
    p.setPipeline(this.pipeAdv);
    p.setBindGroup(0, this.grupoAdv);
    p.dispatchWorkgroups(Math.ceil(this.n / 64));
    p.end();
  }

  desenhar(rp) {
    if (!this.pipeRender) return;
    rp.setPipeline(this.pipeRender);
    rp.setBindGroup(0, this.grupoRender);
    rp.draw(6, this.n);
  }

  destruir() {
    this.bufParts?.destroy();
    this.uRasa?.destroy();
  }
}
