/* ── src/backend/webgpu/solver.js ────────────────────────────────────────────
 *
 * O runtime WebGPU: alocação, pipelines e o laço de passo.
 *
 * Toda a física está nos shaders gerados por core/emit/wgsl.js. Este arquivo
 * cuida só de memória e despacho — e as duas decisões que importam aqui são o
 * fatiamento dos buffers e o ping-pong.
 *
 *
 * PING-PONG
 * ---------
 * O streaming lê o vizinho e escreve a própria célula. Fazer isso em um único
 * buffer é uma corrida: a célula vizinha pode já ter sido sobrescrita por
 * outro workgroup, ou não, dependendo da ordem de escalonamento — e o
 * resultado é um solver que dá respostas ligeiramente diferentes a cada
 * execução, o que é bem pior do que um que dá respostas erradas de forma
 * consistente. Dois conjuntos de buffers e troca a cada passo eliminam a
 * corrida por construção.
 *
 * (Existe um esquema em lugar — "Esoteric Pull", Lehmann 2022 — que elimina o
 * segundo conjunto e com ele metade da memória e da banda. Vale muito, e não
 * entra antes da suíte de validação passar: otimizar um solver que ainda não
 * se sabe correto é o caminho mais curto para não descobrir mais qual dos dois
 * problemas está causando o erro.)
 */

import { Q, MAGIC } from '../../core/lattice.js';
import {
  shaderPasso, shaderInit, shaderMacros, shaderForcas, shaderReduzir,
  planoDeBuffers, TIPO,
} from '../../core/emit/wgsl.js';

/* dim + (omega,magic,les,_) + belt + inlet + sponge = 5 x 16 bytes */
const PARAMS_BYTES = 80;

export class SolverWebGPU {
  constructor(device, { nx, ny, nz, nbuf, limites }) {
    this.device = device;
    this.nx = nx; this.ny = ny; this.nz = nz;
    this.N = nx * ny * nz;
    this.nbuf = nbuf;
    this.limites = limites;
    this.passos = 0;
    this.plano = planoDeBuffers(nbuf);
    this.dirsPorBuffer = Math.ceil(Q / nbuf);
  }

  /**
   * Cria o device, uma vez.
   *
   * Um GPUAdapter é CONSUMIDO pelo primeiro requestDevice: a segunda chamada
   * no mesmo adaptador rejeita com "adapter is consumed". Isso torna
   * impossível o padrão ingênuo de um device por solver — e a falha é
   * silenciosa se a promise não for aguardada, o que faz o app simplesmente
   * parar de progredir sem erro no console.
   *
   * Um device por página, portanto, e todos os solvers o compartilham. É
   * também o que se quer de qualquer jeito: device é objeto pesado, e trocar
   * de preset não deveria reconstruir o mundo.
   */
  static async criarDevice(adapter) {
    const L = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: L.maxBufferSize,
        maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
        maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
        maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
      },
    });
    if (!device) throw new Error('requestDevice devolveu null');

    /* Um device perdido (driver reiniciou, aba em segundo plano por muito
     * tempo, TDR do Windows) invalida tudo. Sem este handler a única pista
     * seria todo comando seguinte falhar em silêncio. */
    device.lost.then(info => {
      if (info.reason !== 'destroyed') {
        console.error(`[webgpu] device perdido: ${info.reason} — ${info.message}`);
      }
    });
    return device;
  }

  /**
   * Cria um solver sobre um device já existente, escolhendo o fatiamento de
   * buffers a partir dos limites reais e tentando a alocação de verdade.
   *
   * Se a alocação falhar — e ela falha, porque maxBufferSize é um limite
   * declarado e não uma reserva de VRAM — o erro sobe com o tamanho que foi
   * pedido, para a interface poder oferecer o preset abaixo em vez de mostrar
   * uma exceção de driver.
   */
  static async criar(device, { nx, ny, nz }) {
    const N = nx * ny * nz;
    const L = device.limits;

    const tetoBuffer = Math.min(L.maxStorageBufferBindingSize, L.maxBufferSize);
    const bytesPorDir = N * 4;

    /* Menos buffers é melhor: menos bindings, menos pressão no descritor. */
    let nbuf = 1;
    while (Math.ceil(Q / nbuf) * bytesPorDir > tetoBuffer) {
      nbuf++;
      if (nbuf > Q) {
        throw new Error(
          `Uma única direção ocupa ${(bytesPorDir / 1e9).toFixed(2)} GB e o ` +
          `device limita buffers a ${(tetoBuffer / 1e9).toFixed(2)} GB. ` +
          'Reduza a resolução.');
      }
    }

    /* 2 conjuntos de populações + tipo + macros (o uniforme não conta) */
    const storage = 2 * nbuf + 2;
    if (storage > L.maxStorageBuffersPerShaderStage) {
      throw new Error(
        `${storage} storage buffers necessários, device oferece ` +
        `${L.maxStorageBuffersPerShaderStage}. Reduza a resolução.`);
    }

    const s = new SolverWebGPU(device, { nx, ny, nz, nbuf, limites: L });
    await s._alocar();
    await s._compilar();
    return s;
  }

  get bytesTotais() {
    return this.N * (2 * Q * 4 + 4 + 16);   // populações + tipo + macros
  }

  /* ─────────────────────────────────────────────────────────────── memória */

  async _alocar() {
    const { device, N } = this;
    const bytesFatia = this.dirsPorBuffer * N * 4;

    const criar = (tam, uso, rotulo) => {
      const b = device.createBuffer({ size: tam, usage: uso, label: rotulo });
      if (!b) throw new Error(`falha ao alocar ${rotulo}`);
      return b;
    };

    const ST = GPUBufferUsage.STORAGE;
    this.popA = [];
    this.popB = [];
    for (let i = 0; i < this.nbuf; i++) {
      this.popA.push(criar(bytesFatia, ST | GPUBufferUsage.COPY_DST, `popA${i}`));
      this.popB.push(criar(bytesFatia, ST | GPUBufferUsage.COPY_DST, `popB${i}`));
    }

    this.tipo = criar(N * 4, ST | GPUBufferUsage.COPY_DST, 'tipo');
    this.macros = criar(N * 16, ST | GPUBufferUsage.COPY_SRC, 'macros');
    this.params = criar(PARAMS_BYTES,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'params');

    /* Um vec4 por workgroup do kernel de forças. */
    this.nWorkgroups = Math.ceil(this.nx / 64) * this.ny * this.nz;
    this.parciais = criar(this.nWorkgroups * 16, ST, 'parciais');
    this.totalForca = criar(16, ST | GPUBufferUsage.COPY_SRC, 'totalForca');
    this.nPart = criar(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'nPart');
    device.queue.writeBuffer(this.nPart, 0,
      new Uint32Array([this.nWorkgroups, 0, 0, 0]));
    this.leituraForca = criar(16,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, 'leituraForca');

    /* Um erro de alocação em WebGPU chega de forma assíncrona no device; sem
     * este dreno ele viraria um `undefined` sem explicação lá na frente. */
    device.pushErrorScope('out-of-memory');
    const erro = await device.popErrorScope();
    if (erro) {
      throw new Error(
        `VRAM insuficiente para ${(this.bytesTotais / 1e9).toFixed(2)} GB ` +
        `(${this.nx}x${this.ny}x${this.nz}). Escolha um preset menor.`);
    }
  }

  /* ────────────────────────────────────────────────────────────── pipelines */

  async _compilar() {
    const { device, nbuf } = this;

    const entradas = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ];
    let b = 1;
    for (let i = 0; i < nbuf; i++) {
      entradas.push({ binding: b++, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
    }
    for (let i = 0; i < nbuf; i++) {
      entradas.push({ binding: b++, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
    }
    entradas.push({ binding: b++, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
    entradas.push({ binding: b++, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
    /* o kernel de forças acrescenta as parciais no fim do mesmo layout, para
     * os quatro kernels compartilharem um bind group só */
    entradas.push({ binding: b++, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });

    this.layout = device.createBindGroupLayout({ entries: entradas, label: 'lbm' });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });

    const fontes = {
      passo: shaderPasso({ nbuf, escreverMacros: true }),
      init: shaderInit({ nbuf }),
      macros: shaderMacros({ nbuf }),
      forcas: shaderForcas({ nbuf }),
      reduzir: shaderReduzir(),
    };
    this.fontes = fontes;

    const pipe = async (nome, codigo, layoutPipeline) => {
      device.pushErrorScope('validation');
      const mod = device.createShaderModule({ code: codigo, label: nome });
      const info = await mod.getCompilationInfo();
      const erros = info.messages.filter(m => m.type === 'error');
      const err = await device.popErrorScope();
      if (erros.length || err) {
        const det = erros.map(m => `  linha ${m.lineNum}: ${m.message}`).join('\n');
        throw new Error(`shader "${nome}" não compilou:\n${det || err?.message}`);
      }
      return device.createComputePipeline({
        layout: layoutPipeline, compute: { module: mod, entryPoint: 'main' },
        label: nome,
      });
    };

    this.pipePasso = await pipe('passo', fontes.passo, pl);
    this.pipeInit = await pipe('init', fontes.init, pl);
    this.pipeMacros = await pipe('macros', fontes.macros, pl);
    this.pipeForcas = await pipe('forcas', fontes.forcas, pl);

    /* Dois bind groups: A->B e B->A. Trocar é trocar de grupo, não recriar. */
    const grupo = (src, dst) => device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        ...src.map((buf, i) => ({ binding: 1 + i, resource: { buffer: buf } })),
        ...dst.map((buf, i) => ({ binding: 1 + nbuf + i, resource: { buffer: buf } })),
        { binding: 1 + 2 * nbuf, resource: { buffer: this.tipo } },
        { binding: 2 + 2 * nbuf, resource: { buffer: this.macros } },
        { binding: 3 + 2 * nbuf, resource: { buffer: this.parciais } },
      ],
    });

    this.grupoAB = grupo(this.popA, this.popB);
    this.grupoBA = grupo(this.popB, this.popA);
    this.frente = 'A';   // onde está o estado atual

    /* A redução tem layout próprio: ela não toca em populações. */
    this.layoutRed = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const plRed = device.createPipelineLayout({ bindGroupLayouts: [this.layoutRed] });
    this.pipeReduzir = await pipe('reduzir', fontes.reduzir, plRed);
    this.grupoRed = device.createBindGroup({
      layout: this.layoutRed,
      entries: [
        { binding: 0, resource: { buffer: this.nPart } },
        { binding: 1, resource: { buffer: this.parciais } },
        { binding: 2, resource: { buffer: this.totalForca } },
      ],
    });
  }

  /* ─────────────────────────────────────────────────────────── configuração */

  /**
   * @param {object} o
   * @param {number} o.omegaPlus  taxa de relaxação par (de Units)
   * @param {number} [o.magic]    Lambda TRT; padrão 3/16 (parede no meio)
   * @param {number} [o.lesCs]    Smagorinsky; 0 desliga
   * @param {number[]} o.inletU   corrente livre em unidades de lattice
   * @param {number[]} [o.beltU]  velocidade da esteira
   */
  configurar({
    omegaPlus, magic = MAGIC.WALL, lesCs = 0.1, inletU,
    beltU = [0, 0, 0], esponja = null, espessuraCL = 0,
  }) {
    const esp = esponja ?? {
      inicio: Math.round(this.nx * 0.82),
      comprimento: Math.max(4, Math.round(this.nx * 0.18)),
    };
    const buf = new ArrayBuffer(PARAMS_BYTES);
    new Uint32Array(buf, 0, 4).set([this.nx, this.ny, this.nz, 0]);
    new Float32Array(buf, 16, 4).set([omegaPlus, magic, lesCs, 0]);
    new Float32Array(buf, 32, 4).set([beltU[0], beltU[1], beltU[2], 0]);
    new Float32Array(buf, 48, 4).set([inletU[0], inletU[1], inletU[2], 0]);
    new Float32Array(buf, 64, 4).set([esp.inicio, esp.comprimento, espessuraCL, 0]);
    this.device.queue.writeBuffer(this.params, 0, buf);
    this.cfg = { omegaPlus, magic, lesCs, inletU, beltU, esponja: esp, espessuraCL };
  }

  /** Carrega o campo de tipos de célula (Uint32Array de N elementos). */
  definirTipos(tipos) {
    if (tipos.length !== this.N) {
      throw new Error(`campo de tipos com ${tipos.length}, esperado ${this.N}`);
    }
    this.device.queue.writeBuffer(this.tipo, 0, tipos);
  }

  /**
   * Parte do repouso e acelera até a velocidade pedida ao longo de `passos`.
   *
   * POR QUE NÃO SE LIGA O TÚNEL DE UMA VEZ
   * --------------------------------------
   * Inicializar o domínio inteiro com escoamento uniforme e um carro já dentro
   * dele é ligar o vento instantaneamente: no primeiro passo o ar bate no
   * nariz a 30 m/s e nasce um pulso de pressão que atravessa o domínio. Num
   * lattice a ω = 1,98 a viscosidade molecular é ~1,7e-3 e esse pulso quase
   * não amortece — ele reverbera entre o corpo e os contornos e cresce.
   *
   * Medido aqui, com um Tesla a 32 células: no passo 200 o desvio de densidade
   * já era de 5,7% (contra os ~1e-4 do regime), e entre os passos 200 e 800
   * mais da metade das células de fluido virava NaN. O solver não estava
   * errado; a partida estava.
   *
   * A rampa resolve porque a escala de tempo do transiente passa a ser a da
   * rampa, e não um degrau. Uma travessia do domínio é generoso e custa pouco:
   * é tempo que o escoamento levaria para se estabelecer de qualquer maneira.
   */
  rampa(passos, aplicarVelocidade) {
    this._rampa = { passos, aplicar: aplicarVelocidade, atual: 0 };
  }

  /** Fração da velocidade final neste instante da rampa (1 quando terminou). */
  get fatorRampa() {
    const r = this._rampa;
    if (!r || r.atual >= r.passos) return 1;
    /* Suavizada nas duas pontas: uma rampa linear tem um degrau de aceleração
     * no início e outro no fim, e degraus de aceleração também geram pulso. */
    const t = r.atual / r.passos;
    return t * t * (3 - 2 * t);
  }

  /** Preenche o domínio com o equilíbrio da corrente livre. */
  inicializar() {
    const enc = this.device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(this.pipeInit);
    p.setBindGroup(0, this.grupoAB);   // init escreve em dst = B
    p.dispatchWorkgroups(Math.ceil(this.nx / 64), this.ny, this.nz);
    p.end();
    this.device.queue.submit([enc.finish()]);
    this.frente = 'B';
    this.passos = 0;
    if (this._rampa) this._rampa.atual = 0;
  }

  /**
   * Preenche o estado a partir de um campo dado na CPU.
   *
   * `fn(x, y, z)` devolve `{ delta, u }` e o equilíbrio correspondente é
   * calculado com a MESMA função de lattice.js que a suíte de validação usa.
   * Isso faz de cada corrida um teste implícito de que o equilíbrio emitido no
   * shader e o escrito em JS concordam: se divergissem, o campo inicial já
   * sairia diferente do esperado no primeiro passo.
   *
   * Caro (constrói N*Q floats na CPU) e é para isso mesmo — casos de
   * validação com perfil analítico, não o caminho normal de partida.
   */
  definirCampo(fn, equilibrium) {
    const { nx, ny, nz, N } = this;
    const fatia = this.dirsPorBuffer;
    const dados = Array.from({ length: this.nbuf },
      () => new Float32Array(fatia * N));

    const g = new Float64Array(Q);
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const cell = (z * ny + y) * nx + x;
          const { delta = 0, u = [0, 0, 0] } = fn(x, y, z) || {};
          equilibrium(delta, u, g);
          for (let i = 0; i < Q; i++) {
            const { buffer, offset } = this.plano[i];
            dados[buffer][offset * N + cell] = g[i];
          }
        }
      }
    }

    const alvo = this.frente === 'A' ? this.popA : this.popB;
    for (let i = 0; i < this.nbuf; i++) {
      this.device.queue.writeBuffer(alvo[i], 0, dados[i]);
    }
    this.passos = 0;
  }

  /* ──────────────────────────────────────────────────────────────── o laço */

  /** Avança `n` passos. Um único command encoder para os n — o custo de
   *  submeter domina o de despachar quando o lattice é pequeno. */
  passo(n = 1) {
    /* Avanca a rampa antes de despachar: o chamador reconfigura o uniforme
       com o fator novo e so entao os n passos rodam. */
    const r = this._rampa;
    if (r && r.atual < r.passos) {
      r.atual += n;
      r.aplicar?.(this.fatorRampa);
    }
    const enc = this.device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(this.pipePasso);
    const gx = Math.ceil(this.nx / 64);
    for (let k = 0; k < n; k++) {
      p.setBindGroup(0, this.frente === 'A' ? this.grupoAB : this.grupoBA);
      p.dispatchWorkgroups(gx, this.ny, this.nz);
      this.frente = this.frente === 'A' ? 'B' : 'A';
    }
    p.end();
    this.device.queue.submit([enc.finish()]);
    this.passos += n;
  }

  /* ─────────────────────────────────────────────────────────────── leitura */

  /**
   * Lê o campo macroscópico de volta para a CPU: Float32Array de N*4 com
   * (ux, uy, uz, delta) por célula.
   *
   * Isto sincroniza com a GPU e é caro. Serve para validação e para o cálculo
   * de forças na CPU enquanto forces.js não existe — não para o laço de
   * render, que lê os buffers direto na GPU sem nunca trazê-los de volta.
   */
  async lerMacros() {
    const bytes = this.N * 16;
    const stage = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.macros, 0, stage, 0, bytes);
    this.device.queue.submit([enc.finish()]);

    await stage.mapAsync(GPUMapMode.READ);
    const saida = new Float32Array(stage.getMappedRange().slice(0));
    stage.unmap();
    stage.destroy();
    return saida;
  }

  /**
   * Força sobre o corpo, em unidades de lattice, por troca de momento.
   *
   * Dois despachos e uma leitura de 16 bytes. A leitura sincroniza com a GPU,
   * então não se mede força todo passo: o coeficiente é uma média sobre
   * centenas de passos de qualquer forma, e amostrar a cada 20 ou 50 passos
   * custa nada e não engasga o pipeline.
   *
   * @returns {Promise<number[]>} [fx, fy, fz]
   */
  /**
   * Mede a força de repouso e passa a subtraí-la de toda medida seguinte.
   *
   * POR QUE UM CORPO PARADO EM FLUIDO PARADO SENTE FORÇA
   * ----------------------------------------------------
   * Em repouso g_i = 0, logo f_i = w_i, e a soma de troca de momento vale
   *
   *     F = soma_links c_i (w_i + w_ī) = soma_links 2 w_i c_i
   *
   * Numa superfície FECHADA isso é exatamente zero: cada direção que sai do
   * corpo num ponto tem a oposta saindo do outro lado, e os termos se cancelam
   * aos pares. Mas um carro repousa sobre o piso, e a parte da carroceria que
   * encosta nele não tem fluido do outro lado — aqueles links não existem. A
   * soma deixa de fechar e sobra uma força constante, para baixo, que não é
   * física nenhuma.
   *
   * Medimos: com um Tesla a 32 células, ela dava fz = -30,15 em unidades de
   * lattice — um coeficiente de sustentação de -146 antes de o escoamento
   * sequer começar. A magnitude absurda foi a única pista.
   *
   * A correção é subtrair o valor medido no estado zerado. Vale para qualquer
   * geometria, incluindo as que encostam em mais de uma parede, e some sozinha
   * quando o corpo está inteiramente imerso (o offset dá zero e a subtração é
   * inócua). Tem de ser chamada DEPOIS de definirTipos e ANTES de inicializar,
   * porque só aí os buffers ainda estão zerados.
   */
  async calibrarRepouso() {
    this.offsetForca = [0, 0, 0];
    this.offsetForca = await this.medirForcas();
    return this.offsetForca;
  }

  async medirForcas() {
    const enc = this.device.createCommandEncoder();
    const p = enc.beginComputePass();

    p.setPipeline(this.pipeForcas);
    /* lê o estado atual, que está no lado `src` do grupo da frente */
    p.setBindGroup(0, this.frente === 'A' ? this.grupoAB : this.grupoBA);
    p.dispatchWorkgroups(Math.ceil(this.nx / 64), this.ny, this.nz);

    p.setPipeline(this.pipeReduzir);
    p.setBindGroup(0, this.grupoRed);
    p.dispatchWorkgroups(1);
    p.end();

    enc.copyBufferToBuffer(this.totalForca, 0, this.leituraForca, 0, 16);
    this.device.queue.submit([enc.finish()]);

    await this.leituraForca.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(this.leituraForca.getMappedRange().slice(0));
    this.leituraForca.unmap();
    const o = this.offsetForca ?? [0, 0, 0];
    return [v[0] - o[0], v[1] - o[1], v[2] - o[2]];
  }

  /** Recalcula macros a partir do estado atual sem avançar o tempo. */
  atualizarMacros() {
    const enc = this.device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(this.pipeMacros);
    /* macros lê de `src`, então precisamos do grupo cujo src é a frente */
    p.setBindGroup(0, this.frente === 'A' ? this.grupoAB : this.grupoBA);
    p.dispatchWorkgroups(Math.ceil(this.nx / 64), this.ny, this.nz);
    p.end();
    this.device.queue.submit([enc.finish()]);
  }

  /** Libera os buffers. NÃO destrói o device — ele é compartilhado. */
  destruir() {
    for (const b of [...this.popA, ...this.popB,
      this.tipo, this.macros, this.params, this.parciais,
      this.totalForca, this.nPart, this.leituraForca]) b.destroy();
  }
}

export { TIPO };
