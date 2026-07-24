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

import { Q } from '../../core/lattice.js';
import { shaderPasso, shaderInit, shaderMacros, planoDeBuffers, TIPO } from '../../core/emit/wgsl.js';
import { MAGIC } from '../../core/lattice.js';

/* Params: vec4<u32> + 4 f32 + vec4<f32> + vec4<f32> = 16 + 16 + 16 + 16 */
const PARAMS_BYTES = 64;

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

    this.layout = device.createBindGroupLayout({ entries: entradas, label: 'lbm' });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });

    const fontes = {
      passo: shaderPasso({ nbuf, escreverMacros: true }),
      init: shaderInit({ nbuf }),
      macros: shaderMacros({ nbuf }),
    };
    this.fontes = fontes;

    const pipe = async (nome, codigo) => {
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
        layout: pl, compute: { module: mod, entryPoint: 'main' }, label: nome,
      });
    };

    this.pipePasso = await pipe('passo', fontes.passo);
    this.pipeInit = await pipe('init', fontes.init);
    this.pipeMacros = await pipe('macros', fontes.macros);

    /* Dois bind groups: A->B e B->A. Trocar é trocar de grupo, não recriar. */
    const grupo = (src, dst) => device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        ...src.map((buf, i) => ({ binding: 1 + i, resource: { buffer: buf } })),
        ...dst.map((buf, i) => ({ binding: 1 + nbuf + i, resource: { buffer: buf } })),
        { binding: 1 + 2 * nbuf, resource: { buffer: this.tipo } },
        { binding: 2 + 2 * nbuf, resource: { buffer: this.macros } },
      ],
    });

    this.grupoAB = grupo(this.popA, this.popB);
    this.grupoBA = grupo(this.popB, this.popA);
    this.frente = 'A';   // onde está o estado atual
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
  configurar({ omegaPlus, magic = MAGIC.WALL, lesCs = 0.1, inletU, beltU = [0, 0, 0] }) {
    const buf = new ArrayBuffer(PARAMS_BYTES);
    new Uint32Array(buf, 0, 4).set([this.nx, this.ny, this.nz, 0]);
    new Float32Array(buf, 16, 4).set([omegaPlus, magic, lesCs, 0]);
    new Float32Array(buf, 32, 4).set([beltU[0], beltU[1], beltU[2], 0]);
    new Float32Array(buf, 48, 4).set([inletU[0], inletU[1], inletU[2], 0]);
    this.device.queue.writeBuffer(this.params, 0, buf);
    this.cfg = { omegaPlus, magic, lesCs, inletU, beltU };
  }

  /** Carrega o campo de tipos de célula (Uint32Array de N elementos). */
  definirTipos(tipos) {
    if (tipos.length !== this.N) {
      throw new Error(`campo de tipos com ${tipos.length}, esperado ${this.N}`);
    }
    this.device.queue.writeBuffer(this.tipo, 0, tipos);
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
      this.tipo, this.macros, this.params]) b.destroy();
  }
}

export { TIPO };
