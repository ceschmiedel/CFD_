/* ── src/backend/webgl2/solver.js ────────────────────────────────────────────
 *
 * O runtime WebGL2: texturas, framebuffers e o laço de passo.
 *
 * Toda a física está nos shaders gerados por core/emit/glsl.js, que saem do
 * MESMO ir.js que gera o WGSL. Este arquivo cuida só de memória e despacho — e
 * a API pública é deliberadamente a mesma do SolverWebGPU, para as suítes de
 * validação rodarem os dois backends com o mesmo código de teste. É esse
 * confronto que dá sentido ao emissor compartilhado: uma discordância entre
 * eles passa a ser, por construção, bug de memória ou de despacho.
 *
 *
 * O QUE MUDA SEM COMPUTE
 * ----------------------
 * WebGL2 não tem compute shader, storage buffer, memória compartilhada de
 * workgroup nem atômico de float. Tudo o que o solver faz tem de caber em
 * "desenhar um triângulo que cobre uma textura":
 *
 *   lattice 3D          -> atlas 2D, fatias em z ladrilhadas
 *   19 populações       -> 5 texturas RGBA32F escritas de uma vez por MRT
 *   ping-pong           -> dois conjuntos de 5, dois framebuffers
 *   redução das forças  -> pirâmide de somas 2x2 até sobrar um texel
 *
 * O ping-pong existe pelo mesmo motivo do WebGPU — o streaming lê o vizinho e
 * escreve a própria célula —, mas aqui ele não é opcional nem por um instante:
 * ler e escrever a mesma textura no mesmo draw é comportamento indefinido, e o
 * driver não avisa.
 *
 *
 * O TETO REAL É A ÁREA DE TEXTURA, NÃO A VRAM
 * -------------------------------------------
 * Um atlas precisa caber em MAX_TEXTURE_SIZE nos dois eixos. Num preset
 * "Extrema" isso dá 7040x7680, dentro dos 16384 usuais — mas as dez texturas
 * RGBA32F somam 8,6 GB, que nenhuma placa de consumo aloca. O `criar` tenta a
 * alocação de verdade e verifica o framebuffer: um teto declarado é hipótese,
 * não contrato.
 */

import { Q, MAGIC } from '../../core/lattice.js';
import { atlasLayout } from '../caps.js';
import {
  shaderPasso, shaderInit, shaderMacros, shaderForcas, shaderReduzir,
  VERTEX_COBERTURA, N_ALVOS, TIPO,
} from '../../core/emit/glsl.js';

export class SolverWebGL2 {
  constructor(gl, { nx, ny, nz, atlas }) {
    this.gl = gl;
    this.nx = nx; this.ny = ny; this.nz = nz;
    this.N = nx * ny * nz;
    this.atlas = atlas;                 // { tx, ty, w, h }
    this.passos = 0;
    this.offsetForca = [0, 0, 0];
  }

  /**
   * Cria o contexto, uma vez.
   *
   * `preserveDrawingBuffer: false` e `depth/stencil: false` porque o canvas
   * aqui é só o dono do contexto — nada é desenhado nele. Todo o trabalho
   * acontece em framebuffers fora da tela.
   *
   * EXT_color_buffer_float é requisito DURO: sem render target de float não há
   * como escrever populações, e não existe contorno. `caps.js` já reprova a
   * máquina antes de chegar aqui; esta segunda checagem existe porque o
   * contexto pode ser outro.
   */
  static criarContexto(canvas = document.createElement('canvas')) {
    const gl = canvas.getContext('webgl2', {
      antialias: false, depth: false, stencil: false,
      preserveDrawingBuffer: false, powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('contexto WebGL2 não pôde ser criado');
    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error(
        'EXT_color_buffer_float ausente — sem render target de float o solver ' +
        'não roda, e não há contorno para isso.');
    }
    return gl;
  }

  static async criar(gl, { nx, ny, nz }) {
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    const atlas = atlasLayout(nx, ny, nz, maxTex);
    if (!atlas) {
      throw new Error(
        `${nx}x${ny}x${nz} não ladrilha dentro de ${maxTex}px de textura. ` +
        'Escolha um preset menor.');
    }
    const alvos = gl.getParameter(gl.MAX_DRAW_BUFFERS);
    if (alvos < N_ALVOS) {
      throw new Error(`${N_ALVOS} alvos de cor necessários, driver oferece ${alvos}.`);
    }

    const s = new SolverWebGL2(gl, { nx, ny, nz, atlas });
    s._compilar();
    s._alocar();
    return s;
  }

  get bytesTotais() {
    const { w, h } = this.atlas;
    /* 2 conjuntos de 5 RGBA32F + macros + forças + tipo (R32UI) */
    return w * h * (2 * N_ALVOS * 16 + 16 + 16 + 4);
  }

  /* ────────────────────────────────────────────────────────────── programas */

  _compilar() {
    const gl = this.gl;

    const shader = (tipo, src, rotulo) => {
      const s = gl.createShader(tipo);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(s);
        /* A linha do log é inútil sem a fonte; anexamos as vizinhas do erro. */
        const n = /(\d+):(\d+)/.exec(log ?? '');
        const trecho = n
          ? src.split('\n').slice(Math.max(0, +n[2] - 4), +n[2] + 2)
            .map((l, k) => `${+n[2] - 3 + k} | ${l}`).join('\n')
          : '';
        throw new Error(`shader "${rotulo}" não compilou:\n${log}\n${trecho}`);
      }
      return s;
    };

    const vs = shader(gl.VERTEX_SHADER, VERTEX_COBERTURA, 'cobertura');

    const programa = (rotulo, fonte) => {
      const fs = shader(gl.FRAGMENT_SHADER, fonte, rotulo);
      const p = gl.createProgram();
      gl.attachShader(p, vs);
      gl.attachShader(p, fs);
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        throw new Error(`programa "${rotulo}" não ligou: ${gl.getProgramInfoLog(p)}`);
      }
      gl.deleteShader(fs);
      /*
       * As localizações são resolvidas UMA vez e guardadas.
       *
       * getUniformLocation devolve null para uniforme que o compilador
       * eliminou por não ser usado, e ligar em null é uma operação válida que
       * não faz nada. É assim que um backend roda inteiro com o valor padrão de
       * um parâmetro e devolve um número plausível e errado — por isso
       * tests/webgl2.html confere que todos sobreviveram.
       */
      const u = new Proxy({}, {
        get: (cache, nome) => {
          if (!(nome in cache)) cache[nome] = gl.getUniformLocation(p, nome);
          return cache[nome];
        },
      });
      return { p, u };
    };

    this.fontes = {
      passo: shaderPasso(),
      init: shaderInit(),
      macros: shaderMacros(),
      forcas: shaderForcas(),
      reduzir: shaderReduzir(),
    };
    this.progPasso = programa('passo', this.fontes.passo);
    this.progInit = programa('init', this.fontes.init);
    this.progMacros = programa('macros', this.fontes.macros);
    this.progForcas = programa('forcas', this.fontes.forcas);
    this.progReduzir = programa('reduzir', this.fontes.reduzir);
    gl.deleteShader(vs);

    /* Um VAO vazio. O triângulo de cobertura sai do gl_VertexID e não tem
     * atributo nenhum, mas o desenho sem VAO ligado é erro em alguns drivers
     * (ANGLE inclusive) e o sintoma é uma tela que simplesmente não muda. */
    this.vao = gl.createVertexArray();
  }

  /* ─────────────────────────────────────────────────────────────── memória */

  _textura(formatoInterno, formato, tipo) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    /* NEAREST e clamp em tudo: cada texel É uma célula, e interpolar entre
     * células vizinhas não significa nada aqui. O acesso é sempre texelFetch,
     * que ignora o filtro — mas uma textura float sem NEAREST é incompleta em
     * drivers sem OES_texture_float_linear, e uma textura incompleta amostra
     * preto sem avisar. */
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texStorage2D(gl.TEXTURE_2D, 1, formatoInterno, this.atlas.w, this.atlas.h);
    const err = gl.getError();
    if (err !== gl.NO_ERROR) {
      throw new Error(
        `falha ao alocar textura ${this.atlas.w}x${this.atlas.h} ` +
        `(GL error 0x${err.toString(16)}) — ${(this.bytesTotais / 1e9).toFixed(2)} GB ` +
        'no total. Escolha um preset menor.');
    }
    return t;
  }

  _fbo(texturas) {
    const gl = this.gl;
    const f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    texturas.forEach((t, i) => {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i,
        gl.TEXTURE_2D, t, 0);
    });
    const estado = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (estado !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`framebuffer incompleto (0x${estado.toString(16)}) ` +
        `com ${texturas.length} anexos de ${this.atlas.w}x${this.atlas.h}`);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return f;
  }

  _alocar() {
    const gl = this.gl;
    const F32 = () => this._textura(gl.RGBA32F, gl.RGBA, gl.FLOAT);

    this.popA = Array.from({ length: N_ALVOS }, F32);
    this.popB = Array.from({ length: N_ALVOS }, F32);
    this.macrosTex = F32();
    this.forcaTex = F32();
    this.tipoTex = this._textura(gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT);

    this.fboA = this._fbo(this.popA);      // escreve em A
    this.fboB = this._fbo(this.popB);      // escreve em B
    this.fboMacros = this._fbo([this.macrosTex]);
    this.fboForca = this._fbo([this.forcaTex]);

    /*
     * A pirâmide de redução. Cada nível tem metade da dimensão do anterior, e
     * o último é 1x1 — é dele que a força sai por readPixels.
     *
     * As texturas são pequenas (o segundo nível já é um quarto do atlas) e
     * ficam alocadas para sempre em vez de criadas por medição: medir força
     * acontece algumas vezes por segundo, e criar e destruir uma dúzia de
     * texturas nesse ritmo é como se descobre o custo de um driver.
     */
    this.piramide = [];
    let w = this.atlas.w, h = this.atlas.h;
    while (w > 1 || h > 1) {
      w = Math.max(1, Math.ceil(w / 2));
      h = Math.max(1, Math.ceil(h / 2));
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, w, h);
      const f = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.piramide.push({ t, f, w, h });
    }

    this.frente = 'A';
    this.leitura = new Float32Array(4);
  }

  /* ───────────────────────────────────────────────── conversão de endereço */

  /** Célula linear (z*ny + y)*nx + x -> texel do atlas, em índice de array. */
  _texelDe(x, y, z, canais = 1) {
    const { tx, w } = this.atlas;
    const cx = x + (z % tx) * this.nx;
    const cy = y + Math.floor(z / tx) * this.ny;
    return (cy * w + cx) * canais;
  }

  /* ─────────────────────────────────────────────────────────── configuração */

  configurar({
    omegaPlus, magic = MAGIC.WALL, lesCs = 0.1, inletU,
    beltU = [0, 0, 0], esponja = null, espessuraCL = 0,
  }) {
    const esp = esponja ?? {
      inicio: Math.round(this.nx * 0.82),
      comprimento: Math.max(4, Math.round(this.nx * 0.18)),
    };
    this.cfg = { omegaPlus, magic, lesCs, inletU, beltU, esponja: esp, espessuraCL };
  }

  /**
   * Liga os uniformes de um programa.
   *
   * Escritos a cada draw e não uma vez na configuração: em WebGL2 o estado de
   * uniforme pertence ao PROGRAMA, e são cinco programas compartilhando os
   * mesmos parâmetros. Cinco escritas de meia dúzia de floats por passo não
   * aparecem em medição nenhuma, e a alternativa — lembrar qual programa está
   * com qual valor — é a fonte clássica de um backend que roda com a
   * viscosidade do caso anterior.
   */
  _ligarUniformes({ p, u }) {
    const gl = this.gl;
    const c = this.cfg;
    gl.useProgram(p);
    gl.uniform3i(u.uDim, this.nx, this.ny, this.nz);
    gl.uniform2i(u.uTiles, this.atlas.tx, this.atlas.ty);
    if (c) {
      gl.uniform1f(u.uOmega, c.omegaPlus);
      gl.uniform1f(u.uMagicU, c.magic);
      gl.uniform1f(u.uLesCsU, c.lesCs);
      gl.uniform3f(u.uBeltU, ...c.beltU);
      gl.uniform3f(u.uInletU, ...c.inletU);
      gl.uniform3f(u.uSponge, c.esponja.inicio, c.esponja.comprimento, c.espessuraCL);
    }
    /* Unidades 0..4 para as populações, 5 para o campo de tipos. */
    const src = this.frente === 'A' ? this.popA : this.popB;
    for (let i = 0; i < N_ALVOS; i++) {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, src[i]);
      gl.uniform1i(u[`uSrc${i}`], i);
    }
    gl.activeTexture(gl.TEXTURE0 + N_ALVOS);
    gl.bindTexture(gl.TEXTURE_2D, this.tipoTex);
    gl.uniform1i(u.uTipo, N_ALVOS);
  }

  /** Um draw cobrindo o alvo inteiro. */
  _desenhar(fbo, w, h, nAlvos = 1) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.drawBuffers(Array.from({ length: nAlvos }, (_, i) => gl.COLOR_ATTACHMENT0 + i));
    gl.viewport(0, 0, w, h);
    gl.bindVertexArray(this.vao);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Carrega o campo de tipos (Uint32Array de N, em ordem linear de célula). */
  definirTipos(tipos) {
    if (tipos.length !== this.N) {
      throw new Error(`campo de tipos com ${tipos.length}, esperado ${this.N}`);
    }
    const { w, h } = this.atlas;
    /*
     * O atlas tem ladrilhos sobrando na última linha (nz raramente é múltiplo
     * de tx). Eles são preenchidos com SOLIDO: o shader do passo já sai cedo
     * neles, o de forças os ignora, e um ladrilho de lixo seria lido como
     * fluido com populações indefinidas — que é o tipo de coisa que aparece
     * como uma força espúria constante e some quando você olha.
     */
    const atlas = new Uint32Array(w * h).fill(TIPO.SOLIDO);
    for (let z = 0; z < this.nz; z++) {
      for (let y = 0; y < this.ny; y++) {
        const base = (z * this.ny + y) * this.nx;
        for (let x = 0; x < this.nx; x++) {
          atlas[this._texelDe(x, y, z)] = tipos[base + x];
        }
      }
    }
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tipoTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h,
      gl.RED_INTEGER, gl.UNSIGNED_INT, atlas);
  }

  /* ────────────────────────────────────────────────────────── partida e laço */

  rampa(passos, aplicarVelocidade) {
    this._rampa = { passos, aplicar: aplicarVelocidade, atual: 0 };
  }

  get fatorRampa() {
    const r = this._rampa;
    if (!r || r.atual >= r.passos) return 1;
    const t = r.atual / r.passos;
    return t * t * (3 - 2 * t);
  }

  /** Preenche o domínio com o equilíbrio da corrente livre. */
  inicializar() {
    const gl = this.gl;
    /* Init lê A e escreve B, como no WebGPU — a frente passa a ser B. */
    this.frente = 'A';
    this._ligarUniformes(this.progInit);
    this._desenhar(this.fboB, this.atlas.w, this.atlas.h, N_ALVOS);
    this.frente = 'B';
    this.passos = 0;
    if (this._rampa) this._rampa.atual = 0;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Preenche o estado a partir de um campo dado na CPU.
   *
   * Mesma assinatura do backend WebGPU: `fn(x,y,z)` devolve `{ delta, u }` e o
   * equilíbrio sai da MESMA função de lattice.js que a suíte usa. Cada corrida
   * vira um teste implícito de que o equilíbrio emitido no shader e o escrito
   * em JS concordam.
   */
  definirCampo(fn, equilibrium) {
    const { nx, ny, nz } = this;
    const { w, h } = this.atlas;
    const dados = Array.from({ length: N_ALVOS }, () => new Float32Array(w * h * 4));
    const g = new Float64Array(Q);

    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const { delta = 0, u = [0, 0, 0] } = fn(x, y, z) || {};
          equilibrium(delta, u, g);
          const base = this._texelDe(x, y, z, 4);
          for (let i = 0; i < Q; i++) {
            dados[Math.floor(i / 4)][base + (i % 4)] = g[i];
          }
        }
      }
    }

    const gl = this.gl;
    const alvo = this.frente === 'A' ? this.popA : this.popB;
    for (let i = 0; i < N_ALVOS; i++) {
      gl.bindTexture(gl.TEXTURE_2D, alvo[i]);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, gl.FLOAT, dados[i]);
    }
    this.passos = 0;
  }

  /** Avança `n` passos. */
  passo(n = 1) {
    const r = this._rampa;
    if (r && r.atual < r.passos) {
      r.atual += n;
      r.aplicar?.(this.fatorRampa);
    }
    const gl = this.gl;
    for (let k = 0; k < n; k++) {
      this._ligarUniformes(this.progPasso);
      this._desenhar(this.frente === 'A' ? this.fboB : this.fboA,
        this.atlas.w, this.atlas.h, N_ALVOS);
      this.frente = this.frente === 'A' ? 'B' : 'A';
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.passos += n;
  }

  /* ─────────────────────────────────────────────────────────────── leitura */

  /** Recalcula o campo macroscópico a partir do estado atual. */
  atualizarMacros() {
    this._ligarUniformes(this.progMacros);
    this._desenhar(this.fboMacros, this.atlas.w, this.atlas.h, 1);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
  }

  /**
   * Lê o campo macroscópico para a CPU: Float32Array de N*4 com
   * (ux, uy, uz, delta) por célula, em ordem linear.
   *
   * Sincroniza com a GPU e desladrilha o atlas na CPU — é caro e é para
   * validação, não para laço de render.
   */
  async lerMacros() {
    this.atualizarMacros();
    const gl = this.gl;
    const { w, h } = this.atlas;
    const bruto = new Float32Array(w * h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboMacros);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, bruto);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    const saida = new Float32Array(this.N * 4);
    for (let z = 0; z < this.nz; z++) {
      for (let y = 0; y < this.ny; y++) {
        for (let x = 0; x < this.nx; x++) {
          const de = this._texelDe(x, y, z, 4);
          const para = ((z * this.ny + y) * this.nx + x) * 4;
          saida[para] = bruto[de];
          saida[para + 1] = bruto[de + 1];
          saida[para + 2] = bruto[de + 2];
          saida[para + 3] = bruto[de + 3];
        }
      }
    }
    return saida;
  }

  /**
   * Mede a força de repouso e passa a subtraí-la de toda medida seguinte.
   *
   * Um corpo que encosta no piso não tem fluido do outro lado na região de
   * contato, a soma de troca de momento deixa de fechar e sobra uma força
   * constante que não é física nenhuma. Ver o comentário extenso no backend
   * WebGPU — o fenômeno é do método, não do backend, e a correção é a mesma.
   * Chamar DEPOIS de definirTipos e ANTES de inicializar.
   */
  async calibrarRepouso() {
    this.offsetForca = [0, 0, 0];
    this.offsetForca = await this.medirForcas();
    return this.offsetForca;
  }

  /**
   * Força sobre o corpo, em unidades de lattice.
   *
   * Um draw por célula somando os links que cruzam a superfície, e depois a
   * pirâmide: cada nível soma blocos 2x2 do anterior até sobrar um texel. Doze
   * draws minúsculos num atlas de 2560², contra o `readPixels` de um pixel só
   * no fim — que é a única sincronização com a GPU.
   */
  async medirForcas() {
    const gl = this.gl;

    /* O passe de forças descarta os ladrilhos além de nz; limpar antes é o que
     * garante que eles somem ZERO na pirâmide em vez do que houvesse ali. */
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboForca);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    this._ligarUniformes(this.progForcas);
    this._desenhar(this.fboForca, this.atlas.w, this.atlas.h, 1);

    const { p, u } = this.progReduzir;
    gl.useProgram(p);
    let fonte = this.forcaTex;
    let tamW = this.atlas.w, tamH = this.atlas.h;
    for (const nivel of this.piramide) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, fonte);
      gl.uniform1i(u.uFonte, 0);
      gl.uniform2i(u.uTam, tamW, tamH);
      this._desenhar(nivel.f, nivel.w, nivel.h, 1);
      fonte = nivel.t; tamW = nivel.w; tamH = nivel.h;
    }

    const topo = this.piramide[this.piramide.length - 1];
    gl.bindFramebuffer(gl.FRAMEBUFFER, topo.f);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, this.leitura);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    const o = this.offsetForca ?? [0, 0, 0];
    return [this.leitura[0] - o[0], this.leitura[1] - o[1], this.leitura[2] - o[2]];
  }

  /** Libera texturas e framebuffers. NÃO destrói o contexto. */
  destruir() {
    const gl = this.gl;
    for (const t of [...this.popA, ...this.popB, this.macrosTex, this.forcaTex,
      this.tipoTex, ...this.piramide.map(n => n.t)]) gl.deleteTexture(t);
    for (const f of [this.fboA, this.fboB, this.fboMacros, this.fboForca,
      ...this.piramide.map(n => n.f)]) gl.deleteFramebuffer(f);
    for (const pr of [this.progPasso, this.progInit, this.progMacros,
      this.progForcas, this.progReduzir]) gl.deleteProgram(pr.p);
    gl.deleteVertexArray(this.vao);
    this.piramide = [];
  }
}

export { TIPO };
