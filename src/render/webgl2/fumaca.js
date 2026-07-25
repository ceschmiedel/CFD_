/* ── src/render/webgl2/fumaca.js ─────────────────────────────────────────────
 *
 * Fumaça volumétrica no caminho WebGL2: um campo de densidade advectado pelo
 * escoamento e renderizado por ray-marching com espalhamento.
 *
 * O par de src/render/fumaca.js. A física da camada é a mesma — traçador
 * passivo, sem massa, que não empurra o ar e não entra na conta do arrasto — e
 * o método também: advecção semi-lagrangiana com correção de MacCormack e
 * limitador, pente de filamentos contínuos, marcha secundária para a luz.
 *
 *
 * AS DUAS DIFERENÇAS
 * ------------------
 * TEXTURA 3D DE VERDADE, e é a boa notícia deste backend. O WebGL2 tem
 * `TEXTURE_3D` com `RGBA16F` filtrável em núcleo — sem extensão —, então a
 * amostragem trilinear que é o coração da advecção semi-lagrangiana e do
 * ray-march sai do hardware. Fosse preciso emular com um atlas 2D, seriam oito
 * `texelFetch` e sete interpolações por amostra, dentro de um laço de 96
 * passos por pixel.
 *
 * ESCRITA POR FATIA. O que não existe é escrever num volume de uma vez: um
 * fragment shader escreve numa superfície 2D. Cada passo de advecção percorre
 * as fatias em z, anexando uma camada por vez com `framebufferTextureLayer`.
 * São 2·nz desenhos por passo de advecção — 256 num preset médio —, e cada um
 * é minúsculo. O custo real não está aí; está nos 96 passos de ray-march por
 * pixel, que é o mesmo dos dois lados.
 */

import { CENA, PRECISAO, VS_COBERTURA, programa } from './comum.js';

const FUMO = `
uniform vec4 uRake;       // x, passoY, passoZ, raio do filamento
uniform vec4 uExtensao;   // y0, y1, z0, z1 do pente, em células
uniform vec4 uFparams;    // dt, dissipação, taxa, avanço deste passo (céls)
uniform vec4 uLuz;        // direção (xyz normalizada), intensidade
uniform int uCamada;      // fatia z sendo escrita
uniform sampler3D uDens;
uniform sampler3D uInter;

vec3 dim3() { return vec3(uDim); }

bool ehSolidoCel(ivec3 c) {
  ivec3 q = clamp(c, ivec3(0), uDim - ivec3(1));
  uint t = texelFetch(uTipoR, emAtlas(q), 0).r;
  return t == 1u || t == 4u || t == 7u;
}

bool dentroDoDominio(vec3 p) {
  return all(greaterThanEqual(p, vec3(0.5))) && all(lessThanEqual(p, dim3() - 0.5));
}

float amostraDens(sampler3D t, vec3 p) {
  return textureLod(t, p / dim3(), 0.0).r;
}

/*
 * O pente: filamentos numa grade (y, z) num plano a montante.
 *
 * FILAMENTO CONTÍNUO, SEM PULSO — um filete picado não mostra linha de
 * corrente, que é a única coisa que esta camada existe para mostrar. E a fatia
 * de emissão ACOMPANHA O PASSO: fatia fixa com passo grande deixa buraco entre
 * uma emissão e a seguinte, e o filamento sai como uma fileira de blocos.
 */
float rake(vec3 p) {
  float meia = max(1.5, uFparams.w * 0.6);
  float dx = abs(p.x - uRake.x);
  if (dx > meia) return 0.0;
  if (p.y < uExtensao.x || p.y > uExtensao.y) return 0.0;
  if (p.z < uExtensao.z || p.z > uExtensao.w) return 0.0;

  /* Fase medida a partir da BORDA do pente: é o que faz o número de filamentos
     ser exatamente o pedido, e não o que couber com a fase que calhar. */
  float fy = abs(fract((p.y - uExtensao.x) / uRake.y) - 0.5) * uRake.y;
  float fz = abs(fract((p.z - uExtensao.z) / uRake.z) - 0.5) * uRake.z;
  float r = length(vec2(fy, fz)) / max(uRake.w, 1e-3);
  return uFparams.z * exp(-r * r * 5.5) * (1.0 - 0.35 * dx / meia);
}`;

/* ─── passe 1: advecção semi-lagrangiana pura, para a intermediária ─── */

const FS_RECUAR = `${PRECISAO}
${CENA}
${FUMO}
out vec4 oDens;
void main() {
  ivec3 ci = ivec3(ivec2(gl_FragCoord.xy), uCamada);
  vec3 p = vec3(ci) + 0.5;
  if (ehSolidoCel(ci)) { oDens = vec4(0.0); return; }

  vec3 anterior = p - amostrar(p).xyz * uFparams.x;
  float d = amostraDens(uDens, anterior);
  /* Fora do domínio a fumaça simplesmente acaba — sem isto o clamp do
     amostrador copia a borda para dentro e o domínio se enche por trás. */
  if (!dentroDoDominio(anterior)) d = 0.0;
  oDens = vec4(d, 0.0, 0.0, 1.0);
}`;

/*
 * ─── passe 2: correção de MacCormack, com limitador ───
 *
 * A advecção semi-lagrangiana é uma interpolação trilinear por passo, e cada
 * interpolação borra: sem correção, os filamentos se dissolvem ANTES de
 * alcançar o corpo e o que sobra é uma névoa cinza a montante — precisamente
 * nada do que se quer ver. MacCormack estima o erro advectando de volta e
 * desconta metade dele.
 *
 * O LIMITADOR NÃO É OPCIONAL: a correção é uma extrapolação e pode passar do
 * ponto, criando densidade negativa e picos que não existiam, o que num campo
 * realimentado explode em poucos passos.
 */
const FS_CORRIGIR = `${PRECISAO}
${CENA}
${FUMO}
out vec4 oDens;
void main() {
  ivec3 ci = ivec3(ivec2(gl_FragCoord.xy), uCamada);
  vec3 p = vec3(ci) + 0.5;
  if (ehSolidoCel(ci)) { oDens = vec4(0.0); return; }

  vec3 u = amostrar(p).xyz;
  vec3 anterior = p - u * uFparams.x;

  float chapeu = texelFetch(uInter, ci, 0).r;          // resultado do passe 1
  float voltando = amostraDens(uInter, p + u * uFparams.x);
  float original = texelFetch(uDens, ci, 0).r;

  float d = chapeu + 0.5 * (original - voltando);

  /* Limitador: o intervalo dos oito cantos de onde a parcela veio. */
  vec3 b = floor(anterior - 0.5);
  float lo = 1e30, hi = -1e30;
  for (int k = 0; k < 8; k++) {
    vec3 o = vec3(float(k & 1), float((k >> 1) & 1), float((k >> 2) & 1));
    ivec3 q = clamp(ivec3(b + o), ivec3(0), uDim - ivec3(1));
    float v = texelFetch(uDens, q, 0).r;
    lo = min(lo, v); hi = max(hi, v);
  }
  d = clamp(d, lo, hi) * uFparams.y;

  if (!dentroDoDominio(anterior)) d = 0.0;
  d = max(d, rake(p));
  oDens = vec4(clamp(d, 0.0, 6.0), 0.0, 0.0, 1.0);
}`;

/* ─────────────────────────────────────────────────────────── o ray-march */

const FS_RENDER = `${PRECISAO}
${CENA}
${FUMO}
uniform sampler2D uProf;
in vec2 vNdc;
out vec4 oCor;

/*
 * Do NDC de volta para o mundo.
 *
 * A profundidade aqui está em [-1,1] — convenção do OpenGL — e o que vem da
 * textura de profundidade está em [0,1]. A conversão é o 2z-1 abaixo, e é a
 * razão de este caminho usar perspectivaGL e não a matriz do WebGPU: com a
 * matriz errada a reconstrução erra a distância e a fumaça atravessa o carro
 * por um motivo que não tem nada a ver com fumaça.
 *
 * (Estas linhas não têm crase de propósito: elas moram DENTRO de um template
 * literal de JavaScript, e uma crase num comentário de GLSL fecha a string.)
 */
vec3 mundoDe(vec2 ndc, float z) {
  vec4 h = uInvViewProj * vec4(ndc, z, 1.0);
  return h.xyz / h.w;
}

/* Henyey-Greenstein. g > 0 espalha para frente, que é o que fumaça faz — e é
   por isso que ela ACENDE quando está entre você e a luz. */
float hg(float cosTheta, float g) {
  float g2 = g * g;
  float d = 1.0 + g2 - 2.0 * g * cosTheta;
  return (1.0 - g2) / (12.566370614 * max(d * sqrt(d), 1e-4));
}

float densidadeEm(vec3 w) {
  vec3 p = paraLattice(w);
  vec3 uvw = p / dim3();
  if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) return 0.0;
  return textureLod(uDens, uvw, 0.0).r;
}

void main() {
  vec3 origem = uOlho;
  vec3 alvo = mundoDe(vNdc, 1.0);
  vec3 dir = normalize(alvo - origem);

  float k = uEscala.x * 2.0;
  vec3 lo = vec3(-float(uDim.x) * 0.5 * k, -float(uDim.y) * 0.5 * k, 0.0);
  vec3 hi = vec3(float(uDim.x) * 0.5 * k, float(uDim.y) * 0.5 * k, float(uDim.z) * k);
  vec2 t = raioCaixa(origem, dir, lo, hi);
  float t0 = max(t.x, 0.0);
  float t1 = t.y;
  if (t1 <= t0) discard;

  /* Para na geometria opaca já desenhada: sem isto a fumaça atravessa o carro
     e o piso, e a cena perde toda a noção de profundidade. */
  float zbuf = texelFetch(uProf, ivec2(gl_FragCoord.xy), 0).r;
  if (zbuf < 1.0) {
    vec3 wOpaco = mundoDe(vNdc, zbuf * 2.0 - 1.0);
    t1 = min(t1, length(wOpaco - origem));
  }
  if (t1 <= t0) discard;

  int nPassos = int(uFumo.z);
  int nLuz = int(uFumo.w);
  float ds = (t1 - t0) / float(nPassos);
  float sigma = uFumo.x;
  float g = uFumo.y;

  vec3 luzDir = normalize(uLuz.xyz);
  float fase = hg(dot(dir, luzDir), g);

  /* Desloca o início do passo por pixel: sem isso o volume vira uma pilha de
     cascas concêntricas — o artefato mais reconhecível de ray-march. */
  float jitter = hash12(gl_FragCoord.xy + uOpcoes.z * 0.618) * ds;

  float T = 1.0;
  vec3 L = vec3(0.0);
  float s = t0 + jitter;

  for (int i = 0; i < nPassos; i++) {
    if (T < 0.01) break;
    vec3 w = origem + dir * s;
    float d = densidadeEm(w);
    s += ds;
    if (d <= 0.001) continue;

    float st = d * sigma;
    float atenua = 1.0 - exp(-st * ds);

    /* Marcha secundária: quanta luz sobreviveu até aqui. Passos longos e
       poucos — a sombra da fumaça é suave por natureza. */
    float tau = 0.0;
    float dl = (hi.z - lo.z) / float(nLuz) * 0.9;
    for (int j = 1; j <= nLuz; j++) {
      tau += densidadeEm(w + luzDir * (float(j) * dl)) * sigma * dl;
    }
    float luz = exp(-tau);

    /* Um piso de espalhamento múltiplo: fumaça de verdade nunca é preta no
       lado escuro, e sem esse termo o volume ganha um núcleo morto. */
    float ambiente = 0.16 + 0.10 * exp(-tau * 0.35);

    L += T * atenua * (luz * fase * 3.6 + ambiente) * uLuz.w * vec3(0.96, 0.97, 1.0);
    T *= exp(-st * ds);
  }

  oCor = vec4(L, 1.0 - T);
}`;

export class VolumeFumacaGL {
  constructor(gl, solver) {
    this.gl = gl;
    this.solver = solver;
    this.nx = solver.nx; this.ny = solver.ny; this.nz = solver.nz;
    this.params = {
      dt: 17,
      saltoCelulas: 2.2,
      dissipacao: 0.999,
      taxa: 2.4,
      colunas: 7,
      linhas: 5,
      densidade: 70,
      g: 0.45,
      passos: 96,
      passosLuz: 5,
      luz: [-0.35, 0.55, 0.76],
      intensidade: 1.0,
    };
    this._acumulado = 0;
  }

  get bytes() { return this.nx * this.ny * this.nz * 8 * 3; }

  preparar() {
    const gl = this.gl;
    const tex = () => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_3D, t);
      /* LINEAR: a amostragem trilinear é o coração da advecção
         semi-lagrangiana e do ray-march, e RGBA16F é filtrável em núcleo no
         WebGL2 — é o que torna este caminho viável sem emulação. */
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
      gl.texStorage3D(gl.TEXTURE_3D, 1, gl.RGBA16F, this.nx, this.ny, this.nz);
      return t;
    };
    this.texA = tex(); this.texB = tex(); this.texC = tex();
    this.frente = 'A';

    this.fbo = gl.createFramebuffer();
    this.progRecuar = programa(gl, VS_COBERTURA, FS_RECUAR, 'fumaca-recuar');
    this.progCorrigir = programa(gl, VS_COBERTURA, FS_CORRIGIR, 'fumaca-corrigir');
    this.progRender = programa(gl, VS_COBERTURA, FS_RENDER, 'fumaca-render');
    this.vao = gl.createVertexArray();
    this.limpar();
  }

  get atual() { return this.frente === 'A' ? this.texA : this.texB; }

  /** Zera as três texturas, fatia por fatia. */
  limpar() {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, this.nx, this.ny);
    for (const t of [this.texA, this.texB, this.texC]) {
      for (let z = 0; z < this.nz; z++) {
        gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, t, 0, z);
        gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Posiciona o pente em relação ao corpo: perto o suficiente para os
   * filamentos chegarem organizados e largo o suficiente para cobrir o corpo
   * com folga.
   */
  posicionarRake(extentos) {
    const e = extentos;
    const largura = Math.max(e.tamanho[1], 4);
    const altura = Math.max(e.tamanho[2], 4);
    this.rake = {
      x: Math.max(3, e.min[0] - Math.max(10, e.tamanho[0] * 0.55)),
      y0: Math.max(1, e.centro[1] - largura * 1.15),
      y1: Math.min(this.ny - 2, e.centro[1] + largura * 1.15),
      z0: 1,
      z1: Math.min(this.nz - 2, altura * 2.3),
    };
    this._recalcularGrade();
  }

  definirGrade({ colunas, linhas } = {}) {
    if (colunas > 0) this.params.colunas = Math.round(colunas);
    if (linhas > 0) this.params.linhas = Math.round(linhas);
    this._recalcularGrade();
  }

  /** O raio sai do espaçamento: menos tubos saem mais grossos, que é o que um
   *  pente com menos saídas e a mesma vazão faz. */
  _recalcularGrade() {
    const r = this.rake;
    if (!r) return;
    r.passoY = (r.y1 - r.y0) / Math.max(this.params.colunas, 1);
    r.passoZ = (r.z1 - r.z0) / Math.max(this.params.linhas, 1);
    r.raio = Math.min(2.8, Math.max(0.7, 0.3 * Math.min(r.passoY, r.passoZ)));
  }

  _uniformesFumo({ u }, dtEfetivo) {
    const gl = this.gl;
    const r = this.rake ?? { x: 10, y0: 1, y1: this.ny - 2, z0: 1, z1: this.nz - 2,
      passoY: 8, passoZ: 8, raio: 1.1 };
    const p = this.params;
    gl.uniform4f(u.uRake, r.x, r.passoY, r.passoZ, r.raio);
    gl.uniform4f(u.uExtensao, r.y0, r.y1, r.z0, r.z1);
    gl.uniform4f(u.uFparams, dtEfetivo, p.dissipacao, p.taxa, dtEfetivo * 0.05);
    const l = p.luz, n = Math.hypot(...l) || 1;
    gl.uniform4f(u.uLuz, l[0] / n, l[1] / n, l[2] / n, p.intensidade);
  }

  /**
   * Avança a fumaça — mas não a cada quadro.
   *
   * A difusão numérica cobra POR PASSO, não por unidade de tempo simulado.
   * Acumular o deslocamento e disparar um passo GRANDE quando ele passa de um
   * par de células dá o mesmo movimento com três a quatro vezes menos
   * interpolações, e é o que faz o filamento sobreviver à travessia inteira.
   */
  avancar(dtVisual, ligarCena) {
    const gl = this.gl;
    if (dtVisual > 0) this.params.dt = dtVisual;
    this._acumulado += this.params.dt;
    if (this._acumulado * 0.05 < this.params.saltoCelulas) return;

    const dt = this._acumulado;
    this._acumulado = 0;

    const src = this.atual;
    const dst = this.frente === 'A' ? this.texB : this.texA;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, this.nx, this.ny);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(this.vao);

    /* Passe 1: src -> intermediária. Passe 2: lê a intermediária e o src, e
       escreve no destino. As três texturas são distintas de propósito: uma
       textura não pode ser alvo de escrita e fonte de amostragem no mesmo
       desenho, e o resultado disso não é erro — é lixo silencioso. */
    for (const [prog, entrada, inter, alvo] of [
      [this.progRecuar, src, src, this.texC],
      [this.progCorrigir, src, this.texC, dst],
    ]) {
      const un = ligarCena(prog);
      this._uniformesFumo(prog, dt);
      gl.activeTexture(gl.TEXTURE0 + un);
      gl.bindTexture(gl.TEXTURE_3D, entrada);
      gl.uniform1i(prog.u.uDens, un);
      gl.activeTexture(gl.TEXTURE0 + un + 1);
      gl.bindTexture(gl.TEXTURE_3D, inter);
      gl.uniform1i(prog.u.uInter, un + 1);

      for (let z = 0; z < this.nz; z++) {
        gl.uniform1i(prog.u.uCamada, z);
        gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, alvo, 0, z);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.frente = this.frente === 'A' ? 'B' : 'A';
  }

  /** Desenha o volume no framebuffer já ligado, lendo a profundidade opaca. */
  desenhar(profTex, ligarCena) {
    const gl = this.gl;
    const prog = this.progRender;
    const un = ligarCena(prog);
    this._uniformesFumo(prog, this.params.dt);

    gl.activeTexture(gl.TEXTURE0 + un);
    gl.bindTexture(gl.TEXTURE_3D, this.atual);
    gl.uniform1i(prog.u.uDens, un);
    gl.activeTexture(gl.TEXTURE0 + un + 1);
    gl.bindTexture(gl.TEXTURE_2D, profTex);
    gl.uniform1i(prog.u.uProf, un + 1);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  destruir() {
    const gl = this.gl;
    for (const t of [this.texA, this.texB, this.texC]) gl.deleteTexture(t);
    gl.deleteFramebuffer(this.fbo);
    gl.deleteVertexArray(this.vao);
  }
}
