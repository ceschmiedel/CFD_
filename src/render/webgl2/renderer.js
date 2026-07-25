/* ── src/render/webgl2/renderer.js ───────────────────────────────────────────
 *
 * A visualização no caminho WebGL2.
 *
 * Vale aqui a mesma regra do renderizador WGSL, e ela é a razão de este arquivo
 * existir em vez de um fallback simplificado: TUDO O QUE APARECE NA TELA É O
 * CAMPO RESOLVIDO. As partículas leem a velocidade das células em que estão, o
 * Cp na carroceria lê a densidade da célula adjacente. Um caminho alternativo
 * que desenhasse enfeite analítico seria pior que não ter caminho nenhum: o
 * visitante sem WebGPU veria uma imagem que não depende do solver e não teria
 * como saber disso.
 *
 *
 * O QUE MUDA SEM COMPUTE
 * ----------------------
 * As partículas viviam num storage buffer atualizado por um kernel de compute.
 * Aqui elas vivem em TEXTURAS — uma para posição e vida, outra para a posição
 * anterior — atualizadas por um fragment shader com dois alvos, um texel por
 * partícula. O desenho lê essas texturas no VERTEX shader (`texelFetch` por
 * `gl_InstanceID`), que é o que o WebGL2 tem no lugar do vertex pulling de
 * storage buffer.
 *
 * A outra mudança é a profundidade. O ray-march da fumaça precisa LER a
 * profundidade da geometria opaca para não atravessá-la, e não dá para
 * amostrar o anexo de profundidade do framebuffer em que se está desenhando.
 * A cena opaca vai para um alvo fora da tela com cor e profundidade em textura;
 * a fumaça desenha num segundo framebuffer que compartilha a MESMA textura de
 * cor mas não tem a de profundidade anexada, e por isso pode lê-la; no fim, a
 * cor é copiada para o canvas.
 */

import { Orbita, inversa } from '../mat4.js';
import { CENA, TURBO, PRECISAO, VS_COBERTURA, programa, textura2D } from './comum.js';
import { VolumeFumacaGL } from './fumaca.js';

/* ─────────────────────────────────────────────────────────────── partículas */

/*
 * O passo de advecção, comum às duas camadas de partícula.
 *
 * `NASCER` é injetado por quem usa: as esteiras renascem espalhadas pelo
 * domínio inteiro, as rasantes renascem enviesadas para o chão. O resto — o
 * Euler subdividido, a detecção de partícula presa dentro do corpo, a vida — é
 * igual nas duas.
 */
const FS_ADVECCAO = (NASCER) => `${PRECISAO}
${CENA}
uniform sampler2D uPos;
uniform sampler2D uAnt;
uniform vec4 uPart;        // altura da faixa, _, _, _
layout(location = 0) out vec4 oPos;
layout(location = 1) out vec4 oAnt;

float hash(uint n) {
  uint x = n * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return float((x >> 22u) ^ x) / 4294967296.0;
}

${NASCER}

void main() {
  ivec2 t = ivec2(gl_FragCoord.xy);
  uint id = uint(t.y) * uint(textureSize(uPos, 0).x) + uint(t.x);
  vec4 p = texelFetch(uPos, t, 0);
  vec4 m = amostrar(p.xyz);

  /* Euler subdividido: mesmo comprimento de caminho, trajetória muito mais
     fiel em volta dos cantos — que é onde o olho procura a separação. A
     contagem vem do relógio, e cresce com a velocidade do túnel. */
  int nSub = max(int(uRelogio.y), 1);
  float dt = uRelogio.x / float(nSub);
  vec3 np = p.xyz;
  for (int k = 0; k < nSub; k++) { np += amostrar(np).xyz * dt; }

  p.w -= 1.0;
  vec3 n = vec3(uDim);
  bool fora = np.x >= n.x - 1.0 || np.x < 1.0 || np.y < 1.0 || np.y >= n.y - 1.0
           || np.z < 0.4 || np.z >= n.z - 1.0;
  /* Campo nulo = entrou no corpo. Ficaria parada para sempre e viraria um
     ponto morto brilhando na carroceria; renascer é mais barato que evitar. */
  bool presa = length(m.xyz) < 1e-5;

  if (fora || presa || p.w <= 0.0) {
    vec4 nova = nascer(id);
    oPos = nova;
    oAnt = vec4(nova.xyz, 0.0);      // sem rastro no quadro do renascimento
  } else {
    oPos = vec4(np, p.w);
    oAnt = vec4(p.xyz, 1.0);
  }
}`;

const NASCER_ESTEIRAS = `
vec4 nascer(uint i) {
  uint s = i * 3u + uint(uOpcoes.z) * 7919u;
  vec3 n = vec3(uDim);
  return vec4(2.0 + hash(s) * 3.0,
              2.0 + hash(s + 1u) * (n.y - 4.0),
              0.5 + hash(s + 2u) * (n.z - 2.0),
              220.0 + hash(s + 5u) * 260.0);
}`;

/* Enviesada para o chão: o expoente cúbico põe a maioria nas primeiras células
   da faixa e ainda deixa algumas mais alto — uniforme dentro da faixa daria uma
   borda superior reta, que o olho lê como uma parede de vidro flutuando. */
const NASCER_RASANTES = `
vec4 nascer(uint i) {
  uint s = i * 3u + uint(uOpcoes.z) * 7919u;
  vec3 n = vec3(uDim);
  float h = hash(s + 2u);
  return vec4(1.5 + hash(s) * 4.0,
              1.0 + hash(s + 1u) * (n.y - 2.0),
              0.6 + h * h * h * uPart.x,
              150.0 + hash(s + 5u) * 320.0);
}`;

/** Lê o estado de uma partícula pelo índice da instância. */
const LER_PARTICULA = `
uniform sampler2D uPos;
uniform sampler2D uAnt;
void lerParticula(int id, out vec4 pos, out vec4 ant) {
  int w = textureSize(uPos, 0).x;
  ivec2 t = ivec2(id % w, id / w);
  pos = texelFetch(uPos, t, 0);
  ant = texelFetch(uAnt, t, 0);
}`;

/* ──────────────────────────────────────────────────────────────── esteiras */

const VS_ESTEIRAS = `${PRECISAO}
${CENA}
${TURBO}
${LER_PARTICULA}
out vec4 vCor;
void main() {
  vec4 pos, ant;
  lerParticula(gl_InstanceID, pos, ant);
  /* dois vértices: a cauda é a posição anterior, a cabeça é a atual */
  vec3 a = gl_VertexID == 0 ? ant.xyz : pos.xyz;
  gl_Position = uViewProj * vec4(paraMundo(a), 1.0);
  float v = length(amostrar(pos.xyz).xyz) / max(uEscala.w, 1e-6);
  float vida = clamp(pos.w / 90.0, 0.0, 1.0);
  vCor = vec4(turbo(v * 0.55), ant.w * vida * 0.85);
}`;

const FS_COR = `${PRECISAO}
in vec4 vCor;
out vec4 oCor;
void main() { oCor = vCor; }`;

/* ──────────────────────────────────────────────────────────────── rasantes */

const VS_RASANTES = `${PRECISAO}
${CENA}
${LER_PARTICULA}
uniform vec4 uRasa;   // alongamento, altura da faixa, meia-largura (px), intensidade
uniform vec4 uRasaB;  // mundo por pixel, traço máximo (céls), distância da câmera, _
out vec2 vUv;
out vec4 vCor;

void main() {
  vec4 pos, ant;
  lerParticula(gl_InstanceID, pos, ant);

  /* O traço é o caminho REAL do passo, esticado ao longo dele mesmo — estica
     comprimento, nunca direção. Com teto: a 90 m/s o traço alongado passaria de
     meio carro e deixaria de ser borrão de movimento. */
  vec3 caminho = ant.xyz - pos.xyz;
  float lc = length(caminho);
  vec3 esticado = caminho * min(uRasa.x, uRasaB.y / max(lc, 1e-6));
  vec3 cabeca = paraMundo(pos.xyz);
  vec3 cauda = paraMundo(pos.xyz + esticado);

  vec2 q[6] = vec2[6](vec2(0.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
                      vec2(0.0, -1.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
  vec2 t = q[gl_VertexID];
  vec3 base = mix(cauda, cabeca, t.x);

  vec3 eixo = cabeca - cauda;
  float comp = length(eixo);
  vec3 dir = comp > 1e-9 ? eixo / comp : vec3(1.0, 0.0, 0.0);

  vec3 paraOlho = normalize(uOlho - base);
  vec3 lat = cross(dir, paraOlho);
  if (length(lat) < 1e-4) { lat = cross(dir, vec3(0.0, 0.0, 1.0)); }
  if (length(lat) < 1e-4) { lat = vec3(0.0, 1.0, 0.0); }
  lat = normalize(lat);

  /* Largura em PIXEL e não no mundo: espessura de mundo engorda com o zoom e
     transforma o risco numa gota alongada. */
  float dist = length(uOlho - base);
  float esp = dist * uRasaB.x * uRasa.z;

  gl_Position = uViewProj * vec4(base + lat * (esp * t.y), 1.0);
  vUv = t;

  float v = length(amostrar(pos.xyz).xyz) / max(uEscala.w, 1e-6);
  float alto = exp(-pos.z / max(uRasa.y * 0.4, 1.0));
  float vida = clamp(pos.w / 60.0, 0.0, 1.0);
  float entrada = smoothstep(0.0, 10.0, pos.x);
  float longe = length(uOlho - cabeca) / max(uRasaB.z, 1e-3);
  float bruma = 1.0 / (1.0 + pow(max(longe - 1.0, 0.0) * 1.6, 2.0));
  float a = 0.6 * uRasa.w * clamp(v * 0.9, 0.0, 1.6) * alto * vida * entrada
          * bruma * ant.w;
  vec3 cor = mix(vec3(0.50, 0.68, 1.0), vec3(0.92, 0.96, 1.0),
                 clamp(v * 0.45, 0.0, 1.0));
  vCor = vec4(cor, a);
}`;

const FS_RASANTES = `${PRECISAO}
in vec2 vUv;
in vec4 vCor;
out vec4 oCor;
void main() {
  /* Núcleo cheio com borda de antialias, e não gaussiana: a queda suave ao
     longo de toda a largura é o que dá aspecto de borrão arredondado. */
  float perfil = 1.0 - smoothstep(0.35, 1.0, abs(vUv.y));
  float cauda = smoothstep(0.0, 0.22, vUv.x);
  float a = vCor.a * perfil * cauda;
  oCor = vec4(vCor.rgb * a, a);      // aditivo, pré-multiplicado
}`;

/* ─────────────────────────────────────────────────────── carroceria com Cp */

const VS_CORPO = `${PRECISAO}
${CENA}
layout(location = 0) in vec3 aPos;   // posição em coordenadas de lattice
out vec3 vMundo;
out vec3 vLattice;
void main() {
  vLattice = aPos;
  vMundo = paraMundo(aPos);
  gl_Position = uViewProj * vec4(vMundo, 1.0);
}`;

const FS_CORPO = `${PRECISAO}
${CENA}
${TURBO}
in vec3 vMundo;
in vec3 vLattice;
out vec4 oCor;

bool ehSolido(vec3 p) {
  uint t = tipoLattice(p);
  return t == 1u || t == 4u || t == 7u;
}

void main() {
  /* A normal sai das derivadas de tela: dá a normal da FACE, que é a certa
     para malha importada onde vértices são compartilhados entre painéis que
     não deviam ser suavizados juntos. Ela aponta para a câmera, não
     necessariamente para FORA — para iluminar tanto faz, para amostrar a
     pressão não: metade das vezes o ponto cairia dentro do corpo. Tentamos os
     dois lados e ficamos com o que está no fluido. */
  vec3 n = normalize(cross(dFdx(vMundo), dFdy(vMundo)));
  vec3 fora = vLattice + n * 1.8;
  if (ehSolido(fora)) { fora = vLattice - n * 1.8; }
  float delta = amostrar(fora).w;

  /* Cp = (p - p_inf)/(1/2 rho U^2); no lattice p = c_s^2 delta com c_s^2 = 1/3
     e rho = 1, então Cp = delta / (1.5 u_lb^2). */
  float cp = delta / max(1.5 * uEscala.w * uEscala.w, 1e-12);

  vec3 luz = normalize(vec3(0.4, 0.7, 0.9));
  float dif = 0.35 + 0.65 * max(dot(n, luz), 0.0);
  float esp = pow(max(dot(reflect(-luz, n), normalize(vec3(0.0, 0.0, 1.0))), 0.0), 24.0);

  vec3 base = vec3(0.72, 0.74, 0.78);
  if (uOpcoes.x > 0.5) {
    base = turbo(clamp(1.0 - (cp * uOpcoes.y + 2.0) / 3.0, 0.0, 1.0));
  }
  oCor = vec4(base * dif + vec3(esp * 0.25), 1.0);
}`;

/* ──────────────────────────────────────────────────────────────────── piso */

const VS_PISO = `${PRECISAO}
${CENA}
out vec2 vUv;
void main() {
  vec2 q[6] = vec2[6](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
                      vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
  vec2 t = q[gl_VertexID];
  vUv = t;
  vec3 p = vec3(t.x * float(uDim.x), t.y * float(uDim.y), 0.0);
  gl_Position = uViewProj * vec4(paraMundo(p), 1.0);
}`;

const FS_PISO = `${PRECISAO}
${CENA}
in vec2 vUv;
out vec4 oCor;
void main() {
  vec2 g = vec2(uDim.xy) / 16.0;
  /* A grade rola com o cinto. O deslocamento chega em células e a grade se
     repete a cada 16 delas, então o renderizador o mantém dentro de um
     período — em float32 um acumulado de milhares de células perderia a fração
     e a grade começaria a andar aos trancos. */
  vec2 uv = vec2(vUv.x - uOpcoes.w / float(uDim.x), vUv.y);
  vec2 l = abs(fract(uv * g) - 0.5) / fwidth(uv * g);
  float linha = 1.0 - min(min(l.x, l.y), 1.0);
  oCor = vec4(vec3(0.16, 0.19, 0.24) + linha * 0.10, 0.55 + linha * 0.25);
}`;

/* ────────────────────────────────────────────────────────────── composição */

const FS_COMPOR = `${PRECISAO}
in vec2 vNdc;
uniform sampler2D uCena;
out vec4 oCor;
void main() {
  oCor = texelFetch(uCena, ivec2(gl_FragCoord.xy), 0);
}`;

/* ──────────────────────────────────────────────────────────────────  classe */

export class RendererWebGL2 {
  constructor(gl, canvas) {
    this.gl = gl;
    this.canvas = canvas;
    this.camera = new Orbita({ alvo: [0, 0, 0.35], distancia: 2.6 });
    this.opcoes = {
      cp: true, ganhoCp: 1.0, esteiras: false, corpo: true, fumaca: true,
      rasantes: true, uCinto: 0, passosPorSegundo: 0,
    };
    this.quadro = 0;
    this.deslocPiso = 0;
    this.nParticulas = 0;
    this._ligarInteracao();
  }

  /**
   * Órbita com um dedo, zoom com dois — o mesmo gesto do caminho WebGPU.
   *
   * Rastrear cada pointerId separadamente e não só o último: com dois dedos na
   * tela, o `pointermove` do segundo compararia a posição dele com a do
   * primeiro e a cena saltaria a cada toque.
   */
  _ligarInteracao() {
    const c = this.canvas;
    const ativos = new Map();
    const sep = () => {
      const [a, b] = [...ativos.values()];
      return Math.hypot(a.x - b.x, a.y - b.y);
    };
    let dist0 = 0;

    c.addEventListener('pointerdown', e => {
      ativos.set(e.pointerId, { x: e.clientX, y: e.clientY });
      c.setPointerCapture(e.pointerId);
      if (ativos.size === 2) dist0 = sep();
    });
    const soltar = e => {
      ativos.delete(e.pointerId);
      if (ativos.size === 2) dist0 = sep();
    };
    c.addEventListener('pointerup', soltar);
    c.addEventListener('pointercancel', soltar);

    c.addEventListener('pointermove', e => {
      const p = ativos.get(e.pointerId);
      if (!p) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      if (ativos.size === 1) {
        this.camera.girar(dx * 0.008, dy * 0.008);
      } else if (ativos.size === 2 && dist0 > 0) {
        const d = sep();
        if (d > 0) { this.camera.aproximar(dist0 / d); dist0 = d; }
      }
    });

    c.addEventListener('wheel', e => {
      e.preventDefault();
      this.camera.aproximar(Math.exp(e.deltaY * 0.0011));
    }, { passive: false });
  }

  /* ───────────────────────────────────────────────────────────── montagem */

  async preparar(solver, { nParticulas = 60000 } = {}) {
    const gl = this.gl;
    this.solver = solver;
    this.nParticulas = nParticulas;
    this.nRasantes = solver.nx >= 280 ? 2000 : 1200;

    this.vao ??= gl.createVertexArray();

    if (!this.progs) {
      this.progs = {
        piso: programa(gl, VS_PISO, FS_PISO, 'piso'),
        corpo: programa(gl, VS_CORPO, FS_CORPO, 'corpo'),
        esteiras: programa(gl, VS_ESTEIRAS, FS_COR, 'esteiras'),
        rasantes: programa(gl, VS_RASANTES, FS_RASANTES, 'rasantes'),
        advEsteiras: programa(gl, VS_COBERTURA, FS_ADVECCAO(NASCER_ESTEIRAS), 'adv-esteiras'),
        advRasantes: programa(gl, VS_COBERTURA, FS_ADVECCAO(NASCER_RASANTES), 'adv-rasantes'),
        compor: programa(gl, VS_COBERTURA, FS_COMPOR, 'compor'),
      };
    }

    this.esteiras?.destruir?.();
    this.rasantes?.destruir?.();
    this.esteiras = this._criarParticulas(nParticulas, (i) => {
      /* Semeadas com vida negativa: o primeiro passo de advecção as renasce em
         posições válidas, e não há um quadro inicial com tudo na origem. */
      return [0, 0, 0, -1];
    });
    this.rasantes = this._criarParticulas(this.nRasantes, null);
    this.rasantes.params = {
      alongamento: 5.0, tracoMaximo: 16,
      altura: Math.max(6, solver.nz * 0.22),
      larguraPixels: 1.1, intensidade: 1.0,
    };
    this._semearRasantes();

    this.fumaca?.destruir();
    this.fumaca = new VolumeFumacaGL(gl, solver);
    this.fumaca.preparar();

    this.alvos = null;   // força recriação no próximo desenho
  }

  /** Duas texturas de estado com ping-pong e um framebuffer por destino. */
  _criarParticulas(n, semente) {
    const gl = this.gl;
    const w = Math.min(256, n);
    const h = Math.ceil(n / w);
    const tex = () => textura2D(gl, w, h, gl.RGBA32F);
    const est = { n, w, h, pos: [tex(), tex()], ant: [tex(), tex()], frente: 0 };

    est.fbo = [0, 1].map(k => {
      const f = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, est.pos[k], 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, est.ant[k], 0);
      return f;
    });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    if (semente) {
      const dados = new Float32Array(w * h * 4);
      for (let i = 0; i < n; i++) dados.set(semente(i), i * 4);
      for (const t of est.pos) {
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, gl.FLOAT, dados);
      }
    }
    est.destruir = () => {
      for (const t of [...est.pos, ...est.ant]) gl.deleteTexture(t);
      for (const f of est.fbo) gl.deleteFramebuffer(f);
    };
    return est;
  }

  /**
   * Semeia as rasantes espalhadas em x.
   *
   * O shader as renasce na entrada, o que em regime dá uma faixa uniforme — mas
   * só depois de uma travessia inteira, e os primeiros segundos ficariam com
   * uma frente de onda subindo a tela.
   */
  _semearRasantes() {
    const gl = this.gl;
    const r = this.rasantes;
    const { nx, ny } = this.solver;
    const h = r.params.altura;
    const dados = new Float32Array(r.w * r.h * 4);
    for (let i = 0; i < r.n; i++) {
      dados[i * 4] = 1.5 + Math.random() * (nx - 3);
      dados[i * 4 + 1] = 1 + Math.random() * (ny - 2);
      dados[i * 4 + 2] = 0.6 + Math.pow(Math.random(), 3) * h;
      dados[i * 4 + 3] = 150 + Math.random() * 320;
    }
    for (const t of r.pos) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, r.w, r.h, gl.RGBA, gl.FLOAT, dados);
    }
  }

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
    if (this.rasantes) {
      const alturaCorpo = Math.max(extentos.tamanho[2], 4);
      this.rasantes.params.altura = Math.min(solver.nz * 0.5, alturaCorpo * 1.5);
      this._semearRasantes();
    }
  }

  /** Sobe a malha do corpo (posições no espaço do lattice). */
  definirMalha(positions, indices) {
    const gl = this.gl;
    if (this.vbo) gl.deleteBuffer(this.vbo);
    if (this.ibo) gl.deleteBuffer(this.ibo);
    this.vbo = null; this.nIndices = 0;
    if (!positions || !indices?.length) return;

    this.vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    /* Uint32 sempre: uma malha de F1 passa dos 65 536 vértices com folga, e
       cair para Uint16 embaralharia os triângulos em silêncio. */
    const ind = indices instanceof Uint32Array ? indices : new Uint32Array(indices);
    this.ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, ind, gl.STATIC_DRAW);
    this.nIndices = ind.length;

    this.vaoCorpo ??= gl.createVertexArray();
    gl.bindVertexArray(this.vaoCorpo);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    gl.bindVertexArray(null);
  }

  /* ──────────────────────────────────────────────────────────── alvos */

  _redimensionar() {
    const gl = this.gl;
    const compacto = Math.min(window.innerWidth, window.innerHeight) < 760;
    const dpr = Math.min(window.devicePixelRatio || 1, compacto ? 1.75 : 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width === w && this.canvas.height === h && this.alvos) return;
    this.canvas.width = w; this.canvas.height = h;

    if (this.alvos) {
      gl.deleteTexture(this.alvos.cor);
      gl.deleteTexture(this.alvos.prof);
      gl.deleteFramebuffer(this.alvos.fboCena);
      gl.deleteFramebuffer(this.alvos.fboCor);
    }
    const cor = textura2D(gl, w, h, gl.RGBA8);
    const prof = textura2D(gl, w, h, gl.DEPTH_COMPONENT24);

    /* DOIS framebuffers sobre a MESMA cor. O primeiro tem profundidade e
       recebe a cena opaca; o segundo NÃO tem, e é nele que a fumaça desenha —
       amostrar o anexo de profundidade do framebuffer em que se está
       desenhando é laço de realimentação, e o resultado é indefinido. */
    const fboCena = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboCena);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, cor, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, prof, 0);
    const fboCor = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboCor);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, cor, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    this.alvos = { cor, prof, fboCena, fboCor, w, h };
  }

  /** Liga os uniformes da cena num programa. */
  _ligarCena({ p, u }, unidadeLivre = 0) {
    const gl = this.gl;
    const s = this.solver;
    gl.useProgram(p);
    gl.uniformMatrix4fv(u.uViewProj, false, this._vp);
    gl.uniformMatrix4fv(u.uInvViewProj, false, this._invVp);
    gl.uniform3i(u.uDim, s.nx, s.ny, s.nz);
    gl.uniform2i(u.uTiles, s.atlas.tx, s.atlas.ty);
    gl.uniform4f(u.uEscala, 1 / s.nx, 1 / s.ny, 1 / s.nz, this._uRef);
    gl.uniform4f(u.uOpcoes, this.opcoes.cp ? 1 : 0, this.opcoes.ganhoCp,
      this.quadro % 4096, this.deslocPiso);
    gl.uniform3f(u.uOlho, ...this.camera.olho);
    const f = this.fumaca;
    gl.uniform4f(u.uFumo, f?.params.densidade ?? 0, f?.params.g ?? 0,
      f?.params.passos ?? 1, f?.params.passosLuz ?? 1);
    gl.uniform4f(u.uRelogio, this._dtVisual, this._subpassos, 0, 0);

    gl.activeTexture(gl.TEXTURE0 + unidadeLivre);
    gl.bindTexture(gl.TEXTURE_2D, s.macrosTex);
    gl.uniform1i(u.uMacros, unidadeLivre);
    gl.activeTexture(gl.TEXTURE0 + unidadeLivre + 1);
    gl.bindTexture(gl.TEXTURE_2D, s.tipoTex);
    gl.uniform1i(u.uTipoR, unidadeLivre + 1);
    return unidadeLivre + 2;
  }

  _ligarParticulas({ u }, est, unidade) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unidade);
    gl.bindTexture(gl.TEXTURE_2D, est.pos[est.frente]);
    gl.uniform1i(u.uPos, unidade);
    gl.activeTexture(gl.TEXTURE0 + unidade + 1);
    gl.bindTexture(gl.TEXTURE_2D, est.ant[est.frente]);
    gl.uniform1i(u.uAnt, unidade + 1);
  }

  /** Um passo de advecção de um conjunto de partículas. */
  _avancarParticulas(prog, est, extra) {
    const gl = this.gl;
    const destino = 1 - est.frente;
    const un = this._ligarCena(prog);
    this._ligarParticulas(prog, est, un);
    gl.uniform4f(prog.u.uPart, extra ?? 0, 0, 0, 0);

    gl.bindFramebuffer(gl.FRAMEBUFFER, est.fbo[destino]);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, est.w, est.h);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    est.frente = destino;
  }

  /* ─────────────────────────────────────────────────────────────── desenho */

  desenhar({ uRef }) {
    if (!this.solver) return;
    const gl = this.gl;
    this._redimensionar();
    this.quadro++;
    this._uRef = uRef;

    /* matrizGL e não matriz: profundidade em [-1,1]. Ver perspectivaGL em
       mat4.js — com a convenção errada a cena aparece certa e só a
       reconstrução do mundo a partir da profundidade erra, que é como a fumaça
       passa a atravessar o carro. */
    const aspecto = this.canvas.width / this.canvas.height;
    const vp = this.camera.matrizGL(aspecto);
    this._vp = vp;
    this._invVp = inversa(vp);

    /* O relógio comum: uma taxa só para todas as camadas de movimento, e ela é
       uma fração do tempo real — é o que faz o controle de velocidade ser
       sentido. Ver o cabeçalho do renderizador WGSL. */
    const agora = performance.now();
    const dtParede = Math.min(
      Math.max((agora - (this._ultimoQuadro ?? agora)) / 1000, 1 / 240), 1 / 24);
    this._ultimoQuadro = agora;
    this._dtVisual = (this.opcoes.passosPorSegundo ?? 0) * dtParede;
    const celulas = this._dtVisual * uRef;
    this._subpassos = Math.max(1, Math.min(8, Math.ceil(celulas / 1.5)));
    this.deslocPiso = (this.deslocPiso + (this.opcoes.uCinto ?? 0) * this._dtVisual) % 16;

    /* ─── passos de simulação das camadas ─── */
    if (this.opcoes.esteiras) {
      this._avancarParticulas(this.progs.advEsteiras, this.esteiras);
    }
    if (this.opcoes.rasantes) {
      this._avancarParticulas(this.progs.advRasantes, this.rasantes,
        this.rasantes.params.altura);
    }
    if (this.opcoes.fumaca && this.fumaca) {
      this.fumaca.avancar(this._dtVisual, (prog) => this._ligarCena(prog));
    }

    /* ─── cena opaca, com profundidade ─── */
    const A = this.alvos;
    gl.bindFramebuffer(gl.FRAMEBUFFER, A.fboCena);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, A.w, A.h);
    gl.clearColor(0.043, 0.051, 0.067, 1);
    gl.clearDepth(1);
    /*
     * depthMask(true) ANTES do clear, e não é detalhe: `glClear` respeita a
     * máscara de profundidade. Com a máscara desligada — que é como o quadro
     * anterior termina, porque a fumaça e os traços não escrevem profundidade —
     * o clear de profundidade não faz NADA.
     *
     * O sintoma foi um dos mais desorientadores deste projeto: no primeiro
     * quadro a carroceria aparecia; do segundo em diante ela sumia e deixava
     * um buraco com a forma dela no piso. A profundidade guardava o carro do
     * quadro passado, o teste LESS rejeitava o carro novo por empate exato, e
     * rejeitava o piso atrás dele — sobrava a cor de fundo recortada em forma
     * de carro. Tudo o mais estava certo: geometria, VAO, uniformes, programa.
     */
    gl.depthMask(true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.disable(gl.CULL_FACE);
    gl.bindVertexArray(this.vao);

    /* piso: testa profundidade, não escreve */
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    this._ligarCena(this.progs.piso);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    if (this.opcoes.corpo && this.vbo && this.nIndices) {
      gl.depthMask(true);
      gl.disable(gl.BLEND);
      this._ligarCena(this.progs.corpo);
      gl.bindVertexArray(this.vaoCorpo);
      gl.drawElements(gl.TRIANGLES, this.nIndices, gl.UNSIGNED_INT, 0);
      gl.bindVertexArray(this.vao);
    }

    /* Os traços testam profundidade e não a escrevem: um traço de um pixel que
       ocultasse a carroceria deixaria a superfície salpicada. */
    gl.depthMask(false);
    if (this.opcoes.esteiras) {
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      const un = this._ligarCena(this.progs.esteiras);
      this._ligarParticulas(this.progs.esteiras, this.esteiras, un);
      gl.drawArraysInstanced(gl.LINES, 0, 2, this.esteiras.n);
    }
    if (this.opcoes.rasantes) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);            // aditivo
      const p = this.progs.rasantes;
      const un = this._ligarCena(p);
      this._ligarParticulas(p, this.rasantes, un);
      const r = this.rasantes.params;
      gl.uniform4f(p.u.uRasa, r.alongamento, r.altura, r.larguraPixels, r.intensidade);
      /* Tamanho de um pixel em unidades de mundo a uma unidade da câmera. O fov
         vem da câmera porque em retrato ele muda com o formato da janela. */
      const mundoPorPixel = 2 * Math.tan(this.camera.fov(aspecto) / 2) / this.canvas.height;
      gl.uniform4f(p.u.uRasaB, mundoPorPixel, r.tracoMaximo, this.camera.distancia, 0);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.rasantes.n);
    }

    /* ─── fumaça: lê a profundidade que a cena opaca acabou de escrever ─── */
    if (this.opcoes.fumaca && this.fumaca) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, A.fboCor);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
      gl.viewport(0, 0, A.w, A.h);
      gl.disable(gl.DEPTH_TEST);
      gl.depthMask(false);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);   // cor pré-multiplicada
      this.fumaca.desenhar(A.prof, (prog) => this._ligarCena(prog));
    }

    /* ─── para a tela ─── */
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(this.progs.compor.p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, A.cor);
    gl.uniform1i(this.progs.compor.u.uCena, 0);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
