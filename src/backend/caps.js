/* ── src/backend/caps.js ─────────────────────────────────────────────────────
 *
 * O que esta máquina consegue rodar, e quanto.
 *
 * Este app oferece mais de um caminho de execução e deixa o usuário escolher.
 * Este módulo é quem levanta os fatos sobre os quais essa escolha é feita —
 * e ele é deliberadamente pessimista, porque a alternativa é o usuário
 * escolher um preset que trava a aba dele.
 *
 *
 * OS BACKENDS
 * -----------
 * webgpu   Compute shaders, storage buffers, layout struct-of-arrays direto.
 *          As dezenove distribuições ficam em buffers lineares, o streaming é
 *          uma leitura indexada, e nada precisa ser empacotado. É o caminho
 *          rápido e o teto alto.
 *
 * webgl2   Sem compute. O lattice 3D vira um atlas 2D (tiles em Z) e as
 *          dezenove distribuições são empacotadas em cinco texturas RGBA32F
 *          escritas por MRT num fragment shader, com ping-pong de framebuffer.
 *          Custa banda extra e um shader bem mais contorcido, mas roda em
 *          praticamente qualquer GPU desta década.
 *
 * Ambos executam A MESMA física: o emissor em core/emit/ir.js gera WGSL para
 * um e GLSL ES 3.00 para o outro a partir da mesma descrição. Se os dois
 * discordarem num caso de validação, é bug de um backend, não de modelagem —
 * e a suíte de validação roda nos dois exatamente para pegar isso.
 *
 *
 * SOBRE OS TETOS DE MEMÓRIA
 * -------------------------
 * Nem WebGPU nem WebGL2 expõem quanta VRAM existe ou quanta está livre. Os
 * números abaixo saem dos limites declarados pelo adaptador, que são um limite
 * SUPERIOR e não uma promessa: um adaptador que anuncia buffers de 2 GiB pode
 * perfeitamente falhar ao alocar o segundo. Por isso o app tenta alocar de
 * verdade antes de aceitar um preset, e cai para o preset abaixo se a alocação
 * falhar. Um teto declarado aqui é uma hipótese, não um contrato.
 */

import { Q } from '../core/lattice.js';

/* fp32 por população, dois conjuntos para o ping-pong do streaming. */
export const BYTES_POR_CELULA = Q * 4 * 2;

/*
 * Presets de qualidade. O domínio de um túnel é comprido em x, e as proporções
 * abaixo dão ~8 comprimentos de corpo à frente e atrás, que é o mínimo para a
 * esteira não bater na saída e voltar.
 */
export const PRESETS = [
  { id: 'minima', nome: 'Mínima', nx: 128, ny: 64, nz: 64 },
  { id: 'baixa', nome: 'Baixa', nx: 192, ny: 96, nz: 96 },
  { id: 'media', nome: 'Média', nx: 256, ny: 128, nz: 128 },
  { id: 'alta', nome: 'Alta', nx: 384, ny: 192, nz: 192 },
  { id: 'extrema', nome: 'Extrema', nx: 512, ny: 256, nz: 256 },
].map(p => ({
  ...p,
  cells: p.nx * p.ny * p.nz,
  bytes: p.nx * p.ny * p.nz * BYTES_POR_CELULA,
}));

/**
 * Escolhe o ladrilhamento em Z que deixa o atlas 2D o mais quadrado possível.
 * O driver aloca o retângulo envolvente de qualquer jeito, então perder para
 * padding é pior que perder para forma.
 *
 * Usado só pelo backend WebGL2 — o WebGPU indexa buffers lineares e não
 * precisa de atlas nenhum.
 */
export function atlasLayout(nx, ny, nz, maxTexture = 16384) {
  let best = null;
  for (let tx = 1; tx <= nz; tx++) {
    const ty = Math.ceil(nz / tx);
    const w = tx * nx, h = ty * ny;
    if (w > maxTexture || h > maxTexture) continue;
    /* preferir quadrado, depois preferir menos padding */
    const score = Math.max(w, h) / Math.min(w, h) + 0.001 * (tx * ty - nz);
    if (!best || score < best.score) best = { tx, ty, w, h, score };
  }
  return best;
}

/* ────────────────────────────────────────────────────────────────── WebGPU */

async function probeWebGPU() {
  if (!globalThis.navigator?.gpu) {
    return { id: 'webgpu', disponivel: false, motivo: 'navigator.gpu ausente — este navegador não expõe WebGPU.' };
  }

  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e) {
    return { id: 'webgpu', disponivel: false, motivo: `requestAdapter lançou: ${e.message}` };
  }
  if (!adapter) {
    return {
      id: 'webgpu', disponivel: false,
      motivo: 'Nenhum adaptador WebGPU. A GPU pode estar na lista de bloqueio do navegador.',
    };
  }

  const L = adapter.limits;
  const info = adapter.info ?? {};

  /*
   * Quantas células cabem. As Q populações são fatiadas entre vários storage
   * buffers para nenhum passar de maxStorageBufferBindingSize, e precisamos de
   * dois conjuntos (leitura e escrita) mais alguns buffers auxiliares —
   * macros, sólido, forças. Reservamos 4 dos bindings para esses.
   */
  const bindingsParaPop = Math.max(2, Math.min(L.maxStorageBuffersPerShaderStage - 4, 12));
  const tetoPorBuffer = Math.min(L.maxStorageBufferBindingSize, L.maxBufferSize);
  const bytesPop = bindingsParaPop * tetoPorBuffer;
  const maxCells = Math.floor(bytesPop / BYTES_POR_CELULA);

  return {
    id: 'webgpu',
    nome: 'WebGPU (compute)',
    disponivel: true,
    adapter,
    prioridade: 0,
    vendor: info.vendor ?? 'desconhecido',
    arquitetura: info.architecture ?? '',
    features: [...adapter.features],
    limites: {
      maxBufferSize: L.maxBufferSize,
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
      maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
    },
    /* subgroups acelera muito a redução das forças; timestamp-query dá
     * medição de kernel de verdade em vez de cronometrar com o relógio de
     * parede, que num pipeline assíncrono mede a coisa errada. */
    temSubgroups: adapter.features.has('subgroups'),
    temTimestamp: adapter.features.has('timestamp-query'),
    maxCells,
    nota: 'Teto vindo dos limites do adaptador; a VRAM real não é consultável.',
  };
}

/* ────────────────────────────────────────────────────────────────── WebGL2 */

function probeWebGL2() {
  let canvas, gl;
  try {
    canvas = document.createElement('canvas');
    gl = canvas.getContext('webgl2', { antialias: false, depth: false });
  } catch (e) {
    return { id: 'webgl2', disponivel: false, motivo: `getContext lançou: ${e.message}` };
  }
  if (!gl) {
    return { id: 'webgl2', disponivel: false, motivo: 'Contexto WebGL2 não pôde ser criado.' };
  }

  /*
   * Sem esta extensão não há render target de float, e sem render target de
   * float não há LBM: as populações precisam ser escritas com precisão total.
   * Não existe contorno para isso — é o requisito duro do caminho WebGL2.
   */
  const corFloat = gl.getExtension('EXT_color_buffer_float');
  if (!corFloat) {
    return {
      id: 'webgl2', disponivel: false,
      motivo: 'EXT_color_buffer_float ausente — sem render targets de float o solver não roda.',
    };
  }

  const maxTexture = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  const maxDraw = gl.getParameter(gl.MAX_DRAW_BUFFERS);
  const maxColor = gl.getParameter(gl.MAX_COLOR_ATTACHMENTS);

  /* Q=19 populações em RGBA32F -> ceil(19/4) = 5 texturas por conjunto. */
  const alvosNecessarios = Math.ceil(Q / 4);
  if (maxDraw < alvosNecessarios || maxColor < alvosNecessarios) {
    return {
      id: 'webgl2', disponivel: false,
      motivo: `MRT insuficiente: ${alvosNecessarios} alvos necessários, ` +
        `driver oferece ${Math.min(maxDraw, maxColor)}.`,
    };
  }

  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : 'desconhecido';

  /* O teto aqui é a área de textura, não bytes: cada tile em Z precisa caber
   * no atlas 2D. Filtramos os presets pelo que o atlas comporta. */
  let maxCells = 0;
  for (const p of PRESETS) {
    if (atlasLayout(p.nx, p.ny, p.nz, maxTexture)) maxCells = Math.max(maxCells, p.cells);
  }

  const perda = gl.getExtension('WEBGL_lose_context');
  if (perda) perda.loseContext();

  return {
    id: 'webgl2',
    nome: 'WebGL2 (fragment)',
    disponivel: true,
    prioridade: 1,
    renderer,
    limites: { maxTexture, maxDrawBuffers: maxDraw, maxColorAttachments: maxColor },
    temFloatLinear: !!gl.getExtension('OES_texture_float_linear'),
    maxCells,
    nota: 'Empacota 19 populações em 5 texturas RGBA32F e o lattice 3D num ' +
      'atlas 2D. Roda quase em qualquer lugar; ~2-4x mais lento que WebGPU.',
  };
}

/* ───────────────────────────────────────────────────────────────── fachada */

/**
 * Sonda todos os backends e devolve o quadro completo para a interface montar
 * o seletor. Nunca lança: um backend indisponível vira uma linha com motivo,
 * porque "por que não posso usar isto?" é a pergunta que o usuário faz.
 *
 * @returns {Promise<{backends: object[], recomendado: string|null, algum: boolean}>}
 */
export async function sondar() {
  const backends = [];

  backends.push(await probeWebGPU().catch(e => ({
    id: 'webgpu', disponivel: false, motivo: `sonda falhou: ${e.message}`,
  })));

  try {
    backends.push(probeWebGL2());
  } catch (e) {
    backends.push({ id: 'webgl2', disponivel: false, motivo: `sonda falhou: ${e.message}` });
  }

  const usaveis = backends.filter(b => b.disponivel);
  usaveis.sort((a, b) => a.prioridade - b.prioridade);

  return {
    backends,
    recomendado: usaveis.length ? usaveis[0].id : null,
    algum: usaveis.length > 0,
  };
}

/**
 * Presets que um backend comporta, cada um marcado com o motivo se não couber.
 * A interface mostra os que não cabem em cinza — saber que "Extrema" existe e
 * por que está fora vale mais do que a linha simplesmente não aparecer.
 */
export function presetsPara(backend) {
  return PRESETS.map(p => {
    if (!backend?.disponivel) {
      return { ...p, cabe: false, motivo: 'backend indisponível' };
    }
    if (p.cells > backend.maxCells) {
      return {
        ...p, cabe: false,
        motivo: `${(p.bytes / 1e9).toFixed(2)} GB excede o teto declarado ` +
          `de ${(backend.maxCells * BYTES_POR_CELULA / 1e9).toFixed(2)} GB`,
      };
    }
    return { ...p, cabe: true, motivo: null };
  });
}

/**
 * Preset inicial: o maior que cabe, recuado um degrau.
 *
 * O recuo é deliberado. O maior preset que "cabe" pelos limites declarados é
 * exatamente o que tem mais chance de falhar na alocação real ou de render a
 * 5 fps, e a primeira impressão de quem abre o link é a que conta. Quem quiser
 * o teto escolhe o teto — a opção está lá, marcada.
 */
export function presetInicial(backend) {
  const cabem = presetsPara(backend).filter(p => p.cabe);
  if (!cabem.length) return null;
  const i = Math.max(0, cabem.length - 2);
  return cabem[i];
}
