/* ── src/geom/parse.js ───────────────────────────────────────────────────────
 *
 * Leitura de malhas: STL, OBJ, PLY, glTF e GLB.
 *
 * Saída única para todos os formatos:
 *
 *     { positions: Float32Array(3*nv), indices: Uint32Array(3*nt),
 *       nome, formato, avisos: string[] }
 *
 * Os triângulos NÃO são soldados. O BVH não se importa com vértices
 * duplicados, e soldar 1,5 milhão de triângulos custa mais do que tudo o que
 * viria depois. Quem precisa de topologia (subdivisão, normais suaves) que
 * solde sob demanda.
 *
 *
 * SOBRE CONFIAR NO CABEÇALHO
 * --------------------------
 * Nenhuma detecção de formato aqui acredita no que o arquivo diz sobre si
 * mesmo, porque os arquivos mentem. Dois exemplos, ambos vindos dos modelos de
 * teste deste repositório:
 *
 *   - Um STL BINÁRIO cujo cabeçalho de 80 bytes começa com a palavra "solid",
 *     que é exatamente como um STL ASCII começa. Quem detecta por prefixo lê
 *     19 MB de bytes binários como texto e devolve zero triângulos, ou pior,
 *     alguns triângulos de lixo. A checagem que funciona é aritmética:
 *     84 + 50*n deve dar o tamanho do arquivo.
 *
 *   - Um STL binário com UM BYTE A MENOS que o necessário — o atributo de
 *     16 bits do último triângulo foi cortado. Rejeitar por isso seria
 *     tecnicamente correto e inútil: os 12 triângulos estão todos lá. Lemos o
 *     que existe e registramos um aviso.
 *
 * A regra geral: aceitar o arquivo se for possível extrair geometria dele, e
 * registrar em `avisos` tudo que foi tolerado. O usuário carregou um modelo
 * para ver o escoamento, não para receber uma aula sobre a especificação.
 */

/* ────────────────────────────────────────────────────────────────────── STL */

const TAM_TRI_BIN = 50;      // 12 floats + uint16 de atributo
const TAM_CABECALHO = 84;    // 80 de cabeçalho + uint32 de contagem

/**
 * Decide se um buffer é STL binário, pela aritmética e não pelo prefixo.
 * Devolve { n, base, truncado } ou null se não for binário.
 *
 * `base` é o deslocamento onde a contagem de triângulos foi encontrada. Ele é
 * 80 em qualquer arquivo que siga a especificação, e a razão de ser um
 * parâmetro é que nem todo arquivo segue: um dos modelos de teste deste
 * repositório tem cabeçalho de SETENTA E NOVE bytes, e com a contagem lida em
 * 80 o número sai zero e o arquivo é rejeitado como "não é STL". Em 79 a conta
 * fecha exata: 79 + 4 + 12*50 = 683, o tamanho do arquivo.
 *
 * A varredura é estreita (76..84) e só aceita fechamento EXATO, o que a torna
 * essencialmente impossível de disparar por acaso: um inteiro arbitrário de 32
 * bits que por coincidência satisfaça base + 4 + 50n = tamanho é raro o
 * bastante para não valer a preocupação.
 */
function contagemSTLBinario(buf) {
  if (buf.byteLength < TAM_CABECALHO) return null;
  const dv = new DataView(buf);

  const tentar = (base) => {
    if (base + 4 > buf.byteLength) return null;
    const n = dv.getUint32(base, true);
    if (n === 0 || n > 200e6) return null;
    return { n, esperado: base + 4 + n * TAM_TRI_BIN };
  };

  /* 1. o caminho da especificação: contagem em 80, conta fechando exata */
  const padrao = tentar(80);
  if (padrao && padrao.esperado === buf.byteLength) {
    return { n: padrao.n, base: 84, truncado: 0, deslocado: 0 };
  }

  /* 2. cabeçalho de tamanho não-padrão, mas com a conta fechando exata */
  for (let base = 76; base <= 84; base++) {
    if (base === 80) continue;
    const t = tentar(base);
    if (t && t.esperado === buf.byteLength) {
      return { n: t.n, base: base + 4, truncado: 0, deslocado: base - 80 };
    }
  }

  /* 3. contagem em 80 e arquivo cortado ou com sobra de poucos bytes.
   *    Tolerância estreita de propósito: além disso, o "n" que lemos
   *    provavelmente é texto ASCII interpretado como inteiro. */
  if (padrao) {
    const folga = buf.byteLength - padrao.esperado;
    const disponivel = Math.floor((buf.byteLength - TAM_CABECALHO) / TAM_TRI_BIN);
    if (folga >= -TAM_TRI_BIN && folga <= 4 && disponivel > 0) {
      const n = Math.min(padrao.n, disponivel);
      return { n, base: 84, truncado: padrao.n - n, deslocado: 0 };
    }
  }

  return null;
}

function parseSTLBinario(buf, { n, base, truncado, deslocado }, avisos) {
  const dv = new DataView(buf);
  const positions = new Float32Array(n * 9);

  if (deslocado) {
    avisos.push(`cabeçalho de ${base - 4} bytes em vez de 80 — a contagem de ` +
      `triângulos está no deslocamento ${base - 4}, onde a conta fecha exata`);
  }

  let p = base;
  for (let t = 0; t < n; t++) {
    p += 12;                                  // a normal do arquivo é ignorada
    for (let k = 0; k < 9; k++) {
      positions[t * 9 + k] = dv.getFloat32(p, true);
      p += 4;
    }
    p += 2;                                   // atributo
  }

  if (truncado > 0) {
    avisos.push(`arquivo termina cedo: ${truncado} triângulo(s) declarado(s) ` +
      'no cabeçalho não estão no arquivo e foram ignorados');
  }

  const indices = new Uint32Array(n * 3);
  for (let i = 0; i < n * 3; i++) indices[i] = i;
  return { positions, indices };
}

function parseSTLAscii(texto, avisos) {
  /* Um regex sobre 70 MB de texto é lento e aloca demais; varremos por token.
   * A gramática de um STL ASCII é rasa o bastante para isso ser trivial. */
  const verts = [];
  const re = /vertex\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)/g;
  let m;
  while ((m = re.exec(texto)) !== null) {
    verts.push(+m[1], +m[2], +m[3]);
  }
  if (verts.length === 0) throw new Error('STL ASCII sem nenhum "vertex"');
  if (verts.length % 9 !== 0) {
    const sobra = (verts.length % 9) / 3;
    avisos.push(`${sobra} vértice(s) soltos no fim do arquivo, descartados`);
    verts.length -= verts.length % 9;
  }

  const positions = new Float32Array(verts);
  const indices = new Uint32Array(positions.length / 3);
  for (let i = 0; i < indices.length; i++) indices[i] = i;
  return { positions, indices };
}

export function parseSTL(buf, avisos = []) {
  const bin = contagemSTLBinario(buf);
  if (bin) {
    const cabecalho = new TextDecoder('latin1')
      .decode(new Uint8Array(buf, 0, Math.min(80, bin.base - 4)))
      .replace(/\0.*$/s, '').trim();
    if (/^solid/i.test(cabecalho)) {
      avisos.push('cabeçalho começa com "solid" mas o arquivo é binário — ' +
        'detectado pela aritmética 84 + 50n, não pelo prefixo');
    }
    return { ...parseSTLBinario(buf, bin, avisos), formato: 'stl-binario', nome: cabecalho };
  }

  const texto = new TextDecoder('utf-8', { fatal: false }).decode(buf);
  if (!/^\s*solid/i.test(texto)) {
    throw new Error(
      'não é STL: a contagem binária não fecha com o tamanho do arquivo e o ' +
      'conteúdo não começa com "solid"');
  }
  const nome = (texto.match(/^\s*solid[ \t]*(.*)$/m)?.[1] ?? '').trim();
  return { ...parseSTLAscii(texto, avisos), formato: 'stl-ascii', nome };
}

/* ────────────────────────────────────────────────────────────────────── OBJ */

export function parseOBJ(texto, avisos = []) {
  const vx = [];
  const tris = [];
  let nome = '';

  /* Índices do OBJ são 1-based e podem ser NEGATIVOS, contando de trás para
   * frente a partir do último vértice lido até ali. Exportadores usam isso. */
  const resolver = (s) => {
    const i = parseInt(s, 10);
    if (Number.isNaN(i)) return -1;
    return i > 0 ? i - 1 : (vx.length / 3) + i;
  };

  for (const linha of texto.split('\n')) {
    if (linha.charCodeAt(0) === 35) continue;            // '#'
    const t = linha.trim();
    if (!t) continue;

    if (t.startsWith('v ')) {
      const p = t.split(/\s+/);
      vx.push(+p[1], +p[2], +p[3]);
    } else if (t.startsWith('f ')) {
      const p = t.split(/\s+/).slice(1);
      /* "v", "v/vt", "v//vn", "v/vt/vn" — só a primeira parte interessa */
      const idx = p.map(s => resolver(s.split('/')[0])).filter(i => i >= 0);
      /* Faces podem ser polígonos de n lados; leque a partir do primeiro. */
      for (let k = 1; k + 1 < idx.length; k++) {
        tris.push(idx[0], idx[k], idx[k + 1]);
      }
    } else if (t.startsWith('o ') || t.startsWith('g ')) {
      if (!nome) nome = t.slice(2).trim();
    }
  }

  if (!tris.length) throw new Error('OBJ sem faces trianguláveis');

  const nv = vx.length / 3;
  const fora = tris.filter(i => i < 0 || i >= nv).length;
  if (fora) {
    avisos.push(`${fora} índice(s) de face fora do intervalo, descartados`);
  }

  return {
    positions: new Float32Array(vx),
    indices: new Uint32Array(tris.filter(i => i >= 0 && i < nv)),
    formato: 'obj', nome,
  };
}

/* ────────────────────────────────────────────────────────────────────── PLY */

export function parsePLY(buf, avisos = []) {
  const cabBytes = new Uint8Array(buf, 0, Math.min(buf.byteLength, 65536));
  const cabTexto = new TextDecoder('latin1').decode(cabBytes);
  const fim = cabTexto.indexOf('end_header');
  if (fim < 0) throw new Error('PLY sem end_header nos primeiros 64 KB');

  const inicioDados = cabTexto.indexOf('\n', fim) + 1;
  const linhas = cabTexto.slice(0, fim).split(/\r?\n/);

  let formato = null, elementoAtual = null;
  const elementos = [];
  const TAM = {
    char: 1, uchar: 1, int8: 1, uint8: 1,
    short: 2, ushort: 2, int16: 2, uint16: 2,
    int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4,
    double: 8, float64: 8,
  };

  for (const l of linhas) {
    const p = l.trim().split(/\s+/);
    if (p[0] === 'format') formato = p[1];
    else if (p[0] === 'element') {
      elementoAtual = { nome: p[1], contagem: +p[2], props: [] };
      elementos.push(elementoAtual);
    } else if (p[0] === 'property' && elementoAtual) {
      if (p[1] === 'list') {
        elementoAtual.props.push({ lista: true, tipoTam: p[2], tipoItem: p[3], nome: p[4] });
      } else {
        elementoAtual.props.push({ lista: false, tipo: p[1], nome: p[2] });
      }
    }
  }

  const elVert = elementos.find(e => e.nome === 'vertex');
  const elFace = elementos.find(e => e.nome === 'face');
  if (!elVert) throw new Error('PLY sem elemento "vertex"');

  const positions = new Float32Array(elVert.contagem * 3);
  const tris = [];

  if (formato === 'ascii') {
    const texto = new TextDecoder('latin1').decode(new Uint8Array(buf));
    const corpo = texto.slice(texto.indexOf('\n', texto.indexOf('end_header')) + 1);
    const tok = corpo.split(/\s+/).filter(s => s.length);
    let p = 0;
    const ix = elVert.props.findIndex(q => q.nome === 'x');
    for (let i = 0; i < elVert.contagem; i++) {
      for (let k = 0; k < elVert.props.length; k++) {
        const v = +tok[p++];
        if (k >= ix && k < ix + 3) positions[i * 3 + (k - ix)] = v;
      }
    }
    if (elFace) {
      for (let i = 0; i < elFace.contagem; i++) {
        const n = +tok[p++];
        const face = [];
        for (let k = 0; k < n; k++) face.push(+tok[p++]);
        for (let k = 1; k + 1 < n; k++) tris.push(face[0], face[k], face[k + 1]);
      }
    }
  } else {
    const le = formato === 'binary_little_endian';
    const dv = new DataView(buf);
    let p = inicioDados;

    const ler = (tipo) => {
      switch (tipo) {
        case 'char': case 'int8': { const v = dv.getInt8(p); p += 1; return v; }
        case 'uchar': case 'uint8': { const v = dv.getUint8(p); p += 1; return v; }
        case 'short': case 'int16': { const v = dv.getInt16(p, le); p += 2; return v; }
        case 'ushort': case 'uint16': { const v = dv.getUint16(p, le); p += 2; return v; }
        case 'int': case 'int32': { const v = dv.getInt32(p, le); p += 4; return v; }
        case 'uint': case 'uint32': { const v = dv.getUint32(p, le); p += 4; return v; }
        case 'float': case 'float32': { const v = dv.getFloat32(p, le); p += 4; return v; }
        case 'double': case 'float64': { const v = dv.getFloat64(p, le); p += 8; return v; }
        default: throw new Error(`tipo PLY desconhecido: ${tipo}`);
      }
    };

    const ix = elVert.props.findIndex(q => q.nome === 'x');
    for (let i = 0; i < elVert.contagem; i++) {
      for (let k = 0; k < elVert.props.length; k++) {
        const v = ler(elVert.props[k].tipo);
        if (k >= ix && k < ix + 3) positions[i * 3 + (k - ix)] = v;
      }
    }
    if (elFace) {
      const pl = elFace.props.find(q => q.lista);
      for (let i = 0; i < elFace.contagem; i++) {
        for (const prop of elFace.props) {
          if (prop !== pl) { ler(prop.tipo); continue; }
          const n = ler(prop.tipoTam);
          const face = [];
          for (let k = 0; k < n; k++) face.push(ler(prop.tipoItem));
          for (let k = 1; k + 1 < n; k++) tris.push(face[0], face[k], face[k + 1]);
        }
      }
    }
  }

  if (!tris.length) throw new Error('PLY sem faces');
  return { positions, indices: new Uint32Array(tris), formato: 'ply', nome: '' };
}

/* ────────────────────────────────────────────────────────────── glTF / GLB */

const COMPONENTE = {
  5120: { arr: Int8Array, tam: 1 },
  5121: { arr: Uint8Array, tam: 1 },
  5122: { arr: Int16Array, tam: 2 },
  5123: { arr: Uint16Array, tam: 2 },
  5125: { arr: Uint32Array, tam: 4 },
  5126: { arr: Float32Array, tam: 4 },
};
const N_COMP = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

function mat4Identidade() {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

/* Coluna-maior, como o glTF armazena. */
function mat4Mul(a, b) {
  const o = new Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] +
        a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

function trsParaMat4(t, r, s) {
  const [x, y, z, w] = r;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  return [
    (1 - (yy + zz)) * s[0], (xy + wz) * s[0], (xz - wy) * s[0], 0,
    (xy - wz) * s[1], (1 - (xx + zz)) * s[1], (yz + wx) * s[1], 0,
    (xz + wy) * s[2], (yz - wx) * s[2], (1 - (xx + yy)) * s[2], 0,
    t[0], t[1], t[2], 1,
  ];
}

function b64ParaBuffer(b64) {
  const bin = atob(b64);
  const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u.buffer;
}

/**
 * Percorre a cena, aplica as transformações de nó e concatena todos os
 * primitivos numa malha só.
 *
 * Concatenar é o certo aqui: o solver quer um campo de sólido, não uma
 * hierarquia. Um GLB de carro traz carroceria, rodas, espelhos e grades como
 * nós separados, e todos são obstáculo do mesmo jeito.
 */
function montarGLTF(json, buffers, avisos) {
  const posAcc = [];
  const idxAcc = [];
  let totalV = 0;

  const bytesDe = (iAcc) => {
    const acc = json.accessors[iAcc];
    const comp = COMPONENTE[acc.componentType];
    const n = N_COMP[acc.type];
    if (!comp || !n) throw new Error(`accessor ${iAcc}: tipo não suportado`);

    if (acc.bufferView === undefined) {
      return new comp.arr(acc.count * n);            // acessor esparso vazio
    }
    const bv = json.bufferViews[acc.bufferView];
    const buf = buffers[bv.buffer ?? 0];
    if (!buf) throw new Error(`bufferView aponta para buffer ausente`);

    const base = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const passo = bv.byteStride ?? 0;

    if (!passo || passo === comp.tam * n) {
      return new comp.arr(buf, base, acc.count * n);
    }
    /* Entrelaçado: copiar desentrelaçando. */
    const saida = new comp.arr(acc.count * n);
    const dv = new DataView(buf);
    const leitor = {
      5120: (o) => dv.getInt8(o), 5121: (o) => dv.getUint8(o),
      5122: (o) => dv.getInt16(o, true), 5123: (o) => dv.getUint16(o, true),
      5125: (o) => dv.getUint32(o, true), 5126: (o) => dv.getFloat32(o, true),
    }[acc.componentType];
    for (let i = 0; i < acc.count; i++) {
      for (let k = 0; k < n; k++) {
        saida[i * n + k] = leitor(base + i * passo + k * comp.tam);
      }
    }
    return saida;
  };

  const visitar = (iNode, pai) => {
    const node = json.nodes[iNode];
    if (!node) return;

    const local = node.matrix ? node.matrix : trsParaMat4(
      node.translation ?? [0, 0, 0],
      node.rotation ?? [0, 0, 0, 1],
      node.scale ?? [1, 1, 1]);
    const M = mat4Mul(pai, local);

    if (node.mesh !== undefined) {
      const mesh = json.meshes[node.mesh];
      for (const prim of mesh.primitives ?? []) {
        /* modo 4 = TRIANGLES. Faixas e leques existem e são raros em modelos
         * de CAD exportados; avisamos em vez de produzir lixo silencioso. */
        if (prim.mode !== undefined && prim.mode !== 4) {
          avisos.push(`primitivo com mode=${prim.mode} (não TRIANGLES) ignorado`);
          continue;
        }
        if (prim.extensions?.KHR_draco_mesh_compression) {
          avisos.push('primitivo comprimido com Draco ignorado — ' +
            'exporte sem compressão Draco');
          continue;
        }
        const iPos = prim.attributes?.POSITION;
        if (iPos === undefined) continue;

        const pos = bytesDe(iPos);
        const nv = pos.length / 3;
        const transformado = new Float32Array(pos.length);
        for (let i = 0; i < nv; i++) {
          const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
          transformado[i * 3] = M[0] * x + M[4] * y + M[8] * z + M[12];
          transformado[i * 3 + 1] = M[1] * x + M[5] * y + M[9] * z + M[13];
          transformado[i * 3 + 2] = M[2] * x + M[6] * y + M[10] * z + M[14];
        }
        posAcc.push(transformado);

        let ind;
        if (prim.indices !== undefined) {
          const src = bytesDe(prim.indices);
          ind = new Uint32Array(src.length);
          for (let i = 0; i < src.length; i++) ind[i] = src[i] + totalV;
        } else {
          ind = new Uint32Array(nv);
          for (let i = 0; i < nv; i++) ind[i] = i + totalV;
        }
        idxAcc.push(ind);
        totalV += nv;
      }
    }

    for (const filho of node.children ?? []) visitar(filho, M);
  };

  const cena = json.scenes?.[json.scene ?? 0];
  const raizes = cena?.nodes ?? json.nodes?.map((_, i) => i) ?? [];
  for (const r of raizes) visitar(r, mat4Identidade());

  if (!posAcc.length) throw new Error('glTF sem geometria triangular');

  const positions = new Float32Array(posAcc.reduce((a, b) => a + b.length, 0));
  let o = 0;
  for (const a of posAcc) { positions.set(a, o); o += a.length; }

  const indices = new Uint32Array(idxAcc.reduce((a, b) => a + b.length, 0));
  o = 0;
  for (const a of idxAcc) { indices.set(a, o); o += a.length; }

  return { positions, indices };
}

export function parseGLB(buf, avisos = []) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('sem a magia "glTF"');
  const versao = dv.getUint32(4, true);
  if (versao !== 2) avisos.push(`GLB versão ${versao}; o leitor é de glTF 2.0`);

  let json = null;
  const bins = [];
  let p = 12;
  while (p + 8 <= buf.byteLength) {
    const tam = dv.getUint32(p, true);
    const tipo = dv.getUint32(p + 4, true);
    const dados = buf.slice(p + 8, p + 8 + tam);
    if (tipo === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(dados));
    else if (tipo === 0x004e4942) bins.push(dados);
    p += 8 + tam + ((4 - (tam % 4)) % 4);
  }
  if (!json) throw new Error('GLB sem chunk JSON');

  /* Buffers externos (uri) não existem num GLB bem formado, mas existem em
   * arquivos convertidos meio caminho. Data URIs a gente resolve; caminhos de
   * arquivo não, porque não há de onde buscar. */
  const buffers = (json.buffers ?? []).map((b, i) => {
    if (b.uri === undefined) return bins[0] ?? bins[i];
    if (b.uri.startsWith('data:')) return b64ParaBuffer(b.uri.split(',')[1]);
    avisos.push(`buffer externo "${b.uri}" não pôde ser carregado`);
    return null;
  });
  if (!buffers.length) buffers.push(bins[0]);

  const nome = json.scenes?.[json.scene ?? 0]?.name ?? json.asset?.generator ?? '';
  return { ...montarGLTF(json, buffers, avisos), formato: 'glb', nome };
}

export function parseGLTF(texto, avisos = []) {
  const json = JSON.parse(texto);
  const buffers = (json.buffers ?? []).map(b => {
    if (b.uri?.startsWith('data:')) return b64ParaBuffer(b.uri.split(',')[1]);
    avisos.push(`buffer externo "${b.uri}" não pôde ser carregado — ` +
      'use .glb, que embute os dados');
    return null;
  });
  const nome = json.scenes?.[json.scene ?? 0]?.name ?? '';
  return { ...montarGLTF(json, buffers, avisos), formato: 'gltf', nome };
}

/* ──────────────────────────────────────────────────────────────── despacho */

/**
 * Lê qualquer formato suportado, decidindo pelo conteúdo e usando o nome do
 * arquivo apenas como desempate.
 *
 * @param {ArrayBuffer} buf
 * @param {string} [nomeArquivo]
 */
export function ler(buf, nomeArquivo = '') {
  const avisos = [];
  const ext = (nomeArquivo.match(/\.([a-z0-9]+)$/i)?.[1] ?? '').toLowerCase();

  const dv = new DataView(buf);
  const magia = buf.byteLength >= 4 ? dv.getUint32(0, true) : 0;

  const inicio = new TextDecoder('latin1')
    .decode(new Uint8Array(buf, 0, Math.min(buf.byteLength, 1024)));

  let r;
  if (magia === 0x46546c67) r = parseGLB(buf, avisos);
  else if (/^ply\s/.test(inicio)) r = parsePLY(buf, avisos);
  else if (ext === 'gltf' || /^\s*\{[\s\S]*"asset"/.test(inicio)) {
    r = parseGLTF(new TextDecoder().decode(buf), avisos);
  } else if (ext === 'obj' || /^\s*(v\s|vn\s|vt\s|#|mtllib|o\s|g\s)/m.test(inicio)) {
    /* OBJ só ganha se o STL não reivindicar o arquivo: um STL binário pode ter
     * qualquer coisa nos 80 bytes de cabeçalho, inclusive algo com cara de OBJ. */
    r = contagemSTLBinario(buf)
      ? parseSTL(buf, avisos)
      : parseOBJ(new TextDecoder().decode(buf), avisos);
  } else {
    r = parseSTL(buf, avisos);
  }

  const nTri = r.indices.length / 3;
  const nVert = r.positions.length / 3;
  if (!nTri) throw new Error(`${r.formato}: nenhum triângulo`);

  return { ...r, avisos, nTriangulos: nTri, nVertices: nVert };
}

/** Caixa envolvente, e uma checagem de que as coordenadas são finitas. */
export function extentos(positions) {
  let minx = Infinity, miny = Infinity, minz = Infinity;
  let maxx = -Infinity, maxy = -Infinity, maxz = -Infinity;
  let naoFinitos = 0;
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i], y = positions[i + 1], z = positions[i + 2];
    if (!Number.isFinite(x + y + z)) { naoFinitos++; continue; }
    if (x < minx) minx = x; if (x > maxx) maxx = x;
    if (y < miny) miny = y; if (y > maxy) maxy = y;
    if (z < minz) minz = z; if (z > maxz) maxz = z;
  }
  return {
    min: [minx, miny, minz], max: [maxx, maxy, maxz],
    tamanho: [maxx - minx, maxy - miny, maxz - minz],
    centro: [(minx + maxx) / 2, (miny + maxy) / 2, (minz + maxz) / 2],
    naoFinitos,
  };
}
