/* ── src/relatorio.js ────────────────────────────────────────────────────────
 *
 * O relatório de uma corrida: um documento HTML autocontido, gerado no
 * navegador a partir do estado do app, que abre numa aba e imprime em PDF.
 *
 * O QUE ELE É, E O QUE NÃO É
 * --------------------------
 * É um relatório QUALITATIVO com ressalvas automáticas. Ele diz o Cd medido,
 * a barra de erro, o Reynolds que se pediu e o que o lattice de fato resolve,
 * a resolução, o bloqueio, a área frontal voxelizada — e um selo que classifica
 * tudo isso antes que alguém leia o número. Nenhuma corrida no navegador sai
 * como "quantitativa validada": a 32 células no corpo e ω ≤ 1,85 o Re do
 * lattice fica cinco ordens de grandeza abaixo do de um carro, e o selo diz
 * isso em cima da página, não numa nota de rodapé.
 *
 * QUANDO ELE EXISTE
 * -----------------
 * Só depois de a simulação ter passado de três travessias do domínio e
 * acumulado amostras suficientes de Cd. Antes disso o Cd é o do transiente de
 * partida, e um relatório de transiente é um relatório de nada. O botão do
 * painel fica desabilitado até lá, mostrando o progresso.
 *
 * O SELO
 * ------
 * Segue o motor de validade do relatório executivo (set/2026), com os estados
 * que fazem sentido no navegador:
 *
 *   inválido       NaN/Inf na força ou no campo
 *   inconclusivo   menos de 3 travessias, ou amostras de menos
 *   instável       modo de paridade acima de 5% da força
 *   qualitativo    tudo certo, mas Re do lattice muito abaixo do físico —
 *                  que é o caso de todo veículo neste app
 *   resolvido      Re físico honrado pelo lattice (corpos pequenos e lentos)
 *
 * "comparativo" e "quantitativo validado" não são emitidos aqui: o primeiro
 * pede duas corridas com setup equivalente, o segundo pede um envelope
 * validado por classe de geometria, e nenhum dos dois existe ainda.
 */

const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const f = (v, n = 3) => Number.isFinite(v) ? v.toFixed(n).replace('.', ',') : '—';
const e = (v, n = 2) => Number.isFinite(v) ? v.toExponential(n).replace('.', ',').replace('e+', 'e') : '—';

/** Mínimos para o relatório existir. */
export const MINIMOS = { travessias: 3, amostras: 60 };

/**
 * O relatório está pronto? Devolve `{ pronto, motivo, travessias, amostras }`.
 *
 * @param {object} app  estado do app (index.html)
 */
export function prontidao(app) {
  const s = app.solver, u = app.units, p = app.preset;
  if (!s || !u || !p) return { pronto: false, motivo: 'sem simulação', travessias: 0, amostras: 0 };
  const travessias = s.passos / Math.ceil(p.nx / u.uLb);
  const amostras = app.janCd.n;
  if (s.fatorRampa < 1) return { pronto: false, motivo: 'acelerando o túnel', travessias, amostras };
  if (travessias < MINIMOS.travessias) {
    return { pronto: false, motivo: `${f(travessias, 1)} de ${MINIMOS.travessias} travessias`, travessias, amostras };
  }
  if (amostras < MINIMOS.amostras) {
    return { pronto: false, motivo: `${amostras} de ${MINIMOS.amostras} amostras`, travessias, amostras };
  }
  if (!Number.isFinite(app.janCd.media)) return { pronto: false, motivo: 'Cd indefinido', travessias, amostras };
  return { pronto: true, motivo: '', travessias, amostras };
}

/** A série da janela de Cd em ordem cronológica (o buffer é circular). */
function serieCronologica(jan) {
  const b = jan.buf;
  if (b.length < jan.capacidade) return b.slice();
  return b.slice(jan.i).concat(b.slice(0, jan.i));
}

/** Classifica a corrida. Ver o cabeçalho. */
export function selo(app, pr) {
  const u = app.units;
  const cd = app.janCd.media;
  if (!Number.isFinite(cd) || !Number.isFinite(app.janCd.desvio)) {
    return { estado: 'inválido', cor: '#c0392b',
      texto: 'A força ou o campo produziram NaN/Inf. Nenhum número desta corrida vale.' };
  }
  if (!pr.pronto) {
    return { estado: 'inconclusivo', cor: '#b7791f',
      texto: `A simulação ainda não estabilizou (${pr.motivo}). O Cd é o do transiente de partida.` };
  }
  const osc = app.oscParidade ?? 0;
  if (osc > 0.05) {
    return { estado: 'instável', cor: '#b7791f',
      texto: `O campo perto da superfície tem um modo de período 2 com ±${(osc * 100).toFixed(0)}% ` +
        'da força. A força publicada é a média dos dois estados, mas o escoamento ali não é estacionário.' };
  }
  if (u.resolved) {
    return { estado: 'resolvido', cor: '#2e7d32',
      texto: 'O lattice honra o Reynolds físico sem modelagem sub-grid. Os números medem o solver, ' +
        'dentro da precisão da resolução e do bloqueio.' };
  }
  return { estado: 'qualitativo', cor: '#1565c0',
    texto: `O lattice resolve Re ${e(u.reLattice, 1)}, e o escoamento real está em Re ${e(u.rePhysical, 1)} ` +
      `(${u.reRatio.toFixed(0)}× maior). O solver entrega a física do Reynolds que alcança, não a do ` +
      'que se pediu: vale para ver onde o escoamento separa e para comparar formas, não para prever ' +
      'o arrasto de um veículo real.' };
}

/** Sparkline SVG da série de Cd. */
function sparkline(serie, media, desvio) {
  if (serie.length < 2) return '';
  const W = 640, H = 120, m = 8;
  const lo = Math.min(...serie), hi = Math.max(...serie);
  const span = Math.max(hi - lo, 1e-9);
  const x = i => m + (W - 2 * m) * i / (serie.length - 1);
  const y = v => H - m - (H - 2 * m) * (v - lo) / span;
  const d = serie.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const faixa = Number.isFinite(desvio)
    ? `<rect x="${m}" y="${y(media + desvio).toFixed(1)}" width="${W - 2 * m}" ` +
      `height="${Math.max(0, y(media - desvio) - y(media + desvio)).toFixed(1)}" fill="#1565c0" opacity="0.12"/>`
    : '';
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px;display:block">` +
    faixa +
    `<line x1="${m}" x2="${W - m}" y1="${y(media).toFixed(1)}" y2="${y(media).toFixed(1)}" stroke="#1565c0" stroke-width="1" stroke-dasharray="4 3"/>` +
    `<path d="${d}" fill="none" stroke="#333" stroke-width="1.2"/>` +
    `<text x="${m}" y="12" font-size="11" fill="#666">${f(hi)}</text>` +
    `<text x="${m}" y="${H - 2}" font-size="11" fill="#666">${f(lo)}</text>` +
    '</svg>';
}

/**
 * Monta o documento.
 *
 * @param {object} app      estado do app
 * @param {object} o
 * @param {string} o.modelo       nome do modelo
 * @param {string} o.backend      'webgpu' | 'webgl2'
 * @param {string} o.gpu          nome da placa/driver
 * @param {string} [o.imagem]     data URL da cena
 * @param {string} [o.url]        endereço do app
 * @param {object} o.controles    { velocidade, comprimento, guinada, esteira, les }
 * @returns {{ html: string, dados: object }}
 */
export function gerar(app, o) {
  const u = app.units, s = app.solver, p = app.preset, co = app.ultimoCo;
  const pr = prontidao(app);
  const sl = selo(app, pr);
  const r = u.report();
  const serie = serieCronologica(app.janCd);
  const cd = app.janCd.media, sd = app.janCd.desvio, cl = app.janCl.media;
  const q = u.dynamicPressure;
  const areaM2 = app.areaFrontal * u.dx * u.dx;
  const arrastoN = cd * q * areaM2;
  const agora = new Date();

  const dados = {
    gerado: agora.toISOString(),
    app: { url: o.url ?? '', backend: o.backend, gpu: o.gpu },
    modelo: { nome: o.modelo, comprimentoM: o.controles.comprimento,
      sentido: app.prep?.sentido ?? null },
    dominio: { preset: p.id, nx: p.nx, ny: p.ny, nz: p.nz, celulas: p.cells,
      celulasNoCorpo: p.celulasNoCorpo, dxM: r.dxM, dtS: r.dtS },
    escoamento: { velocidadeMs: o.controles.velocidade, guinadaGraus: o.controles.guinada,
      esteira: o.controles.esteira, lesCs: r.lesCs, uLb: r.uLb, omegaPlus: r.omegaPlus,
      omegaMax: r.omegaMax, nuLb: r.nuLb, machLattice: r.machLattice },
    reynolds: { fisico: r.rePhysical, lattice: r.reLattice, razao: r.reRatio, resolvido: r.resolved },
    geometria: { areaFrontalCelulas: app.areaFrontal, areaFrontalM2: areaM2,
      bloqueio: app.areaFrontal / (p.ny * p.nz) },
    execucao: { passos: s.passos, tempoFisicoS: s.passos * u.dt, travessias: pr.travessias,
      amostras: app.janCd.n, intervaloAmostraS: 0.5 },
    resultados: { cd, cdDesvio: sd, cdBruto: co?.cdBruto ?? NaN, corrigidoMaskell: co?.corrigido ?? false,
      theta: co?.theta ?? NaN, cl, arrastoN, pressaoDinamicaPa: q, modoParidade: app.oscParidade ?? 0,
      serieCd: serie },
    validade: { estado: sl.estado, texto: sl.texto,
      quantitativoValidado: false,
      limitacoes: [
        'Reynolds do lattice muito abaixo do físico para veículos: o resultado é qualitativo.',
        'Superfície em escada de voxels: a área frontal voxelizada sai maior que a geométrica.',
        'LES grosseiro com Smagorinsky; não é DNS nem RANS validado.',
        'Cd corrigido por bloqueio (Maskell) quando positivo; sem correção espacial de convergência.',
      ] },
  };

  const linha = (k, v) => `<tr><th>${k}</th><td>${v}</td></tr>`;
  const html = `<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relatório — ${esc(o.modelo)} — CFD2026</title>
<style>
  body { font: 14px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif; color: #222; background: #fff;
         margin: 0; padding: 32px 40px; max-width: 900px; }
  h1 { font-size: 22px; margin: 0 0 2px; }
  h2 { font-size: 15px; margin: 28px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #ddd;
       text-transform: uppercase; letter-spacing: .04em; color: #555; }
  .sub { color: #666; margin-bottom: 18px; }
  .selo { border-left: 5px solid; padding: 10px 14px; margin: 14px 0 6px; background: #f7f8fa; }
  .selo b { text-transform: uppercase; letter-spacing: .05em; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; padding: 4px 10px 4px 0; vertical-align: top; border-bottom: 1px solid #eee; }
  th { color: #666; font-weight: 500; width: 46%; }
  td { font-variant-numeric: tabular-nums; }
  .grande { font-size: 30px; font-weight: 600; }
  .dois { display: grid; grid-template-columns: 1fr 1fr; gap: 0 28px; }
  img { max-width: 100%; border: 1px solid #ddd; border-radius: 3px; }
  .acoes { margin: 18px 0; display: flex; gap: 10px; }
  button { font: inherit; padding: 6px 12px; cursor: pointer; }
  ul { padding-left: 20px; } li { margin: 2px 0; }
  .nota { color: #666; font-size: 12.5px; }
  @media print { .acoes { display: none; } body { padding: 0; } }
  @media (max-width: 640px) { body { padding: 18px; } .dois { grid-template-columns: 1fr; } }
</style></head><body>
<h1>Túnel de vento virtual — ${esc(o.modelo)}</h1>
<div class="sub">Relatório qualitativo gerado em ${agora.toLocaleString('pt-BR')} · CFD2026, LBM D3Q19 na GPU do navegador
${o.url ? ` · <a href="${esc(o.url)}">${esc(o.url)}</a>` : ''}</div>

<div class="selo" style="border-color:${sl.cor}"><b style="color:${sl.cor}">${sl.estado}</b><br>${esc(sl.texto)}</div>
<p class="nota">O C<sub>d</sub> deste relatório não é um valor de engenharia e não substitui túnel de vento
nem CFD validado. Serve para visualizar o escoamento, identificar regiões de separação e comparar
formas sob o mesmo setup.</p>

<div class="acoes"><button onclick="print()">imprimir / salvar PDF</button>
<button onclick="baixarJson()">baixar dados (JSON)</button></div>

${o.imagem ? `<img src="${o.imagem}" alt="cena do túnel">` : ''}

<h2>Resultado</h2>
<div class="dois"><div>
<div class="grande">C<sub>d</sub> = ${f(cd)} ± ${f(sd)}</div>
<div class="nota">média e desvio-padrão sobre ${app.janCd.n} amostras a cada 0,5 s de parede,
${co?.corrigido ? `corrigido por bloqueio (Maskell, θ = ${f(co.theta, 1)}; bruto ${f(co.cdBruto)})` : 'sem correção de bloqueio'}</div>
</div><table>
${linha('C<sub>l</sub> (positivo para cima)', f(cl))}
${linha('arrasto a ' + f(o.controles.velocidade, 0) + ' m/s', f(arrastoN, 1) + ' N')}
${linha('pressão dinâmica ½ρU²', f(q, 1) + ' Pa')}
${linha('modo de paridade', (app.oscParidade ?? 0) > 0.02 ? `±${((app.oscParidade ?? 0) * 100).toFixed(0)}% da força` : 'ausente')}
</table></div>
${sparkline(serie, cd, sd)}
<div class="nota">Série do C<sub>d</sub> na janela final (${serie.length} amostras); a linha tracejada é a média, a faixa é ± um desvio.</div>

<h2>Regime</h2>
<table>
${linha('Reynolds físico (U·L/ν do ar)', e(r.rePhysical, 2))}
${linha('Reynolds que o lattice resolve', e(r.reLattice, 2) + (r.resolved ? ' — resolvido diretamente' : ` — ${r.reRatio.toFixed(0)}× menor que o físico`))}
${linha('viscosidade de lattice ν', e(r.nuLb, 3) + ` (ω⁺ = ${f(r.omegaPlus, 3)}, teto ${f(r.omegaMax, 2)})`)}
${linha('modelo sub-grid', r.lesCs > 0 ? `LES Smagorinsky, C<sub>s</sub> = ${f(r.lesCs, 2)}` : 'desligado')}
${linha('Mach de lattice', f(r.machLattice, 3) + ' (erro de compressibilidade ~Ma²)')}
</table>
<p class="nota">${esc(r.verdict)}</p>

<h2>Setup</h2>
<div class="dois"><table>
${linha('modelo', esc(o.modelo))}
${linha('comprimento de referência', f(o.controles.comprimento, 1) + ' m')}
${linha('velocidade', f(o.controles.velocidade, 0) + ' m/s (' + f(o.controles.velocidade * 3.6, 0) + ' km/h)')}
${linha('guinada', f(o.controles.guinada, 0) + '°')}
${linha('esteira rolante', o.controles.esteira ? 'ligada (piso a U∞)' : 'desligada (piso parado, camada limite 1/7 na entrada)')}
${linha('sentido', app.prep?.sentido ? (app.prep.sentido.manual ? 'definido pelo usuário' : 'palpite automático' + (app.prep.sentido.confianca < 0.12 ? ' (fraco)' : '')) : '—')}
</table><table>
${linha('domínio', `${p.nx} × ${p.ny} × ${p.nz} células (${(p.cells / 1e6).toFixed(2)} M)`)}
${linha('células no comprimento do corpo', p.celulasNoCorpo)}
${linha('tamanho da célula', f(r.dxM * 100, 2) + ' cm')}
${linha('passo de tempo', e(r.dtS, 2) + ' s')}
${linha('área frontal voxelizada', `${app.areaFrontal} céls² = ${f(areaM2, 2)} m²`)}
${linha('bloqueio', f(dados.geometria.bloqueio * 100, 2) + ' %')}
${linha('passos simulados', s.passos.toLocaleString('pt-BR') + ` (${f(s.passos * u.dt * 1000, 0)} ms físicos, ${f(pr.travessias, 1)} travessias)`)}
${linha('backend', `${o.backend === 'webgl2' ? 'WebGL2' : 'WebGPU'} · ${esc(o.gpu)}`)}
</table></div>

<h2>Limitações</h2>
<ul>${dados.validade.limitacoes.map(l => `<li>${esc(l)}</li>`).join('')}
<li>A área frontal voxelizada engorda o corpo em cerca de uma célula por lado e entra no denominador do C<sub>d</sub>.</li>
<li>Número de Strouhal e convergência espacial não são estimados neste relatório.</li></ul>
<p class="nota">O que este relatório pode afirmar: onde o escoamento acelera, separa e recircula; a tendência entre
duas formas sob o mesmo setup. O que ele não pode afirmar: o C<sub>d</sub> real do veículo, ganho percentual de
arrasto validado, consumo, ou equivalência a túnel de vento.</p>

<script type="application/json" id="dados">${JSON.stringify(dados).replace(/</g, '\\u003c')}</script>
<script>
function baixarJson() {
  const b = new Blob([document.getElementById('dados').textContent], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(b);
  a.download = ${JSON.stringify(`cfd2026-${o.modelo.replace(/[^\w.-]+/g, '_')}-${agora.toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`)};
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
</script>
</body></html>`;

  return { html, dados };
}

/**
 * Mostra o relatório por cima do app, num iframe.
 *
 * Não em `window.open`: pop-up sem gesto direto do usuário é bloqueado, e no
 * celular é bloqueado até com gesto em alguns navegadores. O iframe com
 * `srcdoc` funciona em todos, imprime (o `print()` de dentro imprime só o
 * relatório) e o botão "nova aba" — este sim num clique — abre o mesmo HTML
 * numa aba para quem quiser guardar o endereço enquanto a sessão dura.
 */
export function abrir(html) {
  document.getElementById('relatorioOverlay')?.remove();
  const ov = document.createElement('div');
  ov.id = 'relatorioOverlay';
  ov.style.cssText = 'position:fixed;inset:0;z-index:50;background:rgba(0,0,0,.65);' +
    'display:flex;flex-direction:column;padding:max(10px,env(safe-area-inset-top)) 10px 10px';
  const barra = document.createElement('div');
  barra.style.cssText = 'display:flex;gap:8px;justify-content:flex-end;margin-bottom:8px';
  const botao = (rotulo, acao) => {
    const b = document.createElement('button');
    b.textContent = rotulo;
    b.style.cssText = 'font:inherit;padding:6px 12px;cursor:pointer;background:#1b2734;' +
      'color:#cfe6ff;border:1px solid #2c4055;border-radius:4px';
    b.onclick = acao;
    barra.appendChild(b);
    return b;
  };
  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const fechar = () => { ov.remove(); URL.revokeObjectURL(url); removeEventListener('keydown', tecla); };
  const tecla = ev => { if (ev.key === 'Escape') fechar(); };
  botao('abrir em nova aba', () => window.open(url, '_blank'));
  botao('✕ fechar', fechar);
  addEventListener('keydown', tecla);

  const fr = document.createElement('iframe');
  fr.srcdoc = html;
  fr.title = 'relatório da corrida';
  fr.style.cssText = 'flex:1;width:100%;border:0;border-radius:4px;background:#fff';
  ov.append(barra, fr);
  document.body.appendChild(ov);
  return true;
}
