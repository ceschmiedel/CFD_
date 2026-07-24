/* ── src/core/forces.js ──────────────────────────────────────────────────────
 *
 * Da força em unidades de lattice ao coeficiente que se publica.
 *
 * O kernel em emit/wgsl.js entrega três números: a força que o escoamento
 * exerce sobre o corpo, em unidades de lattice, por troca de momento. Este
 * módulo converte isso em Cd, Cl e Cs — e aplica a correção que separa um
 * número de túnel de um número de brinquedo.
 */

/**
 * Correção de bloqueio de Maskell (Barlow, Rae & Pope, "Low-Speed Wind Tunnel
 * Testing", 3ª ed., §10.4):
 *
 *     Cd_livre = Cd_medido / (1 + theta * Cd_medido * S / C)
 *
 * S é a área frontal do corpo, C a da seção de teste, e theta ~ 2,5 para um
 * corpo rombudo tridimensional (2,77 para uma placa plana normal ao
 * escoamento).
 *
 * POR QUE ISTO EXISTE
 * -------------------
 * Um corpo dentro de um túnel bloqueia parte da seção. O ar que não passa por
 * ele tem de passar pelos lados, acelera, e a queda de pressão que acompanha
 * essa aceleração puxa o corpo para trás. O arrasto medido é maior que o
 * arrasto do mesmo corpo em ar livre, e a diferença não é pequena: todo túnel
 * real aplica esta correção antes de publicar um número, e este também aplica.
 *
 * ONDE ELA QUEBRA
 * ---------------
 * A correção é derivada para um corpo que DESPRENDE esteira, o que pressupõe
 * Cd > 0. Alimentada com um Cd negativo — um corpo produzindo empuxo, ou o
 * transiente dos primeiros passos antes de o escoamento se estabelecer — o
 * denominador passa por zero em Cd = -1/(theta*b) e troca de sinal: a função
 * devolve arrasto positivo a partir de um negativo e, logo antes disso,
 * amplifica em vez de encolher. Não há o que corrigir num corpo que não está
 * sendo empurrado a favor do escoamento, então devolvemos o valor intacto.
 */
export function maskell(cdMedido, bloqueio, theta = 2.5) {
  if (!(bloqueio > 0) || !Number.isFinite(cdMedido)) return cdMedido;
  if (cdMedido <= 0) return cdMedido;
  return cdMedido / (1 + theta * cdMedido * bloqueio);
}

/**
 * Coeficientes aerodinâmicos a partir da força de lattice.
 *
 * @param {number[]} forcaLb  [fx, fy, fz] em unidades de lattice
 * @param {object} units      instância de Units
 * @param {number} areaFrontalCelulas  área frontal projetada, em células²
 * @param {object} grade      { nx, ny, nz }
 * @param {object} [opt]
 * @param {boolean} [opt.corrigirBloqueio]
 */
export function coeficientes(forcaLb, units, areaFrontalCelulas, grade,
  { corrigirBloqueio = true, theta = 2.5 } = {}) {

  const [fx, fy, fz] = forcaLb;

  /*
   * O denominador. Em unidades de lattice a pressão dinâmica é
   * (1/2) rho u_lb^2 com rho = 1, e a área está em células² — as escalas de
   * comprimento e tempo cancelam exatamente, e o coeficiente sai adimensional
   * sem passar pelo SI. Converter para newtons e metros primeiro daria o mesmo
   * número por um caminho mais longo e com mais chance de erro de fator.
   */
  const q = 0.5 * units.uLb * units.uLb;
  const denom = Math.max(q * areaFrontalCelulas, 1e-30);

  const cdBruto = fx / denom;
  const bloqueio = areaFrontalCelulas / (grade.ny * grade.nz);

  const cd = corrigirBloqueio ? maskell(cdBruto, bloqueio, theta) : cdBruto;

  return {
    cd,
    cdBruto,
    /* Sustentação positiva para cima (z), força lateral em y. O sinal de cl
     * importa: um carro com cl > 0 está sendo levantado, o que é uma
     * afirmação sobre estabilidade, não um detalhe de convenção. */
    cl: fz / denom,
    cs: fy / denom,
    bloqueio,
    theta,
    corrigido: corrigirBloqueio && cdBruto > 0,
    /* Em SI, para quem quer newtons. */
    forcaN: [
      units.forceToSi(fx), units.forceToSi(fy), units.forceToSi(fz),
    ],
    areaFrontalM2: areaFrontalCelulas * units.dx * units.dx,
    /* Arrasto em newtons pela via independente: Cd * q * A. Se este número e
     * forcaN[0] discordarem, há erro de escala em algum lugar. */
    arrastoN: cd * units.dynamicPressure * areaFrontalCelulas * units.dx * units.dx,
  };
}

/**
 * Média e desvio de uma janela deslizante de coeficientes.
 *
 * Um Cd instantâneo de um corpo rombudo não significa nada: a esteira desprende
 * vórtices e o valor oscila com amplitude que chega a 5% do valor médio. O que
 * se publica é a média sobre vários períodos de desprendimento, e a barra de
 * erro é o desvio — reportá-la é a diferença entre "Cd = 0,31" e
 * "Cd = 0,31 ± 0,02", e só a segunda é uma medida.
 */
export class JanelaCd {
  constructor(capacidade = 600) {
    this.capacidade = capacidade;
    this.buf = [];
    this.i = 0;
  }

  adicionar(v) {
    if (!Number.isFinite(v)) return;
    if (this.buf.length < this.capacidade) this.buf.push(v);
    else { this.buf[this.i] = v; this.i = (this.i + 1) % this.capacidade; }
  }

  limpar() { this.buf.length = 0; this.i = 0; }

  get n() { return this.buf.length; }

  get media() {
    if (!this.buf.length) return NaN;
    return this.buf.reduce((a, b) => a + b, 0) / this.buf.length;
  }

  get desvio() {
    const n = this.buf.length;
    if (n < 2) return NaN;
    const m = this.media;
    return Math.sqrt(this.buf.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1));
  }

  /** true quando a janela está cheia — antes disso a média é provisória. */
  get madura() { return this.buf.length >= this.capacidade; }
}

/**
 * Número de Strouhal a partir do histórico da força lateral ou de sustentação.
 *
 * St = f D / U, com f a frequência dominante do desprendimento. Extraída por
 * contagem de cruzamentos por zero da componente oscilante, que para um sinal
 * de banda estreita é mais robusto e muito mais barato que uma FFT.
 *
 * @param {number[]} serie   histórico do coeficiente, um por passo
 * @param {number} passosPorAmostra
 * @param {number} dCelulas  dimensão característica transversal, em células
 * @param {number} uLb
 */
export function strouhal(serie, passosPorAmostra, dCelulas, uLb) {
  const n = serie.length;
  if (n < 32) return { st: NaN, motivo: 'série curta demais' };

  const m = serie.reduce((a, b) => a + b, 0) / n;
  const osc = serie.map(v => v - m);

  const amp = Math.sqrt(osc.reduce((a, b) => a + b * b, 0) / n);
  if (amp < 1e-6) return { st: 0, motivo: 'sem oscilação: esteira estacionária' };

  /* Só contam cruzamentos com amplitude acima do ruído, senão o ruído numérico
   * em torno de zero produz uma frequência alta e inteiramente falsa. */
  const limiar = 0.25 * amp;
  let cruz = 0, ultimoSinal = 0;
  for (const v of osc) {
    const s = v > limiar ? 1 : (v < -limiar ? -1 : 0);
    if (s !== 0 && ultimoSinal !== 0 && s !== ultimoSinal) cruz++;
    if (s !== 0) ultimoSinal = s;
  }
  if (cruz < 4) return { st: NaN, motivo: `só ${cruz} cruzamentos; simule mais` };

  const periodos = cruz / 2;
  const passosTotais = n * passosPorAmostra;
  const freqLb = periodos / passosTotais;

  return { st: freqLb * dCelulas / uLb, periodos, freqLb, amplitude: amp };
}
