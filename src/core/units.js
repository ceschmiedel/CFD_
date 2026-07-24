/* ── src/core/units.js ───────────────────────────────────────────────────────
 *
 * A ponte entre metros por segundo e unidades de lattice, e o lugar onde este
 * programa é honesto sobre o que consegue e o que não consegue resolver.
 *
 *
 * A ARITMÉTICA DESCONFORTÁVEL
 * ---------------------------
 * Um carro de 4,5 m a 30 m/s no ar está em
 *
 *     Re = U L / nu = 30 * 4.5 / 1.506e-5 = 9.0e6
 *
 * Uma simulação direta disso resolve turbilhões até a escala de Kolmogorov, o
 * que pede da ordem de Re^(9/4) ~ 1e15 células. Este programa tem, num bom
 * desktop, 5e7. Estamos sete ordens de grandeza curtos, e nenhuma quantidade
 * de GPU fecha esse buraco — nem a sua, nem a de ninguém, nem em dez anos.
 *
 * O que se faz em vez disso — e é o que a CFD automotiva comercial faz — é
 * resolver as estruturas grandes, que carregam a energia e definem o arrasto
 * (os vórtices do pilar-A, a separação no teto, a bolha de recirculação atrás),
 * e deixar um modelo sub-grid Smagorinsky fornecer a dissipação que os
 * turbilhões não resolvidos teriam fornecido. Isso é um método real, padrão e
 * defensável. Não é DNS, e este módulo nunca deixa a interface fingir que é:
 * ele reporta o Reynolds físico E o Reynolds que o lattice de fato resolve, e
 * diz em uma frase quando os dois se separaram.
 *
 * Um app de túnel de vento que mostra "Re = 9.0e6" e nada mais está mentindo
 * por omissão. Este mostra os dois números lado a lado.
 *
 *
 * O TETO DE VISCOSIDADE
 * ---------------------
 * omega -> 2 leva nu -> 0 e o Re resolvido ao infinito, e compra
 * instabilidade junto. Na prática 1.98 é onde os modos fantasma começam a
 * tocar numa malha grossa:
 *
 *     nu_min = c_s^2 (1/1.98 - 1/2) = 1.6835e-3 unidades de lattice
 *
 * ESSE PISO SÓ VALE SE NADA ESTIVER CISALHANDO CONTRA UMA PAREDE PARADA.
 *
 * Com esteira rolante — o piso movendo-se na velocidade da corrente livre,
 * como todo túnel automotivo sério tem — o escoamento uniforme é solução exata
 * e o solver a reproduz até o zero de máquina em omega = 1.98. Pare a esteira
 * e o piso cria uma camada limite cujo primeiro plano de células tem Reynolds
 * de célula u_lb/nu ~ 30, e o bounce-back ali é linearmente instável. Por isso
 * `rollingRoad` é um parâmetro desta classe e não um detalhe de renderização:
 * ligar a esteira compra Reynolds resolvido de verdade.
 */

import { CS2, omegaFromNu, nuFromOmega } from './lattice.js';

/** Ar a 20 °C, 1 atm. */
export const AR = {
  rho: 1.204,       // kg/m^3
  mu: 1.813e-5,     // Pa.s
  nu: 1.506e-5,     // m^2/s
  c: 343.2,         // m/s — para checar o Mach físico
};

/* Onde os modos fantasma começam a tocar. Medido, não deduzido. Baixar isto
 * compra estabilidade e custa Reynolds resolvido, na proporção exata. */
export const OMEGA_MAX = 1.98;
export const OMEGA_MAX_PISO_PARADO = 1.92;

export const NU_MIN = nuFromOmega(OMEGA_MAX);

/* Velocidade de lattice alvo. Ma_lattice = u_lb/c_s = u_lb*sqrt(3); o erro de
 * compressibilidade cresce com Ma^2, então 0.05 (Ma = 0.087) segura o erro em
 * ~0,8% e ainda dá passos de tempo úteis. */
export const U_LB_PADRAO = 0.05;
export const U_LB_MAX = 0.1;      // Ma = 0.17; acima disso o erro é visível

/**
 * Conversão entre unidades físicas e de lattice para uma corrida.
 *
 * A escala é fixada por três escolhas: o comprimento de referência do corpo
 * (quantas células ele ocupa), a velocidade da corrente livre (quantas
 * unidades de lattice ela vale) e a viscosidade — que é a única das três que
 * normalmente não podemos honrar.
 */
export class Units {
  /**
   * @param {object} o
   * @param {number} o.lengthM        comprimento de referência do corpo (m).
   *                                  O mesmo que entra no Re e no Cd.
   * @param {number} o.speedMs        velocidade da corrente livre (m/s)
   * @param {number} o.cellsPerLength quantas células cobrem lengthM
   * @param {number} [o.uLb]          velocidade da corrente livre no lattice
   * @param {boolean} [o.rollingRoad] piso acompanha a corrente livre
   * @param {object} [o.fluid]        propriedades; padrão = ar a 20 °C
   * @param {number} [o.lesCs]        constante de Smagorinsky; 0 desliga
   */
  constructor({
    lengthM, speedMs, cellsPerLength,
    uLb = U_LB_PADRAO, rollingRoad = true, fluid = AR, lesCs = 0.1,
  }) {
    if (!(lengthM > 0)) throw new Error('lengthM deve ser positivo');
    if (!(speedMs > 0)) throw new Error('speedMs deve ser positivo');
    if (!(cellsPerLength >= 8)) {
      throw new Error(
        `cellsPerLength = ${cellsPerLength}: abaixo de ~8 células o corpo não ` +
        'tem forma, só um degrau. Aumente a resolução.');
    }
    if (!(uLb > 0 && uLb <= U_LB_MAX)) {
      throw new Error(
        `uLb = ${uLb} fora de (0, ${U_LB_MAX}]. Ma = uLb*sqrt(3) e o erro de ` +
        'compressibilidade cresce com Ma^2.');
    }

    this.fluid = { ...fluid };
    this.lengthM = lengthM;
    this.speedMs = speedMs;
    this.cellsPerLength = Math.round(cellsPerLength);
    this.uLb = uLb;
    this.rollingRoad = !!rollingRoad;
    this.lesCs = lesCs;

    /* Escalas fundamentais: quanto vale uma célula e um passo. */
    this.dx = lengthM / this.cellsPerLength;          // m por célula
    this.dt = uLb * this.dx / speedMs;                // s por passo

    /* O Reynolds físico — o número real do escoamento. */
    this.rePhysical = speedMs * lengthM / this.fluid.nu;

    /* A viscosidade de lattice que reproduziria esse Re exatamente. */
    this.nuLbExact = uLb * this.cellsPerLength / this.rePhysical;

    /* O teto imposto pela estabilidade. Com o piso parado, mais baixo. */
    this.omegaMax = this.rollingRoad ? OMEGA_MAX : OMEGA_MAX_PISO_PARADO;
    this.nuLbMin = nuFromOmega(this.omegaMax);

    /* A que de fato usamos, e portanto o Re que o lattice resolve. */
    this.nuLb = Math.max(this.nuLbExact, this.nuLbMin);
    this.omegaPlus = omegaFromNu(this.nuLb);
    this.reLattice = uLb * this.cellsPerLength / this.nuLb;

    /* Mach: o do lattice (erro de compressibilidade) e o físico (o escoamento
     * real é incompressível abaixo de ~0.3, e se não for, este solver está
     * resolvendo o problema errado). */
    this.machLattice = uLb / Math.sqrt(CS2);
    this.machPhysical = speedMs / this.fluid.c;
  }

  /* ───────────────────────────────────────────────────────────────── estado */

  /** true quando o lattice honra o Reynolds físico sem ajuda do LES. */
  get resolved() { return this.nuLbExact >= this.nuLbMin; }

  /** Quantas vezes o Re físico excede o que o lattice resolve. */
  get reRatio() { return this.rePhysical / this.reLattice; }

  /** Uma frase curta e honesta sobre o regime desta corrida. */
  get verdict() {
    const p = this.rePhysical.toPrecision(3);
    const l = this.reLattice.toPrecision(3);
    if (this.resolved) {
      return `Re resolvido diretamente (${p}). Sem modelagem sub-grid necessária.`;
    }
    if (!(this.lesCs > 0)) {
      return `Re físico ${p}, lattice resolve ${l} (${this.reRatio.toFixed(0)}x menor) ` +
        'e o LES está DESLIGADO. As escalas não resolvidas não estão sendo ' +
        'modeladas por nada — trate os números como qualitativos.';
    }
    return `Re físico ${p}, lattice resolve ${l} (${this.reRatio.toFixed(0)}x menor). ` +
      'LES Smagorinsky fornece a dissipação das escalas não resolvidas. ' +
      'É LES grosseiro, não DNS.';
  }

  /* ───────────────────────────────────────────────────────────── conversões */

  velToLb(vMs) { return vMs * this.uLb / this.speedMs; }
  velToSi(vLb) { return vLb * this.speedMs / this.uLb; }
  lenToLb(xM) { return xM / this.dx; }
  lenToSi(xLb) { return xLb * this.dx; }
  timeToSi(steps) { return steps * this.dt; }

  /** delta = rho-1 no lattice -> pressão manométrica em Pa. */
  pressureToSi(delta) {
    const scale = this.fluid.rho * (this.dx / this.dt) ** 2;
    return CS2 * delta * scale;
  }

  /** Força em unidades de lattice -> newtons. Escala: rho dx^4 / dt^2. */
  forceToSi(fLb) {
    return fLb * this.fluid.rho * this.dx ** 4 / this.dt ** 2;
  }

  /** q = 1/2 rho U^2, em Pa. O denominador de todo coeficiente. */
  get dynamicPressure() { return 0.5 * this.fluid.rho * this.speedMs ** 2; }

  /* ───────────────────────────────────────────────────────────────── relato */

  /** Objeto plano para a interface. Nada aqui é arredondado a favor. */
  report() {
    return {
      dxM: this.dx,
      dtS: this.dt,
      cellsPerLength: this.cellsPerLength,
      uLb: this.uLb,
      nuLb: this.nuLb,
      nuLbExact: this.nuLbExact,
      omegaPlus: this.omegaPlus,
      omegaMax: this.omegaMax,
      rePhysical: this.rePhysical,
      reLattice: this.reLattice,
      reRatio: this.reRatio,
      resolved: this.resolved,
      machLattice: this.machLattice,
      machPhysical: this.machPhysical,
      rollingRoad: this.rollingRoad,
      lesCs: this.lesCs,
      dynamicPressurePa: this.dynamicPressure,
      verdict: this.verdict,
    };
  }
}

/** Quantos passos de lattice cobrem um intervalo físico. */
export function stepsForSeconds(units, seconds) {
  return Math.ceil(seconds / units.dt);
}

/**
 * Passos para o escoamento atravessar o domínio `n` vezes.
 *
 * A referência útil para "já transitou o suficiente": um corpo precisa de umas
 * 3 travessias antes que a esteira pare de depender da condição inicial, e
 * qualquer Cd lido antes disso é o Cd do transiente de partida.
 */
export function stepsForFlowthrough(units, n = 1) {
  return Math.ceil(n * units.cellsPerLength / units.uLb);
}
