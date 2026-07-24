/* ── src/core/tunel.js ───────────────────────────────────────────────────────
 *
 * Monta o campo de tipos de célula: o corpo mais as paredes do túnel.
 *
 * As escolhas de contorno aqui não são convenções — cada uma responde a uma
 * pergunta sobre que escoamento estamos modelando.
 *
 * PISO SÓLIDO E MÓVEL, TETO ESPELHO. Um carro na estrada tem asfalto embaixo e
 * nada em cima. O piso é parede de verdade, com não-deslizamento; o teto e as
 * laterais são deslizamento livre, porque as paredes de um túnel numérico não
 * existem no problema real e a camada limite que elas criariam estreitaria a
 * seção de teste ao longo do domínio, aumentando o bloqueio progressivamente.
 *
 * ESTEIRA ROLANTE. O piso acompanha a corrente livre. Todo túnel automotivo
 * sério tem isso, e por dois motivos independentes: sem ela o assoalho do
 * carro vê ar que já frenou contra um chão parado, o que é fisicamente errado
 * (na estrada o chão se move em relação ao carro); e numericamente, escoamento
 * uniforme deixa de ser solução exata e o teto de estabilidade cai de
 * omega = 1.98 para 1.92, custando Reynolds resolvido. Ligar a esteira é de
 * graça e melhora as duas coisas.
 *
 * A ORDEM DE APLICAÇÃO. O corpo entra primeiro e as paredes sobrescrevem. Sem
 * isso um modelo mal posicionado — e eles aparecem, porque a escala vem de um
 * palpite de unidade — poderia tapar a entrada ou a saída, e o túnel viraria
 * uma caixa fechada sem que nada avisasse.
 */

import { CELULA } from './emit/ir.js';
import { VOXEL } from '../geom/voxel.js';

/**
 * @param {object} o
 * @param {object} o.grade            { nx, ny, nz }
 * @param {Uint8Array} [o.corpo]      saída de voxelizar().tipo
 * @param {boolean} [o.esteira]       piso acompanha a corrente livre
 * @param {number} [o.fracaoEsponja]  fração final de nx ocupada pela esponja
 * @returns {{tipo: Uint32Array, esponja: {inicio, comprimento}, contagem: object}}
 */
export function montarTipos({
  grade, corpo = null, esteira = true, fracaoEsponja = 0.18,
}) {
  const { nx, ny, nz } = grade;
  const N = nx * ny * nz;
  const tipo = new Uint32Array(N);

  /* 1. o corpo */
  if (corpo) {
    if (corpo.length !== N) {
      throw new Error(`campo do corpo com ${corpo.length}, esperado ${N}`);
    }
    for (let i = 0; i < N; i++) {
      if (corpo[i] !== VOXEL.FLUIDO) tipo[i] = CELULA.SOLIDO;
    }
  }

  /* 2. as paredes, por cima */
  const piso = esteira ? CELULA.SOLIDO_MOVEL : CELULA.SOLIDO;
  const idx = (x, y, z) => (z * ny + y) * nx + x;

  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      tipo[idx(0, y, z)] = CELULA.ENTRADA;
      tipo[idx(nx - 1, y, z)] = CELULA.SAIDA;
    }
  }
  for (let z = 0; z < nz; z++) {
    for (let x = 0; x < nx; x++) {
      tipo[idx(x, 0, z)] = CELULA.ESPELHO_Y;
      tipo[idx(x, ny - 1, z)] = CELULA.ESPELHO_Y;
    }
  }
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      tipo[idx(x, y, 0)] = piso;
      tipo[idx(x, y, nz - 1)] = CELULA.ESPELHO_Z;
    }
  }
  /* Entrada e saída mandam nos cantos: um canto marcado como espelho no plano
   * de entrada seria um pedaço de parede injetando nada, e o perfil de
   * velocidade sairia com um degrau na borda. */
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      tipo[idx(0, y, z)] = CELULA.ENTRADA;
      tipo[idx(nx - 1, y, z)] = CELULA.SAIDA;
    }
  }

  const comprimento = Math.max(4, Math.round(nx * fracaoEsponja));
  const esponja = { inicio: nx - 1 - comprimento, comprimento };

  const contagem = { fluido: 0, solido: 0, solidoMovel: 0, entrada: 0, saida: 0, espelho: 0 };
  for (let i = 0; i < N; i++) {
    switch (tipo[i]) {
      case CELULA.FLUIDO: contagem.fluido++; break;
      case CELULA.SOLIDO: contagem.solido++; break;
      case CELULA.SOLIDO_MOVEL: contagem.solidoMovel++; break;
      case CELULA.ENTRADA: contagem.entrada++; break;
      case CELULA.SAIDA: contagem.saida++; break;
      default: contagem.espelho++;
    }
  }

  return { tipo, esponja, contagem };
}

/**
 * Área frontal do CORPO, em células², contando só a sombra do obstáculo.
 *
 * Distinta de voxel.areaFrontal porque aqui o piso também é sólido e sombreia
 * o domínio inteiro. Um Cd calculado com essa sombra incluída sairia dividido
 * pela seção de teste toda — ou seja, cerca de vinte vezes menor que o real, e
 * ainda assim com uma cara perfeitamente razoável no visor.
 */
export function areaFrontalCorpo(corpo, grade) {
  const { nx, ny, nz } = grade;
  const sombra = new Uint8Array(ny * nz);
  /* z começa em 1: a linha do piso não é corpo */
  for (let z = 1; z < nz - 1; z++) {
    for (let y = 1; y < ny - 1; y++) {
      const base = (z * ny + y) * nx;
      for (let x = 1; x < nx - 1; x++) {
        if (corpo[base + x] !== VOXEL.FLUIDO) { sombra[z * ny + y] = 1; break; }
      }
    }
  }
  let n = 0;
  for (let i = 0; i < sombra.length; i++) n += sombra[i];
  return n;
}

/**
 * Gera um obstáculo analítico no lattice — esfera ou cilindro.
 *
 * Existe para a suíte de validação, que precisa de geometria EXATA: comparar
 * um Cd medido contra uma correlação de esfera só faz sentido se o corpo for
 * uma esfera de verdade, e não uma esfera passada por um importador de malha,
 * um palpite de eixo e um palpite de unidade. Aqui o raio é o que dizemos que
 * é, e qualquer discrepância no resultado é do solver.
 */
export function corpoAnalitico(grade, forma) {
  const { nx, ny, nz } = grade;
  const N = nx * ny * nz;
  const corpo = new Uint8Array(N);
  const { tipo: f, cx, cy, cz, raio } = forma;

  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const dx = x + 0.5 - cx, dy = y + 0.5 - cy, dz = z + 0.5 - cz;
        let dentro;
        if (f === 'esfera') {
          dentro = dx * dx + dy * dy + dz * dz <= raio * raio;
        } else if (f === 'cilindro') {
          /* eixo em y: um cilindro atravessando o túnel de lado a lado */
          dentro = dx * dx + dz * dz <= raio * raio;
        } else {
          throw new Error(`forma desconhecida: ${f}`);
        }
        if (dentro) corpo[(z * ny + y) * nx + x] = VOXEL.DENTRO;
      }
    }
  }
  return corpo;
}
