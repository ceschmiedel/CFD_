/* ── src/core/validacao.js ───────────────────────────────────────────────────
 *
 * As correlações contra as quais este solver é medido.
 *
 * Um app de túnel de vento sem isto é uma animação. As funções abaixo são
 * ajustes experimentais publicados, e a suíte que as usa (tests/validacao.html)
 * roda dentro do app: quem abre o link pode conferir os números por conta
 * própria, o que é o oposto de pedir confiança.
 *
 * Todas trazem a faixa de Re em que valem, porque uma correlação usada fora da
 * faixa dá um número perfeitamente formatado e sem significado nenhum.
 */

/**
 * Arrasto de uma esfera lisa. Ajuste padrão de Clift-Gauvin, válido de Stokes
 * até um pouco antes da crise de arrasto:
 *
 *     Cd = 24/Re (1 + 0.15 Re^0.687) + 0.42 / (1 + 4.25e4 Re^-1.16)
 *
 * Acima de Re ~ 3e5 a camada limite transiciona para turbulenta, o ponto de
 * separação salta para trás e o Cd DESPENCA de 0,5 para 0,1 — a crise de
 * arrasto. Esta correlação não a descreve, e nenhum LES grosseiro a
 * reproduziria de qualquer forma: é um fenômeno de camada limite que exige
 * resolver a subcamada viscosa.
 */
export function esferaCd(re) {
  if (!(re > 0)) return Infinity;
  if (re < 0.1) return 24 / re;                                  // Stokes
  return (24 / re) * (1 + 0.15 * Math.pow(re, 0.687)) +
    0.42 / (1 + 4.25e4 * Math.pow(re, -1.16));
}

/** Faixa em que esferaCd é confiável. */
export const ESFERA_FAIXA = [0.1, 2e5];

/**
 * Arrasto de um cilindro circular longo em escoamento transversal.
 * Ajuste por partes da curva padrão.
 */
export function cilindroCd(re) {
  if (!(re > 0)) return Infinity;
  if (re < 1) return 8 * Math.PI / (re * (2.002 - Math.log(re)));  // Lamb
  if (re < 2e5) return 1.0 + 10 * Math.pow(re, -2 / 3);
  return 0.35;                                                    // pós-crise
}

/**
 * Strouhal do desprendimento de vórtices atrás de um cilindro.
 *
 * Abaixo de Re ~ 47 a esteira é estacionária e NÃO HÁ desprendimento — a
 * função devolve 0, e isso é uma afirmação física, não um caso de borda: se o
 * solver produzir oscilação abaixo desse Re, a oscilação é numérica.
 *
 * Williamson & Brown (1998) na faixa laminar; Roshko (1954) acima.
 */
export function cilindroStrouhal(re) {
  if (re < 47) return 0;
  if (re < 300) return 0.2684 - 1.0356 / Math.sqrt(re);
  if (re < 2e5) return 0.198 * (1 - 19.7 / re);
  return 0.21;
}

/**
 * Perfil de Poiseuille entre placas planas, adimensionalizado.
 * u(y)/u_max = 1 - (2y/H - 1)^2 com y de 0 a H.
 *
 * Usado para medir a viscosidade EFETIVA que o bounce-back produz — que é o
 * teste que revela se a parede está onde o TRT diz que está. Com Lambda = 3/16
 * ela fica no meio-caminho entre nós e o perfil bate com H = distância entre
 * as paredes efetivas; com outro Lambda, a viscosidade aparente muda com a
 * própria viscosidade, que é o defeito que o TRT existe para curar.
 */
export function poiseuille(y, H) {
  const s = 2 * y / H - 1;
  return Math.max(0, 1 - s * s);
}

/**
 * Compara um valor medido com o de referência e classifica.
 *
 * Os limiares não são arbitrários. 5% é o que um túnel físico bem operado
 * entrega repetindo a mesma medida; 15% é o que um LES grosseiro num lattice
 * de dezenas de células por comprimento consegue honestamente. Além disso, o
 * resultado é qualitativo — e dizer isso é mais útil que pintar de vermelho.
 */
export function classificar(medido, referencia) {
  /* `pct` sai preenchido em TODOS os caminhos, inclusive neste. A versão
   * anterior omitia o campo aqui, e a suíte inteira morria num
   * "cannot read properties of undefined" ao formatar o primeiro caso que
   * divergisse — trocando um resultado ruim, que é informação, por uma página
   * congelada, que não é. */
  if (!Number.isFinite(medido) || !Number.isFinite(referencia) || referencia === 0) {
    return { erro: NaN, pct: NaN, nivel: 'indefinido', ok: false };
  }
  const erro = (medido - referencia) / referencia;
  const a = Math.abs(erro) * 100;
  if (a <= 5) return { erro, pct: a, nivel: 'excelente', ok: true };
  if (a <= 15) return { erro, pct: a, nivel: 'aceitável', ok: true };
  if (a <= 30) return { erro, pct: a, nivel: 'qualitativo', ok: false };
  return { erro, pct: a, nivel: 'discordante', ok: false };
}

/**
 * Os casos de validação, com os parâmetros que os tornam reprodutíveis.
 *
 * Re é escolhido baixo de propósito. Não é para o caso ser rápido — é para ele
 * ser RESOLVIDO: a Re = 100 o lattice honra a viscosidade física sem ajuda do
 * LES (ver units.js), e o número que sai mede o solver, não o modelo sub-grid.
 * Validar contra uma correlação num regime que o lattice não resolve mediria a
 * calibração do Smagorinsky, que não é o que está em questão.
 */
/**
 * Escolhe a velocidade de lattice a partir de um ω ALVO, e não o contrário.
 *
 * O caminho ingênuo é fixar u_lb = 0.05 e deixar ω cair onde cair:
 * ν_lb = u_lb·D/Re, e ω = 1/(3ν_lb + 1/2). Para uma esfera a Re = 200 com
 * D = 24 isso dá ω = 1,93 — encostado no teto de 1,98 onde os modos fantasma
 * começam a tocar. E aí o caso não devolve um Cd ruim: devolve NaN, porque o
 * solver diverge, e a suíte inteira precisa lidar com isso.
 *
 * Fixando ω em 1,90 e derivando u_lb = ν(ω)·Re/D, a estabilidade fica sob
 * controle em todos os casos e o preço é um Mach um pouco maior — que é o
 * lado certo do compromisso, porque o erro de compressibilidade é suave e
 * previsível enquanto a instabilidade não é.
 */
export function uLbParaOmega(re, dCelulas, omegaAlvo = 1.90, uLbMax = 0.1) {
  const nu = (1 / 3) * (1 / omegaAlvo - 0.5);
  return Math.min(nu * re / dCelulas, uLbMax);
}

/*
 * Os casos.
 *
 * Cada um traz o próprio domínio, e essa foi a lição da primeira rodada. Rodar
 * os três num domínio comum de 240×120×96 pareceu econômico e produziu dois
 * resultados errados por dois motivos diferentes:
 *
 * BLOQUEIO. Um cilindro atravessa o túnel de lado a lado, então o bloqueio
 * dele é D/altura — 20/96, ou 21%. Nenhuma correção é confiável aí, e o Cd
 * saiu 38% abaixo da correlação. Um cilindro é um problema essencialmente
 * BIDIMENSIONAL, então a saída é um domínio fino no vão e alto na seção:
 * 480 × 8 × 320 tem 1,2 M células — menos que o domínio "comum" — e derruba o
 * bloqueio para 5%.
 *
 * ESTABILIDADE. A esfera a Re = 200 divergiu no passo 6868. Com u_lb limitado
 * a 0,1 por Mach e D = 24, ω não desce de 1,89 nesse Re — colado no teto. A
 * saída é mais células no corpo (ν_lb = u_lb·D/Re cresce com D), o que pede um
 * domínio maior para o bloqueio não subir junto. O caso fica marcado como
 * `profundo`: ele é real, custa uns dois minutos, e a suíte rápida não o roda.
 */
export const CASOS = [
  {
    id: 'esfera-re50',
    nome: 'Esfera, Re = 50',
    forma: 'esfera', re: 50, diametroCelulas: 24,
    grade: { nx: 240, ny: 120, nz: 96 },
    referencia: () => esferaCd(50),
    travessiasTransiente: 2.0, travessiasMedida: 1.0,
  },
  {
    id: 'esfera-re100',
    nome: 'Esfera, Re = 100',
    forma: 'esfera', re: 100, diametroCelulas: 24,
    grade: { nx: 240, ny: 120, nz: 96 },
    referencia: () => esferaCd(100),
    travessiasTransiente: 2.0, travessiasMedida: 1.5,
  },
  {
    id: 'cilindro-re100',
    nome: 'Cilindro, Re = 100',
    forma: 'cilindro', re: 100, diametroCelulas: 16,
    /* fino no vão (o problema é 2D), alto na seção (bloqueio 5%) */
    grade: { nx: 480, ny: 8, nz: 320 },
    referencia: () => cilindroCd(100),
    referenciaSt: () => cilindroStrouhal(100),
    /*
     * O DESPRENDIMENTO PRECISA SER PROVOCADO.
     *
     * A Re = 100 a esteira simétrica de um cilindro é uma solução válida das
     * equações e é INSTÁVEL — na natureza qualquer perturbação a destrói e
     * nascem os vórtices de von Kármán. Num solver perfeitamente simétrico,
     * alimentado por uma entrada perfeitamente uniforme, não há perturbação
     * nenhuma, e a solução simétrica persiste para sempre.
     *
     * O sintoma na primeira rodada foi inequívoco: Cd com desvio-padrão de
     * 1,8e-5 — ou seja, rigorosamente constante — e um Strouhal de 0,083
     * contra 0,165 esperado, exatamente metade, que era o contador de
     * cruzamentos pegando ruído numérico em vez de um sinal.
     *
     * Inclinamos a entrada 2° por umas poucas centenas de passos e depois
     * endireitamos. A perturbação some; a assimetria que ela semeou cresce.
     */
    perturbacao: { grausGuinada: 2.0, passos: 600 },
    travessiasTransiente: 2.5, travessiasMedida: 4.0,
  },
  {
    id: 'esfera-re200',
    nome: 'Esfera, Re = 200',
    forma: 'esfera', re: 200, diametroCelulas: 40,
    grade: { nx: 480, ny: 240, nz: 192 },
    referencia: () => esferaCd(200),
    travessiasTransiente: 2.0, travessiasMedida: 1.5,
    profundo: true,
  },
];
