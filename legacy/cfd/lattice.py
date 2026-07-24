"""
lattice.py — o conjunto de velocidades D3Q19 e as distribuições deslocadas.

CONVENÇÃO DE ARMAZENAMENTO
--------------------------
Este solver NÃO armazena f_i. Armazena o desvio do repouso:

    g_i = f_i - w_i

O motivo é aritmética de ponto flutuante, e ele decide se o mapa de pressão
sai limpo ou sai granulado.

A densidade é rho = 1 + delta, e nos números de Mach em que rodamos delta é da
ordem de 1e-4. A pressão que pintamos na superfície do corpo é p = c_s^2 * delta.
Se armazenamos f_i — cada um por volta de 0.05, o de repouso por volta de 0.33 —
e somamos dezenove deles em fp32, o erro absoluto da soma fica em torno de 2e-7.
Contra um delta de 1e-4 isso é 0,2% de erro, o que aparece como ruído visível no
campo de Cp. Armazenando g_i, cujos valores já são da ordem do desvio, a mesma
soma erra por volta de 1e-9.

Todas as identidades sobrevivem ao deslocamento porque sum(w_i) = 1 e
sum(w_i * c_i) = 0:

    delta   = sum_i g_i                     (rho = 1 + delta)
    rho * u = sum_i g_i * c_i               (exatamente; os termos w_i c_i cancelam)
    g_i^eq  = w_i * (delta + rho * (3(c.u) + 4.5(c.u)^2 - 1.5 u.u))

E o bounce-back continua sendo g_opp(i) <- g_i, porque w_opp(i) = w_i.

Efeito colateral agradável: o estado de repouso é g_i = 0 em toda parte, então
zerar um tensor inicializa o fluido parado com rho = 1 exatamente.

CUIDADO — onde o deslocamento NÃO cancela
-----------------------------------------
Na troca de momento (forces.py) a contribuição de um link é

    c_i * [ f_i + f_opp ] = c_i * [ g_i + g_opp + 2 w_i ]

O termo 2 w_i c_i não cancela por link (só cancelaria somado sobre pares
opostos, e um link é um só). Quem calcula força precisa recolocá-lo à mão.
"""

import numpy as np

# Número de direções e velocidade do som ao quadrado do lattice.
Q = 19
CS2 = 1.0 / 3.0
INV_CS2 = 3.0
INV_CS4 = 9.0

# c_i em pares opostos adjacentes: (0), (1,2), (3,4), ... Isso torna o índice
# oposto uma expressão fechada — ver OPP abaixo — em vez de uma tabela de busca.
C = np.array([
    (0, 0, 0),      # 0   repouso

    (1, 0, 0),      # 1   +x
    (-1, 0, 0),     # 2   -x
    (0, 1, 0),      # 3   +y
    (0, -1, 0),     # 4   -y
    (0, 0, 1),      # 5   +z
    (0, 0, -1),     # 6   -z

    (1, 1, 0),      # 7
    (-1, -1, 0),    # 8
    (1, -1, 0),     # 9
    (-1, 1, 0),     # 10
    (1, 0, 1),      # 11
    (-1, 0, -1),    # 12
    (1, 0, -1),     # 13
    (-1, 0, 1),     # 14
    (0, 1, 1),      # 15
    (0, -1, -1),    # 16
    (0, 1, -1),     # 17
    (0, -1, 1),     # 18
], dtype=np.int8)

# Pesos do D3Q19: 1/3 no repouso, 1/18 nos eixos, 1/36 nas diagonais de face.
W = np.array(
    [1.0 / 3.0] +
    [1.0 / 18.0] * 6 +
    [1.0 / 36.0] * 12,
    dtype=np.float64,
)

# Índice oposto. Com os pares adjacentes, opp(0) = 0 e opp(i) = i XOR 1 para
# i >= 1 — que é exatamente "i+1 se i é ímpar, i-1 se é par".
OPP = np.array([0] + [((i - 1) ^ 1) + 1 for i in range(1, Q)], dtype=np.int64)

# Valores mágicos de Lambda para o operador TRT (ver lbm.py).
#
#   Lambda = (1/omega_plus - 1/2) * (1/omega_minus - 1/2)
#
# WALL         3/16  põe uma parede reta de bounce-back exatamente no meio do
#                    caminho entre nós, independente da viscosidade
#                    (Ginzburg & Adler 1994). O arrasto depende de onde a
#                    parede *pensa* que está, então este é o que importa aqui.
# STABLE       1/4   o mais estável em tau = 1. Não em geral.
# THIRD_ORDER  1/12  cancela o erro espacial de terceira ordem dominante.
# FOURTH_ORDER 1/6   cancela o de quarta ordem (melhor advecção pura).
MAGIC = {
    'WALL': 3.0 / 16.0,
    'STABLE': 1.0 / 4.0,
    'THIRD_ORDER': 1.0 / 12.0,
    'FOURTH_ORDER': 1.0 / 6.0,
}


def omega_from_nu(nu):
    """Taxa de relaxação simétrica a partir da viscosidade cinemática (lattice).

    nu = c_s^2 (1/omega - 1/2)  =>  omega = 1 / (nu/c_s^2 + 1/2)
    """
    return 1.0 / (nu / CS2 + 0.5)


def nu_from_omega(omega):
    """Viscosidade cinemática (lattice) a partir da taxa de relaxação."""
    return CS2 * (1.0 / omega - 0.5)


def omega_minus_from_lambda(omega_plus, lam):
    """Taxa de relaxação ímpar que produz o Lambda pedido.

    Lambda = (1/omega_plus - 1/2)(1/omega_minus - 1/2)
    """
    tau_minus_half = lam / (1.0 / omega_plus - 0.5)
    return 1.0 / (tau_minus_half + 0.5)


def lambda_from_omegas(omega_plus, omega_minus):
    """O inverso: o parâmetro mágico implicado por um par de taxas."""
    return (1.0 / omega_plus - 0.5) * (1.0 / omega_minus - 0.5)


def equilibrium(delta, u, w=None, c=None):
    """Distribuições de equilíbrio deslocadas, em numpy (referência/testes).

    O caminho quente vive em lbm.py e opera sobre tensores torch; esta versão
    existe para os testes poderem checar as identidades de momento sem GPU.

    Args:
        delta: array (...,) com rho - 1
        u: array (..., 3) com a velocidade em unidades de lattice

    Returns:
        array (..., Q) com g_i^eq
    """
    w = W if w is None else w
    c = C if c is None else c

    delta = np.asarray(delta, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    rho = 1.0 + delta

    cu = u @ c.T.astype(np.float64)                       # (..., Q)
    uu = np.sum(u * u, axis=-1)[..., None]                # (..., 1)

    return w * (delta[..., None] +
                rho[..., None] * (3.0 * cu + 4.5 * cu * cu - 1.5 * uu))


def moments(g, c=None):
    """Momentos hidrodinâmicos a partir das distribuições deslocadas.

    Returns:
        (delta, u) com delta = rho - 1 e u a velocidade de lattice.
    """
    c = C if c is None else c
    g = np.asarray(g, dtype=np.float64)

    delta = np.sum(g, axis=-1)
    rho_u = g @ c.astype(np.float64)                      # (..., 3)
    return delta, rho_u / (1.0 + delta)[..., None]


def _self_check():
    """Invariantes do conjunto de velocidades. Rodam na importação em testes."""
    assert C.shape == (Q, 3)
    assert W.shape == (Q,)
    assert abs(W.sum() - 1.0) < 1e-15, 'sum(w_i) deve ser 1'
    assert np.allclose(W[:, None] * C, 0.0, atol=1e-15) or \
        np.allclose((W[:, None] * C).sum(axis=0), 0.0, atol=1e-15), \
        'sum(w_i c_i) deve ser 0'
    # segundo momento: sum(w_i c_ia c_ib) = c_s^2 delta_ab
    m2 = np.einsum('i,ia,ib->ab', W, C.astype(float), C.astype(float))
    assert np.allclose(m2, CS2 * np.eye(3), atol=1e-15), \
        'sum(w_i c_i c_i) deve ser c_s^2 I'
    # oposição é uma involução e nega a velocidade
    assert np.array_equal(OPP[OPP], np.arange(Q))
    assert np.array_equal(C[OPP], -C)
    assert np.array_equal(W[OPP], W)


_self_check()
