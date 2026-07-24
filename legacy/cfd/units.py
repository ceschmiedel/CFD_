"""
units.py — a ponte entre metros por segundo e unidades de lattice, e o lugar
           onde este programa é honesto sobre o que consegue e o que não
           consegue resolver.

A ARITMÉTICA DESCONFORTÁVEL
---------------------------
Um carro de 4,5 m a 30 m/s no ar está em

    Re = U L / nu = 30 * 4.5 / 1.516e-5 = 8.9e6

Uma simulação direta disso resolve turbilhões até a escala de Kolmogorov, o que
pede da ordem de Re^(9/4) ~ 1e15 células. Esta máquina tem 3e7. Estamos oito
ordens de grandeza curtos, e nenhuma quantidade de GPU fecha esse buraco.

O que se faz em vez disso — e é o que a CFD automotiva comercial faz — é
resolver as estruturas grandes, que carregam a energia e definem o arrasto (os
vórtices do pilar-A, a separação no teto, a bolha de recirculação atrás), e
deixar um modelo sub-grid Smagorinsky fornecer a dissipação que os turbilhões
não resolvidos teriam fornecido. Isso é um método real, padrão e defensável.
Não é DNS, e este módulo nunca deixa a interface fingir que é: ele reporta o
número de Reynolds físico E o número de Reynolds que o lattice de fato resolve,
e diz quando os dois se separaram.

O TETO DE VISCOSIDADE
---------------------
omega -> 2 leva nu -> 0 e o Re resolvido ao infinito, e compra instabilidade
junto. Na prática 1.98 é onde os modos fantasma começam a tocar numa malha
grossa:

    nu_min = c_s^2 (1/1.98 - 1/2) = 1.6835e-3 unidades de lattice

ESSE PISO SÓ VALE SE NADA ESTIVER CISALHANDO CONTRA UMA PAREDE PARADA.

Com esteira rolante — o piso movendo-se na velocidade da corrente livre, como
todo túnel automotivo sério tem — o escoamento uniforme é solução exata e o
solver a reproduz até o zero de máquina em omega = 1.98. Pare a esteira e o
piso cria uma camada limite cujo primeiro plano de células tem Reynolds de
célula u_lb/nu ~ 30, e o bounce-back ali é linearmente instável. Por isso
`rolling_road` é um argumento desta função e não um detalhe de renderização.
"""

import math

from .lattice import CS2, omega_from_nu, nu_from_omega

# Ar a 20 °C, 1 atm.
AR = {
    'rho': 1.204,        # kg/m^3
    'mu': 1.813e-5,      # Pa.s
    'nu': 1.506e-5,      # m^2/s
    'c': 343.2,          # m/s (velocidade do som, para checar Mach físico)
}

# Onde os modos fantasma começam a tocar. Medido, não deduzido — ver o
# comentário de topo. Reduzir isto compra estabilidade e custa Re resolvido.
OMEGA_MAX = 1.98
OMEGA_MAX_PAREDE_PARADA = 1.92    # sem esteira, o piso cisalha e o teto cai

NU_MIN = nu_from_omega(OMEGA_MAX)

# Velocidade de lattice alvo. Ma_lattice = u_lb / c_s = u_lb * sqrt(3); o erro
# de compressibilidade cresce com Ma^2, então 0.05 (Ma = 0.087) mantém o erro
# em ~0,8% e ainda dá passos de tempo úteis.
U_LB_PADRAO = 0.05
U_LB_MAX = 0.1                     # Ma = 0.17; acima disso o erro é visível


class Units:
    """Conversão entre unidades físicas e de lattice para uma dada corrida.

    A escala é fixada por três escolhas: o comprimento de referência do corpo
    (quantas células ele ocupa), a velocidade da corrente livre (quantas
    unidades de lattice ela vale), e a viscosidade — que é a única das três que
    normalmente não podemos honrar.
    """

    def __init__(self, length_m, speed_ms, cells_per_length,
                 u_lb=U_LB_PADRAO, rolling_road=True, fluid=None,
                 les_cs=0.1):
        """
        Args:
            length_m: comprimento de referência do corpo, em metros
                      (o mesmo que entra no Re e no Cd)
            speed_ms: velocidade da corrente livre, em m/s
            cells_per_length: quantas células de lattice cobrem `length_m`
            u_lb: velocidade da corrente livre em unidades de lattice
            rolling_road: se o piso acompanha a corrente livre (esteira)
            fluid: dicionário de propriedades; padrão = ar a 20 °C
            les_cs: constante de Smagorinsky; 0.0 desliga o modelo sub-grid
        """
        if length_m <= 0:
            raise ValueError('length_m deve ser positivo')
        if speed_ms <= 0:
            raise ValueError('speed_ms deve ser positivo')
        if cells_per_length < 8:
            raise ValueError(
                f'cells_per_length = {cells_per_length}: abaixo de ~8 células '
                'o corpo não tem forma, só um degrau. Aumente a resolução.')
        if not 0 < u_lb <= U_LB_MAX:
            raise ValueError(
                f'u_lb = {u_lb} fora de (0, {U_LB_MAX}]. Ma = u_lb*sqrt(3) e '
                'o erro de compressibilidade cresce com Ma^2.')

        self.fluid = dict(fluid or AR)
        self.length_m = float(length_m)
        self.speed_ms = float(speed_ms)
        self.cells_per_length = int(cells_per_length)
        self.u_lb = float(u_lb)
        self.rolling_road = bool(rolling_road)
        self.les_cs = float(les_cs)

        # Escalas fundamentais: quanto vale uma célula e um passo de tempo.
        self.dx = self.length_m / self.cells_per_length          # m / célula
        self.dt = self.u_lb * self.dx / self.speed_ms            # s / passo

        # Reynolds físico — o número real do escoamento.
        self.re_physical = self.speed_ms * self.length_m / self.fluid['nu']

        # A viscosidade de lattice que reproduziria esse Re exatamente.
        self.nu_lb_exact = self.u_lb * self.cells_per_length / self.re_physical

        # O teto imposto pela estabilidade. Com o piso parado, mais baixo.
        self.omega_max = (OMEGA_MAX if self.rolling_road
                          else OMEGA_MAX_PAREDE_PARADA)
        self.nu_lb_min = nu_from_omega(self.omega_max)

        # A que de fato usamos, e portanto o Re que o lattice resolve.
        self.nu_lb = max(self.nu_lb_exact, self.nu_lb_min)
        self.omega_plus = omega_from_nu(self.nu_lb)
        self.re_lattice = self.u_lb * self.cells_per_length / self.nu_lb

        # Mach: o do lattice (erro de compressibilidade) e o físico (o
        # escoamento real é incompressível abaixo de ~0.3).
        self.mach_lattice = self.u_lb / math.sqrt(CS2)
        self.mach_physical = self.speed_ms / self.fluid['c']

    # ------------------------------------------------------------------ estado

    @property
    def resolved(self):
        """True quando o lattice honra o Reynolds físico sem ajuda do LES."""
        return self.nu_lb_exact >= self.nu_lb_min

    @property
    def re_ratio(self):
        """Quantas vezes o Re físico excede o que o lattice resolve."""
        return self.re_physical / self.re_lattice

    @property
    def verdict(self):
        """Uma frase curta e honesta sobre o regime desta corrida."""
        if self.resolved:
            return (f'Re resolvido diretamente ({self.re_physical:.3g}). '
                    'Sem modelagem sub-grid necessária.')
        if self.les_cs <= 0:
            return (f'Re físico {self.re_physical:.3g}, lattice resolve '
                    f'{self.re_lattice:.3g} ({self.re_ratio:.0f}x menor) e o '
                    'LES está DESLIGADO. As escalas não resolvidas não estão '
                    'sendo modeladas por nada — trate os números como '
                    'qualitativos.')
        return (f'Re físico {self.re_physical:.3g}, lattice resolve '
                f'{self.re_lattice:.3g} ({self.re_ratio:.0f}x menor). '
                'LES Smagorinsky fornece a dissipação das escalas não '
                'resolvidas. É LES grosseiro, não DNS.')

    # -------------------------------------------------------------- conversões

    def vel_to_lb(self, v_ms):
        """m/s -> unidades de lattice."""
        return v_ms * self.u_lb / self.speed_ms

    def vel_to_si(self, v_lb):
        """unidades de lattice -> m/s."""
        return v_lb * self.speed_ms / self.u_lb

    def len_to_lb(self, x_m):
        """metros -> células."""
        return x_m / self.dx

    def len_to_si(self, x_lb):
        """células -> metros."""
        return x_lb * self.dx

    def time_to_si(self, steps):
        """passos de lattice -> segundos."""
        return steps * self.dt

    def pressure_to_si(self, delta):
        """delta = rho-1 no lattice -> pressão manométrica em Pa.

        p_lb = c_s^2 * delta, e a escala de pressão é rho_fisico * (dx/dt)^2.
        """
        scale = self.fluid['rho'] * (self.dx / self.dt) ** 2
        return CS2 * delta * scale

    def force_to_si(self, f_lb):
        """força em unidades de lattice -> newtons (por profundidade completa).

        A escala de força é rho * dx^4 / dt^2.
        """
        return f_lb * self.fluid['rho'] * self.dx ** 4 / self.dt ** 2

    @property
    def dynamic_pressure(self):
        """q = 1/2 rho U^2, em Pa. O denominador de todo coeficiente."""
        return 0.5 * self.fluid['rho'] * self.speed_ms ** 2

    # ------------------------------------------------------------------ relato

    def report(self):
        """Dicionário plano para a interface. Nada aqui é arredondado a favor."""
        return {
            'dx_m': self.dx,
            'dt_s': self.dt,
            'cells_per_length': self.cells_per_length,
            'u_lb': self.u_lb,
            'nu_lb': self.nu_lb,
            'nu_lb_exact': self.nu_lb_exact,
            'omega_plus': self.omega_plus,
            'omega_max': self.omega_max,
            're_physical': self.re_physical,
            're_lattice': self.re_lattice,
            're_ratio': self.re_ratio,
            'resolved': self.resolved,
            'mach_lattice': self.mach_lattice,
            'mach_physical': self.mach_physical,
            'rolling_road': self.rolling_road,
            'les_cs': self.les_cs,
            'dynamic_pressure_pa': self.dynamic_pressure,
            'verdict': self.verdict,
        }

    def __repr__(self):
        return (f'<Units L={self.length_m:g}m U={self.speed_ms:g}m/s '
                f'N={self.cells_per_length} omega={self.omega_plus:.4f} '
                f'Re_fis={self.re_physical:.3g} Re_lat={self.re_lattice:.3g}>')


def steps_for_seconds(units, seconds):
    """Quantos passos de lattice cobrem um intervalo físico."""
    return int(math.ceil(seconds / units.dt))


def steps_for_flowthrough(units, domain_lengths=1.0):
    """Passos para o escoamento atravessar o domínio `domain_lengths` vezes.

    A referência útil para "já transitou o suficiente": um corpo precisa de
    ~3 travessias antes que a esteira pare de depender da condição inicial.
    """
    return int(math.ceil(domain_lengths * units.cells_per_length / units.u_lb))
