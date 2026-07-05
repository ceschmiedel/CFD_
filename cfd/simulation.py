"""
Módulo de simulação CFD usando FiPy
Contém classes e funções para simulações de dinâmica de fluidos computacional
"""

import numpy as np
import torch
import time
from fipy import CellVariable, FaceVariable, Grid2D, TransientTerm, DiffusionTerm, ConvectionTerm

class CFDSimulation:
    """Classe principal para simulações CFD usando FiPy"""
    
    def __init__(self, domain_size=(4.0, 2.0), resolution=(40, 20), device=None,
                 object_mask=None):
        """
        Inicializa simulação CFD

        Args:
            domain_size: Tupla (Lx, Ly) com dimensões do domínio
            resolution: Tupla (nx, ny) com resolução da malha
            device: Dispositivo PyTorch (cuda/cpu) para otimizações
            object_mask: Máscara booleana (nx*ny) da geometria imersa;
                         se None, usa aerofólio NACA de fallback
        """
        self.Lx, self.Ly = domain_size
        self.nx, self.ny = resolution
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.external_mask = None
        if object_mask is not None:
            self.set_object_mask(object_mask)
        
        # Configurar dispositivo
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Criar malha FiPy
        self.mesh = Grid2D(dx=self.dx, dy=self.dy, nx=self.nx, ny=self.ny)
        
        # Variáveis CFD
        self.u = None  # Velocidade em x
        self.v = None  # Velocidade em y
        self.p = None  # Pressão
        
        # Propriedades do fluido
        self.rho = 1.225  # Densidade do ar (kg/m³)
        self.mu = 1.8e-5  # Viscosidade dinâmica (Pa·s)
        self.nu = self.mu / self.rho  # Viscosidade cinemática
        
        # Parâmetros de simulação
        self.dt = 0.001  # Passo de tempo
        self.inlet_velocity = 30.0  # Velocidade de entrada
        
        # Histórico de convergência
        self.residuals = []
        
    def initialize_variables(self):
        """Inicializa variáveis CFD"""
        self.u = CellVariable(name="velocidade_x", mesh=self.mesh, value=self.inlet_velocity)
        self.v = CellVariable(name="velocidade_y", mesh=self.mesh, value=0.0)
        self.p = CellVariable(name="pressao", mesh=self.mesh, value=0.0)
        # Campo vetorial de velocidade nas faces (coeficiente de convecção)
        self.velocity = FaceVariable(name="velocidade", mesh=self.mesh, rank=1)
        self._update_face_velocity()

    def _update_face_velocity(self):
        """Atualiza o campo vetorial de faces a partir de u e v"""
        self.velocity[0] = self.u.arithmeticFaceValue
        self.velocity[1] = self.v.arithmeticFaceValue
        
    def apply_boundary_conditions(self):
        """Aplica condições de contorno"""
        # Entrada (esquerda): velocidade constante
        self.u.constrain(self.inlet_velocity, where=self.mesh.facesLeft)
        self.v.constrain(0.0, where=self.mesh.facesLeft)
        
        # Saída (direita): gradiente zero
        self.u.faceGrad.constrain(0.0, where=self.mesh.facesRight)
        self.v.faceGrad.constrain(0.0, where=self.mesh.facesRight)
        
        # Paredes (topo e fundo): no-slip
        self.u.constrain(0.0, where=self.mesh.facesTop | self.mesh.facesBottom)
        self.v.constrain(0.0, where=self.mesh.facesTop | self.mesh.facesBottom)

        # Pressão: referência (gauge zero) na saída; demais contornos com
        # gradiente nulo (condição natural do FiPy)
        self.p.constrain(0.0, where=self.mesh.facesRight)
        
    def add_obstacle(self, center=(1.0, 1.0), radius=0.3):
        """
        Adiciona obstáculo circular ao domínio
        
        Args:
            center: Centro do obstáculo (x, y)
            radius: Raio do obstáculo
        """
        x, y = self.mesh.cellCenters
        cx, cy = center
        
        # Máscara para células dentro do obstáculo
        obstacle_mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        
        # Aplicar condição no-slip no obstáculo
        self.u.setValue(0.0, where=obstacle_mask)
        self.v.setValue(0.0, where=obstacle_mask)
        
        return obstacle_mask
    
    def set_object_mask(self, mask):
        """
        Define a máscara de células sólidas a partir da geometria real
        (gerada por GeometryProcessor.build_occupancy_mask).

        Args:
            mask: Array booleano com nx*ny elementos (ordem FiPy, x rápido)
        """
        mask = np.asarray(mask).astype(bool).ravel()
        if mask.size != self.nx * self.ny:
            raise ValueError(
                f"Máscara com {mask.size} células; esperado {self.nx * self.ny} "
                f"({self.nx}x{self.ny})")
        self.external_mask = mask

    def create_object_mask(self):
        """
        Retorna a máscara do objeto imerso: usa a geometria real carregada
        quando disponível, senão um aerofólio NACA 0012 de fallback.
        """
        if self.external_mask is not None:
            return self.external_mask

        x = self.mesh.cellCenters[0].value
        y = self.mesh.cellCenters[1].value

        # Criar obstáculo mais realista (formato de aerofólio)
        center_x, center_y = 1.5, 1.0
        
        # Forma de aerofólio NACA aproximada (vetorizado sobre as células)
        dx = np.abs(x - center_x)
        dy = np.abs(y - center_y)

        # Meia-espessura NACA 00xx (fórmula padrão, com fator 5) avaliada na
        # posição relativa da meia-corda e escalada pela corda de 0.8 m
        t = 0.12       # 12% de espessura (NACA 0012)
        chord = 0.8    # corda total do obstáculo (m)
        xi = np.clip(dx / (chord / 2), 0.0, 1.0)
        thickness = 5 * t * chord * (0.2969 * np.sqrt(xi) -
                                     0.1260 * xi -
                                     0.3516 * xi**2 +
                                     0.2843 * xi**3 -
                                     0.1015 * xi**4)

        mask = (dx < chord / 2) & (dy < thickness)
        return mask
    
    def initialize_potential_flow(self, object_mask):
        """Inicializa campo de velocidade com fluxo potencial ao redor
        do obstáculo real (doublet centrado no centroide da máscara)"""
        x = self.mesh.cellCenters[0].value
        y = self.mesh.cellCenters[1].value

        object_mask = np.asarray(object_mask).astype(bool)
        if np.any(object_mask):
            cx = float(np.mean(x[object_mask]))
            cy = float(np.mean(y[object_mask]))
            # raio efetivo do cilindro equivalente à área sólida
            solid_area = np.count_nonzero(object_mask) * self.dx * self.dy
            R2 = max(solid_area / np.pi, 1e-4)
        else:
            cx, cy = 1.5, 1.0
            R2 = 0.25

        dx = x - cx
        dy = y - cy
        r2 = dx**2 + dy**2
        u_vals = self.inlet_velocity * (1.0 - R2 * (dx**2 - dy**2) / r2**2)
        v_vals = self.inlet_velocity * (-R2 * 2.0 * dx * dy / r2**2)

        # Fluxo uniforme onde está muito perto da singularidade ou dentro do objeto
        near_singularity = r2 <= 0.01
        u_vals = np.where(near_singularity, self.inlet_velocity, u_vals)
        v_vals = np.where(near_singularity, 0.0, v_vals)
        u_vals = np.where(object_mask, 0.0, u_vals)
        v_vals = np.where(object_mask, 0.0, v_vals)

        self.u.setValue(u_vals)
        self.v.setValue(v_vals)
    
    def calculate_pressure_field(self, velocity_magnitude):
        """Calcula campo de pressão usando equação de Bernoulli"""
        # Pressão de referência (entrada)
        p_ref = 101325  # Pa
        
        # Equação de Bernoulli: p + 0.5*rho*v^2 = constante
        pressure = p_ref + 0.5 * self.rho * (self.inlet_velocity**2 - velocity_magnitude**2)
        
        return pressure
    
    def calculate_aero_forces(self, pressure, object_mask):
        """
        Calcula forças aerodinâmicas integrando a pressão nas células de
        fluido adjacentes à superfície do obstáculo.

        Arrasto: diferença de pressão frente/trás integrada por linha.
        Sustentação: diferença de pressão baixo/cima integrada por coluna.

        Returns:
            dict: drag_coefficient, lift_coefficient, drag_force, lift_force,
                  frontal_area (por unidade de profundidade)
        """
        p2 = np.asarray(pressure).reshape(self.ny, self.nx)
        m2 = np.asarray(object_mask).astype(bool).reshape(self.ny, self.nx)

        p_ref = 101325.0
        drag_force = 0.0
        lift_force = 0.0
        n_rows_solid = 0

        # arrasto: para cada linha, pressão imediatamente antes e depois do sólido
        for j in range(self.ny):
            cols = np.flatnonzero(m2[j])
            if cols.size == 0:
                continue
            n_rows_solid += 1
            i0, i1 = cols[0], cols[-1]
            p_front = p2[j, i0 - 1] if i0 > 0 else p_ref
            p_back = p2[j, i1 + 1] if i1 < self.nx - 1 else p_ref
            drag_force += (p_front - p_back) * self.dy

        # sustentação: para cada coluna, pressão logo abaixo e logo acima
        for i in range(self.nx):
            rows = np.flatnonzero(m2[:, i])
            if rows.size == 0:
                continue
            j0, j1 = rows[0], rows[-1]
            p_bottom = p2[j0 - 1, i] if j0 > 0 else p_ref
            p_top = p2[j1 + 1, i] if j1 < self.ny - 1 else p_ref
            lift_force += (p_bottom - p_top) * self.dx

        frontal_area = max(n_rows_solid * self.dy, 1e-12)
        dynamic_pressure = 0.5 * self.rho * self.inlet_velocity**2

        return {
            'drag_force': drag_force,
            'lift_force': lift_force,
            'frontal_area': frontal_area,
            'drag_coefficient': abs(drag_force) / (dynamic_pressure * frontal_area),
            'lift_coefficient': lift_force / (dynamic_pressure * frontal_area),
        }

    def calculate_drag_coefficient(self, pressure, object_mask):
        """Calcula coeficiente de arrasto baseado no campo de pressão"""
        if not np.any(object_mask):
            return 0.0
        return self.calculate_aero_forces(pressure, object_mask)['drag_coefficient']
    
    def solve_steady_state(self, max_iterations=100, tolerance=1e-4, progress_callback=None):
        """
        Resolve simulação CFD em estado estacionário com geometria real
        
        Args:
            max_iterations: Número máximo de iterações
            tolerance: Tolerância de convergência
            progress_callback: Função para atualizar progresso
            
        Returns:
            dict: Resultados da simulação
        """
        self.initialize_variables()
        self.apply_boundary_conditions()
        
        # Criar máscara do objeto baseada na geometria real
        object_mask = self.create_object_mask()
        
        # Inicializar campo de velocidade com fluxo potencial
        self.initialize_potential_flow(object_mask)
        self._update_face_velocity()

        start_time = time.time()
        residual = float('inf')

        dt = 0.01  # passo de pseudo-tempo

        for iteration in range(max_iterations):
            # Salvar valores anteriores para calcular convergência
            u_old = self.u.value.copy()
            v_old = self.v.value.copy()

            # Passo 1 (preditor): momento sem gradiente de pressão -> u*
            eq_u = (TransientTerm(var=self.u) +
                   ConvectionTerm(coeff=self.velocity, var=self.u) -
                   DiffusionTerm(coeff=self.nu, var=self.u))

            eq_v = (TransientTerm(var=self.v) +
                   ConvectionTerm(coeff=self.velocity, var=self.v) -
                   DiffusionTerm(coeff=self.nu, var=self.v))

            eq_u.solve(var=self.u, dt=dt)
            eq_v.solve(var=self.v, dt=dt)

            # No-slip no obstáculo antes da projeção
            self.u.setValue(0.0, where=object_mask)
            self.v.setValue(0.0, where=object_mask)

            # Passo 2 (projeção de Chorin): resolver Poisson da pressão
            # ∇²p = (ρ/dt) ∇·u* para impor incompressibilidade
            div_star = self.u.grad[0] + self.v.grad[1]
            p_eq = DiffusionTerm(var=self.p) == (self.rho / dt) * div_star
            p_eq.solve(var=self.p)

            # Passo 3 (corretor): u = u* - (dt/ρ) ∇p
            p_grad = self.p.grad
            self.u.setValue(self.u.value - (dt / self.rho) * p_grad[0].value)
            self.v.setValue(self.v.value - (dt / self.rho) * p_grad[1].value)

            # Reaplicar não-deslizamento no objeto após a correção
            self.u.setValue(0.0, where=object_mask)
            self.v.setValue(0.0, where=object_mask)

            # Atualizar campo de velocidade combinado (valores nas faces)
            self._update_face_velocity()
            
            # Calcular convergência
            du = np.mean(np.abs(self.u.value - u_old))
            dv = np.mean(np.abs(self.v.value - v_old))
            residual = np.sqrt(du**2 + dv**2) / self.inlet_velocity
            self.residuals.append(residual)
            
            # Atualizar progresso
            if progress_callback:
                elapsed_time = time.time() - start_time
                progress = (iteration + 1) / max_iterations
                progress_callback(progress, iteration + 1, residual, elapsed_time)
            
            # Verificar convergência
            if residual < tolerance:
                break
                
        # Calcular campos derivados
        velocity_magnitude = np.sqrt(self.u.value**2 + self.v.value**2)
        
        # Pressão absoluta a partir do campo resolvido pela projeção
        # (self.p é pressão gauge relativa à saída do túnel)
        pressure = self.p.value + 101325.0

        # Calcular forças aerodinâmicas na superfície do obstáculo
        aero = self.calculate_aero_forces(pressure, object_mask)
        drag_coefficient = aero['drag_coefficient']
        
        # Coordenadas dos eixos (centros de célula), no formato esperado
        # pela camada de visualização: x com nx pontos e y com ny pontos
        x_axis = np.linspace(self.dx / 2, self.Lx - self.dx / 2, self.nx)
        y_axis = np.linspace(self.dy / 2, self.Ly - self.dy / 2, self.ny)

        return {
            'u': self.u.value,
            'v': self.v.value,
            'velocity_u': self.u.value,
            'velocity_v': self.v.value,
            'velocity_magnitude': velocity_magnitude,
            'pressure': pressure,
            'x': x_axis,
            'y': y_axis,
            'mesh_x': x_axis,
            'mesh_y': y_axis,
            'residuals': self.residuals,
            'drag_coefficient': drag_coefficient,
            'lift_coefficient': aero['lift_coefficient'],
            'drag_force': aero['drag_force'],
            'lift_force': aero['lift_force'],
            'frontal_area': aero['frontal_area'],
            'iterations': len(self.residuals),
            'object_mask': np.asarray(object_mask).astype(bool),
            'converged': residual < tolerance,
            'final_residual': residual
        }

    def calculate_streamlines(self, n_streamlines=15):
        """
        Calcula streamlines do campo de velocidade
        
        Args:
            n_streamlines: Número de streamlines
            
        Returns:
            list: Lista de streamlines
        """
        x, y = self.mesh.cellCenters
        u_vals = self.u.value.reshape(self.ny, self.nx)
        v_vals = self.v.value.reshape(self.ny, self.nx)
        
        # Pontos de início das streamlines (entrada do domínio)
        y_starts = np.linspace(0.2, self.Ly - 0.2, n_streamlines)
        streamlines = []
        
        for y_start in y_starts:
            streamline = self._trace_streamline(0.1, y_start, u_vals, v_vals)
            if len(streamline) > 10:  # Apenas streamlines com pontos suficientes
                streamlines.append(streamline)
                
        return streamlines
    
    def _trace_streamline(self, x_start, y_start, u_field, v_field, max_steps=1000):
        """
        Traça uma streamline individual
        
        Args:
            x_start, y_start: Ponto inicial
            u_field, v_field: Campos de velocidade
            max_steps: Número máximo de passos
            
        Returns:
            np.array: Pontos da streamline
        """
        points = [(x_start, y_start)]
        x, y = x_start, y_start
        dt_stream = 0.01
        
        for _ in range(max_steps):
            # Interpolar velocidade na posição atual
            i = int(x / self.dx)
            j = int(y / self.dy)
            
            if i >= self.nx - 1 or j >= self.ny - 1 or i < 0 or j < 0:
                break
                
            u_interp = u_field[j, i]
            v_interp = v_field[j, i]
            
            # Integração Euler
            x += u_interp * dt_stream
            y += v_interp * dt_stream
            
            points.append((x, y))
            
            # Parar se saiu do domínio
            if x > self.Lx or y > self.Ly or y < 0:
                break
                
        return np.array(points)

def create_simple_simulation(simulation_points=None, device=None):
    """
    Cria simulação CFD simples
    
    Args:
        simulation_points: Pontos de simulação (não usado nesta implementação)
        device: Dispositivo PyTorch
        
    Returns:
        CFDSimulation: Instância da simulação
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    sim = CFDSimulation(device=device)
    return sim

def create_gpu_optimized_simulation(domain_size=(4.0, 2.0), resolution=(40, 20),
                                    object_mask=None):
    """
    Cria simulação otimizada para o hardware disponível

    Args:
        domain_size: Tamanho do domínio
        resolution: Resolução base da malha
        object_mask: Máscara da geometria imersa (opcional); deve ter sido
                     gerada na resolução final retornada por esta função —
                     prefira criar a simulação primeiro e chamar
                     set_object_mask() com sim.nx/sim.ny

    Returns:
        CFDSimulation: Simulação otimizada
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolução maior quando há GPU (a silhueta do modelo precisa de células
    # suficientes para ser bem representada), moderada em CPU
    if device.type == 'cuda':
        resolution = (120, 60)
    else:
        resolution = (60, 30)

    sim = CFDSimulation(domain_size=domain_size, resolution=resolution,
                        device=device, object_mask=object_mask)
    return sim

