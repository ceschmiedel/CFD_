"""
Módulo de simulação CFD usando FiPy
Contém classes e funções para simulações de dinâmica de fluidos computacional
"""

import numpy as np
import torch
import time
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ConvectionTerm, DefaultSolver
from fipy.tools import numerix
import streamlit as st

class CFDSimulation:
    """Classe principal para simulações CFD usando FiPy"""
    
    def __init__(self, domain_size=(4.0, 2.0), resolution=(40, 20), device=None):
        """
        Inicializa simulação CFD
        
        Args:
            domain_size: Tupla (Lx, Ly) com dimensões do domínio
            resolution: Tupla (nx, ny) com resolução da malha
            device: Dispositivo PyTorch (cuda/cpu) para otimizações
        """
        self.Lx, self.Ly = domain_size
        self.nx, self.ny = resolution
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        
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
    
    def create_object_mask(self):
        """Cria máscara do objeto baseada na geometria real"""
        # Por enquanto, usar obstáculo circular melhorado
        # Em implementação futura, usar geometria STL real
        x = self.mesh.cellCenters[0].value
        y = self.mesh.cellCenters[1].value
        
        # Criar obstáculo mais realista (formato de aerofólio)
        center_x, center_y = 1.5, 1.0
        
        # Forma de aerofólio NACA aproximada
        mask = np.zeros_like(x, dtype=bool)
        for i in range(len(x)):
            for j in range(len(y)):
                dx = x[i] - center_x
                dy = y[j] - center_y
                
                # Equação aproximada de aerofólio NACA
                if abs(dx) < 0.5 and abs(dy) < 0.1:
                    # Espessura do aerofólio
                    t = 0.12  # 12% de espessura
                    if abs(dx) < 0.4:
                        thickness = t * (0.2969*np.sqrt(abs(dx)/0.4) - 
                                       0.1260*(abs(dx)/0.4) - 
                                       0.3516*(abs(dx)/0.4)**2 + 
                                       0.2843*(abs(dx)/0.4)**3 - 
                                       0.1015*(abs(dx)/0.4)**4)
                        if abs(dy) < thickness:
                            mask[i] = True
        
        return mask
    
    def initialize_potential_flow(self, object_mask):
        """Inicializa campo de velocidade com fluxo potencial"""
        x = self.mesh.cellCenters[0].value
        y = self.mesh.cellCenters[1].value
        
        # Inicializar com fluxo uniforme
        self.u.setValue(self.inlet_velocity)
        self.v.setValue(0.0)
        
        # Modificar fluxo ao redor do objeto
        for i in range(len(x)):
            for j in range(len(y)):
                if not object_mask[i]:
                    # Fluxo potencial ao redor de cilindro (aproximação)
                    dx = x[i] - 1.5  # Centro do objeto
                    dy = y[j] - 1.0
                    r = np.sqrt(dx**2 + dy**2)
                    
                    if r > 0.1:  # Evitar singularidade
                        # Velocidade modificada pelo objeto
                        u_mod = self.inlet_velocity * (1 + 0.25/r**2 * (dx**2 - dy**2))
                        v_mod = self.inlet_velocity * (0.25/r**2 * 2*dx*dy)
                        
                        self.u.setValue(u_mod, where=(self.mesh.cellCenters[0] == x[i]) & 
                                                    (self.mesh.cellCenters[1] == y[j]))
                        self.v.setValue(v_mod, where=(self.mesh.cellCenters[0] == x[i]) & 
                                                    (self.mesh.cellCenters[1] == y[j]))
    
    def calculate_pressure_field(self, velocity_magnitude):
        """Calcula campo de pressão usando equação de Bernoulli"""
        # Pressão de referência (entrada)
        p_ref = 101325  # Pa
        
        # Equação de Bernoulli: p + 0.5*rho*v^2 = constante
        pressure = p_ref + 0.5 * self.rho * (self.inlet_velocity**2 - velocity_magnitude**2)
        
        return pressure
    
    def calculate_drag_coefficient(self, pressure, object_mask):
        """Calcula coeficiente de arrasto baseado no campo de pressão"""
        # Integrar pressão na superfície do objeto
        # Simplificação: usar diferença de pressão média
        
        if np.any(object_mask):
            # Pressão na frente do objeto
            front_pressure = np.mean(pressure[object_mask])
            
            # Pressão de referência
            ref_pressure = 101325
            
            # Força de arrasto aproximada
            drag_force = (front_pressure - ref_pressure) * 0.5  # Área aproximada
            
            # Coeficiente de arrasto
            dynamic_pressure = 0.5 * self.rho * self.inlet_velocity**2
            drag_coefficient = drag_force / (dynamic_pressure * 0.5)  # Área de referência
            
            return abs(drag_coefficient)
        else:
            return 0.0
    
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
        
        start_time = time.time()
        
        for iteration in range(max_iterations):
            # Salvar valores anteriores para calcular convergência
            u_old = self.u.value.copy()
            v_old = self.v.value.copy()
            
            # Resolver equações de Navier-Stokes simplificadas
            # Equação de momento em x
            eq_u = (TransientTerm(var=self.u) + 
                   ConvectionTerm(coeff=self.velocity, var=self.u) - 
                   DiffusionTerm(coeff=self.nu, var=self.u))
            
            # Equação de momento em y  
            eq_v = (TransientTerm(var=self.v) + 
                   ConvectionTerm(coeff=self.velocity, var=self.v) - 
                   DiffusionTerm(coeff=self.nu, var=self.v))
            
            # Resolver com relaxação
            eq_u.solve(var=self.u, dt=0.01)
            eq_v.solve(var=self.v, dt=0.01)
            
            # Aplicar condições de contorno
            self.apply_boundary_conditions()
            
            # Aplicar condição de não-deslizamento no objeto
            self.u.setValue(0.0, where=object_mask)
            self.v.setValue(0.0, where=object_mask)
            
            # Atualizar campo de velocidade combinado
            self.velocity[0] = self.u
            self.velocity[1] = self.v
            
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
        
        # Calcular pressão usando equação de Bernoulli modificada
        pressure = self.calculate_pressure_field(velocity_magnitude)
        
        # Calcular coeficiente de arrasto
        drag_coefficient = self.calculate_drag_coefficient(pressure, object_mask)
        
        return {
            'u': self.u.value,
            'v': self.v.value,
            'velocity_u': self.u.value,
            'velocity_v': self.v.value,
            'velocity_magnitude': velocity_magnitude,
            'pressure': pressure,
            'x': self.mesh.cellCenters[0].value,
            'y': self.mesh.cellCenters[1].value,
            'mesh_x': self.mesh.cellCenters[0].value,
            'mesh_y': self.mesh.cellCenters[1].value,
            'residuals': self.residuals,
            'drag_coefficient': drag_coefficient,
            'iterations': len(self.residuals),
            'object_mask': object_mask
        }        
        # Calcular coeficiente de arrasto (aproximação)
        drag_force = self.calculate_drag_force(obstacle_mask)
        drag_coefficient = drag_force / (0.5 * self.rho * self.inlet_velocity**2 * 2 * 0.3)  # Aproximação
        
        # Preparar resultados
        x, y = self.mesh.cellCenters
        
        results = {
            'x': x,
            'y': y,
            'u': self.u.value,
            'v': self.v.value,
            'p': np.zeros_like(self.u.value),  # Pressão simplificada
            'velocity_magnitude': velocity_magnitude,
            'drag_coefficient': drag_coefficient,
            'residuals': self.residuals,
            'iterations': iteration + 1,
            'converged': residual < tolerance,
            'final_residual': residual
        }
        
        return results
    
    def calculate_drag_force(self, obstacle_mask):
        """
        Calcula força de arrasto no obstáculo
        
        Args:
            obstacle_mask: Máscara do obstáculo
            
        Returns:
            float: Força de arrasto
        """
        # Aproximação simples baseada na pressão
        pressure_on_obstacle = np.mean(self.p.value[obstacle_mask])
        drag_force = pressure_on_obstacle * 0.6  # Área frontal aproximada
        return abs(drag_force)
    
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

def create_gpu_optimized_simulation(domain_size=(4.0, 2.0), resolution=(40, 20)):
    """
    Cria simulação otimizada para GPU
    
    Args:
        domain_size: Tamanho do domínio
        resolution: Resolução da malha
        
    Returns:
        CFDSimulation: Simulação otimizada
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Usar resolução maior se GPU disponível, mas limitada para evitar problemas de memória
    if device.type == 'cuda':
        resolution = (min(resolution[0] * 1.5, 60), min(resolution[1] * 1.5, 30))
    else:
        # Para CPU, manter resolução baixa
        resolution = (30, 15)
        
    sim = CFDSimulation(domain_size=domain_size, resolution=resolution, device=device)
    return sim

