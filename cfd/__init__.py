"""
Módulo CFD - Simulação de Dinâmica de Fluidos Computacional
Contém classes e funções para simulações CFD usando FiPy e modelos de IA
"""

from .simulation import CFDSimulation, create_simple_simulation
from .geometry import GeometryProcessor, process_model_for_cfd, process_stl_for_cfd
from .ai_model import CFDAIModel, create_pretrained_model

__all__ = [
    'CFDSimulation',
    'create_simple_simulation',
    'GeometryProcessor',
    'process_model_for_cfd',
    'process_stl_for_cfd',
    'CFDAIModel',
    'create_pretrained_model'
]

