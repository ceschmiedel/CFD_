"""
Módulo de Visualização - Gráficos 2D e 3D para resultados CFD
Contém classes para criação de visualizações usando matplotlib, plotly e pyvista
"""

from .plots_2d import CFDPlotter2D, display_results_summary, create_comparison_plot
from .plots_3d import CFDPlotter3D, create_3d_visualization_from_stl

__all__ = [
    'CFDPlotter2D',
    'display_results_summary', 
    'create_comparison_plot',
    'CFDPlotter3D',
    'create_3d_visualization_from_stl'
]

