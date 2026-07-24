"""
Módulo de Utilitários - Funções auxiliares e de suporte
Contém funções gerais para o aplicativo CFD
"""

from .helpers import (
    create_sample_stl_data,
    calculate_reynolds_number,
    estimate_drag_coefficient_sphere,
    format_scientific_notation,
    create_colorbar_legend,
    save_figure_to_base64,
    validate_stl_file,
    create_mesh_quality_report,
    interpolate_field_to_points,
    create_domain_info
)

__all__ = [
    'create_sample_stl_data',
    'calculate_reynolds_number',
    'estimate_drag_coefficient_sphere', 
    'format_scientific_notation',
    'create_colorbar_legend',
    'save_figure_to_base64',
    'validate_stl_file',
    'create_mesh_quality_report',
    'interpolate_field_to_points',
    'create_domain_info'
]

