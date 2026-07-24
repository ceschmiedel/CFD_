"""
Funções auxiliares para o aplicativo CFD
Contém utilitários gerais e funções de suporte
"""

import numpy as np
import tempfile
import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt

def create_sample_stl_data():
    """
    Cria dados de exemplo de um cubo simples em formato STL
    
    Returns:
        bytes: Dados STL em formato binário
    """
    # Vértices de um cubo simples
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Base inferior
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Base superior
    ], dtype=np.float32)
    
    # Faces do cubo (triângulos)
    faces = np.array([
        # Base inferior
        [0, 1, 2], [0, 2, 3],
        # Base superior  
        [4, 6, 5], [4, 7, 6],
        # Laterais
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0]
    ], dtype=np.uint32)
    
    # Criar arquivo STL em memória
    stl_data = BytesIO()
    
    # Header (80 bytes)
    header = b"Simple cube for CFD testing" + b"\x00" * (80 - 28)
    stl_data.write(header)
    
    # Número de triângulos
    stl_data.write(len(faces).to_bytes(4, byteorder='little'))
    
    # Escrever cada triângulo
    for face in faces:
        # Normal (calculada automaticamente como [0,0,0])
        stl_data.write(b"\x00" * 12)
        
        # Vértices do triângulo
        for vertex_idx in face:
            vertex = vertices[vertex_idx]
            for coord in vertex:
                stl_data.write(coord.tobytes())
        
        # Attribute byte count (2 bytes, sempre 0)
        stl_data.write(b"\x00\x00")
    
    return stl_data.getvalue()

def calculate_reynolds_number(velocity, length, kinematic_viscosity=1.5e-5):
    """
    Calcula número de Reynolds
    
    Args:
        velocity (float): Velocidade característica (m/s)
        length (float): Comprimento característico (m)
        kinematic_viscosity (float): Viscosidade cinemática (m²/s)
        
    Returns:
        float: Número de Reynolds
    """
    return velocity * length / kinematic_viscosity

def estimate_drag_coefficient_sphere(reynolds):
    """
    Estima coeficiente de arrasto para esfera baseado no número de Reynolds
    
    Args:
        reynolds (float): Número de Reynolds
        
    Returns:
        float: Coeficiente de arrasto estimado
    """
    if reynolds < 1:
        # Regime de Stokes
        return 24 / reynolds
    elif reynolds < 1000:
        # Regime intermediário
        return 24 / reynolds * (1 + 0.15 * reynolds**0.687)
    else:
        # Regime turbulento
        return 0.44

def format_scientific_notation(value, precision=2):
    """
    Formata número em notação científica
    
    Args:
        value (float): Valor a ser formatado
        precision (int): Número de casas decimais
        
    Returns:
        str: Valor formatado
    """
    if abs(value) < 1e-10:
        return "0"
    elif abs(value) >= 1000 or abs(value) < 0.01:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"

def create_colorbar_legend(values, colormap='viridis', title="Valores"):
    """
    Cria legenda de colorbar
    
    Args:
        values (array): Valores para a escala
        colormap (str): Nome do colormap
        title (str): Título da legenda
        
    Returns:
        matplotlib.figure.Figure: Figura com a legenda
    """
    fig, ax = plt.subplots(figsize=(8, 1))
    
    # Criar colorbar horizontal
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=np.min(values), vmax=np.max(values))
    
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal'
    )
    
    cb.set_label(title)
    
    return fig

def save_figure_to_base64(fig):
    """
    Converte figura matplotlib para base64
    
    Args:
        fig (matplotlib.figure.Figure): Figura a ser convertida
        
    Returns:
        str: String base64 da imagem
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    
    return img_base64

def validate_stl_file(file_data):
    """
    Valida se os dados são de um arquivo STL válido
    
    Args:
        file_data (bytes): Dados do arquivo
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        if len(file_data) < 84:
            return False, "Arquivo muito pequeno para ser um STL válido"
        
        # Verificar se é STL ASCII ou binário
        header = file_data[:80].decode('ascii', errors='ignore')
        
        if header.strip().lower().startswith('solid'):
            # Possível STL ASCII
            try:
                text = file_data.decode('ascii')
                if 'facet normal' in text and 'vertex' in text:
                    return True, "STL ASCII válido"
                else:
                    return False, "Formato STL ASCII inválido"
            except:
                return False, "Erro ao decodificar STL ASCII"
        else:
            # STL binário
            if len(file_data) >= 84:
                # Ler número de triângulos
                num_triangles = int.from_bytes(file_data[80:84], byteorder='little')
                expected_size = 84 + num_triangles * 50
                
                if len(file_data) == expected_size:
                    return True, "STL binário válido"
                else:
                    return False, f"Tamanho incorreto para STL binário (esperado: {expected_size}, atual: {len(file_data)})"
            else:
                return False, "STL binário incompleto"
                
    except Exception as e:
        return False, f"Erro na validação: {str(e)}"

def create_mesh_quality_report(mesh_info):
    """
    Cria relatório de qualidade da malha
    
    Args:
        mesh_info (dict): Informações da malha
        
    Returns:
        dict: Relatório de qualidade
    """
    report = {
        'overall_quality': 'Unknown',
        'issues': [],
        'recommendations': []
    }
    
    # Verificar se é watertight
    if not mesh_info.get('is_watertight', False):
        report['issues'].append("Malha não é watertight (possui buracos)")
        report['recommendations'].append("Reparar buracos na malha antes da simulação")
    
    # Verificar consistência de orientação
    if not mesh_info.get('is_winding_consistent', False):
        report['issues'].append("Orientação das faces inconsistente")
        report['recommendations'].append("Corrigir orientação das normais das faces")
    
    # Verificar número de faces
    num_faces = mesh_info.get('faces', 0)
    if num_faces < 100:
        report['issues'].append("Malha muito grosseira (poucas faces)")
        report['recommendations'].append("Usar malha com maior resolução")
    elif num_faces > 100000:
        report['issues'].append("Malha muito densa (muitas faces)")
        report['recommendations'].append("Simplificar malha para melhor performance")
    
    # Verificar volume
    volume = mesh_info.get('volume', 0)
    if volume <= 0:
        report['issues'].append("Volume inválido ou negativo")
        report['recommendations'].append("Verificar orientação das faces")
    
    # Determinar qualidade geral
    if len(report['issues']) == 0:
        report['overall_quality'] = 'Excelente'
    elif len(report['issues']) <= 2:
        report['overall_quality'] = 'Boa'
    else:
        report['overall_quality'] = 'Ruim'
    
    return report

def interpolate_field_to_points(field_values, field_coords, target_points):
    """
    Interpola campo para pontos específicos
    
    Args:
        field_values (array): Valores do campo
        field_coords (array): Coordenadas dos pontos do campo
        target_points (array): Pontos onde interpolar
        
    Returns:
        array: Valores interpolados
    """
    from scipy.spatial import cKDTree
    from scipy.interpolate import griddata
    
    try:
        # Usar interpolação por vizinhos mais próximos se griddata falhar
        if len(field_coords.shape) == 1:
            # Coordenadas 1D, assumir malha estruturada
            return np.interp(target_points.flatten(), field_coords, field_values)
        else:
            # Coordenadas 2D/3D
            interpolated = griddata(
                field_coords, field_values, target_points, 
                method='linear', fill_value=0.0
            )
            return interpolated
            
    except Exception as e:
        print(f"Erro na interpolação: {e}")
        # Fallback: usar valor médio
        return np.full(len(target_points), np.mean(field_values))

def create_domain_info(domain_size=(4.0, 2.0), object_size=0.3):
    """
    Cria informações sobre o domínio de simulação
    
    Args:
        domain_size (tuple): Tamanho do domínio (Lx, Ly)
        object_size (float): Tamanho relativo do objeto
        
    Returns:
        dict: Informações do domínio
    """
    Lx, Ly = domain_size
    
    return {
        'domain_width': Lx,
        'domain_height': Ly,
        'domain_area': Lx * Ly,
        'object_scale': object_size,
        'inlet_position': 0.0,
        'outlet_position': Lx,
        'object_position': Lx * 0.25,
        'aspect_ratio': Lx / Ly,
        'recommended_mesh_size': {
            'coarse': (40, 20),
            'medium': (80, 40),
            'fine': (160, 80)
        }
    }

