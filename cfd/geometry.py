"""
Módulo de geometria para manipulação de modelos 3D e projeções 2D
Processa arquivos STL e prepara geometrias para simulação CFD
"""

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class GeometryProcessor:
    """Classe para processamento de geometrias 3D e 2D"""
    
    def __init__(self):
        self.mesh_3d = None
        self.mesh_2d = None
        self.projected_points = None
        self.bounding_box = None
        
    def load_stl(self, file_path_or_buffer):
        """
        Carrega arquivo STL
        
        Args:
            file_path_or_buffer: Caminho do arquivo ou buffer de dados
            
        Returns:
            trimesh.Trimesh: Malha 3D carregada
        """
        try:
            if hasattr(file_path_or_buffer, 'read'):
                # É um buffer/stream
                self.mesh_3d = trimesh.load(file_path_or_buffer, file_type='stl')
            else:
                # É um caminho de arquivo
                self.mesh_3d = trimesh.load(file_path_or_buffer)
            
            # Centralizar e normalizar a malha
            self.mesh_3d = self._normalize_mesh(self.mesh_3d)
            
            return self.mesh_3d
            
        except Exception as e:
            raise ValueError(f"Erro ao carregar arquivo STL: {str(e)}")
    
    def _normalize_mesh(self, mesh):
        """
        Normaliza a malha (centraliza e escala)
        
        Args:
            mesh (trimesh.Trimesh): Malha original
            
        Returns:
            trimesh.Trimesh: Malha normalizada
        """
        # Centralizar na origem
        mesh.vertices -= mesh.centroid
        
        # Escalar para tamanho padrão (maior dimensão = 1.0)
        max_dimension = np.max(mesh.extents)
        if max_dimension > 0:
            mesh.vertices /= max_dimension
        
        return mesh
    
    def project_to_2d(self, projection_plane='xy', simplify_factor=0.1):
        """
        Projeta malha 3D para 2D
        
        Args:
            projection_plane (str): Plano de projeção ('xy', 'xz', 'yz')
            simplify_factor (float): Fator de simplificação do contorno
            
        Returns:
            np.array: Pontos 2D do contorno projetado
        """
        if self.mesh_3d is None:
            raise ValueError("Nenhuma malha 3D carregada")
        
        # Selecionar coordenadas baseado no plano de projeção
        if projection_plane == 'xy':
            coords = self.mesh_3d.vertices[:, [0, 1]]
        elif projection_plane == 'xz':
            coords = self.mesh_3d.vertices[:, [0, 2]]
        elif projection_plane == 'yz':
            coords = self.mesh_3d.vertices[:, [1, 2]]
        else:
            raise ValueError("Plano de projeção deve ser 'xy', 'xz' ou 'yz'")
        
        # Calcular envoltória convexa para obter contorno
        try:
            hull = ConvexHull(coords)
            hull_points = coords[hull.vertices]
            
            # Simplificar contorno se necessário
            if simplify_factor > 0:
                hull_points = self._simplify_contour(hull_points, simplify_factor)
            
            self.projected_points = hull_points
            
            # Calcular bounding box
            self.bounding_box = {
                'min_x': np.min(hull_points[:, 0]),
                'max_x': np.max(hull_points[:, 0]),
                'min_y': np.min(hull_points[:, 1]),
                'max_y': np.max(hull_points[:, 1])
            }
            
            return hull_points
            
        except Exception as e:
            # Fallback: usar todos os pontos únicos
            unique_coords = np.unique(coords, axis=0)
            self.projected_points = unique_coords
            return unique_coords
    
    def _simplify_contour(self, points, factor):
        """
        Simplifica contorno removendo pontos próximos
        
        Args:
            points (np.array): Pontos do contorno
            factor (float): Fator de simplificação
            
        Returns:
            np.array: Pontos simplificados
        """
        if len(points) < 3:
            return points
        
        # Calcular distâncias entre pontos consecutivos
        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        total_perimeter = np.sum(distances)
        
        # Threshold baseado no fator de simplificação
        threshold = factor * total_perimeter / len(points)
        
        # Manter pontos que estão suficientemente distantes
        simplified = [points[0]]  # Sempre manter o primeiro ponto
        
        for i in range(1, len(points)):
            dist_to_last = np.sqrt(np.sum((points[i] - simplified[-1])**2))
            if dist_to_last > threshold:
                simplified.append(points[i])
        
        return np.array(simplified)
    
    def get_frontal_area(self):
        """
        Calcula área frontal projetada
        
        Returns:
            float: Área frontal
        """
        if self.projected_points is None:
            return 0.0
        
        # Usar fórmula de Shoelace para calcular área do polígono
        x = self.projected_points[:, 0]
        y = self.projected_points[:, 1]
        
        # Fechar o polígono se necessário
        if not np.allclose(self.projected_points[0], self.projected_points[-1]):
            x = np.append(x, x[0])
            y = np.append(y, y[0])
        
        area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
        return area
    
    def get_characteristic_length(self):
        """
        Calcula comprimento característico
        
        Returns:
            float: Comprimento característico
        """
        if self.bounding_box is None:
            return 1.0
        
        width = self.bounding_box['max_x'] - self.bounding_box['min_x']
        height = self.bounding_box['max_y'] - self.bounding_box['min_y']
        
        return max(width, height)
    
    def create_airfoil_points(self, n_points=50):
        """
        Cria pontos de um aerofólio NACA simples (para teste)
        
        Args:
            n_points (int): Número de pontos
            
        Returns:
            np.array: Pontos do aerofólio
        """
        # NACA 0012 simplificado
        x = np.linspace(0, 1, n_points)
        
        # Espessura do aerofólio NACA 0012
        t = 0.12  # 12% de espessura
        
        y_upper = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 
                          0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        y_lower = -y_upper
        
        # Combinar superfícies superior e inferior
        x_airfoil = np.concatenate([x, x[::-1]])
        y_airfoil = np.concatenate([y_upper, y_lower[::-1]])
        
        # Centralizar e escalar
        x_airfoil -= 0.5
        
        return np.column_stack([x_airfoil, y_airfoil])
    
    def adapt_to_simulation_domain(self, domain_width=4.0, domain_height=2.0, 
                                 object_scale=0.3):
        """
        Adapta geometria para domínio de simulação
        
        Args:
            domain_width (float): Largura do domínio
            domain_height (float): Altura do domínio
            object_scale (float): Escala do objeto no domínio
            
        Returns:
            np.array: Pontos adaptados para o domínio
        """
        if self.projected_points is None:
            # Usar aerofólio padrão se não há geometria carregada
            points = self.create_airfoil_points()
        else:
            points = self.projected_points.copy()
        
        # Normalizar pontos para [-1, 1]
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            if x_range > 0:
                points[:, 0] = (points[:, 0] - np.mean(points[:, 0])) / x_range * 2
            if y_range > 0:
                points[:, 1] = (points[:, 1] - np.mean(points[:, 1])) / y_range * 2
        
        # Escalar para o domínio
        scale_x = domain_width * object_scale * 0.5
        scale_y = domain_height * object_scale * 0.5
        
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y
        
        # Posicionar no domínio (1/4 da largura a partir da entrada)
        offset_x = domain_width * 0.25
        offset_y = domain_height * 0.5
        
        points[:, 0] += offset_x
        points[:, 1] += offset_y
        
        return points
    
    def get_mesh_info(self):
        """
        Retorna informações da malha 3D
        
        Returns:
            dict: Informações da malha
        """
        if self.mesh_3d is None:
            return {}
        
        return {
            'vertices': len(self.mesh_3d.vertices),
            'faces': len(self.mesh_3d.faces),
            'volume': self.mesh_3d.volume,
            'surface_area': self.mesh_3d.area,
            'centroid': self.mesh_3d.centroid,
            'extents': self.mesh_3d.extents,
            'is_watertight': self.mesh_3d.is_watertight,
            'is_winding_consistent': self.mesh_3d.is_winding_consistent
        }


def process_stl_for_cfd(file_path_or_buffer, projection_plane='xy'):
    """
    Função utilitária para processar STL para CFD
    
    Args:
        file_path_or_buffer: Arquivo STL ou buffer
        projection_plane (str): Plano de projeção
        
    Returns:
        tuple: (GeometryProcessor, pontos_2d, info_malha)
    """
    processor = GeometryProcessor()
    
    # Carregar STL
    mesh_3d = processor.load_stl(file_path_or_buffer)
    
    # Projetar para 2D
    points_2d = processor.project_to_2d(projection_plane)
    
    # Adaptar para domínio de simulação
    simulation_points = processor.adapt_to_simulation_domain()
    
    # Obter informações
    mesh_info = processor.get_mesh_info()
    
    return processor, simulation_points, mesh_info

