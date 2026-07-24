"""
Módulo de geometria para manipulação de modelos 3D e projeções 2D
Processa arquivos STL/GLB/GLTF/OBJ e prepara geometrias para simulação CFD.

Convenção canônica após a importação:
    X = eixo longitudinal (direção do fluxo no túnel de vento)
    Y = eixo lateral (largura)
    Z = eixo vertical (altura)
A malha é reorientada automaticamente para esta convenção (maior extensão
vira X, menor vira Z), o que garante que streamlines e simulação fluam
sempre ao longo do comprimento do modelo.
"""

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy import ndimage
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt

SUPPORTED_EXTENSIONS = ('stl', 'glb', 'gltf', 'obj')


class GeometryProcessor:
    """Classe para processamento de geometrias 3D e 2D"""

    def __init__(self):
        self.mesh_3d = None
        self.mesh_2d = None
        self.projected_points = None
        self.bounding_box = None
        self.orientation_order = None  # permutação de eixos aplicada

    def load_mesh(self, file_path_or_buffer, file_type=None):
        """
        Carrega modelo 3D em qualquer formato suportado (STL, GLB, GLTF, OBJ).

        Args:
            file_path_or_buffer: Caminho do arquivo ou buffer de dados
            file_type (str): Extensão do arquivo (obrigatória para buffers)

        Returns:
            trimesh.Trimesh: Malha 3D canonicalizada e normalizada
        """
        try:
            if hasattr(file_path_or_buffer, 'read'):
                if file_type is None:
                    file_type = getattr(file_path_or_buffer, 'name', 'model.stl')
                if '.' in str(file_type):
                    file_type = str(file_type).rsplit('.', 1)[-1]
                file_type = file_type.lower()
                # force='mesh' concatena cenas GLB/GLTF (múltiplos nós) em
                # uma única malha já com as transformações aplicadas
                mesh = trimesh.load(file_path_or_buffer, file_type=file_type,
                                    force='mesh')
            else:
                mesh = trimesh.load(file_path_or_buffer, force='mesh')

            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                raise ValueError("Arquivo não contém malha triangular válida")

            mesh = self._canonicalize_orientation(mesh)
            mesh = self._normalize_mesh(mesh)
            self.mesh_3d = mesh
            return mesh

        except Exception as e:
            raise ValueError(f"Erro ao carregar modelo 3D: {str(e)}")

    # Alias mantido por compatibilidade com código existente
    def load_stl(self, file_path_or_buffer):
        return self.load_mesh(file_path_or_buffer, file_type='stl')

    def _canonicalize_orientation(self, mesh):
        """
        Reorienta a malha para a convenção do túnel de vento:
        maior dimensão -> X (fluxo), menor dimensão -> Z (vertical).

        Modelos vêm com eixos arbitrários (GLB usa Y-up, STLs variam);
        sem esta etapa o fluxo fica perpendicular ao modelo.
        """
        extents = mesh.extents
        # ordem decrescente de extensão: [longitudinal, lateral, vertical]
        order = np.argsort(extents)[::-1]

        if not np.array_equal(order, [0, 1, 2]):
            mesh.vertices = mesh.vertices[:, order]
            # permutação ímpar inverte a orientação (det = -1); inverter o
            # winding das faces preserva as normais para fora
            parity = np.linalg.det(np.eye(3)[order])
            if parity < 0:
                mesh.faces = mesh.faces[:, ::-1]

        self.orientation_order = order
        return mesh

    def _normalize_mesh(self, mesh):
        """
        Normaliza a malha (centraliza no centro da bounding box e escala
        para maior dimensão = 1.0)
        """
        bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        mesh.vertices -= bbox_center

        max_dimension = np.max(mesh.extents)
        if max_dimension > 0:
            mesh.vertices /= max_dimension

        return mesh

    def project_to_2d(self, projection_plane='xz', simplify_factor=0.1):
        """
        Projeta malha 3D para 2D. Com a malha canonicalizada, o plano 'xz'
        é a vista lateral (comprimento x altura) usada no túnel de vento.

        Args:
            projection_plane (str): Plano de projeção ('xy', 'xz', 'yz')
            simplify_factor (float): Fator de simplificação do contorno

        Returns:
            np.array: Pontos 2D do contorno projetado
        """
        if self.mesh_3d is None:
            raise ValueError("Nenhuma malha 3D carregada")

        if projection_plane == 'xy':
            coords = self.mesh_3d.vertices[:, [0, 1]]
        elif projection_plane == 'xz':
            coords = self.mesh_3d.vertices[:, [0, 2]]
        elif projection_plane == 'yz':
            coords = self.mesh_3d.vertices[:, [1, 2]]
        else:
            raise ValueError("Plano de projeção deve ser 'xy', 'xz' ou 'yz'")

        try:
            hull = ConvexHull(coords)
            hull_points = coords[hull.vertices]

            if simplify_factor > 0:
                hull_points = self._simplify_contour(hull_points, simplify_factor)

            self.projected_points = hull_points

            self.bounding_box = {
                'min_x': np.min(hull_points[:, 0]),
                'max_x': np.max(hull_points[:, 0]),
                'min_y': np.min(hull_points[:, 1]),
                'max_y': np.max(hull_points[:, 1])
            }

            return hull_points

        except Exception:
            unique_coords = np.unique(coords, axis=0)
            self.projected_points = unique_coords
            return unique_coords

    def _simplify_contour(self, points, factor):
        """Simplifica contorno removendo pontos próximos"""
        if len(points) < 3:
            return points

        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        total_perimeter = np.sum(distances)
        threshold = factor * total_perimeter / len(points)

        simplified = [points[0]]
        for i in range(1, len(points)):
            dist_to_last = np.sqrt(np.sum((points[i] - simplified[-1])**2))
            if dist_to_last > threshold:
                simplified.append(points[i])

        return np.array(simplified)

    @staticmethod
    def _map_points_to_domain(points, domain_width=4.0, domain_height=2.0,
                              object_scale=0.3):
        """
        Mapeia pontos 2D (vista lateral do modelo) para coordenadas do
        domínio de simulação, preservando a proporção do modelo.

        A escala é uniforme: o comprimento vira `domain_width * object_scale`
        e a altura acompanha a razão de aspecto real do modelo. O objeto é
        posicionado a 1/4 da entrada, centrado verticalmente.
        """
        points = np.asarray(points, dtype=float).copy()
        if len(points) == 0:
            return points

        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        x_range = max(x_max - x_min, 1e-12)

        # escala uniforme baseada no comprimento
        scale = (domain_width * object_scale) / x_range

        # limitar altura a 60% do domínio (evita bloquear o túnel)
        y_range = max(y_max - y_min, 1e-12)
        max_height = 0.6 * domain_height
        if y_range * scale > max_height:
            scale = max_height / y_range

        center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
        points = (points - center) * scale

        points[:, 0] += domain_width * 0.25
        points[:, 1] += domain_height * 0.5

        return points

    def adapt_to_simulation_domain(self, domain_width=4.0, domain_height=2.0,
                                   object_scale=0.3):
        """
        Adapta geometria (contorno projetado) para o domínio de simulação.

        Returns:
            np.array: Pontos adaptados para o domínio
        """
        if self.projected_points is None:
            points = self.create_airfoil_points()
        else:
            points = self.projected_points.copy()

        return self._map_points_to_domain(points, domain_width,
                                          domain_height, object_scale)

    def build_occupancy_mask(self, nx, ny, domain_width=4.0, domain_height=2.0,
                             object_scale=0.3, n_samples=150000):
        """
        Rasteriza a silhueta real do modelo na grade da simulação, gerando a
        máscara de células sólidas (obstáculo) para o solver CFD.

        Usa uma nuvem densa de pontos da superfície (robusto mesmo para
        malhas não-watertight) projetada na vista lateral, seguida de
        fechamento morfológico e preenchimento de buracos.

        Args:
            nx, ny: Resolução da grade da simulação
            domain_width, domain_height: Dimensões do domínio
            object_scale: Fração do domínio ocupada pelo comprimento do objeto
            n_samples: Nº de pontos amostrados da superfície

        Returns:
            np.array: Máscara booleana achatada (ordem FiPy: x mais rápido,
                      tamanho nx*ny)
        """
        dx = domain_width / nx
        dy = domain_height / ny

        if self.mesh_3d is not None:
            pts3 = self.mesh_3d.vertices
            if len(pts3) < n_samples:
                try:
                    samples, _ = trimesh.sample.sample_surface(
                        self.mesh_3d, n_samples - len(pts3))
                    pts3 = np.vstack([pts3, samples])
                except Exception:
                    pass
            side_pts = pts3[:, [0, 2]]  # vista lateral: X (fluxo) x Z (altura)
            mapped = self._map_points_to_domain(
                side_pts, domain_width, domain_height, object_scale)

            x_edges = np.linspace(0, domain_width, nx + 1)
            y_edges = np.linspace(0, domain_height, ny + 1)
            hist, _, _ = np.histogram2d(mapped[:, 0], mapped[:, 1],
                                        bins=[x_edges, y_edges])
            mask_2d = hist.T > 0  # (ny, nx)

            # fechar lacunas entre pontos amostrados e preencher o interior
            mask_2d = ndimage.binary_closing(mask_2d, iterations=2)
            mask_2d = ndimage.binary_fill_holes(mask_2d)

        elif self.projected_points is not None:
            # fallback: rasterizar polígono do contorno (ex.: aerofólio NACA)
            outline = self._map_points_to_domain(
                self.projected_points, domain_width, domain_height,
                object_scale)
            path = MplPath(outline)
            xc = (np.arange(nx) + 0.5) * dx
            yc = (np.arange(ny) + 0.5) * dy
            xx, yy = np.meshgrid(xc, yc)
            centers = np.column_stack([xx.ravel(), yy.ravel()])
            mask_2d = path.contains_points(centers).reshape(ny, nx)

        else:
            raise ValueError("Nenhuma geometria carregada para gerar máscara")

        # nunca bloquear entrada, saída, teto ou piso do túnel
        mask_2d[:, 0] = False
        mask_2d[:, -1] = False
        mask_2d[0, :] = False
        mask_2d[-1, :] = False

        return mask_2d.ravel()

    def get_frontal_area(self):
        """Calcula área frontal projetada (fórmula de Shoelace)"""
        if self.projected_points is None:
            return 0.0

        x = self.projected_points[:, 0]
        y = self.projected_points[:, 1]

        if not np.allclose(self.projected_points[0], self.projected_points[-1]):
            x = np.append(x, x[0])
            y = np.append(y, y[0])

        area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
        return area

    def get_characteristic_length(self):
        """Calcula comprimento característico"""
        if self.bounding_box is None:
            return 1.0

        width = self.bounding_box['max_x'] - self.bounding_box['min_x']
        height = self.bounding_box['max_y'] - self.bounding_box['min_y']

        return max(width, height)

    def create_airfoil_points(self, n_points=50):
        """Cria pontos de um aerofólio NACA 0012 (geometria de teste)"""
        x = np.linspace(0, 1, n_points)
        t = 0.12  # 12% de espessura

        y_upper = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x -
                           0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        y_lower = -y_upper

        x_airfoil = np.concatenate([x, x[::-1]])
        y_airfoil = np.concatenate([y_upper, y_lower[::-1]])
        x_airfoil -= 0.5

        points = np.column_stack([x_airfoil, y_airfoil])
        self.projected_points = points
        self.bounding_box = {
            'min_x': points[:, 0].min(), 'max_x': points[:, 0].max(),
            'min_y': points[:, 1].min(), 'max_y': points[:, 1].max()
        }
        return points

    def get_mesh_info(self):
        """Retorna informações da malha 3D"""
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


def process_model_for_cfd(file_path_or_buffer, projection_plane='xz',
                          file_type=None):
    """
    Função utilitária para processar modelo 3D (STL/GLB/GLTF/OBJ) para CFD.

    Args:
        file_path_or_buffer: Arquivo ou buffer
        projection_plane (str): Plano de projeção ('xz' = vista lateral)
        file_type (str): Extensão, necessária para buffers sem nome

    Returns:
        tuple: (GeometryProcessor, pontos_2d, info_malha)
    """
    processor = GeometryProcessor()

    processor.load_mesh(file_path_or_buffer, file_type=file_type)
    processor.project_to_2d(projection_plane)
    simulation_points = processor.adapt_to_simulation_domain()
    mesh_info = processor.get_mesh_info()

    return processor, simulation_points, mesh_info


# Alias mantido por compatibilidade
def process_stl_for_cfd(file_path_or_buffer, projection_plane='xz'):
    return process_model_for_cfd(file_path_or_buffer,
                                 projection_plane=projection_plane,
                                 file_type='stl')
