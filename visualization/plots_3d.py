"""
Módulo de visualizações 3D para resultados CFD
Contém classes para criação de visualizações tridimensionais usando PyVista e Plotly
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None

class CFDPlotter3D:
    """Classe para criação de visualizações 3D de resultados CFD"""
    
    def __init__(self):
        self.fig = None
        
    def plot_mesh_interactive(self, mesh):
        """
        Cria visualização 3D interativa do modelo usando Plotly
        
        Args:
            mesh: Objeto trimesh com o modelo 3D
            
        Returns:
            plotly.graph_objects.Figure: Figura interativa
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Criar figura Plotly 3D interativa
        fig = go.Figure(data=[go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightblue',
            opacity=0.7,
            lighting=dict(
                ambient=0.18,
                diffuse=1,
                fresnel=0.1,
                specular=1,
                roughness=0.05,
                facenormalsepsilon=1e-15,
                vertexnormalsepsilon=1e-15
            ),
            lightposition=dict(
                x=100,
                y=200,
                z=0
            )
        )])
        
        fig.update_layout(
            title="Modelo 3D Interativo",
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600,
            margin=dict(r=20, l=10, b=10, t=40)
        )
        
        return fig
    
    def plot_mesh_with_streamlines(self, mesh, velocity_field=None):
        """
        Visualiza modelo 3D com streamlines
        
        Args:
            mesh: Objeto trimesh
            velocity_field: Campo de velocidade (opcional)
            
        Returns:
            plotly.graph_objects.Figure: Figura com streamlines
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        fig = go.Figure()
        
        # Adicionar malha do modelo
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightgray',
            opacity=0.6,
            name='Modelo 3D'
        ))
        
        # Gerar streamlines sintéticas se não fornecidas
        if velocity_field is None:
            # Criar streamlines sintéticas ao redor do modelo
            bounds = mesh.bounds
            x_min, y_min, z_min = bounds[0]
            x_max, y_max, z_max = bounds[1]
            
            # Pontos de origem das streamlines
            n_streams = 20
            y_start = np.linspace(y_min - 0.5, y_max + 0.5, n_streams)
            z_start = np.linspace(z_min - 0.3, z_max + 0.3, n_streams)
            
            for i, y in enumerate(y_start):
                for j, z in enumerate(z_start):
                    # Streamline simples (fluxo em X)
                    x_stream = np.linspace(x_min - 1.0, x_max + 1.0, 50)
                    y_stream = np.full_like(x_stream, y)
                    z_stream = np.full_like(x_stream, z)
                    
                    # Adicionar perturbação para simular fluxo ao redor do objeto
                    center_y = (y_min + y_max) / 2
                    center_z = (z_min + z_max) / 2
                    
                    # Desvio baseado na distância do centro
                    dist_y = abs(y - center_y)
                    dist_z = abs(z - center_z)
                    
                    if dist_y < (y_max - y_min) / 2 and dist_z < (z_max - z_min) / 2:
                        # Streamlines que passam perto do objeto são desviadas
                        deviation = 0.3 * np.exp(-(x_stream - (x_min + x_max)/2)**2 / 0.5)
                        y_stream += deviation * np.sign(y - center_y)
                        z_stream += deviation * np.sign(z - center_z)
                    
                    # Colorir baseado na velocidade (simulada)
                    velocity_mag = 1.0 + 0.3 * np.sin(x_stream * 2)
                    
                    fig.add_trace(go.Scatter3d(
                        x=x_stream,
                        y=y_stream,
                        z=z_stream,
                        mode='lines',
                        line=dict(
                            color=velocity_mag,
                            colorscale='Viridis',
                            width=3
                        ),
                        name=f'Streamline {i*len(z_start)+j+1}' if i == 0 and j == 0 else None,
                        showlegend=True if i == 0 and j == 0 else False
                    ))
        
        fig.update_layout(
            title="Modelo 3D com Streamlines",
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=2.0, y=2.0, z=1.5)
                )
            ),
            width=900,
            height=700,
            margin=dict(r=20, l=10, b=10, t=40)
        )
        
        return fig
    
    def create_wind_tunnel_animation(self, mesh, n_frames=50):
        """
        Cria animação do túnel de vento
        
        Args:
            mesh: Objeto trimesh
            n_frames: Número de frames da animação
            
        Returns:
            plotly.graph_objects.Figure: Figura com animação
        """
        vertices = mesh.vertices
        faces = mesh.faces
        bounds = mesh.bounds
        
        # Configurar frames da animação
        frames = []
        
        for frame in range(n_frames):
            frame_data = []
            
            # Adicionar modelo (estático)
            frame_data.append(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightgray',
                opacity=0.7,
                name='Modelo'
            ))
            
            # Adicionar partículas do vento (animadas)
            time_offset = frame * 0.1
            
            # Gerar partículas
            n_particles = 100
            x_min, y_min, z_min = bounds[0]
            x_max, y_max, z_max = bounds[1]
            
            # Posições das partículas
            x_particles = np.linspace(x_min - 2.0, x_max + 2.0, n_particles)
            y_particles = np.random.uniform(y_min - 1.0, y_max + 1.0, n_particles)
            z_particles = np.random.uniform(z_min - 0.5, z_max + 0.5, n_particles)
            
            # Animar movimento das partículas
            x_particles = x_particles + time_offset * 2.0
            
            # Resetar partículas que saíram do domínio
            x_particles = np.where(x_particles > x_max + 2.0, x_min - 2.0, x_particles)
            
            # Velocidade das partículas (colormap)
            velocities = 1.0 + 0.5 * np.sin(x_particles + time_offset)
            
            frame_data.append(go.Scatter3d(
                x=x_particles,
                y=y_particles,
                z=z_particles,
                mode='markers',
                marker=dict(
                    size=3,
                    color=velocities,
                    colorscale='Viridis',
                    opacity=0.8
                ),
                name='Partículas do Vento'
            ))
            
            frames.append(go.Frame(data=frame_data, name=str(frame)))
        
        # Criar figura inicial
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Configurar animação
        fig.update_layout(
            title="Animação do Túnel de Vento",
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=2.5, y=2.5, z=1.5)
                )
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 50}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            width=900,
            height=700,
            margin=dict(r=20, l=10, b=10, t=40)
        )
        
        return fig
    
    def plot_mesh_matplotlib(self, mesh, title="Modelo 3D"):
        """
        Visualização 3D usando matplotlib (fallback)
        
        Args:
            mesh: Objeto trimesh
            title: Título do gráfico
            
        Returns:
            matplotlib.figure.Figure: Figura matplotlib
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plotar faces do modelo
        vertices = mesh.vertices
        faces = mesh.faces
        
        for face in faces[::10]:  # Plotar apenas algumas faces para performance
            triangle = vertices[face]
            ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                           alpha=0.3, color='lightblue')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        
        return fig

    def visualize_3d_streamlines_pyvista(self, stl_file, output_png="streamlines_pyvista.png"):
        """
        Visualização 3D com PyVista (se disponível)
        
        Args:
            stl_file: Caminho para arquivo STL
            output_png: Caminho para salvar imagem
            
        Returns:
            str: Caminho da imagem gerada ou None se erro
        """
        if not PYVISTA_AVAILABLE:
            st.error("PyVista não está disponível. Usando visualização alternativa.")
            return None
        
        try:
            # Ler arquivo STL
            mesh = pv.read(stl_file)
            
            # Adicionar campo vetorial uniforme
            n_points = mesh.n_points
            velocity = np.tile([1.0, 0.0, 0.0], (n_points, 1))
            mesh["velocity"] = velocity
            
            # Gerar streamlines
            streamlines = mesh.streamlines('velocity', source_center=mesh.center, 
                                         n_points=100, max_time=100.0)
            
            # Criar visualização
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(mesh, opacity=0.5, color="white")
            plotter.add_mesh(streamlines, color="yellow", line_width=2)
            plotter.add_axes()
            plotter.show(screenshot=output_png)
            
            return output_png
            
        except Exception as e:
            st.error(f"Erro na visualização PyVista: {str(e)}")
            return None

def create_3d_visualization_from_stl(stl_file_path):
    """
    Função auxiliar para criar visualização 3D a partir de arquivo STL
    
    Args:
        stl_file_path: Caminho para arquivo STL
        
    Returns:
        plotly.graph_objects.Figure: Figura 3D
    """
    import trimesh
    
    try:
        mesh = trimesh.load(stl_file_path)
        plotter = CFDPlotter3D()
        return plotter.plot_mesh_interactive(mesh)
    except Exception as e:
        st.error(f"Erro ao carregar STL: {str(e)}")
        return None

