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

def _decimate_for_display(mesh, max_bytes=150_000_000):
    """
    Prepara a malha para exibição no navegador sem estourar o limite de
    mensagem do Streamlit (~400 MB serializado pelo Plotly).

    Só decima quando o tamanho serializado estimado excede o orçamento —
    modelos típicos (até ~2M de faces) passam inteiros, preservando todos
    os detalhes. Modelos de F1 são "sopa de triângulos" com milhares de
    peças finas desconectadas: decimação agressiva os pulveriza, então
    quando inevitável usa-se colapso suave (aggression=0) e, na falta do
    decimador, mantêm-se as maiores faces (nunca amostragem aleatória).
    A física continua usando a malha completa em qualquer caso.
    """
    import trimesh

    # coordenadas arredondadas encurtam o JSON (~30%) sem perda visual
    # (malha normalizada para dimensão 1.0; 5 casas ≈ precisão de 10 µm)
    verts = np.round(np.asarray(mesh.vertices, dtype=np.float64), 5)
    faces = np.asarray(mesh.faces)

    # ~9 bytes/coordenada e ~8 bytes/índice no JSON
    estimated = verts.shape[0] * 3 * 9 + faces.shape[0] * 3 * 8

    if estimated > max_bytes:
        target = max(int(faces.shape[0] * max_bytes / estimated), 20000)
        simplified = None
        try:
            # aggression=0: colapso conservador (pode parar acima do alvo,
            # mas preserva muito mais superfície que o padrão)
            simplified = mesh.simplify_quadric_decimation(
                face_count=target, aggression=0)
        except BaseException:
            pass

        if simplified is not None and len(simplified.faces) > 0:
            verts = np.round(np.asarray(simplified.vertices), 5)
            faces = np.asarray(simplified.faces)
        else:
            # manter as maiores faces preserva carroceria/painéis e
            # descarta micro-triângulos (grades, rebites)
            order = np.argsort(mesh.area_faces)[::-1][:target]
            sub = mesh.submesh([order], append=True)
            verts = np.round(np.asarray(sub.vertices), 5)
            faces = np.asarray(sub.faces)

    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


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
        mesh = _decimate_for_display(mesh)
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
        display_mesh = _decimate_for_display(mesh)
        vertices = display_mesh.vertices
        faces = display_mesh.faces

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
        
        # Gerar streamlines sintéticas se não fornecidas.
        # A malha é canonicalizada na importação (X = eixo longitudinal),
        # então o fluxo ao longo de +X está sempre alinhado ao modelo.
        if velocity_field is None:
            bounds = mesh.bounds
            x_min, y_min, z_min = bounds[0]
            x_max, y_max, z_max = bounds[1]

            length = x_max - x_min
            cy = (y_min + y_max) / 2
            cz = (z_min + z_max) / 2
            # semi-eixos da seção transversal do corpo (com folga de 10%)
            ay = max((y_max - y_min) / 2 * 1.1, 1e-6)
            az = max((z_max - z_min) / 2 * 1.1, 1e-6)
            xc = (x_min + x_max) / 2

            n_streams = 9
            y_seeds = np.linspace(cy - 2.0 * ay, cy + 2.0 * ay, n_streams)
            z_seeds = np.linspace(cz - 1.6 * az, cz + 1.6 * az, n_streams)

            x_stream = np.linspace(x_min - 0.8 * length,
                                   x_max + 0.8 * length, 60)
            # intensidade da deflexão ao longo do corpo (bump gaussiano)
            bump = np.exp(-((x_stream - xc) / (0.4 * length))**2)

            # Acumular todas as linhas em um único trace usando NaN como
            # separador (muito mais leve que centenas de traces)
            xs, ys, zs, speed = [], [], [], []

            for y0 in y_seeds:
                for z0 in z_seeds:
                    ry = (y0 - cy) / ay
                    rz = (z0 - cz) / az
                    r = np.sqrt(ry**2 + rz**2)

                    if r < 0.15:
                        continue  # linha colidiria de frente com o corpo

                    # empurrar para fora as linhas que passariam por dentro
                    # da seção do corpo; decai suavemente para longe dele
                    push = max(0.0, 1.15 - r)
                    scale = 1.0 + push * bump

                    y_line = cy + (y0 - cy) * scale
                    z_line = cz + (z0 - cz) * scale

                    # aceleração local onde o fluxo contorna o corpo
                    v_line = 1.0 + 0.6 * push * bump

                    xs.extend(x_stream.tolist() + [np.nan])
                    ys.extend(y_line.tolist() + [np.nan])
                    zs.extend(z_line.tolist() + [np.nan])
                    speed.extend(v_line.tolist() + [np.nan])

            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='lines',
                line=dict(
                    color=speed,
                    colorscale='Viridis',
                    width=3,
                    colorbar=dict(title='Velocidade relativa')
                ),
                name='Streamlines'
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
        # A malha é serializada uma única vez (frames só têm partículas)
        display_mesh = _decimate_for_display(mesh)
        vertices = display_mesh.vertices
        faces = display_mesh.faces
        bounds = mesh.bounds

        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]

        # Trace 0: modelo (estático — fica fora dos frames)
        mesh_trace = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightgray',
            opacity=0.7,
            name='Modelo'
        )

        # Posições base das partículas (fixas entre frames; só x anima)
        n_particles = 100
        rng = np.random.default_rng(42)
        x_base = np.linspace(x_min - 2.0, x_max + 2.0, n_particles)
        y_particles = rng.uniform(y_min - 1.0, y_max + 1.0, n_particles)
        z_particles = rng.uniform(z_min - 0.5, z_max + 0.5, n_particles)

        def particle_trace(frame):
            time_offset = frame * 0.1
            x_particles = x_base + time_offset * 2.0
            # recircular partículas que saíram do domínio
            span = (x_max + 2.0) - (x_min - 2.0)
            x_particles = (x_particles - (x_min - 2.0)) % span + (x_min - 2.0)
            velocities = 1.0 + 0.5 * np.sin(x_particles + time_offset)
            return go.Scatter3d(
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
            )

        # Frames atualizam APENAS o trace das partículas (traces=[1]);
        # repetir a malha em cada frame multiplicava o tamanho da mensagem
        # e estourava o limite do Streamlit
        frames = [
            go.Frame(data=[particle_trace(f)], traces=[1], name=str(f))
            for f in range(n_frames)
        ]

        # Criar figura inicial (malha + primeiro frame de partículas)
        fig = go.Figure(
            data=[mesh_trace, particle_trace(0)],
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

