"""
Módulo de visualizações 2D para resultados CFD
Cria gráficos de contorno, streamlines e outras visualizações
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import streamlit as st

class CFDPlotter2D:
    """Classe para criação de gráficos 2D de CFD"""
    
    def __init__(self, figsize=(12, 6)):
        """
        Inicializa o plotter
        
        Args:
            figsize (tuple): Tamanho da figura matplotlib
        """
        self.figsize = figsize
        self.colormap = 'viridis'
        
    def plot_velocity_field(self, results, title="Campo de Velocidade"):
        """
        Plota campo de velocidade com contornos e vetores
        
        Args:
            results (dict): Resultados da simulação CFD
            title (str): Título do gráfico
            
        Returns:
            matplotlib.figure.Figure: Figura criada
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Extrair dados com fallback para chaves diferentes
        if 'mesh_x' in results and 'mesh_y' in results:
            x = results['mesh_x']
            y = results['mesh_y']
        elif 'x' in results and 'y' in results:
            x = results['x']
            y = results['y']
        else:
            # Fallback: criar coordenadas
            u_data = results.get('velocity_u', results.get('u', []))
            if len(u_data) > 0:
                if len(u_data.shape) == 1:
                    size = int(np.sqrt(len(u_data)))
                    x = np.linspace(0, 4, size)
                    y = np.linspace(0, 2, size)
                else:
                    ny, nx = u_data.shape
                    x = np.linspace(0, 4, nx)
                    y = np.linspace(0, 2, ny)
            else:
                x = np.linspace(0, 4, 40)
                y = np.linspace(0, 2, 20)
        
        u = results.get('velocity_u', results.get('u', np.zeros_like(x)))
        v = results.get('velocity_v', results.get('v', np.zeros_like(y)))
        velocity_mag = results.get('velocity_magnitude', np.sqrt(u**2 + v**2))
        
        # Reformatar para contorno se necessário
        if len(x.shape) == 1:
            # Assumir malha estruturada
            nx = len(np.unique(x))
            ny = len(np.unique(y))
            X = x.reshape(ny, nx)
            Y = y.reshape(ny, nx)
            U = u.reshape(ny, nx)
            V = v.reshape(ny, nx)
            Vmag = velocity_mag.reshape(ny, nx)
        else:
            X, Y, U, V, Vmag = x, y, u, v, velocity_mag
        
        # Gráfico 1: Magnitude da velocidade
        contour1 = ax1.contourf(X, Y, Vmag, levels=20, cmap=self.colormap)
        ax1.set_title(f"{title} - Magnitude")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_aspect('equal')
        plt.colorbar(contour1, ax=ax1, label="Velocidade (m/s)")
        
        # Adicionar vetores de velocidade (subamostrados)
        skip = max(1, len(X) // 20)  # Mostrar ~20 vetores por direção
        ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  U[::skip, ::skip], V[::skip, ::skip], 
                  alpha=0.7, scale=None, width=0.003)
        
        # Gráfico 2: Componente u
        contour2 = ax2.contourf(X, Y, U, levels=20, cmap=self.colormap)
        ax2.set_title(f"{title} - Componente U")
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("y (m)")
        ax2.set_aspect('equal')
        plt.colorbar(contour2, ax=ax2, label="Velocidade U (m/s)")
        
        plt.tight_layout()
        return fig
    
    def plot_pressure_field(self, results, title="Campo de Pressão"):
        """
        Plota campo de pressão
        
        Args:
            results (dict): Resultados da simulação CFD
            title (str): Título do gráfico
            
        Returns:
            matplotlib.figure.Figure: Figura criada
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Extrair dados
        x = results['mesh_x']
        y = results['mesh_y']
        p = results['pressure']
        
        # Reformatar para contorno se necessário
        if len(x.shape) == 1:
            nx = len(np.unique(x))
            ny = len(np.unique(y))
            X = x.reshape(ny, nx)
            Y = y.reshape(ny, nx)
            P = p.reshape(ny, nx)
        else:
            X, Y, P = x, y, p
        
        # Contorno de pressão
        contour = ax.contourf(X, Y, P, levels=20, cmap='RdBu_r')
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal')
        plt.colorbar(contour, ax=ax, label="Pressão (Pa)")
        
        # Adicionar linhas de contorno
        ax.contour(X, Y, P, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_streamlines(self, results, streamlines=None, title="Linhas de Corrente"):
        """
        Plota linhas de corrente
        
        Args:
            results (dict): Resultados da simulação CFD
            streamlines (list): Lista de streamlines [(x_array, y_array), ...]
            title (str): Título do gráfico
            
        Returns:
            matplotlib.figure.Figure: Figura criada
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Extrair dados
        x = results['mesh_x']
        y = results['mesh_y']
        u = results['velocity_u']
        v = results['velocity_v']
        velocity_mag = results['velocity_magnitude']
        
        # Reformatar para contorno se necessário
        if len(x.shape) == 1:
            nx = len(np.unique(x))
            ny = len(np.unique(y))
            X = x.reshape(ny, nx)
            Y = y.reshape(ny, nx)
            U = u.reshape(ny, nx)
            V = v.reshape(ny, nx)
            Vmag = velocity_mag.reshape(ny, nx)
        else:
            X, Y, U, V, Vmag = x, y, u, v, velocity_mag
        
        # Campo de velocidade como fundo
        contour = ax.contourf(X, Y, Vmag, levels=20, cmap=self.colormap, alpha=0.7)
        plt.colorbar(contour, ax=ax, label="Velocidade (m/s)")
        
        # Plotar streamlines se fornecidas
        if streamlines:
            for x_line, y_line in streamlines:
                ax.plot(x_line, y_line, 'white', linewidth=1.5, alpha=0.8)
        else:
            # Usar matplotlib streamplot
            ax.streamplot(X, Y, U, V, color='white', linewidth=1, density=2)
        
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def plot_geometry_2d(self, points, title="Geometria 2D"):
        """
        Plota geometria 2D
        
        Args:
            points (np.array): Pontos da geometria
            title (str): Título do gráfico
            
        Returns:
            matplotlib.figure.Figure: Figura criada
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if points is not None and len(points) > 0:
            # Plotar contorno
            ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Contorno')
            
            # Fechar o contorno se necessário
            if not np.allclose(points[0], points[-1]):
                ax.plot([points[-1, 0], points[0, 0]], 
                       [points[-1, 1], points[0, 1]], 'b-', linewidth=2)
            
            # Preencher área
            ax.fill(points[:, 0], points[:, 1], alpha=0.3, color='blue')
            
            # Configurar limites
            margin = 0.1
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            ax.set_xlim(np.min(points[:, 0]) - margin * x_range,
                       np.max(points[:, 0]) + margin * x_range)
            ax.set_ylim(np.min(points[:, 1]) - margin * y_range,
                       np.max(points[:, 1]) + margin * y_range)
        
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_interactive_plot(self, results, plot_type="velocity"):
        """
        Cria gráfico interativo com Plotly
        
        Args:
            results (dict): Resultados da simulação CFD
            plot_type (str): Tipo de gráfico ("velocity", "pressure", "streamlines")
            
        Returns:
            plotly.graph_objects.Figure: Figura Plotly
        """
        # Extrair dados
        x = results['mesh_x']
        y = results['mesh_y']
        
        if plot_type == "velocity":
            z = results['velocity_magnitude']
            title = "Campo de Velocidade (Interativo)"
            colorbar_title = "Velocidade (m/s)"
            colorscale = 'Viridis'
        elif plot_type == "pressure":
            z = results['pressure']
            title = "Campo de Pressão (Interativo)"
            colorbar_title = "Pressão (Pa)"
            colorscale = 'RdBu'
        else:
            z = results['velocity_magnitude']
            title = "Campo CFD (Interativo)"
            colorbar_title = "Magnitude"
            colorscale = 'Viridis'
        
        # Reformatar dados se necessário
        if len(x.shape) == 1:
            nx = len(np.unique(x))
            ny = len(np.unique(y))
            X = x.reshape(ny, nx)
            Y = y.reshape(ny, nx)
            Z = z.reshape(ny, nx)
        else:
            X, Y, Z = x, y, z
            ny, nx = X.shape
        
        # Criar gráfico de contorno
        fig = go.Figure(data=go.Contour(
            x=X[0, :],  # Primeira linha para coordenadas x
            y=Y[:, 0],  # Primeira coluna para coordenadas y
            z=Z,
            colorscale=colorscale,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color="white")
            ),
            colorbar=dict(title=colorbar_title)
        ))
        
        # Adicionar vetores de velocidade se for campo de velocidade
        if plot_type == "velocity" and 'velocity_u' in results:
            u = results['velocity_u']
            v = results['velocity_v']
            
            if len(u.shape) == 1:
                U = u.reshape(ny, nx)
                V = v.reshape(ny, nx)
            else:
                U, V = u, v
            
            # Subamostrar vetores para reduzir dados
            skip = max(1, min(nx, ny) // 8)  # Reduzido de 15 para 8
            
            fig.add_trace(go.Scatter(
                x=X[::skip, ::skip].flatten(),
                y=Y[::skip, ::skip].flatten(),
                mode='markers',
                marker=dict(
                    size=8,
                    symbol='arrow',
                    angle=np.arctan2(V[::skip, ::skip], U[::skip, ::skip]).flatten() * 180 / np.pi,
                    color='white',
                    line=dict(width=1, color='black')
                ),
                name='Vetores de Velocidade',
                showlegend=False
            ))
        
        # Configurar layout
        fig.update_layout(
            title=title,
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            width=800,
            height=500,
            showlegend=False
        )
        
        # Manter proporção
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        return fig
    
    def plot_convergence(self, residuals, title="Convergência da Simulação"):
        """
        Plota histórico de convergência
        
        Args:
            residuals (list): Lista de resíduos por iteração
            title (str): Título do gráfico
            
        Returns:
            matplotlib.figure.Figure: Figura criada
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        iterations = range(1, len(residuals) + 1)
        ax.semilogy(iterations, residuals, 'b-', linewidth=2)
        
        ax.set_title(title)
        ax.set_xlabel("Iteração")
        ax.set_ylabel("Resíduo (log)")
        ax.grid(True, alpha=0.3)
        
        # Adicionar linha de tolerância se aplicável
        if len(residuals) > 0:
            tolerance = min(residuals) * 10
            ax.axhline(y=tolerance, color='r', linestyle='--', 
                      label=f'Tolerância: {tolerance:.2e}')
            ax.legend()
        
        plt.tight_layout()
        return fig


def display_results_summary(results):
    """
    Exibe resumo dos resultados no Streamlit
    
    Args:
        results (dict): Resultados da simulação CFD
    """
    st.subheader("Resumo dos Resultados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'drag_coefficient' in results:
            st.metric("Coeficiente de Arrasto", f"{results['drag_coefficient']:.4f}")
        
        if 'velocity_magnitude' in results:
            max_vel = np.max(results['velocity_magnitude'])
            st.metric("Velocidade Máxima", f"{max_vel:.2f} m/s")
    
    with col2:
        if 'pressure' in results:
            max_pressure = np.max(results['pressure'])
            min_pressure = np.min(results['pressure'])
            st.metric("Pressão Máxima", f"{max_pressure:.0f} Pa")
            st.metric("Pressão Mínima", f"{min_pressure:.0f} Pa")
    
    with col3:
        if 'iterations' in results:
            st.metric("Iterações", results['iterations'])
        
        if 'converged' in results:
            status = "✅ Convergiu" if results['converged'] else "❌ Não convergiu"
            st.metric("Status", status)


def create_comparison_plot(results_traditional, results_ai, field_type="velocity"):
    """
    Cria gráfico de comparação entre métodos tradicional e IA
    
    Args:
        results_traditional (dict): Resultados do método tradicional
        results_ai (dict): Resultados do método IA
        field_type (str): Tipo de campo a comparar
        
    Returns:
        matplotlib.figure.Figure: Figura de comparação
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    if field_type == "velocity":
        field_trad = results_traditional['velocity_magnitude']
        field_ai = results_ai['velocity_magnitude']
        title_base = "Velocidade"
        unit = "m/s"
    else:
        field_trad = results_traditional['pressure']
        field_ai = results_ai['pressure']
        title_base = "Pressão"
        unit = "Pa"
    
    # Método tradicional
    x_trad = results_traditional['mesh_x']
    y_trad = results_traditional['mesh_y']
    
    if len(x_trad.shape) == 1:
        nx = len(np.unique(x_trad))
        ny = len(np.unique(y_trad))
        X_trad = x_trad.reshape(ny, nx)
        Y_trad = y_trad.reshape(ny, nx)
        Field_trad = field_trad.reshape(ny, nx)
    else:
        X_trad, Y_trad, Field_trad = x_trad, y_trad, field_trad
    
    # Método IA
    X_ai = results_ai['mesh_x']
    Y_ai = results_ai['mesh_y']
    Field_ai = field_ai
    
    # Plotar resultados
    im1 = axes[0, 0].contourf(X_trad, Y_trad, Field_trad, levels=20, cmap='viridis')
    axes[0, 0].set_title(f"{title_base} - Método Tradicional")
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0], label=unit)
    
    im2 = axes[0, 1].contourf(X_ai, Y_ai, Field_ai, levels=20, cmap='viridis')
    axes[0, 1].set_title(f"{title_base} - Método IA")
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1], label=unit)
    
    # Diferença (se as malhas forem compatíveis)
    try:
        if X_trad.shape == X_ai.shape:
            diff = Field_trad - Field_ai
            im3 = axes[1, 0].contourf(X_trad, Y_trad, diff, levels=20, cmap='RdBu_r')
            axes[1, 0].set_title(f"Diferença ({title_base})")
            axes[1, 0].set_aspect('equal')
            plt.colorbar(im3, ax=axes[1, 0], label=f"Δ{unit}")
        else:
            axes[1, 0].text(0.5, 0.5, "Malhas incompatíveis\npara comparação", 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
    except:
        axes[1, 0].text(0.5, 0.5, "Erro na comparação", 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Estatísticas
    stats_text = f"""
    Método Tradicional:
    Máximo: {np.max(field_trad):.2f} {unit}
    Mínimo: {np.min(field_trad):.2f} {unit}
    Média: {np.mean(field_trad):.2f} {unit}
    
    Método IA:
    Máximo: {np.max(field_ai):.2f} {unit}
    Mínimo: {np.min(field_ai):.2f} {unit}
    Média: {np.mean(field_ai):.2f} {unit}
    """
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                   verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_title("Estatísticas Comparativas")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig

