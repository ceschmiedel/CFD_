"""
Módulo de visualizações 2D para simulação CFD
Inclui todas as funções necessárias incluindo plot_geometry_2d
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

class CFDPlotter2D:
    """Classe para criar visualizações 2D dos resultados CFD"""
    
    def __init__(self):
        """Inicializa o plotter 2D"""
        self.fig_size = (12, 8)
        self.dpi = 100
    
    def _extract_coordinates(self, results):
        """
        Extrai coordenadas dos resultados com fallbacks
        
        Args:
            results: Dicionário com resultados da simulação
            
        Returns:
            tuple: (x, y) coordenadas
        """
        # Tentar diferentes chaves para coordenadas
        x = results.get('x', results.get('mesh_x', np.linspace(0, 4, 40)))
        y = results.get('y', results.get('mesh_y', np.linspace(0, 2, 20)))
        
        return x, y
    
    def plot_geometry_2d(self, vertices, faces=None, title="Geometria 2D"):
        """
        Plota geometria 2D (projeção de modelo 3D)
        
        Args:
            vertices: Array de vértices da geometria
            faces: Array de faces (opcional)
            title: Título do gráfico
            
        Returns:
            matplotlib.figure.Figure: Figura criada
        """
        fig, ax = plt.subplots(1, 1, figsize=self.fig_size)
        
        try:
            # Verificar se vertices é válido
            if vertices is None or len(vertices) == 0:
                ax.text(0.5, 0.5, 'Nenhuma geometria carregada', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                return fig
            
            # Converter para numpy array se necessário
            if not isinstance(vertices, np.ndarray):
                vertices = np.array(vertices)
            
            # Garantir que temos pelo menos 2D
            if vertices.ndim == 1:
                vertices = vertices.reshape(-1, 2)
            elif vertices.shape[1] > 2:
                # Usar apenas x e y (projeção 2D)
                vertices = vertices[:, :2]
            
            # Plotar vértices
            ax.scatter(vertices[:, 0], vertices[:, 1], c='blue', s=1, alpha=0.6)
            
            # Plotar faces se disponíveis
            if faces is not None and len(faces) > 0:
                # Limitar número de faces para performance
                max_faces = min(1000, len(faces))
                for i in range(max_faces):
                    if i < len(faces):
                        face = faces[i]
                        if len(face) >= 3:  # Triângulo ou mais
                            # Conectar vértices da face
                            face_vertices = vertices[face[:3]]  # Usar apenas 3 primeiros pontos
                            # Fechar o triângulo
                            face_vertices = np.vstack([face_vertices, face_vertices[0]])
                            ax.plot(face_vertices[:, 0], face_vertices[:, 1], 'r-', alpha=0.3, linewidth=0.5)
            
            ax.set_title(title)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            # Em caso de erro, mostrar mensagem
            ax.text(0.5, 0.5, f'Erro ao plotar geometria: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def plot_velocity_field(self, results):
        """Plota campo de velocidade"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extrair dados com fallbacks
        x, y = self._extract_coordinates(results)
        u = results.get('velocity_u', results.get('u', np.ones_like(x)))
        v = results.get('velocity_v', results.get('v', np.zeros_like(y)))
        velocity_mag = results.get('velocity_magnitude', np.sqrt(u**2 + v**2))
        
        # Reformatar se necessário
        if len(x.shape) == 1:
            X, Y = np.meshgrid(x, y)
            if len(u.shape) == 1:
                U = u.reshape(len(y), len(x))
                V = v.reshape(len(y), len(x))
                Vel_mag = velocity_mag.reshape(len(y), len(x))
            else:
                U, V, Vel_mag = u, v, velocity_mag
        else:
            X, Y = x, y
            U, V, Vel_mag = u, v, velocity_mag
        
        # Plot 1: Magnitude da velocidade
        contour1 = ax1.contourf(X, Y, Vel_mag, levels=20, cmap='viridis')
        ax1.set_title('Magnitude da Velocidade')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        plt.colorbar(contour1, ax=ax1, label='Velocidade (m/s)')
        
        # Plot 2: Vetores de velocidade
        skip = max(1, min(len(x), len(y)) // 10)
        ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  U[::skip, ::skip], V[::skip, ::skip], 
                  Vel_mag[::skip, ::skip], cmap='plasma')
        ax2.set_title('Vetores de Velocidade')
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def plot_pressure_field(self, results):
        """Plota campo de pressão"""
        fig, ax = plt.subplots(1, 1, figsize=self.fig_size)
        
        # Extrair coordenadas
        x, y = self._extract_coordinates(results)
        pressure = results.get('pressure', np.zeros_like(x))
        
        # Reformatar se necessário
        if len(x.shape) == 1:
            X, Y = np.meshgrid(x, y)
            if len(pressure.shape) == 1:
                P = pressure.reshape(len(y), len(x))
            else:
                P = pressure
        else:
            X, Y = x, y
            P = pressure
        
        # Plotar campo de pressão
        contour = ax.contourf(X, Y, P, levels=20, cmap='RdBu_r')
        ax.set_title('Campo de Pressão')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        plt.colorbar(contour, ax=ax, label='Pressão (Pa)')
        
        plt.tight_layout()
        return fig
    
    def plot_streamlines(self, results):
        """
        Plota linhas de corrente realistas ao redor do objeto
        
        Args:
            results: Dicionário com resultados da simulação
            
        Returns:
            matplotlib.figure.Figure: Figura criada
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Extrair coordenadas
        x, y = self._extract_coordinates(results)
        
        # Extrair dados de velocidade
        u = results.get('velocity_u', results.get('u', np.zeros_like(x)))
        v = results.get('velocity_v', results.get('v', np.zeros_like(y)))
        velocity_mag = results.get('velocity_magnitude', np.sqrt(u**2 + v**2))
        
        # Extrair máscara do objeto se disponível
        object_mask = results.get('object_mask', None)
        
        # Reformatar para streamlines
        if len(x.shape) == 1:
            nx, ny = len(x), len(y)
            X, Y = np.meshgrid(x, y)
            
            if len(u.shape) == 1:
                U = u.reshape(ny, nx)
                V = v.reshape(ny, nx)
                Vel_mag = velocity_mag.reshape(ny, nx)
            else:
                U, V, Vel_mag = u, v, velocity_mag
        else:
            X, Y = x, y
            U, V, Vel_mag = u, v, velocity_mag
        
        # Mascarar velocidades dentro do objeto
        if object_mask is not None:
            if len(object_mask.shape) == 1:
                mask_2d = object_mask.reshape(ny, nx)
            else:
                mask_2d = object_mask
            
            # Zerar velocidades dentro do objeto
            U_masked = np.ma.masked_where(mask_2d, U)
            V_masked = np.ma.masked_where(mask_2d, V)
            Vel_mag_masked = np.ma.masked_where(mask_2d, Vel_mag)
        else:
            U_masked = U
            V_masked = V
            Vel_mag_masked = Vel_mag
        
        # Plotar campo de velocidade como fundo
        contour = ax.contourf(X, Y, Vel_mag_masked, levels=20, cmap='viridis', alpha=0.7)
        plt.colorbar(contour, ax=ax, label='Velocidade (m/s)')
        
        # Plotar streamlines com pontos de partida estratégicos
        # Pontos de partida na entrada do domínio
        start_points = []
        y_start = np.linspace(0.2, 1.8, 15)  # Distribuir pontos na entrada
        for y_pos in y_start:
            start_points.append([0.1, y_pos])
        
        # Converter para array numpy
        start_points = np.array(start_points)
        
        # Plotar streamlines
        streams = ax.streamplot(X, Y, U_masked, V_masked, 
                               start_points=start_points.T,
                               color='white', 
                               linewidth=1.5,
                               density=1,
                               arrowsize=1.2,
                               arrowstyle='->')
        
        # Plotar contorno do objeto se disponível
        if object_mask is not None:
            if len(object_mask.shape) == 1:
                mask_2d = object_mask.reshape(ny, nx)
            else:
                mask_2d = object_mask
            
            # Contorno do objeto
            ax.contour(X, Y, mask_2d.astype(float), levels=[0.5], colors='red', linewidths=2)
            ax.contourf(X, Y, mask_2d.astype(float), levels=[0.5, 1.5], colors=['red'], alpha=0.8)
        
        ax.set_title('Linhas de Corrente ao Redor do Objeto')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Definir limites do domínio
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 2)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_plot(self, results, field_name="velocity"):
        """
        Cria plot interativo com Plotly
        
        Args:
            results: Dicionário com resultados da simulação
            field_name: Nome do campo a plotar
            
        Returns:
            plotly.graph_objects.Figure: Figura Plotly
        """
        # Extrair coordenadas
        x, y = self._extract_coordinates(results)
        
        # Extrair dados baseado no field_name
        if field_name == "velocity":
            data = results.get('velocity_magnitude', np.sqrt(
                results.get('u', np.ones_like(x))**2 + 
                results.get('v', np.zeros_like(y))**2
            ))
            title = "Magnitude da Velocidade"
            colorbar_title = "Velocidade (m/s)"
        elif field_name == "pressure":
            data = results.get('pressure', np.zeros_like(x))
            title = "Campo de Pressão"
            colorbar_title = "Pressão (Pa)"
        else:
            data = results.get(field_name, np.zeros_like(x))
            title = f"Campo de {field_name}"
            colorbar_title = field_name
        
        # Reformatar dados se necessário
        if len(x.shape) == 1:
            nx, ny = len(x), len(y)
            X, Y = np.meshgrid(x, y)
            
            if len(data.shape) == 1:
                Z = data.reshape(ny, nx)
            else:
                Z = data
        else:
            X, Y = x, y
            Z = data
        
        # Criar figura Plotly
        fig = go.Figure()
        
        # Adicionar contorno
        fig.add_trace(go.Contour(
            x=X[0, :] if len(X.shape) > 1 else x,
            y=Y[:, 0] if len(Y.shape) > 1 else y,
            z=Z,
            colorscale='viridis',
            showscale=True,
            colorbar=dict(title=colorbar_title)
        ))
        
        # Adicionar vetores de velocidade se disponível
        if 'velocity_u' in results and 'velocity_v' in results:
            u = results['velocity_u']
            v = results['velocity_v']
            
            # Reformatar dados se necessário
            if len(u.shape) == 1:
                U = u.reshape(ny, nx)
                V = v.reshape(ny, nx)
            else:
                U, V = u, v
            
            # Subamostrar vetores para reduzir dados
            skip = max(1, min(nx, ny) // 8)  # Reduzido de 15 para 8
            
            # Adicionar vetores
            fig.add_trace(go.Scatter(
                x=X[::skip, ::skip].flatten(),
                y=Y[::skip, ::skip].flatten(),
                mode='markers',
                marker=dict(
                    size=3,
                    symbol='arrow',
                    angle=np.degrees(np.arctan2(V[::skip, ::skip], U[::skip, ::skip])).flatten(),
                    color='white',
                    line=dict(width=1, color='black')
                ),
                showlegend=False,
                name='Vetores de Velocidade'
            ))
        
        # Configurar layout
        fig.update_layout(
            title=title,
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            width=800,
            height=500
        )
        
        # Manter proporção
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        return fig
    
    def plot_convergence(self, residuals):
        """
        Plota histórico de convergência
        
        Args:
            residuals: Lista de resíduos por iteração
            
        Returns:
            matplotlib.figure.Figure: Figura criada
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if residuals and len(residuals) > 0:
            iterations = range(1, len(residuals) + 1)
            ax.semilogy(iterations, residuals, 'b-', linewidth=2)
            ax.set_xlabel('Iteração')
            ax.set_ylabel('Resíduo')
            ax.set_title('Convergência da Simulação')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Nenhum dado de convergência disponível', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Convergência da Simulação')
        
        plt.tight_layout()
        return fig

def display_results_summary(results):
    """
    Exibe resumo dos resultados da simulação
    
    Args:
        results: Dicionário com resultados da simulação
    """
    st.subheader("📊 Resumo dos Resultados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Velocidade máxima
        if 'velocity_magnitude' in results:
            max_vel = np.max(results['velocity_magnitude'])
            st.metric("Velocidade Máxima", f"{max_vel:.2f} m/s")
        
    with col2:
        # Coeficiente de arrasto
        if 'drag_coefficient' in results:
            drag_coef = results['drag_coefficient']
            st.metric("Coeficiente de Arrasto", f"{drag_coef:.3f}")
        
    with col3:
        # Número de iterações
        if 'iterations' in results:
            iterations = results['iterations']
            st.metric("Iterações", f"{iterations}")
    
    # Informações adicionais
    if 'residuals' in results and len(results['residuals']) > 0:
        final_residual = results['residuals'][-1]
        st.info(f"Resíduo final: {final_residual:.2e}")



def create_comparison_plot(traditional_results, ai_results):
    """
    Cria gráfico de comparação entre simulação tradicional e IA
    
    Args:
        traditional_results: Resultados da simulação tradicional
        ai_results: Resultados da simulação IA
        
    Returns:
        plotly.graph_objects.Figure: Figura de comparação
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Criar subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Tradicional - Velocidade', 'IA - Velocidade',
                       'Tradicional - Pressão', 'IA - Pressão'),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Função auxiliar para extrair dados
    def extract_data(results, field_name):
        if results is None:
            return np.zeros((10, 10)), np.linspace(0, 4, 10), np.linspace(0, 2, 10)
        
        x = results.get('x', results.get('mesh_x', np.linspace(0, 4, 40)))
        y = results.get('y', results.get('mesh_y', np.linspace(0, 2, 20)))
        
        if field_name == "velocity":
            data = results.get('velocity_magnitude', np.sqrt(
                results.get('u', np.ones_like(x))**2 + 
                results.get('v', np.zeros_like(y))**2
            ))
        elif field_name == "pressure":
            data = results.get('pressure', np.zeros_like(x))
        else:
            data = results.get(field_name, np.zeros_like(x))
        
        # Reformatar se necessário
        if len(x.shape) == 1:
            X, Y = np.meshgrid(x, y)
            if len(data.shape) == 1:
                Z = data.reshape(len(y), len(x))
            else:
                Z = data
        else:
            Z = data
            
        return Z, x, y
    
    try:
        # Extrair dados de velocidade
        vel_trad, x_trad, y_trad = extract_data(traditional_results, "velocity")
        vel_ai, x_ai, y_ai = extract_data(ai_results, "velocity")
        
        # Extrair dados de pressão
        press_trad, _, _ = extract_data(traditional_results, "pressure")
        press_ai, _, _ = extract_data(ai_results, "pressure")
        
        # Plot 1: Velocidade Tradicional
        fig.add_trace(
            go.Contour(
                z=vel_trad,
                x=x_trad,
                y=y_trad,
                colorscale='viridis',
                showscale=False,
                name='Vel. Tradicional'
            ),
            row=1, col=1
        )
        
        # Plot 2: Velocidade IA
        fig.add_trace(
            go.Contour(
                z=vel_ai,
                x=x_ai,
                y=y_ai,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Velocidade (m/s)", x=0.48),
                name='Vel. IA'
            ),
            row=1, col=2
        )
        
        # Plot 3: Pressão Tradicional
        fig.add_trace(
            go.Contour(
                z=press_trad,
                x=x_trad,
                y=y_trad,
                colorscale='RdBu_r',
                showscale=False,
                name='Press. Tradicional'
            ),
            row=2, col=1
        )
        
        # Plot 4: Pressão IA
        fig.add_trace(
            go.Contour(
                z=press_ai,
                x=x_ai,
                y=y_ai,
                colorscale='RdBu_r',
                showscale=True,
                colorbar=dict(title="Pressão (Pa)", x=1.02),
                name='Press. IA'
            ),
            row=2, col=2
        )
        
    except Exception as e:
        # Em caso de erro, criar plots vazios com mensagem
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_trace(
                    go.Scatter(
                        x=[0.5], y=[0.5],
                        text=[f"Erro: {str(e)}"],
                        mode="text",
                        showlegend=False
                    ),
                    row=row, col=col
                )
    
    # Atualizar layout
    fig.update_layout(
        title="Comparação: Simulação Tradicional vs IA",
        height=600,
        showlegend=False
    )
    
    # Atualizar eixos
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="x (m)", row=i, col=j)
            fig.update_yaxes(title_text="y (m)", row=i, col=j)
    
    return fig

def create_performance_comparison(traditional_time, ai_time, traditional_accuracy=None, ai_accuracy=None):
    """
    Cria gráfico de comparação de performance
    
    Args:
        traditional_time: Tempo da simulação tradicional
        ai_time: Tempo da simulação IA
        traditional_accuracy: Precisão da simulação tradicional (opcional)
        ai_accuracy: Precisão da simulação IA (opcional)
        
    Returns:
        plotly.graph_objects.Figure: Figura de comparação de performance
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Criar subplot
    if traditional_accuracy is not None and ai_accuracy is not None:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Tempo de Execução', 'Precisão'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Gráfico de precisão
        fig.add_trace(
            go.Bar(
                x=['Tradicional', 'IA'],
                y=[traditional_accuracy, ai_accuracy],
                name='Precisão',
                marker_color=['blue', 'orange']
            ),
            row=1, col=2
        )
        
        fig.update_yaxes(title_text="Precisão (%)", row=1, col=2)
        
    else:
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=('Tempo de Execução',)
        )
    
    # Gráfico de tempo
    fig.add_trace(
        go.Bar(
            x=['Tradicional', 'IA'],
            y=[traditional_time, ai_time],
            name='Tempo',
            marker_color=['blue', 'orange'],
            text=[f'{traditional_time:.2f}s', f'{ai_time:.2f}s'],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    fig.update_yaxes(title_text="Tempo (s)", row=1, col=1)
    
    # Calcular speedup
    speedup = traditional_time / ai_time if ai_time > 0 else 1
    
    fig.update_layout(
        title=f"Comparação de Performance - Speedup: {speedup:.1f}x",
        showlegend=False,
        height=400
    )
    
    return fig

