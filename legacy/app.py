"""
Aplicação Principal - Simulador CFD Aerodinâmico
Aplicativo Streamlit para simulação de dinâmica de fluidos em modelos 3D
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import tempfile
import os
import torch
from io import BytesIO

# Importar módulos locais
from cfd.simulation import CFDSimulation, create_simple_simulation
from cfd.geometry import GeometryProcessor, process_model_for_cfd
from cfd.ai_model import CFDAIModel, create_pretrained_model
from visualization.plots_2d import CFDPlotter2D, display_results_summary, create_comparison_plot
from visualization.plots_3d import CFDPlotter3D, create_3d_visualization_from_stl

# Configuração da página
st.set_page_config(
    page_title="Simulador CFD Aerodinâmico",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Inicializa variáveis de estado da sessão"""
    if 'geometry_processor' not in st.session_state:
        st.session_state.geometry_processor = None
    if 'mesh_3d' not in st.session_state:
        st.session_state.mesh_3d = None
    if 'simulation_points' not in st.session_state:
        st.session_state.simulation_points = None
    if 'cfd_results' not in st.session_state:
        st.session_state.cfd_results = None
    if 'ai_results' not in st.session_state:
        st.session_state.ai_results = None
    if 'ai_model' not in st.session_state:
        st.session_state.ai_model = None

def create_progress_callback():
    """Cria callback para atualizar progresso da simulação"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    device_info = st.empty()
    
    def update_progress(progress, iteration, residual, elapsed_time):
        progress_bar.progress(progress)
        
        # Estimar tempo restante
        if progress > 0:
            total_time = elapsed_time / progress
            remaining_time = total_time - elapsed_time
            time_str = f"Tempo restante: {remaining_time:.1f}s"
        else:
            time_str = "Calculando..."
        
        status_text.text(
            f"Iteração {iteration} | Resíduo: {residual:.2e} | {time_str}"
        )
        
        # Mostrar informações da GPU se disponível
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1e6  # MB
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e6  # MB
            gpu_usage = gpu_mem / total_mem * 100
            gpu_name = torch.cuda.get_device_name(0)
            
            device_info.info(
                f"🚀 GPU: {gpu_name} | Uso de memória: {gpu_mem:.1f}/{total_mem:.1f} MB ({gpu_usage:.1f}%)"
            )
        else:
            device_info.info("💻 Executando em CPU")
    
    return update_progress

def load_model_file(uploaded_file):
    """Carrega e processa modelo 3D (STL, GLB, GLTF ou OBJ)"""
    try:
        file_type = uploaded_file.name.rsplit('.', 1)[-1].lower()

        # Processar modelo (vista lateral 'xz': X = fluxo, Z = altura)
        processor, simulation_points, mesh_info = process_model_for_cfd(
            uploaded_file, projection_plane='xz', file_type=file_type
        )

        # Salvar no estado da sessão
        st.session_state.geometry_processor = processor
        st.session_state.mesh_3d = processor.mesh_3d
        st.session_state.simulation_points = simulation_points

        return processor, simulation_points, mesh_info

    except Exception as e:
        st.error(f"Erro ao carregar modelo 3D: {str(e)}")
        return None, None, None

def run_traditional_simulation(simulation_points, progress_callback=None):
    """Executa simulação CFD tradicional com otimização GPU"""
    try:
        # Detectar dispositivo disponível
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Criar simulação otimizada
        from cfd.simulation import create_gpu_optimized_simulation
        sim = create_gpu_optimized_simulation()

        # Configurar parâmetros baseados na interface
        sim.inlet_velocity = st.session_state.get('inlet_velocity', 30.0)

        # Imergir a geometria real carregada no domínio (túnel de vento):
        # rasteriza a silhueta lateral do modelo na grade do solver
        processor = st.session_state.get('geometry_processor')
        if processor is not None:
            try:
                mask = processor.build_occupancy_mask(
                    sim.nx, sim.ny,
                    domain_width=sim.Lx, domain_height=sim.Ly
                )
                sim.set_object_mask(mask)
            except Exception as mask_err:
                st.warning(
                    f"Não foi possível usar a geometria carregada como "
                    f"obstáculo ({mask_err}); usando aerofólio padrão."
                )
        
        # Executar simulação
        results = sim.solve_steady_state(
            max_iterations=st.session_state.get('max_iterations', 200),
            tolerance=st.session_state.get('tolerance', 1e-5),
            progress_callback=progress_callback
        )
        
        # Calcular streamlines
        streamlines = sim.calculate_streamlines(n_streamlines=15)
        results['streamlines'] = streamlines
        
        # Adicionar informações do dispositivo
        results['device_used'] = str(device)
        results['gpu_available'] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            results['gpu_memory_used'] = torch.cuda.memory_allocated(device) / 1e6  # MB
            results['gpu_name'] = torch.cuda.get_device_name(device)
        
        return results
        
    except Exception as e:
        st.error(f"Erro na simulação tradicional: {str(e)}")
        return None

def run_ai_simulation():
    """Executa simulação usando IA"""
    try:
        # Verificar se modelo existe
        if st.session_state.ai_model is None:
            with st.spinner("Treinando modelo de IA..."):
                if torch.cuda.is_available():
                    # liberar VRAM em cache antes de alocar o modelo
                    torch.cuda.empty_cache()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                st.session_state.ai_model = create_pretrained_model(device=device)
        
        # Executar predição
        results = st.session_state.ai_model.predict_field(
            domain_size=(4.0, 2.0),
            resolution=(80, 40)
        )
        
        # Calcular coeficientes aerodinâmicos aproximados
        velocity_mag = results['velocity_magnitude']
        results['drag_coefficient'] = np.mean(velocity_mag) * 0.01  # Aproximação simples
        
        return results
        
    except Exception as e:
        st.error(f"Erro na simulação IA: {str(e)}")
        return None

def main():
    """Função principal da aplicação"""
    # Inicializar estado
    initialize_session_state()
    
    # Cabeçalho
    st.markdown('<h1 class="main-header">🌪️ Simulador CFD Aerodinâmico</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Bem-vindo ao Simulador CFD Aerodinâmico!</strong><br>
    Este aplicativo permite carregar modelos 3D em formato STL e realizar análises 
    de dinâmica de fluidos usando métodos tradicionais (FiPy) ou modelos de IA.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Informações do sistema
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"**Dispositivo:** {device}")
        
        if torch.cuda.is_available():
            st.info(f"**GPU:** {torch.cuda.get_device_name(0)}")
        
        st.markdown("---")
        
        # Parâmetros de simulação
        st.subheader("Parâmetros de Simulação")
        
        st.session_state.inlet_velocity = st.slider(
            "Velocidade de Entrada (m/s)",
            min_value=10.0, max_value=100.0, value=30.0, step=5.0
        )
        
        st.session_state.max_iterations = st.slider(
            "Máximo de Iterações",
            min_value=50, max_value=300, value=100, step=25  # Reduzido para evitar dados grandes
        )
        
        st.session_state.tolerance = st.select_slider(
            "Tolerância de Convergência",
            options=[1e-3, 1e-4, 1e-5, 1e-6],
            value=1e-4,  # Aumentado para convergir mais rápido
            format_func=lambda x: f"{x:.0e}"
        )
        
        st.markdown("---")
        
        # Opções de visualização
        st.subheader("Visualizações")
        show_3d = st.checkbox("Mostrar visualizações 3D", value=True)
        show_comparison = st.checkbox("Comparar métodos", value=False)
    
    # Área principal
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Upload & Geometria", 
        "🔬 Simulação Tradicional", 
        "🤖 Simulação IA", 
        "📊 Comparação & Resultados"
    ])
    
    # Tab 1: Upload e Geometria
    with tab1:
        st.markdown('<h2 class="section-header">Upload do Modelo 3D</h2>', 
                    unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Carregue um modelo 3D (STL, GLB, GLTF ou OBJ)",
            type=['stl', 'glb', 'gltf', 'obj'],
            help="Selecione um modelo 3D para análise aerodinâmica. "
                 "A orientação é normalizada automaticamente: o maior eixo "
                 "vira a direção do fluxo."
        )

        if uploaded_file is not None:
            with st.spinner("Processando modelo 3D..."):
                processor, simulation_points, mesh_info = load_model_file(uploaded_file)

            if processor is not None:
                st.success("✅ Modelo 3D carregado com sucesso!")
                
                # Informações da malha
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Informações da Malha")
                    st.metric("Vértices", f"{mesh_info['vertices']:,}")
                    st.metric("Faces", f"{mesh_info['faces']:,}")
                    st.metric("Volume", f"{mesh_info['volume']:.4f}")
                    st.metric("Área Superficial", f"{mesh_info['surface_area']:.4f}")
                
                with col2:
                    st.subheader("🔍 Qualidade da Malha")
                    st.metric("Watertight", "✅" if mesh_info['is_watertight'] else "❌")
                    st.metric("Winding Consistent", "✅" if mesh_info['is_winding_consistent'] else "❌")
                    
                    # Dimensões
                    extents = mesh_info['extents']
                    st.metric("Dimensões (X×Y×Z)", f"{extents[0]:.3f}×{extents[1]:.3f}×{extents[2]:.3f}")
                
                # Visualizações da geometria
                st.subheader("🎨 Visualizações da Geometria")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Visualização 2D
                    plotter_2d = CFDPlotter2D()
                    fig_2d = plotter_2d.plot_geometry_2d(
                        simulation_points, 
                        "Projeção 2D para Simulação"
                    )
                    st.pyplot(fig_2d)
                
                with viz_col2:
                    # Visualização 3D interativa
                    if show_3d:
                        plotter_3d = CFDPlotter3D()
                        fig_3d_interactive = plotter_3d.plot_mesh_interactive(processor.mesh_3d)
                        st.plotly_chart(fig_3d_interactive, use_container_width=True)
                        
                        # Opção de visualizar com streamlines
                        if st.button("🌪️ Visualizar com Streamlines 3D"):
                            fig_streamlines = plotter_3d.plot_mesh_with_streamlines(processor.mesh_3d)
                            st.plotly_chart(fig_streamlines, use_container_width=True)
                        
                        # Opção de animação do túnel de vento
                        if st.button("🎬 Animação do Túnel de Vento"):
                            with st.spinner("Gerando animação do túnel de vento..."):
                                fig_animation = plotter_3d.create_wind_tunnel_animation(processor.mesh_3d)
                                st.plotly_chart(fig_animation, use_container_width=True)
                                st.success("✅ Animação gerada! Use os controles Play/Pause para controlar a animação.")
        
        else:
            st.info("👆 Carregue um modelo 3D (STL/GLB) para começar a análise")
            
            # Opção de usar geometria padrão
            if st.button("🔧 Usar Aerofólio NACA Padrão"):
                with st.spinner("Criando geometria padrão..."):
                    processor = GeometryProcessor()
                    simulation_points = processor.create_airfoil_points(n_points=50)
                    simulation_points = processor.adapt_to_simulation_domain()
                    
                    st.session_state.geometry_processor = processor
                    st.session_state.simulation_points = simulation_points
                    
                st.success("✅ Aerofólio NACA 0012 carregado!")
                st.rerun()
    
    # Tab 2: Simulação Tradicional
    with tab2:
        st.markdown('<h2 class="section-header">Simulação CFD Tradicional (FiPy)</h2>', 
                    unsafe_allow_html=True)
        
        if st.session_state.simulation_points is not None:
            
            if st.button("🚀 Executar Simulação Tradicional", type="primary"):
                
                with st.spinner("Executando simulação CFD..."):
                    # Criar callback de progresso
                    progress_callback = create_progress_callback()
                    
                    # Executar simulação
                    results = run_traditional_simulation(
                        st.session_state.simulation_points,
                        progress_callback
                    )
                
                if results is not None:
                    st.session_state.cfd_results = results
                    st.success("✅ Simulação tradicional concluída!")
                    
                    # Exibir resumo
                    display_results_summary(results)
                    
                    # Visualizações
                    st.subheader("📈 Resultados da Simulação")
                    
                    plotter = CFDPlotter2D()
                    
                    # Gráficos de velocidade
                    fig_vel = plotter.plot_velocity_field(results)
                    st.pyplot(fig_vel)
                    
                    # Gráfico de pressão
                    fig_press = plotter.plot_pressure_field(results)
                    st.pyplot(fig_press)
                    
                    # Streamlines
                    if 'streamlines' in results:
                        fig_stream = plotter.plot_streamlines(results)
                        st.pyplot(fig_stream)
                    
                    # Convergência
                    if 'residuals' in results:
                        fig_conv = plotter.plot_convergence(results['residuals'])
                        st.pyplot(fig_conv)
                    
                    # Gráfico interativo
                    st.subheader("🎯 Visualização Interativa")
                    interactive_fig = plotter.create_interactive_plot(results, "velocity")
                    st.plotly_chart(interactive_fig, use_container_width=True)
        
        else:
            st.warning("⚠️ Carregue um modelo 3D primeiro na aba 'Upload & Geometria'")
    
    # Tab 3: Simulação IA
    with tab3:
        st.markdown('<h2 class="section-header">Simulação com Modelo de IA</h2>', 
                    unsafe_allow_html=True)
        
        st.info("""
        **Modelo Substituto de IA**: Utiliza redes neurais treinadas para predizer 
        campos de velocidade e pressão de forma rápida, ideal para análises preliminares.
        """)
        
        if st.button("🤖 Executar Simulação IA", type="primary"):
            
            with st.spinner("Executando simulação com IA..."):
                results = run_ai_simulation()
            
            if results is not None:
                st.session_state.ai_results = results
                st.success("✅ Simulação IA concluída!")
                
                # Exibir resumo
                display_results_summary(results)
                
                # Visualizações
                st.subheader("📈 Resultados da Simulação IA")
                
                plotter = CFDPlotter2D()
                
                # Gráficos de velocidade
                fig_vel = plotter.plot_velocity_field(results, "Campo de Velocidade (IA)")
                st.pyplot(fig_vel)
                
                # Gráfico de pressão
                fig_press = plotter.plot_pressure_field(results, "Campo de Pressão (IA)")
                st.pyplot(fig_press)
                
                # Gráfico interativo
                st.subheader("🎯 Visualização Interativa")
                interactive_fig = plotter.create_interactive_plot(results, "velocity")
                st.plotly_chart(interactive_fig, use_container_width=True)
    
    # Tab 4: Comparação e Resultados
    with tab4:
        st.markdown('<h2 class="section-header">Comparação de Métodos</h2>', 
                    unsafe_allow_html=True)
        
        if (st.session_state.cfd_results is not None and 
            st.session_state.ai_results is not None):
            
            st.success("✅ Ambas as simulações foram executadas!")
            
            # Comparação de métricas
            st.subheader("📊 Comparação de Métricas")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Coef. Arrasto (Tradicional)",
                    f"{st.session_state.cfd_results.get('drag_coefficient', 0):.4f}"
                )
                
            with col2:
                st.metric(
                    "Coef. Arrasto (IA)",
                    f"{st.session_state.ai_results.get('drag_coefficient', 0):.4f}"
                )
            
            with col3:
                # Diferença percentual
                cd_trad = st.session_state.cfd_results.get('drag_coefficient', 0)
                cd_ai = st.session_state.ai_results.get('drag_coefficient', 0)
                if cd_trad > 0:
                    diff_percent = abs(cd_ai - cd_trad) / cd_trad * 100
                    st.metric("Diferença (%)", f"{diff_percent:.1f}%")
            
            # Gráficos de comparação (velocidade e pressão no mesmo painel)
            st.subheader("🔍 Comparação Visual")

            fig_comp = create_comparison_plot(
                st.session_state.cfd_results,
                st.session_state.ai_results
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
        else:
            st.info("ℹ️ Execute ambas as simulações para ver a comparação")
            
            # Mostrar status
            status_text = "Status das simulações:\n"
            status_text += f"• Tradicional: {'✅ Concluída' if st.session_state.cfd_results else '❌ Não executada'}\n"
            status_text += f"• IA: {'✅ Concluída' if st.session_state.ai_results else '❌ Não executada'}"
            
            st.text(status_text)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p>🌪️ Simulador CFD Aerodinâmico | Desenvolvido com Streamlit, FiPy e PyTorch</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

