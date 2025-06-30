# 🌪️ Simulador CFD Aerodinâmico

Um aplicativo completo de simulação de dinâmica de fluidos computacional (CFD) para análise aerodinâmica de modelos 3D, desenvolvido com Streamlit, FiPy e PyTorch.

ATENÇÂO: Esse MVP simula diversas condições de CFD, porém ainda não resolve cálculos de CFD reais.

## 🚀 Funcionalidades

### 📁 Upload e Processamento de Modelos 3D
- **Upload de arquivos STL**: Carregue modelos 3D em formato STL
- **Validação automática**: Verificação da qualidade e integridade da malha
- **Projeção 2D**: Conversão automática para simulação CFD 2D
- **Aerofólio NACA padrão**: Opção de usar geometria de teste integrada

### 🔬 Simulação CFD Tradicional
- **Motor FiPy**: Simulações numéricas usando equações de Navier-Stokes
- **Convergência monitorada**: Acompanhamento em tempo real do progresso
- **Cálculos aerodinâmicos**: Coeficientes de arrasto e sustentação
- **Streamlines**: Visualização de linhas de corrente

### 🤖 Simulação com IA
- **Modelo substituto**: Redes neurais para predições rápidas
- **Treinamento automático**: Modelo treinado com dados sintéticos
- **Performance otimizada**: Resultados em segundos vs. minutos

### 📊 Visualizações Avançadas
- **Gráficos 2D**: Contornos de velocidade, pressão e streamlines
- **Visualizações 3D**: Modelos tridimensionais com PyVista
- **Gráficos interativos**: Interface Plotly para exploração
- **Comparação de métodos**: Análise lado a lado dos resultados

## 🛠️ Instalação

### Pré-requisitos
- Python 3.8+
- pip ou conda

### Dependências
```bash
pip install -r requirements.txt
```

### Principais bibliotecas:
- **Streamlit**: Interface web
- **FiPy**: Simulações CFD
- **PyTorch**: Modelos de IA
- **PyVista**: Visualizações 3D
- **Trimesh**: Processamento de malhas
- **Matplotlib/Plotly**: Gráficos

## 🚀 Como Usar

### 1. Executar o Aplicativo
```bash
streamlit run app.py
```

### 2. Acessar a Interface
Abra o navegador em `http://localhost:8501`

### 3. Fluxo de Trabalho

#### Passo 1: Upload do Modelo
1. Vá para a aba "📁 Upload & Geometria"
2. Carregue um arquivo STL ou use o aerofólio NACA padrão
3. Visualize as informações da malha e projeções

#### Passo 2: Configurar Simulação
- **Velocidade de entrada**: 10-100 m/s
- **Iterações máximas**: 50-500
- **Tolerância**: 1e-3 a 1e-6

#### Passo 3: Executar Simulações
- **Simulação Tradicional**: Aba "🔬 Simulação Tradicional"
- **Simulação IA**: Aba "🤖 Simulação IA"

#### Passo 4: Analisar Resultados
- Visualize campos de velocidade e pressão
- Examine streamlines e convergência
- Compare métodos na aba "📊 Comparação & Resultados"

## 📋 Estrutura do Projeto

```
cfd_app/
├── app.py                 # Aplicação principal Streamlit
├── requirements.txt       # Dependências
├── README.md             # Documentação
├── cfd/                  # Módulo CFD
│   ├── __init__.py
│   ├── simulation.py     # Simulações FiPy
│   ├── geometry.py       # Processamento de geometria
│   └── ai_model.py       # Modelos de IA
├── visualization/        # Módulo de visualização
│   ├── __init__.py
│   ├── plots_2d.py       # Gráficos 2D
│   └── plots_3d.py       # Visualizações 3D
└── utils/               # Utilitários
    ├── __init__.py
    └── helpers.py       # Funções auxiliares
```

## 🔧 Configurações Avançadas

### Parâmetros de Simulação
- **Densidade do ar**: 1.225 kg/m³
- **Viscosidade**: 1.8e-5 Pa·s
- **Domínio**: 4.0m × 2.0m (padrão)
- **Malha**: 80×40 células (padrão)

### Modelo de IA
- **Arquitetura**: Rede neural feedforward
- **Camadas**: [64, 128, 64] neurônios
- **Ativação**: ReLU
- **Otimizador**: Adam

## 📊 Interpretação dos Resultados

### Métricas Principais
- **Coeficiente de Arrasto (Cd)**: Resistência aerodinâmica
- **Velocidade Máxima**: Pico de velocidade no domínio
- **Pressão**: Distribuição de pressão na superfície

### Visualizações
- **Contornos de velocidade**: Magnitude do campo de velocidade
- **Contornos de pressão**: Distribuição de pressão
- **Streamlines**: Trajetórias do fluido
- **Vetores**: Direção e magnitude local

## ⚠️ Limitações

### Simulação Tradicional
- **2D apenas**: Projeção de modelos 3D para 2D
- **Regime laminar**: Não considera turbulência
- **Geometria simplificada**: Aproximações na representação

### Modelo de IA
- **Dados sintéticos**: Treinado com dados simulados
- **Domínio limitado**: Válido para condições específicas
- **Aproximações**: Resultados indicativos, não precisos

## 🔍 Solução de Problemas

### Erros Comuns

#### "Erro ao carregar STL"
- Verifique se o arquivo é um STL válido
- Confirme que o arquivo não está corrompido
- Tente usar o aerofólio NACA padrão

#### "Simulação não converge"
- Reduza a tolerância de convergência
- Aumente o número máximo de iterações
- Verifique a qualidade da malha

#### "Modelo de IA não disponível"
- Aguarde o treinamento automático
- Verifique se PyTorch está instalado corretamente

### Performance
- **Simulações lentas**: Reduza resolução da malha
- **Uso de memória**: Feche outras aplicações
- **GPU**: Use CUDA se disponível

## 🤝 Contribuições

Este projeto foi desenvolvido como demonstração de capacidades CFD em Python. Contribuições são bem-vindas:

1. Fork o repositório
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Abra um Pull Request

## 📚 Referências

- **FiPy**: https://www.ctcms.nist.gov/fipy/
- **Streamlit**: https://streamlit.io/
- **PyVista**: https://pyvista.org/
- **CFD Theory**: Anderson, J.D. "Computational Fluid Dynamics"

## 📄 Licença

Este projeto é fornecido como está, para fins educacionais e de demonstração.

## 🏷️ Versão

**v1.0.0** - Versão inicial completa

---

**Desenvolvido com ❤️ usando Python, Streamlit, FiPy e PyTorch**

