# 🌪️ Simulador CFD Aerodinâmico

Aplicativo de simulação de dinâmica de fluidos computacional (CFD) para análise
aerodinâmica de modelos 3D, desenvolvido com Streamlit, FiPy e PyTorch.

Carregue um carro de F1 (ou qualquer modelo 3D), e o app o coloca dentro de um
túnel de vento virtual 2D: a silhueta real do modelo vira obstáculo no solver,
que calcula campos de velocidade e pressão, coeficientes de arrasto e
sustentação, e visualizações com streamlines alinhadas ao fluxo.

> ⚠️ **Escopo**: este é um MVP educacional. O solver resolve equações de
> momento simplificadas (sem acoplamento pressão-velocidade nem turbulência) em
> uma projeção 2D do modelo — os resultados são indicativos, não substituem CFD
> de engenharia (OpenFOAM, Ansys Fluent etc.).

## 🚀 Funcionalidades

### 📁 Upload e Processamento de Modelos 3D
- **Formatos**: STL, GLB, GLTF e OBJ (cenas GLB com múltiplos meshes são
  concatenadas automaticamente)
- **Orientação canônica automática**: qualquer que seja a convenção de eixos do
  arquivo, o modelo é reorientado na importação — maior eixo → X (direção do
  fluxo), menor eixo → Z (altura). Isso garante que o vento sempre sopre ao
  longo do comprimento do modelo
- **Normalização**: centralização e escala uniforme (sem distorção de aspecto)
- **Validação da malha**: vértices, faces, volume, watertightness e winding

### 🔬 Simulação Tradicional (túnel de vento)
- **Geometria real imersa**: a silhueta lateral do modelo carregado é
  rasterizada na grade do solver (robusto até para malhas não-watertight) e
  aplicada como obstáculo com condição de não-deslizamento
- **Solver FiPy**: equações de momento com convecção e difusão, inicializadas
  com fluxo potencial ao redor do obstáculo
- **Forças aerodinâmicas**: arrasto e sustentação por integração de pressão nas
  células adjacentes à superfície do obstáculo (Cd, Cl, forças e área frontal)
- **Resolução adaptativa**: 120×60 células com GPU (CUDA), 60×30 em CPU
- **Fallback**: aerofólio NACA 0012 integrado quando nenhum modelo é carregado

### 🤖 Simulação com IA
- **Modelo substituto**: rede neural feedforward (PyTorch) para predição rápida
  de campos de velocidade e pressão
- **Resiliência a falta de VRAM**: se a GPU estiver sem memória, treino e
  predição caem automaticamente para CPU (o modelo é pequeno; leva segundos)

### 📊 Visualizações
- **Campos 2D**: contornos de velocidade e pressão, vetores, streamlines sobre
  o campo resolvido, histórico de convergência (Matplotlib + Plotly)
- **3D interativo**: malha do modelo, streamlines 3D alinhadas ao fluxo que
  defletem ao redor do corpo, e animação de túnel de vento com partículas
- **Malhas pesadas sem sustos**: modelos até ~2M de faces são exibidos com
  qualidade total; a decimação só entra se o tamanho serializado exceder o
  orçamento (e nunca duplica a malha entre frames de animação)
- **Comparação**: métricas e campos lado a lado (tradicional × IA)

## 🛠️ Instalação

### Pré-requisitos
- Python 3.10+
- GPU NVIDIA com CUDA (opcional — o app roda em CPU)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abra o navegador em `http://localhost:8501`.

## 🚀 Fluxo de Uso

1. **Upload & Geometria**: carregue um STL/GLB (ou use o aerofólio NACA
   padrão). Confira as informações da malha e a visualização 3D
2. **Parâmetros** (barra lateral): velocidade de entrada (10–100 m/s),
   iterações máximas e tolerância de convergência
3. **Simulação Tradicional**: executa o túnel de vento com a silhueta real do
   modelo; acompanhe o progresso e o resíduo em tempo real
4. **Simulação IA**: predição rápida com o modelo substituto
5. **Comparação & Resultados**: métricas e campos lado a lado

### Modelos de exemplo (`examples/`)
| Arquivo | Observações |
|---|---|
| `RB16B_FIXED!.stl` | **Recomendado** — F1 Red Bull RB16B, malha watertight e winding consistente, 111k faces |
| `ferrari_f75.glb` | GLB leve (80k faces), bom para testar o suporte a GLB |
| `tesla.glb` | Carro de rua, 93k faces |
| `audi.stl` | Carro de rua, 143k faces (não-watertight — funciona para simulação) |
| `cube_example.stl` | Geometria mínima para testes rápidos |
| `DeLorean.STL` | Modelo detalhado, 19 MB |

Malhas não-watertight funcionam: a rasterização da silhueta usa nuvem densa de
pontos da superfície + fechamento morfológico, sem depender de volume fechado.

## 📋 Estrutura do Projeto

```
CFD2026/
├── app.py                 # Aplicação Streamlit (4 abas)
├── requirements.txt
├── cfd/
│   ├── simulation.py      # Solver FiPy + máscara de obstáculo + forças aero
│   ├── geometry.py        # Import multi-formato, orientação canônica,
│   │                      #   projeção 2D e rasterização da silhueta
│   └── ai_model.py        # Rede neural substituta (fallback GPU→CPU)
├── visualization/
│   ├── plots_2d.py        # Campos, streamlines 2D, plots interativos
│   └── plots_3d.py        # Malha 3D, streamlines 3D, animação do túnel
├── utils/helpers.py
├── examples/              # Modelos 3D de teste
└── tests/test_smoke.py    # Suíte de smoke tests
```

## 🧪 Testes

```bash
python tests/test_smoke.py    # ou: pytest tests/
```

Cobrem: solver de ponta a ponta em malha pequena, contrato de dados da
visualização, geração de streamlines, carga de GLB com canonicalização de
orientação, rasterização da silhueta e obstáculo real no solver, e o exemplo
watertight RB16B.

## 🔧 Parâmetros Físicos

- **Fluido**: ar (ρ = 1.225 kg/m³, μ = 1.8e-5 Pa·s)
- **Domínio**: 4.0 m × 2.0 m; objeto ocupa ~30% do comprimento, posicionado a
  1/4 da entrada
- **Contornos**: entrada com velocidade fixa, saída com gradiente zero, paredes
  no-slip, obstáculo no-slip
- **Pressão**: estimada por Bernoulli a partir do campo de velocidade

## ⚠️ Limitações Conhecidas

- **2D**: o solver opera na projeção lateral do modelo, não no volume 3D
- **Física simplificada**: sem acoplamento pressão-velocidade (SIMPLE/PISO) nem
  modelo de turbulência; Cd/Cl são qualitativos
- **Streamlines 3D**: sintéticas (alinhadas e defletidas pelo corpo), não
  derivadas do campo resolvido — as streamlines 2D sim usam o campo real
- **Modelo de IA**: treinado com dados sintéticos genéricos; não considera a
  geometria carregada

## 📚 Referências

- FiPy: https://www.ctcms.nist.gov/fipy/
- Streamlit: https://streamlit.io/
- Trimesh: https://trimesh.org/
- Anderson, J.D. — *Computational Fluid Dynamics: The Basics with Applications*

## 📄 Licença

Projeto fornecido como está, para fins educacionais e de demonstração.
