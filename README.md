# CFD2026 — túnel de vento no navegador

Túnel de vento virtual que roda inteiramente na GPU do navegador. Carregue um
modelo 3D (STL, OBJ, PLY, GLB), e ele é voxelizado dentro de um domínio de
túnel e resolvido por um **Lattice-Boltzmann D3Q19** com colisão TRT e modelo
sub-grid de Smagorinsky. Fumaça volumétrica advectada pelo campo, C<sub>p</sub>
pintado na carroceria, forças por troca de momento.

Sem servidor de simulação, sem instalação: o solver é WebGPU compute, e o que o
visitante abre roda na placa dele.

> **Estado**: o núcleo é validado contra correlações experimentais (6/6, tabela
> abaixo) e a suíte roda dentro do próprio app. O C<sub>d</sub> de veículos
> ainda **não converge** — ver [Limitações](#limitações). Corpos canônicos sim.

---

## Como rodar

Precisa de um navegador com **WebGPU** (Chrome/Edge, Safari 18+, Firefox 141+)
e de um servidor HTTP local — módulos ES não carregam de `file://`.

```bash
python tools/servidor.py 8601
```

Abra `http://localhost:8601/`. Nada para instalar, nada para compilar.

O servidor manda `Cache-Control: no-store`. Isso não é detalhe: com o
`http.server` padrão o navegador recarrega o HTML e mantém os módulos ES em
cache, então uma correção de shader some sem aviso e você depura código que o
navegador nunca carregou.

### No celular

A interface se adapta: o painel de controle vira uma gaveta atrás do botão
**⚙ controles**, a visualização fica com a tela inteira e o C<sub>d</sub> sobe
para o canto superior, que está sempre à vista. Um dedo arrastando gira a
câmera, dois dedos aproximam e afastam.

A resolução inicial é escolhida mais baixa em telas pequenas — uma GPU móvel
passa nos limites declarados de presets que ela não consegue avançar em tempo
útil. As outras continuam na lista, com o tamanho de cada uma.

## Publicar (Vercel)

Site estático puro: **sem build, sem dependências, sem passo de instalação**. O
repositório é o deploy.

Na Vercel, *Add New → Project → Import* este repositório e deixe:

| Campo | Valor |
|---|---|
| Framework Preset | **Other** |
| Build Command | *(vazio)* |
| Output Directory | *(vazio — serve a raiz)* |
| Install Command | *(vazio)* |

`vercel.json` já cuida dos cabeçalhos: `examples/` com cache imutável de um ano
(são arquivos grandes que nunca mudam), `src/` com revalidação de uma hora.
`.vercelignore` deixa de fora `legacy/`, `tools/` e um modelo de 28 MB que não
está na lista do console — 40 % do peso do deploy por nada.

Feita a importação, cada `git push` na `master` publica sozinho.

> **Peso.** O modelo padrão tem 7,5 MB e todo visitante o baixa na primeira
> visita — a tela de carregamento mostra o progresso em MB. O `DeLorean.STL`
> tem 18 MB e só é baixado se escolhido.

> **Quem abre sem WebGPU** vê uma explicação do porquê, não uma tela preta.
> Chrome/Edge, Safari 18+ e Firefox 141+ têm; navegadores mais antigos e
> algumas GPUs em lista de bloqueio, não.

---

## Validação

Cada caso roda o solver completo — voxelização, contornos, troca de momento — e
compara com um ajuste experimental publicado. Abra
`http://localhost:8601/tests/validacao.html` e confira você mesmo.

| Caso | Medido | Referência | Erro |
|---|---|---|---|
| Couette: inclinação do perfil | exata | U/H | **0,00 %** |
| Couette: posição efetiva da parede | z = 0,500 / 32,501 | meio-caminho | espalhamento **abaixo de 1e-3 célula** entre ω = 0,7 e 1,6 |
| Esfera, Re = 50 | C<sub>d</sub> 1,480 | 1,539 | 3,8 % |
| Esfera, Re = 100 | C<sub>d</sub> 1,070 | 1,094 | **2,2 %** |
| Cilindro, Re = 100 | C<sub>d</sub> 1,285 | 1,464 | 12,2 % |
| Cilindro, Re = 100 | St 0,1663 | 0,1648 | **0,9 %** |

O teste de Couette é o que mede se a parede está onde o TRT diz que está. Com
Λ = 3/16 ela fica no meio-caminho entre nós **independente da viscosidade** —
e é esse *independente* que o espalhamento de 3e-4 célula confirma. O arrasto
depende inteiramente de onde a parede pensa que está.

Correlações usadas: Clift-Gauvin (esfera), curva padrão por partes (cilindro),
Williamson & Brown 1998 / Roshko 1954 (Strouhal), Maskell para bloqueio
(Barlow, Rae & Pope §10.4).

### Outras suítes

| Página | O que verifica |
|---|---|
| `tests/nucleo.html` | Invariantes D3Q19, escoamento uniforme preservado a 1,9e-8, viscosidade medida por decaimento de onda de cisalhamento (0,04–1,07 % de erro em ω de 0,6 a 1,8), conservação de massa |
| `tests/geometria.html` | Leitura dos seis modelos do repositório, voxelização, silhuetas em corte |
| `tests/estabilidade.html` | Onde o solver perde estabilidade **com um carro dentro** |
| `tests/desempenho.html` | MLUPS e banda efetiva por preset |

---

## Desempenho

Medido numa RTX 4080, WebGPU:

| Preset | Domínio | Células | MLUPS | Banda efetiva | Passos/s |
|---|---|---|---|---|---|
| Mínima | 160×80×64 | 0,82 M | 2 835 | 431 GB/s | 3 460 |
| Baixa | 240×120×96 | 2,76 M | 2 876 | 437 GB/s | 1 040 |
| Média | 320×160×128 | 6,55 M | 2 782 | 423 GB/s | 424 |
| Alta | 480×240×192 | 22,1 M | 2 765 | 420 GB/s | 125 |

~60 % do pico teórico da placa. O LBM é limitado por banda: 152 bytes por
célula por passo (19 populações lidas, 19 escritas). A fumaça volumétrica custa
cerca de 5 fps sobre isso.

---

## Como funciona

### Uma física, dois backends

A colisão TRT, o Smagorinsky e o bounce-back estão escritos **uma vez**, em
`src/core/emit/ir.js`, em termos de um *dialeto* que sabe declarar uma variável
e endereçar uma população na linguagem alvo. `wgsl.js` gera WGSL; `glsl.js`
geraria GLSL para o caminho WebGL2.

O motivo é direto: um solver com dois backends e duas cópias da colisão tem,
mais cedo ou mais tarde, duas físicas diferentes — e isso não aparece como erro
de compilação, aparece como um C<sub>d</sub> que difere 4 % entre backends sem
ninguém saber qual está certo.

O gerador também dá coisas que ninguém escreveria à mão: as 19 direções viram
código reto sem laço nem indireção (o kernel tem **zero laços**), os `c_i`
entram como literais, e a colisão sai por par oposto — metade das
multiplicações, porque g⁺ é simétrico e g⁻ antissimétrico no par.

### Decisões que importam

**Populações deslocadas.** Armazenamos `g_i = f_i − w_i`, não `f_i`. Somar 19
valores de ~0,05 em fp32 erra δ = ρ−1 (que é da ordem de 1e-4) em 0,2 %, o que
aparece como ruído visível no mapa de C<sub>p</sub>. Com `g_i` o mesmo erro cai
para ~1e-9.

**Bounce-back no nó de fluido, não no sólido.** A versão de tutorial trata o
sólido como um nó que participa do passo. Funciona com parede parada; com
esteira rolante vaza — o par para cima/para baixo do nó de piso apenas troca de
lugar a cada passo enquanto o termo de esteira soma momento nele toda vez.
Puxando com bounce-back, o nó sólido deixa de ter estado.

**Voxelização sem winding number nem pseudo-normais.** Os dois assumem
propriedades que sopa de triângulos não tem (uma asa de espessura zero devolve
winding ≈ 0 dos dois lados). O escoamento não pergunta se um ponto está dentro
no sentido topológico — pergunta se o ar consegue chegar lá. Sobreposição exata
triângulo-caixa marca a casca, inundação a partir da borda decide o resto.

**Forças por troca de momento.** Soma sobre os links de bounce-back. Entrega
pressão **e** atrito viscoso juntos sem reconstruir normal nenhuma a partir de
uma escada de voxels — e o atrito responde por 10 a 25 % do arrasto de um
carro. Com populações deslocadas, o termo `2·w_i·c_i` **não** cancela por link.

**Fumaça: rake, não névoa.** Um volume homogêneo não tem contraste. Todo túnel
físico usa um pente de tubos finos soltando filamentos paralelos, e o que se vê
é o que acontece com eles. Advecção semi-lagrangiana com correção de MacCormack
e limitador — sem ela os filamentos se dissolvem antes de alcançar o corpo. O
rake é pulsado: o espaçamento entre as contas de fumaça é proporcional à
velocidade local, o que é medição por tempo de voo.

### Honestidade sobre Reynolds

Um carro de 4,5 m a 30 m/s está em Re = 9,0 × 10⁶. Resolver isso diretamente
pede da ordem de 10¹⁵ células; este programa tem 10⁷. O app reporta **os dois
números** — o Re físico e o Re que o lattice de fato resolve — e diz em uma
frase quando os dois se separaram. Um túnel virtual que mostra "Re = 9,0e6" e
nada mais está mentindo por omissão.

---

## Estrutura

```
index.html                    console do túnel
src/
  core/
    lattice.js                D3Q19, pesos, TRT, magic numbers
    units.js                  ponte m/s ↔ lattice, Re físico vs resolvido
    tunel.js                  montagem dos tipos de célula, corpos analíticos
    forces.js                 coeficientes, Maskell, janela de média, Strouhal
    validacao.js              correlações experimentais e casos
    emit/
      ir.js                   A FÍSICA, uma vez só
      wgsl.js                 → WGSL (WebGPU)
  backend/
    caps.js                   sondagem de WebGPU/WebGL2, presets
    webgpu/solver.js          alocação, pipelines, laço de passo
  geom/
    parse.js                  STL, OBJ, PLY, glTF, GLB
    voxel.js                  casca por eixo separador + inundação selada
    prepare.js                orientação, unidade, sentido, colocação
  render/
    renderer.js               carroceria com Cp, esteiras, piso
    fumaca.js                 volume, advecção MacCormack, ray-march
    comum.js, mat4.js
tests/                        as suítes de verificação (abra no navegador)
tools/servidor.py             servidor de desenvolvimento sem cache
examples/                     modelos 3D
legacy/                       o app Streamlit + FiPy anterior, arquivado
```

---

## Modelos de exemplo

| Arquivo | Triângulos | Observações |
|---|---|---|
| `tesla.glb` | 93 k | Cybertruck; orientação detectada corretamente |
| `RB16B_FIXED!.stl` | 111 k | F1 Red Bull; watertight e winding consistente |
| `ferrari_f75.glb` | 80 k | F1; **entra de ré** no arquivo — a detecção de sentido corrige |
| `audi.stl` | 143 k | Não-watertight; funciona |
| `DeLorean.STL` | 380 k | Binário com cabeçalho começando em `"solid"` — a armadilha clássica; **entra de ré** |
| `cube_example.stl` | 12 | Binário com cabeçalho de **79 bytes** em vez de 80 |

Os dois últimos são o motivo de nenhuma detecção de formato aqui acreditar no
que o arquivo diz sobre si mesmo. STL binário é detectado pela aritmética
(84 + 50n fechando com o tamanho), não pelo prefixo.

**Orientação.** O eixo do escoamento é a maior extensão; o eixo vertical sai de
**simetria** (veículos são espelhados no plano longitudinal e não no
horizontal); e o *sentido* sai da **altura** — veículo de estrada tem nariz
baixo e traseira mais alta. Acerta os seis modelos, e há inversão manual no
painel para quando não acertar.

---

## Limitações

**O C<sub>d</sub> de veículo não converge.** É o problema aberto principal. O
valor é finito e a barra de erro **cresce** com o tempo em vez de encolher. A
causa provável está visível no painel: com o teto de ω em 1,90, o Re do lattice
cai para ~180, e a 180 um corpo rombudo tem C<sub>d</sub> genuinamente alto e
esteira muito instável. Se for isso, o remédio é mais células no corpo — não
mais passos. Corpos canônicos (esfera, cilindro) convergem e validam.

**Backend WebGL2 não implementado.** A sondagem e o emissor de shaders estão
prontos; falta o runtime (empacotamento das 19 populações em 5 texturas
RGBA32F e atlas 2D do lattice 3D). Até lá, sem WebGPU o app mostra um aviso
explicando o porquê.

**É LES grosseiro, não DNS.** O modelo sub-grid de Smagorinsky fornece a
dissipação das escalas não resolvidas. É o método que a CFD automotiva
comercial usa, e não é simulação direta.

**Resolução no corpo.** O preset decide o compromisso entre células no corpo e
bloqueio, e os dois números aparecem no seletor. No preset Média o corpo tem
32 células de comprimento — o suficiente para a forma, não para um retrovisor.

**Sem build de arquivo único ainda.** O plano é um `.html` autocontido para
distribuição, gerado por script a partir dos módulos.

---

## Referências

- Ginzburg & Adler (1994) — parâmetro mágico TRT, posição da parede
- Lehmann (2022) — *Esoteric Pull*, otimização de memória do LBM (ainda não aplicada)
- Barlow, Rae & Pope — *Low-Speed Wind Tunnel Testing*, 3ª ed. (correção de Maskell)
- Clift, Grace & Weber — *Bubbles, Drops and Particles* (correlação da esfera)
- Williamson & Brown (1998), Roshko (1954) — Strouhal do cilindro
- Akenine-Möller (2001) — sobreposição triângulo-caixa por eixo separador
- Selle, Fedkiw et al. — advecção MacCormack não-condicionalmente estável

## Licença

Fornecido como está, para fins educacionais e de demonstração.
