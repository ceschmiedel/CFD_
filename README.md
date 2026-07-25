# CFD2026 — túnel de vento no navegador

Túnel de vento virtual que roda inteiramente na GPU do navegador. Carregue um
modelo 3D (STL, OBJ, PLY, GLB), e ele é voxelizado dentro de um domínio de
túnel e resolvido por um **Lattice-Boltzmann D3Q19** com colisão TRT e modelo
sub-grid de Smagorinsky. Fumaça volumétrica advectada pelo campo, C<sub>p</sub>
pintado na carroceria, traços de ar rasgando o chão e forças por troca de
momento.

Sem servidor de simulação, sem instalação: roda na placa de quem abre o link —
em WebGPU quando ele existe, em WebGL2 quando não, com a mesma física e a mesma
imagem nos dois.

> **Estado**: o núcleo é validado contra correlações experimentais (6/6, tabela
> abaixo) e a suíte roda dentro do próprio app — **nos dois backends, com os
> mesmos números**. O app inteiro, solver e visualização, roda em **WebGPU ou
> WebGL2**. O C<sub>d</sub> de veículos ainda **não converge** — ver
> [Limitações](#limitações). Corpos canônicos sim.

---

## Como rodar

Precisa de um navegador com **WebGPU** (Chrome/Edge, Safari 18+, Firefox 141+)
**ou WebGL2** (praticamente qualquer um desta década) e de um servidor HTTP
local — módulos ES não carregam de `file://`.

```bash
python tools/servidor.py 8601
```

Abra `http://localhost:8601/`. Nada para instalar, nada para compilar.

O caminho é escolhido sozinho — WebGPU quando existe, WebGL2 quando não — e o
seletor no painel troca entre eles (`?backend=webgl2` na URL faz o mesmo). A
troca recarrega a página: um `<canvas>` aceita um tipo de contexto só na vida
dele.

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

> **Quem abre sem WebGPU** cai no caminho WebGL2 sem perceber — solver e
> visualização completos, ~10 % mais lento na mesma máquina. Só quem não tem
> nenhum dos dois vê uma explicação, e não uma tela preta.

---

## Validação

Cada caso roda o solver completo — voxelização, contornos, troca de momento — e
compara com um ajuste experimental publicado. Abra
`http://localhost:8601/tests/validacao.html` e confira você mesmo;
`?backend=webgl2` roda a mesma suíte no outro caminho.

**Os seis casos passam nos dois backends, com os mesmos números** — a tabela
abaixo vale para WebGPU e WebGL2 indistintamente. É o que se espera de uma
física escrita uma vez só, e é o único jeito de saber que ela foi mesmo escrita
uma vez só.

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
| `tests/desempenho.html` | MLUPS e banda efetiva por preset, nos dois backends (`?backend=webgl2`) |
| `tests/webgl2.html` | O caminho WebGL2 inteiro: os cinco shaders compilam no driver desta máquina, escoamento uniforme preservado, viscosidade medida por decaimento de onda, e o **confronto com o WebGPU** no mesmo caso — campo e força lado a lado |

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

### E o WebGL2 custa quanto?

Medido em seguida, na mesma máquina e na mesma sessão (`?backend=webgl2`):

| Preset | WebGL2 | WebGPU na mesma sessão | razão |
|---|---|---|---|
| Mínima | 1 544 MLUPS | 1 286 | **0,83×** |
| Baixa | 1 396 | 1 313 | 0,94× |
| Média | 1 345 | 1 322 | 0,98× |
| Alta | 1 111 | 1 290 | 1,16× |

Empate, e o caminho sem compute chega a ganhar no domínio pequeno — onde o
custo por despacho pesa mais que a banda, e um `drawArrays` de três vértices é
mais barato que um dispatch de compute. O prognóstico de "2 a 4× mais lento"
que este README carregou desde antes de existir runtime estava errado, e a
razão é que o LBM é limitado por BANDA de memória: quem decide o número é
quanto se lê e se escreve por célula, não se o kernel é compute ou fragment.

**A ressalva, e ela é grande.** O WebGPU mediu 1 300 MLUPS nesta sessão contra
os 2 800 da tabela acima, medidos em outra. A máquina não estava no mesmo
estado — mesma placa, condição diferente —, e por isso só a RAZÃO medida
lado a lado vale alguma coisa; os absolutos das duas tabelas não se comparam
entre si. Rode as duas você mesmo antes de acreditar em qualquer das duas.

Uma consequência prática: com 6,55 M de células o WebGL2 dá 205 passos/s, ou
três subpassos por quadro de 60 Hz. Isso é escoamento vivo na tela — o caminho
sem WebGPU não é um modo degradado do solver, é o mesmo solver.

---

## Como funciona

### Uma física, dois backends

A colisão TRT, o Smagorinsky e o bounce-back estão escritos **uma vez**, em
`src/core/emit/ir.js`, em termos de um *dialeto* que sabe declarar uma variável
e endereçar uma população na linguagem alvo. `wgsl.js` gera WGSL para o WebGPU;
`glsl.js` gera GLSL ES 3.00 para o WebGL2. **Os dois rodam**, e resolvem o
mesmo escoamento: no mesmo caso, o campo de velocidade concorda em 3,3e-8 e a
força sobre o corpo em 5e-7 sobre 0,72 — arredondamento de fp32, não física.
`tests/webgl2.html` é o confronto.

No WebGL2 não há compute nem storage buffer: o lattice 3D vira um atlas 2D
(ladrilhos em z), as 19 populações vão em cinco texturas RGBA32F escritas de
uma vez por MRT, o kernel do passo é um fragment shader com ping-pong de
framebuffer, e a redução das forças — que no WebGPU é memória compartilhada de
workgroup — vira uma pirâmide de somas 2×2 até sobrar um texel. Três
construções de WGSL que vazaram para o IR (`select`, o cast `f32` e `gid`) são
definidas no preâmbulo GLSL em vez de encher o IR de condicionais por backend,
que é como uma física escrita uma vez volta a ser duas.

**A imagem também.** As quatro camadas — carroceria com C<sub>p</sub>, piso
rolante, traços rasantes e fumaça volumétrica — existem nas duas linguagens, e
a regra que as governa é a mesma dos dois lados: tudo o que aparece na tela é o
campo resolvido. Sem compute, as partículas passam a viver em texturas
atualizadas por um fragment shader e são lidas no vertex shader por
`gl_InstanceID`; o volume de fumaça é uma `TEXTURE_3D` de verdade — `RGBA16F`
filtrável em núcleo no WebGL2 —, escrita fatia por fatia com
`framebufferTextureLayer`, o que dá a trilinear do hardware na advecção e no
ray-march. A cena opaca vai para um alvo fora da tela com profundidade em
textura, porque o ray-march precisa LER essa profundidade para não atravessar o
carro.

Medido na mesma máquina, preset Média com tudo ligado: **54 fps em WebGL2
contra 62 em WebGPU**, e o mesmo C<sub>d</sub> (2,63 contra 2,65 na mesma
janela de amostragem).

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
pente é dito em número de tubos (colunas e linhas): a espessura de cada
filamento sai do espaçamento, então tirar tubos os deixa mais grossos, que é o
que um pente com menos saídas e a mesma vazão faz.

E o filete é **contínuo**. Houve aqui uma emissão pulsada, que dava movimento a
uma cena estacionária ao custo de picar o filamento em contas — e um filamento
picado não mostra linha de corrente, que é a única coisa que essa camada existe
para mostrar. O movimento veio para as rasantes e o piso. A fatia de emissão do
pente acompanha o passo de advecção: fatia fixa com passo grande deixa buraco
entre uma emissão e a seguinte, e o filamento sai como uma fileira de blocos —
o que fica óbvio assim que o relógio passa a seguir a velocidade do túnel.

**Velocidade se lê no chão.** Um túnel em regime estacionário é uma cena parada
— o campo não muda, as streaklines ficam onde estão, e quem chega sem contexto
vê um carro de enfeite numa caixa. A sensação de 30 m/s vem de onde ela vem na
janela de um carro: do solo borrando. Um punhado de traços rasantes advectados
pelo campo, enviesados para o chão, mais compridos onde o ar corre e curtos e
embolados na esteira; e a grade do piso rolando à velocidade do cinto, que é o
que a condição de contorno da esteira rolante diz que o chão está fazendo. A
direção e a velocidade são do campo resolvido — o exagero é o comprimento do
traço, e ele estica ao longo do caminho que a partícula de fato percorreu.

**Um relógio só, e ele é tempo físico.** `u_lb` vale 0,05 a 5 m/s e a 90: mudar
a velocidade não mexe no campo de lattice, mexe em quanto tempo **físico** vale
um passo. Animar a uma taxa fixa de células por quadro — o caminho óbvio — dá
portanto a mesma imagem nas duas pontas do controle, e a velocidade do túnel não
se sente em lugar nenhum da tela. As camadas de movimento (rasantes, esteiras,
fumaça e a grade do piso) penduram todas num único relógio que reproduz tempo
físico a ritmo constante: 0,6 s de escoamento por segundo de tela, medido contra
o relógio de parede e não contado em quadros. O deslocamento na imagem fica
proporcional aos metros por segundo, e o HUD anuncia o fator — câmera lenta que
não se declara é mais uma forma de mentir sobre a escala de tempo.

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
    renderer.js               carroceria com Cp, esteiras, piso rolante
    fumaca.js                 volume, advecção MacCormack, ray-march
    rasante.js                traços rente ao chão, a velocidade da cena
    comum.js, mat4.js
    webgl2/                   as mesmas quatro camadas em GLSL ES 3.00
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
