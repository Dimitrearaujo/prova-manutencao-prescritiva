# Resultados

Reprodução: `python scripts/evaluate.py`. Amostras estratificadas de ~2.000
eventos por partição, semente fixa em 42.

---

## 1. A partição aleatória é inválida neste conjunto de dados

Antes de qualquer número, o ponto metodológico que condiciona todos eles.

O ensaio gravou **um defeito de cada vez, em blocos contínuos de tempo**. Cada
rótulo ocupa uma janela própria, sem sobreposição com os outros — verificável em
`scripts/eda.py`. Dentro de um bloco, as leituras acontecem a segundos de
distância e são quase idênticas.

Sortear treino e teste ao acaso coloca a mesma medição dos dois lados da
partição. O vizinho mais próximo de um evento de teste passa a ser um evento
gravado dois segundos antes, que está no treino com o rótulo certo. A acurácia
que sai disso não mede generalização: mede a taxa de amostragem do sensor.

Por isso a partição aleatória aparece aqui **apenas como contraste**, para
mostrar o tamanho da distorção.

---

## 2. Desempenho por partição

Cada partição é medida em dois níveis.

**Acerto de rótulo** cobra o nome exato do defeito. **Acerto de procedimento**
cobra o que o sistema entrega ao técnico: os quatro defeitos de rolamento levam
ao mesmo Doc1 e à mesma ação corretiva, então trocar anel interno por anel
externo não muda uma linha da instrução. O segundo número mede o produto; o
primeiro é detalhe de nomenclatura.

| Partição | Acerto de procedimento | Acerto de rótulo | Taxa de rejeição | Eventos |
|---|---|---|---|---|
| Aleatória — **inválida**, só contraste | 95,9% | 91,8% | 18,8% | 2.001 |
| **Temporal — a medida principal** | **73,1%** | 48,5% | 31,8% | 1.998 |
| Nova campanha — estresse | 51,1% | 26,4% | 40,3% | 2.002 |

Acurácia entre os eventos que o sistema aceitou diagnosticar. Os rejeitados não
recebem palpite: recebem "não reconheço este padrão".

### Leitura

**A queda de 95,9% para 73,1% é a medida do vazamento.** Vinte e três pontos de
acurácia que a partição aleatória entrega de graça e que não existem em operação.

**73,1% é o número real do produto.** De cada dez eventos diagnosticados, sete
levam ao procedimento correto, e um em cada três eventos nem chega a ser
diagnosticado — é devolvido como não reconhecido em vez de virar um palpite.

**A queda para 51,1% na nova campanha não é ruído: é deriva medida.** A segunda
rodada de coletas foi gravada depois, e a linha de base mudou junto. A própria
condição `normal` se deslocou entre campanhas — a aceleração RMS mediana sobe 25%
de uma para a outra. Quando o referencial se move, a comparação com o histórico
antigo perde valor, e é isso que o número mostra.

Esse resultado tem consequência de projeto, não é só um número ruim. Ele é a
justificativa concreta para o **monitoramento de deriva** descrito em
[`ARQUITETURA.md`](ARQUITETURA.md): a taxa de rejeição sobe de 31,8% para 40,3%
sob deriva, o que a torna um alarme utilizável. Quando ela sobe de forma
sustentada em campo, o índice precisa ser reconstruído com dados recentes.

---

## 3. Defeito inédito: o teste que separa esta solução de um classificador fechado

Uma classe inteira é removida do índice e depois consultada. É a situação que o
enunciado quer evitar: um defeito que o sistema nunca viu.

Um classificador supervisionado não tem saída aqui — ele devolve, com confiança,
a classe conhecida menos errada. Aqui há duas saídas aceitáveis: **rejeitar**, ou
**errar o nome mas acertar o procedimento**.

| Defeito removido | Rejeitou | Acertou o procedimento mesmo assim | Desfecho útil | Para onde foi |
|---|---|---|---|---|
| rolamento_combination | 52,3% | 94,4% | **97,3%** | rolamento_inner |
| rolamento_inner | 29,0% | 93,0% | **95,0%** | rolamento_combination |
| rolamento_ball | 33,0% | 82,6% | **88,3%** | rolamento_inner |
| rolamento_outer | 44,7% | 77,7% | **87,7%** | rolamento_inner |
| desbalanceado_1parafuso | 52,7% | 49,3% | 76,0% | eccentric_rotor |
| eccentric_rotor | 49,3% | 46,1% | 72,7% | cocked_rotor |
| cocked_rotor | 56,0% | 26,5% | 67,7% | correia |
| polia | 43,3% | 4,1% | 45,7% | normal |
| desalinhado | 44,3% | 0,0% | 44,3% | correia |
| ventoinha | 39,3% | 0,0% | 39,3% | polia |
| correia | 35,7% | 1,0% | 36,3% | cocked_rotor |
| **média** | **43,6%** | — | **68,2%** | |

### Leitura

O resultado se separa em dois grupos, e a separação tem explicação física.

**Quando o defeito removido tem irmão na mesma família, o sistema degrada com
elegância.** Tire `rolamento_combination` do índice e ele cai em
`rolamento_inner` — outro rolamento, mesmo Doc1, mesma ação corretiva. O nome
está errado, a prescrição está certa. Os quatro defeitos de rolamento ficam entre
87,7% e 97,3% de desfecho útil.

**Quando o defeito é o único da sua família, a degradação é feia.** `correia`,
`ventoinha`, `polia` e `desalinhado` não têm par no histórico: removidos, o
vizinho mais próximo passa a ser um defeito de outro componente, e a prescrição
sairia errada. A rejeição segura entre 36% e 46% desses casos; o restante
passaria.

**O pior caso está em `polia`, que cai em `normal`.** Chamar defeito de condição
normal é o erro mais caro que este sistema pode cometer, porque não gera nem
ação nem alerta. É a limitação mais séria medida aqui.

---

## 4. Ablação de features

Reprodução: `python scripts/ablacao_features.py`, sobre a partição temporal.

| Variante | Features | Acerto de rótulo | Rejeição |
|---|---|---|---|
| A. tudo | 25 | 46,6% | 28,4% |
| B. sem frequência de pico | 23 | 46,7% | 28,1% |
| C. sem temperatura | 24 | 47,3% | 30,6% |
| D. sem frequência e temperatura | 22 | 47,6% | 30,9% |
| **E. produção — sem frequência, temperatura e ordem** | **20** | **48,5%** | 32,4% |

O ganho é de dois pontos, dentro da margem de ruído da amostra. **As duas colunas
foram removidas pela análise, não pela métrica.**

**Frequência de pico** tem 14 valores distintos, e 61 Hz responde por 61% das
linhas — inclusive nos 347 registros de motor desligado. Vibração rotacional não
existe com o motor parado: 61 Hz é a frequência da rede elétrica. E como o
`RobustScaler` divide por um IQR de 1,25, um deslocamento dessa coluna entre
coletas vira dezenas de unidades de distância e domina a busca. As features
`ordem_z` e `ordem_x` saíram junto: derivam dessa coluna, e se a origem é
artefato, a razão também é.

**Temperatura** correlaciona entre 0,50 e 0,93 com a posição dentro do bloco de
gravação, subindo em alguns blocos e caindo em outros — aquecimento e
resfriamento ambiente, não física de falha. E separa mal: a variação entre
defeitos (1,18 °C) é menor que a variação dentro de cada defeito (1,63 °C).

Manter uma coluna que a análise mostrou não medir o defeito, só porque ela não
piora o número, é carregar dívida para o dia em que a instrumentação mudar.

---

## 5. Cobertura documental

`tests/test_cobertura.py` trava o mapa contra a leitura humana dos seis PDFs.

| | Menor aderência | Maior aderência |
|---|---|---|
| Defeitos cobertos (9) | 2,14 | 2,83 |
| Segundo colocado de cada busca | — | 0,50 |
| Defeitos sem procedimento (2) | 0,00 | 0,00 |

Com o limiar em **1,0**, há uma faixa vazia entre 0,50 e 2,14. A decisão de
cobertura não está no fio da navalha: nenhum ajuste pequeno de parâmetro faz o
mapa virar. O teste verifica essa folga explicitamente, não só o resultado.

---

## 6. O que estes números não dizem

**Não medem detecção precoce.** Todos os eventos do conjunto são de defeito já
instalado ou de operação normal. Não há trajetória de degradação, então não dá
para avaliar antecedência — que é metade do valor de manutenção preditiva.

**Não medem qualidade do texto gerado.** O acerto de procedimento mede se o
documento certo foi selecionado, não se a instrução redigida a partir dele é boa.
Avaliar isso exige um humano da manutenção lendo as saídas.

**Vêm de uma bancada, não de um chão de fábrica.** Quatro rotações fixas, um
defeito por vez, ambiente controlado. Em operação real há carga variável,
múltiplos defeitos simultâneos e transição gradual entre estados. A partição por
campanha é a melhor aproximação disponível de "o mundo mudou", e ela já mostra
queda de 22 pontos.
