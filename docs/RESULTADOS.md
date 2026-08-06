# Resultados

Cada número aqui declara de onde vem.

| Origem | O que produz |
|---|---|
| `python scripts/evaluate.py` | seções 2 e 3, e o `data/index/avaliacao.json` que as guarda — amostras estratificadas de ~2.000 eventos por partição, semente fixa em 42 |
| `python scripts/ablacao_features.py` | seção 4 |
| `python scripts/eda.py` | a sobreposição de blocos da seção 1 e os números de deriva na leitura da seção 2 (hipótese H5) |
| `pytest tests/test_cobertura.py` | seção 5 |

A tabela antes/depois da seção 1 e a atribuição de rejeição por portão da seção 7
não saem de nenhum script versionado: foram medidas à parte sobre os artefatos
entregues, consultando `IndiceSimilaridade` com e sem os parâmetros novos, e o
procedimento está descrito no ponto em que cada uma aparece.

---

## 1. A partição aleatória é inválida neste conjunto de dados

Antes de qualquer número, o ponto metodológico que condiciona todos eles.

O ensaio gravou **um defeito de cada vez**, e dentro de cada gravação as leituras
acontecem a segundos de distância e são quase idênticas. Sortear treino e teste
ao acaso coloca a mesma medição dos dois lados da partição: o vizinho mais
próximo de um evento de teste passa a ser um evento gravado dois segundos antes,
que está no treino com o rótulo certo. A acurácia que sai disso não mede
generalização — mede a taxa de amostragem do sensor.

Os blocos de gravação **não** são disjuntos entre si, e o argumento não precisa
que sejam. `scripts/eda.py` reporta 16 dos 26 blocos brutos com sobreposição
temporal, porque alguns defeitos foram retomados semanas depois:
`desbalanceado_1parafuso` ocupa 32,7 dias e `cocked_rotor`, 14,0 dias. O que
sustenta a rejeição do sorteio é a proximidade **dentro** de cada bloco, e é por
isso que `split_temporal` corta dentro de cada bloco — `groupby("fault_original")`,
corte cronológico e intervalo de guarda no meio —, em vez de separar blocos
inteiros.

A mesma medição vale para a partição por campanha, e vale contra ela: 7.207
eventos da primeira campanha, todos de `desalinhado` e
`desbalanceado_1parafuso`, foram gravados **depois** do início da segunda. O bloco
`desalinhado` inteiro foi gravado entre `correia_2` e `normal_2`. Para essas duas
classes, a queda de desempenho na partição por campanha não pode ser lida só como
deriva no tempo.

O mesmo cuidado vale para a demonstração ao vivo, e não só para a avaliação. A aba
Diagnóstico do painel sorteia um evento do próprio histórico e consulta o índice
de produção, que contém as leituras vizinhas daquele mesmo bloco. Excluir só o id
do evento consultado não resolve — a leitura gravada dois segundos depois continua
no índice, a distância praticamente zero. Por isso a consulta exclui a
**vizinhança temporal**: sai da busca todo registro gravado a menos de 300 s do
evento. Um evento novo de verdade não tem gêmeo no passado, então a janela não
muda nada para ele; ela existe para que a demonstração rode contra o mesmo sistema
que foi medido.

O tamanho do efeito está medido. Nos **mesmos 300 eventos sorteados do histórico**
(semente 42), consultados contra o índice de produção sob os dois critérios:

| | Critério anterior<br>só o próprio id fora, peso `1/(d+1e-6)` | Critério atual<br>vizinhança de 300 s fora, peso com piso |
|---|---|---|
| Peso do vizinho mais próximo (mediana) | 0,99997 | 0,05089 |
| Consultas com um vizinho acima de 50% do peso | 62,3% | 0,0% |
| Taxa de rejeição | 16,3% | 42,3% |

À esquerda o kNN é um 1-NN disfarçado: o gêmeo gravado dois segundos antes carrega
praticamente todo o peso e traz o rótulo certo de graça, e a rejeição cai para
16,3%. À direita — que é o que a demonstração roda hoje — o peso do vizinho mais
próximo fica em 0,051 contra 0,040 do peso uniforme entre 25 vizinhos, ou seja
quase plano, e a rejeição vai para 42,3%, a mesma ordem de grandeza dos 33,5% da
partição temporal. Num levantamento maior, de 1.000 eventos, a taxa fica em 39,5%.

As duas mudanças — a janela temporal e o piso no peso — entraram juntas e foram
medidas juntas, então os 26 pontos são o efeito somado. É a diferença entre
demonstrar o sistema e demonstrar a taxa de amostragem do sensor, e é a mesma
objeção desta seção aplicada à tela.

---

## 2. Desempenho por partição

Cada partição é medida em três níveis, porque acertar o nome do defeito, entregar
o procedimento certo e terminar o atendimento de forma correta são coisas
diferentes.

**Acerto de rótulo** cobra o nome exato do defeito.

**Documento certo** cobra o que o sistema põe na mão de quem vai abrir a máquina:
qual dos procedimentos cadastrados foi entregue. O mapa rótulo → documento não
está escrito na avaliação — é descoberto pela mesma `BaseConhecimento.cobertura()`
que o motor usa, então a avaliação mede o que a solução de fato entrega. Quando o
defeito real não tem procedimento cadastrado, **não existe documento certo a
entregar**, e o evento não pode contar como acerto aqui.

**Desfecho correto** cobra o fim de linha. Soma ao anterior o acerto que não cabe
nele por definição: o defeito real não tem procedimento e o sistema não entregou
procedimento nenhum. Um evento de ventoinha reconhecido como ventoinha termina em
"não existe procedimento cadastrado, cadastre um" — que é exatamente o que o
enunciado pede.

**Prescrição indevida** é o erro que o enunciado proíbe de forma direta, e que
nenhuma métrica de acerto mostra: a fração de eventos reconhecidos em que o
defeito real não tem procedimento, o sistema o confundiu com um que tem, e por
isso entregaria um procedimento.

| Partição | Eventos | Rejeição | Acerto de rótulo | Documento certo | Desfecho correto | Prescrição indevida |
|---|---|---|---|---|---|---|
| Aleatória — **inválida**, só contraste | 2.001 | 42,3% | 88,8% | 77,9% | 93,9% | 1,0% |
| **Temporal — a medida principal** | 1.998 | 33,5% | **48,8%** | **54,8%** | **72,2%** | **4,1%** |
| Nova campanha — estresse | 2.002 | 42,5% | 27,0% | 38,0% | 48,5% | 5,4% |

As quatro colunas de acerto são calculadas entre os eventos que o sistema aceitou
diagnosticar. Os rejeitados não recebem palpite: recebem "não reconheço este
padrão".

### Leitura

**A queda de 88,8% para 48,8% no acerto de rótulo é a medida do vazamento.**
Quarenta pontos que a partição aleatória entrega de graça e que não existem em
operação.

**Na partição temporal, documento certo (54,8%) fica acima do acerto de rótulo
(48,8%)**, e a diferença tem uma causa só: os quatro defeitos de rolamento levam
ao mesmo Doc1 e à mesma ação corretiva, então trocar anel interno por anel externo
não muda uma linha da instrução entregue ao técnico. O experimento da seção 3 mede
esse argumento em vez de assumi-lo: sob a métrica estrita, a família rolamento
mantém entre 0,77 e 0,95 de documento ainda correto, e **todos os outros defeitos
vão a zero**. A justificativa vale para a família rolamento e só para ela — por
isso a métrica compara documento entregue, e não componente.

**Na partição aleatória a ordem se inverte** — 88,8% de rótulo contra 77,9% de
documento — pelo mesmo mecanismo lido ao contrário: ali o sistema acerta o nome de
quase tudo, inclusive de `eccentric_rotor` e `ventoinha`, e esses acertos não
entram em "documento certo" porque não há documento a entregar. Eles reaparecem em
"desfecho correto", que sobe para 93,9%. As três colunas medem coisas diferentes
de propósito, e a diferença entre elas é informação, não ruído.

**4,1% de prescrição indevida na medida principal.** De cada cem eventos que o
sistema aceita diagnosticar, quatro são de um defeito sem procedimento cadastrado
e receberiam um procedimento assim mesmo. É o erro mais caro que esta solução pode
cometer sob o enunciado, e o número existe justamente para não ficar escondido
atrás de uma média de acerto.

**A queda para 27,0% de acerto de rótulo na nova campanha não é ruído: é deriva
medida.** A linha de base se deslocou entre coletas. Restrito à condição `normal`
— máquina sem defeito nas duas rodadas —, a mediana da aceleração RMS no eixo Z
sobe 13,6% da primeira campanha para a segunda, e 25,2% no regime de 2000 rpm
(`python scripts/eda.py`, hipótese H5). Quando o referencial se move, a comparação
com o histórico antigo perde valor, e é isso que o número mostra.

Esse resultado tem consequência de projeto, não é só um número ruim. Ele é a
justificativa concreta para o **monitoramento de deriva** descrito em
[`ARQUITETURA.md`](ARQUITETURA.md): a taxa de rejeição sobe de 33,5% para 42,5%
sob deriva, o que a torna um alarme utilizável. Quando ela sobe de forma
sustentada em campo, o índice precisa ser reconstruído com dados recentes.

---

## 3. Defeito inédito: o teste que separa esta solução de um classificador fechado

Uma classe inteira é removida do índice e depois consultada, 300 eventos por
classe. É a situação que o enunciado quer evitar: um defeito que o sistema nunca
viu.

Um classificador supervisionado não tem saída aqui — ele devolve, com confiança, a
classe conhecida menos errada. Aqui há duas saídas aceitáveis: **rejeitar**, ou
**errar o nome e ainda assim terminar certo** (entregando o mesmo documento, ou
não entregando documento nenhum quando não existe um).

| Defeito removido | Rejeitou | Documento ainda correto | Prescrição indevida | Desfecho útil | Para onde foi |
|---|---|---|---|---|---|
| rolamento_combination | 53,0% | 95,0% | 0,0% | **97,7%** | rolamento_inner |
| rolamento_inner | 31,3% | 93,2% | 0,0% | **95,3%** | rolamento_combination |
| rolamento_ball | 35,7% | 83,4% | 0,0% | **89,3%** | rolamento_inner |
| rolamento_outer | 46,0% | 77,2% | 0,0% | **87,7%** | rolamento_inner |
| eccentric_rotor | 53,0% | 0,0% | **66,7%** | 60,3% | cocked_rotor |
| cocked_rotor | 59,3% | 0,0% | 0,0% | 59,3% | correia |
| ventoinha | 40,7% | 0,0% | **74,7%** | 55,3% | polia |
| desbalanceado_1parafuso | 54,7% | 0,0% | 0,0% | 54,7% | eccentric_rotor |
| desalinhado | 47,3% | 0,0% | 0,0% | 47,3% | correia |
| polia | 44,7% | 0,0% | 0,0% | 44,7% | normal |
| correia | 37,7% | 0,0% | 0,0% | 37,7% | cocked_rotor |
| **média** | **45,8%** | — | — | **66,3%** | |

> **Como ler as colunas.** "Documento ainda correto" e "prescrição indevida" são
> frações dos eventos **reconhecidos**; "rejeitou" e "desfecho útil" são frações
> do **total**. Para `eccentric_rotor` e `ventoinha` a coluna de documento é zero
> **por construção** — esses dois defeitos não têm procedimento cadastrado, então
> não existe documento certo a entregar. Quem descreve essas duas linhas é a
> coluna de prescrição indevida. O próprio `scripts/evaluate.py` imprime essa
> ressalva ao final da tabela.

### Leitura

O resultado se separa em três grupos, e a separação tem explicação física.

**Quando o defeito removido tem irmão na mesma família, o sistema degrada com
elegância.** Tire `rolamento_combination` do índice e ele cai em
`rolamento_inner` — outro rolamento, mesmo Doc1, mesma ação corretiva. O nome
está errado, a prescrição está certa. Os quatro defeitos de rolamento ficam entre
87,7% e 97,7% de desfecho útil, e são os **únicos** com documento ainda correto
diferente de zero. É esse resultado, e não uma suposição sobre o catálogo, que
autoriza tratar os quatro rolamentos como um destino só na seção 2.

**Quando o defeito é o único da sua família, a degradação é feia.** `correia`,
`polia`, `desalinhado`, `cocked_rotor` e `desbalanceado_1parafuso` não têm par no
histórico: removidos, o vizinho mais próximo passa a ser um defeito de outro
componente, e sob a métrica estrita o documento entregue está errado em 100% dos
casos que passam. A rejeição segura entre 37,7% e 59,3% deles; o restante passaria
com o procedimento de outro defeito.

**O pior caso está em `polia`, que cai em `normal`.** Chamar defeito de condição
normal é o pior desfecho quando o defeito é inédito, porque não gera nem ação
nem alerta.

**E o número mais desconfortável do projeto está nas duas linhas sem documento.**
Removidos do índice, `eccentric_rotor` e `ventoinha` são confundidos com defeitos
que **têm** procedimento, e o sistema prescreveria sem lastro em 66,7% e 74,7% dos
eventos que aceita diagnosticar. É a violação direta do requisito central do
enunciado, medida em vez de escondida. Ela não aparece na operação normal do
sistema — com as duas classes no índice, a prescrição indevida na partição
temporal é de 4,1% — mas mostra onde o desenho tem limite: a recusa por falta de
documento depende de o defeito ser **reconhecido** primeiro, e um defeito inédito
não é. O caminho para reduzir isso é a rejeição ficar mais sensível, e é isso que
os dois portões da seção 3.4 da arquitetura controlam.

---

## 4. Ablação de features

Reprodução: `python scripts/ablacao_features.py`, sobre a partição temporal
(treino 86.543 eventos, teste avaliado 1.501).

| Variante | Features | Acerto entre reconhecidos | Rejeição | Acerto global |
|---|---|---|---|---|
| A. tudo | 25 | 47,4% | 30,0% | 33,2% |
| B. sem frequência de pico | 23 | 47,6% | 30,2% | 33,2% |
| C. sem temperatura | 24 | 47,8% | 32,6% | 32,2% |
| D. sem frequência e temperatura | 22 | 47,9% | 32,6% | 32,2% |
| **E. produção — sem frequência, temperatura e ordem** | **20** | **49,0%** | 34,2% | 32,2% |

A variante de produção é ao mesmo tempo a de **maior acurácia entre os
reconhecidos** (49,0%) e a de **maior rejeição** (34,2%). É exatamente o
comportamento desejado num sistema de conjunto aberto: remover as colunas-artefato
faz o sistema reconhecer menos e acertar mais quando reconhece. O ganho de
acurácia é de 1,6 ponto, dentro da margem de ruído da amostra — **as duas colunas
foram removidas pela análise, não pela métrica.**

Os números diferem levemente dos da seção 2 (49,0% contra 48,8% de rótulo, 34,2%
contra 33,5% de rejeição) por dois motivos declarados: a amostra é menor (1.501
contra 1.998) e a calibração dos limiares usa 800 eventos em vez de 1.500. A
ablação existe para comparar variantes entre si, e todas as cinco rodam sob as
mesmas condições.

**Frequência de pico** tem 14 valores distintos no eixo Z e 19 no X, e 61 Hz
responde por 61,5% das linhas em Z e 47,8% em X — inclusive nos 347 registros de
motor desligado, onde a mediana é exatamente 61,0 Hz. Vibração rotacional não
existe com o motor parado: 61 Hz é a frequência da rede elétrica. E o
`RobustScaler` é ajustado por regime, onde o IQR dessa coluna varia de 1,0 a 44,0;
onde ele é pequeno, a amplitude da coluna sozinha vale até 52 unidades de
distância, contra as 2,1 que separam dois eventos parecidos. As features `ordem_z`
e `ordem_x` saíram junto: derivam dessa coluna, e se a origem é artefato, a razão
também é.

**Temperatura** correlaciona com a posição dentro do bloco de gravação de 0,009 a
0,933, com o sinal trocando entre blocos, e os sete blocos acima de 0,80 são todos
da segunda rodada de gravação — aquecimento e resfriamento ambiente da sessão, não
física de falha. E separa mal: agrupando pelo rótulo normalizado, a variação entre
defeitos (1,18 °C) é menor que a variação dentro de cada defeito (1,63 °C), razão
0,72. Medida pelo rótulo bruto a razão sai 1,09 e diria o contrário — mas ali a
diferença entre duas campanhas do *mesmo* defeito entra na conta como se fosse
separação entre defeitos diferentes, que é justamente o que a normalização de
rótulo existe para desfazer.

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

O mapa resultante, gravado em `data/index/avaliacao.json` no campo
`mapa_documento` e servido em tempo real por `GET /cobertura`:

| Destino | Defeitos |
|---|---|
| Doc1 | rolamento_ball, rolamento_combination, rolamento_inner, rolamento_outer |
| Doc2 | desalinhado |
| Doc3 | desbalanceado_1parafuso |
| Doc4 | correia |
| Doc5 | polia |
| Doc6 | cocked_rotor |
| sem documento | eccentric_rotor, ventoinha |
| condição operacional | normal, baseline, teste, acelerando, motor_desligado |

Base: 6 documentos, 170 trechos.

---

## 6. Auditoria adversarial do chat

A tabela da seção 5 mede a cobertura a partir do **catálogo**. Ela não mede o que
acontece quando um técnico digita a pergunta com as palavras dele. Esta seção mede.

**Como o corpus foi feito.** 101 perguntas geradas por três agentes adversariais
com ângulos distintos — descrever só o sintoma, trocar o componente por sinônimo,
e atacar a estrutura do portão — e depois **curadas** por revisores que julgaram
cada uma contra o texto real dos seis procedimentos. A curadoria não foi
cosmética: **22 das 101 descreviam defeito legitimamente coberto** e viraram
controle, e **13 ficaram ambíguas** a ponto de nenhum engenheiro decidir sem
espectro ou medição de entreferro, e foram excluídas dos dois denominadores.
Reportar contra o corpus bruto teria errado nos dois sentidos.

Restam **66 ataques** (o sistema tem de recusar). Os **52 controles** (o sistema
tem de responder, e pelo procedimento certo) são essas 22 reclassificadas mais 30
perguntas de manutenção bem formadas, que não vieram da geração adversarial.
Corpus em
`tests/dados/corpus_chat_adversarial.json`; `python scripts/auditoria_chat.py`
refaz a conta em segundos, com um dublê no lugar do modelo.

### Antes e depois

| | portão anterior | portão atual |
|---|---|---|
| paráfrase | 12/46 vazaram | **0/32** |
| seletor da tela | 22/23 vazaram | **0/11** |
| co-citação | 32/32 vazaram | 16/23 |

*As colunas têm denominadores diferentes porque a coluna da esquerda foi medida
antes da curadoria, sobre os rótulos do gerador. É o número honesto de cada
momento; o que se compara entre elas é o mecanismo, não a fração.*

### Os quatro números do portão atual

| | |
|---|---|
| vazamento | 16/66 — todos de co-citação |
| mudez, pergunta bem formada | **1/30** |
| mudez, pergunta só de sintoma | 17/22 |
| desvio (respondeu pelo procedimento errado) | 1/52 |

Mudez e desvio saem separados de propósito: o técnico percebe que ficou sem
resposta e **não** percebe que recebeu o procedimento de outro defeito.

### Ablação do portão

`python scripts/ablacao_portao.py` — cada linha é o portão completo menos uma regra.

| configuração | vazou | calou | desviou |
|---|---|---|---|
| portão completo | 16/66 | 17/52 | 1/52 |
| sem `termos_ambiguos` | 18/66 | 14/52 | 4/52 |
| sem `termos_pergunta` | 21/66 | 22/52 | 2/52 |

`termos_pergunta` é ganho puro: tirá-lo piora as três colunas. `termos_ambiguos`
compra 2 vazamentos e 3 desvios a menos por 3 recusas a mais — troca boa, porque
desvio é o erro invisível.

### O resíduo, declarado

**16 vazamentos, todos de co-citação, e não foram consertados de propósito.** O
padrão é este: a pergunta nomeia um defeito **coberto**, quase sempre para
descartá-lo, e descreve sem nomear um defeito sem procedimento — *"já troquei os
rolamentos dos dois lados e a trepidação continua"*. O sistema responde com o
procedimento de rolamento.

Ele não inventa texto e não prescreve para o defeito sem documento; entrega o
procedimento do defeito que a pergunta nomeou, **com a etiqueta dizendo de qual
defeito ele é** — a tela e a API mostram isso antes das instruções.

Consertar exigiria detectar que o defeito citado foi *descartado* pelo técnico. As
formas de escrever isso em português não cabem numa lista sem virar o mesmo jogo de
adivinhação de que este portão acabou de sair. Fica declarado e travado por
catraca em `tests/test_auditoria_portao.py`, que falha se piorar.

### A mudez de sintoma não é falha do portão

17 dos 22 controles reclassificados descrevem **só o sintoma**, sem nomear defeito:
*"o motor treme na frequência de giro e o ronco sobe e desce"*. O chat recusa e
pede o nome do defeito. É o comportamento pretendido: transformar sintoma em
defeito é trabalho do **caminho do diagnóstico**, que tem o sensor. No chat não há
sensor, e foi medido nesta mesma seção que a semelhança de palavras não decide.

---

## 7. O que estes números não dizem

**Não medem detecção precoce.** Todos os eventos do conjunto são de defeito já
instalado ou de operação normal. Não há trajetória de degradação, então não dá
para avaliar antecedência — que é metade do valor de manutenção preditiva.

**Não medem qualidade do texto gerado.** "Documento certo" mede se o procedimento
certo foi selecionado, não se a instrução redigida a partir dele é boa. Avaliar
isso exige um humano da manutenção lendo as saídas.

**Não há varredura de sensibilidade dos hiperparâmetros.** `n_neighbors: 25`,
`reject_distance_percentile: 99.0` e `min_consensus: 0.45` foram escolhidos e não
varridos — a ablação varia features, não constantes. Isso importa porque a origem
da rejeição está medida: reconstruindo o índice sobre a partição temporal e
classificando cada recusa pelo portão que a produziu, dos **125 eventos rejeitados
em 399, 6 caíram pelo limiar de distância e 119 pelo portão de consenso**. O
consenso responde por 95,2% do trabalho, ou seja a propriedade que define o
projeto repousa quase toda sobre o `0.45`. É a lacuna metodológica mais visível
desta entrega, e a curva rejeição × acurácia em torno desse valor — com a
assimetria de custo declarada, já que chamar defeito de `normal` custa mais que um
falso alarme — é o próximo trabalho.

**Vêm de uma bancada, não de um chão de fábrica.** Quatro rotações fixas, um
defeito por vez, ambiente controlado. Em operação real há carga variável,
múltiplos defeitos simultâneos e transição gradual entre estados. A partição por
campanha é a melhor aproximação disponível de "o mundo mudou", e ela já mostra
queda de 22 pontos no acerto de rótulo.
