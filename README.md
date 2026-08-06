# Manutenção Prescritiva por Similaridade Histórica

Pipeline completo de manutenção prescritiva para máquinas rotativas: um evento de
vibração entra, o sistema encontra ocorrências semelhantes no histórico
operacional, decide se aquilo é um defeito, verifica se existe procedimento
cadastrado para ele e só então prescreve a correção.

Desenvolvido como resposta ao desafio técnico de Desenvolvedor Full Stack —
I.A. e Python.

---

## O que a solução faz

| Entrada | Saída |
|---|---|
| Evento novo em JSON, com 22 métricas de sensores de vibração e a rotação | Tipo de defeito, quantidade e frequência de ocorrências, distribuição no tempo, contexto operacional e instruções de correção extraídas do procedimento aplicável |

Quatro desfechos possíveis, e três deles **não** prescrevem nada:

| Desfecho | Quando acontece | O que o sistema responde |
|---|---|---|
| Defeito documentado | Padrão reconhecido, é defeito, há procedimento | Diagnóstico + histórico + instruções de correção |
| Defeito sem documentação | Padrão reconhecido, é defeito, **não** há procedimento | Informa a lacuna e pede o cadastro do documento |
| Condição de operação | Padrão reconhecido, mas é `normal`, `acelerando`, `motor_desligado`… | Informa que não é problema, nenhuma ação corretiva |
| Padrão não reconhecido | Não se parece com nada do histórico, ou chegou numa rotação sem histórico comparável | Recusa o diagnóstico e pede registro da condição |

---

## A decisão que define o projeto

O enunciado é explícito: a solução **não deve depender da classificação prévia de
falhas conhecidas**. Isso descarta o caminho óbvio.

Um classificador supervisionado treinado com `fault` como alvo aprende um
conjunto fechado de classes. Diante de um defeito que nunca viu, ele não tem como
dizer "não sei": devolve, com confiança alta, a classe conhecida menos errada. Em
manutenção industrial isso significa mandar um técnico trocar um mancal por causa
de um problema que era outro.

Aqui **o histórico é o modelo**. Não existe treino com rótulo como alvo. Um evento
novo é comparado ao passado por vizinhança em espaço de features padronizadas, os
vizinhos encontrados trazem seus próprios rótulos, e o diagnóstico é o consenso
deles. Um defeito inédito não tem vizinho próximo, aparece como distância alta e
é recusado.

Isso dá três propriedades que um classificador não teria: **rastreabilidade** —
cada diagnóstico vem acompanhado dos eventos históricos que o sustentam, com id e
data, auditáveis pelo técnico; **atualização sem retreino** — indexar um evento
novo é acrescentar uma linha; e **conjunto aberto** — o sistema sabe dizer que não
sabe.

O detalhamento das decisões está em [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md).

---

## Como executar

Requer Python 3.11 ou 3.12 (a imagem Docker usa 3.12). Em 3.13+ a instalação não
resolve: `rapidocr-onnxruntime` 1.3.24 declara `Requires-Python <3.13` e `numpy`
1.26.4 não publica wheel para cp313.

Os seis PDFs de procedimento estão versionados em `data/docs/`; o `banner.csv` não
está — copie-o da pasta do enunciado para `data/raw/banner.csv`.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
```

Linux/Mac:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

No Windows, use `.venv\Scripts\python.exe` no lugar de `python` nos comandos
abaixo — os blocos seguintes supõem a venv ativa, e a forma sem ativação evita
depender da `ExecutionPolicy` do PowerShell.

Construção dos artefatos, nesta ordem:

```bash
python scripts/ingest.py           # CSV -> parquet + SQLite indexado
python scripts/build_index.py      # indice de vizinhos + calibracao dos limiares
python scripts/build_knowledge.py  # PDFs -> texto (OCR no Doc1) -> indice + mapa de cobertura
```

Interface e API:

```bash
streamlit run app/streamlit_app.py
uvicorn prescritiva.api.main:app --reload
```

Análise e avaliação:

```bash
python scripts/demo.py               # os 4 desfechos em linha de comando (~1 min)
python scripts/eda.py                # as 5 hipoteses que justificam o tratamento dos dados (~15 s)
python scripts/evaluate.py           # os 4 experimentos de avaliacao (~25 min - rode antes, nao ao vivo)
python scripts/ablacao_features.py   # o que cada grupo de features contribui (~20 min - rode antes, nao ao vivo)
pytest                               # testes
```

### Modelo de linguagem

A geração das instruções usa um modelo local via [Ollama](https://ollama.com),
conforme a restrição de operar em estação de trabalho sem depender de serviço
externo:

```bash
ollama pull qwen2.5:3b
```

Para comparar modelos no mesmo caso real, com tempo medido na máquina onde vai
rodar:

```bash
python scripts/comparar_modelos.py qwen2.5:3b llama3.2:3b
```

**Sem o Ollama a solução continua funcionando por inteiro.** A camada de geração é
plugável e cai num gerador determinístico, que monta a resposta recortando os
trechos recuperados do procedimento. A troca acontece em dois momentos: na
seleção do gerador, se o Ollama não responder ao subir; e a cada chamada, se ele
cair depois — a geração é protegida, o recorte assume no mesmo pedido e a
resposta sai marcada como `deterministico (fallback em execucao)`, para o técnico
saber que o texto veio recortado e não reescrito. Perde-se a redação fluida; não
se perde diagnóstico, histórico, cobertura nem instrução correta. O
determinístico também serve de linha de base: como só reproduz texto já aprovado
pela engenharia, ele não tem como alucinar, e é contra ele que o modelo precisa
provar que vale a pena.

A contenção de alucinação tem quatro camadas, e a mais fraca delas — a instrução
que proíbe o modelo de acrescentar o que não está no texto — **falhou numa
medição**: em duas execuções do mesmo caso, uma das saídas citou dois instrumentos
que não aparecem nos trechos recuperados. As três camadas acima dela são
determinísticas e seguraram. O caso está descrito em
[`docs/ARQUITETURA.md`](docs/ARQUITETURA.md) §3.9, com o que ele implica sobre
quando usar o gerador determinístico em produção.

---

## API

`uvicorn prescritiva.api.main:app` sobe o contrato HTTP com o sistema que já
recebe os eventos dos sensores. OpenAPI interativo em `/docs`.

| Método | Rota | O que faz |
|---|---|---|
| `GET` | `/saude` | gerador em uso, documentos e trechos carregados, regimes indexados |
| `POST` | `/diagnosticar` | diagnostica um evento e registra o resultado na tabela `diagnosticos` |
| `GET` | `/diagnosticos` | histórico de diagnósticos emitidos, com a contagem por desfecho |
| `POST` | `/perguntar` | consulta livre à base documental (o chat), com o mesmo portão de cobertura |
| `GET` | `/documentos` | procedimentos indexados, com título, páginas e método de extração |
| `POST` | `/documentos` | cadastra um PDF novo e reindexa a base |
| `GET` | `/cobertura` | mapa defeito → documento como ele está agora |

`POST /documentos` recusa antes de gravar: `400` para nome de arquivo com
caminho, fora da allowlist, que não abre como PDF ou cujo texto extraído não
sustenta um procedimento consultável; `413` acima de 25 MB; `401` se
`cadastro.chave_acesso` estiver configurada e o cabeçalho `X-Prescritiva-Key`
vier ausente ou errado. A validação roda numa área temporária de propósito —
um arquivo ruim já gravado em `data/docs/` derrubaria toda reindexação
seguinte, que é justamente o mecanismo de recuperação do desfecho "defeito sem
documentação". A chave é opcional e vem desligada por padrão (nenhuma
instalação nova fica travada de saída); configurada por
`PRESCRITIVA_CADASTRO_KEY`, ela vale igual para a API e para o cadastro pelo
painel — reescrever o procedimento que o técnico vai seguir não pode ser ação
de qualquer um numa planta segmentada, e essa é uma decisão de arquitetura, não
um extra de segurança avulso. Detalhe completo em
[`docs/ARQUITETURA.md`](docs/ARQUITETURA.md) §3.10.

O painel Streamlit tem quatro abas — Dados, Diagnóstico, Procedimentos e
Avaliação — e importa o motor em processo, sem passar pela API. O cadastro de
documento novo mora na aba Procedimentos e passa pelas mesmas proteções do
`POST /documentos`, incluindo a chave de acesso quando configurada.

---

## Integração em ambiente industrial

O sensor não pergunta: ele publica. `scripts/consumidor_mqtt.py` assina o tópico
de eventos, chama o **mesmo** `MotorDiagnostico` que a API usa e publica o
diagnóstico de volta.

```bash
python scripts/consumidor_mqtt.py --host localhost --port 1883
python scripts/simular_chao_de_fabrica.py --modo roteiro --intervalo 5
```

| Tópico | Conteúdo |
|---|---|
| `planta/vibracao/eventos` | entrada, um registro de sensor por mensagem |
| `planta/vibracao/diagnosticos` | saída, todos os quatro desfechos |
| `planta/vibracao/diagnosticos/erros` | payload que o sistema não conseguiu interpretar |

Erro de integração e recusa de diagnóstico saem por canais diferentes de
propósito: quem integra do outro lado precisa distinguir "não entendi a
mensagem", que é problema de quem publicou, de "não reconheci o padrão", que é
informação de manutenção. A confirmação ao broker é manual e só acontece depois
de publicar o diagnóstico, para que uma queda no meio do processamento faça o
broker reentregar em vez de perder o evento.

O simulador ocupa o lugar do gateway na demonstração e remove o rótulo anotado
pelo operador antes de publicar — enviá-lo junto faria o consumidor parecer
acertar quando na verdade recebeu a resposta pronta.

Todo diagnóstico emitido, por HTTP ou por fila, é gravado na tabela
`diagnosticos` do mesmo SQLite de onde o histórico é lido. A contagem por
situação é o indicador operacional: `padrao_desconhecido` subindo significa que a
máquina saiu da condição em que o índice foi construído.

---

## Deploy

```bash
docker compose up -d --build
```

Uma imagem para os dois processos (API na 8000, painel na 8501), dado por volume
e Ollama no host, onde está a GPU. Os artefatos de índice **precisam existir
antes do primeiro `up`** — sem eles a API responde `503` e o container aparece
como `unhealthy`, que é o comportamento desejado.

O raciocínio inteiro — o que muda na estação industrial, por que o SQLite tem
prazo de validade, o consumo de memória medido e as limitações conhecidas do
empacotamento — está em [`docs/DEPLOY.md`](docs/DEPLOY.md).

---

## O que a análise dos dados revelou

`scripts/eda.py` testa cinco hipóteses antes de qualquer decisão virar código.
Duas viraram regra de ingestão, duas descartaram features e uma mede a deriva de
linha de base que a avaliação por campanha encontra depois.

**As colunas imperiais são redundantes.** Cinco pares medem a mesma grandeza em
unidades diferentes, com correlação entre 0,999997 e 0,999999. Nos quatro pares de
velocidade a razão medida é 25,41 — polegada para milímetro; o quinto é
Fahrenheit/Celsius, conversão afim, e por isso correlaciona igual sem ter razão
constante. Manter os dois lados dá peso duplo à mesma medida na distância.
Descartadas.

**O sufixo `_2`/`_3` é campanha de coleta.** Confirmado pelo tempo: 26 rótulos
brutos viram 14 depois da normalização. Não é uma linha limpa — 16 dos 26 blocos
de gravação se sobrepõem no tempo, porque alguns defeitos foram retomados semanas
depois, e 7.207 eventos da primeira rodada (`desbalanceado_1parafuso` e
`desalinhado`) foram gravados já durante a segunda. O argumento não depende de os
blocos serem disjuntos: `split_temporal` corta **dentro** de cada bloco. Mas a
exceção importa, e está declarada, porque enfraquece a leitura de deriva temporal
para essas duas classes.

**A frequência de pico não mede o defeito.** As duas colunas têm apenas 14 e 19
valores distintos, e 61 Hz responde por 61,5% das linhas no eixo Z e 47,8% no X —
inclusive nos 347 registros de **motor desligado**, onde a mediana é exatamente
61,0 Hz. Vibração rotacional não existe com o motor parado: 61 Hz é a frequência
da rede elétrica. E o `RobustScaler` é ajustado por regime de rotação,
onde o IQR dessa coluna varia de **1,0 a 44,0** — coluna que mede defeito não
muda de escala assim. Onde o IQR é pequeno, a amplitude da coluna sozinha vale
até **52 unidades de distância**, contra as **2,1** que separam dois eventos que o
sistema considera parecidos: um deslocamento entre coletas domina a busca por
vizinhos sozinho.

**A temperatura é um relógio, não um sintoma.** A correlação com a posição dentro
do bloco de gravação vai de 0,009 a 0,933, com o sinal trocando entre blocos, e
os sete blocos acima de 0,80 são **todos da segunda rodada** — aquecimento e
resfriamento ambiente da sessão, não física de falha. E separa mal: agrupando
pelo rótulo normalizado, a variação entre defeitos (1,18 °C) é menor que a
variação dentro de cada defeito (1,63 °C), razão 0,72.

**A linha de base se deslocou entre as coletas.** Comparando só a condição
`normal`, a mediana da aceleração RMS no eixo Z sobe **13,6%** da primeira para a
segunda campanha, e **25,2%** no regime de 2000 rpm. Como a máquina está sem
defeito nos dois casos, a diferença é de instrumentação ou de ambiente — é a
causa física da queda de desempenho na partição por campanha, e a justificativa
do monitoramento de deriva pela taxa de rejeição.

---

## Cobertura documental

Dos onze defeitos presentes no histórico, **nove têm procedimento cadastrado e
dois não têm**. Isso não é uma falha da solução: é exatamente o caso que o
enunciado manda tratar informando que o problema ainda não está documentado.

| Defeito | Procedimento | Registros |
|---|---|---|
| Rolamento — anel interno, externo, elemento rolante, combinado | Doc1 (escaneado, extraído por OCR) | 54.016 |
| Desalinhamento de eixo | Doc2 | 4.117 |
| Desbalanceamento de rotor | Doc3 | 10.316 |
| Falha em correia | Doc4 | 11.999 |
| Falha em polia | Doc5 | 12.000 |
| Rotor inclinado (cocked rotor) | Doc6 | 13.075 |
| **Rotor excêntrico** | **nenhum** | 14.808 |
| **Falha em ventoinha** | **nenhum** | 11.999 |

Os dois defeitos sem procedimento somam **26.807 registros, 18,6% do histórico**.

**A associação defeito → documento não está escrita no código.** É descoberta em
tempo de execução comparando o defeito com o *escopo* de cada documento — título e
objetivo —, nunca com o corpo inteiro, porque título declara o que o documento
trata e corpo apenas menciona.

Neste acervo a regra do corpo produziria **o mesmo mapa** — os seis procedimentos
são monotemáticos e nomeiam o componente no título. A escolha pelo escopo não
corrige um erro observado aqui: protege a propriedade quando o acervo crescer, que
é justamente o que o cadastro de documento novo permite. O raciocínio completo, e
a justificativa errada que esta seção já publicou, estão em
[`docs/ARQUITETURA.md`](docs/ARQUITETURA.md) §3.6.

Como a descoberta é dinâmica, **cadastrar um PDF novo pelo painel faz a cobertura
mudar sem alteração de código**. `tests/test_cobertura.py` trava o mapa contra a
leitura humana dos seis documentos e verifica que o limiar está numa faixa vazia,
não no fio da navalha: a menor aderência coberta é 2,14 e a maior rejeitada é
0,50, com o limiar em 1,0.

### O mesmo portão quando a pergunta é texto livre

A tabela acima parte do defeito. O chat parte da frase que o técnico digitou, e é
aí que a regra é mais fácil de contornar. Uma auditoria adversarial de 101
perguntas — depois curadas em 66 ataques, 52 controles e 13 ambíguas — mediu o
portão anterior deixando passar **paráfrase (12/46), o combo de restringir
procedimento na tela (22/23) e a citação de um defeito coberto junto (32/32)**.

O portão foi refeito sobre uma frase:

> **O catálogo decide SE o chat responde e quais procedimentos são candidatos; a
> aderência do texto só escolhe entre candidatos — nunca cria um.**

Hoje: **paráfrase 0/32, seletor 0/11, co-citação 16/23**, com **1/30 de recusa
indevida** em perguntas de manutenção bem formadas. O resíduo de co-citação está
declarado, explicado e travado por catraca em
[`docs/RESULTADOS.md`](docs/RESULTADOS.md) §6.

O piso de aderência BM25 que existia foi **removido, não recalibrado**: as
distribuições de score de pergunta legítima e de paráfrase de defeito sem
procedimento se sobrepõem inteiras, e o piso antigo (2,5) chegava a emudecer
`"como balancear o rotor desbalanceado?"`, que pontua 2,01 dentro do próprio Doc3.

```bash
python scripts/auditoria_chat.py
```

---

## Avaliação

**O ponto metodológico central: uma partição aleatória é inválida neste conjunto
de dados.** O ensaio gravou um defeito de cada vez, e dentro de cada gravação as
leituras acontecem a segundos de distância e são quase idênticas. Sortear treino e
teste coloca a mesma medição dos dois lados da partição — o vizinho mais próximo
de um evento de teste vira um evento gravado dois segundos antes, já no treino com
o rótulo certo. Todas as partições usadas respeitam o tempo, e a aleatória aparece
só como contraste.

Cada partição é medida em três níveis, porque acertar o nome do defeito, entregar
o procedimento certo e terminar o atendimento de forma correta são coisas
diferentes.

| Partição | Acerto de rótulo | Documento certo | Desfecho correto | Prescrição indevida | Rejeição |
|---|---|---|---|---|---|
| Aleatória — **inválida**, só contraste | 88,8% | 77,9% | 93,9% | 1,0% | 42,3% |
| **Temporal — a medida principal** | **48,8%** | **54,8%** | **72,2%** | **4,1%** | 33,5% |
| Nova campanha — estresse | 27,0% | 38,0% | 48,5% | 5,4% | 42,5% |

**A queda de 88,8% para 48,8% no acerto de rótulo é a medida do vazamento.**
Quarenta pontos que a partição aleatória entrega de graça e que não existem em
operação.

**Documento certo fica acima do acerto de rótulo na partição temporal** porque os
quatro defeitos de rolamento levam ao mesmo Doc1: confundir anel interno com anel
externo não muda uma linha da instrução entregue. Esse argumento vale para a
família rolamento e **só para ela** — a métrica compara o documento que o sistema
de fato entregaria, descoberto pela mesma função que o motor usa, e não um
agrupamento por componente.

**Prescrição indevida é o número que o enunciado cobra**, e nenhuma métrica de
acerto mostra: a fração de eventos reconhecidos em que o sistema entregaria um
procedimento para um defeito que não tem procedimento nenhum. Na medida principal
são 4,1%.

**A queda na nova campanha é deriva medida, não ruído.** A linha de base se
deslocou entre coletas — na própria condição `normal`, a mediana da aceleração RMS
no eixo Z sobe 13,6%, e 25,2% no regime de 2000 rpm (`python scripts/eda.py`,
hipótese H5). Isso não é só um número ruim: é a justificativa concreta para o
monitoramento de deriva pela taxa de rejeição, descrito na arquitetura.

**O teste que separa esta solução de um classificador fechado** remove uma classe
inteira do índice e pergunta por ela. Média de 45,8% rejeitados corretamente e
66,3% de desfecho útil. O resultado se separa por família: tirando
`rolamento_combination` o sistema cai em `rolamento_inner`, que leva ao mesmo
Doc1 — 97,7% de desfecho útil. Já `correia`, que não tem par no histórico, cai
para 37,7%. E quando o defeito removido é um dos dois **sem** documento, o sistema
prescreve sem lastro em 66,7% (`eccentric_rotor`) e 74,7% (`ventoinha`) dos
eventos que aceita diagnosticar — o número mais desconfortável do projeto, e o
que sustenta a discussão sobre o limite da abordagem.

Números completos, matrizes de confusão e as limitações do que foi medido em
[`docs/RESULTADOS.md`](docs/RESULTADOS.md).

---

## Estrutura

```
config/            settings.yaml e catalogo de defeitos
data/              raw (nao versionado), docs (os 6 PDFs), processed, index, knowledge
docs/              arquitetura, resultados, deploy, enunciado
src/prescritiva/   pacote da solucao (data, features, similarity, knowledge,
                   llm, diagnosis, integracao, api)
scripts/           etapas do pipeline, analise, avaliacao, demo, consumidor MQTT
app/               interface Streamlit
tests/             testes
Dockerfile         imagem unica para API e painel
docker-compose.yml os dois processos, volumes e variaveis de ambiente
```
