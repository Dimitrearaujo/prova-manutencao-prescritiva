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
| Evento novo em JSON, com 24 métricas de sensores de vibração e a rotação | Tipo de defeito, quantidade e frequência de ocorrências, distribuição no tempo, e instruções de correção extraídas do procedimento aplicável |

Quatro desfechos possíveis, e três deles **não** prescrevem nada:

| Desfecho | Quando acontece | O que o sistema responde |
|---|---|---|
| Defeito documentado | Padrão reconhecido, é defeito, há procedimento | Diagnóstico + histórico + instruções de correção |
| Defeito sem documentação | Padrão reconhecido, é defeito, **não** há procedimento | Informa a lacuna e pede o cadastro do documento |
| Condição de operação | Padrão reconhecido, mas é `normal`, `acelerando`, `motor_desligado`… | Informa que não é problema, nenhuma ação corretiva |
| Padrão não reconhecido | Não se parece com nada do histórico | Recusa o diagnóstico e pede registro da condição |

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
ponderado deles. Um defeito inédito não tem vizinho próximo, aparece como
distância alta e é recusado.

Isso dá três propriedades que um classificador não teria: **rastreabilidade** —
cada diagnóstico vem acompanhado dos eventos históricos que o sustentam, com id e
data, auditáveis pelo técnico; **atualização sem retreino** — indexar um evento
novo é acrescentar uma linha; e **conjunto aberto** — o sistema sabe dizer que não
sabe.

O detalhamento das decisões está em [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md).

---

## Como executar

Requer Python 3.11+. O `banner.csv` e os PDFs não estão versionados; baixe-os da
pasta do enunciado para `data/raw/banner.csv` e `data/docs/`.

```bash
python -m venv .venv && .venv/Scripts/activate     # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

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
python scripts/eda.py                # as 4 hipoteses que justificam o tratamento dos dados
python scripts/evaluate.py           # os 4 experimentos de avaliacao
python scripts/ablacao_features.py   # o que cada grupo de features contribui
pytest                               # testes
```

### Modelo de linguagem

A geração das instruções usa um modelo local via [Ollama](https://ollama.com),
conforme a restrição de operar em estação de trabalho sem depender de serviço
externo:

```bash
ollama pull qwen2.5:3b
```

**Sem o Ollama a solução continua funcionando por inteiro.** A camada de geração é
plugável e cai automaticamente num gerador determinístico, que monta a resposta
recortando os trechos recuperados do procedimento. Perde-se a redação fluida;
não se perde diagnóstico, histórico, cobertura nem instrução correta.

---

## O que a análise dos dados revelou

`scripts/eda.py` testa quatro hipóteses antes de qualquer decisão virar código.
Duas viraram regra de ingestão, duas eliminaram features.

**As colunas imperiais são redundantes.** Cinco pares medem a mesma grandeza em
unidades diferentes, com correlação 1,000000 e fator 25,4. Manter os dois lados
dá peso duplo à mesma medida na distância. Descartadas.

**O sufixo `_2`/`_3` é campanha de coleta.** Confirmado pelo tempo, não pela
distribuição: cada rótulo ocupa um bloco contínuo e sem sobreposição, e a segunda
rodada roda inteira no fim do período. Os rótulos foram unificados para
diagnóstico e cobertura, mas o original ficou preservado — a campanha é o grupo
de vazamento que a avaliação precisa separar.

**A frequência de pico não mede o defeito.** A coluna tem apenas 14 valores
distintos, e 61 Hz responde por 61% das linhas — inclusive nos 347 registros de
**motor desligado**. Vibração rotacional não existe com o motor parado: 61 Hz é a
frequência da rede elétrica. Como o `RobustScaler` divide por um IQR de 1,25, um
deslocamento dessa coluna entre coletas vira dezenas de unidades de distância e
domina a busca por vizinhos.

**A temperatura é um relógio, não um sintoma.** Correlaciona com a posição dentro
do bloco de gravação entre 0,50 e 0,93 — subindo em alguns blocos e caindo em
outros, o que é aquecimento e resfriamento ambiente. E separa mal: a variação
entre defeitos (1,18 °C) é menor que a variação dentro de cada defeito (1,63 °C),
com as faixas quase totalmente sobrepostas.

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
objetivo —, nunca com o corpo inteiro. A distinção importa: o procedimento de
rolamentos lista "Ventiladores" entre os equipamentos onde rolamentos são
críticos, e uma regra que olhasse o corpo faria a falha de ventoinha receber o
procedimento de rolamento.

Como a descoberta é dinâmica, **cadastrar um PDF novo pelo painel faz a cobertura
mudar sem alteração de código**. `tests/test_cobertura.py` trava o mapa contra a
leitura humana dos seis documentos e verifica que o limiar está numa faixa vazia,
não no fio da navalha: a menor aderência coberta é 2,14 e a maior rejeitada é
0,50, com o limiar em 1,0.

---

## Avaliação

Ver [`docs/RESULTADOS.md`](docs/RESULTADOS.md) para os números completos e a
leitura de cada experimento.

O ponto metodológico central: **um split aleatório é inválido neste conjunto de
dados.** O ensaio gravou um defeito de cada vez, em blocos contínuos de tempo, com
leituras a segundos de distância. Sortear treino e teste coloca a mesma medição
dos dois lados da partição. Todas as partições usadas respeitam o tempo, e o
split aleatório aparece apenas como contraste — para mostrar o tamanho da ilusão
que produz.

---

## Estrutura

```
config/            settings.yaml e catalogo de defeitos
data/              raw, docs, processed, index (nao versionados)
docs/              arquitetura, resultados, enunciado
src/prescritiva/   pacote da solucao
scripts/           etapas do pipeline, analise e avaliacao
app/               interface Streamlit
tests/             testes
```
