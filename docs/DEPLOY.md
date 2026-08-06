# Deploy

Como empacotar e executar a solução em container, e o que muda quando ela sai
da máquina de desenvolvimento e vai para a estação industrial descrita no
enunciado (32 GB de RAM, GPU de 16 GB dedicada).

Arquivos envolvidos: `Dockerfile`, `docker-compose.yml`, `.dockerignore`.

---

## 1. O desenho em uma frase

A imagem carrega **o código e as dependências**. O **dado** entra por volume e o
**modelo de linguagem roda fora do container**, no host.

```
   ┌─────────────── estação / servidor ───────────────┐
   │                                                  │
   │   ┌──────────────┐      ┌──────────────┐         │
   │   │ prescritiva- │      │ prescritiva- │         │
   │   │ api  :8000   │      │ painel :8501 │         │
   │   └──────┬───────┘      └──────┬───────┘         │
   │          │  mesma imagem       │                 │
   │          └──────────┬──────────┘                 │
   │                     │                            │
   │        volumes ─────┼───── ./data/{index,        │
   │                     │       processed,docs,      │
   │                     │       knowledge}, .db      │
   │                     │                            │
   │        HTTP ────────┴──► Ollama (host, GPU)      │
   │                          :11434                  │
   └──────────────────────────────────────────────────┘
```

Três decisões sustentam esse desenho, e as três são discutíveis — por isso estão
justificadas aqui.

**Uma imagem para os dois processos.** API e painel compartilham o mesmo pacote,
o mesmo índice e o mesmo conjunto de dependências pesadas. Duas imagens
duplicariam centenas de megabytes de wheels para trocar uma linha de comando. O
que separa os dois está no `docker-compose.yml`, não no artefato.

**O dado não entra na imagem.** O `.dockerignore` exclui `data/` inteiro — são
cerca de 85 MB entre o CSV bruto, o banco SQLite, o índice serializado e os
PDFs. Manter isso fora tem duas consequências boas: reconstruir o índice não
exige reconstruir a imagem, e o histórico operacional da planta nunca vira uma
camada distribuível.

**O Ollama fica no host.** Ele é o único componente que precisa de GPU, e expor
uma GPU a um container amarra a solução à plataforma (`nvidia-container-toolkit`,
driver do host casado com o da imagem, e nada disso existe no Docker Desktop
para Windows sem WSL2 configurado). Com o serviço no host, a **mesma imagem**
roda no notebook do avaliador e na estação industrial; o que muda é uma variável
de ambiente. É também o desenho correto do ponto de vista de recurso: o modelo é
carregado uma vez e atende a todos os processos, em vez de uma cópia por
container.

---

## 2. O que precisa existir ANTES de subir

Este é o ponto que mais causa confusão, então vem antes das instruções.

A imagem **não contém os artefatos de índice**. Ela sobe vazia e a API responde
`503` até que os arquivos abaixo existam no host:

| Caminho no host | Gerado por | Para que serve |
|---|---|---|
| `data/raw/banner.csv` | fornecido com o desafio — **não vem no repositório nem no .zip**, copie antes de tudo | entrada de `scripts/ingest.py`; sem ele o primeiro comando abaixo falha com `FileNotFoundError` |
| `data/prescritiva.db` | `scripts/ingest.py` | histórico consultado para contagem, frequência e distribuição no tempo |
| `data/processed/eventos.parquet` | `scripts/ingest.py` | base das telas do painel |
| `data/index/indice_similaridade.joblib` | `scripts/build_index.py` | índice de vizinhos por regime de rotação |
| `data/index/base_conhecimento.json` | `scripts/build_knowledge.py` | trechos e escopo dos procedimentos |
| `data/knowledge/*.json` | `scripts/build_knowledge.py` | cache do OCR (o Doc1 é escaneado e custa ~25 s por página) |
| `data/docs/*.pdf` | fornecidos com o desafio | os procedimentos em si |
| `data/index/avaliacao.json` | `scripts/evaluate.py` — **já versionado**, não precisa gerar | aba "Avaliação" do painel |

Ou seja, na primeira vez, **no host**, com o venv do projeto:

```powershell
.venv\Scripts\python.exe scripts\ingest.py
.venv\Scripts\python.exe scripts\build_index.py
.venv\Scripts\python.exe scripts\build_knowledge.py
.venv\Scripts\python.exe scripts\evaluate.py   # opcional
```

> **Armadilha real.** O compose monta o banco como **arquivo**
> (`./data/prescritiva.db:/app/data/prescritiva.db`). Se o arquivo não existir no
> host na hora do `up`, o Docker cria um **diretório** com esse nome e o SQLite
> falha com `unable to open database file` — erro que não diz nada sobre a causa.
> Rode o `ingest.py` antes do primeiro `up`. Se já caiu nisso: pare os
> containers, apague o diretório `data/prescritiva.db`, gere o banco, suba de
> novo.

Esses passos poderiam rodar dentro do container (a imagem tem tudo que eles
precisam), mas rodam no host de propósito: são o trabalho pesado da solução, o
enunciado permite explicitamente infraestrutura de alto desempenho nessa etapa, e
mantê-los fora do ciclo de vida do container deixa claro que **construir o índice
e servir o índice são coisas diferentes**.

---

## 3. Construir

```powershell
docker build -t prescritiva:0.1.0 .
```

O `Dockerfile` é multi-stage por duas razões concretas.

**Tamanho.** O estágio de construção instala `build-essential` (rede de segurança
para o caso de alguma dependência não ter wheel pronto para a arquitetura alvo).
Compilador, cabeçalhos e cache do pip ficam para trás; a imagem final recebe
apenas o virtualenv pronto e a aplicação. Isso elimina o peso do *toolchain*, mas
não o das dependências: a imagem ainda tem cerca de 2,1 GB, e a §7 mostra onde
esse peso está e o que daria para cortar.

**Cache.** `requirements.txt` é copiado **sozinho, antes do código**. PyMuPDF,
`onnxruntime` e o `opencv-python` que entra como dependência do `rapidocr` são a
maior parte do download. Se o código entrasse junto, cada mudança em `src/`
invalidaria essa camada e cada build baixaria tudo de novo. Com a ordem atual,
alterar código refaz apenas as camadas finais, que são de kilobytes.

### Três detalhes que não são óbvios

**A instalação é editável — `pip install -e .` — e não `pip install .`.** Não é
preferência: `src/prescritiva/config.py` calcula
`PROJECT_ROOT = Path(__file__).resolve().parents[2]`. Se o pacote fosse copiado
para `site-packages`, esse cálculo apontaria para dentro da biblioteca padrão do
Python e nem `config/settings.yaml` nem `data/` seriam encontrados. Editável, o
pacote continua morando em `/app/src` e a raiz resolvida é `/app`, que é o que os
volumes esperam. O `setuptools` é instalado explicitamente porque desde o Python
3.12 o `venv` não o traz mais e o build editável depende dele.

**Bibliotecas de sistema para o caminho de OCR.** A imagem final instala
`libgomp1` (exigida pelo `onnxruntime`), `libgl1` e `libglib2.0-0` (exigidas pelo
`opencv-python`). Nenhuma é usada no caminho comum — elas só importam quando um
PDF sem camada de texto é enviado ao endpoint de cadastro e o OCR roda. Estão
instaladas mesmo assim porque a ausência delas não aparece no build nem na subida
do container: aparece no meio de um upload do usuário, que é o pior lugar
possível para descobrir.

**O diretório de código não é gravável pela aplicação.** Só `/app/data` pertence
ao usuário `prescritiva`; `/app` continua sendo do root. Não é descuido: o
processo não tem por que reescrever o próprio código, e a única coisa que ele
precisa gravar — documento novo, cache de OCR, índice reconstruído — está sob
`data/`. O efeito colateral visível é que rodar `pytest` dentro do container
emite um aviso por não conseguir criar `/app/.pytest_cache`; os testes passam
assim mesmo.

**Um worker só.** O `CMD` fixa `--workers 1`. O índice de similaridade e a base
de conhecimento ficam em memória, e o endpoint `POST /documentos` reindexa esse
objeto em processo. Com vários workers, um cadastro atualizaria apenas o worker
que atendeu a requisição e os demais continuariam respondendo com a base antiga —
um bug intermitente e difícil de reproduzir. Escalar horizontalmente exige antes
tirar o índice do processo (ver §7).

---

## 4. Subir

```powershell
docker compose up -d --build
docker compose ps
```

| Serviço | URL | O que é |
|---|---|---|
| `api` | http://localhost:8000/docs | OpenAPI interativo do FastAPI |
| `api` | http://localhost:8000/saude | estado do motor (usado pelo HEALTHCHECK) |
| `painel` | http://localhost:8501 | painel Streamlit, 4 abas |

Para acompanhar e derrubar:

```powershell
docker compose logs -f api
docker compose down
```

### O HEALTHCHECK diz mais do que "está de pé"

O `HEALTHCHECK` do `Dockerfile` bate em `GET /saude` usando o próprio Python
(sem `curl`, para não adicionar pacote à imagem). Esse endpoint depende de
`MotorDiagnostico.carregar()`, ou seja: **ele só responde 200 se o índice de
similaridade e a base documental estiverem realmente carregados**. Um container
sem artefatos montados aparece como `unhealthy`, e isso é o comportamento
desejado — um container que não consegue diagnosticar não deve receber tráfego.

Se `docker compose ps` mostrar `unhealthy`, a causa quase sempre é a §2. Confirme
com:

```powershell
docker compose exec api python -c "from prescritiva.config import load_settings; p=load_settings().paths; print(p.index_dir, list(p.index_dir.iterdir()))"
```

O `start-period` é de 60 s porque a primeira chamada lê cerca de 10 MB de joblib
e ainda sonda o Ollama antes de responder.

### Terceiro processo: o consumidor da fila

A integração com o chão de fábrica (`scripts/consumidor_mqtt.py`) é um terceiro
processo, e não um modo da API: ela consome eventos de um tópico MQTT em vez de
esperar requisições HTTP. A imagem já carrega tudo que ele precisa — `paho-mqtt`
está no `requirements.txt` e o script vem copiado —, então subir esse serviço é
só uma variação de comando sobre a **mesma imagem**:

```yaml
  consumidor:
    <<: *comum
    container_name: prescritiva-consumidor
    command: ["python", "scripts/consumidor_mqtt.py"]
    environment:
      PRESCRITIVA_LLM_BASE_URL: http://host.docker.internal:11434
      PRESCRITIVA_MQTT_HOST: host.docker.internal   # ou o endereço do broker da planta
```

O endereço do broker segue a mesma regra da URL do Ollama e pelo mesmo motivo:
é da instalação, não da solução, e por isso é variável de ambiente
(`PRESCRITIVA_MQTT_HOST`) e não linha editada no `settings.yaml`.

O serviço **não** está declarado no `docker-compose.yml` porque exige um broker,
que esta entrega não sobe. Antes de ativá-lo, leia a observação sobre escrita
concorrente no SQLite em §6.

---

## 5. Apontar para o Ollama

O `settings.yaml` traz `base_url: http://localhost:11434`, que é o certo para
execução direta no host. Dentro do container, `localhost` é o **próprio
container**: o Ollama do host ficaria inalcançável, `OllamaGerador.disponivel()`
retornaria `False` e o motor cairia — em silêncio, por design — no gerador
determinístico. A solução continuaria respondendo, mas com trechos recortados em
vez de instrução reescrita, e ninguém perceberia a troca olhando só o container.

Por isso a configuração aceita sobrescrita por variável de ambiente:

| Variável | Sobrescreve | Padrão |
|---|---|---|
| `PRESCRITIVA_LLM_BASE_URL` | `llm.base_url` | `http://localhost:11434` |
| `PRESCRITIVA_LLM_MODEL` | `llm.model` | `qwen2.5:3b` |
| `PRESCRITIVA_MQTT_HOST` | `mqtt.host` | `localhost` |

A regra que decide o que entra nessa tabela: **endereço de serviço externo é do
ambiente, o resto é da solução.** Limiar de rejeição, tamanho de trecho e
temperatura descrevem como a solução funciona e continuam no `settings.yaml`,
versionados e revisáveis; onde o Ollama e o broker atendem muda por máquina e não
pode obrigar a editar arquivo versionado nem a reconstruir imagem. A lista vive
em `_SOBRESCRITAS_ENV`, em `src/prescritiva/config.py`.

Uma quarta variável, `PRESCRITIVA_CADASTRO_KEY` (sobrescreve
`cadastro.chave_acesso`, padrão nenhum = cadastro aberto), entra pelo mesmo
mecanismo por um motivo diferente: não é endereço de serviço, é segredo da
instalação — um `settings.yaml` versionado com a chave real a tornaria pública
no primeiro commit. Detalhe em [`ARQUITETURA.md`](ARQUITETURA.md) §3.10.

O `docker-compose.yml` já define a primeira como
`http://host.docker.internal:11434`. Esse nome é resolvido automaticamente pelo
Docker Desktop; a linha `extra_hosts: host.docker.internal:host-gateway` faz o
mesmo arquivo funcionar em Docker Engine sobre Linux, que é o caso provável da
estação industrial.

### Pré-requisitos no host

```powershell
ollama pull qwen2.5:3b
ollama serve
```

No Linux, o Ollama escuta apenas em `127.0.0.1` por padrão e o container **não**
o alcança. É preciso publicar na interface do bridge:

```bash
sudo systemctl edit ollama    # Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
```

Verificando de dentro do container:

```powershell
docker compose exec api python -c "import requests; print(requests.get('http://host.docker.internal:11434/api/tags', timeout=3).json())"
```

E confirmando qual gerador o motor de fato escolheu — que é o que importa:

```powershell
curl http://localhost:8000/saude
```

Se `gerador` vier como `deterministico`, o Ollama não foi alcançado. **A solução
não quebra nesse caso**, e essa é uma decisão de projeto, não um acidente: o
gerador determinístico recorta os trechos do procedimento cadastrado e entrega
instrução verificável. O que se perde é a reescrita, não a correção.

---

## 6. Na estação industrial (32 GB RAM / GPU 16 GB)

O que muda em relação à máquina de desenvolvimento:

**O modelo pode crescer, e é onde a GPU deve ser gasta.** O `qwen2.5:3b` foi
escolhido para caber em máquina sem GPU. Com 16 GB dedicados cabe um modelo
bem maior quantizado (`qwen2.5:14b-instruct-q4_K_M` ocupa aproximadamente 9 GB de
VRAM), o que melhora a qualidade da reescrita sem tocar em uma linha de código:

```yaml
environment:
  PRESCRITIVA_LLM_MODEL: qwen2.5:14b-instruct-q4_K_M
```

Vale medir antes de adotar. A tarefa aqui é reescrever um procedimento já
recuperado, com temperatura 0,1 — é uma tarefa de reescrita, não de raciocínio, e
o ganho de um modelo maior pode não pagar a latência adicional para o técnico que
está esperando na frente do equipamento.

**A RAM não é o gargalo desta solução.** Medido com `docker stats` logo após um
diagnóstico completo: **254 MiB na API** (com o índice de similaridade e os 170
trechos carregados) e **52 MiB no painel**. Cerca de 306 MB para a stack inteira.
O índice serializado tem ~10 MB e o `NearestNeighbors` do scikit-learn fica
inteiro em memória, mas 144 mil eventos com o conjunto de features usado são
poucas dezenas de MB. Ou seja: dos 32 GB da estação, praticamente toda a folga
deve ser reservada ao Ollama, não aos containers. Se um dia a RAM apertar, o
suspeito é o modelo, não a aplicação.

Medido em `docker compose up -d --build` no Docker Desktop (WSL2) de
desenvolvimento — o valor absoluto pode variar alguns MB numa engine Docker
diferente (Linux nativo na estação industrial, por exemplo), mas a ordem de
grandeza e a conclusão (a aplicação é irrelevante perto do modelo) não mudam.

**A rede muda de forma.** Em desenvolvimento as portas são publicadas em
`localhost`. Numa planta, o correto é não expor `8000` e `8501` diretamente:
colocar um proxy reverso à frente (nginx ou Traefik) terminando TLS e cuidando
da autenticação, que **esta entrega não tem** — a API é aberta e o painel também.
Isso é limitação declarada, não esquecimento: o desafio pede a solução técnica, e
autenticação sem um diretório corporativo definido seria invenção.

**O SQLite tem prazo de validade, e o consumidor MQTT antecipa esse prazo.**
Ele foi a escolha certa para o volume atual — o banco tem 44 MB e as consultas de
estatística são agregações simples e somente-leitura. O consumidor da fila muda
essa conta: ele **grava** cada diagnóstico. Um escritor só, sobre bind mount,
funciona; vários containers escrevendo no mesmo arquivo montado, não — o
travamento de arquivo do SQLite é conhecido por ser pouco confiável sobre
sistemas de arquivos de rede e sobre o compartilhamento do Docker Desktop no
Windows. Enquanto houver um único consumidor, está de pé. Se a planta passar a
ingerir continuamente ou com mais de um consumidor, o caminho é PostgreSQL (com
TimescaleDB, dado que a série é temporal); os módulos que falam com o banco são
poucos e curtos.

**Reinício e atualização.** `restart: unless-stopped` já cobre queda de container
e reboot da estação com o Docker configurado para iniciar com o sistema. Para
atualizar a solução: reconstruir a imagem, `docker compose up -d` — os volumes
sobrevivem, e os artefatos de índice não são reconstruídos, o que é o
comportamento desejado. Reconstruir o índice é uma operação separada e explícita.

---

## 7. Limitações conhecidas desta configuração

Estão aqui para serem discutidas, não escondidas.

**O painel não consome a API.** O `app/streamlit_app.py` importa
`MotorDiagnostico` diretamente e carrega sua própria cópia do índice. Os dois
containers, portanto, mantêm estado independente: se um documento novo for
cadastrado pela API, o painel só enxerga depois de reiniciar (e vice-versa). Os
volumes garantem que o **arquivo** seja o mesmo; a **memória** não é. O conserto
é transformar o painel em cliente HTTP da API — mudança de escopo maior que
"empacotar", e por isso não foi feita aqui.

**Sem autenticação e sem TLS.** Ver §6.

**Um worker.** Ver §3. O limite prático é a fila de requisições, não a CPU: cada
diagnóstico com LLM leva segundos porque espera o Ollama.

**Build validado apenas para `linux/amd64`.** As dependências fixadas têm wheel
pronto para essa plataforma. Em `arm64` o `build-essential` do primeiro estágio
cobre o que precisar compilar, mas isso não foi testado.

**Bind mounts, não volumes nomeados.** É o certo para desenvolvimento e para
demonstração — os artefatos ficam visíveis e editáveis no host. Em produção,
`data/index` e `data/processed` deveriam ser volumes nomeados ou montados
somente-leitura, já que a aplicação não precisa escrever neles.

**A imagem tem ~2,1 GB, e isso é muito.** O multi-stage fez o trabalho dele — não
há compilador nem cache de pip na imagem final, verificado com `which gcc` lá
dentro —, mas o peso está nas próprias dependências. Medido no
`site-packages` (1,2 GB no total):

| Pacote | MB | Quem precisa |
|---|---|---|
| plotly | 187 | apenas o painel |
| opencv (`cv2` + `.libs`) | 170 | apenas o OCR do cadastro |
| pyarrow | 131 | leitura do parquet |
| scipy | 111 | scikit-learn |
| pandas | 75 | tudo |
| onnxruntime | 58 | apenas o OCR do cadastro |

Existem dois cortes concretos, e nenhum dos dois foi aplicado aqui:

1. **Separar os requisitos por processo.** `plotly` e `streamlit` só servem ao
   painel; a API carrega 187 MB de JavaScript de gráfico que nunca usa. Isso
   contradiz em parte a decisão de imagem única da §1 — a troca é entre um
   artefato simples de operar e dois artefatos menores. Para o porte desta
   solução, operar um só ainda parece o certo, mas é uma decisão de conveniência,
   não de engenharia, e muda se a imagem precisar trafegar por uma rede industrial
   lenta.
2. **Trocar `opencv-python` por `opencv-python-headless`.** Economiza cerca de
   90 MB e dispensa `libgl1` na imagem. Não foi feito porque o `opencv-python`
   entra como dependência transitiva do `rapidocr-onnxruntime`, e substituí-la
   exigiria desinstalar e reinstalar por cima dentro do build — um truque comum,
   mas que deixa o `requirements.txt` mentindo sobre o que está instalado.
