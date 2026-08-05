# syntax=docker/dockerfile:1.7

# Imagem unica para os dois processos da solucao (API e painel). Eles partilham
# o mesmo pacote, o mesmo indice e o mesmo conjunto de dependencias pesadas;
# manter duas imagens duplicaria ~700 MB de wheels para trocar uma linha de
# comando. O que separa API de painel esta no compose, nao no artefato.

# ---------------------------------------------------------------------------
# Estagio 1 - construcao. Carrega compilador e cabecalhos, que nao vao para a
# imagem final. Tudo e instalado num virtualenv proprio para que o estagio
# seguinte copie um diretorio so e nao precise repetir o pip.
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Rede de seguranca: as dependencias fixadas tem wheel para linux/amd64, mas se
# alguma precisar compilar numa outra arquitetura o build falha aqui, no
# estagio descartavel, e nao engorda a imagem final.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Esta e a camada cara: PyMuPDF, onnxruntime e o opencv que vem junto do
# rapidocr somam a maior parte da imagem e levam minutos para baixar. Por isso
# o requirements.txt entra sozinho, ANTES do codigo: mudanca em src/ nao pode
# invalidar o cache do pip. O setuptools e explicito porque desde o Python 3.12
# o venv nao o traz mais, e o build editavel do proximo passo depende dele.
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

WORKDIR /app

# Instalacao editavel, e nao "pip install .", por uma razao concreta: o
# config.py resolve PROJECT_ROOT subindo dois niveis a partir do proprio
# arquivo. Copiado para site-packages, esse calculo apontaria para dentro da
# biblioteca padrao e config/ e data/ ficariam invisiveis. Editavel, o pacote
# continua morando em /app/src e a raiz resolvida e /app.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-deps --no-build-isolation -e .

# ---------------------------------------------------------------------------
# Estagio 2 - execucao. Sem compilador, sem cache de pip, sem codigo-fonte de
# dependencia: apenas o venv pronto e a aplicacao.
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# libgomp1 e exigida pelo onnxruntime; libgl1 e libglib2.0-0 pelo
# opencv-python, que entra como dependencia do rapidocr. Nenhuma e usada no
# caminho comum: elas so importam quando um PDF sem camada de texto e enviado
# ao endpoint de cadastro e o OCR roda. Sao instaladas mesmo assim porque a
# falta delas nao aparece no build nem na subida do container, e sim no meio de
# um upload do usuario - o pior lugar possivel para descobrir.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin prescritiva

COPY --from=builder /opt/venv /opt/venv

# HOME explicito porque o Streamlit grava configuracao e credenciais em
# ~/.streamlit ao subir. Docker deriva HOME do /etc/passwd, mas o comportamento
# ja variou entre versoes, e um HOME apontando para diretorio nao gravavel
# derruba o painel na inicializacao - nao vale depender disso. Declarado DEPOIS
# da copia do venv: e uma camada de metadado, e posta antes invalidaria o cache
# da camada mais cara da imagem a cada ajuste.
ENV HOME=/home/prescritiva

WORKDIR /app

# Ordem por frequencia de mudanca: o que muda pouco vem primeiro para que uma
# alteracao no codigo refaca o menor numero possivel de camadas.
COPY --chown=prescritiva:prescritiva pyproject.toml ./
COPY --chown=prescritiva:prescritiva config/ ./config/
COPY --chown=prescritiva:prescritiva scripts/ ./scripts/
COPY --chown=prescritiva:prescritiva tests/ ./tests/
COPY --chown=prescritiva:prescritiva src/ ./src/
COPY --chown=prescritiva:prescritiva app/ ./app/

# Os diretorios de dados sao pontos de montagem, mas precisam existir e
# pertencer ao usuario da aplicacao antes disso: o endpoint de cadastro grava
# em docs/, knowledge/ e index/, e um container subido sem volume falharia por
# permissao em vez de falhar por dado ausente, que e o diagnostico correto.
RUN mkdir -p data/raw data/docs data/index data/processed data/knowledge \
    && chown -R prescritiva:prescritiva /app/data

USER prescritiva

EXPOSE 8000

# Sem curl nem wget: o interpretador ja esta na imagem e resolve sem adicionar
# pacote. /saude carrega o indice de similaridade e a base documental, entao
# responde 503 enquanto os artefatos nao estiverem montados. Isso e proposital:
# um container sem indice nao serve, e deve aparecer como unhealthy.
# start-period folgado porque a primeira chamada le ~10 MB de joblib e ainda
# sonda o Ollama antes de responder.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/saude', timeout=8)"]

# Um worker so, de proposito. O indice fica em memoria e o cadastro de
# documento reindexa esse objeto em processo: com varios workers o upload
# atualizaria um deles e os demais seguiriam respondendo com a base antiga.
# Escalar aqui exige antes tirar o indice do processo - esta anotado no
# docs/DEPLOY.md.
CMD ["uvicorn", "prescritiva.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
