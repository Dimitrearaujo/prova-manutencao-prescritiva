"""Interface da solucao de manutencao prescritiva.

Quatro telas, na ordem em que a apresentacao precisa delas: o que os dados
dizem, o diagnostico de um evento novo, a consulta livre a base documental e o
que a avaliacao mediu.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.graph_objects import Figure
from pydantic import ValidationError

from prescritiva.config import load_catalog, load_settings
from prescritiva.diagnosis.contexto import ATIPICO, INEDITO, formatar_regime
from prescritiva.diagnosis.engine import MotorDiagnostico
from prescritiva.knowledge.extract import extract_all
from prescritiva.knowledge.store import BaseConhecimento

st.set_page_config(page_title="Manutencao Prescritiva", page_icon="⚙", layout="wide")

CORES_SITUACAO = {
    "defeito_documentado": "#B3261E",
    "defeito_sem_documentacao": "#B26A00",
    "padrao_desconhecido": "#5B5B66",
    "estado_operacional": "#1B6E3C",
}
ROTULO_SITUACAO = {
    "defeito_documentado": "Defeito identificado e documentado",
    "defeito_sem_documentacao": "Defeito identificado, sem procedimento cadastrado",
    "padrao_desconhecido": "Padrao nao reconhecido",
    "estado_operacional": "Condicao de operacao, nao e problema",
}


def _milhar(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def _rpm_do_regime(regime: str) -> float:
    """Extrai o numero de "2000rpm" para ordenar os regimes pela rotacao."""
    digitos = "".join(caractere for caractere in regime if caractere.isdigit())
    return float(digitos) if digitos else 0.0


@st.cache_resource
def carregar_motor() -> MotorDiagnostico:
    return MotorDiagnostico.carregar()


@st.cache_data
def carregar_eventos() -> pd.DataFrame:
    caminho = load_settings().paths.processed_dir / "eventos.parquet"
    return pd.read_parquet(caminho)


@st.cache_data
def carregar_avaliacao() -> dict | None:
    caminho = load_settings().paths.index_dir / "avaliacao.json"
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))


def tela_dados(eventos: pd.DataFrame, motor: MotorDiagnostico) -> None:
    st.subheader("O que o historico contem")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Eventos", f"{len(eventos):,}".replace(",", "."))
    col2.metric("Defeitos distintos", int(eventos.loc[eventos["e_defeito"], "fault"].nunique()))
    col3.metric("Procedimentos", len(motor.base.documentos))
    col4.metric("Periodo", f"{eventos['created_at'].min():%d/%m} a {eventos['created_at'].max():%d/%m}")

    esquerda, direita = st.columns([3, 2])
    with esquerda:
        contagem = (
            eventos.groupby(["fault", "e_defeito"]).size().reset_index(name="registros")
            .sort_values("registros", ascending=True)
        )
        contagem["tipo"] = contagem["e_defeito"].map({True: "Defeito", False: "Condicao de operacao"})
        st.plotly_chart(
            px.bar(
                contagem, x="registros", y="fault", color="tipo", orientation="h",
                title="Registros por rotulo",
                color_discrete_map={"Defeito": "#B3261E", "Condicao de operacao": "#1B6E3C"},
            ),
            use_container_width=True,
        )
    with direita:
        st.plotly_chart(
            px.pie(
                eventos.groupby("regime_rpm").size().reset_index(name="n"),
                names="regime_rpm", values="n", hole=0.45, title="Regime de rotacao",
            ),
            use_container_width=True,
        )

    st.markdown(
        "**Dentro de cada gravação as leituras acontecem a segundos de distância e são quase "
        "idênticas**, o que torna um sorteio inválido para avaliar: a mesma medição cairia nos "
        "dois lados da partição. Os blocos não são disjuntos entre si — `scripts/eda.py` reporta "
        "16 dos 26 com sobreposição temporal, porque alguns defeitos foram retomados semanas "
        "depois —, e o argumento não precisa que sejam: a partição corta **dentro** de cada bloco."
    )
    linha = eventos.copy()
    linha["dia"] = linha["created_at"].dt.floor("D")
    st.plotly_chart(
        px.scatter(
            linha.groupby(["dia", "fault"]).size().reset_index(name="registros"),
            x="dia", y="fault", size="registros", color="fault",
            title="Janela de gravacao de cada rotulo", height=430,
        ).update_layout(showlegend=False),
        use_container_width=True,
    )

    st.subheader("Cobertura documental")
    catalogo = load_catalog()["faults"]
    linhas = []
    for fault, entrada in catalogo.items():
        cobertura = motor.base.cobertura(entrada["termo_chave"], entrada["termos_busca"], entrada["rotulo"])
        linhas.append(
            {
                "Defeito": entrada["rotulo"],
                "Coberto": "sim" if cobertura.coberto else "NAO",
                "Procedimento": cobertura.titulo or "-",
                "Aderencia": round(cobertura.score, 2),
                "Registros": int((eventos["fault"] == fault).sum()),
            }
        )
    tabela = pd.DataFrame(linhas)
    sem = tabela[tabela["Coberto"] == "NAO"]
    if not sem.empty:
        # O separador de milhar e trocado so no numero. Aplicar o replace na
        # frase inteira comeria a virgula do texto, porque literais adjacentes
        # viram uma string so antes de o metodo ser chamado.
        registros = f"{sem['Registros'].sum():,}".replace(",", ".")
        fatia = f"{sem['Registros'].sum() / len(eventos):.1%}".replace(".", ",")
        st.warning(
            f"{len(sem)} defeitos sem procedimento cadastrado, somando {registros} registros "
            f"({fatia} do histórico). Para eles o sistema informa a lacuna e pede o cadastro, "
            "em vez de improvisar instrução."
        )
    st.dataframe(tabela, use_container_width=True, hide_index=True)


def _data_br(carimbo: str | None) -> str:
    partes = (carimbo or "")[:10].split("-")
    return f"{partes[2]}/{partes[1]}/{partes[0]}" if len(partes) == 3 else "-"


def _grafico_regimes(por_regime: dict[str, int], regime_evento: str) -> Figure:
    """Distribuicao do padrao por rotacao, com o regime do evento destacado.

    O destaque nao pode depender so de cor: a barra do evento atual recebe
    tambem hachura, sufixo no rotulo do eixo e o valor escrito na propria barra.
    """
    total = sum(por_regime.values())
    linhas = []
    for regime, ocorrencias in sorted(por_regime.items(), key=lambda item: _rpm_do_regime(item[0])):
        atual = regime == regime_evento
        linhas.append(
            {
                "regime": formatar_regime(regime) + (" (evento atual)" if atual else ""),
                "fatia": ocorrencias / total,
                "valor": f"{_milhar(ocorrencias)} ({ocorrencias / total:.0%})",
                "origem": "Regime deste evento" if atual else "Demais regimes",
            }
        )
    quadro = pd.DataFrame(linhas)
    figura = px.bar(
        quadro,
        x="fatia", y="regime", orientation="h", text="valor",
        color="origem", pattern_shape="origem",
        color_discrete_map={"Regime deste evento": "#1F4E79", "Demais regimes": "#9AA5B1"},
        pattern_shape_map={"Regime deste evento": "/", "Demais regimes": ""},
        category_orders={"regime": list(quadro["regime"])},
        labels={"fatia": "Fatia das ocorrencias do padrao", "regime": "", "origem": ""},
        title="Contexto operacional: em que rotacao este padrao ja apareceu",
    )
    figura.update_traces(textposition="auto", cliponaxis=False)
    figura.update_layout(
        height=330,
        xaxis_tickformat=".0%",
        xaxis_range=[0, min(1.0, float(quadro["fatia"].max()) * 1.25)],
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.28, "x": 0},
        margin={"t": 60, "b": 30},
    )
    return figura


def tela_diagnostico(eventos: pd.DataFrame, motor: MotorDiagnostico) -> None:
    st.subheader("Diagnostico de um evento novo")

    modo = st.radio(
        "Origem do evento", ["Sortear do historico", "Colar JSON"], horizontal=True, label_visibility="collapsed"
    )
    if modo == "Sortear do historico":
        opcoes = ["(qualquer)"] + sorted(eventos["fault"].unique())
        alvo = st.selectbox("Filtrar por rotulo real", opcoes)
        universo = eventos if alvo == "(qualquer)" else eventos[eventos["fault"] == alvo]
        if st.button("Sortear evento", type="primary") or "evento" not in st.session_state:
            st.session_state.evento = universo.sample(1).iloc[0].to_dict()
        evento = dict(st.session_state.evento)
        rotulo_real = evento.get("fault")
        st.caption(
            f"Rotulo anotado pelo operador neste registro: **{rotulo_real}**. Ele nao e enviado "
            "ao motor, e o proprio registro fica de fora da busca de vizinhos."
        )
    else:
        texto = st.text_area("JSON do evento", height=150, placeholder='{"rpm": 1000.0, ...}')
        if not texto.strip():
            st.info("Cole o JSON de um evento no formato do banner.csv.")
            return
        try:
            evento = json.loads(texto)
        except json.JSONDecodeError as erro:
            st.error(f"JSON invalido: {erro}")
            return
        rotulo_real = evento.get("fault")

    evento.pop("fault", None)
    rpm_informado = evento.get("rpm")
    try:
        with st.spinner("Consultando historico e procedimentos..."):
            diagnostico = motor.diagnosticar(evento)
    except ValidationError as erro:
        # Colar um JSON incompleto e o erro mais provavel numa demonstracao, e
        # ele nao pode virar traceback na tela: o campo que falta e a resposta.
        faltando = ", ".join(
            str(detalhe["loc"][0]) if detalhe["loc"] else detalhe["msg"] for detalhe in erro.errors()
        )
        st.error(f"O evento nao tem os campos exigidos pelo contrato de entrada: {faltando}.")
        return

    cor = CORES_SITUACAO[diagnostico.situacao]
    st.markdown(
        f"<div style='background:{cor};color:#fff;padding:14px 18px;border-radius:8px;"
        f"font-weight:600'>{ROTULO_SITUACAO[diagnostico.situacao]}</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Padrao identificado", diagnostico.rotulo or "nao determinado",
        help="Rotulo que os vizinhos mais proximos carregam. Nao vem de classificador treinado.",
    )
    col2.metric(
        "Consenso dos vizinhos", f"{diagnostico.confianca:.0%}",
        help=(
            f"Peso somado dos {motor.indice.n_neighbors} vizinhos mais proximos que carregam esse "
            f"rotulo. Abaixo de {motor.indice.min_consensus:.0%} o evento e tratado como padrao "
            "nao reconhecido."
        ),
    )
    col3.metric(
        "Regime de rotacao", formatar_regime(diagnostico.regime_rpm),
        help="A busca e particionada por rotacao: so entram no calculo eventos do mesmo regime.",
    )
    if rpm_informado is not None and abs(float(rpm_informado) - _rpm_do_regime(diagnostico.regime_rpm)) >= 1:
        st.caption(
            f"O evento chegou com {float(rpm_informado):.0f} rpm. O regime com historico mais "
            f"proximo e {formatar_regime(diagnostico.regime_rpm)}, e a comparacao foi feita contra ele."
        )

    if rotulo_real:
        acertou = diagnostico.fault == rotulo_real
        st.caption(("Confere com o rotulo do operador." if acertou else
                    f"Diverge do rotulo do operador ({rotulo_real})."))

    st.info(diagnostico.mensagem)
    if diagnostico.situacao == "defeito_sem_documentacao":
        st.caption(
            "O cadastro e feito na aba **Procedimentos**, em *Cadastrar novo documento "
            "orientativo*. Depois de indexado, o mesmo evento passa a receber instrucao."
        )

    historico = diagnostico.historico
    if historico.get("ocorrencias"):
        st.markdown(f"#### Historico de {diagnostico.rotulo}")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Ocorrencias registradas", _milhar(historico["ocorrencias"]),
            help="Cada ocorrencia e uma leitura de vibracao gravada, nao uma parada de maquina.",
        )
        c2.metric(
            "Media por dia com registro", f"{historico['ocorrencias_por_dia']:.0f}",
            help="Ocorrencias divididas pelos dias em que houve registro, nao pelo periodo inteiro.",
        )
        c3.metric(
            "Dias com registro", historico["dias_cobertos"],
            help=f"De {_data_br(historico['primeira'])} a {_data_br(historico['ultima'])}.",
        )

        esquerda, direita = st.columns([3, 2])
        with esquerda:
            serie = pd.DataFrame(
                {"dia": list(historico["por_dia"].keys()), "registros": list(historico["por_dia"].values())}
            )
            st.plotly_chart(
                px.bar(
                    serie, x="dia", y="registros",
                    labels={"dia": "Dia", "registros": "Registros"},
                    title="Distribuicao ao longo do tempo",
                ).update_layout(height=330, margin={"t": 60, "b": 30}),
                use_container_width=True,
            )
        with direita:
            st.plotly_chart(
                _grafico_regimes(historico["por_regime"], diagnostico.regime_rpm),
                use_container_width=True,
            )

        # Caixa neutra quando o regime confere: verde, abaixo da tarja vermelha
        # de defeito, seria lido como "esta tudo bem com a maquina". O que muda
        # de cor e so o que exige conferencia, e a frase diz qual e o caso.
        contexto = diagnostico.contexto_operacional
        if contexto.get("alinhamento") in {ATIPICO, INEDITO}:
            st.warning(contexto["leitura"])
        elif contexto.get("leitura"):
            st.info(contexto["leitura"])

        campanhas = historico.get("por_campanha") or {}
        if len(campanhas) > 1:
            detalhe = ", ".join(
                f"campanha {campanha} com {_milhar(registros)} registros"
                for campanha, registros in campanhas.items()
            )
            st.caption(
                f"Gravado em {len(campanhas)} campanhas de coleta: {detalhe}. Cada campanha e um "
                "bloco continuo do ensaio, o que explica os blocos do grafico de tempo."
            )

    if diagnostico.instrucoes:
        st.markdown("#### Instrucoes de correcao")
        st.caption(
            f"Gerado por `{diagnostico.gerador}` a partir de "
            f"{diagnostico.cobertura.get('documento')}, usando apenas os trechos listados abaixo."
        )
        st.markdown(diagnostico.instrucoes)

    if diagnostico.trechos:
        with st.expander(f"Trechos do procedimento usados ({len(diagnostico.trechos)})"):
            st.caption("Aderencia = score BM25 do trecho para a consulta do defeito; maior e melhor.")
            for trecho in diagnostico.trechos:
                st.markdown(f"**{trecho['documento']} / {trecho['secao']}** — aderencia {trecho['score']:.1f}")
                st.text(trecho["texto"][:900])

    with st.expander("Como a similaridade decidiu"):
        similaridade = diagnostico.similaridade
        # Rotacao fora da grade nao chega a consultar o indice: nao ha vizinho,
        # nao ha distribuicao. A tela precisa explicar isso em vez de quebrar,
        # porque e o desfecho que a pergunta "e se a maquina rodar a 1500 rpm?"
        # produz - a mais previsivel de todas numa apresentacao.
        if not similaridade.get("distribuicao"):
            st.info(
                similaridade.get("motivo")
                or "Nenhum vizinho foi consultado, entao nao ha distribuicao a mostrar."
            )
            return
        distribuicao = pd.DataFrame(
            {"rotulo": list(similaridade["distribuicao"].keys()),
             "peso": list(similaridade["distribuicao"].values())}
        ).sort_values("peso")
        st.plotly_chart(
            px.bar(
                distribuicao, x="peso", y="rotulo", orientation="h",
                labels={"peso": "Peso no consenso", "rotulo": ""},
                title=f"Peso dos {motor.indice.n_neighbors} vizinhos por rotulo",
            ).update_layout(xaxis_tickformat=".0%"),
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        c1.metric(
            "Distancia media aos vizinhos", f"{similaridade['distancia_media']:.2f}",
            help="Distancia euclidiana em escala padronizada, nao em unidade fisica.",
        )
        c2.metric(
            "Limiar de rejeicao do regime", f"{similaridade['limiar_rejeicao']:.2f}",
            help="Percentil da distancia observada no proprio historico deste regime.",
        )
        if similaridade["reconhecido"]:
            st.caption(
                f"Distancia media abaixo do limiar e consenso de {diagnostico.confianca:.0%} acima "
                f"do minimo de {motor.indice.min_consensus:.0%}: o padrao foi aceito."
            )
        else:
            st.caption(f"Padrao recusado. {similaridade['motivo']}")
        vizinhos = pd.DataFrame(similaridade["vizinhos"]).rename(
            columns={
                "id": "Evento", "fault": "Rotulo", "fault_original": "Rotulo bruto",
                "created_at": "Registrado em", "rpm": "rpm", "distancia": "Distancia",
            }
        )
        st.caption(
            f"Os {len(vizinhos)} vizinhos mais proximos, dos {motor.indice.n_neighbors} que "
            "entraram no calculo do consenso."
        )
        st.dataframe(vizinhos, use_container_width=True, hide_index=True)


def tela_chat(motor: MotorDiagnostico) -> None:
    st.subheader("Consulta a base de procedimentos")

    with st.expander("Cadastrar novo documento orientativo"):
        enviado = st.file_uploader("PDF do procedimento", type="pdf")
        if enviado is not None and st.button("Indexar documento", type="primary"):
            settings = load_settings()
            destino: Path = settings.paths.docs_dir / enviado.name
            destino.write_bytes(enviado.getvalue())
            cfg = settings.knowledge
            with st.spinner("Extraindo e reindexando (documento escaneado exige OCR)..."):
                extraidos = extract_all(
                    settings.paths.docs_dir, settings.paths.knowledge_dir,
                    dpi=cfg["ocr_dpi"], min_confidence=cfg["ocr_min_confidence"],
                )
                base = BaseConhecimento(
                    tamanho_trecho=cfg["chunk_chars"], sobreposicao=cfg["chunk_overlap"],
                    chars_escopo=cfg["scope_chars"], score_minimo_cobertura=cfg["coverage_min_score"],
                ).indexar(extraidos)
                base.salvar(settings.paths.index_dir)
                motor.base = base
            st.success(f"{enviado.name} indexado. A base agora tem {len(base.documentos)} procedimentos.")
            st.cache_data.clear()

    documentos = ["(todos)"] + [d.nome for d in motor.base.documentos]
    escolhido = st.selectbox("Restringir a um procedimento", documentos)

    if "conversa" not in st.session_state:
        st.session_state.conversa = []
    for papel, conteudo in st.session_state.conversa:
        with st.chat_message(papel):
            st.markdown(conteudo)

    pergunta = st.chat_input("Ex.: como corrigir desalinhamento angular?")
    if not pergunta:
        return
    st.session_state.conversa.append(("user", pergunta))
    with st.chat_message("user"):
        st.markdown(pergunta)
    with st.chat_message("assistant"), st.spinner("Consultando procedimentos..."):
        resultado = motor.perguntar(pergunta, documento=None if escolhido == "(todos)" else escolhido)
        st.markdown(resultado["resposta"])
        if resultado["trechos"]:
            with st.expander(f"Fontes ({len(resultado['trechos'])})"):
                for trecho in resultado["trechos"]:
                    st.markdown(f"**{trecho['documento']} / {trecho['secao']}**")
                    st.text(trecho["texto"][:700])
    st.session_state.conversa.append(("assistant", resultado["resposta"]))


def tela_avaliacao() -> None:
    st.subheader("O que a avaliacao mediu")
    avaliacao = carregar_avaliacao()
    if not avaliacao or "splits" not in avaliacao:
        st.info("Execute `python scripts/evaluate.py` para gerar os resultados.")
        return

    st.markdown(
        "A partição temporal é a medida principal. A aleatória está aqui apenas para mostrar "
        "o tamanho da ilusão que produz neste conjunto de dados: dentro de cada gravação as "
        "leituras acontecem a segundos de distância e são quase idênticas, então sortear coloca "
        "a mesma medição dos dois lados da partição."
    )
    st.markdown(
        "Cada partição é medida em três níveis. **Documento certo** cobra qual procedimento foi "
        "entregue a quem vai abrir a máquina — quando o defeito real não tem procedimento "
        "cadastrado, não existe documento certo e o evento não conta como acerto aqui. "
        "**Desfecho correto** soma o acerto que não cabe no anterior: o defeito não tem "
        "procedimento e o sistema não entregou nenhum. **Prescrição indevida** é o erro que o "
        "enunciado proíbe e que nenhuma métrica de acerto mostra."
    )

    # As chaves vem do avaliacao.json, que e regravado por scripts/evaluate.py.
    # Ler com .get evita que uma metrica renomeada derrube a tela inteira: o
    # painel mostra o que existe e avisa o que falta, em vez de KeyError.
    def _pct(metricas: dict, chave: str) -> str:
        valor = metricas.get(chave)
        return f"{valor:.1%}" if isinstance(valor, (int, float)) else "-"

    tabela = pd.DataFrame(
        [
            {
                "Partição": nome,
                "Rejeição": _pct(m, "taxa_rejeicao"),
                "Acerto de rótulo": _pct(m, "acuracia_rotulo"),
                "Documento certo": _pct(m, "acuracia_documento"),
                "Desfecho correto": _pct(m, "desfecho_correto"),
                "Prescrição indevida": _pct(m, "prescricao_indevida"),
                "Eventos": m.get("eventos", "-"),
            }
            for nome, m in avaliacao["splits"].items()
        ]
    )
    st.dataframe(tabela, use_container_width=True, hide_index=True)
    if tabela.drop(columns=["Partição", "Eventos"]).eq("-").any().any():
        st.info(
            "Colunas com `-` não existem no `avaliacao.json` atual. Rode "
            "`python scripts/evaluate.py` para regravá-lo com as métricas desta versão."
        )

    st.markdown("#### Defeito inédito: a classe some do índice e é consultada")
    st.caption(
        "Um classificador fechado não tem saída aqui — devolve, com confiança, a classe "
        "conhecida menos errada. Aqui há duas saídas aceitáveis: rejeitar, ou errar o nome "
        "mas ainda acertar o procedimento."
    )
    inedito = pd.DataFrame(avaliacao["defeito_inedito"])
    if "desfecho_util" not in inedito.columns:
        st.info("Rode `python scripts/evaluate.py` novamente para gerar a métrica de desfecho útil.")
        st.dataframe(inedito, use_container_width=True, hide_index=True)
        return

    longo = inedito.assign(
        rejeitou=inedito["rejeitado_corretamente"],
        salvou=inedito["desfecho_util"] - inedito["rejeitado_corretamente"],
    ).melt(
        id_vars="defeito_removido",
        value_vars=["rejeitou", "salvou"],
        var_name="desfecho",
        value_name="fracao",
    )
    st.plotly_chart(
        px.bar(
            longo, x="fracao", y="defeito_removido", color="desfecho", orientation="h",
            title="Desfecho útil: rejeitou corretamente, ou acertou o procedimento mesmo errando o nome",
            color_discrete_map={"rejeitou": "#1B6E3C", "salvou": "#7A9CC6"},
            category_orders={
                "defeito_removido": list(
                    inedito.sort_values("desfecho_util")["defeito_removido"]
                )
            },
        ).update_layout(height=440, xaxis_tickformat=".0%"),
        use_container_width=True,
    )
    st.markdown(
        "O resultado se separa por família de componente. Removido `rolamento_combination`, "
        "o sistema cai em `rolamento_inner` — outro rolamento, mesmo Doc1, mesma ação: o nome "
        "está errado e a prescrição está certa. Já `correia` e `ventoinha` não têm par no "
        "histórico, e removidas caem em defeito de outro componente. **O pior caso é `polia`, "
        "que cai em `normal`** — chamar defeito de condição normal não gera nem ação nem alerta."
    )
    st.dataframe(inedito, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Manutencao Prescritiva")
    st.caption(
        "Diagnostico por similaridade no historico operacional, com prescricao restrita ao "
        "que os procedimentos cadastrados sustentam."
    )
    try:
        motor = carregar_motor()
        eventos = carregar_eventos()
    except FileNotFoundError:
        st.error(
            "Indices ausentes. Rode, nesta ordem: `scripts/ingest.py`, `scripts/build_index.py` "
            "e `scripts/build_knowledge.py`."
        )
        return

    st.sidebar.success(f"Gerador de texto: `{motor.gerador.nome}`")
    if motor.gerador.nome == "deterministico":
        st.sidebar.caption(
            "Ollama nao respondeu. A solucao segue funcionando: as instrucoes vem recortadas "
            "direto do procedimento, sem reescrita."
        )

    dados, diagnostico, chat, avaliacao = st.tabs(
        ["Dados", "Diagnostico", "Procedimentos", "Avaliacao"]
    )
    with dados:
        tela_dados(eventos, motor)
    with diagnostico:
        tela_diagnostico(eventos, motor)
    with chat:
        tela_chat(motor)
    with avaliacao:
        tela_avaliacao()


if __name__ == "__main__":
    main()
