"""Testes de ponta a ponta do motor, contra os artefatos construidos.

Cada teste cobre um dos quatro desfechos possiveis do diagnostico.

Nenhum deles condiciona assercao ao resultado do kNN. A versao anterior pulava o
teste do desfecho principal quando o evento sorteado caia na regra de rejeicao, e
guardava as assercoes dos testes de recusa dentro de um `if` sobre o rotulo
predito: com isso o unico teste do produto - reconhecer o defeito, achar o
procedimento e gerar a instrucao - passava mesmo quebrado, e continuaria verde se
o Doc2 deixasse de cobrir desalinhamento.

O padrao adotado aqui e procurar deterministicamente, dentro de uma amostra fixa,
um evento que chegue ao desfecho, e entao cobrar o desfecho inteiro sem desvio.
Nenhuma amostra chegar ao desfecho e falha, nao motivo de skip. Isso mantem o
teste preso ao CONTRATO e nao a um valor: os pesos do kNN e os limiares podem
mudar sem que o teste precise ser reescrito, mas se a prescricao parar de sair, o
teste quebra.
"""

from __future__ import annotations

import pandas as pd
import pytest

from prescritiva.config import load_settings
from prescritiva.diagnosis.engine import Diagnostico, MotorDiagnostico

# Amostra por defeito. Um evento em cada tres cai na regra de rejeicao, e entre os
# reconhecidos nem todos recebem o rotulo certo; 25 tentativas deixam o fracasso
# de todas elas fora do alcance do azar de sorteio e mantem a suite em minutos.
TENTATIVAS = 25


@pytest.fixture(scope="module")
def motor() -> MotorDiagnostico:
    settings = load_settings()
    if not (settings.paths.index_dir / "indice_similaridade.joblib").exists():
        pytest.skip("execute scripts/build_index.py e build_knowledge.py antes")
    return MotorDiagnostico.carregar()


@pytest.fixture(scope="module")
def eventos() -> pd.DataFrame:
    caminho = load_settings().paths.processed_dir / "eventos.parquet"
    if not caminho.exists():
        pytest.skip("execute scripts/ingest.py antes")
    return pd.read_parquet(caminho)


def _amostras(eventos: pd.DataFrame, fault: str, n: int = TENTATIVAS) -> list[dict]:
    disponiveis = eventos[eventos["fault"] == fault]
    amostra = disponiveis.sample(n=min(n, len(disponiveis)), random_state=7)
    linhas = []
    for posicao in range(len(amostra)):
        linha = amostra.iloc[posicao].to_dict()
        linha.pop("fault", None)
        linha.pop("fault_original", None)
        linhas.append(linha)
    return linhas


def _amostra(eventos: pd.DataFrame, fault: str) -> dict:
    return _amostras(eventos, fault, 1)[0]


def _primeiro_com_desfecho(
    motor: MotorDiagnostico, eventos: pd.DataFrame, fault: str, desfecho: str
) -> Diagnostico:
    """Primeiro evento do defeito que chega ao desfecho, ou falha o teste.

    Rejeitar parte dos eventos e o comportamento desejado, entao procurar nao e
    mascarar defeito. Nao achar nenhum em TENTATIVAS eventos e outra coisa: quer
    dizer que o caminho parou de funcionar.
    """
    for entrada in _amostras(eventos, fault):
        diagnostico = motor.diagnosticar(entrada)
        if diagnostico.situacao == desfecho:
            return diagnostico
    pytest.fail(f"nenhum dos {TENTATIVAS} eventos de {fault} chegou a {desfecho}")


def test_defeito_com_procedimento_recebe_instrucao(motor, eventos):
    """O desfecho que E o produto: reconhecer, achar o procedimento e instruir."""
    diagnostico = _primeiro_com_desfecho(motor, eventos, "desalinhado", "defeito_documentado")

    assert diagnostico.situacao == "defeito_documentado"
    assert diagnostico.fault == "desalinhado"
    assert diagnostico.e_defeito is True
    assert diagnostico.cobertura["coberto"] is True
    assert diagnostico.cobertura["documento"] == "Doc2"
    assert diagnostico.instrucoes
    assert diagnostico.trechos
    # O contexto entregue ao modelo fica restrito ao documento que cobre o
    # defeito. Trecho de outro procedimento aqui e a porta da alucinacao.
    assert {t["documento"] for t in diagnostico.trechos} == {"Doc2"}


@pytest.mark.parametrize("fault", ["eccentric_rotor", "ventoinha"])
def test_defeito_sem_procedimento_pede_cadastro(motor, eventos, fault):
    """Os dois defeitos que o enunciado quer ver recusados.

    O sistema pode nao reconhecer o padrao, e tudo bem. O que nao pode acontecer
    e reconhecer o defeito e mesmo assim entregar instrucao de correcao, porque
    nao existe procedimento cadastrado para nenhum dos dois.
    """
    entrada = motor.catalogo[fault]
    # Fixa a premissa do teste sem depender do kNN: o defeito realmente nao tem
    # documento. Se alguem cadastrar um PDF que passe a cobri-lo, este teste
    # precisa cair aqui, com essa mensagem, e nao mais adiante por outro motivo.
    cobertura = motor.base.cobertura(
        entrada["termo_chave"], entrada["termos_busca"], entrada["rotulo"]
    )
    assert not cobertura.coberto

    diagnosticos = [motor.diagnosticar(e) for e in _amostras(eventos, fault)]

    # A garantia real vale para qualquer rotulo predito: instrucao so existe
    # quando ha documento cobrindo o defeito que o sistema apontou. Note que ela
    # nao e "este evento nunca recebe instrucao" - um evento de rotor excentrico
    # confundido com correia recebe o Doc4 e prescreve, o que este teste nao
    # proibe e a metrica `prescricao_indevida` de scripts/evaluate.py mede.
    for diagnostico in diagnosticos:
        assert not diagnostico.instrucoes or diagnostico.cobertura["coberto"]
        assert not diagnostico.trechos or diagnostico.cobertura["coberto"]

    proprios = [d for d in diagnosticos if d.fault == fault]
    assert proprios, f"nenhum dos {TENTATIVAS} eventos de {fault} foi reconhecido como {fault}"
    for diagnostico in proprios:
        assert diagnostico.situacao == "defeito_sem_documentacao"
        assert diagnostico.cobertura["coberto"] is False
        assert diagnostico.cobertura["documento"] is None
        assert not diagnostico.instrucoes
        assert not diagnostico.trechos
        assert "cadastre" in diagnostico.mensagem.lower()


def test_motor_desligado_nao_recebe_acao_corretiva(motor, eventos):
    diagnostico = _primeiro_com_desfecho(
        motor, eventos, "motor_desligado", "estado_operacional"
    )
    assert diagnostico.situacao == "estado_operacional"
    assert diagnostico.fault == "motor_desligado"
    assert diagnostico.e_defeito is False
    assert not diagnostico.instrucoes
    assert not diagnostico.trechos


def test_evento_absurdo_e_recusado(motor):
    """Valores muito fora de qualquer faixa fisica plausivel.

    Um classificador fechado responderia com a classe menos errada. Aqui a
    resposta correta e admitir que nao ha nada parecido no historico.
    """
    absurdo = {
        "rpm": 1000.0,
        "z_rms_velocity_mm_s": 900.0,
        "x_rms_velocity_mm_s": 850.0,
        "temperature_c": 240.0,
        "z_peak_acceleration_g": 400.0,
        "x_peak_acceleration_g": 380.0,
        "z_peak_vel_comp_freq_hz": 61.0,
        "x_peak_vel_comp_freq_hz": 61.0,
        "z_rms_acceleration_g": 300.0,
        "x_rms_acceleration_g": 290.0,
        "z_kurtosis": 90.0,
        "x_kurtosis": 88.0,
        "z_crest_factor": 70.0,
        "x_crest_factor": 68.0,
        "z_peak_velocity_mm_s": 950.0,
        "x_peak_velocity_mm_s": 940.0,
        "z_high_freq_rms_accel_g": 200.0,
        "x_high_freq_rms_accel_g": 210.0,
    }
    diagnostico = motor.diagnosticar(absurdo)
    assert diagnostico.situacao == "padrao_desconhecido"
    assert diagnostico.fault is None
    assert not diagnostico.instrucoes


def test_pergunta_sem_procedimento_nao_inventa_resposta(motor):
    """Assunto que nenhum dos procedimentos cadastrados trata.

    tests/test_motor_chat.py cobre o portao do chat em profundidade, com um
    gerador duble que denuncia a chamada ao modelo. Este aqui roda a mesma recusa
    pelo motor montado por MotorDiagnostico.carregar() - base, indice e gerador
    reais, como a API e a tela usam - para que a recusa nao dependa da montagem
    do teste.
    """
    resultado = motor.perguntar("como calibrar o sensor de pressao da caldeira a vapor?")
    assert resultado["situacao"] == "sem_defeito_nomeado"
    assert not resultado["trechos"]
    # A recusa precisa dizer o que falta e o que existe. "Reformule" sozinho
    # devolve o problema ao tecnico sem dizer para onde reformular.
    assert "nao nomeia nenhum defeito" in resultado["resposta"].lower()
    assert "ha procedimento para" in resultado["resposta"].lower()


def test_diagnostico_sempre_expoe_as_evidencias(motor, eventos):
    diagnostico = motor.diagnosticar(_amostra(eventos, "correia"))
    assert diagnostico.similaridade["vizinhos"]
    for vizinho in diagnostico.similaridade["vizinhos"]:
        assert {"id", "fault", "created_at", "distancia"} <= vizinho.keys()
