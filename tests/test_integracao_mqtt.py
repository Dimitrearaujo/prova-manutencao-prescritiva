"""Testes do consumidor MQTT com duble de cliente, sem broker nenhum.

O que precisa ser provado aqui nao e que o paho funciona, e sim que o consumidor
decide certo: qual topico recebe cada resultado, o que acontece com uma mensagem
que ele nao entende, quando a mensagem e confirmada ao broker e o que sobra no
banco depois. Nada disso precisa de rede, e amarrar o teste a um broker o
tornaria lento e intermitente.

O motor tambem e um duble. Carregar o motor real exigiria os indices construidos
e o modelo de linguagem no ar; a decisao do motor ja e testada em test_engine.py.

Cada teste entrega a mensagem pelo mesmo callback que o paho chamaria e depois
roda um passo do trabalhador, que e exatamente o caminho de producao - so que
sem depender de temporizacao de thread.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from prescritiva.diagnosis.engine import Diagnostico
from prescritiva.integracao.mqtt import ConfiguracaoMQTT, ConsumidorMQTT
from prescritiva.integracao.repositorio import RepositorioDiagnosticos

CONFIG = ConfiguracaoMQTT(
    topico_entrada="planta/vibracao/eventos",
    topico_saida="planta/vibracao/diagnosticos",
    topico_erros="planta/vibracao/diagnosticos/erros",
    qos=1,
)

EVENTO_VALIDO: dict[str, Any] = {
    "id": 4242,
    "created_at": "2023-03-15T10:00:00+00:00",
    "rpm": 1000.0,
    "z_rms_velocity_mm_s": 1.8,
    "x_rms_velocity_mm_s": 1.4,
    "temperature_c": 31.2,
    "z_peak_acceleration_g": 0.9,
    "x_peak_acceleration_g": 0.7,
    "z_peak_vel_comp_freq_hz": 61.0,
    "x_peak_vel_comp_freq_hz": 61.0,
    "z_rms_acceleration_g": 0.31,
    "x_rms_acceleration_g": 0.28,
    "z_kurtosis": 3.1,
    "x_kurtosis": 3.0,
    "z_crest_factor": 4.2,
    "x_crest_factor": 4.0,
    "z_peak_velocity_mm_s": 3.4,
    "x_peak_velocity_mm_s": 3.1,
    "z_high_freq_rms_accel_g": 0.12,
    "x_high_freq_rms_accel_g": 0.11,
}


@dataclass
class MensagemFalsa:
    payload: bytes
    mid: int = 1
    qos: int = 1
    topic: str = CONFIG.topico_entrada


@dataclass
class ClienteFalso:
    """Duble do cliente paho: anota o que foi assinado, publicado e confirmado."""

    on_connect: Any = None
    on_message: Any = None
    on_disconnect: Any = None
    assinaturas: list[tuple[str, int]] = field(default_factory=list)
    publicacoes: list[tuple[str, str, int]] = field(default_factory=list)
    conexoes: list[tuple[str, int, int]] = field(default_factory=list)
    confirmadas: list[tuple[int, int]] = field(default_factory=list)
    recuo: tuple[int, int] | None = None
    no_ar: bool = False
    confirmacao_manual: bool = False

    def connect_async(self, host: str, port: int, keepalive: int) -> None:
        self.conexoes.append((host, port, keepalive))

    def subscribe(self, topic: str, qos: int = 0) -> None:
        self.assinaturas.append((topic, qos))

    def publish(self, topic: str, payload: str, qos: int = 0) -> None:
        self.publicacoes.append((topic, payload, qos))

    def ack(self, mid: int, qos: int) -> None:
        self.confirmadas.append((mid, qos))

    def manual_ack_set(self, on: bool) -> None:
        self.confirmacao_manual = on

    def reconnect_delay_set(self, min_delay: int, max_delay: int) -> None:
        self.recuo = (min_delay, max_delay)

    def loop_forever(self, retry_first_connection: bool = False) -> None:
        self.no_ar = retry_first_connection

    def disconnect(self) -> None:
        self.no_ar = False

    def entregar(self, payload: bytes | str, *, mid: int = 1) -> None:
        """Simula a chegada de uma mensagem no topico assinado."""
        corpo = payload.encode("utf-8") if isinstance(payload, str) else payload
        self.on_message(self, None, MensagemFalsa(corpo, mid=mid))

    @property
    def ultima_publicacao(self) -> tuple[str, dict[str, Any]]:
        topico, corpo, _ = self.publicacoes[-1]
        return topico, json.loads(corpo)


class MotorFalso:
    """Devolve um diagnostico combinado, sem indice nem modelo de linguagem."""

    def __init__(self, situacao: str = "defeito_documentado", *, erro: Exception | None = None):
        self.situacao = situacao
        self.erro = erro
        self.recebidos: list[Any] = []

    def diagnosticar(self, evento: Any) -> Diagnostico:
        self.recebidos.append(evento)
        if self.erro is not None:
            raise self.erro
        return _diagnostico(self.situacao)


def _diagnostico(situacao: str) -> Diagnostico:
    prescreve = situacao == "defeito_documentado"
    desconhecido = situacao == "padrao_desconhecido"
    return Diagnostico(
        situacao=situacao,
        fault=None if desconhecido else "desalinhado",
        rotulo=None if desconhecido else "Desalinhamento de eixo",
        confianca=0.0 if desconhecido else 0.87,
        regime_rpm="1000rpm",
        e_defeito=situacao.startswith("defeito"),
        mensagem=f"mensagem de {situacao}",
        similaridade={"distancia_media": float("inf") if desconhecido else 1.23, "vizinhos": []},
        cobertura={"documento": "Doc2" if prescreve else None},
        instrucoes="1. Bloquear o equipamento." if prescreve else "",
    )


@pytest.fixture
def repositorio(tmp_path) -> RepositorioDiagnosticos:
    return RepositorioDiagnosticos(tmp_path / "prescritiva.db")


def _montar(
    repositorio: RepositorioDiagnosticos | None,
    situacao: str = "defeito_documentado",
    *,
    erro: Exception | None = None,
    fila_maxima: int = 100,
) -> tuple[ConsumidorMQTT, ClienteFalso, MotorFalso]:
    cliente = ClienteFalso()
    motor = MotorFalso(situacao, erro=erro)
    consumidor = ConsumidorMQTT(
        motor, repositorio, CONFIG, cliente=cliente, fila_maxima=fila_maxima
    )
    return consumidor, cliente, motor


def _ciclo(consumidor: ConsumidorMQTT, cliente: ClienteFalso, payload: bytes | str, *, mid: int = 1):
    """Recebe pela rede e roda um passo do trabalhador, como em producao."""
    cliente.entregar(payload, mid=mid)
    return consumidor.atender_um(timeout=1.0)


def test_ao_conectar_assina_o_topico_de_entrada(repositorio):
    consumidor, cliente, _ = _montar(repositorio)
    consumidor.executar()

    assert cliente.confirmacao_manual, "confirmar antes de diagnosticar perderia o evento"
    assert cliente.conexoes == [(CONFIG.host, CONFIG.port, CONFIG.keepalive_s)]
    assert cliente.recuo == (CONFIG.reconexao_min_s, CONFIG.reconexao_max_s)
    assert cliente.no_ar, "o laco precisa tolerar o broker ainda nao estar de pe"

    cliente.on_connect(cliente, None, {}, 0, None)
    assert cliente.assinaturas == [(CONFIG.topico_entrada, CONFIG.qos)]


def test_conexao_recusada_nao_assina(repositorio):
    """Assinar sobre uma conexao recusada mascara a falha de credencial."""

    class RecusaFalsa:
        is_failure = True

    _, cliente, _ = _montar(repositorio)
    cliente.on_connect(cliente, None, {}, RecusaFalsa(), None)

    assert cliente.assinaturas == []


def test_mensagem_boa_publica_diagnostico_e_grava_no_banco(repositorio):
    consumidor, cliente, motor = _montar(repositorio)
    _ciclo(consumidor, cliente, json.dumps(EVENTO_VALIDO))

    topico, envelope = cliente.ultima_publicacao
    assert topico == CONFIG.topico_saida
    assert envelope["evento_id"] == 4242
    assert envelope["persistido"] is True
    assert envelope["diagnostico"]["situacao"] == "defeito_documentado"
    assert envelope["diagnostico"]["instrucoes"]

    registros = repositorio.listar()
    assert len(registros) == 1
    assert registros[0].evento_id == 4242
    assert registros[0].origem == "mqtt"
    assert registros[0].id == envelope["registro_id"]

    # O rotulo do operador nunca entra: o motor recebe medicao, nao resposta.
    assert motor.recebidos[0].fault is None


def test_recebimento_nao_diagnostica_na_thread_de_rede(repositorio):
    """O callback do paho so enfileira.

    Diagnosticar ali dentro bloqueia o keepalive pelo tempo do modelo de
    linguagem, o broker derruba a conexao e reentrega a mensagem: o mesmo evento
    acaba diagnosticado duas vezes, com duas prescricoes diferentes.
    """
    consumidor, cliente, motor = _montar(repositorio)
    cliente.entregar(json.dumps(EVENTO_VALIDO))

    assert motor.recebidos == [], "o diagnostico nao pode acontecer no callback"
    assert cliente.publicacoes == []

    consumidor.atender_um(timeout=1.0)
    assert len(motor.recebidos) == 1
    assert len(cliente.publicacoes) == 1


def test_confirma_ao_broker_somente_depois_de_publicar(repositorio):
    """Confirmar antes de publicar transformaria uma queda em evento perdido."""
    consumidor, cliente, _ = _montar(repositorio)

    cliente.entregar(json.dumps(EVENTO_VALIDO), mid=77)
    assert cliente.confirmadas == []

    consumidor.atender_um(timeout=1.0)
    assert cliente.confirmadas == [(77, 1)]
    assert len(cliente.publicacoes) == 1


def test_fila_cheia_nao_confirma_a_mensagem(repositorio):
    """Sem confirmacao o broker reentrega; descartar perderia o evento."""
    consumidor, cliente, motor = _montar(repositorio, fila_maxima=1)

    cliente.entregar(json.dumps(EVENTO_VALIDO), mid=1)
    cliente.entregar(json.dumps(EVENTO_VALIDO), mid=2)

    assert consumidor._fila.qsize() == 1
    assert cliente.confirmadas == []
    assert motor.recebidos == []


@pytest.mark.parametrize(
    "situacao",
    [
        "defeito_documentado",
        "defeito_sem_documentacao",
        "estado_operacional",
        "padrao_desconhecido",
    ],
)
def test_os_quatro_desfechos_sao_publicados(repositorio, situacao):
    """Recusar-se a prescrever e uma resposta, nao um erro.

    Se o consumidor descartasse padrao_desconhecido e defeito_sem_documentacao,
    a planta so ouviria a solucao quando ela ja sabe a resposta, que e o oposto
    do comportamento pedido.
    """
    consumidor, cliente, _ = _montar(repositorio, situacao)
    _ciclo(consumidor, cliente, json.dumps(EVENTO_VALIDO))

    topico, envelope = cliente.ultima_publicacao
    assert topico == CONFIG.topico_saida
    assert envelope["diagnostico"]["situacao"] == situacao
    assert repositorio.listar()[0].situacao == situacao


def test_padrao_desconhecido_serializa_sem_infinito(repositorio):
    """`Infinity` nao e JSON valido e derrubaria o parser do outro lado."""
    consumidor, cliente, _ = _montar(repositorio, "padrao_desconhecido")
    _ciclo(consumidor, cliente, json.dumps(EVENTO_VALIDO))

    _, corpo, _ = cliente.publicacoes[-1]
    assert "Infinity" not in corpo
    assert json.loads(corpo)["diagnostico"]["similaridade"]["distancia_media"] is None
    assert repositorio.listar()[0].distancia_media is None


@pytest.mark.parametrize(
    "payload",
    [
        b"isto nao e json",
        b"",
        b"[1, 2, 3]",
        b'"apenas um texto"',
        b"\xff\xfe nao e utf-8",
    ],
)
def test_payload_malformado_vai_para_o_topico_de_erros(repositorio, payload):
    consumidor, cliente, motor = _montar(repositorio)
    _ciclo(consumidor, cliente, payload)

    topico, corpo = cliente.ultima_publicacao
    assert topico == CONFIG.topico_erros
    assert corpo["erro"] == "payload_invalido"
    assert motor.recebidos == [], "nao se diagnostica o que nao foi entendido"
    assert repositorio.listar() == []


def test_mensagem_rejeitada_tambem_e_confirmada(repositorio):
    """Reentregar um payload malformado geraria um laco infinito no broker."""
    consumidor, cliente, _ = _montar(repositorio)
    _ciclo(consumidor, cliente, b"isto nao e json", mid=9)

    assert cliente.confirmadas == [(9, 1)]


def test_campo_faltando_diz_qual_campo(repositorio):
    """Quem publicou precisa saber o que corrigir, nao so que deu errado."""
    incompleto = {k: v for k, v in EVENTO_VALIDO.items() if k != "z_kurtosis"}
    consumidor, cliente, _ = _montar(repositorio)
    _ciclo(consumidor, cliente, json.dumps(incompleto))

    topico, corpo = cliente.ultima_publicacao
    assert topico == CONFIG.topico_erros
    assert corpo["erro"] == "evento_invalido"
    assert "z_kurtosis" in corpo["detalhe"]
    assert corpo["evento_id"] == 4242
    assert repositorio.listar() == []


def test_campo_com_tipo_errado_e_recusado(repositorio):
    torto = dict(EVENTO_VALIDO, rpm="mil e quinhentos")
    consumidor, cliente, _ = _montar(repositorio)
    _ciclo(consumidor, cliente, json.dumps(torto))

    topico, corpo = cliente.ultima_publicacao
    assert topico == CONFIG.topico_erros
    assert "rpm" in corpo["detalhe"]


def test_falha_do_motor_nao_derruba_o_consumidor(repositorio):
    """Um evento ruim nao pode parar a fila inteira."""
    consumidor, cliente, _ = _montar(repositorio, erro=RuntimeError("indice corrompido"))
    _ciclo(consumidor, cliente, json.dumps(EVENTO_VALIDO))

    topico, corpo = cliente.ultima_publicacao
    assert topico == CONFIG.topico_erros
    assert corpo["erro"] == "falha_no_diagnostico"
    assert "indice corrompido" in corpo["detalhe"]


def test_banco_indisponivel_nao_impede_a_publicacao():
    """O tecnico precisa da resposta agora; a falta do registro fica explicita."""

    class RepositorioQuebrado:
        def registrar(self, *args: Any, **kwargs: Any) -> int:
            raise OSError("disco cheio")

    cliente = ClienteFalso()
    consumidor = ConsumidorMQTT(MotorFalso(), RepositorioQuebrado(), CONFIG, cliente=cliente)
    _ciclo(consumidor, cliente, json.dumps(EVENTO_VALIDO))

    topico, envelope = cliente.ultima_publicacao
    assert topico == CONFIG.topico_saida
    assert envelope["persistido"] is False
    assert envelope["registro_id"] is None
    assert envelope["diagnostico"]["situacao"] == "defeito_documentado"


def test_consumidor_sem_repositorio_continua_publicando():
    consumidor, cliente, _ = _montar(None)
    _ciclo(consumidor, cliente, json.dumps(EVENTO_VALIDO))

    topico, envelope = cliente.ultima_publicacao
    assert topico == CONFIG.topico_saida
    assert envelope["persistido"] is False


def test_publica_com_o_qos_configurado(repositorio):
    """Qos 1 e o que garante que um diagnostico nao se perca numa queda de link."""
    consumidor, cliente, _ = _montar(repositorio)
    _ciclo(consumidor, cliente, json.dumps(EVENTO_VALIDO))

    assert cliente.publicacoes[-1][2] == 1


def test_fila_vazia_nao_publica_nada(repositorio):
    consumidor, cliente, _ = _montar(repositorio)

    assert consumidor.atender_um(timeout=0.01) is None
    assert cliente.publicacoes == []


def test_varias_mensagens_sao_atendidas_em_ordem(repositorio):
    consumidor, cliente, _ = _montar(repositorio)
    for mid in (1, 2, 3):
        cliente.entregar(json.dumps(dict(EVENTO_VALIDO, id=mid)), mid=mid)

    while consumidor.atender_um(timeout=0.01) is not None:
        pass

    assert [c[0] for c in cliente.confirmadas] == [1, 2, 3]
    assert [r.evento_id for r in repositorio.listar()] == [3, 2, 1], "listar traz o mais recente primeiro"


def test_configuracao_ignora_chave_desconhecida():
    config = ConfiguracaoMQTT.de_settings({"host": "broker.planta", "chave_do_futuro": 1})

    assert config.host == "broker.planta"
    assert config.port == 1883


def test_configuracao_aceita_bloco_ausente():
    assert ConfiguracaoMQTT.de_settings(None).host == "localhost"
