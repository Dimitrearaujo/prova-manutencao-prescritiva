"""Consumidor MQTT: o caminho por onde o chao de fabrica fala com a solucao.

A API HTTP atende quem pergunta. O sensor nao pergunta: ele publica, sem esperar
resposta e sem saber quem esta ouvindo. MQTT e o protocolo padrao dessa conversa
em planta, leve o bastante para rodar em gateway e CLP, com entrega ao menos uma
vez e reconexao automatica quando o link cai.

Este modulo assina o topico dos eventos, chama o MESMO MotorDiagnostico que a API
usa (nao existe uma segunda regra de decisao aqui) e publica o resultado.

Quatro decisoes que so aparecem quando isso roda de verdade:

- Todo desfecho e publicado, inclusive padrao_desconhecido e
  defeito_sem_documentacao. Sao respostas, nao falhas; silenciar o que o sistema
  nao sabe e justamente o que o enunciado manda evitar.
- Mensagem que o sistema nao consegue interpretar vai para um topico proprio de
  erros, nunca para o de diagnosticos. Quem integra do outro lado precisa
  distinguir "nao entendi a mensagem" de "nao reconheci o padrao": a primeira e
  problema de integracao, a segunda e informacao de manutencao.
- O diagnostico roda numa thread separada da rede, e a mensagem so e confirmada
  ao broker depois de publicada. Os dois detalhes vem de uma falha observada, e
  estao explicados em `_ao_receber` e `atender_um`.
- O consumidor conversa com uma interface minima de cliente, e nao com o paho
  direto, para que o comportamento inteiro seja exercitado em teste sem nenhum
  broker no ar.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
from dataclasses import asdict, dataclass, fields
from typing import Any, Protocol

from pydantic import ValidationError

from prescritiva.config import load_settings
from prescritiva.data.schema import VibrationEvent
from prescritiva.diagnosis.engine import Diagnostico, MotorDiagnostico
from prescritiva.integracao.repositorio import RepositorioDiagnosticos, agora

LOGGER = logging.getLogger(__name__)

ORIGEM = "mqtt"


class ClienteMQTT(Protocol):
    """A parte do cliente paho que o consumidor usa.

    Declarada explicitamente para que o teste possa substituir o cliente por um
    duble sem broker e sem rede, e para deixar visivel o quanto deste modulo
    depende da biblioteca: cinco metodos e tres callbacks.
    """

    on_connect: Any
    on_message: Any
    on_disconnect: Any

    def connect_async(self, host: str, port: int, keepalive: int) -> Any: ...

    def subscribe(self, topic: str, qos: int = 0) -> Any: ...

    def publish(self, topic: str, payload: str, qos: int = 0) -> Any: ...

    def ack(self, mid: int, qos: int) -> Any: ...

    def manual_ack_set(self, on: bool) -> Any: ...

    def reconnect_delay_set(self, min_delay: int, max_delay: int) -> Any: ...

    def loop_forever(self, retry_first_connection: bool = False) -> Any: ...

    def disconnect(self) -> Any: ...


@dataclass(frozen=True)
class Pendente:
    """Mensagem recebida e ainda nao confirmada ao broker."""

    payload: bytes
    mid: int | None
    qos: int


@dataclass(frozen=True)
class ConfiguracaoMQTT:
    host: str = "localhost"
    port: int = 1883
    keepalive_s: int = 60
    topico_entrada: str = "planta/vibracao/eventos"
    topico_saida: str = "planta/vibracao/diagnosticos"
    topico_erros: str = "planta/vibracao/diagnosticos/erros"
    qos: int = 1
    client_id: str = "prescritiva-consumidor"
    reconexao_min_s: int = 1
    reconexao_max_s: int = 60

    @classmethod
    def de_settings(cls, bruto: dict[str, Any] | None) -> "ConfiguracaoMQTT":
        """Le o bloco mqtt do settings.yaml ignorando chave que nao conhece.

        Configuracao de planta e editada por quem opera, nao por quem escreveu o
        codigo. Uma chave a mais no arquivo nao pode derrubar o consumidor.
        """
        conhecidos = {campo.name for campo in fields(cls)}
        return cls(**{k: v for k, v in (bruto or {}).items() if k in conhecidos})


class ConsumidorMQTT:
    def __init__(
        self,
        motor: MotorDiagnostico,
        repositorio: RepositorioDiagnosticos | None,
        config: ConfiguracaoMQTT,
        *,
        cliente: ClienteMQTT | None = None,
        fila_maxima: int = 100,
    ) -> None:
        self.motor = motor
        self.repositorio = repositorio
        self.config = config
        self.cliente = cliente if cliente is not None else construir_cliente_paho(config)
        self.cliente.on_connect = self._ao_conectar
        self.cliente.on_message = self._ao_receber
        self.cliente.on_disconnect = self._ao_desconectar
        self._fila: queue.Queue[Pendente] = queue.Queue(maxsize=fila_maxima)
        self._parando = threading.Event()
        self._trabalhador: threading.Thread | None = None

    def executar(self) -> None:
        """Sobe o consumidor e fica no ar ate ser interrompido.

        `connect_async` com `retry_first_connection` cobre o caso em que o broker
        ainda nao subiu quando o servico sobe. Em planta a ordem de boot nao e
        garantida, e um consumidor que morre porque o broker atrasou vira chamado
        de suporte. Quedas depois da primeira conexao sao tratadas pelo proprio
        laco do paho, com o recuo configurado abaixo.
        """
        # Confirmacao manual: o paho confirmaria a mensagem assim que o callback
        # retorna, ou seja, antes de o diagnostico existir. Se a estacao caisse
        # no meio do processamento, o evento ja teria sido dado como entregue e
        # sumiria. Confirmando so depois de publicar, uma queda faz o broker
        # reentregar, que e o comportamento que o qos 1 promete.
        self.cliente.manual_ack_set(True)
        self.cliente.reconnect_delay_set(
            min_delay=self.config.reconexao_min_s, max_delay=self.config.reconexao_max_s
        )
        self._parando.clear()
        self._trabalhador = threading.Thread(
            target=self._trabalhar, name="prescritiva-diagnostico", daemon=True
        )
        self._trabalhador.start()

        self.cliente.connect_async(self.config.host, self.config.port, self.config.keepalive_s)
        LOGGER.info(
            "consumidor ligado a %s:%s | entrada=%s saida=%s",
            self.config.host,
            self.config.port,
            self.config.topico_entrada,
            self.config.topico_saida,
        )
        try:
            self.cliente.loop_forever(retry_first_connection=True)
        finally:
            self.encerrar()

    def encerrar(self) -> None:
        self._parando.set()
        if self._trabalhador is not None:
            self._trabalhador.join(timeout=5.0)
            self._trabalhador = None

    def atender_um(self, *, timeout: float = 0.5) -> tuple[str, str] | None:
        """Diagnostica e publica a proxima mensagem da fila, se houver.

        E o corpo do trabalhador, exposto como um passo para que o teste
        exercite exatamente o caminho que roda em producao, sem depender de
        temporizacao de thread.
        """
        try:
            pendente = self._fila.get(timeout=timeout)
        except queue.Empty:
            return None

        try:
            topico, corpo = self.processar(pendente.payload)
            self.cliente.publish(topico, corpo, qos=self.config.qos)
            # So agora a mensagem deixa de ser responsabilidade do broker.
            if pendente.mid is not None:
                self.cliente.ack(pendente.mid, pendente.qos)
            return topico, corpo
        finally:
            self._fila.task_done()

    def _trabalhar(self) -> None:
        while not self._parando.is_set():
            try:
                self.atender_um()
            except Exception:  # noqa: BLE001
                # `processar` nao levanta; sobra falha de rede na publicacao.
                # Derrubar o trabalhador deixaria o servico conectado e mudo.
                LOGGER.exception("falha ao publicar um diagnostico")

    def processar(self, payload: bytes | str) -> tuple[str, str]:
        """Transforma o payload cru na dupla (topico, corpo) a publicar.

        Isolada da rede de proposito: e aqui que mora a decisao de o que vai para
        o topico de diagnosticos e o que vai para o de erros, e e isso que o
        teste precisa exercitar. Nunca levanta excecao, porque uma falha aqui
        pararia o trabalhador e deixaria o servico conectado e mudo.
        """
        recebido_em = agora()

        try:
            dados = _ler_json(payload)
        except ValueError as erro:
            return self._erro("payload_invalido", str(erro), recebido_em, None)

        try:
            evento = VibrationEvent(**dados)
        except ValidationError as erro:
            return self._erro(
                "evento_invalido", _resumo_validacao(erro), recebido_em, dados.get("id")
            )

        try:
            diagnostico = self.motor.diagnosticar(evento)
        except Exception as erro:  # noqa: BLE001
            # Um unico evento estranho nao pode derrubar o servico e parar a
            # linha inteira. O erro vai para o topico de erros e o laco segue.
            LOGGER.exception("falha ao diagnosticar o evento %s", evento.id)
            return self._erro("falha_no_diagnostico", repr(erro), recebido_em, evento.id)

        registro_id = self._registrar(diagnostico, evento.id)
        envelope = {
            "evento_id": evento.id,
            "recebido_em": recebido_em,
            "origem": ORIGEM,
            "registro_id": registro_id,
            "persistido": registro_id is not None,
            "diagnostico": asdict(diagnostico),
        }
        LOGGER.info(
            "evento %s -> %s (%s)", evento.id, diagnostico.situacao, diagnostico.rotulo or "-"
        )
        return self.config.topico_saida, _serializar(envelope)

    def _registrar(self, diagnostico: Diagnostico, evento_id: int | None) -> int | None:
        if self.repositorio is None:
            return None
        try:
            return self.repositorio.registrar(diagnostico, evento_id=evento_id, origem=ORIGEM)
        except Exception:  # noqa: BLE001
            # Banco indisponivel nao pode reter o diagnostico: o tecnico precisa
            # da resposta agora. A falta do registro fica explicita no envelope,
            # em vez de virar um sucesso aparente.
            LOGGER.exception("falha ao gravar o diagnostico do evento %s", evento_id)
            return None

    def _erro(
        self, tipo: str, detalhe: str, recebido_em: str, evento_id: Any
    ) -> tuple[str, str]:
        LOGGER.warning("mensagem rejeitada (%s): %s", tipo, detalhe)
        corpo = {
            "evento_id": evento_id,
            "recebido_em": recebido_em,
            "origem": ORIGEM,
            "erro": tipo,
            "detalhe": detalhe,
        }
        return self.config.topico_erros, _serializar(corpo)

    def _ao_conectar(
        self, cliente: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any = None
    ) -> None:
        # A assinatura e refeita a cada conexao de proposito. Se o broker perder
        # a sessao, as assinaturas vao junto, e o consumidor ficaria no ar sem
        # receber nada: conectado, silencioso e errado.
        if getattr(reason_code, "is_failure", False):
            LOGGER.error("o broker recusou a conexao: %s", reason_code)
            return
        cliente.subscribe(self.config.topico_entrada, qos=self.config.qos)
        LOGGER.info("assinado %s (qos %s)", self.config.topico_entrada, self.config.qos)

    def _ao_receber(self, cliente: Any, userdata: Any, mensagem: Any) -> None:
        """Enfileira e retorna imediatamente.

        Diagnosticar aqui dentro parece mais simples e esta errado: este callback
        roda na thread de rede do paho, e um diagnostico com modelo de linguagem
        leva dezenas de segundos a minutos. Durante esse tempo o keepalive nao e
        respondido, o broker derruba a conexao e reentrega a mensagem que ainda
        nao foi confirmada. Foi o que aconteceu no primeiro teste com broker
        real: um evento diagnosticado duas vezes, com duas prescricoes
        diferentes gravadas para o mesmo id.
        """
        try:
            self._fila.put_nowait(
                Pendente(mensagem.payload, getattr(mensagem, "mid", None), mensagem.qos)
            )
        except queue.Full:
            # Sem confirmar: a mensagem continua com o broker e sera reentregue.
            # Perder o evento seria pior do que processa-lo atrasado.
            LOGGER.warning(
                "fila cheia (%s pendentes); a estacao nao acompanha o ritmo da planta",
                self._fila.maxsize,
            )

    def _ao_desconectar(
        self,
        cliente: Any,
        userdata: Any,
        flags: Any = None,
        reason_code: Any = None,
        properties: Any = None,
    ) -> None:
        LOGGER.warning("desconectado do broker (%s); reconexao automatica em curso", reason_code)


def construir_cliente_paho(config: ConfiguracaoMQTT) -> ClienteMQTT:
    """Cliente real, com sessao persistente.

    O import e tardio para que o modulo continue importavel, e testavel com
    duble, numa maquina sem o paho instalado.

    `clean_session=False` com um client_id fixo faz o broker guardar as
    mensagens qos 1 enquanto o consumidor esta fora do ar. E o que garante que
    um restart do servico nao perca os eventos coletados durante a janela.
    """
    import paho.mqtt.client as mqtt

    return mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=config.client_id,
        clean_session=False,
        manual_ack=True,
    )


def construir_consumidor(*, cliente: ClienteMQTT | None = None) -> ConsumidorMQTT:
    """Monta o consumidor a partir do settings.yaml e dos artefatos ja gerados."""
    settings = load_settings()
    return ConsumidorMQTT(
        motor=MotorDiagnostico.carregar(),
        repositorio=RepositorioDiagnosticos(settings.paths.database),
        config=ConfiguracaoMQTT.de_settings(settings.mqtt),
        cliente=cliente,
    )


def _ler_json(payload: bytes | str) -> dict[str, Any]:
    """Decodifica o payload exigindo um objeto JSON.

    As tres falhas possiveis (byte que nao e utf-8, texto que nao e JSON e JSON
    que nao e objeto) sao todas ValueError, entao quem chama trata uma so.
    """
    texto = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload
    dados = json.loads(texto)
    if not isinstance(dados, dict):
        raise ValueError(f"esperado um objeto JSON, veio {type(dados).__name__}")
    return dados


def _resumo_validacao(erro: ValidationError) -> str:
    """Diz qual campo faltou ou veio errado, para quem integra poder corrigir."""
    return "; ".join(
        f"{'.'.join(str(parte) for parte in falha['loc'])}: {falha['msg']}"
        for falha in erro.errors()
    )


def _sanear(valor: Any) -> Any:
    """Troca float nao finito por nulo, recursivamente.

    `distancia_media` vale infinito quando nao existe historico no regime do
    evento. json.dumps escreveria `Infinity`, que nao e JSON valido e faz o
    parser estrito do outro lado da fila recusar a mensagem inteira.
    """
    if isinstance(valor, float):
        return valor if math.isfinite(valor) else None
    if isinstance(valor, dict):
        return {chave: _sanear(item) for chave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_sanear(item) for item in valor]
    return valor


def _serializar(conteudo: dict[str, Any]) -> str:
    # allow_nan=False depois do saneamento funciona como conferencia: se algum
    # nao finito escapar, quebra aqui e nao no consumidor do outro lado.
    return json.dumps(_sanear(conteudo), ensure_ascii=False, default=str, allow_nan=False)
