"""Publica eventos reais do historico num broker MQTT, no lugar do gateway.

Existe para a demonstracao ao vivo: sem ele, provar a integracao exigiria um CLP
e um motor girando. O simulador ocupa o lugar do gateway e publica exatamente o
que ele publicaria, um registro de sensor por mensagem, no ritmo escolhido.

O rotulo anotado pelo operador (`fault`) e removido antes de publicar, junto com
as colunas derivadas na ingestao. Ele existe no historico porque alguem escreveu
em campo; um evento novo chega sem ele, e envia-lo junto tornaria a demonstracao
uma mentira - o consumidor pareceria acertar quando na verdade recebeu a
resposta pronta.

    python scripts/simular_chao_de_fabrica.py --modo roteiro --intervalo 5
    python scripts/simular_chao_de_fabrica.py --modo amostra --total 20 --intervalo 1
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import pandas as pd

from prescritiva.config import fault_label, load_settings
from prescritiva.integracao.mqtt import ConfiguracaoMQTT

# Anotacao do operador e colunas calculadas na ingestao. Nada disso existe num
# evento que acaba de sair do sensor.
COLUNAS_DE_ANOTACAO: frozenset[str] = frozenset(
    {"fault", "fault_original", "e_defeito", "campanha", "regime_rpm"}
)

# Um caso de cada desfecho, na mesma ordem de scripts/demo.py, para que a
# apresentacao ao vivo mostre os quatro caminhos sem depender de sorteio.
ROTEIRO: tuple[tuple[str, str], ...] = (
    ("defeito com procedimento cadastrado", "desalinhado"),
    ("defeito SEM procedimento cadastrado", "ventoinha"),
    ("condicao de operacao, nao e problema", "motor_desligado"),
)

# Colunas amplificadas para forjar o quarto desfecho. Multiplicar um evento real
# em vez de inventar numeros mantem a coerencia interna do registro (razoes
# X/Z, fator de crista) e representa o que de fato acontece em campo: um modo de
# falha que nunca esteve no historico, nao um valor digitado errado.
COLUNAS_AMPLIFICADAS: tuple[str, ...] = (
    "z_rms_velocity_mm_s",
    "x_rms_velocity_mm_s",
    "z_peak_acceleration_g",
    "x_peak_acceleration_g",
    "z_rms_acceleration_g",
    "x_rms_acceleration_g",
    "z_peak_velocity_mm_s",
    "x_peak_velocity_mm_s",
    "z_high_freq_rms_accel_g",
    "x_high_freq_rms_accel_g",
)
FATOR_AMPLIFICACAO = 25.0


def parse_args() -> argparse.Namespace:
    padrao = ConfiguracaoMQTT.de_settings(load_settings().mqtt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=padrao.host)
    parser.add_argument("--port", type=int, default=padrao.port)
    parser.add_argument("--topico", default=padrao.topico_entrada)
    parser.add_argument("--qos", type=int, choices=(0, 1, 2), default=padrao.qos)
    parser.add_argument(
        "--modo",
        choices=("roteiro", "amostra"),
        default="roteiro",
        help="roteiro: um evento de cada desfecho; amostra: eventos sorteados do historico",
    )
    parser.add_argument("--intervalo", type=float, default=5.0, help="segundos entre mensagens")
    parser.add_argument("--total", type=int, default=10, help="quantos eventos no modo amostra")
    parser.add_argument("--fault", action="append", help="restringe o sorteio a estes rotulos")
    # Semente fixa para que a apresentacao ao vivo seja reproduzivel. Ela foi
    # escolhida entre as que exibem os quatro desfechos, e nao ao acaso: parte
    # dos eventos de desalinhamento cai na regra de rejeicao, o que e o
    # comportamento calibrado do indice e nao um defeito. Trocar a semente sorteia
    # outros eventos e alguns serao rejeitados; o modo amostra mostra essa
    # proporcao sem selecao.
    parser.add_argument("--semente", type=int, default=3)
    return parser.parse_args()


def carregar_historico() -> pd.DataFrame:
    caminho = load_settings().paths.processed_dir / "eventos.parquet"
    if not caminho.exists():
        raise SystemExit(f"{caminho} nao existe. Execute scripts/ingest.py antes.")
    return pd.read_parquet(caminho)


def montar_fila(eventos: pd.DataFrame, args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    """Escolhe o que publicar e em que ordem."""
    if args.modo == "amostra":
        candidatos = eventos
        if args.fault:
            candidatos = candidatos[candidatos["fault"].isin(args.fault)]
        if candidatos.empty:
            raise SystemExit(f"nenhum evento com fault em {args.fault}")
        sorteados = candidatos.sample(
            n=min(args.total, len(candidatos)), random_state=args.semente
        )
        return [
            (f"anotado como {fault_label(linha['fault'])}", _payload(linha))
            for _, linha in sorteados.iterrows()
        ]

    fila: list[tuple[str, dict[str, Any]]] = []
    for descricao, fault in ROTEIRO:
        linha = eventos[eventos["fault"] == fault].sample(1, random_state=args.semente).iloc[0]
        fila.append((f"{descricao} (anotado: {fault})", _payload(linha)))

    base = eventos[eventos["fault"] == "normal"].sample(1, random_state=args.semente).iloc[0]
    fila.append(("modo de falha inedito (evento real amplificado)", _amplificar(_payload(base))))
    return fila


def _payload(linha: pd.Series) -> dict[str, Any]:
    return {
        coluna: _para_json(valor)
        for coluna, valor in linha.items()
        if coluna not in COLUNAS_DE_ANOTACAO
    }


def _para_json(valor: Any) -> Any:
    """Converte o que o pandas devolve para tipo que o json entende."""
    if isinstance(valor, pd.Timestamp):
        return valor.isoformat()
    # Escalares numpy (np.float64, np.int64) nao sao serializaveis; .item() devolve
    # o equivalente nativo.
    return valor.item() if hasattr(valor, "item") else valor


def _amplificar(payload: dict[str, Any]) -> dict[str, Any]:
    ampliado = dict(payload)
    ampliado.pop("id", None)  # evento inedito nao tem id no historico
    for coluna in COLUNAS_AMPLIFICADAS:
        if coluna in ampliado:
            ampliado[coluna] = round(float(ampliado[coluna]) * FATOR_AMPLIFICACAO, 4)
    return ampliado


def main() -> None:
    args = parse_args()
    import paho.mqtt.client as mqtt

    fila = montar_fila(carregar_historico(), args)

    cliente = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="prescritiva-gateway-simulado",
    )
    cliente.connect(args.host, args.port, 60)
    # loop_start em thread propria: sem ele o handshake de qos 1 nao completa e
    # wait_for_publish trava para sempre.
    cliente.loop_start()

    print(f"publicando {len(fila)} eventos em {args.host}:{args.port} -> {args.topico}\n")
    try:
        for ordem, (descricao, payload) in enumerate(fila, start=1):
            corpo = json.dumps(payload, ensure_ascii=False)
            info = cliente.publish(args.topico, corpo, qos=args.qos)
            info.wait_for_publish(timeout=10)
            print(
                f"[{ordem}/{len(fila)}] id={payload.get('id', '-'):<8} "
                f"rpm={payload['rpm']:<7} {descricao}"
            )
            if ordem < len(fila):
                time.sleep(args.intervalo)
    except KeyboardInterrupt:
        print("\ninterrompido.")
    finally:
        cliente.loop_stop()
        cliente.disconnect()


if __name__ == "__main__":
    main()
