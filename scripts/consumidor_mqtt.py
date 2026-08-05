"""Sobe o consumidor MQTT como servico.

Entrypoint separado do modulo de proposito: `prescritiva.integracao.mqtt` guarda
o comportamento e nao sabe nada de linha de comando, o que mantem o modulo
testavel e permite que outro processo (a API, um agendador) reuse o consumidor
sem herdar um argparse.

    python scripts/consumidor_mqtt.py --host localhost --port 1883
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from prescritiva.config import load_settings
from prescritiva.diagnosis.engine import MotorDiagnostico
from prescritiva.integracao.mqtt import ConfiguracaoMQTT, ConsumidorMQTT
from prescritiva.integracao.repositorio import RepositorioDiagnosticos


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    padrao = ConfiguracaoMQTT.de_settings(settings.mqtt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=padrao.host)
    parser.add_argument("--port", type=int, default=padrao.port)
    parser.add_argument("--topico-entrada", default=padrao.topico_entrada)
    parser.add_argument("--topico-saida", default=padrao.topico_saida)
    parser.add_argument("--topico-erros", default=padrao.topico_erros)
    parser.add_argument("--qos", type=int, choices=(0, 1, 2), default=padrao.qos)
    parser.add_argument(
        "--sem-persistencia",
        action="store_true",
        help="nao grava os diagnosticos no SQLite (util para testar a fila isolada)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )

    settings = load_settings()
    config = replace(
        ConfiguracaoMQTT.de_settings(settings.mqtt),
        host=args.host,
        port=args.port,
        topico_entrada=args.topico_entrada,
        topico_saida=args.topico_saida,
        topico_erros=args.topico_erros,
        qos=args.qos,
    )

    # O motor e carregado antes de conectar de proposito: se os indices faltam,
    # o servico falha agora, com mensagem clara, em vez de aceitar eventos da
    # planta e errar um por um depois de ja estar assinado no topico.
    motor = MotorDiagnostico.carregar()
    repositorio = None if args.sem_persistencia else RepositorioDiagnosticos(settings.paths.database)

    consumidor = ConsumidorMQTT(motor=motor, repositorio=repositorio, config=config)
    print(f"gerador de texto em uso: {motor.gerador.nome}")
    try:
        consumidor.executar()
    except KeyboardInterrupt:
        consumidor.cliente.disconnect()
        print("\nconsumidor encerrado.")


if __name__ == "__main__":
    main()
