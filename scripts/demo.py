"""Demonstracao dos quatro desfechos do diagnostico, em linha de comando.

Serve de roteiro para a apresentacao: um exemplo de cada caminho que o motor
pode tomar, incluindo os dois em que ele se recusa a prescrever.

Cada caso procura, dentro de uma amostra fixa, o primeiro evento que chega ao
desfecho que aquele caso existe para mostrar, e informa quantos foram descartados
no caminho. Sortear um evento so e escolher uma semente feliz: cerca de um em
cada tres eventos e recusado pela regra de rejeicao, e uma semente que caia num
deles faz a demo do caso "defeito com procedimento" terminar sem prescrever nada.
Exibir a contagem e mais honesto que esconde-la - a taxa de descarte e o
comportamento de conjunto aberto que o projeto defende, nao um defeito a
disfarcar.
"""

from __future__ import annotations

import pandas as pd

from prescritiva.config import load_settings
from prescritiva.diagnosis.engine import Diagnostico, MotorDiagnostico

# Quantos eventos de cada defeito o roteiro pode percorrer atras do desfecho.
# Com ~2/3 de aproveitamento, 30 tentativas tornam o fracasso de todas elas um
# sinal de regressao, nao azar de sorteio.
TENTATIVAS = 30

CASOS = [
    ("defeito com procedimento cadastrado", "desalinhado", "defeito_documentado"),
    ("defeito SEM procedimento cadastrado", "ventoinha", "defeito_sem_documentacao"),
    ("condicao de operacao, nao e problema", "motor_desligado", "estado_operacional"),
]

EVENTO_ABSURDO = {
    "rpm": 1000.0, "z_rms_velocity_mm_s": 900.0, "x_rms_velocity_mm_s": 850.0,
    "temperature_c": 240.0, "z_peak_acceleration_g": 400.0, "x_peak_acceleration_g": 380.0,
    "z_peak_vel_comp_freq_hz": 61.0, "x_peak_vel_comp_freq_hz": 61.0,
    "z_rms_acceleration_g": 300.0, "x_rms_acceleration_g": 290.0,
    "z_kurtosis": 90.0, "x_kurtosis": 88.0, "z_crest_factor": 70.0, "x_crest_factor": 68.0,
    "z_peak_velocity_mm_s": 950.0, "x_peak_velocity_mm_s": 940.0,
    "z_high_freq_rms_accel_g": 200.0, "x_high_freq_rms_accel_g": 210.0,
}


def procurar_desfecho(
    motor: MotorDiagnostico, eventos: pd.DataFrame, fault: str, desfecho: str
) -> tuple[Diagnostico | None, int, int]:
    """Primeiro evento do defeito que chega ao desfecho pedido.

    Devolve tambem quantos foram descartados antes dele e quantos foram
    examinados no total, que e o que transforma a demo numa medida em vez de uma
    ilustracao.
    """
    disponiveis = eventos[eventos["fault"] == fault]
    amostra = disponiveis.sample(n=min(TENTATIVAS, len(disponiveis)), random_state=7)
    for posicao in range(len(amostra)):
        linha = amostra.iloc[posicao].to_dict()
        linha.pop("fault", None)
        linha.pop("fault_original", None)
        diagnostico = motor.diagnosticar(linha)
        if diagnostico.situacao == desfecho:
            return diagnostico, posicao, len(amostra)
    return None, len(amostra), len(amostra)


def mostrar(titulo: str, rotulo_real: str | None, diagnostico: Diagnostico) -> None:
    print("\n" + "=" * 78)
    print(titulo.upper())
    print("=" * 78)
    if rotulo_real:
        print(f"rotulo anotado pelo operador: {rotulo_real}  (nao enviado ao motor)")
    print(f"desfecho    : {diagnostico.situacao}")
    print(f"defeito     : {diagnostico.rotulo or '-'}")
    similaridade = diagnostico.similaridade
    votos = similaridade.get("votos", {})
    # O rotulo mais pesado, e nao diagnostico.fault: num evento rejeitado o fault
    # sai None e a contagem apareceria como "0 de 25" ao lado de um percentual
    # diferente de zero. E o candidato que perdeu o portao que interessa mostrar.
    predominante = next(iter(similaridade.get("distribuicao", {})), None)
    apoios = votos.get(predominante, 0)
    consultados = similaridade.get("vizinhos_consultados", 0)
    # O peso e o voto contam coisas diferentes e a demo mostra os dois: o peso
    # diz o quanto o vizinho vencedor esta perto, o voto diz quantos concordam.
    print(
        f"consenso    : {apoios} de {consultados} vizinhos "
        f"({similaridade.get('consenso_simples', 0.0):.0%})  | "
        f"peso {diagnostico.confianca:.0%}  | regime: {diagnostico.regime_rpm}"
    )
    print(f"\n{diagnostico.mensagem}")
    if diagnostico.trechos:
        print(f"\ntrechos usados: {[t['secao'][:38] for t in diagnostico.trechos]}")
    if diagnostico.instrucoes:
        print(f"\n--- instrucoes ({diagnostico.gerador}) ---")
        print(diagnostico.instrucoes[:1400])
    vizinhos = similaridade.get("vizinhos", [])[:5]
    if vizinhos:
        print("\nevidencia (5 vizinhos mais proximos):")
        for v in vizinhos:
            print(f"  id {v['id']:<8} {v['fault_original']:<26} d={v['distancia']:.3f}  {v['created_at'][:19]}")


def main() -> None:
    settings = load_settings()
    eventos = pd.read_parquet(settings.paths.processed_dir / "eventos.parquet")
    motor = MotorDiagnostico.carregar()
    print(f"gerador de texto em uso: {motor.gerador.nome}")

    for titulo, fault, desfecho in CASOS:
        diagnostico, descartados, examinados = procurar_desfecho(motor, eventos, fault, desfecho)
        if diagnostico is None:
            print("\n" + "=" * 78)
            print(titulo.upper())
            print("=" * 78)
            print(
                f"nenhum dos {examinados} eventos de {fault} chegou a {desfecho}. "
                "Isso e regressao, nao azar de sorteio: investigue antes de apresentar."
            )
            continue
        mostrar(titulo, fault, diagnostico)
        print(
            f"\nselecao: {descartados} de {examinados} eventos de {fault} foram descartados "
            f"antes deste por nao chegarem a {desfecho}."
        )

    mostrar("evento fora de qualquer faixa fisica plausivel", None, motor.diagnosticar(EVENTO_ABSURDO))


if __name__ == "__main__":
    main()
