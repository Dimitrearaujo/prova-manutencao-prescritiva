"""Demonstracao dos quatro desfechos do diagnostico, em linha de comando.

Serve de roteiro para a apresentacao: um exemplo de cada caminho que o motor
pode tomar, incluindo os dois em que ele se recusa a prescrever.
"""

from __future__ import annotations

import pandas as pd

from prescritiva.config import load_settings
from prescritiva.diagnosis.engine import MotorDiagnostico

CASOS = [
    ("defeito com procedimento cadastrado", "desalinhado"),
    ("defeito SEM procedimento cadastrado", "ventoinha"),
    ("condicao de operacao, nao e problema", "motor_desligado"),
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


def mostrar(titulo: str, rotulo_real: str | None, diagnostico) -> None:
    print("\n" + "=" * 78)
    print(titulo.upper())
    print("=" * 78)
    if rotulo_real:
        print(f"rotulo anotado pelo operador: {rotulo_real}  (nao enviado ao motor)")
    print(f"desfecho    : {diagnostico.situacao}")
    print(f"defeito     : {diagnostico.rotulo or '-'}")
    print(f"consenso    : {diagnostico.confianca:.0%}  | regime: {diagnostico.regime_rpm}")
    print(f"\n{diagnostico.mensagem}")
    if diagnostico.trechos:
        print(f"\ntrechos usados: {[t['secao'][:38] for t in diagnostico.trechos]}")
    if diagnostico.instrucoes:
        print(f"\n--- instrucoes ({diagnostico.gerador}) ---")
        print(diagnostico.instrucoes[:1400])
    vizinhos = diagnostico.similaridade.get("vizinhos", [])[:5]
    if vizinhos:
        print("\nevidencia (5 vizinhos mais proximos):")
        for v in vizinhos:
            print(f"  id {v['id']:<8} {v['fault_original']:<26} d={v['distancia']:.3f}  {v['created_at'][:19]}")


def main() -> None:
    settings = load_settings()
    eventos = pd.read_parquet(settings.paths.processed_dir / "eventos.parquet")
    motor = MotorDiagnostico.carregar()
    print(f"gerador de texto em uso: {motor.gerador.nome}")

    for titulo, fault in CASOS:
        linha = eventos[eventos["fault"] == fault].sample(1, random_state=7).iloc[0].to_dict()
        linha.pop("fault", None)
        linha.pop("fault_original", None)
        mostrar(titulo, fault, motor.diagnosticar(linha))

    mostrar("evento fora de qualquer faixa fisica plausivel", None, motor.diagnosticar(EVENTO_ABSURDO))


if __name__ == "__main__":
    main()
