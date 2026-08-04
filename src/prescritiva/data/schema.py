"""Contrato do evento de vibracao que entra na solucao."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# Pares da mesma grandeza em unidades diferentes. Manter os dois lados infla a
# dimensionalidade e da peso duplo a mesma medida na distancia do kNN, entao a
# coluna imperial e descartada e o Sistema Internacional fica como canonico.
UNIT_DUPLICATES: dict[str, str] = {
    "z_rms_velocity_in_s": "z_rms_velocity_mm_s",
    "x_rms_velocity_in_s": "x_rms_velocity_mm_s",
    "z_peak_velocity_in_s": "z_peak_velocity_mm_s",
    "x_peak_velocity_in_s": "x_peak_velocity_mm_s",
    "temperature_f": "temperature_c",
}

SENSOR_COLUMNS: tuple[str, ...] = (
    "z_rms_velocity_mm_s",
    "x_rms_velocity_mm_s",
    "temperature_c",
    "z_peak_acceleration_g",
    "x_peak_acceleration_g",
    "z_peak_vel_comp_freq_hz",
    "x_peak_vel_comp_freq_hz",
    "z_rms_acceleration_g",
    "x_rms_acceleration_g",
    "z_kurtosis",
    "x_kurtosis",
    "z_crest_factor",
    "x_crest_factor",
    "z_peak_velocity_mm_s",
    "x_peak_velocity_mm_s",
    "z_high_freq_rms_accel_g",
    "x_high_freq_rms_accel_g",
)


class VibrationEvent(BaseModel):
    """Um registro de sensor, no formato exato do banner.csv e do JSON de entrada."""

    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    created_at: datetime | None = None
    rpm: float

    z_rms_velocity_in_s: float | None = None
    z_rms_velocity_mm_s: float
    temperature_f: float | None = None
    temperature_c: float
    x_rms_velocity_in_s: float | None = None
    x_rms_velocity_mm_s: float
    z_peak_acceleration_g: float
    x_peak_acceleration_g: float
    z_peak_vel_comp_freq_hz: float
    x_peak_vel_comp_freq_hz: float
    z_rms_acceleration_g: float
    x_rms_acceleration_g: float
    z_kurtosis: float
    x_kurtosis: float
    z_crest_factor: float
    x_crest_factor: float
    z_peak_velocity_in_s: float | None = None
    z_peak_velocity_mm_s: float
    x_peak_velocity_in_s: float | None = None
    x_peak_velocity_mm_s: float
    z_high_freq_rms_accel_g: float
    x_high_freq_rms_accel_g: float

    # Presente no historico (anotacao do operador) e ausente em um evento novo.
    fault: str | None = Field(default=None)
