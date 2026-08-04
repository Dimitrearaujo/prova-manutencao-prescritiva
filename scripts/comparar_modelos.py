"""Compara geradores de texto no mesmo caso real, com tempo medido.

A escolha do modelo nao deveria sair de reputacao: sai de ver a resposta que ele
da para o defeito que o sistema diagnosticou, com o texto de procedimento que o
sistema recuperou, na maquina onde vai rodar.

Uso:
    python scripts/comparar_modelos.py qwen2.5:3b llama3.2:3b
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from prescritiva.config import load_settings
from prescritiva.diagnosis.engine import INSTRUCAO, MotorDiagnostico, _montar_pergunta
from prescritiva.diagnosis.historico import estatisticas
from prescritiva.llm.deterministico import GeradorDeterministico
from prescritiva.llm.ollama_client import OllamaGerador

DEFEITO_DEMO = "desalinhado"


def main() -> None:
    modelos = sys.argv[1:] or [load_settings().llm["model"]]
    settings = load_settings()
    motor = MotorDiagnostico.carregar()

    eventos = pd.read_parquet(settings.paths.processed_dir / "eventos.parquet")
    linha = eventos[eventos["fault"] == DEFEITO_DEMO].sample(1, random_state=7).iloc[0].to_dict()
    linha.pop("fault", None)
    linha.pop("fault_original", None)

    diagnostico = motor.diagnosticar(linha)
    entrada = motor.catalogo[diagnostico.fault]
    trechos = motor.base.buscar(
        f"{entrada['rotulo']} {entrada['termos_busca']}",
        top_k=settings.knowledge["retrieval_top_k"],
        documento=diagnostico.cobertura["documento"],
        preferir_secoes=__import__(
            "prescritiva.diagnosis.engine", fromlist=["SECOES_DE_ACAO"]
        ).SECOES_DE_ACAO,
    )
    pergunta = _montar_pergunta(
        entrada["rotulo"], estatisticas(settings.paths.database, diagnostico.fault), trechos
    )

    print(f"defeito diagnosticado: {diagnostico.rotulo}")
    print(f"procedimento: {diagnostico.cobertura['documento']}")
    print(f"trechos recuperados: {[t.secao[:34] for t in trechos]}")

    geradores = [GeradorDeterministico()] + [
        OllamaGerador(
            m,
            temperatura=settings.llm["temperature"],
            timeout_s=settings.llm["timeout_s"],
            max_tokens=settings.llm["max_tokens"],
            contexto=settings.llm["num_ctx"],
        )
        for m in modelos
    ]

    for gerador in geradores:
        print("\n" + "=" * 78)
        print(gerador.nome.upper())
        print("=" * 78)
        if not gerador.disponivel():
            print("indisponivel (modelo nao instalado ou servico fora do ar)")
            continue
        inicio = time.time()
        try:
            resposta = gerador.gerar(INSTRUCAO, pergunta)
        except Exception as erro:  # noqa: BLE001 - o objetivo aqui e comparar, nao propagar
            print(f"falhou: {type(erro).__name__}: {erro}")
            continue
        decorrido = time.time() - inicio
        palavras = len(resposta.split())
        print(f"tempo: {decorrido:.1f}s | {palavras} palavras | {palavras / max(decorrido, 0.1):.1f} palavras/s\n")
        print(resposta)


if __name__ == "__main__":
    main()
