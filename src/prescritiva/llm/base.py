"""Contrato do gerador de texto.

O motor de diagnostico nao conhece nenhum modelo especifico. Ele conhece esta
interface. Trocar o modelo local por outro, ou por um servico remoto, e escrever
uma classe nova aqui e mudar uma linha do settings.yaml.
"""

from __future__ import annotations

from typing import Protocol


class GeradorTexto(Protocol):
    nome: str

    def disponivel(self) -> bool:
        """Se falso, o motor cai no gerador deterministico sem quebrar."""

    def gerar(self, instrucao: str, pergunta: str) -> str: ...
