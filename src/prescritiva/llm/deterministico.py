"""Gerador sem modelo de linguagem: monta a resposta a partir dos trechos.

Existe por dois motivos. Operacional: a solucao continua entregando instrucao
correta se o servico do modelo cair, e a planta nao fica sem resposta. Tecnico:
e a linha de base contra a qual o modelo precisa provar que vale a pena, e como
ele so recorta texto ja aprovado pela engenharia, nao tem como alucinar.
"""

from __future__ import annotations


class GeradorDeterministico:
    nome = "deterministico"

    def disponivel(self) -> bool:
        return True

    def gerar(self, instrucao: str, pergunta: str) -> str:
        trechos = _extrair_trechos(pergunta)
        if not trechos:
            return (
                "Nao ha trecho de procedimento recuperado para este caso. "
                "Consulte a documentacao completa do equipamento."
            )
        corpo = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(trechos, start=1))
        return (
            "Trechos do procedimento aplicaveis a este defeito, na ordem de "
            f"aderencia ao evento:\n\n{corpo}\n\n"
            "Texto reproduzido do procedimento cadastrado, sem reescrita."
        )


def _extrair_trechos(pergunta: str) -> list[str]:
    marcador = "TRECHOS DO PROCEDIMENTO:"
    if marcador not in pergunta:
        return []
    bloco = pergunta.split(marcador, 1)[1]
    return [parte.strip() for parte in bloco.split("\n---\n") if parte.strip()]
