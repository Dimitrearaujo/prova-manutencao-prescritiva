"""Leitura do contexto operacional em que um padrao ja apareceu.

O enunciado pede, ao lado da quantidade e da frequencia, o "contexto operacional
associado". A estatistica historica ja traz a distribuicao por regime de
rotacao; o que faltava era confrontar essa distribuicao com o regime do evento
que acabou de chegar. A diferenca muda a conduta do tecnico: um defeito que o
historico registra sobretudo a 500 rpm chegando a 2000 rpm nao e a ocorrencia
de sempre, e vale conferir a condicao de operacao antes de executar o
procedimento.

Fica fora do motor porque nao decide nada. Nenhum dos quatro desfechos muda em
funcao desta leitura: ela explica o diagnostico ja tomado, e por isso a tela e
quem consome a API leem exatamente o mesmo texto.
"""

from __future__ import annotations

from dataclasses import dataclass

# O alerta e relativo ao regime dominante, nunca a um piso fixo de fatia. Um
# corte absoluto acusaria fora do padrao quase sempre: no historico do ensaio a
# maioria dos defeitos foi gravada nas tres rotacoes em proporcao parecida, e
# 25% tanto pode ser distribuicao equilibrada entre quatro regimes quanto
# concentracao real quando o defeito so aparece em dois.
FATOR_REGIME_ATIPICO = 0.5

TIPICO = "tipico"
ATIPICO = "atipico"
INEDITO = "inedito"
SEM_HISTORICO = "sem_historico"


@dataclass
class ContextoRegime:
    """Como o regime do evento se posiciona na historia do padrao identificado."""

    regime_evento: str
    ocorrencias_no_regime: int
    fracao_no_regime: float
    regime_dominante: str | None
    fracao_dominante: float
    total: int
    alinhamento: str
    leitura: str


def formatar_regime(regime: str) -> str:
    """Deixa o regime legivel na frase: 2000rpm vira 2000 rpm."""
    return regime.replace("rpm", " rpm").strip() if regime else "regime desconhecido"


def contexto_regime(por_regime: dict[str, int], regime_evento: str, rotulo: str) -> ContextoRegime:
    """Compara o regime do evento com a distribuicao historica do padrao.

    `rotulo` entra apenas para a frase sair legivel para quem opera a maquina;
    nenhuma regra depende dele.
    """
    total = sum(por_regime.values())
    regime_visivel = formatar_regime(regime_evento)
    if not total:
        return ContextoRegime(
            regime_evento=regime_evento,
            ocorrencias_no_regime=0,
            fracao_no_regime=0.0,
            regime_dominante=None,
            fracao_dominante=0.0,
            total=0,
            alinhamento=SEM_HISTORICO,
            leitura="Sem ocorrencias anteriores no historico, nao ha regime de rotacao a comparar.",
        )

    dominante, ocorrencias_dominante = max(por_regime.items(), key=lambda item: item[1])
    no_regime = por_regime.get(regime_evento, 0)
    fracao = no_regime / total
    fracao_dominante = ocorrencias_dominante / total

    if not no_regime:
        alinhamento = INEDITO
        leitura = (
            f"Atencao: {rotulo} nunca foi registrado em {regime_visivel}. As "
            f"{_milhar(total)} ocorrencias do historico estao todas em outros regimes de "
            f"rotacao, com pico em {formatar_regime(dominante)} "
            f"({fracao_dominante:.0%}). Nao existe caso anterior comparavel nesta rotacao."
        )
    elif fracao < FATOR_REGIME_ATIPICO * fracao_dominante:
        alinhamento = ATIPICO
        leitura = (
            f"Atencao: este evento chegou em {regime_visivel}, mas {rotulo} se concentra "
            f"em {formatar_regime(dominante)} ({fracao_dominante:.0%} das ocorrencias). "
            f"Apenas {fracao:.0%} do historico deste padrao "
            f"({_milhar(no_regime)} de {_milhar(total)} registros) foi gravado na rotacao "
            "atual, entao vale confirmar a condicao de operacao antes de executar."
        )
    elif regime_evento == dominante:
        alinhamento = TIPICO
        leitura = (
            f"Contexto operacional esperado: {regime_visivel} e a rotacao em que "
            f"{rotulo} mais aparece, com {fracao:.0%} das {_milhar(total)} ocorrencias "
            f"do historico ({_milhar(no_regime)} registros)."
        )
    else:
        alinhamento = TIPICO
        leitura = (
            f"Contexto operacional compativel: {fracao:.0%} das {_milhar(total)} "
            f"ocorrencias de {rotulo} foram registradas em {regime_visivel} "
            f"({_milhar(no_regime)} registros). O pico historico e "
            f"{formatar_regime(dominante)}, com {fracao_dominante:.0%}."
        )

    return ContextoRegime(
        regime_evento=regime_evento,
        ocorrencias_no_regime=no_regime,
        fracao_no_regime=fracao,
        regime_dominante=dominante,
        fracao_dominante=fracao_dominante,
        total=total,
        alinhamento=alinhamento,
        leitura=leitura,
    )


def _milhar(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")
