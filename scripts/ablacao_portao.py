"""Ablacao do portao do chat: quanto cada regra contribui, medido.

O projeto ja mede a contribuicao de cada familia de feature em
scripts/ablacao_features.py. O portao do chat merece o mesmo tratamento: cada
regra dele custa recusa de pergunta legitima, e uma regra que nao paga esse
custo com reducao de vazamento nao deveria estar la.

As regras ablacionadas:

    ancora        exigir que a pergunta nomeie um defeito do catalogo. Sem isso
                  a decisao volta para o BM25 sobre o corpo inteiro.
    ambiguos      recusar fenomeno que pertence a mais de um defeito quando o
                  dono legitimo nao foi nomeado ("excentricidade" sem dizer se e
                  do rotor ou da polia).
    sinonimos     as formas alternativas de nomear um defeito coberto
                  ("balanceamento", "cocked rotor", "pe manco").
    restricao     o documento escolhido na tela so pode estreitar candidatos.

Cada linha e o portao COMPLETO menos aquela regra, para que a contribuicao
apareca como piora, e nao como um numero solto.

Uso:  python scripts/ablacao_portao.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))
if str(RAIZ / "scripts") not in sys.path:
    sys.path.insert(0, str(RAIZ / "scripts"))

from auditoria_chat import (  # noqa: E402
    CORPUS_PADRAO,
    SITUACAO_RESPONDIDA,
    GeradorEspiao,
    carregar_motor,
)


def _sem_campo(catalogo: dict, campo: str) -> dict:
    limpo = copy.deepcopy(catalogo)
    for entrada in limpo.values():
        entrada.pop(campo, None)
    return limpo


def medir(motor, corpus, *, ignorar_restricao: bool = False) -> dict[str, int]:
    vazou = 0
    for caso in corpus["ataques"]:
        documento = None if ignorar_restricao else (caso.get("documento_seletor") or None)
        saida = motor.perguntar(caso["pergunta"], documento=documento)
        if saida["situacao"] == SITUACAO_RESPONDIDA:
            vazou += 1

    calou = desviou = 0
    for caso in corpus["controles"]:
        saida = motor.perguntar(caso["pergunta"])
        if saida["situacao"] != SITUACAO_RESPONDIDA:
            calou += 1
        elif caso.get("documento_esperado") and saida["documento"] != caso["documento_esperado"]:
            desviou += 1
    return {"vazou": vazou, "calou": calou, "desviou": desviou}


def main() -> int:
    corpus = json.loads(CORPUS_PADRAO.read_text(encoding="utf-8"))
    motor = carregar_motor(GeradorEspiao())
    catalogo_completo = copy.deepcopy(motor.catalogo)
    n_a, n_c = len(corpus["ataques"]), len(corpus["controles"])

    print(f"corpus curado: {n_a} ataques (tem que recusar), {n_c} controles (tem que responder)")
    if corpus.get("ambiguos_excluidos"):
        print(
            f"{len(corpus['ambiguos_excluidos'])} perguntas ambiguas ficaram FORA dos dois "
            "denominadores, por decisao dos curadores"
        )
    print()
    print(f"{'configuracao':<34} {'vazou':>8} {'calou':>8} {'desviou':>8}")
    print("-" * 62)

    completo = medir(motor, corpus)
    print(
        f"{'portao completo':<34} {completo['vazou']:>4}/{n_a:<3} "
        f"{completo['calou']:>4}/{n_c:<3} {completo['desviou']:>4}/{n_c:<3}"
    )

    for rotulo, campo in [
        ("sem termos_ambiguos", "termos_ambiguos"),
        ("sem termos_pergunta (sinonimos)", "termos_pergunta"),
    ]:
        motor.catalogo = _sem_campo(catalogo_completo, campo)
        r = medir(motor, corpus)
        print(
            f"{rotulo:<34} {r['vazou']:>4}/{n_a:<3} {r['calou']:>4}/{n_c:<3} "
            f"{r['desviou']:>4}/{n_c:<3}"
        )
    motor.catalogo = catalogo_completo

    r = medir(motor, corpus, ignorar_restricao=True)
    print(
        f"{'ataques sem usar o seletor':<34} {r['vazou']:>4}/{n_a:<3} "
        f"{r['calou']:>4}/{n_c:<3} {r['desviou']:>4}/{n_c:<3}"
    )

    print(
        "\nLeitura: 'calou' e recusa de pergunta legitima e 'desviou' e resposta pelo "
        "procedimento errado.\nSao erros diferentes e nao devem sair somados - o tecnico "
        "percebe o primeiro e nao percebe o segundo."
    )
    print(
        "\nA ultima linha nao e ablacao da regra de restricao: ela refaz os ataques sem\n"
        "fixar documento. Dar o mesmo numero e o resultado esperado e mede outra coisa -\n"
        "que o seletor deixou de ser atalho, porque a ancora ja recusa essas perguntas\n"
        "antes de a restricao ser consultada. A regra em si e exercitada em\n"
        "tests/test_motor_chat.py, num caso onde a ancora PASSA e so a restricao barra."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
