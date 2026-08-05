"""Auditoria adversarial do portao de cobertura do chat.

O enunciado exige que a solucao se detenha unicamente a problemas que possuem
documentos. No caminho do diagnostico isso e decidido pelo ESCOPO do documento.
No chat a pergunta e texto livre, e ate aqui a decisao de responder caia no
CORPO (BM25) sempre que a pergunta nao nomeava o defeito com as palavras do
catalogo. Uma parafrase - "o rotor esta batendo" no lugar de "rotor
excentrico" - contornava o portao e o sistema prescrevia para um defeito que
ele mesmo declara sem procedimento.

Este script mede esse contorno. Ele nao opina: roda um corpus e conta.

Tres mecanismos de contorno sao exercitados, porque tapar so o primeiro deixa
os outros dois abertos:

    parafrase  a pergunta descreve o defeito sem usar as palavras do catalogo;
    cocitacao  a pergunta cita junto um defeito COBERTO, o que fazia o codigo
               eleger o documento desse defeito e pular as checagens seguintes;
    seletor    a pergunta vem com o documento fixado, como quando o tecnico usa
               o combo "Restringir a um procedimento" da tela.

O corpus de CONTROLE existe porque zerar vazamento e trivial: bastaria recusar
tudo. Um portao que emudece o produto e uma falha pior que a que ele conserta,
entao as duas taxas sao publicadas sempre juntas.

O gerador usado e um duble que registra a chamada em vez de chamar o modelo.
Assim a auditoria mede o PORTAO, roda em segundos e nao depende do Ollama estar
no ar. A checagem separada com o GeradorDeterministico responde outra pergunta:
quando o portao vaza, o texto que sai chega a ser prescritivo?

Uso:
    python scripts/auditoria_chat.py
    python scripts/auditoria_chat.py --corpus caminho/outro_corpus.json
    python scripts/auditoria_chat.py --json     # saida legivel por maquina

Sai com codigo 1 se houver vazamento ou mudez, para servir de portao em CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from prescritiva.config import load_settings  # noqa: E402
from prescritiva.diagnosis.engine import MotorDiagnostico  # noqa: E402
from prescritiva.knowledge.store import BaseConhecimento  # noqa: E402
from prescritiva.llm.deterministico import GeradorDeterministico  # noqa: E402
from prescritiva.similarity.index import IndiceSimilaridade  # noqa: E402

CORPUS_PADRAO = RAIZ / "tests" / "dados" / "corpus_chat_adversarial.json"

# A auditoria reconhece RESPOSTA, nao recusa. Manter a lista pelo lado da recusa
# ja custou um numero errado: as situacoes novas do portao ("sem_defeito_nomeado",
# "restricao_incompativel") ficaram de fora da lista e 40 recusas corretas foram
# contadas como vazamento. Do lado da resposta so existe um nome, e ele so muda
# se o contrato do motor mudar de verdade.
SITUACAO_RESPONDIDA = "respondido"


class GeradorEspiao:
    """Registra a chamada em vez de gerar.

    Se ele for chamado numa pergunta de ataque, o portao deixou passar: existe
    contexto aprovado o bastante para o modelo escrever procedimento em cima.
    """

    nome = "espiao"

    def __init__(self) -> None:
        self.chamadas: list[tuple[str, str]] = []

    def disponivel(self) -> bool:
        return True

    def gerar(self, instrucao: str, pergunta: str) -> str:
        self.chamadas.append((instrucao, pergunta))
        return "[resposta do duble]"


@dataclass
class Resultado:
    pergunta: str
    esperado: str  # "recusa" ou "resposta"
    situacao: str
    documento: str | None
    recusou: bool
    passou: bool
    alvo: str = ""
    mecanismo: str = ""
    documento_esperado: str = ""
    documento_seletor: str = ""
    origem: str = ""
    melhor_score: float = 0.0
    defeitos_citados: list[str] = field(default_factory=list)
    chamou_modelo: bool = False


def carregar_motor(gerador: Any) -> MotorDiagnostico:
    settings = load_settings()
    if not (settings.paths.index_dir / "base_conhecimento.json").exists():
        raise SystemExit(
            "Base de conhecimento ausente. Rode scripts/build_knowledge.py e "
            "scripts/build_index.py antes da auditoria."
        )
    return MotorDiagnostico(
        indice=IndiceSimilaridade.carregar(settings.paths.index_dir),
        base=BaseConhecimento.carregar(settings.paths.index_dir),
        database=settings.paths.database,
        gerador=gerador,
        top_k=settings.knowledge["retrieval_top_k"],
    )


def _defeitos_citados(motor: MotorDiagnostico, pergunta: str) -> list[str]:
    """Quais defeitos do catalogo a pergunta nomeia literalmente.

    Serve para explicar o vazamento, nao para decidir nada: e essa lista vazia
    que jogava a pergunta no ramo permissivo.
    """
    try:
        return [rotulo for rotulo, _ in motor._defeitos_citados(pergunta)]
    except Exception:  # noqa: BLE001 - a auditoria nao pode quebrar por detalhe interno
        return []


def rodar_caso(
    motor: MotorDiagnostico,
    espiao: GeradorEspiao,
    pergunta: str,
    *,
    esperado: str,
    documento: str | None = None,
    **extra: Any,
) -> Resultado:
    antes = len(espiao.chamadas)
    saida = motor.perguntar(pergunta, documento=documento)
    situacao = saida.get("situacao", "?")
    recusou = situacao != SITUACAO_RESPONDIDA
    trechos = saida.get("trechos") or []
    return Resultado(
        pergunta=pergunta,
        esperado=esperado,
        situacao=situacao,
        documento=saida.get("documento"),
        recusou=recusou,
        passou=recusou if esperado == "recusa" else (not recusou),
        melhor_score=float(trechos[0]["score"]) if trechos else 0.0,
        defeitos_citados=_defeitos_citados(motor, pergunta),
        chamou_modelo=len(espiao.chamadas) > antes,
        **extra,
    )


def auditar(corpus: dict[str, Any]) -> tuple[list[Resultado], list[Resultado]]:
    espiao = GeradorEspiao()
    motor = carregar_motor(espiao)

    ataques = [
        rodar_caso(
            motor,
            espiao,
            caso["pergunta"],
            esperado="recusa",
            documento=caso.get("documento_seletor") or None,
            alvo=caso.get("alvo", ""),
            mecanismo=caso.get("mecanismo", ""),
            documento_seletor=caso.get("documento_seletor", ""),
        )
        for caso in corpus["ataques"]
    ]
    controles = [
        rodar_caso(
            motor,
            espiao,
            caso["pergunta"],
            esperado="resposta",
            documento_esperado=caso.get("documento_esperado", ""),
            origem=caso.get("origem", "pergunta de manutencao bem formada"),
        )
        for caso in corpus["controles"]
    ]
    # Um controle respondido pelo documento errado nao e sucesso: e a resposta
    # certa tirada do procedimento errado, que e o mesmo erro com outra cara.
    for caso in controles:
        if caso.passou and caso.documento_esperado:
            caso.passou = caso.documento == caso.documento_esperado
    return ataques, controles


def texto_prescritivo_nos_vazamentos(vazamentos: list[Resultado]) -> list[dict[str, Any]]:
    """Refaz os vazamentos com o GeradorDeterministico.

    O README apresenta esse gerador como o caminho que nao tem como alucinar,
    porque so recorta texto aprovado. A afirmacao vale para o TEXTO e nao para a
    ESCOLHA do texto: recortar o procedimento errado entrega ao tecnico uma
    instrucao real de outro defeito, que e pior do que um paragrafo vago.
    """
    if not vazamentos:
        return []
    motor = carregar_motor(GeradorDeterministico())
    saidas = []
    for caso in vazamentos:
        resposta = motor.perguntar(
            caso.pergunta, documento=caso.documento_seletor or None
        )
        texto = resposta.get("resposta", "")
        saidas.append(
            {
                "pergunta": caso.pergunta,
                "documento": resposta.get("documento"),
                "prescritivo": resposta.get("situacao") == SITUACAO_RESPONDIDA and bool(texto.strip()),
                "primeiros_200": texto[:200].replace("\n", " "),
            }
        )
    return saidas


def relatorio(ataques: list[Resultado], controles: list[Resultado]) -> str:
    vazou = [r for r in ataques if not r.passou]
    mudos = [r for r in controles if not r.passou]
    linhas: list[str] = []

    def secao(titulo: str) -> None:
        linhas.append("")
        linhas.append(titulo)
        linhas.append("-" * len(titulo))

    linhas.append("=" * 78)
    linhas.append("AUDITORIA DO PORTAO DE COBERTURA DO CHAT")
    linhas.append("=" * 78)

    secao("Resumo")
    total_a, total_c = len(ataques), len(controles)
    taxa_v = 100 * len(vazou) / total_a if total_a else 0.0
    taxa_m = 100 * len(mudos) / total_c if total_c else 0.0
    # Mudez e desvio sao erros diferentes e nao podem sair somados. Recusar uma
    # pergunta legitima deixa o tecnico sem resposta e ele percebe; responder pelo
    # procedimento errado entrega instrucao de outro defeito e ele nao percebe.
    calados = [r for r in mudos if r.recusou]
    desviados = [r for r in mudos if not r.recusou]
    linhas.append(f"  vazamento : {len(vazou):3d}/{total_a:<3d} ({taxa_v:5.1f}%)  <- tem que ser 0")
    linhas.append(f"  mudez     : {len(mudos):3d}/{total_c:<3d} ({taxa_m:5.1f}%)  <- tem que ser 0")
    linhas.append(f"     recusou : {len(calados):3d}  (pergunta legitima sem resposta)")
    linhas.append(f"     desviou : {len(desviados):3d}  (respondeu pelo procedimento errado)")

    if any(r.origem for r in controles):
        secao("Mudez por tipo de pergunta legitima")
        origens = sorted({r.origem for r in controles})
        for origem in origens:
            do_tipo = [r for r in controles if r.origem == origem]
            perdidos = [r for r in do_tipo if not r.passou]
            linhas.append(f"  {len(perdidos):3d}/{len(do_tipo):<3d}  {origem}")

    if total_a:
        secao("Qualidade da recusa nos ataques")
        # O enunciado exige recusa DUPLA: reportar que o problema nao esta
        # documentado E sugerir o cadastro. Recusa generica cumpre metade.
        recusados = [r for r in ataques if r.passou]
        especificas = [r for r in recusados if r.situacao == "defeito_sem_documentacao"]
        ambiguas = [r for r in recusados if r.situacao == "fenomeno_ambiguo"]
        genericas = [r for r in recusados if r.situacao == "sem_defeito_nomeado"]
        linhas.append(f"  nomeou o defeito que falta documentar : {len(especificas):3d}/{len(recusados)}")
        linhas.append(f"  pediu o componente (fenomeno ambiguo) : {len(ambiguas):3d}/{len(recusados)}")
        linhas.append(f"  recusa generica, pede nomear o defeito: {len(genericas):3d}/{len(recusados)}")

    if total_a:
        secao("Vazamento por mecanismo")
        por_mec = Counter(r.mecanismo for r in ataques)
        vaz_mec = Counter(r.mecanismo for r in vazou)
        for mecanismo, total in sorted(por_mec.items()):
            linhas.append(f"  {mecanismo:<12} {vaz_mec[mecanismo]:3d}/{total:<3d}")

        secao("Vazamento por alvo")
        por_alvo = Counter(r.alvo for r in ataques)
        vaz_alvo = Counter(r.alvo for r in vazou)
        for alvo, total in sorted(por_alvo.items()):
            linhas.append(f"  {alvo:<18} {vaz_alvo[alvo]:3d}/{total:<3d}")

    if vazou:
        secao(f"Perguntas que vazaram ({len(vazou)})")
        for r in vazou:
            citados = ", ".join(r.defeitos_citados) or "-"
            sel = f" [seletor={r.documento_seletor}]" if r.documento_seletor else ""
            linhas.append(f"  {r.mecanismo:<10} {r.alvo:<16} {r.pergunta}{sel}")
            linhas.append(
                f"     -> respondeu por {r.documento} (score {r.melhor_score:.2f}); "
                f"catalogo casou: {citados}"
            )

    if mudos:
        secao(f"Controles que o portao emudeceu ou desviou ({len(mudos)})")
        for r in mudos:
            linhas.append(
                f"  {r.pergunta}\n     -> esperado {r.documento_esperado}, "
                f"veio {r.documento} ({r.situacao})"
            )

    return "\n".join(linhas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PADRAO)
    parser.add_argument("--json", action="store_true", help="saida em JSON")
    parser.add_argument(
        "--deterministico",
        action="store_true",
        help="refaz os vazamentos com o GeradorDeterministico e mostra o texto",
    )
    args = parser.parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"Corpus nao encontrado: {args.corpus}")
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))

    ataques, controles = auditar(corpus)
    vazou = [r for r in ataques if not r.passou]
    mudos = [r for r in controles if not r.passou]

    if args.json:
        print(
            json.dumps(
                {
                    "corpus": str(args.corpus),
                    "ataques": len(ataques),
                    "controles": len(controles),
                    "vazamentos": len(vazou),
                    "mudez": len(mudos),
                    "detalhe_ataques": [vars(r) for r in ataques],
                    "detalhe_controles": [vars(r) for r in controles],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(relatorio(ataques, controles))
        if args.deterministico and vazou:
            print("\nTexto entregue nos vazamentos, com o GeradorDeterministico")
            print("-" * 58)
            for item in texto_prescritivo_nos_vazamentos(vazou):
                marca = "PRESCREVEU" if item["prescritivo"] else "recusou"
                print(f"  [{marca}] {item['pergunta']}  ({item['documento']})")
                print(f"      {item['primeiros_200']}")

    return 1 if (vazou or mudos) else 0


if __name__ == "__main__":
    raise SystemExit(main())
