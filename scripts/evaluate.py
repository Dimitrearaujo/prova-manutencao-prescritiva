"""Avaliacao do motor de similaridade.

Quatro experimentos, cada um respondendo a uma pergunta diferente:

aleatorio  Nao e a avaliacao da solucao. Esta aqui para mostrar por que ele nao
           vale neste conjunto de dados: o ensaio gravou cada defeito em um bloco
           continuo de tempo, e sortear coloca leituras a segundos de distancia
           nos dois lados da particao.

temporal   A metrica principal. Dentro de cada bloco, treino no inicio e teste no
           fim, com intervalo de guarda no meio. Representa o uso real:
           reconhecer um defeito ja visto a partir de uma leitura posterior.

campanha   Teste de estresse. Indexa a primeira campanha e consulta as seguintes,
           coletadas depois. Mede o que acontece sob recalibracao da
           instrumentacao.

inedito    Remove uma classe inteira do indice e pergunta por ela. Mede a unica
           coisa que separa esta solucao de um classificador fechado: dizer "nao
           reconheco" em vez de escolher o rotulo menos errado.

Cada um mede em tres niveis, porque acertar o nome do defeito, entregar o
procedimento certo e terminar o atendimento de forma correta sao coisas
diferentes:

acuracia_rotulo     cobra o nome exato do defeito.
acuracia_documento  cobra o que o sistema poe na mao de quem vai abrir a maquina:
                    qual dos procedimentos cadastrados foi entregue. Difere do
                    rotulo porque varios rotulos levam ao mesmo documento - os
                    quatro defeitos de rolamento levam ao Doc1, e trocar anel
                    interno por anel externo nao muda uma linha da instrucao.
desfecho_correto    cobra o fim de linha. Soma ao anterior o acerto que nao cabe
                    nele por definicao: o defeito real nao tem procedimento e o
                    sistema nao entregou procedimento nenhum.

A separacao dos dois ultimos existe porque um evento de ventoinha reconhecido
como ventoinha termina em "nao existe procedimento cadastrado, cadastre um" - o
comportamento que o enunciado exige - e nao tem documento certo a entregar. Ele
nao pode entrar em acuracia_documento sem inflar a metrica, e nao pode ficar de
fora do desfecho sem chamar de erro justamente o acerto mais importante.

O mapa rotulo -> documento nao esta escrito aqui. E o mesmo que o motor descobre
em tempo de execucao, por BaseConhecimento.cobertura(), entao a avaliacao mede o
que a solucao de fato entrega. A versao anterior desta avaliacao colapsava pelo
campo "componente" do catalogo, e o componente nao e o documento: contava correia
(Doc4) e polia (Doc5) como o mesmo procedimento, e juntava sob "rotor" dois
defeitos com documento e um SEM documento - de modo que confundir rotor
excentrico com rotor inclinado, e receber o Doc6 por um defeito que nao tem
procedimento nenhum, entrava como procedimento correto.

O erro oposto tem numero proprio, `prescricao_indevida`: a fracao de eventos em
que o sistema entrega procedimento para um defeito que nao tem procedimento. O
enunciado e literal ao exigir que a solucao se detenha aos problemas que possuem
documentos, e nenhuma metrica de acerto mostra essa violacao.
"""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from prescritiva.config import load_catalog, load_settings
from prescritiva.evaluation.splits import amostrar, split_aleatorio, split_campanha, split_temporal
from prescritiva.knowledge.store import BaseConhecimento
from prescritiva.similarity.index import IndiceSimilaridade

AMOSTRA_TESTE = 2000
AMOSTRA_INEDITO = 300

# Os dois destinos que nao sao documento. Separa-los do nome de um PDF e o que
# permite perguntar "houve prescricao?" sem consultar o catalogo de novo.
SEM_DOCUMENTO = "sem_documento"
CONDICAO_OPERACIONAL = "condicao_operacional"


@lru_cache(maxsize=1)
def _mapa_documento() -> dict[str, str]:
    """Rotulo -> documento que o sistema entregaria para ele.

    Descoberto pela mesma funcao que o motor usa, e nao declarado a mao, para que
    a avaliacao nao possa divergir do produto: cadastrar um PDF novo muda o mapa
    dos dois lados ao mesmo tempo.
    """
    catalogo = load_catalog()
    base = BaseConhecimento.carregar(load_settings().paths.index_dir)
    mapa = {
        fault: base.cobertura(e["termo_chave"], e["termos_busca"], e["rotulo"]).documento
        or SEM_DOCUMENTO
        for fault, e in catalogo["faults"].items()
    }
    mapa.update({estado: CONDICAO_OPERACIONAL for estado in catalogo["estados_operacionais"]})
    return mapa


def _prescreve(destino: str) -> bool:
    """O destino corresponde a entregar um procedimento ao tecnico?"""
    return destino not in {SEM_DOCUMENTO, CONDICAO_OPERACIONAL}


def _consultar_muitos(indice: IndiceSimilaridade, teste: pd.DataFrame) -> pd.DataFrame:
    mapa = _mapa_documento()
    linhas = []
    for posicao in range(len(teste)):
        evento = teste.iloc[[posicao]]
        resultado = indice.consultar(evento)
        verdadeiro = evento["fault"].iloc[0]
        previsto = resultado.fault_predominante
        linhas.append(
            {
                "verdadeiro": verdadeiro,
                "previsto": previsto,
                "doc_verdadeiro": mapa.get(verdadeiro, SEM_DOCUMENTO),
                "doc_previsto": mapa.get(previsto, SEM_DOCUMENTO),
                "confianca": resultado.confianca,
                "distancia": resultado.distancia_media,
                "reconhecido": resultado.reconhecido,
            }
        )
    return pd.DataFrame(linhas)


def _acertou_documento(reconhecidos: pd.DataFrame) -> pd.Series:
    """Entregou o documento que o defeito real exige.

    A segunda condicao e o que separa esta metrica da anterior: quando o defeito
    real nao tem procedimento cadastrado nao existe documento certo, entao o
    evento nao pode contar como acerto por mais que os rotulos combinem.
    """
    return (reconhecidos["doc_verdadeiro"] == reconhecidos["doc_previsto"]) & (
        reconhecidos["doc_verdadeiro"] != SEM_DOCUMENTO
    )


def _desfecho_correto(reconhecidos: pd.DataFrame) -> pd.Series:
    """O fim de linha foi o certo para o tecnico.

    Duas formas de acertar, e a segunda nao cabe em `acuracia_documento` por
    definicao: entregar o documento que o defeito exige, ou nao entregar
    documento nenhum quando o defeito real nao tem procedimento cadastrado. Um
    evento de ventoinha reconhecido como ventoinha termina em "nao existe
    procedimento, cadastre um" - que e exatamente o que o enunciado pede, e seria
    perverso contar como erro so porque nenhum PDF foi entregue.
    """
    recusa_correta = (reconhecidos["doc_verdadeiro"] == SEM_DOCUMENTO) & (
        reconhecidos["doc_previsto"] == SEM_DOCUMENTO
    )
    return _acertou_documento(reconhecidos) | recusa_correta


def _prescreveu_sem_documentacao(reconhecidos: pd.DataFrame) -> pd.Series:
    """Prescreveu procedimento para defeito que nao tem procedimento nenhum.

    O evento e de um defeito sem documento, o sistema o confundiu com um defeito
    que tem, e por isso entregou um procedimento. E o erro que o enunciado proibe
    de forma direta, e ele fica invisivel em qualquer metrica de acerto.
    """
    return (reconhecidos["doc_verdadeiro"] == SEM_DOCUMENTO) & reconhecidos["doc_previsto"].map(
        _prescreve
    )


def _metricas(res: pd.DataFrame) -> dict:
    reconhecidos = res[res["reconhecido"]]
    if not len(reconhecidos):
        return {"eventos": len(res), "eventos_reconhecidos": 0, "taxa_rejeicao": 1.0,
                "acuracia_rotulo": 0.0, "acuracia_documento": 0.0, "desfecho_correto": 0.0,
                "prescricao_indevida": 0.0,
                "acuracia_global": 0.0, "distancia_media": float(res["distancia"].mean())}
    return {
        "eventos": len(res),
        "eventos_reconhecidos": int(len(reconhecidos)),
        "taxa_rejeicao": float((~res["reconhecido"]).mean()),
        "acuracia_rotulo": float((reconhecidos["verdadeiro"] == reconhecidos["previsto"]).mean()),
        "acuracia_documento": float(_acertou_documento(reconhecidos).mean()),
        "desfecho_correto": float(_desfecho_correto(reconhecidos).mean()),
        "prescricao_indevida": float(_prescreveu_sem_documentacao(reconhecidos).mean()),
        "acuracia_global": float((res["verdadeiro"] == res["previsto"]).mean()),
        "distancia_media": float(res["distancia"].mean()),
    }


def _rodar(treino: pd.DataFrame, teste: pd.DataFrame, cfg: dict, *, amostra_calibracao: int = 1500):
    indice = IndiceSimilaridade(cfg["n_neighbors"], cfg["min_consensus"]).fit(treino)
    indice.calibrar_limiares(treino, cfg["reject_distance_percentile"], amostra=amostra_calibracao)
    return _consultar_muitos(indice, teste)


def experimento_defeito_inedito(eventos: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    linhas = []
    for alvo in sorted(eventos.loc[eventos["e_defeito"], "fault"].unique()):
        treino = eventos[eventos["fault"] != alvo]
        teste = amostrar(eventos[eventos["fault"] == alvo], AMOSTRA_INEDITO)
        resultado = _rodar(treino, teste, cfg, amostra_calibracao=600)
        reconhecidos = resultado[resultado["reconhecido"]]
        # Quando nao rejeita, o rotulo escolhido ainda leva ao MESMO documento?
        # Remover rolamento_ball e receber rolamento_inner no lugar mantem o Doc1
        # e a mesma acao corretiva: para o tecnico, a prescricao continua certa.
        # Para eccentric_rotor e ventoinha a coluna e zero por construcao - nao ha
        # documento certo a entregar - e quem os descreve e o par
        # desfecho_util / prescricao_indevida.
        salvos = float(_acertou_documento(reconhecidos).mean()) if len(reconhecidos) else 0.0
        util = float(_desfecho_correto(reconhecidos).mean()) if len(reconhecidos) else 0.0
        indevida = (
            float(_prescreveu_sem_documentacao(reconhecidos).mean()) if len(reconhecidos) else 0.0
        )
        rejeitado = float((~resultado["reconhecido"]).mean())
        linhas.append(
            {
                "defeito_removido": alvo,
                "eventos": len(resultado),
                "rejeitado_corretamente": rejeitado,
                "documento_ainda_correto": salvos,
                "prescricao_indevida": indevida,
                "desfecho_util": rejeitado + (1 - rejeitado) * util,
                "rotulo_mais_dado": reconhecidos["previsto"].value_counts().idxmax()
                if len(reconhecidos)
                else "-",
            }
        )
    return pd.DataFrame(linhas)


def main() -> None:
    settings = load_settings()
    cfg = settings.similarity
    eventos = pd.read_parquet(settings.paths.processed_dir / "eventos.parquet")

    if not (settings.paths.index_dir / "base_conhecimento.json").exists():
        raise SystemExit(
            "a base de conhecimento nao existe; execute scripts/build_knowledge.py antes. "
            "A avaliacao depende dela para saber qual documento cada defeito recebe."
        )

    particoes = {
        "Aleatorio (invalido)": split_aleatorio(eventos),
        "Temporal (principal)": split_temporal(eventos),
        "Nova campanha (estresse)": split_campanha(eventos),
    }

    splits: dict[str, dict] = {}
    matrizes: dict[str, dict] = {}
    for nome, (treino, teste_todo) in particoes.items():
        teste = amostrar(teste_todo, AMOSTRA_TESTE)
        resultado = _rodar(treino, teste, cfg)
        splits[nome] = _metricas(resultado)
        reconhecidos = resultado[resultado["reconhecido"]]
        matrizes[nome] = (
            pd.crosstab(reconhecidos["verdadeiro"], reconhecidos["previsto"], normalize="index")
            .round(2)
            .to_dict()
            if len(reconhecidos)
            else {}
        )

        print("=" * 78)
        print(nome.upper())
        print("=" * 78)
        for chave, valor in splits[nome].items():
            print(f"  {chave:<32} {valor:.4f}" if isinstance(valor, float) else f"  {chave:<32} {valor}")
        print()

    print("=" * 78)
    print("DEFEITO INEDITO (a classe some do indice e e consultada)")
    print("=" * 78)
    inedito = experimento_defeito_inedito(eventos, cfg)
    print(inedito.to_string(index=False))
    print(f"\n  rejeicao media: {inedito['rejeitado_corretamente'].mean():.1%}")
    print(f"  desfecho util medio (rejeitou OU terminou certo para o tecnico): "
          f"{inedito['desfecho_util'].mean():.1%}")
    print("\n  documento_ainda_correto e prescricao_indevida sao fracoes dos eventos")
    print("  RECONHECIDOS; rejeitado_corretamente e desfecho_util, do total.")
    print("  Para eccentric_rotor e ventoinha nao existe documento certo a entregar:")
    print("  documento_ainda_correto e zero por construcao, e o que descreve essas duas")
    print("  linhas e prescricao_indevida - quanto o sistema prescreveu sem lastro.")

    destino = settings.paths.index_dir / "avaliacao.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(
            {
                "splits": splits,
                "matrizes_confusao": matrizes,
                "defeito_inedito": inedito.to_dict(orient="records"),
                "mapa_documento": _mapa_documento(),
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"\nresultados salvos em {destino}")


if __name__ == "__main__":
    main()
