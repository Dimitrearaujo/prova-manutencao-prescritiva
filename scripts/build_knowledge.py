"""Etapa 2 do conhecimento: indexa os documentos e imprime o mapa de cobertura.

O mapa e o resultado mais importante desta etapa: ele mostra, defeito a defeito,
se existe procedimento cadastrado. Os defeitos que ficam de fora sao exatamente
os que a solucao deve recusar a prescrever.
"""

from __future__ import annotations

from prescritiva.config import load_catalog, load_settings
from prescritiva.knowledge.extract import extract_all
from prescritiva.knowledge.store import BaseConhecimento


def main() -> None:
    settings = load_settings()
    settings.paths.ensure()
    cfg = settings.knowledge

    documentos = extract_all(
        settings.paths.docs_dir,
        settings.paths.knowledge_dir,
        dpi=cfg["ocr_dpi"],
        min_confidence=cfg["ocr_min_confidence"],
    )
    base = BaseConhecimento(
        tamanho_trecho=cfg["chunk_chars"],
        sobreposicao=cfg["chunk_overlap"],
        chars_escopo=cfg["scope_chars"],
        score_minimo_cobertura=cfg["coverage_min_score"],
    ).indexar(documentos)
    base.salvar(settings.paths.index_dir)

    print(f"{'documento':<10} {'pgs':>4} {'trechos':>8}  titulo")
    for documento in base.documentos:
        print(f"{documento.nome:<10} {documento.paginas:>4} {documento.n_trechos:>8}  {documento.titulo[:70]}")
    print(f"\ntotal de trechos indexados: {len(base.trechos)}")

    catalogo = load_catalog()["faults"]
    print("\n" + "=" * 92)
    print("MAPA DE COBERTURA DOCUMENTAL")
    print("=" * 92)
    print(f"{'defeito':<26} {'coberto':<9} {'doc':<6} {'score':>7} {'2o lugar':>10}  titulo do procedimento")
    sem_cobertura = []
    for fault, entrada in catalogo.items():
        cobertura = base.cobertura(entrada["termo_chave"], entrada["termos_busca"], entrada["rotulo"])
        ranking = base.pontuar_cobertura(entrada["termo_chave"], entrada["termos_busca"])
        segundo = f"{ranking[1][0]:.2f}" if len(ranking) > 1 else "-"
        marca = "sim" if cobertura.coberto else "NAO"
        print(
            f"{fault:<26} {marca:<9} {(cobertura.documento or '-'):<6} "
            f"{cobertura.score:>7.2f} {segundo:>10}  {(cobertura.titulo or '')[:40]}"
        )
        if not cobertura.coberto:
            sem_cobertura.append(fault)

    print(f"\ndefeitos sem procedimento cadastrado: {sem_cobertura or 'nenhum'}")
    print(f"limiar de cobertura em uso: {cfg['coverage_min_score']}")
    print(f"base salva em {settings.paths.index_dir / 'base_conhecimento.json'}")


if __name__ == "__main__":
    main()
