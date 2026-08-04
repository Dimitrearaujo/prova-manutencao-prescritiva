"""Fecha o ciclo que o diagrama do enunciado descreve.

O sistema recusa prescrever para um defeito sem procedimento e pede o cadastro.
Este teste verifica a outra metade: que cadastrar o documento resolve de fato a
recusa, sem alterar codigo nem reconstruir o indice de similaridade.

Usa um procedimento de exemplo gerado aqui, claramente identificado como tal, e
o remove no fim para nao contaminar a base real.
"""

from __future__ import annotations

import fitz
import pytest

from prescritiva.config import load_catalog, load_settings
from prescritiva.knowledge.extract import extract_all
from prescritiva.knowledge.store import BaseConhecimento

NOME_ARQUIVO = "DocExemploVentoinha.pdf"

TEXTO = """Procedimento para Diagnostico e Correcao de Problemas em Ventoinha

DOCUMENTO DE EXEMPLO, criado para testar o cadastro de novos procedimentos.
Nao e um documento tecnico real e nao deve ser usado em manutencao.

1. Objetivo
Este documento estabelece o procedimento para identificar, diagnosticar e
corrigir falhas em ventoinhas e rotores de ventilacao acoplados a maquinas
rotativas, reduzindo vibracao e desbalanceamento causados pelas pas.

2. Principais Falhas
2.1 Acumulo de sujeira nas pas
Deposito irregular de material desloca o centro de massa e gera vibracao.
2.2 Pa trincada ou deformada
Impacto mecanico ou fadiga altera a geometria da helice.
2.3 Folga na fixacao do cubo
Aperto insuficiente permite deslocamento axial durante a operacao.

3. Ferramentas Necessarias
Chave de torque, analisador de vibracao, escova de limpeza, paquimetro.

4. Procedimento de Correcao
1. Bloquear e sinalizar o equipamento antes de qualquer intervencao.
2. Remover a protecao e inspecionar visualmente as pas.
3. Limpar o deposito de material acumulado em todas as pas.
4. Medir a espessura das pas e comparar com a especificacao.
5. Substituir a ventoinha caso exista trinca ou deformacao permanente.
6. Reapertar o cubo com o torque especificado pelo fabricante.

5. Validacao Final
Apos a correcao, ligar o equipamento em vazio, medir a vibracao radial e
comparar com o valor registrado antes da intervencao.
"""


@pytest.fixture(scope="module")
def base_com_documento_novo():
    settings = load_settings()
    if not (settings.paths.index_dir / "base_conhecimento.json").exists():
        pytest.skip("execute scripts/build_knowledge.py antes")

    destino = settings.paths.docs_dir / NOME_ARQUIVO
    documento = fitz.open()
    pagina = documento.new_page()
    pagina.insert_textbox(fitz.Rect(50, 50, 545, 790), TEXTO, fontsize=10, fontname="helv")
    documento.save(destino)
    documento.close()

    # O cache real e reaproveitado de proposito: extrair para um cache vazio
    # refaria o OCR das 17 paginas do Doc1, treze minutos para testar outra
    # coisa. Apenas o documento novo e extraido de fato.
    cache = settings.paths.knowledge_dir
    cfg = settings.knowledge
    try:
        extraidos = extract_all(settings.paths.docs_dir, cache, dpi=cfg["ocr_dpi"])
        yield BaseConhecimento(
            tamanho_trecho=cfg["chunk_chars"],
            sobreposicao=cfg["chunk_overlap"],
            chars_escopo=cfg["scope_chars"],
            score_minimo_cobertura=cfg["coverage_min_score"],
        ).indexar(extraidos)
    finally:
        destino.unlink(missing_ok=True)
        (cache / f"{destino.stem}.json").unlink(missing_ok=True)


def test_ventoinha_deixa_de_ser_defeito_sem_documentacao(base_com_documento_novo):
    entrada = load_catalog()["faults"]["ventoinha"]
    cobertura = base_com_documento_novo.cobertura(
        entrada["termo_chave"], entrada["termos_busca"], entrada["rotulo"]
    )
    assert cobertura.coberto, "cadastrar o procedimento deveria resolver a recusa"
    assert cobertura.documento == "DocExemploVentoinha"


def test_o_documento_novo_fica_consultavel(base_com_documento_novo):
    trechos = base_com_documento_novo.buscar("como corrigir sujeira nas pas da ventoinha", top_k=3)
    assert trechos
    assert any(t.documento == "DocExemploVentoinha" for t in trechos)


def test_cadastro_nao_muda_a_cobertura_dos_outros_defeitos(base_com_documento_novo):
    """Um documento novo nao pode roubar a cobertura de quem ja estava atendido."""
    catalogo = load_catalog()["faults"]
    for fault in ("rolamento_inner", "desalinhado", "correia", "polia", "cocked_rotor"):
        entrada = catalogo[fault]
        cobertura = base_com_documento_novo.cobertura(
            entrada["termo_chave"], entrada["termos_busca"], entrada["rotulo"]
        )
        assert cobertura.coberto
        assert cobertura.documento != "DocExemploVentoinha"


def test_eccentric_rotor_continua_sem_documentacao(base_com_documento_novo):
    """O outro defeito descoberto continua recusado: um documento resolve um caso."""
    entrada = load_catalog()["faults"]["eccentric_rotor"]
    cobertura = base_com_documento_novo.cobertura(
        entrada["termo_chave"], entrada["termos_busca"], entrada["rotulo"]
    )
    assert not cobertura.coberto
