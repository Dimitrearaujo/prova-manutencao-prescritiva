"""Base de conhecimento documental: indexacao, busca e cobertura.

Duas buscas distintas convivem aqui, de proposito diferente:

- busca de trecho, sobre o corpo inteiro dos documentos, para alimentar a
  resposta com o texto que ensina a corrigir o defeito;
- busca de cobertura, restrita ao ESCOPO de cada documento (titulo e objetivo),
  para decidir se o defeito e coberto por algum procedimento.

A separacao existe porque titulo declara o que o documento TRATA e corpo apenas
MENCIONA. Um procedimento de rolamentos cita acoplamento, polia e correia entre
as causas e os sintomas; decidir cobertura por presenca no corpo faria a falha
desses componentes parecer documentada por um procedimento que nao os trata.

Honestidade sobre o tamanho do efeito NESTE acervo: rodando a mesma regra sobre
o corpo, o mapa dos onze defeitos sai identico - os seis procedimentos sao
monotematicos e nomeiam o componente no titulo, entao nao ha caso onde as duas
regras discordem. A escolha pelo escopo nao corrige um erro observado aqui: ela
protege a propriedade que precisa continuar valendo quando o acervo crescer, que
e justamente o que o cadastro de documento novo permite. Um manual generico, ou
um procedimento longo que discuta varios componentes, quebraria a regra do corpo
e nao quebra a do escopo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from prescritiva.knowledge.chunk import Trecho, dividir
from prescritiva.knowledge.extract import ExtractedDocument
from prescritiva.text import stems, tokenize


_PESO_SECAO_PREFERIDA = 1.8


@dataclass
class DocumentoIndexado:
    nome: str
    arquivo: str
    titulo: str
    escopo: str
    paginas: int
    metodo: str
    n_trechos: int


@dataclass
class TrechoRecuperado:
    documento: str
    secao: str
    texto: str
    score: float


@dataclass
class Cobertura:
    coberto: bool
    documento: str | None
    titulo: str | None
    score: float
    motivo: str


class BaseConhecimento:
    def __init__(self, *, tamanho_trecho: int = 900, sobreposicao: int = 150,
                 chars_escopo: int = 700, score_minimo_cobertura: float = 3.0) -> None:
        self.tamanho_trecho = tamanho_trecho
        self.sobreposicao = sobreposicao
        self.chars_escopo = chars_escopo
        self.score_minimo_cobertura = score_minimo_cobertura
        self.documentos: list[DocumentoIndexado] = []
        self.trechos: list[Trecho] = []
        self._bm25_trechos: BM25Okapi | None = None

    def indexar(self, documentos: list[ExtractedDocument]) -> "BaseConhecimento":
        self.documentos = []
        self.trechos = []
        for documento in documentos:
            trechos = dividir(
                documento.nome,
                documento.texto,
                tamanho=self.tamanho_trecho,
                sobreposicao=self.sobreposicao,
            )
            self.documentos.append(
                DocumentoIndexado(
                    nome=documento.nome,
                    arquivo=documento.arquivo,
                    titulo=documento.titulo,
                    escopo=documento.texto[: self.chars_escopo],
                    paginas=documento.paginas,
                    metodo=documento.metodo,
                    n_trechos=len(trechos),
                )
            )
            self.trechos.extend(trechos)
        self._reconstruir()
        return self

    def _reconstruir(self) -> None:
        # BM25 sobre os 185 trechos funciona bem. Sobre os 6 documentos nao
        # funcionaria: com corpus desse tamanho o IDF de um termo presente em
        # metade dos documentos zera, e "rolamento" desapareceria justamente do
        # documento de rolamentos. Por isso a cobertura usa regra propria.
        self._bm25_trechos = BM25Okapi([tokenize(t.texto) for t in self.trechos])

    def buscar(
        self,
        consulta: str,
        *,
        top_k: int = 4,
        documento: str | None = None,
        preferir_secoes: str | None = None,
    ) -> list[TrechoRecuperado]:
        """Recupera trechos por BM25.

        `preferir_secoes` inclina o resultado para as secoes que ensinam a agir.
        Perguntar "como corrigir desalinhamento angular" casa fortissimo com a
        secao que DEFINE desalinhamento angular, porque e onde os termos se
        repetem, e a secao que ensina a corrigir fica de fora. Para prescrever,
        a definicao nao serve: o tecnico precisa dos passos.
        """
        if not self._bm25_trechos or not self.trechos:
            return []
        scores = self._bm25_trechos.get_scores(tokenize(consulta))
        preferidos = stems(preferir_secoes) if preferir_secoes else set()
        candidatos = [
            (score * (_PESO_SECAO_PREFERIDA if preferidos & stems(trecho.secao) else 1.0), trecho)
            for score, trecho in zip(scores, self.trechos, strict=True)
            if documento is None or trecho.documento == documento
        ]
        candidatos.sort(key=lambda item: item[0], reverse=True)
        return [
            TrechoRecuperado(
                documento=trecho.documento, secao=trecho.secao, texto=trecho.texto, score=float(score)
            )
            for score, trecho in candidatos[:top_k]
            if score > 0
        ]

    def pontuar_cobertura(self, termo_chave: str, termos_busca: str) -> list[tuple[float, DocumentoIndexado]]:
        """Aderencia de cada documento ao defeito, do maior para o menor.

        A regra tem duas partes. O termo-chave e uma exigencia: todos os seus
        radicais precisam estar no escopo do documento, senao o documento nem
        entra na disputa. Os termos de contexto sao um desempate, para escolher
        entre dois procedimentos que citam o mesmo componente.

        Casar no titulo vale mais que casar no objetivo porque o titulo diz do
        que o procedimento TRATA, enquanto o objetivo pode apenas mencionar.
        """
        radicais_chave = stems(termo_chave)
        radicais_contexto = stems(termos_busca)
        pontuados: list[tuple[float, DocumentoIndexado]] = []

        for documento in self.documentos:
            no_titulo = stems(documento.titulo)
            no_escopo = stems(documento.escopo) | no_titulo
            if not radicais_chave <= no_escopo:
                continue
            peso_titulo = len(radicais_chave & no_titulo) / len(radicais_chave)
            peso_contexto = (
                len(radicais_contexto & no_escopo) / len(radicais_contexto)
                if radicais_contexto
                else 0.0
            )
            pontuados.append((2.0 * peso_titulo + peso_contexto, documento))

        pontuados.sort(key=lambda item: item[0], reverse=True)
        return pontuados

    def cobertura(self, termo_chave: str, termos_busca: str, rotulo: str) -> Cobertura:
        if not self.documentos:
            return Cobertura(False, None, None, 0.0, "A base de conhecimento esta vazia.")

        pontuados = self.pontuar_cobertura(termo_chave, termos_busca)
        if not pontuados or pontuados[0][0] < self.score_minimo_cobertura:
            melhor = f" O mais proximo foi {pontuados[0][1].titulo}." if pontuados else ""
            score = pontuados[0][0] if pontuados else 0.0
            return Cobertura(
                coberto=False,
                documento=None,
                titulo=None,
                score=score,
                motivo=(
                    f'Nenhum procedimento cadastrado trata de "{rotulo}".{melhor} '
                    f"Aderencia {score:.2f}, abaixo do minimo de {self.score_minimo_cobertura:.2f}."
                ),
            )

        score, documento = pontuados[0]
        return Cobertura(
            coberto=True,
            documento=documento.nome,
            titulo=documento.titulo,
            score=score,
            motivo=f"Coberto por {documento.arquivo} (aderencia {score:.2f}).",
        )

    def salvar(self, destino: Path) -> None:
        destino.mkdir(parents=True, exist_ok=True)
        (destino / "base_conhecimento.json").write_text(
            json.dumps(
                {
                    "config": {
                        "tamanho_trecho": self.tamanho_trecho,
                        "sobreposicao": self.sobreposicao,
                        "chars_escopo": self.chars_escopo,
                        "score_minimo_cobertura": self.score_minimo_cobertura,
                    },
                    "documentos": [asdict(d) for d in self.documentos],
                    "trechos": [asdict(t) for t in self.trechos],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def carregar(cls, origem: Path) -> "BaseConhecimento":
        dados = json.loads((origem / "base_conhecimento.json").read_text(encoding="utf-8"))
        base = cls(**dados["config"])
        base.documentos = [DocumentoIndexado(**d) for d in dados["documentos"]]
        base.trechos = [Trecho(**t) for t in dados["trechos"]]
        base._reconstruir()
        return base
