"""Adaptador para um modelo de linguagem local servido pelo Ollama.

Local por exigencia do enunciado: a inferencia precisa caber numa estacao de
trabalho comercial, e o dado de chao de fabrica nao sai da planta.
"""

from __future__ import annotations

import requests


class OllamaGerador:
    def __init__(
        self,
        modelo: str,
        base_url: str = "http://localhost:11434",
        temperatura: float = 0.1,
        timeout_s: int = 300,
        max_tokens: int = 600,
        contexto: int = 4096,
    ) -> None:
        self.modelo = modelo
        self.base_url = base_url.rstrip("/")
        self.temperatura = temperatura
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.contexto = contexto
        self.nome = f"ollama:{modelo}"

    def disponivel(self) -> bool:
        try:
            resposta = requests.get(f"{self.base_url}/api/tags", timeout=3)
            resposta.raise_for_status()
            instalados = {m["name"].split(":")[0] for m in resposta.json().get("models", [])}
            return self.modelo.split(":")[0] in instalados
        except requests.RequestException:
            return False

    def gerar(self, instrucao: str, pergunta: str) -> str:
        resposta = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.modelo,
                "stream": False,
                "options": {
                    "temperature": self.temperatura,
                    # Teto de tokens: numa estacao sem GPU cada token custa tempo
                    # real, e uma resposta que divaga vira minutos de espera para
                    # o tecnico. Limitar o tamanho e limitar a latencia.
                    "num_predict": self.max_tokens,
                    "num_ctx": self.contexto,
                },
                "messages": [
                    {"role": "system", "content": instrucao},
                    {"role": "user", "content": pergunta},
                ],
            },
            timeout=self.timeout_s,
        )
        resposta.raise_for_status()
        return resposta.json()["message"]["content"].strip()
