"""Integracao com o ambiente industrial.

Separa o que fala com o mundo de fora do que decide. O motor de diagnostico nao
sabe se o evento chegou por HTTP, por fila ou por arquivo, e nao sabe se o
resultado sera gravado; quem sabe disso mora aqui.
"""
