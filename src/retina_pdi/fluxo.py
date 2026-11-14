"""Fluxo completo de uma captura: controle de qualidade → FOV → tensor pronto para o modelo.

É o "contrato" entre a aquisição (Frente F2) e a inferência (Frente F3): a captura só segue para os
modelos de RD, Glaucoma e DMRI se passar no controle de qualidade.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .pipeline import mascara_fov, preparar_para_inferencia
from .qualidade import LIMIARES_PADRAO, Limiares, ResultadoQualidade, avaliar_qualidade


@dataclass
class ResultadoFluxo:
    aprovada: bool
    tensor: np.ndarray | None
    qualidade: ResultadoQualidade
    tempos_ms: dict[str, float] = field(default_factory=dict)


def processar_captura(rgb: np.ndarray, tamanho: int = 224, realce: str | None = None, normalizacao: str = "0_255",
                      limiares: Limiares = LIMIARES_PADRAO, exigir_qualidade: bool = True) -> ResultadoFluxo:
    """Executa o fluxo. Com `exigir_qualidade=True`, capturas reprovadas devolvem `tensor=None`
    (o software de aquisição deve pedir uma nova captura ao operador, com os motivos em `qualidade.motivos`)."""
    t = {}
    t0 = time.perf_counter()
    fov = mascara_fov(rgb)
    t["fov"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    q = avaliar_qualidade(rgb, limiares, fov)
    t["qualidade"] = (time.perf_counter() - t0) * 1e3

    if exigir_qualidade and not q.adequada:
        t["total"] = sum(t.values())
        return ResultadoFluxo(False, None, q, t)

    t0 = time.perf_counter()
    x = preparar_para_inferencia(rgb, tamanho, realce, normalizacao, fov=fov)
    t["preparacao"] = (time.perf_counter() - t0) * 1e3
    t["total"] = sum(t.values())
    return ResultadoFluxo(q.adequada, x, q, t)
