"""Métricas de avaliação de realce/filtragem usadas nos notebooks 03–06."""
from __future__ import annotations

import cv2
import numpy as np

from .qualidade import estimar_ruido


def anel_ao_redor(alvo: np.ndarray, fov: np.ndarray, largura: int = 10, folga: int = 3) -> np.ndarray:
    """Anel de pixels vizinhos ao `alvo` (ex.: vasos) que NÃO são alvo — o "fundo local" para medir contraste."""
    a = alvo.astype(np.uint8)
    externo = cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (folga + largura) + 1,) * 2)).astype(bool)
    interno = cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * folga + 1,) * 2)).astype(bool)
    return externo & ~interno & fov


def cohen_d(canal: np.ndarray, alvo: np.ndarray, fov: np.ndarray, largura_anel: int = 10, folga: int = 3) -> float:
    """Tamanho do efeito de Cohen entre o fundo local e o alvo: (μ_fundo − μ_alvo)/σ_combinado.

    Positivo = alvo mais escuro que o fundo (caso dos vasos no canal verde). Quanto maior |d|, mais separável.
    `largura_anel` e `folga` (px) definem o fundo local; reduza-os em imagens pequenas (vasos de 1–2 px).
    """
    v = canal[alvo & fov].astype(np.float32)
    f = canal[anel_ao_redor(alvo, fov, largura_anel, folga)].astype(np.float32)
    return float((f.mean() - v.mean()) / np.sqrt((f.var() + v.var()) / 2 + 1e-9))


def entropia(canal: np.ndarray, mascara: np.ndarray | None = None) -> float:
    """Entropia de Shannon do histograma (bits) — quanto da faixa de níveis é efetivamente usada."""
    v = canal[mascara] if mascara is not None else canal.ravel()
    h = np.bincount(v.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    h /= h.sum()
    h = h[h > 0]
    return float(-(h * np.log2(h)).sum())


def contraste_rms(canal: np.ndarray, mascara: np.ndarray | None = None) -> float:
    """Contraste RMS = desvio-padrão dos níveis de cinza dentro da máscara."""
    v = canal[mascara] if mascara is not None else canal.ravel()
    return float(v.astype(np.float32).std())


def ruido_sigma(canal: np.ndarray, mascara: np.ndarray) -> float:
    """σ do ruído estimado (Immerkaer) dentro da máscara."""
    return estimar_ruido(canal, mascara)
