"""Degradações sintéticas de retinografias (foco, iluminação, ruído).

Servem para (i) mostrar aos alunos o que cada defeito de aquisição faz na matriz de pixels (notebook 01)
e (ii) validar avaliadores de qualidade com verdade conhecida (notebook 05): sabemos exatamente
qual defeito foi aplicado e com que intensidade.

O parâmetro `nivel` vai de 0 (sem defeito) a 1 (severo). Escalas de comprimento são relativas ao tamanho
da imagem, então o mesmo `nivel` produz um defeito visualmente equivalente em 512 ou em 3500 px.
"""
from __future__ import annotations

import cv2
import numpy as np

from .pipeline import mascara_fov

TIPOS = ("desfoque", "movimento", "escuro", "claro", "desigual", "ruido")
DESCRICAO = {
    "desfoque": "fora de foco (desfoque gaussiano)",
    "movimento": "tremor / movimento (desfoque direcional)",
    "escuro": "subexposição (pouca luz)",
    "claro": "superexposição (luz em excesso)",
    "desigual": "iluminação desigual (sombra / vinheta)",
    "ruido": "ruído do sensor",
}


def _para_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def desfocar(rgb: np.ndarray, nivel: float) -> np.ndarray:
    """Desfoque gaussiano; σ = nivel · 1,2 % do menor lado."""
    sigma = max(1e-3, nivel * 0.012 * min(rgb.shape[:2]))
    return cv2.GaussianBlur(rgb, (0, 0), sigma)


def desfoque_movimento(rgb: np.ndarray, nivel: float, angulo: float = 20.0) -> np.ndarray:
    """Desfoque direcional (kernel linear); comprimento = nivel · 3 % do menor lado."""
    comp = max(1, int(round(nivel * 0.03 * min(rgb.shape[:2]))) | 1)
    k = np.zeros((comp, comp), np.float32)
    k[comp // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((comp / 2 - 0.5, comp / 2 - 0.5), angulo, 1.0)
    k = cv2.warpAffine(k, rot, (comp, comp))
    return cv2.filter2D(rgb, -1, k / max(k.sum(), 1e-6))


def escurecer(rgb: np.ndarray, nivel: float) -> np.ndarray:
    """Multiplica a imagem por (1 − 0,85·nivel): nivel 1 mantém 15 % do brilho."""
    return _para_uint8(rgb.astype(np.float32) * (1 - 0.85 * nivel))


def clarear(rgb: np.ndarray, nivel: float) -> np.ndarray:
    """Superexposição: ganho 1 + 2,5·nivel, com saturação em 255 (perde-se informação nas regiões claras)."""
    return _para_uint8(rgb.astype(np.float32) * (1 + 2.5 * nivel))


def iluminacao_desigual(rgb: np.ndarray, nivel: float, angulo: float = 30.0, fov: np.ndarray | None = None) -> np.ndarray:
    """Rampa de iluminação linear: um lado fica até `nivel·85 %` mais escuro que o outro."""
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.deg2rad(angulo)
    proj = (xx - w / 2) * np.cos(t) + (yy - h / 2) * np.sin(t)
    proj /= max(np.abs(proj).max(), 1e-6)  # −1 … 1
    ganho = 1 - 0.85 * nivel * (proj + 1) / 2
    return _para_uint8(rgb.astype(np.float32) * ganho[..., None])


def ruido(rgb: np.ndarray, nivel: float, seed: int = 0, fov: np.ndarray | None = None) -> np.ndarray:
    """Ruído gaussiano aditivo (σ = nivel · 40 níveis) aplicado somente dentro do FOV."""
    rng = np.random.default_rng(seed)
    fov = mascara_fov(rgb) if fov is None else fov
    r = rng.normal(0, nivel * 40, rgb.shape).astype(np.float32)
    r[~fov] = 0
    return _para_uint8(rgb.astype(np.float32) + r)


def aplicar(rgb: np.ndarray, tipo: str, nivel: float, seed: int = 0) -> np.ndarray:
    """Aplica a degradação `tipo` com intensidade `nivel` ∈ [0, 1]."""
    if tipo == "desfoque":
        return desfocar(rgb, nivel)
    if tipo == "movimento":
        return desfoque_movimento(rgb, nivel, angulo=float(np.random.default_rng(seed).uniform(0, 180)))
    if tipo == "escuro":
        return escurecer(rgb, nivel)
    if tipo == "claro":
        return clarear(rgb, nivel)
    if tipo == "desigual":
        return iluminacao_desigual(rgb, nivel, angulo=float(np.random.default_rng(seed).uniform(0, 360)))
    if tipo == "ruido":
        return ruido(rgb, nivel, seed)
    raise ValueError(f"tipo desconhecido: {tipo!r}; use um de {TIPOS}")
