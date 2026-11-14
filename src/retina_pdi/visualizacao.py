"""Funções de visualização compartilhadas pelos notebooks."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .dados import FIGURAS


def configurar_estilo() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "image.interpolation": "nearest",
        "font.family": "DejaVu Sans",
    })


def grade(imagens: Sequence[np.ndarray], titulos: Sequence[str] | None = None, colunas: int = 4,
          largura: float = 3.2, cmap: str | None = None, suptitulo: str | None = None):
    """Mostra várias imagens em uma grade sem eixos. Retorna a figura."""
    n = len(imagens)
    linhas = int(np.ceil(n / colunas))
    fig, axs = plt.subplots(linhas, colunas, figsize=(largura * colunas, largura * linhas * 0.85), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for i, img in enumerate(imagens):
        ax = axs.ravel()[i]
        ax.imshow(img, cmap=(cmap or "gray") if img.ndim == 2 else None)
        if titulos is not None:
            ax.set_title(titulos[i])
    if suptitulo:
        fig.suptitle(suptitulo, y=1.0)
    fig.tight_layout()
    return fig


def mostrar(img: np.ndarray, titulo: str | None = None, largura: float = 6, cmap: str | None = None):
    fig, ax = plt.subplots(figsize=(largura, largura * img.shape[0] / img.shape[1]))
    ax.imshow(img, cmap=(cmap or "gray") if img.ndim == 2 else None)
    ax.axis("off")
    if titulo:
        ax.set_title(titulo)
    fig.tight_layout()
    return fig


def sobrepor(rgb: np.ndarray, mascara: np.ndarray, cor: Sequence[int] = (0, 255, 0), alpha: float = 0.6) -> np.ndarray:
    """Pinta `mascara` (bool) sobre `rgb` com transparência `alpha`."""
    out = rgb.astype(np.float32).copy()
    m = mascara.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(cor, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def histogramas_rgb(rgb: np.ndarray, mascara: np.ndarray | None = None, ax=None, bins: int = 64):
    """Histogramas dos canais R, G e B (opcionalmente só dentro de `mascara`)."""
    ax = ax or plt.gca()
    for i, (nome, cor) in enumerate(zip("RGB", ("tab:red", "tab:green", "tab:blue"))):
        canal = rgb[..., i] if mascara is None else rgb[..., i][mascara]
        ax.hist(canal.ravel(), bins=bins, range=(0, 255), color=cor, alpha=0.5, label=nome, density=True)
    ax.set_xlabel("intensidade (0–255)")
    ax.set_ylabel("densidade")
    ax.legend(frameon=False)
    return ax


def salvar_figura(fig, nome: str) -> str:
    """Salva a figura em outputs/figures/<nome>.png e devolve o caminho."""
    FIGURAS.mkdir(parents=True, exist_ok=True)
    destino = FIGURAS / f"{nome}.png"
    fig.savefig(destino, bbox_inches="tight", facecolor="white")
    return str(destino)
