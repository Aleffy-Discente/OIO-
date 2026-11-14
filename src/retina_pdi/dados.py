"""Acesso ao manifesto e às imagens públicas de fundo de olho (baixadas por scripts/download_data.py)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data"
MANIFESTO = DATA / "manifest.csv"
PROCESSADOS = DATA / "processed"
FIGURAS = RAIZ / "outputs" / "figures"

CONDICOES = ("normal", "rd", "glaucoma", "dmri")
NOMES_CONDICAO = {"normal": "Normal", "rd": "Retinopatia diabética", "glaucoma": "Glaucoma", "dmri": "DMRI"}


def carregar_manifesto() -> pd.DataFrame:
    """Lê `data/manifest.csv`. Levanta um erro claro se os dados ainda não foram baixados."""
    if not MANIFESTO.exists():
        raise FileNotFoundError(
            f"{MANIFESTO} não existe. Rode `python scripts/download_data.py` na raiz do projeto.")
    return pd.read_csv(MANIFESTO, keep_default_na=False)


def selecionar(condicao: str | None = None, fonte: str | None = None, n: int | None = None,
               seed: int = 0) -> pd.DataFrame:
    """Filtra o manifesto por condição/fonte e (opcionalmente) sorteia `n` linhas de forma reprodutível."""
    df = carregar_manifesto()
    if condicao is not None:
        df = df[df["condicao"] == condicao]
    if fonte is not None:
        df = df[df["fonte"].str.lower() == fonte.lower()]
    if n is not None and len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return df.sort_values("id").reset_index(drop=True)


def _ler_bytes(caminho: str | Path) -> np.ndarray:
    p = Path(caminho)
    if not p.is_absolute():
        p = RAIZ / p
    if not p.exists():
        raise FileNotFoundError(p)
    # np.fromfile + imdecode: cv2.imread não abre caminhos com acentos no Windows ("Módulo 2").
    return np.fromfile(str(p), dtype=np.uint8)


def carregar_rgb(caminho: str | Path) -> np.ndarray:
    """Carrega uma imagem colorida como matriz uint8 H×W×3 em ordem **RGB**.

    (O OpenCV lê em BGR; convertemos aqui para o matplotlib/PIL/TensorFlow verem as cores certas.)
    """
    bgr = cv2.imdecode(_ler_bytes(caminho), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Não foi possível decodificar {caminho}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def carregar_mascara(caminho: str | Path) -> np.ndarray:
    """Carrega uma máscara binária (TIFF/PNG) como matriz booleana H×W."""
    m = cv2.imdecode(_ler_bytes(caminho), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Não foi possível decodificar {caminho}")
    return m > 127


def carregar_registro(linha: pd.Series) -> dict:
    """Carrega imagem (+ máscaras, quando existirem) de uma linha do manifesto."""
    out = {"rgb": carregar_rgb(linha["arquivo"]), "meta": linha}
    if linha.get("mascara_vasos"):
        out["vasos"] = carregar_mascara(linha["mascara_vasos"])
    if linha.get("mascara_fov"):
        out["fov"] = carregar_mascara(linha["mascara_fov"])
    return out


def reduzir(img: np.ndarray, largura_max: int = 1024, interpolacao: int = cv2.INTER_AREA) -> np.ndarray:
    """Reduz mantendo a proporção (INTER_AREA é a melhor escolha para diminuir imagens)."""
    h, w = img.shape[:2]
    if w <= largura_max:
        return img
    f = largura_max / w
    return cv2.resize(img, (largura_max, round(h * f)), interpolation=interpolacao)


def carregar_hrf(largura: int = 1168, condicao: str | None = None, n: int | None = None) -> list[dict]:
    """Carrega imagens do HRF reduzidas para `largura` px, com as máscaras de vasos e FOV reamostradas.

    As máscaras binárias são reduzidas com INTER_AREA e binarizadas em 0,5 (preserva melhor os vasos finos
    do que o vizinho mais próximo). Cada item é um dict: ``rgb``, ``vasos``, ``fov`` (oficial) e ``meta``.
    """
    df = selecionar(condicao=condicao, fonte="hrf")
    if n is not None:
        df = df.head(n)
    saida = []
    for _, linha in df.iterrows():
        reg = carregar_registro(linha)
        rgb = reduzir(reg["rgb"], largura)
        h, w = rgb.shape[:2]
        item = {"rgb": rgb, "meta": linha}
        for chave in ("vasos", "fov"):
            item[chave] = cv2.resize(reg[chave].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= 0.5
        saida.append(item)
    return saida
