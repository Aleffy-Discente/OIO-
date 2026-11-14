"""Avaliação automática da qualidade de retinografias (antes da inferência).

Traduz os critérios visuais do Módulo 1 (foco, iluminação, centralização, artefatos) em propriedades
computáveis da matriz de pixels — ver Aula 1 do Módulo 2. Todas as métricas são calculadas numa versão
**padronizada** da imagem (recorte do FOV, quadrada, 512×512): sem isso, nitidez e contraste dependeriam
da resolução da câmera e não da qualidade da captura.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .pipeline import erodir_fov, gaussiano_mascarado, mascara_fov, recortar_fov

LADO_PADRAO = 512


def padronizar(rgb: np.ndarray, fov: np.ndarray | None = None, lado: int = LADO_PADRAO):
    """Recorta o FOV, completa até ficar quadrada e reduz para `lado`×`lado` (INTER_AREA).

    Devolve (rgb_padrao, fov_padrao, info) — `info` traz a geometria original do FOV.
    """
    fov = mascara_fov(rgb) if fov is None else fov
    h, w = fov.shape
    ys, xs = np.where(fov)
    if len(ys) == 0:
        raise ValueError("FOV vazio: imagem inteiramente escura?")
    bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    info = {
        "cobertura_fov": float(fov.mean()),
        "razao_bbox": float(bw / bh),
        "preenchimento_bbox": float(fov.sum() / (bh * bw)),
        "desvio_centro": float(np.hypot(xs.mean() - w / 2, ys.mean() - h / 2) / min(h, w)),
        "toca_borda": bool(ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1),
    }
    img, m = recortar_fov(rgb, fov, quadrado=True)
    img = cv2.resize(img, (lado, lado), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m.astype(np.uint8), (lado, lado), interpolation=cv2.INTER_NEAREST).astype(bool)
    return img, m, info


def estimar_ruido(canal: np.ndarray, mascara: np.ndarray) -> float:
    """Desvio-padrão do ruído pelo método de Immerkaer (1996): resposta a um Laplaciano "cego" a estruturas suaves.

    σ ≈ √(π/2)/6 · média(|I ∗ N|), N = [[1,−2,1],[−2,4,−2],[1,−2,1]] — bordas e vasos contribuem pouco;
    ruído branco contribui muito.
    """
    n = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(canal.astype(np.float32), -1, n))
    return float(np.sqrt(np.pi / 2) / 6 * r[mascara].mean())


def metricas_qualidade(rgb: np.ndarray, fov: np.ndarray | None = None) -> dict[str, float]:
    """Calcula as métricas de qualidade (todas dentro do FOV, na imagem padronizada 512×512).

    * foco:        ``nitidez`` (variância do Laplaciano do verde suavizado) e ``nitidez_rel`` (idem,
                   dividida pela variância do próprio canal, para não punir imagens de baixo contraste);
                   ``tenengrad`` (energia do gradiente de Sobel).
    * iluminação:  ``brilho`` (média da luminância), ``p_saturado`` / ``p_escuro`` (fração de pixels
                   saturados ou muito escuros) e ``desigualdade`` (variação da iluminação de fundo).
    * contraste:   ``contraste`` (desvio-padrão do canal verde), ``entropia`` (bits) e ``ruido`` (σ estimado, Immerkaer).
    * enquadramento: ``cobertura_fov``, ``preenchimento_bbox`` (≈0,79 = círculo inteiro; →1 = FOV cortado),
                   ``desvio_centro`` e ``toca_borda``.
    """
    img, m, info = padronizar(rgb, fov)
    fe = erodir_fov(m, 6)  # evita a borda do círculo, que tem gradiente alto em qualquer imagem
    verde = img[..., 1]
    g = cv2.GaussianBlur(verde, (0, 0), 1.0).astype(np.float32)  # σ=1: reduz a influência do ruído no Laplaciano
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    var_lap = float(lap[fe].var())
    var_verde = float(verde[fe].astype(np.float32).var())
    sx, sy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    lum = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fundo = gaussiano_mascarado(lum, m, 0.12 * LADO_PADRAO)
    hist = np.bincount(verde[fe], minlength=256).astype(np.float64)
    hist /= hist.sum()
    nz = hist[hist > 0]
    return {
        "nitidez": var_lap,
        "nitidez_rel": var_lap / max(var_verde, 1e-6),
        "tenengrad": float(np.sqrt(sx ** 2 + sy ** 2)[fe].mean()),
        "brilho": float(lum[fe].mean()),
        "p_saturado": float((img.max(axis=2)[fe] >= 250).mean()),
        "p_escuro": float((lum[fe] < 25).mean()),
        "desigualdade": float(fundo[fe].std() / max(fundo[fe].mean(), 1e-6)),
        "ruido": estimar_ruido(verde, fe),
        "contraste": float(np.sqrt(var_verde)),
        "entropia": float(-(nz * np.log2(nz)).sum()),
        **info,
    }


@dataclass(frozen=True)
class Limiares:
    """Limiares das regras de qualidade.

    Os valores padrão foram calibrados no notebook 05 com HRF + ODIR + RFMiD (percentis das imagens originais e
    degradações sintéticas). **Não são universais**: nitidez e contraste dependem da câmera. No dispositivo
    final (Pi Camera Module 3), recalibre com capturas boas do próprio retinógrafo.
    """
    nitidez_rel_min: float = 0.035       # abaixo disso: fora de foco / movimento
    ruido_max: float = 2.0               # σ estimado (níveis de 0-255) acima disso: ruído excessivo
    brilho_min: float = 45.0             # luminância média no FOV
    brilho_max: float = 175.0
    p_saturado_max: float = 0.10         # fração do FOV com pixels ≥ 250
    p_escuro_max: float = 0.03           # fração do FOV com luminância < 25
    desigualdade_max: float = 0.25       # desvio-padrão relativo da iluminação de fundo
    contraste_min: float = 9.0           # desvio-padrão do canal verde
    desvio_centro_max: float = 0.05      # deslocamento do FOV em relação ao centro do quadro (fração do menor lado)


LIMIARES_PADRAO = Limiares()


@dataclass
class ResultadoQualidade:
    adequada: bool
    motivos: list[str]
    metricas: dict[str, float] = field(default_factory=dict)


def julgar_metricas(m: dict[str, float], limiares: Limiares = LIMIARES_PADRAO) -> ResultadoQualidade:
    """Aplica as regras de qualidade a métricas JÁ calculadas (permite testar limiares sem reprocessar imagens)."""
    motivos = []
    if m["nitidez_rel"] < limiares.nitidez_rel_min:
        motivos.append("fora de foco / borrada")
    if m["ruido"] > limiares.ruido_max:
        motivos.append("ruído excessivo")
    if m["brilho"] < limiares.brilho_min:
        motivos.append("subexposta (escura)")
    if m["brilho"] > limiares.brilho_max or m["p_saturado"] > limiares.p_saturado_max:
        motivos.append("superexposta (saturada)")
    if m["p_escuro"] > limiares.p_escuro_max:
        motivos.append("grandes regiões escuras")
    if m["desigualdade"] > limiares.desigualdade_max:
        motivos.append("iluminação desigual")
    if m["contraste"] < limiares.contraste_min:
        motivos.append("baixo contraste")
    if m["desvio_centro"] > limiares.desvio_centro_max:
        motivos.append("retina descentralizada no quadro")
    return ResultadoQualidade(not motivos, motivos, dict(m))


def avaliar_qualidade(rgb: np.ndarray, limiares: Limiares = LIMIARES_PADRAO, fov: np.ndarray | None = None
                      ) -> ResultadoQualidade:
    """Calcula as métricas da imagem e aplica as regras; `motivos` explica cada reprovação."""
    return julgar_metricas(metricas_qualidade(rgb, fov), limiares)


def calibrar_limiares(metricas, pct: float = 2.0, folga: float = 0.85) -> Limiares:
    """Deriva `Limiares` das métricas de capturas **boas do próprio dispositivo** (DataFrame de `metricas_qualidade`).

    Cada limiar é o percentil extremo (`pct`) da distribuição, afrouxado pelo fator `folga` (< 1): mínimos são
    multiplicados e máximos divididos por ele, para não reprovar capturas boas que só estejam no limite.
    Pisos evitam limiares degenerados quando o conjunto tem pouca variação (ex.: nenhuma saturação).
    """
    def p(col: str, q: float) -> float:
        return float(np.percentile(np.asarray(metricas[col], dtype=float), q))

    return Limiares(
        nitidez_rel_min=p("nitidez_rel", pct) * folga,
        ruido_max=p("ruido", 100 - pct) / folga,
        brilho_min=p("brilho", pct) * folga,
        brilho_max=min(255.0, p("brilho", 100 - pct) / folga),
        p_saturado_max=max(p("p_saturado", 100 - pct) / folga, 0.02),
        p_escuro_max=max(p("p_escuro", 100 - pct) / folga, 0.01),
        desigualdade_max=p("desigualdade", 100 - pct) / folga,
        contraste_min=p("contraste", pct) * folga,
        desvio_centro_max=max(p("desvio_centro", 100 - pct) / folga, 0.02),
    )


MENSAGENS_OPERADOR = {
    "fora de foco / borrada": "Imagem borrada: ajuste o foco e mantenha o aparelho firme.",
    "ruído excessivo": "Muito ruído: aumente a iluminação em vez do ganho da câmera.",
    "subexposta (escura)": "Imagem escura: aumente a intensidade do LED ou o tempo de exposição.",
    "superexposta (saturada)": "Imagem estourada: reduza a intensidade do LED ou o tempo de exposição.",
    "grandes regiões escuras": "Regiões escuras (pálpebra/cílios?): peça ao paciente para abrir mais o olho e realinhe.",
    "iluminação desigual": "Iluminação desigual: realinhe o aparelho com a pupila para evitar sombras.",
    "baixo contraste": "Baixo contraste: verifique foco e opacidades (ex.: catarata) e repita a captura.",
    "retina descentralizada no quadro": "Retina fora do centro: recentralize o campo de visão.",
}


def mensagens_para_operador(res: ResultadoQualidade) -> list[str]:
    """Traduz os motivos de reprovação em instruções curtas para quem opera o retinógrafo."""
    return [MENSAGENS_OPERADOR.get(m, m) for m in res.motivos]
