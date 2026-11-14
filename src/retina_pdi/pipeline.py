"""Pipeline clássico de PDI para retinografias.

As funções aqui foram construídas passo a passo nos notebooks 02–06; os notebooks mostram o raciocínio
e este módulo guarda a versão final, testada (`tests/`) e reutilizável pela Frente F2 (software de aquisição
e pré-processamento embarcado no Raspberry Pi).

Convenções: imagens são `uint8` H×W×3 em **RGB**; máscaras são `bool` H×W.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology

# =========================================================================== campo de visão (FOV)


def mascara_fov(rgb: np.ndarray, limiar: float | None = None, largura_trab: int = 512) -> np.ndarray:
    """Máscara do campo de visão (a região circular da retina, sem a moldura preta).

    1. usa o canal de maior valor por pixel (robusto a imagens muito vermelhas ou esverdeadas);
    2. limiariza — por padrão, 12 % do percentil 99, limitado a [8, 30];
    3. abertura morfológica remove marcas/etiquetas coladas à borda; fica o maior componente conexo;
    4. preenche buracos (vasos e lesões escuras dentro do círculo).

    O FOV é uma estrutura "grande e lisa": as etapas 1–4 rodam numa versão com no máximo `largura_trab`
    pixels de largura e a máscara é ampliada de volta (vizinho mais próximo) — ~30× mais rápido em 12 MP.
    """
    h, w = rgb.shape[:2]
    if w > largura_trab:
        pequena = cv2.resize(rgb, (largura_trab, round(h * largura_trab / w)), interpolation=cv2.INTER_AREA)
        m = mascara_fov(pequena, limiar, largura_trab=largura_trab)
        return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    cinza = cv2.GaussianBlur(rgb.max(axis=2), (0, 0), 2)
    if limiar is None:
        limiar = float(np.clip(0.12 * np.percentile(cinza, 99), 8, 30))
    m = (cinza > limiar).astype(np.uint8)
    k = max(3, int(0.02 * min(m.shape)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, rot, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:  # imagem inteiramente escura: devolve máscara vazia
        return np.zeros(m.shape, bool)
    maior = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ndi.binary_fill_holes(rot == maior)


def caixa_fov(fov: np.ndarray) -> tuple[int, int, int, int]:
    """Caixa delimitadora (y0, y1, x0, x1) do FOV, com y1/x1 exclusivos."""
    ys, xs = np.where(fov)
    if len(ys) == 0:
        return 0, fov.shape[0], 0, fov.shape[1]
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def recortar_fov(rgb: np.ndarray, fov: np.ndarray | None = None, quadrado: bool = True):
    """Recorta a imagem no FOV. Com `quadrado=True`, completa com preto (padding centrado) até ficar quadrada,
    preservando toda a retina sem distorcer a proporção. Devolve (rgb_recortada, fov_recortado)."""
    fov = mascara_fov(rgb) if fov is None else fov
    y0, y1, x0, x1 = caixa_fov(fov)
    img, m = rgb[y0:y1, x0:x1].copy(), fov[y0:y1, x0:x1]
    img[~m] = 0
    if quadrado:
        h, w = img.shape[:2]
        lado = max(h, w)
        pad_y, pad_x = (lado - h) // 2, (lado - w) // 2
        img = cv2.copyMakeBorder(img, pad_y, lado - h - pad_y, pad_x, lado - w - pad_x, cv2.BORDER_CONSTANT, value=0)
        m = cv2.copyMakeBorder(m.astype(np.uint8), pad_y, lado - h - pad_y, pad_x, lado - w - pad_x,
                               cv2.BORDER_CONSTANT, value=0).astype(bool)
    return img, m


# =========================================================================== filtragem e realce


def gaussiano_mascarado(canal: np.ndarray, fov: np.ndarray, sigma: float) -> np.ndarray:
    """Suavização gaussiana que ignora os pixels fora do FOV (convolução normalizada).

    Sem isso, o preto da moldura "vaza" para dentro da borda da retina e escurece o resultado.
    """
    c = canal.astype(np.float32) * fov
    num = cv2.GaussianBlur(c, (0, 0), sigma)
    den = cv2.GaussianBlur(fov.astype(np.float32), (0, 0), sigma)
    return num / np.maximum(den, 1e-6)


def clahe(canal: np.ndarray, clip: float = 2.0, grade: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE em um canal uint8. `clip` limita a amplificação de contraste; `grade` é o nº de blocos (linhas, colunas)."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=grade).apply(canal)


def clahe_lab(rgb: np.ndarray, clip: float = 2.0, grade: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE só na luminância (L do LAB): realça contraste sem alterar a cromaticidade."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = clahe(lab[..., 0], clip, grade)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def corrigir_iluminacao(canal: np.ndarray, fov: np.ndarray, sigma_rel: float = 0.05) -> np.ndarray:
    """Remove o gradiente lento de iluminação: canal − fundo (gaussiano largo) + média do FOV.

    `sigma_rel` é o desvio do fundo em fração do menor lado da imagem.
    """
    sigma = sigma_rel * min(canal.shape)
    fundo = gaussiano_mascarado(canal, fov, sigma)
    media = float(canal[fov].mean())
    out = canal.astype(np.float32) - fundo + media
    out[~fov] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def realce_ben_graham(rgb: np.ndarray, fov: np.ndarray | None = None, sigma_rel: float = 0.05) -> np.ndarray:
    """Realce popular em competições de RD (Kaggle/APTOS): 4·(I − gaussiana(I)) + 128, dentro do FOV."""
    fov = mascara_fov(rgb) if fov is None else fov
    sigma = sigma_rel * min(rgb.shape[:2])
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        fundo = gaussiano_mascarado(rgb[..., c], fov, sigma)
        out[..., c] = 4.0 * (rgb[..., c].astype(np.float32) - fundo) + 128.0
    out[~fov] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================================================================== segmentação


def remover_componentes_pequenos(binaria: np.ndarray, area_min: int) -> np.ndarray:
    """Remove componentes conexos (8-conectividade) com menos de `area_min` pixels."""
    n, rot, stats, _ = cv2.connectedComponentsWithStats(binaria.astype(np.uint8), connectivity=8)
    manter = 1 + np.where(stats[1:, cv2.CC_STAT_AREA] >= area_min)[0]
    return np.isin(rot, manter)


def erodir_fov(fov: np.ndarray, raio: int) -> np.ndarray:
    """Erosão do FOV — descarta a faixa da borda, onde filtros de crista respondem ao contorno do círculo."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * raio + 1, 2 * raio + 1))
    return cv2.erode(fov.astype(np.uint8), k).astype(bool)


def preencher_fora_do_fov(canal: np.ndarray, fov: np.ndarray, sigma_rel: float = 0.03) -> np.ndarray:
    """Substitui o preto fora do FOV pela média local, evitando uma 'borda' artificial no filtro."""
    fundo = gaussiano_mascarado(canal, fov, sigma_rel * min(canal.shape))
    return np.where(fov, canal, fundo).astype(np.uint8)


def resposta_vasos(rgb: np.ndarray, fov: np.ndarray | None = None,
                   sigmas: tuple[float, ...] = (1.5, 2.5, 3.5)) -> np.ndarray:
    """Mapa contínuo de "vesselness" (filtro de Frangi multi-escala) sobre o canal verde.

    Passos: canal verde → preenche fora do FOV → corrige iluminação → Frangi com cristas escuras.
    `sigmas` estão em pixels da imagem de entrada: para imagens grandes reduza a imagem antes
    (o notebook 04 trabalha em ~1200 px de largura).
    """
    fov = mascara_fov(rgb) if fov is None else fov
    g = preencher_fora_do_fov(rgb[..., 1], fov)
    g = corrigir_iluminacao(g, fov, 0.05)
    return filters.frangi(g.astype(np.float32) / 255, sigmas=sigmas, black_ridges=True, beta=0.5, gamma=None
                          ).astype(np.float32)


def segmentar_vasos(rgb: np.ndarray, fov: np.ndarray | None = None, sigmas: tuple[float, ...] = (1.5, 2.5, 3.5),
                    pct_baixo: float = 88.0, pct_alto: float = 95.0, area_min_rel: float = 3e-4) -> np.ndarray:
    """Segmenta a árvore vascular: Frangi + limiar por histerese (percentis do FOV) + limpeza de ruído.

    No HRF (1168 px de largura) obtém Dice ≈ 0,75 contra a segmentação manual.
    """
    fov = mascara_fov(rgb) if fov is None else fov
    resp = resposta_vasos(rgb, fov, sigmas)
    fe = erodir_fov(fov, max(3, int(0.01 * min(fov.shape))))
    v = resp[fe]
    b = filters.apply_hysteresis_threshold(resp, np.percentile(v, pct_baixo), np.percentile(v, pct_alto)) & fe
    return remover_componentes_pequenos(b, int(area_min_rel * b.size))


def metricas_segmentacao(pred: np.ndarray, ref: np.ndarray, fov: np.ndarray | None = None) -> dict[str, float]:
    """Sensibilidade, especificidade, acurácia, precisão, Dice e IoU (somente dentro do FOV, se fornecido)."""
    if fov is not None:
        pred, ref = pred[fov], ref[fov]
    p, r = pred.astype(bool).ravel(), ref.astype(bool).ravel()
    tp, tn = int((p & r).sum()), int((~p & ~r).sum())
    fp, fn = int((p & ~r).sum()), int((~p & r).sum())
    d = lambda a, b: a / b if b else 0.0  # noqa: E731
    return {"sensibilidade": d(tp, tp + fn), "especificidade": d(tn, tn + fp), "acuracia": d(tp + tn, p.size),
            "precisao": d(tp, tp + fp), "dice": d(2 * tp, 2 * tp + fp + fn), "iou": d(tp, tp + fp + fn)}


@dataclass
class DiscoOptico:
    """Resultado da localização do disco óptico (coordenadas em pixels da imagem analisada)."""
    candidato: tuple[int, int]            # (y, x) do máximo de brilho suavizado
    centro: tuple[float, float] | None    # (y, x) do centroide da região segmentada
    raio: float | None                    # raio do círculo de mesma área
    mascara: np.ndarray | None = field(default=None, repr=False)


def localizar_disco_optico(rgb: np.ndarray, fov: np.ndarray | None = None) -> DiscoOptico:
    """Localiza e segmenta grosseiramente o disco óptico (a região mais clara da retina).

    1. brilho = (R+G)/2; um fechamento morfológico "apaga" os vasos escuros que cruzam o disco;
    2. suaviza (dentro do FOV) e toma o máximo como candidato — longe da borda, onde há reflexos;
    3. Otsu numa janela em torno do candidato e componente conexo mais próximo = disco;
    4. se o raio for implausível (> 8,5 % do diâmetro do FOV — o disco "vazou" para o halo claro ao redor),
       repete o Otsu apenas entre os pixels mais claros, até 3 vezes.

    É um método didático: falha com exsudatos muito grandes ou reflexos intensos (veja o notebook 04).
    """
    fov = mascara_fov(rgb) if fov is None else fov
    h, w = fov.shape
    lado = min(h, w)
    y0f, y1f, x0f, x1f = caixa_fov(fov)
    r_max = 0.085 * max(y1f - y0f, x1f - x0f)
    brilho = (rgb[..., 0].astype(np.float32) + rgb[..., 1]) / 2
    k = int(0.05 * lado) | 1
    fechado = cv2.morphologyEx(brilho, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    suave = gaussiano_mascarado(fechado, fov, 0.03 * lado)
    fe = erodir_fov(fov, int(0.05 * lado))
    cy, cx = np.unravel_index(np.argmax(np.where(fe, suave, 0)), suave.shape)
    r0 = int(0.14 * lado)
    y0, y1, x0, x1 = max(0, cy - r0), min(h, cy + r0), max(0, cx - r0), min(w, cx + r0)
    jan = fechado[y0:y1, x0:x1].astype(np.uint8)
    limiar, _ = cv2.threshold(jan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    comp = None
    for _ in range(4):
        b = ((jan > limiar) & fov[y0:y1, x0:x1]).astype(np.uint8)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n, rot, st, cen = cv2.connectedComponentsWithStats(b)
        cands = [(np.hypot(cen[i][0] - (cx - x0), cen[i][1] - (cy - y0)) - 0.001 * st[i, cv2.CC_STAT_AREA], i)
                 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 0.002 * lado * lado]
        if not cands:
            break
        comp = rot == min(cands)[1]
        if np.sqrt(comp.sum() / np.pi) <= r_max:
            break
        acima = jan[jan > limiar]                      # região grande demais: sobe o limiar entre os pixels mais claros
        if acima.size < 50:
            break
        limiar, _ = cv2.threshold(acima.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if comp is None:
        return DiscoOptico((int(cy), int(cx)), None, None, None)
    ys, xs = np.where(comp)
    mascara = np.zeros((h, w), bool)
    mascara[y0:y1, x0:x1] = comp
    return DiscoOptico((int(cy), int(cx)), (float(ys.mean() + y0), float(xs.mean() + x0)),
                       float(np.sqrt(comp.sum() / np.pi)), mascara)


# =========================================================================== preparação para inferência

NORMALIZACOES = {
    "0_255": "float32 em [0, 255] — modelos Keras com pré-processamento embutido (ex.: MobileNetV3, EfficientNet)",
    "0_1": "float32 em [0, 1]",
    "menos1_1": "float32 em [−1, 1] — convenção do EfficientNet-Lite e da família MobileNet clássica",
    "imagenet": "float32, (x/255 − média)/desvio com estatísticas do ImageNet",
    "uint8": "uint8 [0, 255] — entrada de modelos TFLite totalmente quantizados (int8/uint8)",
}
_MEDIA_IMAGENET = np.array([0.485, 0.456, 0.406], np.float32)
_DESVIO_IMAGENET = np.array([0.229, 0.224, 0.225], np.float32)
REALCES = (None, "clahe", "ben_graham")


def normalizar(rgb: np.ndarray, modo: str) -> np.ndarray:
    """Converte uma imagem uint8 RGB para a escala numérica esperada pelo modelo (ver `NORMALIZACOES`)."""
    if modo not in NORMALIZACOES:
        raise ValueError(f"normalização desconhecida: {modo!r}; use uma de {list(NORMALIZACOES)}")
    if modo == "uint8":
        return rgb.astype(np.uint8)
    x = rgb.astype(np.float32)
    if modo == "0_255":
        return x
    if modo == "0_1":
        return x / 255.0
    if modo == "menos1_1":
        return x / 127.5 - 1.0
    return (x / 255.0 - _MEDIA_IMAGENET) / _DESVIO_IMAGENET


def preparar_para_inferencia(rgb: np.ndarray, tamanho: int = 224, realce: str | None = None,
                             normalizacao: str = "0_255", fov: np.ndarray | None = None,
                             recortar: bool = True) -> np.ndarray:
    """Transforma uma retinografia bruta no tensor de entrada de um modelo (1, tamanho, tamanho, 3).

    Ordem das etapas — **deve ser idêntica à usada no treinamento** (defina-a com a Frente F3):

    1. recorte do FOV e completamento até ficar quadrada (sem distorcer a proporção);
    2. redimensionamento para `tamanho`×`tamanho` (INTER_AREA ao reduzir, INTER_CUBIC ao ampliar);
    3. realce opcional: ``"clahe"`` (na luminância) ou ``"ben_graham"``; a moldura volta a ficar preta;
    4. conversão de escala numérica (`normalizacao`) e inclusão da dimensão de lote (NHWC).
    """
    if realce not in REALCES:
        raise ValueError(f"realce desconhecido: {realce!r}; use um de {REALCES}")
    if recortar:
        img, m = recortar_fov(rgb, fov, quadrado=True)
    else:
        img, m = rgb, (mascara_fov(rgb) if fov is None else fov)
    interp = cv2.INTER_AREA if img.shape[0] >= tamanho else cv2.INTER_CUBIC
    img = cv2.resize(img, (tamanho, tamanho), interpolation=interp)
    m = cv2.resize(m.astype(np.uint8), (tamanho, tamanho), interpolation=cv2.INTER_NEAREST).astype(bool)
    if realce == "clahe":
        img = clahe_lab(img)
    elif realce == "ben_graham":
        img = realce_ben_graham(img, m)
    img = img.copy()
    img[~m] = 0
    return normalizar(img, normalizacao)[None, ...]
