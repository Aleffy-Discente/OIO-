#!/usr/bin/env python
"""Baixa as imagens públicas de fundo de olho usadas nos notebooks do Módulo 02.

Fontes (todas abertas, sem cadastro):

* HRF   — High-Resolution Fundus (FAU Erlangen). Saudável, RD e glaucoma, 3504x2336,
          com segmentação manual de vasos e máscara do campo de visão. CC BY 4.0.
* ODIR  — Ocular Disease Intelligent Recognition (ODIR-5K), 512x512, rótulos por doença
          (normal, diabetes/RD, glaucoma, DMRI). Espelho no Hugging Face Hub.
* RFMiD — Retinal Fundus Multi-disease Image Dataset (Pachade et al., 2021), 2144x1424,
          rótulos multi-doença (RD, DMRI, escavação do disco...). CC BY 4.0.

Só o necessário é baixado: para os datasets do Hub lemos o rodapé do Parquet, a coluna de
rótulos e apenas os *row groups* que contêm as imagens escolhidas (HTTP Range).

Uso:
    python scripts/download_data.py              # HRF + ODIR + RFMiD (~165 MB)
    python scripts/download_data.py --only hrf   # só uma fonte
    python scripts/download_data.py --force      # rebaixa tudo
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
CACHE = RAIZ / "data" / "cache"
MANIFESTO = RAIZ / "data" / "manifest.csv"

HRF_BASE = "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images"
UA = {"User-Agent": "RetinaEmFoco-IFPE/1.0 (material didatico; download de amostras)"}

REF_HRF = ("Budai et al. Robust Vessel Segmentation in Fundus Images. "
           "Int. J. Biomed. Imaging, 2013. https://www5.cs.fau.de/research/data/fundus-images/")
REF_ODIR = ("ODIR-2019, Peking University (Shanggong Medical Technology). "
            "Espelho: hf.co/datasets/bumbledeep/odir. Uso educacional/pesquisa.")
REF_RFMID = ("Pachade et al. Retinal Fundus Multi-Disease Image Dataset (RFMiD). Data 6(2):14, 2021. "
             "Espelho: hf.co/datasets/bumbledeep/rfmid")

COLUNAS = ["id", "fonte", "condicao", "rotulo_original", "arquivo", "largura", "altura",
           "mascara_vasos", "mascara_fov", "licenca", "referencia", "observacao"]


# --------------------------------------------------------------------------- utilidades
def _tentar(fn, tentativas: int = 5, espera: float = 4.0):
    """Repete `fn` com backoff — conexões longas ao Hub às vezes caem."""
    for k in range(tentativas):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if k == tentativas - 1:
                raise
            print(f"   ! {type(e).__name__}: {e} — nova tentativa em {espera * (k + 1):.0f}s")
            time.sleep(espera * (k + 1))


def _baixar(url: str, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and destino.stat().st_size > 0:
        return

    def _get():
        with requests.get(url, headers=UA, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(destino, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                                desc=destino.name, leave=False) as barra:
                for parte in r.iter_content(1 << 16):
                    f.write(parte)
                    barra.update(len(parte))

    _tentar(_get)


def _salvar_bytes(dados: bytes, destino: Path) -> tuple[int, int]:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(dados)
    with Image.open(io.BytesIO(dados)) as im:
        return im.size  # (largura, altura)


def _rel(p: Path) -> str:
    return p.relative_to(RAIZ).as_posix()


# --------------------------------------------------------------------------- HRF
def baixar_hrf(force: bool) -> list[dict]:
    """HRF: imagens + segmentação manual de vasos + máscara FOV (45 imagens)."""
    destino = RAW / "hrf"
    if force and destino.exists():
        shutil.rmtree(destino)
    pastas = {"images": destino / "images", "manualsegm": destino / "vasos", "fovmask": destino / "fov"}
    grupos = {"images": ["healthy", "diabetic_retinopathy", "glaucoma"],
              "manualsegm": ["healthy_manualsegm", "diabetic_retinopathy_manualsegm", "glaucoma_manualsegm"],
              "fovmask": ["healthy_fovmask", "diabetic_retinopathy_fovmask", "glaucoma_fovmask"]}
    for tipo, nomes in grupos.items():
        for nome in nomes:
            zpath = CACHE / "hrf" / f"{nome}.zip"
            _baixar(f"{HRF_BASE}/{nome}.zip", zpath)
            with zipfile.ZipFile(zpath) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    alvo = pastas[tipo] / Path(info.filename).name
                    alvo.parent.mkdir(parents=True, exist_ok=True)
                    alvo.write_bytes(z.read(info))

    cond = {"h": "normal", "dr": "rd", "g": "glaucoma"}
    regs = []
    for img in sorted(pastas["images"].glob("*.*")):
        stem = img.stem  # ex.: 01_dr
        sufixo = stem.split("_")[-1]
        vasos = next(iter(pastas["manualsegm"].glob(f"{stem}.*")), None)
        fov = next(iter(pastas["fovmask"].glob(f"{stem}_mask.*")), None)
        with Image.open(img) as im:
            w, h = im.size
        regs.append(dict(
            id=f"hrf_{stem}", fonte="HRF", condicao=cond[sufixo], rotulo_original=sufixo, arquivo=_rel(img),
            largura=w, altura=h, mascara_vasos=_rel(vasos) if vasos else "", mascara_fov=_rel(fov) if fov else "",
            licenca="CC BY 4.0", referencia=REF_HRF,
            observacao="Segmentação manual de vasos (padrão-ouro) e máscara do FOV disponíveis."))
    print(f"HRF: {len(regs)} imagens")
    return regs


# --------------------------------------------------------------------------- Hub (Parquet + HTTP Range)
def _hf():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    return pq, HfFileSystem()


def baixar_odir(force: bool, cotas: dict[str, int] | None = None) -> list[dict]:
    """ODIR-5K (512x512): pacientes distintos por classe, lendo poucos row groups."""
    cotas = cotas or {"normal": 8, "diabetes": 10, "glaucoma": 12, "age_related_macular_degeneration": 12}
    mapa = {"normal": "normal", "diabetes": "rd", "glaucoma": "glaucoma",
            "age_related_macular_degeneration": "dmri"}
    destino = RAW / "odir"
    if force and destino.exists():
        shutil.rmtree(destino)
    pq, fs = _hf()
    caminho = "datasets/bumbledeep/odir/data/train-00000-of-00001.parquet"

    def _rotulos():
        with fs.open(caminho, "rb") as f:
            return pq.ParquetFile(f).read(columns=["patient_id", "label"]).to_pandas()

    tab = _tentar(_rotulos)
    escolhidos, vistos = {}, set()
    for i, (pid, rot) in enumerate(zip(tab.patient_id, tab.label)):
        if rot in cotas and pid not in vistos and sum(1 for r in escolhidos.values() if r[0] == rot) < cotas[rot]:
            escolhidos[i] = (rot, pid)
            vistos.add(pid)
        if all(sum(1 for r in escolhidos.values() if r[0] == c) >= n for c, n in cotas.items()):
            break

    regs, contador = [], {c: 0 for c in cotas}

    def _ler_grupo(g):
        with fs.open(caminho, "rb") as f:
            return pq.ParquetFile(f).read_row_group(g, columns=["image", "patient_id"]).column("image").to_pylist()

    grupos = sorted({i // 100 for i in escolhidos})
    for g in tqdm(grupos, desc="ODIR row groups"):
        imgs = _tentar(lambda g=g: _ler_grupo(g))
        for i in sorted(k for k in escolhidos if k // 100 == g):
            rot, pid = escolhidos[i]
            contador[rot] += 1
            c = mapa[rot]
            arq = destino / c / f"odir_{c}_{contador[rot]:02d}.jpg"
            w, h = _salvar_bytes(imgs[i - g * 100]["bytes"], arq)
            regs.append(dict(id=arq.stem, fonte="ODIR", condicao=c, rotulo_original=rot, arquivo=_rel(arq),
                             largura=w, altura=h, mascara_vasos="", mascara_fov="", licenca="uso educacional/pesquisa",
                             referencia=REF_ODIR, observacao=f"paciente {pid}; imagem redimensionada para 512x512"))
    print(f"ODIR: {len(regs)} imagens")
    return regs


def baixar_rfmid(force: bool) -> list[dict]:
    """RFMiD (2144x1424, resolução original) — só 2 row groups (~340 MB no total).

    Colunas do vetor de rótulos (45 posições, sem Disease_Risk): 0=DR, 1=ARMD (DMRI), 11=ODC (escavação do disco
    óptico, sinal sugestivo de glaucoma). Usamos apenas exemplos com UM único rótulo (ou nenhum = normal).
    """
    destino = RAW / "rfmid"
    if force and destino.exists():
        shutil.rmtree(destino)
    pq, fs = _hf()
    base = "datasets/bumbledeep/rfmid/data/"
    # (arquivo, row group, {classe: cota})
    plano = [("test-00000-of-00004.parquet", 0, {"ARMD": 8, "DR": 4}),
             ("train-00006-of-00007.parquet", 1, {"normal": 8, "ODC": 8})]
    idx = {"DR": 0, "ARMD": 1, "ODC": 11}
    mapa = {"normal": "normal", "DR": "rd", "ARMD": "dmri", "ODC": "glaucoma"}

    def classe(v):
        if sum(v) == 0:
            return "normal"
        if sum(v) == 1:
            return next((k for k, i in idx.items() if v[i]), None)
        return None

    regs, contador = [], {}
    for arq_pq, g, cotas in plano:

        def _ler(arq_pq=arq_pq, g=g):
            with fs.open(base + arq_pq, "rb") as f:
                return pq.ParquetFile(f).read_row_group(g, columns=["image", "label"]).to_pylist()

        print(f"RFMiD: lendo {arq_pq} grupo {g} (~170 MB)...")
        linhas = _tentar(_ler)
        for lin in linhas:
            c = classe(json.loads(lin["label"]))
            if c not in cotas or contador.get(c, 0) >= cotas[c]:
                continue
            contador[c] = contador.get(c, 0) + 1
            cond = mapa[c]
            arq = destino / cond / f"rfmid_{cond}_{contador[c]:02d}.jpg"
            w, h = _salvar_bytes(lin["image"]["bytes"], arq)
            obs = "ODC = escavação do disco óptico (sinal sugestivo de glaucoma; não é diagnóstico)" if c == "ODC" else ""
            regs.append(dict(id=arq.stem, fonte="RFMiD", condicao=cond, rotulo_original=c, arquivo=_rel(arq),
                             largura=w, altura=h, mascara_vasos="", mascara_fov="", licenca="CC BY 4.0",
                             referencia=REF_RFMID, observacao=obs))
    print(f"RFMiD: {len(regs)} imagens")
    return regs


# --------------------------------------------------------------------------- manifesto
FONTES = {"hrf": baixar_hrf, "odir": baixar_odir, "rfmid": baixar_rfmid}


def atualizar_manifesto(novos: list[dict], fontes_atualizadas: set[str]) -> pd.DataFrame:
    """Mescla os novos registros ao manifesto existente (preserva fontes não reprocessadas)."""
    antigo = pd.read_csv(MANIFESTO) if MANIFESTO.exists() else pd.DataFrame(columns=COLUNAS)
    antigo = antigo[~antigo["fonte"].str.lower().isin(fontes_atualizadas)]
    df = pd.concat([antigo, pd.DataFrame(novos, columns=COLUNAS)], ignore_index=True)
    df = df.fillna("").sort_values(["fonte", "condicao", "id"]).reset_index(drop=True)
    MANIFESTO.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFESTO, index=False, encoding="utf-8")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(FONTES), help="baixa somente estas fontes")
    ap.add_argument("--force", action="store_true", help="apaga e rebaixa as fontes selecionadas")
    args = ap.parse_args()

    selecionadas = args.only or list(FONTES)
    novos: list[dict] = []
    for nome in selecionadas:
        print(f"\n=== {nome.upper()} ===")
        novos += FONTES[nome](args.force)

    df = atualizar_manifesto(novos, set(selecionadas))
    print("\nManifesto:", _rel(MANIFESTO))
    print(df.groupby(["fonte", "condicao"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
