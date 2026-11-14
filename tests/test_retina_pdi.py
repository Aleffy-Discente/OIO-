"""Testes do pacote retina_pdi. Os que usam imagens reais são pulados se os dados ainda não foram baixados."""
import cv2
import numpy as np
import pytest

from retina_pdi import dados, degradacoes, pipeline as P, qualidade as Q

TEM_DADOS = dados.MANIFESTO.exists()
precisa_dados = pytest.mark.skipif(not TEM_DADOS, reason="rode scripts/download_data.py")


def retina_sintetica(h=400, w=560, raio=180, seed=0):
    """Disco alaranjado com textura suave sobre fundo preto + 'vasos' escuros."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    disco = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < raio ** 2
    img = np.zeros((h, w, 3), np.uint8)
    base = np.array([190, 100, 40], np.float32) * (0.85 + 0.15 * rng.random((h, w, 1)))
    img[disco] = base[disco].astype(np.uint8)
    cv2.line(img, (w // 2 - 150, h // 2), (w // 2 + 150, h // 2 + 30), (120, 40, 20), 3)
    img[~disco] = 0
    return img, disco


def test_mascara_fov_recupera_disco():
    img, disco = retina_sintetica()
    m = P.mascara_fov(img)
    iou = (m & disco).sum() / (m | disco).sum()
    assert iou > 0.97


def test_mascara_fov_imagem_preta():
    assert not P.mascara_fov(np.zeros((100, 100, 3), np.uint8)).any()


def test_recortar_fov_quadrado_e_sem_fora():
    img, _ = retina_sintetica()
    rec, m = P.recortar_fov(img)
    assert rec.shape[0] == rec.shape[1] and m.shape == rec.shape[:2]
    assert (rec[~m] == 0).all()


def test_gaussiano_mascarado_nao_escurece_borda():
    canal = np.full((200, 200), 100, np.uint8)
    fov = np.zeros((200, 200), bool)
    fov[50:150, 50:150] = True
    canal[~fov] = 0
    suave = P.gaussiano_mascarado(canal, fov, 10)
    assert np.allclose(suave[fov], 100, atol=1e-3)  # sem "vazamento" do preto para dentro do FOV


def test_clahe_aumenta_contraste():
    rng = np.random.default_rng(0)
    c = (120 + rng.normal(0, 4, (256, 256))).clip(0, 255).astype(np.uint8)
    assert P.clahe(c, 3.0).std() > c.std()


def test_metricas_segmentacao_perfeita_e_oposta():
    ref = np.zeros((10, 10), bool)
    ref[2:5, 2:5] = True
    assert P.metricas_segmentacao(ref, ref)["dice"] == 1.0
    assert P.metricas_segmentacao(~ref, ref)["dice"] == 0.0


@pytest.mark.parametrize("norm,dtype,lo,hi", [("0_255", np.float32, 0, 255), ("0_1", np.float32, 0, 1),
                                              ("menos1_1", np.float32, -1, 1), ("uint8", np.uint8, 0, 255)])
@pytest.mark.parametrize("realce", [None, "clahe", "ben_graham"])
def test_preparar_para_inferencia(norm, dtype, lo, hi, realce):
    img, _ = retina_sintetica()
    x = P.preparar_para_inferencia(img, 96, realce, norm)
    assert x.shape == (1, 96, 96, 3) and x.dtype == dtype
    assert x.min() >= lo - 1e-6 and x.max() <= hi + 1e-6


def test_preparar_para_inferencia_argumentos_invalidos():
    img, _ = retina_sintetica()
    with pytest.raises(ValueError):
        P.preparar_para_inferencia(img, 96, "inexistente")
    with pytest.raises(ValueError):
        P.preparar_para_inferencia(img, 96, None, "inexistente")


def test_degradacoes_mudam_a_imagem_e_preservam_formato():
    img, _ = retina_sintetica()
    for tipo in degradacoes.TIPOS:
        out = degradacoes.aplicar(img, tipo, 0.8)
        assert out.shape == img.shape and out.dtype == np.uint8
        assert np.abs(out.astype(int) - img.astype(int)).mean() > 0.5, tipo
    with pytest.raises(ValueError):
        degradacoes.aplicar(img, "nada", 0.5)


@precisa_dados
def test_manifesto_e_carregamento_com_caminho_acentuado():
    df = dados.carregar_manifesto()
    assert {"normal", "rd", "glaucoma", "dmri"} <= set(df.condicao)
    rgb = dados.carregar_rgb(df.iloc[0]["arquivo"])
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8


@precisa_dados
def test_fov_bate_com_mascara_oficial_do_hrf():
    r = dados.selecionar(fonte="hrf", n=1).iloc[0]
    reg = dados.carregar_registro(r)
    m = P.mascara_fov(reg["rgb"])
    assert (m & reg["fov"]).sum() / (m | reg["fov"]).sum() > 0.98


@precisa_dados
def test_segmentacao_de_vasos_dice_minimo_no_hrf():
    r = dados.selecionar("normal", "hrf", 1).iloc[0]
    reg = dados.carregar_registro(r)
    rgb = dados.reduzir(reg["rgb"], 1168)
    gt = cv2.resize(reg["vasos"].astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA) > 0
    fov = P.mascara_fov(rgb)
    pred = P.segmentar_vasos(rgb, fov)
    assert P.metricas_segmentacao(pred, gt, fov)["dice"] > 0.65


@precisa_dados
def test_qualidade_detecta_defeitos_graves_em_imagem_real():
    r = dados.selecionar("normal", "hrf", 1).iloc[0]
    rgb = dados.reduzir(dados.carregar_rgb(r.arquivo), 1024)
    assert Q.avaliar_qualidade(rgb).adequada
    assert not Q.avaliar_qualidade(degradacoes.aplicar(rgb, "escuro", 1.0)).adequada
    assert not Q.avaliar_qualidade(degradacoes.aplicar(rgb, "claro", 1.0)).adequada
    assert not Q.avaliar_qualidade(degradacoes.aplicar(rgb, "ruido", 1.0)).adequada
    assert not Q.avaliar_qualidade(degradacoes.aplicar(rgb, "desfoque", 1.0)).adequada


@precisa_dados
def test_disco_optico_localizado_em_imagem_saudavel():
    r = dados.selecionar("normal", "hrf", 1).iloc[0]
    rgb = dados.reduzir(dados.carregar_rgb(r.arquivo), 512)
    d = P.localizar_disco_optico(rgb)
    assert d.centro is not None and 5 < d.raio < 60


@precisa_dados
def test_fluxo_completo_aprova_boa_e_barra_ruim():
    from retina_pdi import fluxo
    r = dados.selecionar("normal", "hrf", 1).iloc[0]
    rgb = dados.reduzir(dados.carregar_rgb(r.arquivo), 1024)
    ok = fluxo.processar_captura(rgb, 224, "clahe", "menos1_1")
    assert ok.aprovada and ok.tensor.shape == (1, 224, 224, 3) and ok.tempos_ms["total"] > 0
    ruim = fluxo.processar_captura(degradacoes.aplicar(rgb, "escuro", 1.0))
    assert not ruim.aprovada and ruim.tensor is None and "subexposta (escura)" in ruim.qualidade.motivos


@precisa_dados
def test_calibrar_limiares_aprova_as_capturas_de_referencia_e_gera_mensagens():
    import pandas as pd
    df = dados.selecionar("normal", "hrf")
    imgs = [dados.reduzir(dados.carregar_rgb(a), 768) for a in df.arquivo[:8]]
    met = pd.DataFrame([Q.metricas_qualidade(i) for i in imgs])
    lim = Q.calibrar_limiares(met)
    assert all(Q.avaliar_qualidade(i, lim).adequada for i in imgs)
    res = Q.avaliar_qualidade(degradacoes.aplicar(imgs[0], "escuro", 1.0), lim)
    assert not res.adequada and Q.mensagens_para_operador(res)
