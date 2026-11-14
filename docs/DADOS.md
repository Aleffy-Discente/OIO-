# Dados: fontes, licenças e cuidados

Todas as imagens são **retinografias reais, públicas e anonimizadas**, baixadas por `python scripts/download_data.py`
(≈ 165 MB, sem cadastro). O arquivo `data/manifest.csv` descreve cada imagem (fonte, condição, rótulo original, tamanho, licença, referência, máscaras).

| Fonte | Conteúdo usado aqui | Resolução | Licença / uso | Onde |
|---|---|---|---|---|
| **HRF** — High-Resolution Fundus (FAU Erlangen) | 45 imagens: 15 saudáveis, 15 RD, 15 glaucoma; **segmentação manual de vasos** e **máscara do FOV** | 3504×2336 | CC BY 4.0 | <https://www5.cs.fau.de/research/data/fundus-images/> |
| **RFMiD** — Retinal Fundus Multi-disease Image Dataset | 28 imagens: 8 DMRI, 4 RD, 8 "escavação do disco" (rotulada como *glaucoma*), 8 normais | ≈ 2144×1424 | CC BY 4.0 | espelho: `hf.co/datasets/bumbledeep/rfmid` (original: Grand Challenge RIADD) |
| **ODIR-5K** — Ocular Disease Intelligent Recognition | 42 imagens: 12 DMRI, 10 RD, 12 glaucoma, 8 normais (pacientes distintos) | 512×512 | uso educacional / pesquisa | espelho: `hf.co/datasets/bumbledeep/odir` |

Total: **115 imagens** (HRF 45 · ODIR 42 · RFMiD 28).

## Como citar
* HRF — Budai, A. et al. *Robust Vessel Segmentation in Fundus Images.* International Journal of Biomedical Imaging, 2013.
* RFMiD — Pachade, S. et al. *Retinal Fundus Multi-Disease Image Dataset (RFMiD): A Dataset for Multi-Disease Detection Research.* Data, 6(2):14, 2021.
* ODIR-5K — Peking University / Shanggong Medical Technology, *ODIR-2019 (Ocular Disease Intelligent Recognition)*.

Se você publicar resultados, cite as fontes originais e **confira a licença vigente** em cada página: os espelhos do Hugging Face são cópias de terceiros e
suas etiquetas de licença não substituem as dos autores.

## Cuidados de interpretação
1. **Rótulos herdados, não revistos.** As condições vêm das bases. Este projeto não revisou nenhum rótulo com oftalmologista;
   use-os como "condição declarada pela fonte".
2. **"Glaucoma" do RFMiD = `ODC`** (*optic disc cupping*, escavação do disco): sinal **sugestivo** de glaucoma, não diagnóstico.
   No script, as posições do vetor de rótulos usadas são `0 = DR`, `1 = ARMD (DMRI)`, `11 = ODC`, e só entram imagens com **um único** rótulo (ou nenhum, para "normal").
   A correspondência foi conferida visualmente (drusas/atrofia nas DMRI; discos com escavação nas ODC).
3. **ODIR está em 512×512** (reamostrado e, às vezes, recortado): serve para diversidade de câmeras e de qualidade, não para medir detalhe fino.
4. **Sem verdade de qualidade de captura.** Nenhuma base tem rótulo confiável de qualidade; o notebook 05 usa degradações sintéticas e inspeção visual.
5. **Amostra pequena e desbalanceada por fonte** (o HRF não tem DMRI). Resultados dos notebooks são **ilustrativos**, não estimativas de desempenho clínico.
6. **Ética.** Bases públicas e anonimizadas; o projeto (Edital nº 14/2026) usa somente bases públicas e ensaios de bancada, sem procedimentos com seres humanos.

## Bancos de treinamento do edital (não incluídos aqui)
Messidor-2, APTOS 2019, DDR (RD), REFUGE (glaucoma) e ADAM (DMRI) serão usados no **Módulo 4** (treinamento dos modelos).
Quando forem adicionados, basta acrescentar linhas ao manifesto (esquema abaixo) para os notebooks os enxergarem.

> Nota técnica: o APTOS 2019 disponível como espelho no Hub vem em *shards* de ≈ 500 MB ordenados por classe, o que inviabilizou uma amostra pequena; use o download oficial (Kaggle).

## Esquema de `data/manifest.csv`
| Coluna | Significado |
|---|---|
| `id` | identificador único (`fonte_condicao_NN`) |
| `fonte` | `HRF`, `RFMiD` ou `ODIR` |
| `condicao` | `normal`, `rd`, `glaucoma`, `dmri` |
| `rotulo_original` | rótulo na base de origem |
| `arquivo` | caminho relativo à raiz do projeto |
| `largura`, `altura` | tamanho em pixels |
| `mascara_vasos`, `mascara_fov` | caminhos das máscaras de referência (só HRF) |
| `licenca`, `referencia`, `observacao` | proveniência |

Para adicionar imagens do **retinógrafo** (sessão do Lab Maker), use `fonte = "OIO-M2"` e as mesmas colunas. Os cálculos dos notebooks 05 e 06 percorrem todo o manifesto,
mas alguns gráficos e tabelas listam as três fontes explicitamente (variáveis `fontes` / `["HRF", "RFMiD", "ODIR"]`) — inclua a nova fonte nessas listas.
