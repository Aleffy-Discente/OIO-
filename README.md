# Retina em Foco · Módulo 02 — Processamento Digital de Imagens Biomédicas

Material prático do **Módulo 02** (16 h · Semanas 1–5) do curso de formação técnica do projeto *Retina em Foco*
(Edital nº 14/2026 FACEPE — Compet Superior · IFPE Campus Palmares · Centro Médico Senador José Ermírio de Moraes).

Ementa: **01** aquisição de imagens de fundo de olho · **02** OpenCV e espaços de cor · **03** filtragem e realce de contraste (CLAHE) ·
**04** segmentação de estruturas retinianas · **05** avaliação automática da qualidade · **06** preparação de imagens para inferência.
Cada item tem um notebook executável, com **imagens reais e públicas** de retinopatia diabética (RD), glaucoma, DMRI e olhos normais.

## Início rápido

**Windows (PowerShell)**
```powershell
.\scripts\setup.ps1            # cria .venv, instala dependências, baixa as imagens (~165 MB) e roda os testes
.\.venv\Scripts\jupyter lab notebooks
```
**Linux / macOS / Raspberry Pi OS**
```bash
./scripts/setup.sh
.venv/bin/jupyter lab notebooks
```
No **VS Code**, abra a pasta, escolha o interpretador `.venv` e o kernel *Python 3*.
Já existe um ambiente pronto em `.venv/` (Python 3.12) neste computador; os comandos acima só são necessários em outra máquina.

Passo a passo manual, se preferir: `python -m venv .venv` → `pip install -r requirements.txt` → `pip install -e ".[dev]"` → `python scripts/download_data.py`.
`requirements-lock.txt` guarda as versões exatas com que tudo foi testado (Python 3.12, OpenCV 4.14, NumPy 2.5, pandas 3.0).

## Notebooks (`notebooks/`)

| # | Notebook | Você vai… |
|---|---|---|
| 00 | `00_Visao_Geral_e_Dados` | conhecer o roteiro, o manifesto de dados e as convenções |
| 01 | `01_Aquisicao_de_Imagens` | seguir da Aula 1: do fóton ao pixel; resolução, bits, JPEG; foco/iluminação/centralização/artefatos como números; roteiro da sessão no Lab Maker |
| 02 | `02_OpenCV_e_Espacos_de_Cor` | dominar as armadilhas da OpenCV; **medir** qual canal realça os vasos (G, d de Cohen); HSV/LAB/YCrCb; cor × câmera; máscara do FOV |
| 03 | `03_Filtragem_e_CLAHE` | comparar filtros contra dois tipos de ruído; equalização × CLAHE; grade de parâmetros; onde aplicar (L do LAB); correção de iluminação; ordem das operações |
| 04 | `04_Segmentacao_Estruturas_Retinianas` | segmentar vasos (Frangi/Sato/black-hat) e **avaliar** com o padrão-ouro do HRF; ROC/PR; resolução × tempo; disco óptico; candidatos a lesão e seus limites |
| 05 | `05_Avaliacao_Qualidade_Imagem` | métricas de qualidade, degradações sintéticas, calibração de limiares, regras × classificador, calibração por câmera, latência |
| 06 | `06_Preparacao_para_Inferencia` | recorte/padding, interpolação, realce, normalização, fluxo completo com orçamento de tempo, paridade treino × inferência, exportação |

Os notebooks vêm **já executados** (saídas embutidas). Para reexecutar tudo: `python scripts/executar_notebooks.py` (≈ 6–8 min; aceita prefixos, ex.: `03 04`).
Cada notebook termina com **exercícios propostos** — base da entrega "Exercícios de programação — filtragem, CLAHE e segmentação".

## Estrutura

```
Módulo 2/
├── notebooks/            # 00–06 (executados)
├── src/retina_pdi/       # pacote reutilizável (versão final das funções dos notebooks)
│   ├── dados.py          #   manifesto, leitura de imagens (funciona com acentos no caminho), HRF reduzido
│   ├── pipeline.py       #   FOV, CLAHE, iluminação, vasos, disco óptico, preparação p/ inferência
│   ├── qualidade.py      #   métricas, limiares, calibração no dispositivo, mensagens ao operador
│   ├── degradacoes.py    #   defeitos sintéticos (foco, luz, ruído) com nível conhecido
│   ├── metricas.py       #   Cohen d, entropia, contraste RMS
│   ├── fluxo.py          #   captura → qualidade → tensor (contrato F2 ↔ F3)
│   └── visualizacao.py
├── scripts/              # download_data.py · executar_notebooks.py · setup.ps1 / setup.sh
├── tests/                # pytest (27 testes)
├── data/                 # manifest.csv (versionável) · raw/ · processed/ (gerados)
├── docs/DADOS.md         # fontes, licenças, cuidados de interpretação
├── outputs/figures/      # figuras salvas pelos notebooks
├── Aula1-Aquisicao-Fundo-Olho.pdf · Modulo02.html   # materiais originais do módulo
└── requirements.txt · requirements-lock.txt · pyproject.toml
```

## Usando o pacote fora dos notebooks

```python
from retina_pdi import dados, fluxo

rgb = dados.carregar_rgb("caminho/da/captura.png")        # uint8, RGB
r = fluxo.processar_captura(rgb, tamanho=224, realce="clahe", normalizacao="menos1_1")
if r.aprovada:
    tensor = r.tensor                                     # (1, 224, 224, 3) pronto para o modelo
else:
    print(r.qualidade.motivos)                            # pedir nova captura ao operador
```

## Testes

```bash
.venv/Scripts/python -m pytest -q     # Windows   ·   .venv/bin/python -m pytest -q  (Linux)
```
Cobrem FOV, CLAHE, convolução mascarada, métricas, degradações, normalizações, qualidade, calibração e o fluxo completo (os que usam imagens reais são pulados se os dados não foram baixados).
