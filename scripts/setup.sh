#!/usr/bin/env bash
# Prepara o ambiente do Módulo 02 em Linux/macOS/Raspberry Pi OS.
#   ./scripts/setup.sh              # venv + dependências + pacote + dados + testes
#   ./scripts/setup.sh --sem-dados  # pula o download das imagens
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || python3 -m venv .venv
PY=.venv/bin/python
$PY -m pip install --upgrade pip
$PY -m pip install -r requirements.txt
$PY -m pip install -e ".[dev]"
$PY -m ipykernel install --user --name retina-modulo02 --display-name "Python (Retina em Foco - M02)"

[ "${1:-}" = "--sem-dados" ] || $PY scripts/download_data.py
$PY -m pytest -q
echo
echo "Pronto. Abra o VS Code / JupyterLab e escolha o interpretador .venv (kernel 'Python 3')."
echo "  JupyterLab: .venv/bin/jupyter lab notebooks"
