# Prepara o ambiente do Módulo 02 no Windows (PowerShell).
#   .\scripts\setup.ps1              # venv + dependências + pacote + dados + testes
#   .\scripts\setup.ps1 -SemDados    # pula o download das imagens
param([switch]$SemDados)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path .venv)) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt
& $py -m pip install -e ".[dev]"
& $py -m ipykernel install --user --name retina-modulo02 --display-name "Python (Retina em Foco - M02)"

if (-not $SemDados) { & $py scripts\download_data.py }
& $py -m pytest -q
Write-Host "`nPronto. Abra o VS Code / JupyterLab e escolha o interpretador .venv (kernel 'Python 3')."
Write-Host "  JupyterLab: .\.venv\Scripts\jupyter lab notebooks"
