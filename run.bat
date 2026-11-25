@echo off
rem One command: makes a private Python environment in this folder, installs the pinned
rem packages, and opens the local page. Models download on first use (see README: sizes).
setlocal
cd /d "%~dp0"

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo No NVIDIA driver found ^(nvidia-smi is missing^). This needs an NVIDIA card with 8 GB or more.
  exit /b 1
)

set "PY="
py -3.12 -c "" >nul 2>nul && set "PY=py -3.12"
if not defined PY (
  python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python 3.12 was not found. Install it from https://www.python.org/downloads/ ^(tick "Add to PATH"^) and run this again.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the Python environment in .venv ...
  %PY% -m venv .venv || exit /b 1
)
if not exist ".venv\installed.txt" (
  echo Installing packages ^(about 3 GB, once^) ...
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu130 || exit /b 1
  echo ok> ".venv\installed.txt"
)

".venv\Scripts\python.exe" -m app.server --open %*
