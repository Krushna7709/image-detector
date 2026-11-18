@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%src"

if exist "%ROOT%.venv\Scripts\python.exe" (
  if "%~1"=="" (
    "%ROOT%.venv\Scripts\python.exe" -m aiforensics serve
  ) else (
    "%ROOT%.venv\Scripts\python.exe" -m aiforensics %*
  )
  exit /b %errorlevel%
)

py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 (
  set "PYTHON=py -3"
) else (
  python -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo Python 3.10 or newer is required. Install it from https://www.python.org/downloads/
    exit /b 1
  )
  set "PYTHON=python"
)

if "%~1"=="" (
  %PYTHON% -m aiforensics serve
) else (
  %PYTHON% -m aiforensics %*
)
