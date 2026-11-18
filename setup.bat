@echo off
setlocal
cd /d "%~dp0"

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

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  %PYTHON% -m venv .venv
  if errorlevel 1 (
    echo Could not create the virtual environment. Delete .venv and run setup.bat again.
    exit /b 1
  )
)

echo Installing ForensIQ and development dependencies ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -e ".[dev]"
if errorlevel 1 exit /b 1

call :check_sklearn
if errorlevel 1 (
  echo.
  echo The installed scikit-learn build is blocked by this machine's Application Control policy.
  echo Falling back to the known-good scikit-learn 1.8.0 ...
  ".venv\Scripts\python.exe" -m pip install --force-reinstall "scikit-learn==1.8.0"
  if errorlevel 1 exit /b 1
  call :check_sklearn
  if errorlevel 1 (
    echo error: scikit-learn still cannot load. See the troubleshooting note in README.md.
    exit /b 1
  )
)

echo.
echo Setup complete. Start the app with:
echo   run.bat
echo.
echo Or run the server directly with:
echo   .venv\Scripts\python.exe -m aiforensics serve
echo.
echo Run the tests with:
echo   .venv\Scripts\python.exe -m pytest
exit /b 0

:check_sklearn
".venv\Scripts\python.exe" -c "import sklearn.linear_model" >nul 2>nul
exit /b %errorlevel%
