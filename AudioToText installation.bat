@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title AudioToText — setup and launch

REM --- Find Python ---
set "PYEXE="
where python >nul 2>&1 && set "PYEXE=python"
if not defined PYEXE (
  where py >nul 2>&1 && set "PYEXE=py -3"
)
if not defined PYEXE (
  echo.
  echo [ERROR] Python 3 was not found.
  echo Install Python 3.10 or newer from https://www.python.org/
  echo Enable "Add python.exe to PATH" during setup.
  echo.
  pause
  exit /b 1
)

REM --- Virtual environment ---
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo Creating virtual environment in .venv ...
  %PYEXE% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Could not create .venv — try: %PYEXE% -m venv .venv
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Could not activate .venv
  pause
  exit /b 1
)

REM --- Dependencies ---
echo.
echo Installing / updating packages from requirements.txt ...
python -m pip install --upgrade pip -q
pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] pip install failed.
  pause
  exit /b 1
)

REM --- Run app ---
echo.
echo Starting AudioToText...
echo.
python main.py
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
  echo.
  echo App exited with code %EXITCODE%.
  pause
)
exit /b %EXITCODE%
