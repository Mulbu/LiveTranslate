@echo off
cd /d "%~dp0"
set PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%

:: Locate interpreter: prefer the conda env recorded by install.ps1, else .venv
set "PY="
if exist ".conda-python" set /p PY=<.conda-python
if defined PY if not exist "%PY%" set "PY="
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"

if not defined PY (
    echo [ERROR] No environment found ^(no conda env or .venv^).
    echo Please run install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

if not exist ".livetranslate-ready" (
    echo [ERROR] Environment setup is incomplete.
    echo Please run install.bat again to finish installing and verifying dependencies.
    echo.
    pause
    exit /b 1
)

echo Starting LiveTranslate...
"%PY%" main.py
if errorlevel 1 (
    echo.
    echo [ERROR] LiveTranslate exited with an error.
    pause
)
