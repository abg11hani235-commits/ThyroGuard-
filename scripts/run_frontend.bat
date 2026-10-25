@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."

echo ============================================================
echo   Starting ThyroGuard Clinical Web UI (Port 3000)
echo ============================================================

:: Detect Python Executable dynamically
set "PYTHON_EXE="
if exist "%CD%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
) else if exist "%CD%\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\venv\Scripts\python.exe"
) else if exist "D:\New folder\programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=D:\New folder\programs\Python\Python311\python.exe"
) else (
    for /f "tokens=*" %%i in ('where python 2^>nul') do (
        if not defined PYTHON_EXE set "PYTHON_EXE=%%i"
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] No Python found! Run setup_local_env.bat first.
    pause
    exit /b 1
)

echo Using Python: %PYTHON_EXE%
echo Modern Web UI accessible at: http://localhost:3000
"%PYTHON_EXE%" -m http.server 3000 --directory frontend
pause
