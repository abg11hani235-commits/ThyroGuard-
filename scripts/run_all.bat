@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."

echo ====================================================================
echo        Starting ThyroGuard Clinical Evidence RAG System
echo ====================================================================

:: 1. Detect Python Executable dynamically
set "PYTHON_EXE="
if exist "%CD%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
    echo [Environment] Using virtual environment: .venv\Scripts\python.exe
) else if exist "%CD%\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\venv\Scripts\python.exe"
    echo [Environment] Using virtual environment: venv\Scripts\python.exe
) else if exist "D:\New folder\programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=D:\New folder\programs\Python\Python311\python.exe"
    echo [Environment] Using Python 3.11: D:\New folder\programs\Python\Python311\python.exe
) else (
    for /f "tokens=*" %%i in ('where python 2^>nul') do (
        if not defined PYTHON_EXE (
            set "PYTHON_EXE=%%i"
            echo [Environment] Using system Python: %%i
        )
    )
)

if not defined PYTHON_EXE (
    echo.
    echo [ERROR] No Python executable found!
    echo Please run 'setup_local_env.bat' to create your local virtual environment,
    echo or ensure Python 3.10+ is added to your Windows PATH.
    echo.
    pause
    exit /b 1
)

echo.
echo [1/3] Launching FastAPI Backend on Port 8000...
start "ThyroGuard Clinical RAG API (Port 8000)" cmd /k ""%PYTHON_EXE%" -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload"

echo [2/3] Launching Modern Web UI on Port 3000...
start "ThyroGuard Web UI (Port 3000)" cmd /k ""%PYTHON_EXE%" -m http.server 3000 --directory frontend"

echo [3/3] Launching Streamlit Clinical UI on Port 8501...
start "ThyroGuard Streamlit Clinical UI (Port 8501)" cmd /k ""%PYTHON_EXE%" -m streamlit run ui/streamlit_app.py --server.port 8501"

echo.
echo ====================================================================
echo [SUCCESS] All ThyroGuard native services have been launched!
echo ====================================================================
echo.
echo  - Modern Clinical Web UI:  http://localhost:3000
echo  - FastAPI Endpoints:       http://localhost:8000/app/
echo  - Interactive Swagger Docs: http://localhost:8000/docs
echo  - Streamlit Benchmark UI:  http://localhost:8501
echo.
echo (Keep the background terminal windows open while using ThyroGuard)
echo.
pause
