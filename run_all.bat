@echo off
echo ====================================================================
echo Starting Medical Evidence RAG System (Backend + Web UI + Streamlit)
echo ====================================================================

:: 1. Launch FastAPI Backend on Port 8000
start "Medical Evidence RAG API" cmd /k "D:\New folder\programs\Python\Python311\python.exe" -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

:: 2. Launch Modern Web UI on Port 3000 (localhost:3000 as seen in screenshots)
start "Medical Evidence Web UI (Port 3000)" cmd /k "D:\New folder\programs\Python\Python311\python.exe" -m http.server 3000 --directory frontend

:: 3. Launch Streamlit Clinical UI on Port 8501
start "Medical Evidence Streamlit UI (Port 8501)" cmd /k "D:\New folder\programs\Python\Python311\python.exe" -m streamlit run ui/streamlit_app.py --server.port 8501

echo.
echo [OK] All services launched!
echo.
echo - Modern Web UI:        http://localhost:3000
echo - FastAPI & Web UI:     http://localhost:8000/app/
echo - API Documentation:    http://localhost:8000/docs
echo - Streamlit Clinical UI: http://localhost:8501
echo.
pause
