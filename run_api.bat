@echo off
echo Starting ThyroGuard Clinical RAG API Server...
"D:\New folder\programs\Python\Python311\python.exe" -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
pause
