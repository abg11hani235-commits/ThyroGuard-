@echo off
echo ============================================================
echo Starting Medical Evidence RAG Web UI (Port 3000)...
echo ============================================================
echo Access the UI at: http://localhost:3000
"D:\New folder\programs\Python\Python311\python.exe" -m http.server 3000 --directory frontend
pause
