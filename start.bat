@echo off
echo ========================================================
echo  Launching Medical Anomaly Detection System (Beta-VAE)
echo ========================================================

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Medical Anomaly Detection - FastAPI Server" cmd /k "set PYTHONPATH=. && python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 3 /nobreak > nul

echo [2/2] Starting React Frontend on http://localhost:5173 ...
start "Medical Anomaly Detection - React Dashboard" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers started! Open your browser at: http://localhost:5173
echo ========================================================
pause
