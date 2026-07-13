@echo off
echo ============================================================
echo Starting GraphGuard FastAPI Backend
echo ============================================================
cd fraud-system
call venv\Scripts\activate

echo Launching FastAPI Backend...
echo (Loading models and graph data may take a few seconds...)
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

pause
