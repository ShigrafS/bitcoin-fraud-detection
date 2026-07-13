@echo off
echo ============================================================
echo Shutting down GraphGuard Backend and Frontend Services
echo ============================================================

echo Stopping FastAPI Backend (Port 8000)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo Killing PID %%a
    taskkill /F /PID %%a
)

echo Stopping Streamlit Frontend (Port 8501)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501 ^| findstr LISTENING') do (
    echo Killing PID %%a
    taskkill /F /PID %%a
)

echo Shutdown completed successfully.
pause
