@echo off
echo ============================================================
echo Starting GraphGuard Streamlit Frontend
echo ============================================================
cd fraud-system
call venv\Scripts\activate

echo Launching Streamlit Dashboard...
python -m streamlit run dashboard/app.py

pause
