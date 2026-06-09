@echo off
cd /d "C:\Users\pauli\Downloads\TP imagenes tumores"
python -m streamlit run app.py --browser.gatherUsageStats false
pause
