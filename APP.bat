@echo off
cd /d "%~dp0"

:: Evitar que Streamlit pida email la primera vez
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    mkdir "%USERPROFILE%\.streamlit" 2>nul
    echo [general] > "%USERPROFILE%\.streamlit\credentials.toml"
    echo email = "" >> "%USERPROFILE%\.streamlit\credentials.toml"
)

python -m streamlit run app.py --browser.gatherUsageStats false
pause
