@echo off

REM ============================================================
REM Streamlit App Launcher
REM Runs the grape disease detection web interface
REM ============================================================

cd /d %~dp0

echo Starting Streamlit application...
echo.

streamlit run app.py

pause
