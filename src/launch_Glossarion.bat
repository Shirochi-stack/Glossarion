@echo off
cd /d "%~dp0"

where pythonw >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ pythonw.exe not found. Please install Python and ensure it is in your PATH.
    pause
    exit /b
)

start python.exe translator_gui.py
exit
