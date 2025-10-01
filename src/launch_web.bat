@echo off
title Glossarion Web Interface
echo.
echo ========================================
echo    Glossarion Web Interface Launcher
echo ========================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Starting Glossarion Web Interface...
echo.
echo The browser will open automatically once the server is ready.
echo Press Ctrl+C in the console to stop the server when done.
echo.

REM Start PowerShell script in background to wait for server and open browser
start "" /B powershell -ExecutionPolicy Bypass -File "%~dp0wait_and_open.ps1" -url "http://127.0.0.1:7860"

REM Start the web interface
python app.py

echo.
echo ========================================
echo Server stopped. You can close this window.
echo ========================================
pause
