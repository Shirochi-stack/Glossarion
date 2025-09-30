@echo off
title Glossarion Web Interface - Advanced Launcher
color 0A
echo.
echo ========================================
echo   Glossarion Web Interface
echo   Advanced Launcher
echo ========================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Select launch mode:
echo.
echo [1] Local Only (http://127.0.0.1:7860)
echo [2] Network Accessible (http://0.0.0.0:7860)
echo [3] Public Share Link (uses Gradio sharing)
echo [4] Custom Port (specify your own)
echo [5] Exit
echo.
set /p choice="Enter choice (1-5): "

if "%choice%"=="1" (
    set SERVER_NAME=127.0.0.1
    set SERVER_PORT=7860
    set SHARE=False
    goto :launch
)

if "%choice%"=="2" (
    set SERVER_NAME=0.0.0.0
    set SERVER_PORT=7860
    set SHARE=False
    echo.
    echo WARNING: This will make the server accessible to other devices on your network.
    echo.
    goto :launch
)

if "%choice%"=="3" (
    set SERVER_NAME=0.0.0.0
    set SERVER_PORT=7860
    set SHARE=True
    echo.
    echo NOTE: This will create a public link that expires in 72 hours.
    echo.
    goto :launch
)

if "%choice%"=="4" (
    set /p SERVER_PORT="Enter port number (default 7860): "
    if "%SERVER_PORT%"=="" set SERVER_PORT=7860
    set SERVER_NAME=127.0.0.1
    set SHARE=False
    goto :launch
)

if "%choice%"=="5" (
    exit /b 0
)

echo Invalid choice. Exiting.
pause
exit /b 1

:launch
echo.
echo ========================================
echo Starting Glossarion Web Interface...
echo ========================================
echo.
echo Configuration:
echo - Host: %SERVER_NAME%
echo - Port: %SERVER_PORT%
echo - Public Share: %SHARE%
echo.
echo The browser will open automatically once the server is ready.
echo Press Ctrl+C in the console to stop the server when done.
echo.

REM Set environment variables for Python script to use
set GRADIO_SERVER_NAME=%SERVER_NAME%
set GRADIO_SERVER_PORT=%SERVER_PORT%
set GRADIO_SHARE=%SHARE%

REM Start PowerShell script in background to wait for server and open browser
start "" /B powershell -ExecutionPolicy Bypass -File "%~dp0wait_and_open.ps1" -url "http://127.0.0.1:%SERVER_PORT%"

REM Start the web interface
python glossarion_web.py

echo.
echo ========================================
echo Server stopped. You can close this window.
echo ========================================
pause