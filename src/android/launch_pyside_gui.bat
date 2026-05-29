@echo off
title Glossarion PySide Android Launcher
setlocal

echo.
echo ========================================
echo   Glossarion PySide Android Launcher
echo ========================================
echo.

REM Run from src/ so desktop resource and module lookups match the main app.
cd /d "%~dp0.."

REM Find system Python first before adding MSYS2 DLLs.
for /f "tokens=*" %%i in ('where python.exe 2^>nul') do (
    set "SYSTEM_PYTHON=%%i"
    goto :found_python
)

:found_python

REM Match the regular desktop launcher's DLL path behavior.
set "PATH=C:\msys64\mingw64\bin;%PATH%"

if defined SYSTEM_PYTHON (
    "%SYSTEM_PYTHON%" android\pyside_launcher\main.py %*
) else (
    python android\pyside_launcher\main.py %*
)

echo.
echo ========================================
echo Launcher exited with code %ERRORLEVEL%.
echo ========================================
pause
