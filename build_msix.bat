@echo off
setlocal

cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\tools\msix\build-msix.ps1" -DistDir "%~dp0src\dist" -ExecutablePattern "Glossarion v*.exe"

if errorlevel 1 (
    echo.
    echo MSIX build failed.
    pause
    exit /b 1
)

echo.
echo MSIX build complete.
echo Output folder: %~dp0dist\msix
pause
