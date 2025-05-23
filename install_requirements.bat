@echo off
echo ===============================
echo Installing Glossarion Dependencies
echo ===============================

REM Check if pythonw is available
where pythonw >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Error: pythonw.exe not found in PATH. Glossarion requires pythonw to run GUI silently.
    pause
    exit /b
)

REM Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

if %ERRORLEVEL% equ 0 (
    echo ✅ All dependencies installed successfully.
) else (
    echo ❌ Failed to install dependencies.
)
pause
