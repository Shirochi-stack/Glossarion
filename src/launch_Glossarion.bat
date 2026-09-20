@echo off
REM ensure we're in the script's folder:
cd /d "%~dp0"

REM Use the restored Python 3.12.9 installation directly.  Do not rely on the
REM PATH inherited from Explorer, which can remain stale after a Windows reset.
SET "SYSTEM_PYTHON=C:\Users\ADMIN\AppData\Local\Programs\Python\Python312\python.exe"

IF NOT EXIST "%SYSTEM_PYTHON%" (
    echo Error: Python 3.12.9 was not found at:
    echo %SYSTEM_PYTHON%
    pause
    exit /b 1
)

REM Add MSYS2 DLLs to PATH for WeasyPrint (PREPEND to override Tesseract-OCR's incompatible DLLs)
SET PATH=C:\msys64\mingw64\bin;%PATH%

REM Launch with the exact interpreter whose packages were restored.
"%SYSTEM_PYTHON%" translator_gui.py

REM Pause to see any errors
pause
