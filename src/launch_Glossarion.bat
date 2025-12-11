@echo off
REM ensure we're in the script's folder:
cd /d "%~dp0"

REM Find system Python first before adding MSYS2
FOR /F "tokens=*" %%i IN ('where python.exe 2^>nul') DO SET SYSTEM_PYTHON=%%i & GOTO :found_python
:found_python

REM Add MSYS2 DLLs to PATH for WeasyPrint (PREPEND to override Tesseract-OCR's incompatible DLLs)
SET PATH=C:\msys64\mingw64\bin;%PATH%

REM call the real python using the path we found earlier
IF DEFINED SYSTEM_PYTHON (
    "%SYSTEM_PYTHON%" translator_gui.py
) ELSE (
    python translator_gui.py
)

REM or, alternatively:
REM py -3 translator_gui.py

REM Pause to see any errors
pause
