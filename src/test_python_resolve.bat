@echo off
cd /d "%~dp0"

echo === Before MSYS2 PATH ===
where python.exe

REM Simulate the FOR loop from launch_Glossarion.bat
FOR /F "tokens=*" %%i IN ('where python.exe 2^>nul') DO SET SYSTEM_PYTHON=%%i & GOTO :found_python
:found_python

echo.
echo SYSTEM_PYTHON resolved to: %SYSTEM_PYTHON%
echo.

REM Now add MSYS2
SET PATH=C:\msys64\mingw64\bin;%PATH%

echo === After MSYS2 PATH ===
where python.exe

echo.
echo Testing PySide6 import with resolved Python:
"%SYSTEM_PYTHON%" -c "import PySide6; print('PySide6 OK - version', PySide6.__version__)"

pause
