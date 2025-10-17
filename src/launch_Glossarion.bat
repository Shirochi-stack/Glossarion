@echo off
REM ensure we're in the script's folder:
cd /d "%~dp0"

REM call the real python
python translator_gui.py

REM or, alternatively:
REM py -3 translator_gui.py

REM Pause to keep window open after program exits
pause
