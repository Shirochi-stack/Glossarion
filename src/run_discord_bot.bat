@echo off
REM Glossarion Discord Bot Launcher

echo ========================================
echo   Glossarion Discord Bot
echo ========================================
echo.

REM Check if Discord bot token is set
if "%DISCORD_BOT_TOKEN%"=="" (
    echo [ERROR] DISCORD_BOT_TOKEN not set!
    echo.
    echo Please set your Discord bot token:
    echo   set DISCORD_BOT_TOKEN=your_token_here
    echo.
    echo Or edit this file and add it below.
    echo.
    pause
    exit /b 1
)

REM Change to the Glossarion directory
cd /d "%~dp0"

echo [INFO] Starting bot...
echo [INFO] Working directory: %CD%
echo.

REM Run the bot
python discord_bot.py

REM Keep window open if bot crashes
if errorlevel 1 (
    echo.
    echo [ERROR] Bot crashed or failed to start!
    echo Check the error messages above.
    echo.
    pause
)
