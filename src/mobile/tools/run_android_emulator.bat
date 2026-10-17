@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem ---------------------------------------------------------------------------
rem Glossarion Mobile on the local Android emulator (AVD "glossarion",
rem API 35 x86_64, 16 GB RAM, 8 cores, 24 GB data - enough for the 9B model).
rem
rem   run_android_emulator.bat                 start the emulator, open the app
rem   run_android_emulator.bat path\to.apk     ... and install that x86_64 APK first
rem   run_android_emulator.bat latest          ... and install the x86_64 APK of the
rem                                            newest successful GitHub Build Mobile
rem                                            run on main (needs the gh CLI)
rem   run_android_emulator.bat smoke [apk]     run the CI smoke + E2E script instead
rem
rem Override the AVD with GLOSSARION_AVD, RAM (MB) with GLOSSARION_EMU_RAM.
rem ---------------------------------------------------------------------------

set "SDK=%LOCALAPPDATA%\Android\Sdk"
set "JAVA_HOME=%LOCALAPPDATA%\Android\jdk17"
set "ANDROID_HOME=%SDK%"
set "ADB=%SDK%\platform-tools\adb.exe"
set "EMU=%SDK%\emulator\emulator.exe"
set "PKG=com.glossarion.app"
if not defined GLOSSARION_AVD set "GLOSSARION_AVD=glossarion"
if not defined GLOSSARION_EMU_RAM set "GLOSSARION_EMU_RAM=16384"
set "REPO=%~dp0..\..\.."
set "WORK=%LOCALAPPDATA%\Glossarion\emulator"

if not exist "%EMU%" (
    echo Android emulator not found at "%EMU%".
    echo Install the SDK first ^(see src\mobile\README.md^).
    exit /b 1
)

set "MODE=run"
set "APK="
if /i "%~1"=="smoke" (
    set "MODE=smoke"
    set "APK=%~2"
) else (
    set "APK=%~1"
)

rem --- "latest": download the newest x86_64 APK from GitHub -----------------
if /i "%APK%"=="latest" (
    where gh >nul 2>nul || (echo The gh CLI is needed for "latest". & exit /b 1)
    set "RUNID="
    for /f "usebackq delims=" %%R in (`gh run list --repo Shirochi-stack/Glossarion --workflow build-mobile.yml --branch main --status success --limit 1 --json databaseId --jq ".[0].databaseId"`) do set "RUNID=%%R"
    if not defined RUNID (echo No successful Build Mobile run on main found. & exit /b 1)
    echo Downloading the APK of Build Mobile run !RUNID! ...
    if exist "%WORK%\apk" rmdir /s /q "%WORK%\apk"
    mkdir "%WORK%\apk"
    gh run download !RUNID! --repo Shirochi-stack/Glossarion --name X_Glossarion-Android-APK --dir "%WORK%\apk" || exit /b 1
    set "APK="
    for /r "%WORK%\apk" %%F in (*x86_64*.apk) do set "APK=%%F"
    if not defined APK (echo No x86_64 APK in that run. & exit /b 1)
)

rem --- start the emulator unless one is already attached ---------------------
"%ADB%" start-server >nul 2>nul
set "RUNNING="
for /f "skip=1 tokens=1,2" %%A in ('"%ADB%" devices') do if "%%B"=="device" set "RUNNING=%%A"
if defined RUNNING (
    echo Using the running device !RUNNING!.
) else (
    echo Starting emulator "%GLOSSARION_AVD%" with %GLOSSARION_EMU_RAM% MB RAM ...
    start "Android emulator" "%EMU%" -avd %GLOSSARION_AVD% -memory %GLOSSARION_EMU_RAM% -cores 8 -gpu auto -no-boot-anim -no-audio
)

echo Waiting for Android to finish booting ...
"%ADB%" wait-for-device
set /a TRIES=0
:waitboot
set "BOOTED="
for /f "delims=" %%B in ('"%ADB%" shell getprop sys.boot_completed 2^>nul') do set "BOOTED=%%B"
if "%BOOTED%"=="1" goto booted
set /a TRIES+=1
if %TRIES% GEQ 300 (echo The emulator did not boot within 10 minutes. & exit /b 1)
"%SystemRoot%\System32\timeout.exe" /t 2 /nobreak >nul
goto waitboot
:booted
echo Booted.

rem --- smoke mode: the same script CI runs ------------------------------------
if "%MODE%"=="smoke" (
    if not defined APK (echo smoke needs an APK path or "latest". & exit /b 1)
    set "PATH=%SDK%\platform-tools;%PATH%"
    set "BASH=%ProgramFiles%\Git\bin\bash.exe"
    if not exist "!BASH!" (echo Git Bash not found at "!BASH!". & exit /b 1)
    "!BASH!" "%REPO%\src\mobile\ci\android_smoke.sh" "%APK%" "%WORK%\smoke-logs"
    exit /b !ERRORLEVEL!
)

rem --- install (optional) and open the app -------------------------------------
if defined APK (
    if not exist "%APK%" (echo APK not found: "%APK%" & exit /b 1)
    echo Installing "%APK%" ...
    "%ADB%" install -r -g "%APK%"
    if errorlevel 1 (
        echo Install failed. An app signed with a different key may be installed:
        echo   "%ADB%" uninstall %PKG%   ^(this deletes the app's data on the emulator^)
        exit /b 1
    )
)
"%ADB%" shell pm path %PKG% >nul 2>nul
if errorlevel 1 (
    echo Glossarion is not installed on the emulator yet. Run this again with an APK path or "latest".
    exit /b 1
)
echo Opening Glossarion ...
rem Open the launcher activity with "am start" (monkey would also send one random tap).
set "ACTIVITY="
for /f "delims=" %%A in ('"%ADB%" shell cmd package resolve-activity --brief -c android.intent.category.LAUNCHER %PKG% ^| findstr /c:"/"') do set "ACTIVITY=%%A"
if not defined ACTIVITY set "ACTIVITY=%PKG%/.MainActivity"
"%ADB%" shell am start -n !ACTIVITY! >nul 2>&1
echo Done. Close the emulator window when you are finished.
exit /b 0
