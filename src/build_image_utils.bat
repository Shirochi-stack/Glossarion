@echo off
REM Build script for image_utils.dll
REM Requires MSVC (Visual Studio) or MinGW-w64

echo Building C++ Image Utils...
echo.

REM Try MSVC first (cl.exe)
where cl.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using MSVC compiler...
    cl.exe /LD /O2 /EHsc image_utils.cpp /Fe:libimage_utils.dll
    if %ERRORLEVEL% EQU 0 (
        echo ✓ Build successful: libimage_utils.dll
        exit /b 0
    )
)

REM Try MinGW-w64 (g++)
where g++.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using MinGW-w64 compiler...
    g++ -shared -O3 -fPIC image_utils.cpp -o libimage_utils.dll -static-libgcc -static-libstdc++
    if %ERRORLEVEL% EQU 0 (
        echo ✓ Build successful: libimage_utils.dll
        exit /b 0
    )
)

echo ✗ Error: No compiler found!
echo Please install Visual Studio (MSVC) or MinGW-w64
exit /b 1
