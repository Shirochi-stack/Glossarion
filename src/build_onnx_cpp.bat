@echo off
REM Build script for ONNX C++ backend on Windows

echo ========================================
echo ONNX C++ Backend Build Script
echo ========================================
echo.

REM Check if CMake is available
where cmake >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake not found. Please install CMake from https://cmake.org/download/
    pause
    exit /b 1
)

REM Check for ONNX Runtime
if "%ONNXRUNTIME_DIR%"=="" (
    echo [INFO] ONNXRUNTIME_DIR not set. Checking common locations...
    
    REM Try to find ONNX Runtime in common locations
    if exist "%USERPROFILE%\onnxruntime\lib\onnxruntime.lib" (
        set ONNXRUNTIME_DIR=%USERPROFILE%\onnxruntime
        echo [INFO] Found ONNX Runtime at: %ONNXRUNTIME_DIR%
    ) else if exist "C:\Program Files\onnxruntime\lib\onnxruntime.lib" (
        set ONNXRUNTIME_DIR=C:\Program Files\onnxruntime
        echo [INFO] Found ONNX Runtime at: %ONNXRUNTIME_DIR%
    ) else (
        echo.
        echo [ERROR] ONNX Runtime not found!
        echo.
        echo Please download ONNX Runtime from:
        echo   https://github.com/microsoft/onnxruntime/releases
        echo.
        echo Download: onnxruntime-win-x64-*.zip
        echo Extract to: %USERPROFILE%\onnxruntime
        echo.
        echo Or set ONNXRUNTIME_DIR environment variable:
        echo   set ONNXRUNTIME_DIR=C:\path\to\onnxruntime
        echo.
        pause
        exit /b 1
    )
)

echo [INFO] Using ONNX Runtime: %ONNXRUNTIME_DIR%
echo.

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
echo [1/3] Configuring with CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake configuration failed
    cd ..
    pause
    exit /b 1
)

REM Build
echo.
echo [2/3] Building...
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed
    cd ..
    pause
    exit /b 1
)

REM Install (copy to src directory)
echo.
echo [3/3] Installing...
cmake --install . --config Release
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Install failed
    cd ..
    pause
    exit /b 1
)

cd ..

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo DLL location: %cd%\onnx_inpainter.dll
echo.
echo To use the C++ backend, run:
echo   python onnx_cpp_backend.py
echo.
pause
