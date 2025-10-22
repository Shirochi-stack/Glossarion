@echo off
REM Build script for ONNX Inpainter C++ library (Windows)
REM This script compiles the thread-safe C++ backend

echo ========================================
echo  ONNX Inpainter C++ Builder
echo ========================================
echo.

REM Check if CMake is available
where cmake >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CMake not found in PATH
    echo Please install CMake: https://cmake.org/download/
    pause
    exit /b 1
)

REM Detect compiler
echo Detecting available compiler...
where cl >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Found MSVC compiler
    set GENERATOR="Visual Studio 17 2022"
    goto :build
)

where g++ >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Found MinGW/GCC compiler
    set GENERATOR="MinGW Makefiles"
    goto :build
)

echo ERROR: No suitable C++ compiler found
echo Please install either:
echo   - Visual Studio 2022 (with C++ tools)
echo   - MinGW-w64
pause
exit /b 1

:build
echo.
echo Building with %GENERATOR%...
echo.

REM Create build directory
if not exist build mkdir build
cd build

REM Configure
echo [1/3] Configuring CMake...
cmake .. -G %GENERATOR% -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: CMake configuration failed
    echo Check that ONNX Runtime is installed and ONNXRUNTIME_ROOT is set correctly
    cd ..
    pause
    exit /b 1
)

REM Build
echo.
echo [2/3] Building library...
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed
    cd ..
    pause
    exit /b 1
)

REM Copy to parent directory
echo.
echo [3/3] Copying library to src directory...
cd ..

if exist build\Release\onnx_inpainter.dll (
    copy /Y build\Release\onnx_inpainter.dll onnx_inpainter.dll
    echo [OK] Copied: onnx_inpainter.dll
) else if exist build\onnx_inpainter.dll (
    copy /Y build\onnx_inpainter.dll onnx_inpainter.dll
    echo [OK] Copied: onnx_inpainter.dll
) else (
    echo WARNING: Could not find compiled DLL
)

echo.
echo ========================================
echo  Build Complete!
echo ========================================
echo.
echo The thread-safe ONNX backend is ready.
echo You can now run your manga translation pipeline.
echo.
pause
