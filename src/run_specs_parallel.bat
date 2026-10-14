@echo off
setlocal
pushd "%~dp0"

echo Running spec files sequentially with PyInstaller...
echo.
echo [1/5] Building TurboLite (smallest - no Vertex AI, no EPUB reader, no PDF)...
python -m PyInstaller --clean translator_TurboLite.spec
if errorlevel 1 goto :build_failed

echo.
echo [2/5] Building Lite (Vertex AI, no EPUB reader, no PDF)...
python -m PyInstaller --clean translator_lite.spec
if errorlevel 1 goto :build_failed

echo.
echo [3/5] Building (full novel translaton build)...
python -m PyInstaller --clean translator.spec
if errorlevel 1 goto :build_failed

echo.
echo [4/5] Building NoCuda (full Manga build)...
python -m PyInstaller --clean translator_NoCuda.spec
if errorlevel 1 goto :build_failed

echo.
echo [5/5] Building Structured file (Performance build)...
python -m PyInstaller --clean translatoronefileoff.spec
if errorlevel 1 goto :build_failed

echo.
echo Done!
echo Excluded: translator_Heavy.spec
echo.
echo Package structure (smallest to largest):
echo   TurboLite  = no Vertex AI, no EPUB reader, no PDF
echo   Lite       = Vertex AI included, no EPUB reader, no PDF
echo   NoCuda     = full Manga translation build
popd
exit /b 0

:build_failed
set "build_exit=%errorlevel%"
echo.
echo Build failed with exit code %build_exit%.
popd
exit /b %build_exit%
