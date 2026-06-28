@echo off
echo Running spec files sequentially with PyInstaller...
echo.
echo [1/5] Building TurboLite (smallest - no Vertex AI, no EPUB reader, no PDF)...
pyinstaller --clean translator_TurboLite.spec

echo.
echo [2/5] Building Lite (Vertex AI, no EPUB reader, no PDF)...
pyinstaller --clean translator_lite.spec

echo.
echo [3/5] Building (full novel translaton build)...
pyinstaller --clean translator.spec

echo.
echo [4/5] Building NoCuda (full Manga build)...
pyinstaller --clean translator_NoCuda.spec

echo.
echo [4/5] Building Structured file (Performance build)...
pyinstaller --clean translatoronefileoff.spec

echo.
echo Done!
echo Excluded: translator.spec (Standard/bloat), translator_Heavy.spec
echo.
echo Package structure (smallest to largest):
echo   TurboLite  = no Vertex AI, no EPUB reader, no PDF
echo   Lite       = Vertex AI included, no EPUB reader, no PDF
echo   NoCuda     = full Manga translation build
