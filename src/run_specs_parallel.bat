@echo off
echo Running spec files sequentially with PyInstaller...

echo.
echo Building 3 variants...

pyinstaller --clean translator_TurboLite.spec
pyinstaller --clean translator_lite.spec
pyinstaller --clean translator.spec
pyinstaller --clean translator_NoCuda.spec


echo.
echo Done!
echo Excluded: translator_Heavy.spec
echo.
echo Lite variant sizes:
echo   SuperLite  = Lite minus EPUB Library reader GUI/Chromium  (~228 MB)
echo   OmegaLite  = OpenAI-only, no PDF gen, no EPUB Library reader GUI  (~50-80 MB est.)
echo   NOTE: EPUB translation/processing is fully retained in ALL builds.
