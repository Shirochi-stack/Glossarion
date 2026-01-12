@echo off
echo Running spec files sequentially with PyInstaller...

pyinstaller --clean translator_lite.spec
pyinstaller --clean translator.spec
pyinstaller --clean translator_NoCuda.spec

echo Done!
echo Excluded: translator_Heavy.spec
