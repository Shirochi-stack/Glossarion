# Disposable GitLab-hosted Windows setup. No signing or credential installation.
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$toolRoot = Join-Path $env:CI_PROJECT_DIR '.gitlab-toolchain'
New-Item -ItemType Directory -Path $toolRoot -Force | Out-Null
$uvArchive = Join-Path $toolRoot 'uv.zip'
Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $uvArchive
Expand-Archive -LiteralPath $uvArchive -DestinationPath (Join-Path $toolRoot 'uv') -Force
$uvExecutable = (Get-ChildItem -LiteralPath (Join-Path $toolRoot 'uv') -Filter uv.exe -Recurse | Select-Object -First 1).FullName
if (-not $uvExecutable) { throw 'Downloaded uv executable is missing.' }
$desktopVenv = Join-Path $toolRoot 'desktop'
& $uvExecutable venv --python 3.12 $desktopVenv
$env:PATH = (Join-Path $desktopVenv 'Scripts') + ';' + $env:PATH
& $uvExecutable pip install --python (Join-Path $desktopVenv 'Scripts\python.exe') pip
python -m pip install -r .gitlab/requirements.txt
$msysArchive = Join-Path $toolRoot 'msys2.tar.xz'
Invoke-WebRequest -Uri 'https://repo.msys2.org/distrib/msys2-x86_64-latest.tar.xz' -OutFile $msysArchive
tar.exe -xf $msysArchive -C $toolRoot
$env:MSYS2_ROOT = Join-Path $toolRoot 'msys64'
$env:MSYSTEM = 'MINGW64'
$env:CHERE_INVOKING = '1'
$msysBash = Join-Path $env:MSYS2_ROOT 'usr\bin\bash.exe'
& $msysBash --login -c 'pacman -Syu --noconfirm'
& $msysBash --login -c 'pacman -Su --noconfirm'
& $msysBash --login -c 'pacman -S --noconfirm --needed mingw-w64-x86_64-gtk3 mingw-w64-x86_64-pango mingw-w64-x86_64-cairo mingw-w64-x86_64-gdk-pixbuf2 mingw-w64-x86_64-libffi mingw-w64-x86_64-glib2 mingw-w64-x86_64-harfbuzz mingw-w64-x86_64-fribidi mingw-w64-x86_64-fontconfig mingw-w64-x86_64-freetype'
python --version
