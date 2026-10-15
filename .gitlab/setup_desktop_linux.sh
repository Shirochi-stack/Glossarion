#!/usr/bin/env bash
# Native prerequisites from build-linux.yml, adapted to the Debian CI image.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends build-essential python3-dev pkg-config cmake zip libpango1.0-dev libpangocairo-1.0-0 libcairo2-dev libgdk-pixbuf-2.0-dev libffi-dev libgirepository1.0-dev gir1.2-pango-1.0 libgl1 libegl1 libopengl0 libdbus-1-3 libnss3 libasound2 libgbm1 libxcomposite1 libxdamage1 libxrandr2 libxtst6 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libxcb-xinerama0 libxcb-image0 libxcb-render-util0
python -c 'import ctypes; ctypes.CDLL("libdbus-1.so.3"); ctypes.CDLL("libOpenGL.so.0")'
python -m pip install -r .gitlab/requirements.txt
