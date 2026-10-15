#!/usr/bin/env bash
# Each GitLab-hosted job has its own disposable macOS VM.
set -euo pipefail
export HOMEBREW_NO_AUTO_UPDATE=1
command -v uv >/dev/null || brew install uv
mkdir -p "$CI_PROJECT_DIR/.gitlab-toolchain"
if [ "${DESKTOP_ARCH:-arm64}" = x86_64 ]; then
  # A native x86_64 interpreter and native libraries are required: an ARM
  # Python with an Intel filename would not be an Intel package.
  arch -x86_64 /usr/bin/true || { echo 'Rosetta is unavailable on this hosted image.' >&2; return 1; }
  if [ ! -x /usr/local/bin/brew ]; then
    curl --fail --location --retry 3 https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh -o "$CI_PROJECT_DIR/.gitlab-toolchain/install-brew.sh"
    NONINTERACTIVE=1 arch -x86_64 /bin/bash "$CI_PROJECT_DIR/.gitlab-toolchain/install-brew.sh"
  fi
  arch -x86_64 /usr/local/bin/brew install create-dmg pango cairo libffi gdk-pixbuf gobject-introspection pkg-config cmake
  export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
  export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/opt/libffi/lib/pkgconfig"
  export DYLD_FALLBACK_LIBRARY_PATH=/usr/local/lib
  desktop_python=cpython-3.12-macos-x86_64-none
else
  brew install create-dmg pango cairo libffi gdk-pixbuf gobject-introspection pkg-config cmake
  desktop_python=cpython-3.12-macos-aarch64-none
fi
uv venv --python "$desktop_python" "$CI_PROJECT_DIR/.gitlab-toolchain/desktop"
source "$CI_PROJECT_DIR/.gitlab-toolchain/desktop/bin/activate"
uv pip install pip
python -m pip install -r .gitlab/requirements.txt
python -c 'import platform,sys; print(sys.version); print(platform.machine())'
