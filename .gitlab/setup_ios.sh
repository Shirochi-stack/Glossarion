#!/usr/bin/env bash
# Only disposable GitLab-hosted macOS; no certificates or keychains touched.
set -euo pipefail
export HOMEBREW_NO_AUTO_UPDATE=1
command -v uv >/dev/null || brew install uv
brew install coreutils pkg-config sqlite
export PATH="$(brew --prefix coreutils)/libexec/gnubin:$PATH"
uv python install 3.13
uv venv --python 3.13 "$CI_PROJECT_DIR/.gitlab-toolchain/host"
source "$CI_PROJECT_DIR/.gitlab-toolchain/host/bin/activate"
uv pip install pip
eval "$(python src/mobile/ci/wheels/wheels.py env --platform ios)"
rustup toolchain install "$RUST_VERSION" --profile minimal --target "$RUST_TARGETS"
rustup default "$RUST_VERSION"
xcodebuild -version
python --version
