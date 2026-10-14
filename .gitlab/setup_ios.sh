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
# The image's Homebrew Rust path is discarded by forge's cross environment.
# Install into the standard location in this disposable VM, which forge retains.
export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"
export PATH="$CARGO_HOME/bin:$PATH"
curl --fail --location --retry 3 https://sh.rustup.rs -o "$CI_PROJECT_DIR/.gitlab-toolchain/rustup-init.sh"
bash "$CI_PROJECT_DIR/.gitlab-toolchain/rustup-init.sh" -y --profile minimal --default-toolchain "$RUST_VERSION" --target "$RUST_TARGETS" --no-modify-path
env -i HOME="$HOME" PATH="$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin" cargo --version
xcodebuild -version
python --version
