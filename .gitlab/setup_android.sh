#!/usr/bin/env bash
# Bootstrap only the disposable hosted Linux job. No runner registration or secrets.
# Source this script so the subsequent Python bridge inherits SDK/Rust paths.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends ca-certificates curl unzip zip git build-essential pkg-config sqlite3 libsqlite3-dev openjdk-17-jdk-headless
python -m pip install uv
export ANDROID_HOME="$CI_PROJECT_DIR/.gitlab-toolchain/android"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")"
export CARGO_HOME="$CI_PROJECT_DIR/.gitlab-toolchain/cargo"
export RUSTUP_HOME="$CI_PROJECT_DIR/.gitlab-toolchain/rustup"
export PATH="$CARGO_HOME/bin:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH"
python .gitlab/setup_android_sdk.py
# sdkmanager reports success only after licenses are accepted, just as the
# original Actions/Flet build environment does for these build dependencies.
set +o pipefail
yes | sdkmanager --licenses > /dev/null
set -o pipefail
sdkmanager 'platform-tools' 'platforms;android-35' 'build-tools;35.0.0'
eval "$(python src/mobile/ci/wheels/wheels.py env --platform android)"
curl --fail --location --retry 3 https://sh.rustup.rs -o "$CI_PROJECT_DIR/.gitlab-toolchain/rustup-init.sh"
bash "$CI_PROJECT_DIR/.gitlab-toolchain/rustup-init.sh" -y --profile minimal --default-toolchain "$RUST_VERSION" --target "$RUST_TARGETS" --no-modify-path
rustc --version
java -version
