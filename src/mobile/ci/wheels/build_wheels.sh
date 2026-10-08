#!/usr/bin/env bash
# Build the self-built mobile wheels of one platform with flet-dev/mobile-forge.
#
# Usage: build_wheels.sh <android|ios> <package>...
#   (packages from wheelhouse.toml: android = cryptography pillow, ios = cryptography)
#
# Run by .github/workflows/mobile-wheels.yml on ubuntu-24.04 (android) or macos-26 (ios) after the
# toolchain steps (Rust targets; Xcode on iOS). Every pinned input comes from wheelhouse.toml
# through wheels.py, so this script holds no versions or hashes of its own:
#   1. wheels.py env: forge + crossenv revisions, python-build release, CPython 3.13.15, ABIs,
#      Rust targets and the PIP_CONSTRAINT file for forge's build environments;
#   2. clone mobile-forge at the pinned commit and pin its crossenv dependency to a commit;
#   3. wheels.py fetch: the support tarball, BeeWare OpenSSL 3.5.9 and the sdists (from uv.lock),
#      sha256-checked and stored under the names forge expects, so forge downloads nothing itself;
#   4. Android: the pinned NDK (forge's .ci/install_ndk.sh, a no-op when installed) -> NDK_HOME;
#   5. source forge's setup.sh: support tree, uv venv on CPython 3.13.15, dependency wheels;
#   6. wheels.py stage-openssl: static OpenSSL 3.5.9 replaces the support tree's 3.0.x, and every
#      other libcrypto.a/libssl.a in the support tree is deleted (see "Link order" below);
#   7. iOS: include/<arch>-<sdk> -> python3.13 symlinks, because cryptography-cffi's build.rs
#      derives Python.h's directory from PYO3_CROSS_LIB_DIR (platform-config/<arch>-<sdk>);
#   8. forge <host> <recipe directory> for each package (src/mobile/ci/wheels/recipes/<package>);
#   9. wheels.py collect: only the promised wheels go to $WHEELCACHE/<package>/.
#
# Environment: FORGE_DIR (default $RUNNER_TEMP/forge), WHEELCACHE (default $RUNNER_TEMP/wheelcache).
# Needs git, uv (a Python 3.11+ for wheels.py) and, on Android, ANDROID_HOME with sdkmanager.
#
# What reaches the compile: forge's compile_env() builds the environment of the `python -m build`
# (maturin, cargo, rustc, the C compiler) from scratch: its own keys, the recipe's
# build.script_env, and TMPDIR/USER/HOME/LANG/TERM. Nothing exported here or in the workflow gets
# there, so per-platform compile settings live in recipes/<package>/meta.yaml (OPENSSL_STATIC on
# Android, IPHONEOS_DEPLOYMENT_TARGET on iOS). Read by forge's own process or inherited by its
# pip installs, and so effective from here: NDK_HOME, MOBILE_FORGE_*_SUPPORT_PATH (setup.sh),
# PIP_CONSTRAINT. rustc comes from rustup's default toolchain (`rustup default`, found via HOME).
#
# Link order: forge sets CARGO_TARGET_<triple>_RUSTFLAGS to " -L{prefix}/lib ..." and cargo puts
# RUSTFLAGS before the -L native=$OPENSSL_DIR/lib of openssl-sys's build script, so its
# `-l static=crypto` / `static=ssl` take the first libcrypto.a / libssl.a on that path. On Android
# {prefix}/lib is the support tree's install/android/<abi>/python-3.13.x/lib, which ships CPython's
# static OpenSSL 3.0.x. A module compiled against the 3.5.9 headers got linked with it, the 3.2+
# symbols stayed undefined and dlopen failed on the device. Hence stage-openssl's deletion, the
# recipe's `-z defs` (Android) and wheels.py verify-native's import checks (every build and cache hit).
set -euo pipefail

die() {
  echo "::error title=Mobile wheels::$*" >&2
  exit 1
}

platform="${1:-}"
case "$platform" in
  android|ios) ;;
  *) echo "usage: build_wheels.sh <android|ios> <package>..." >&2; exit 2 ;;
esac
shift
[ "$#" -gt 0 ] || { echo "build_wheels.sh: no packages given" >&2; exit 2; }
packages=("$@")

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wheels_py="$here/wheels.py"
tmp_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
FORGE_DIR="${FORGE_DIR:-$tmp_root/forge}"
WHEELCACHE="${WHEELCACHE:-$tmp_root/wheelcache}"
case "$FORGE_DIR" in
  *" "*) die "FORGE_DIR must not contain spaces ($FORGE_DIR)" ;;
esac

host_py() {
  # A Python 3.11+ for wheels.py until forge's own venv exists.
  uv run --no-project python "$@"
}

echo "::group::Pinned build inputs ($platform: ${packages[*]})"
env_sh="$(host_py "$wheels_py" env --platform "$platform" --constraints-out "$tmp_root/mobile-wheels-constraints.txt")"
echo "$env_sh"
eval "$env_sh"
recipes=()
for pkg in "${packages[@]}"; do
  recipes+=("$(host_py "$wheels_py" recipe --platform "$platform" "$pkg")")
done
echo "::endgroup::"

echo "::group::mobile-forge $FORGE_REV (crossenv $CROSSENV_REV)"
rm -rf "$FORGE_DIR"
git init -q "$FORGE_DIR"
git -C "$FORGE_DIR" remote add origin "$FORGE_REPO"
git -C "$FORGE_DIR" fetch -q --depth 1 origin "$FORGE_REV"
git -C "$FORGE_DIR" checkout -q --detach FETCH_HEAD
[ "$(git -C "$FORGE_DIR" rev-parse HEAD)" = "$FORGE_REV" ] || die "mobile-forge checkout is not $FORGE_REV"
cd "$FORGE_DIR"
# forge depends on the moving branch crossenv@flet; build against a fixed commit instead.
sed -i.bak "s#crossenv@flet\"#crossenv@${CROSSENV_REV}\"#" pyproject.toml
rm -f pyproject.toml.bak
grep -q -F "\"crossenv @ git+${CROSSENV_REPO}@${CROSSENV_REV}\"" pyproject.toml \
  || die "could not pin crossenv in mobile-forge's pyproject.toml"
echo "::endgroup::"

echo "::group::Fetch and verify the build inputs"
host_py "$wheels_py" fetch --platform "$platform" --dest "$FORGE_DIR/downloads" "${packages[@]}"
echo "::endgroup::"

if [ "$platform" = android ]; then
  echo "::group::Android NDK $ANDROID_NDK_VERSION"
  : "${ANDROID_HOME:?ANDROID_HOME must point at the Android SDK}"
  bash .ci/install_ndk.sh "$ANDROID_NDK_VERSION"
  export NDK_HOME="$ANDROID_HOME/ndk/$ANDROID_NDK_VERSION"
  compgen -G "$NDK_HOME/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android24-clang" > /dev/null \
    || die "NDK $ANDROID_NDK_VERSION has no aarch64-linux-android24-clang under $NDK_HOME"
  echo "::endgroup::"
fi

echo "::group::forge setup.sh $PY_FULL $FORGE_HOST (python-build $PYTHON_BUILD_RELEASE)"
# setup.sh reads the support tarball from downloads/ (fetched and verified above), creates the uv
# venv on exactly CPython $PY_FULL and runs make_dep_wheels. It is written for interactive use
# (plain `return` on errors), so check its results instead of trusting its status.
unset UV_PYTHON
set +eu
# shellcheck disable=SC1091
source ./setup.sh "$PY_FULL" "$FORGE_HOST"
set -eu
[ -n "${VIRTUAL_ENV:-}" ] || die "setup.sh did not activate the forge venv"
command -v forge > /dev/null || die "setup.sh did not install forge"
venv_py="$(python -c 'import platform; print(platform.python_version())')"
[ "$venv_py" = "$PY_FULL" ] || die "the forge venv runs Python $venv_py, not $PY_FULL"
if [ "$platform" = android ]; then
  [ -n "${MOBILE_FORGE_ANDROID_SUPPORT_PATH:-}" ] || die "setup.sh did not set MOBILE_FORGE_ANDROID_SUPPORT_PATH"
else
  [ -n "${MOBILE_FORGE_IOS_SUPPORT_PATH:-}" ] || die "setup.sh did not set MOBILE_FORGE_IOS_SUPPORT_PATH"
fi
echo "::endgroup::"

echo "::group::Static OpenSSL for cryptography"
python "$wheels_py" stage-openssl --platform "$platform" --downloads "$FORGE_DIR/downloads" --forge "$FORGE_DIR" \
  "${packages[@]}"
echo "::endgroup::"

if [ "$platform" = ios ]; then
  echo "::group::Python.h for cryptography-cffi (iOS)"
  xcf="$MOBILE_FORGE_IOS_SUPPORT_PATH/support/$PY_SHORT/iOS/Python.xcframework"
  for pair in "ios-arm64:arm64-iphoneos" "ios-arm64_x86_64-simulator:arm64-iphonesimulator" \
              "ios-arm64_x86_64-simulator:x86_64-iphonesimulator"; do
    slice="${pair%%:*}"
    name="${pair#*:}"
    [ -f "$xcf/$slice/include/python$PY_SHORT/Python.h" ] || die "no $xcf/$slice/include/python$PY_SHORT/Python.h"
    ln -sfn "python$PY_SHORT" "$xcf/$slice/include/$name"
    [ -f "$xcf/$slice/include/$name/Python.h" ] || die "$xcf/$slice/include/$name does not resolve to Python.h"
    echo "$slice/include/$name -> python$PY_SHORT"
  done
  echo "::endgroup::"
fi

for recipe in "${recipes[@]}"; do
  echo "::group::forge $FORGE_HOST $recipe"
  forge "$FORGE_HOST" "$recipe"
  echo "::endgroup::"
done

echo "::group::Collect the promised wheels"
python "$wheels_py" collect --platform "$platform" --dist "$FORGE_DIR/dist" --logs "$FORGE_DIR/logs" \
  --cache "$WHEELCACHE" "${packages[@]}"
echo "::endgroup::"
