#!/usr/bin/env bash
# Android UI tests for Glossarion Mobile: `flet test android` (src/mobile/tests) on a booted emulator.
#
# Usage: android_ui_tests.sh [log-dir]        (CI: the android-ui-tests job of build-mobile.yml)
#
#  1. picks the emulator (ANDROID_SERIAL, else the first `adb devices` entry) and streams logcat;
#  2. runs `uv run --locked flet test android --device-id <serial>` from src/mobile: flet builds a
#     debug test host with the app embedded (the same pins; PIP_FIND_LINKS = the self-built
#     wheelhouse), installs it and runs pytest on tests/ with the flet_app fixture;
#  3. the tests write screenshots, uiautomator dumps and the step log into the log dir
#     (GLOSSARION_UI_ARTIFACTS); junit.xml and logcat land there too.
#
# Environment overrides: GLOSSARION_UI_TIMEOUT (60 s per finder), UI_TESTS_K (pytest -k),
# SCIKIT_IMAGE (the flet[test] extra flet.testing imports; pinned here, not in uv.lock).
set -euo pipefail

LOG_DIR="${1:-$PWD/mobile-ui-test-logs}"
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
SCIKIT_IMAGE="${SCIKIT_IMAGE:-scikit-image==0.25.2}"

serial="${ANDROID_SERIAL:-}"
if [ -z "$serial" ]; then
  serial="$(adb devices | awk 'NR > 1 && $2 == "device" { print $1; exit }')"
fi
if [ -z "$serial" ]; then
  echo "::error title=Android UI tests::no booted emulator or device (adb devices)"
  exit 1
fi
export ANDROID_SERIAL="$serial"
echo "Using $serial"

adb -s "$serial" logcat -c || true
adb -s "$serial" logcat -v threadtime > "$LOG_DIR/logcat.txt" 2>&1 &
logcat_pid=$!
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() {
  kill "$logcat_pid" 2>/dev/null || true
  adb -s "$serial" exec-out screencap -p > "$LOG_DIR/final_screen.png" 2>/dev/null || true
}
trap cleanup EXIT

export GLOSSARION_UI_ARTIFACTS="$LOG_DIR"
GLOSSARION_UI_ADB="$(command -v adb)"
export GLOSSARION_UI_ADB
export FLET_CLI_NO_RICH_OUTPUT=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8

cd "$(dirname "$0")/.."
pytest_args=(-p no:cacheprovider -o console_output_style=classic -rA --junitxml="$LOG_DIR/junit.xml")
if [ -n "${UI_TESTS_K:-}" ]; then
  pytest_args+=(-k "$UI_TESTS_K")
fi
rc=0
uv run --locked --with "$SCIKIT_IMAGE" flet test android --device-id "$serial" --yes -v -- "${pytest_args[@]}" \
  2>&1 | tee "$LOG_DIR/flet_test.txt" || rc=$?
echo "flet test exit status: $rc"
exit "$rc"
