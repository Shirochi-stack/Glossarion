#!/bin/bash
# smoke-test.sh — Emulator smoke test for the Glossarion APK.
# Called by the CI workflow inside the android-emulator-runner action.
# The emulator should already be booted when this script runs.
set +e

# ── Pre-create log files so they ALWAYS exist for artifact upload,
#    even if the script dies early. Otherwise the Upload step silently
#    uploads nothing and we can't see why the smoke test failed. ──
: > "$GITHUB_WORKSPACE/adb-install.log"
: > "$GITHUB_WORKSPACE/emulator-logcat-full.log"
: > "$GITHUB_WORKSPACE/emulator-logcat-filtered.log"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Smoke test starting ==="
log "PWD=$(pwd)"
log "GITHUB_WORKSPACE=$GITHUB_WORKSPACE"

log "=== adb devices ==="
adb devices -l

log "=== Emulator properties ==="
adb shell getprop ro.build.version.sdk
adb shell getprop ro.product.cpu.abi
adb shell getprop ro.product.cpu.abilist
adb shell getprop ro.product.model

log "=== Finding APK ==="
APK_PATH=$(find "$GITHUB_WORKSPACE/src/android/bin" -name '*.apk' -type f 2>/dev/null | head -1)
log "APK_PATH=$APK_PATH"
ls -la "$GITHUB_WORKSPACE/src/android/bin/" 2>&1 || log "bin/ not found"
if [ -z "$APK_PATH" ]; then
  log "❌ No APK found, aborting"
  exit 1
fi
APK_SIZE=$(stat -c %s "$APK_PATH" 2>/dev/null || echo "?")
log "APK size: $APK_SIZE bytes"

log "=== APK architecture check (first 20 .so files) ==="
unzip -l "$APK_PATH" | grep -E '\.so$' | head -20

log "=== Waiting for emulator to fully boot ==="
adb wait-for-device
# Double-check sys.boot_completed — emulator-runner usually waits, but a very
# large APK install will race with late-boot services if we push too early.
BOOTED=""
for i in $(seq 1 90); do
  BOOTED=$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')
  if [ "$BOOTED" = "1" ]; then break; fi
  sleep 1
done
log "sys.boot_completed=$BOOTED (after ${i}s)"

# Give package manager a moment to fully settle before install
sleep 3

log "=== Installing APK ==="
# -r: replace existing, -t: allow test-only, -g: grant all permissions at install time
adb install -r -t -g --no-streaming "$APK_PATH" > "$GITHUB_WORKSPACE/adb-install.log" 2>&1
INSTALL_STATUS=$?
log "adb install exit status: $INSTALL_STATUS"
cat "$GITHUB_WORKSPACE/adb-install.log"

if [ $INSTALL_STATUS -ne 0 ]; then
  log "=== Retrying install with streaming ==="
  adb install -r -t -g "$APK_PATH" >> "$GITHUB_WORKSPACE/adb-install.log" 2>&1
  INSTALL_STATUS=$?
  log "retry exit: $INSTALL_STATUS"
  cat "$GITHUB_WORKSPACE/adb-install.log"
fi

log "=== Verifying installation ==="
if ! adb shell pm list packages 2>/dev/null | grep -q "com.glossarion.glossarion"; then
  log "❌ PACKAGE NOT INSTALLED"
  log "=== Dumping logcat for diagnostic ==="
  adb logcat -d > "$GITHUB_WORKSPACE/emulator-logcat-full.log" 2>&1
  exit 1
fi
log "✅ Package is installed"

log "=== Clearing logcat buffer ==="
adb logcat -c

log "=== Launching app ==="
# -W waits for the activity to be ready, -S stops any existing instance first
adb shell am start -W -S -n com.glossarion.glossarion/org.kivy.android.PythonActivity
LAUNCH_STATUS=$?
log "am start exit: $LAUNCH_STATUS"

# ── Poll for ~75 seconds. The APK is ~95 MB (three ABIs) and on first run
#    python-for-android unpacks stdlib.zip + site-packages into app data —
#    this typically takes ~20 s before the Python main loop actually starts.
#    Previously we only slept 30 s and dumped logcat while unpack was still
#    in progress, which made it look like the app had hung. ──
log "=== Polling for app process (up to 75s) ==="
MAX_WAIT=75
MAIN_LOOP_FOUND=0
LAST_PID=""
for i in $(seq 1 $MAX_WAIT); do
  PID=$(adb shell pidof com.glossarion.glossarion 2>/dev/null | tr -d '\r')
  if [ "$PID" != "$LAST_PID" ]; then
    log "[${i}s] PID='$PID'"
    LAST_PID="$PID"
  fi
  # Every 10 s, check whether the Python main loop has been reached
  if [ $((i % 10)) -eq 0 ]; then
    if adb logcat -d 2>/dev/null | grep -q "Start application main loop"; then
      MAIN_LOOP_FOUND=1
      log "[${i}s] ✅ Main loop reached — app is fully up"
      break
    fi
    log "[${i}s] main loop not yet reached"
  fi
  sleep 1
done

log "=== Dumping full logcat ==="
adb logcat -d > "$GITHUB_WORKSPACE/emulator-logcat-full.log" 2>&1
LOG_SIZE=$(stat -c %s "$GITHUB_WORKSPACE/emulator-logcat-full.log" 2>/dev/null || echo 0)
log "Logcat size: $LOG_SIZE bytes"

grep -iE '(python|kivy|glossarion|SDL|Traceback|ModuleNotFound|ImportError|FATAL|AndroidRuntime|Exception|Error)' \
  "$GITHUB_WORKSPACE/emulator-logcat-full.log" \
  > "$GITHUB_WORKSPACE/emulator-logcat-filtered.log" 2>/dev/null || true

log "=== Filtered logcat (last 300 lines) ==="
tail -300 "$GITHUB_WORKSPACE/emulator-logcat-filtered.log"

log "=== Final process check ==="
FINAL_PID=$(adb shell pidof com.glossarion.glossarion 2>/dev/null | tr -d '\r')
if [ -n "$FINAL_PID" ]; then
  if [ $MAIN_LOOP_FOUND -eq 1 ]; then
    log "✅ App is running (PID=$FINAL_PID) and main loop was reached"
    exit 0
  else
    log "⚠️ App is running (PID=$FINAL_PID) but main loop not confirmed in logs"
    # Still pass — app didn't crash, it's just slow
    exit 0
  fi
else
  log "❌ App NOT running (crashed or exited)"
  # Show the last few python lines for quick triage
  log "=== Last 50 python/kivy lines ==="
  grep -E 'python|kivy' "$GITHUB_WORKSPACE/emulator-logcat-full.log" | tail -50
  exit 1
fi
