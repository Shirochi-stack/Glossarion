#!/usr/bin/env bash
# Android emulator smoke test for Glossarion Mobile.
#
# Usage: android_smoke.sh <x86_64.apk> [log-dir]
#
#  1. install the APK (runtime permissions granted) and launch com.glossarion.app;
#  2. wait for GLOSSARION_READY and GLOSSARION_BACKEND_READY in `adb logcat -s flet.python`;
#  3. fire the self-test deep link glossarion://app/__selftest__?suite=smoke;
#  4. wait for GLOSSARION_SELFTEST PASS (FAIL or timeout exits 1);
#  5. always save logcat (flet.python + full + crash buffer), meminfo, pidof, package dump and a
#     screenshot into the log dir.
#
# Environment overrides: GLOSSARION_PACKAGE, READY_TIMEOUT (300), SELFTEST_TIMEOUT (900),
# SELFTEST_URL, LOGCAT_TAG (flet.python).
set -uo pipefail

APK="${1:?usage: android_smoke.sh <x86_64.apk> [log-dir]}"
LOG_DIR="${2:-mobile-smoke-logs}"
PKG="${GLOSSARION_PACKAGE:-com.glossarion.app}"
READY_TIMEOUT="${READY_TIMEOUT:-300}"
SELFTEST_TIMEOUT="${SELFTEST_TIMEOUT:-900}"
SELFTEST_URL="${SELFTEST_URL:-glossarion://app/__selftest__?suite=smoke}"
TAG="${LOGCAT_TAG:-flet.python}"

mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
TAG_LOG="$LOG_DIR/logcat_flet_python.txt"
FULL_LOG="$LOG_DIR/logcat_full_stream.txt"
SUMMARY="$LOG_DIR/summary.txt"
: > "$TAG_LOG"
: > "$FULL_LOG"
: > "$SUMMARY"

RESULT="FAIL"
STAGE="setup"
STREAM_PIDS=()

log() {
  printf '[smoke %s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$SUMMARY"
}

fail() {
  echo "::error title=Android smoke (${STAGE})::$*"
  log "FAIL during ${STAGE}: $*"
  exit 1
}

start_streams() {
  # Two followers: the Python tag only (what the contract promises) and everything (fallback +
  # crash context). Appending keeps earlier output when a follower is restarted.
  adb logcat -v threadtime -s "$TAG" >> "$TAG_LOG" 2>&1 &
  STREAM_PIDS[0]=$!
  adb logcat -v threadtime >> "$FULL_LOG" 2>&1 &
  STREAM_PIDS[1]=$!
}

ensure_streams() {
  local pid
  for pid in "${STREAM_PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "logcat follower exited; restarting followers"
      stop_streams
      start_streams
      return
    fi
  done
}

stop_streams() {
  local pid
  for pid in "${STREAM_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}

marker_regex() {
  # Whole-marker match: GLOSSARION_READY must not match GLOSSARION_BACKEND_READY's tail etc.
  printf '(^|[^A-Z_])%s([^A-Z_]|$)' "$1"
}

# Returns 0 when the marker is in the flet.python log, 2 when it is only in the full log,
# 1 when it is nowhere.
find_marker() {
  local regex
  regex="$(marker_regex "$1")"
  if grep -a -q -E "$regex" "$TAG_LOG"; then
    return 0
  fi
  if grep -a -q -E "$regex" "$FULL_LOG"; then
    return 2
  fi
  return 1
}

app_pid() {
  adb shell pidof "$PKG" 2>/dev/null | tr -d '\r' | awk '{print $1}'
}

# wait_for_marker <marker> <timeout-seconds>
wait_for_marker() {
  local marker="$1" timeout="$2" start=$SECONDS rc dead_checks=0 pid
  while true; do
    find_marker "$marker"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      log "found '$marker' in $TAG after $((SECONDS - start))s"
      return 0
    fi
    if [ "$rc" -eq 2 ]; then
      log "found '$marker' after $((SECONDS - start))s, but NOT under logcat tag $TAG (see logcat_full_stream.txt)"
      echo "::warning title=Android smoke::'$marker' was logged outside the $TAG logcat tag; check how the app prints its markers."
      return 0
    fi
    find_marker 'GLOSSARION_SELFTEST FAIL'
    if [ $? -ne 1 ]; then
      grep -a -h -E "$(marker_regex 'GLOSSARION_SELFTEST FAIL')" "$TAG_LOG" "$FULL_LOG" | tail -n 1 > "$LOG_DIR/selftest_result.txt"
      fail "the app reported GLOSSARION_SELFTEST FAIL: $(cut -c1-500 "$LOG_DIR/selftest_result.txt")"
    fi
    # runtime_bootstrap prints GLOSSARION_BACKEND_FAIL {json} when the warm backend import fails.
    find_marker 'GLOSSARION_BACKEND_FAIL'
    if [ $? -ne 1 ]; then
      grep -a -h -E "$(marker_regex 'GLOSSARION_BACKEND_FAIL')" "$TAG_LOG" "$FULL_LOG" | tail -n 1 > "$LOG_DIR/backend_fail.txt"
      fail "the app reported GLOSSARION_BACKEND_FAIL: $(cut -c1-500 "$LOG_DIR/backend_fail.txt")"
    fi
    pid="$(app_pid)"
    if [ -z "$pid" ]; then
      dead_checks=$((dead_checks + 1))
      # A few consecutive misses: the process really is gone (not just restarting an activity).
      if [ "$dead_checks" -ge 3 ]; then
        fail "$PKG is not running while waiting for '$marker' (crashed or exited; see logcat_crash.txt)"
      fi
    else
      dead_checks=0
    fi
    if [ $((SECONDS - start)) -ge "$timeout" ]; then
      if ! grep -a -q -E 'GLOSSARION_[A-Z_]+' "$TAG_LOG" "$FULL_LOG"; then
        fail "timed out after ${timeout}s waiting for '$marker', and no GLOSSARION_* marker reached logcat at all. Flet forwards sys.stdout/sys.stderr (print) to the $TAG tag; writes to sys.__stderr__ (fd 2) go to /dev/null on Android."
      fi
      fail "timed out after ${timeout}s waiting for '$marker'"
    fi
    ensure_streams
    sleep 5
  done
}

# shellcheck disable=SC2329  # invoked through the EXIT trap below
collect_diagnostics() {
  local exit_code=$?
  STAGE="diagnostics"
  stop_streams
  adb logcat -d -v threadtime > "$LOG_DIR/logcat_dump.txt" 2>&1 || true
  adb logcat -d -v threadtime -b crash > "$LOG_DIR/logcat_crash.txt" 2>&1 || true
  adb shell dumpsys meminfo "$PKG" > "$LOG_DIR/meminfo.txt" 2>&1 || true
  {
    echo "pidof $PKG: $(app_pid)"
    adb shell ps -A -o PID,RSS,NAME 2>/dev/null | grep -F "$PKG" || true
  } > "$LOG_DIR/pidof.txt" 2>&1
  adb shell dumpsys package "$PKG" > "$LOG_DIR/dumpsys_package.txt" 2>&1 || true
  adb shell dumpsys activity activities 2>/dev/null | grep -E "mResumedActivity|topResumedActivity|$PKG" | head -n 50 > "$LOG_DIR/activities.txt" || true
  adb shell getprop > "$LOG_DIR/getprop.txt" 2>&1 || true
  adb exec-out screencap -p > "$LOG_DIR/screen.png" 2>/dev/null || rm -f "$LOG_DIR/screen.png"
  grep -a -h -o -E 'GLOSSARION_[A-Z_]+( (PASS|FAIL))?' "$TAG_LOG" "$FULL_LOG" 2>/dev/null | sort | uniq -c > "$LOG_DIR/markers.txt" || true
  grep -a -c -F 'Traceback (most recent call last)' "$TAG_LOG" > "$LOG_DIR/python_tracebacks_count.txt" 2>/dev/null || true
  log "result: $RESULT (exit $exit_code)"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
      echo "## Android emulator smoke: $RESULT"
      echo
      echo '```'
      cat "$SUMMARY"
      echo
      echo "Markers seen:"
      cat "$LOG_DIR/markers.txt" 2>/dev/null
      echo '```'
    } >> "$GITHUB_STEP_SUMMARY"
  fi
  exit "$exit_code"
}
trap collect_diagnostics EXIT

[ -f "$APK" ] || fail "APK not found: $APK"
command -v adb > /dev/null || fail "adb is not on PATH"

STAGE="boot"
log "waiting for the emulator"
timeout 300 adb wait-for-device || fail "no device after 300s"
boot_start=$SECONDS
until [ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ]; do
  if [ $((SECONDS - boot_start)) -ge 300 ]; then
    fail "sys.boot_completed did not become 1 within 300s"
  fi
  sleep 3
done
log "device: $(adb shell getprop ro.product.model | tr -d '\r'), Android $(adb shell getprop ro.build.version.release | tr -d '\r') (API $(adb shell getprop ro.build.version.sdk | tr -d '\r')), ABI $(adb shell getprop ro.product.cpu.abi | tr -d '\r')"
adb shell input keyevent 82 > /dev/null 2>&1 || true   # wake/unlock
adb shell svc power stayon true > /dev/null 2>&1 || true

STAGE="install"
log "installing $(basename "$APK") ($(du -h "$APK" | cut -f1))"
installed=false
for attempt in 1 2 3; do
  if timeout 600 adb install -r -g "$APK" > "$LOG_DIR/adb_install.txt" 2>&1; then
    installed=true
    break
  fi
  log "adb install attempt $attempt failed: $(tail -n 1 "$LOG_DIR/adb_install.txt")"
  sleep 10
done
[ "$installed" = true ] || fail "adb install failed: $(tail -n 3 "$LOG_DIR/adb_install.txt" | tr '\n' ' ')"
adb shell pm list packages "$PKG" | tr -d '\r' | grep -q -x "package:$PKG" || fail "$PKG is not installed after adb install"
adb shell dumpsys deviceidle whitelist "+$PKG" > /dev/null 2>&1 || true

adb logcat -G 16M > /dev/null 2>&1 || true
adb logcat -c > /dev/null 2>&1 || true
start_streams

STAGE="launch"
log "launching $PKG"
if ! adb shell monkey -p "$PKG" -c android.intent.category.LAUNCHER 1 > "$LOG_DIR/launch.txt" 2>&1; then
  log "monkey launch failed; trying am start on the resolved launcher activity"
  activity="$(adb shell cmd package resolve-activity --brief "$PKG" | tr -d '\r' | tail -n 1)"
  adb shell am start -W -n "$activity" >> "$LOG_DIR/launch.txt" 2>&1 || fail "could not launch $PKG"
fi

STAGE="ready"
wait_for_marker GLOSSARION_READY "$READY_TIMEOUT"
wait_for_marker GLOSSARION_BACKEND_READY "$READY_TIMEOUT"

STAGE="selftest"
log "starting self-test: $SELFTEST_URL"
# adb shell joins its arguments into one device-side command line: keep the URL single-quoted
# so the device shell does not glob the '?'.
adb shell "am start -W -a android.intent.action.VIEW -d '$SELFTEST_URL' $PKG" > "$LOG_DIR/selftest_intent.txt" 2>&1 \
  || fail "am start for the self-test deep link failed: $(tr '\n' ' ' < "$LOG_DIR/selftest_intent.txt")"
if grep -q -E '^Error' "$LOG_DIR/selftest_intent.txt"; then
  fail "the self-test deep link was not delivered: $(tr '\n' ' ' < "$LOG_DIR/selftest_intent.txt")"
fi
wait_for_marker 'GLOSSARION_SELFTEST PASS' "$SELFTEST_TIMEOUT"
grep -a -h -E "$(marker_regex 'GLOSSARION_SELFTEST PASS')" "$TAG_LOG" "$FULL_LOG" | tail -n 1 > "$LOG_DIR/selftest_result.txt" || true

STAGE="post"
sleep 2
[ -n "$(app_pid)" ] || fail "$PKG exited right after the self-test passed"
RESULT="PASS"
log "smoke test passed"
exit 0
