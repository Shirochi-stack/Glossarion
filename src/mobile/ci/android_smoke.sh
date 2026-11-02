#!/usr/bin/env bash
# Android emulator smoke test for Glossarion Mobile.
#
# Usage: android_smoke.sh <x86_64.apk> [log-dir]
#
#  1. install the APK (runtime permissions granted) and launch com.glossarion.app;
#  2. wait for GLOSSARION_READY and GLOSSARION_BACKEND_READY in `adb logcat -s flet.python`;
#  3. fire the self-test deep link glossarion://app/__selftest__?suite=smoke;
#  4. wait for GLOSSARION_SELFTEST PASS {"suite":"smoke" (FAIL or timeout exits 1);
#  5. then the offline end-to-end suite: glossarion://app/__selftest__?suite=e2e (real chat /
#     translate / graceful + force stop / kill & resume jobs against the app's built-in fake
#     OpenAI server on 127.0.0.1, see glossarion_mobile/diagnostics/e2e.py) and wait for
#     GLOSSARION_SELFTEST PASS {"suite":"e2e" (blocking; E2E_TIMEOUT, default 20 min);
#  5b. the in-app browser in the background (hotfix 4; RUN_BG_WEB, default = RUN_E2E):
#     glossarion://app/__selftest__?suite=background holds the foreground service like a job and
#     loads a loopback page that beats every second; on GLOSSARION_BGWEB_READY this script presses
#     HOME; the check watches the page while Glossarion is hidden (document.visibilityState stays
#     visible, timers and fetches keep running) and prints GLOSSARION_BGWEB {json} (saved to
#     bg_web_result.txt) before GLOSSARION_SELFTEST PASS {"suite":"background" (BG_WEB_TIMEOUT,
#     default 6 min); then Glossarion is brought back to the screen;
#  6. with RUN_LLAMA=1 only: the built-in llama.cpp engine suite glossarion://app/__selftest__?suite=llama
#     (downloads the GLOSSARION_LLAMA_SUITE_PRESET GGUF from Hugging Face, default "Garbage"
#     = Qwen3.5 0.8B, loads it, generates a few tokens through the in-app OpenAI-compatible
#     server, unloads), waits for GLOSSARION_SELFTEST PASS {"suite":"llama" (LLAMA_TIMEOUT,
#     default 30 min) and saves the GLOSSARION_LLAMA {json} line to llama_result.txt; its second check
#     (llama_vision_describe) adds the preset's vision projector and describes a tiny test picture
#     through the app's vision path: the GLOSSARION_LLAMA_VISION {json} line goes to llama_vision_result.txt;
#  7. always save logcat (flet.python + full + crash buffer), meminfo, pidof, package dump, the
#     app's last self-test report (debuggable builds) and a screenshot into the log dir; the smoke
#     suite's GLOSSARION_INTERPRETERS {json} line (a QA scan batch in a child interpreter: pass, or
#     fallback, which the suite itself fails on Android) goes to interpreters_result.txt and the job summary.
#
# Environment overrides: GLOSSARION_PACKAGE, READY_TIMEOUT (300), SELFTEST_TIMEOUT (900),
# SELFTEST_URL, RUN_E2E (1; 0 skips step 5), E2E_TIMEOUT (1200), E2E_URL, RUN_BG_WEB (RUN_E2E; 0 skips
# step 5b), BG_WEB_TIMEOUT (360), BG_WEB_URL, RUN_LLAMA (0; 1 runs step 6), LLAMA_TIMEOUT (1800), LLAMA_URL,
# LOGCAT_TAG (flet.python).
set -uo pipefail

APK="${1:?usage: android_smoke.sh <x86_64.apk> [log-dir]}"
LOG_DIR="${2:-mobile-smoke-logs}"
PKG="${GLOSSARION_PACKAGE:-com.glossarion.app}"
READY_TIMEOUT="${READY_TIMEOUT:-300}"
SELFTEST_TIMEOUT="${SELFTEST_TIMEOUT:-900}"
SELFTEST_URL="${SELFTEST_URL:-glossarion://app/__selftest__?suite=smoke}"
RUN_E2E="${RUN_E2E:-1}"
E2E_TIMEOUT="${E2E_TIMEOUT:-1200}"
E2E_URL="${E2E_URL:-glossarion://app/__selftest__?suite=e2e}"
RUN_BG_WEB="${RUN_BG_WEB:-$RUN_E2E}"
BG_WEB_TIMEOUT="${BG_WEB_TIMEOUT:-360}"
BG_WEB_URL="${BG_WEB_URL:-glossarion://app/__selftest__?suite=background}"
RUN_LLAMA="${RUN_LLAMA:-0}"
LLAMA_TIMEOUT="${LLAMA_TIMEOUT:-1800}"
LLAMA_URL="${LLAMA_URL:-glossarion://app/__selftest__?suite=llama}"
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
# 1 when it is nowhere. With "fixed" as the second argument the marker is a literal string
# (the per-suite PASS lines carry JSON: GLOSSARION_SELFTEST PASS {"suite":"e2e",...).
find_marker() {
  local regex
  if [ "${2:-regex}" = "fixed" ]; then
    if grep -a -q -F -- "$1" "$TAG_LOG"; then
      return 0
    fi
    if grep -a -q -F -- "$1" "$FULL_LOG"; then
      return 2
    fi
    return 1
  fi
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

# wait_for_marker <marker> <timeout-seconds> [fixed]
wait_for_marker() {
  local marker="$1" timeout="$2" mode="${3:-regex}" start=$SECONDS rc dead_checks=0 pid
  while true; do
    find_marker "$marker" "$mode"
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
  # The full JSON of the last self-test run (every check's detail and traceback). Only
  # debuggable (debug-signed) builds allow run-as; release builds skip this.
  local report
  report="$(adb shell "run-as $PKG find . -maxdepth 8 -name selftest-last.json" 2>/dev/null | tr -d '\r' | head -n 1)"
  if [ -n "$report" ]; then
    adb exec-out run-as "$PKG" cat "$report" > "$LOG_DIR/selftest-last.json" 2>/dev/null || rm -f "$LOG_DIR/selftest-last.json"
  fi
  grep -a -h -o -E 'GLOSSARION_[A-Z_]+( (PASS|FAIL))?' "$TAG_LOG" "$FULL_LOG" 2>/dev/null | sort | uniq -c > "$LOG_DIR/markers.txt" || true
  if [ "$RUN_LLAMA" = "1" ] && [ ! -s "$LOG_DIR/llama_result.txt" ]; then
    grep -a -h -E "$(marker_regex 'GLOSSARION_LLAMA')" "$TAG_LOG" "$FULL_LOG" 2>/dev/null | tail -n 1 > "$LOG_DIR/llama_result.txt" || true
  fi
  if [ "$RUN_BG_WEB" = "1" ] && [ ! -s "$LOG_DIR/bg_web_result.txt" ]; then
    grep -a -h -E "$(marker_regex 'GLOSSARION_BGWEB')" "$TAG_LOG" "$FULL_LOG" 2>/dev/null | tail -n 1 > "$LOG_DIR/bg_web_result.txt" || true
  fi
  if [ "$RUN_LLAMA" = "1" ] && [ ! -s "$LOG_DIR/llama_vision_result.txt" ]; then
    grep -a -h -E "$(marker_regex 'GLOSSARION_LLAMA_VISION')" "$TAG_LOG" "$FULL_LOG" 2>/dev/null | tail -n 1 > "$LOG_DIR/llama_vision_result.txt" || true
  fi
  grep -a -h -E "$(marker_regex 'GLOSSARION_INTERPRETERS')" "$TAG_LOG" "$FULL_LOG" 2>/dev/null | tail -n 1 > "$LOG_DIR/interpreters_result.txt" || true
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
      echo
      echo "QA scan in a child interpreter:"
      cut -c1-500 "$LOG_DIR/interpreters_result.txt" 2>/dev/null
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

# start_suite <suite> <deep-link> <timeout-seconds>: fire the self-test deep link of <suite>.
start_suite() {
  local suite="$1" url="$2" timeout="$3"
  STAGE="selftest-$suite"
  log "starting self-test suite '$suite': $url (timeout ${timeout}s)"
  # adb shell joins its arguments into one device-side command line: keep the URL single-quoted
  # so the device shell does not glob the '?'.
  adb shell "am start -W -a android.intent.action.VIEW -d '$url' $PKG" > "$LOG_DIR/selftest_${suite}_intent.txt" 2>&1 \
    || fail "am start for the self-test deep link failed: $(tr '\n' ' ' < "$LOG_DIR/selftest_${suite}_intent.txt")"
  if grep -q -E '^Error' "$LOG_DIR/selftest_${suite}_intent.txt"; then
    fail "the self-test deep link was not delivered: $(tr '\n' ' ' < "$LOG_DIR/selftest_${suite}_intent.txt")"
  fi
}

# wait_suite <suite> <timeout-seconds>: wait for that suite's PASS line (any GLOSSARION_SELFTEST FAIL,
# a crash or the timeout fails the run).
wait_suite() {
  local suite="$1" timeout="$2" pass_line
  pass_line="GLOSSARION_SELFTEST PASS {\"suite\":\"$suite\""
  wait_for_marker "$pass_line" "$timeout" fixed
  grep -a -h -F -- "$pass_line" "$TAG_LOG" "$FULL_LOG" | tail -n 1 > "$LOG_DIR/selftest_${suite}_result.txt" || true
}

# run_suite <suite> <deep-link> <timeout-seconds>: fire the self-test deep link and wait for
# that suite's PASS line.
run_suite() {
  start_suite "$1" "$2" "$3"
  wait_suite "$1" "$3"
}

# run_background_suite: suite=background (step 5b). The check holds the foreground service, loads its
# page and prints GLOSSARION_BGWEB_READY; HOME then sends Glossarion to the background while the
# check watches the page; afterwards Glossarion is brought back to the screen.
run_background_suite() {
  start_suite background "$BG_WEB_URL" "$BG_WEB_TIMEOUT"
  wait_for_marker GLOSSARION_BGWEB_READY 180
  log "pressing HOME: Glossarion goes to the background while the check watches its in-app page"
  adb shell input keyevent KEYCODE_HOME > /dev/null 2>&1 || fail "adb could not press HOME"
  wait_suite background "$BG_WEB_TIMEOUT"
  grep -a -h -E "$(marker_regex 'GLOSSARION_BGWEB')" "$TAG_LOG" "$FULL_LOG" | tail -n 1 > "$LOG_DIR/bg_web_result.txt" || true
  log "background in-app browser: $(cut -c1-500 "$LOG_DIR/bg_web_result.txt" 2>/dev/null)"
  adb shell monkey -p "$PKG" -c android.intent.category.LAUNCHER 1 > /dev/null 2>&1 || true
  sleep 5
}

run_suite smoke "$SELFTEST_URL" "$SELFTEST_TIMEOUT"
cp "$LOG_DIR/selftest_smoke_result.txt" "$LOG_DIR/selftest_result.txt" 2>/dev/null || true

if [ "$RUN_E2E" = "1" ]; then
  run_suite e2e "$E2E_URL" "$E2E_TIMEOUT"
else
  log "RUN_E2E=$RUN_E2E: skipping the end-to-end suite"
fi

if [ "$RUN_BG_WEB" = "1" ]; then
  run_background_suite
else
  log "RUN_BG_WEB=$RUN_BG_WEB: skipping the in-app browser background suite"
fi

if [ "$RUN_LLAMA" = "1" ]; then
  # The suite prints GLOSSARION_LLAMA {"model":...,"sha256_ok":...,"load_s":...,"tok_s":...} before
  # its PASS line; keep it even when the suite fails (the EXIT trap still collects the rest).
  run_suite llama "$LLAMA_URL" "$LLAMA_TIMEOUT"
  grep -a -h -E "$(marker_regex 'GLOSSARION_LLAMA')" "$TAG_LOG" "$FULL_LOG" | tail -n 1 > "$LOG_DIR/llama_result.txt" || true
  grep -a -h -E "$(marker_regex 'GLOSSARION_LLAMA_VISION')" "$TAG_LOG" "$FULL_LOG" | tail -n 1 > "$LOG_DIR/llama_vision_result.txt" || true
  log "llama suite: $(cut -c1-500 "$LOG_DIR/llama_result.txt" 2>/dev/null)"
  log "llama vision: $(cut -c1-500 "$LOG_DIR/llama_vision_result.txt" 2>/dev/null)"
else
  log "RUN_LLAMA=$RUN_LLAMA: skipping the built-in engine (llama) suite"
fi

STAGE="post"
sleep 2
[ -n "$(app_pid)" ] || fail "$PKG exited right after the self-test passed"
RESULT="PASS"
log "smoke test passed"
exit 0
