#!/usr/bin/env bash
# Start an isolated API-35 emulator within this job; no external device access.
set -euo pipefail
source .gitlab/setup_android.sh
apt-get install -y --no-install-recommends libgl1 libpulse0 libnss3 libx11-6 libxcb1 libxcomposite1 libxcursor1 libxi6 libxtst6 libxrandr2 libxrender1 libxkbcommon0
sdkmanager emulator 'system-images;android-35;google_apis;x86_64'
export PATH="$ANDROID_HOME/emulator:$PATH"
export ANDROID_AVD_HOME="$CI_PROJECT_DIR/.gitlab-toolchain/avd"
mkdir -p "$ANDROID_AVD_HOME" mobile-smoke-logs
echo no | avdmanager create avd --force --name glossarion --package 'system-images;android-35;google_apis;x86_64'
python - "$ANDROID_AVD_HOME/glossarion.avd/config.ini" <<'PY'
from pathlib import Path
import sys
config = Path(sys.argv[1])
settings = {"hw.ramSize": "4096", "disk.dataPartition.size": "6144M"}
lines = [line for line in config.read_text().splitlines() if line.partition("=")[0].strip() not in settings]
config.write_text("\n".join(lines + [f"{key}={value}" for key, value in settings.items()]) + "\n")
PY
export ANDROID_SERIAL=emulator-5554
accel=off
if [ -r /dev/kvm ] && [ -w /dev/kvm ]; then accel=on; fi
echo "Starting API-35 emulator (acceleration: $accel)"
emulator -avd glossarion -port 5554 -no-window -no-audio -no-boot-anim -no-snapshot -camera-back none -camera-front none -gpu swiftshader_indirect -accel "$accel" > mobile-smoke-logs/emulator.log 2>&1 &
emulator_pid=$!
trap 'kill "$emulator_pid" 2>/dev/null || true' EXIT
deadline=$((SECONDS + 900))
while [ "$SECONDS" -lt "$deadline" ]; do
  if ! kill -0 "$emulator_pid" 2>/dev/null; then cat mobile-smoke-logs/emulator.log; exit 1; fi
  if [ "$(adb -s "$ANDROID_SERIAL" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = 1 ]; then
    adb -s "$ANDROID_SERIAL" shell input keyevent 82
    adb -s "$ANDROID_SERIAL" shell settings put global window_animation_scale 0
    adb -s "$ANDROID_SERIAL" shell settings put global transition_animation_scale 0
    adb -s "$ANDROID_SERIAL" shell settings put global animator_duration_scale 0
    echo "Emulator boot complete"
    break
  fi
  sleep 5
done
if [ "$(adb -s "$ANDROID_SERIAL" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" != 1 ]; then
  echo 'Android emulator did not boot within 900 seconds; device smoke cannot pass.' >&2
  tail -n 100 mobile-smoke-logs/emulator.log
  exit 1
fi
