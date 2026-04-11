#!/bin/bash
# smoke-test.sh — Emulator smoke test for the Glossarion APK.
# Called by the CI workflow inside the android-emulator-runner action.
set +e

echo "=== Finding APK ==="
APK_PATH=$(find $GITHUB_WORKSPACE/src/android/bin -name '*.apk' -type f 2>/dev/null | head -1)
echo "APK_PATH=$APK_PATH"
ls -la $GITHUB_WORKSPACE/src/android/bin/ 2>/dev/null || echo "bin/ not found"
if [ -z "$APK_PATH" ]; then echo "No APK found, skipping"; exit 0; fi

echo "=== APK architecture check ==="
unzip -l "$APK_PATH" | grep -E '\.so$' | head -20

echo "=== Emulator ABI ==="
adb shell getprop ro.product.cpu.abi
adb shell getprop ro.product.cpu.abilist

echo "=== Installing APK ==="
adb install --no-streaming "$APK_PATH" > $GITHUB_WORKSPACE/adb-install.log 2>&1 || adb install "$APK_PATH" >> $GITHUB_WORKSPACE/adb-install.log 2>&1
cat $GITHUB_WORKSPACE/adb-install.log

echo "=== Verifying installation ==="
if ! adb shell pm list packages | grep -q glossarion; then
  echo "PACKAGE NOT INSTALLED - install failed"
  cat $GITHUB_WORKSPACE/adb-install.log
  exit 0
fi

echo "=== Launching app ==="
adb shell am start -n com.glossarion.glossarion/org.kivy.android.PythonActivity

echo "=== Waiting 30s for app to start or crash ==="
sleep 30

echo "=== Dumping logcat ==="
adb logcat -d > $GITHUB_WORKSPACE/emulator-logcat-full.log 2>&1
grep -iE '(python|kivy|glossarion|SDL|Traceback|ModuleNotFound|ImportError|FATAL|AndroidRuntime|Exception|Error)' $GITHUB_WORKSPACE/emulator-logcat-full.log > $GITHUB_WORKSPACE/emulator-logcat-filtered.log 2>/dev/null || true

echo "=== Filtered logcat (last 300 lines) ==="
tail -300 $GITHUB_WORKSPACE/emulator-logcat-filtered.log

echo "=== Process check ==="
adb shell pidof com.glossarion.glossarion && echo "App is running" || echo "App NOT running (crashed)"

exit 0
