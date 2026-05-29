# Experimental PySide6 Android Launcher

This folder is a sidecar launcher for testing the desktop PySide6 GUI on
Android. It does not replace `src/android/main.py`, so the current Kivy
Buildozer path stays intact.

For local Windows smoke testing, run `src/android/launch_pyside_gui.bat`.

For GitHub Actions Android packaging, run the `Build Android PySide APK`
workflow. It uses this folder's `pysidedeploy.spec` and downloads the matching
official PySide6/Shiboken6 Android wheels before invoking
`pyside6-android-deploy`.

`main.py` prepares Android-friendly paths, points `translator_gui.py` at a
writable app-data directory via `GLOSSARION_APP_DIR`, registers Android stubs
for modules that are unavailable or too heavy on mobile, then runs
`src/translator_gui.py` as `__main__`.

When testing with `pyside6-android-deploy`, package the whole `src` tree and
use this file as the entry point:

```text
src/android/pyside_launcher/main.py
```

If the deploy tool insists on an entry file literally named `main.py` at the
project root, use this launcher from a temporary build directory rather than
overwriting the tracked Kivy launcher.
