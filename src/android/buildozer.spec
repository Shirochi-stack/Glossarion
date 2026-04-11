[app]

# App metadata
title = Glossarion
package.name = glossarion
package.domain = com.glossarion
source.dir = .
source.include_exts = py,kv,png,jpg,ico,json,csv,txt,otf,ttf,ttc,gradle
source.exclude_dirs = .buildozer,bin,dist,__pycache__,.git,.github

# Versioning
version = 1.0.0

# Dependencies (python-for-android recipes)
# NOTE: lxml removed — p4a's lxml 4.8.0 recipe is incompatible with Python 3.11.
# All BeautifulSoup usage uses html.parser instead.
requirements = python3,kivy==2.3.1,kivymd==1.2.0,pillow,beautifulsoup4,requests,certifi,charset-normalizer,chardet,html2text,tqdm,plyer,pyjnius,six,setuptools

# Entry point
entrypoint = main.py

# App icon and presplash
icon.filename = assets/Halgakos.png
# presplash.filename = assets/presplash.png

# Orientation
orientation = portrait

# Fullscreen
fullscreen = 0

# Android-specific
android.api = 34
android.minapi = 26
android.ndk = 25b
android.accept_sdk_license = True

# Android permissions
android.permissions = INTERNET,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,MANAGE_EXTERNAL_STORAGE,FOREGROUND_SERVICE,POST_NOTIFICATIONS,WAKE_LOCK

# Android features
android.enable_androidx = True

# Gradle dependencies for AndroidX notifications
android.gradle_dependencies = androidx.core:core:1.12.0,androidx.appcompat:appcompat:1.6.1

# Services (foreground translation service)
# services = GlossarionTranslation:service.py:foreground

# Arch — ARM64 + ARM for real devices, x86_64 for CI emulator smoke test
android.archs = arm64-v8a,armeabi-v7a,x86_64

# Allow backup
android.allow_backup = True

# Exclude large unused packages from the build to reduce APK size
# These are in the main requirements.txt but NOT compatible with Android

# Presplash color
android.presplash_color = #121217

# Python optimizations
android.optimize_python = True

# Logcat
log_level = 2

# Strip debug symbols to reduce APK size
android.strip = True

[buildozer]
# Log level (0=error, 1=info, 2=debug)
log_level = 2

# Build warnings
warn_on_root = 1
