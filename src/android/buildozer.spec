[app]

# App metadata
title = Glossarion
package.name = glossarion
package.domain = com.glossarion
source.dir = .
source.include_exts = py,kv,png,jpg,ico,json,csv,txt
source.exclude_dirs = .buildozer,bin,dist,__pycache__,.git,.github

# Versioning
version = 1.0.0

# Dependencies (python-for-android recipes)
# NOTE: Some packages may need recipes. Major ones like kivy, pillow, lxml
# have recipes in python-for-android. Others compile from pip.
requirements = python3,kivy==2.3.1,kivymd==2.0.1,pillow,beautifulsoup4,lxml,html5lib,ebooklib,requests,httpx,certifi,charset-normalizer,chardet,html2text,regex,langdetect,rapidfuzz,tqdm,plyer,pyjnius,cryptography,openssl

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
android.sdk = 34
android.accept_sdk_license = True

# Android permissions
android.permissions = INTERNET,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,MANAGE_EXTERNAL_STORAGE,FOREGROUND_SERVICE,POST_NOTIFICATIONS,WAKE_LOCK

# Android features
android.enable_androidx = True

# Gradle dependencies for AndroidX notifications
android.gradle_dependencies = androidx.core:core:1.12.0,androidx.appcompat:appcompat:1.6.1

# Services (foreground translation service)
# services = GlossarionTranslation:service.py:foreground

# Arch — build for ARM64 (most modern devices) and ARM (older devices)
android.archs = arm64-v8a,armeabi-v7a

# Allow backup
android.allow_backup = True

# Exclude large unused packages from the build to reduce APK size
# These are in the main requirements.txt but NOT compatible with Android
android.add_compile_options = -Xlint:deprecation

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
