[app]

# App metadata
title = Glossarion
package.name = glossarion
package.domain = com.glossarion
source.dir = .
source.include_exts = py,kv,png,jpg,ico,json,csv,txt,otf,ttf,ttc,gradle,xml
source.exclude_dirs = .buildozer,bin,dist,__pycache__,.git,.github

# Versioning
version = 1.0.0

# Dependencies (python-for-android recipes and pip packages)
# Keep python3 on 3.10 where lxml + ebooklib are reliable on Android.
# BeautifulSoup remains required by EPUB parsing paths.
requirements = python3==3.10.14,kivy==2.3.1,kivymd==1.2.0,pillow,beautifulsoup4,soupsieve,ebooklib,lxml,requests,certifi,charset-normalizer,chardet,html2text,tqdm,plyer,pyjnius,six,setuptools,filetype,typing_extensions

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

# Intent filters - register as handler for EPUB, TXT, PDF files
# (allows app to appear in "Open with" dialogs)
android.manifest.intent_filters = intent_filters.xml

# Gradle dependencies for AndroidX notifications
android.gradle_dependencies = androidx.core:core:1.12.0,androidx.appcompat:appcompat:1.6.1

# Services (foreground translation service)
# services = GlossarionTranslation:service.py:foreground

# Arch - ARM64 + ARM for real devices, x86_64 for CI emulator smoke test
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
