[app]

# Experimental PySide6 Android build for the desktop Glossarion GUI.
# Run pyside6-android-deploy from this directory so input_file resolves to
# this launcher while project_dir still packages the full src tree.
title = Glossarion
project_dir = ../..
input_file = main.py
exec_directory =
project_file =
icon = ../../Halgakos.png

[python]

python_path =
packages = Nuitka==4.0
android_packages = buildozer==1.5.0,cython==0.29.33,virtualenv

[qt]

qml_files =
excluded_qml_plugins =
modules = Core,Gui,Widgets,Network,Multimedia
plugins =

[android]

# Filled by the GitHub workflow through command-line arguments after qtpip
# downloads architecture-specific Android wheels.
wheel_pyside =
wheel_shiboken =
plugins =

[nuitka]

macos.permissions =
mode = onefile
extra_args = --quiet --noinclude-qt-translations

[buildozer]

# debug creates an APK; release creates an AAB.
mode = debug
recipe_dir =
jars_dir =
ndk_path =
sdk_path =
local_libs =

# Official PySide Android wheels are available for aarch64 and x86_64.
arch = aarch64
