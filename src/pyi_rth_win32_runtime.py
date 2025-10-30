"""
PyInstaller runtime hook to fix R6034 errors on Windows
Forces the application to use the system's MSVC runtime
"""
import os
import sys

# Set environment variable to prevent mixed runtime loading
os.environ['PYINSTALLER_DISABLE_PYC_ARCHIVE'] = '1'

# For Windows, ensure we use the correct runtime
if sys.platform == 'win32':
    # Remove any paths that might contain conflicting runtimes
    if hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller bundle
        import ctypes
        # Force Windows to use system runtime libraries
        kernel32 = ctypes.windll.kernel32
        kernel32.SetDefaultDllDirectories(0x00001000)  # LOAD_LIBRARY_SEARCH_SYSTEM32
