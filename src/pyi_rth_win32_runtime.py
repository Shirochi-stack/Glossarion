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
        # Allow system libraries AND application directory for SSL certs
        kernel32 = ctypes.windll.kernel32
        # 0x00001000: LOAD_LIBRARY_SEARCH_SYSTEM32
        # 0x00000002: LOAD_LIBRARY_SEARCH_APPLICATION_DIR
        # 0x00000100: LOAD_LIBRARY_SEARCH_USER_DIRS
        kernel32.SetDefaultDllDirectories(0x00001000 | 0x00000002 | 0x00000100)
        
        # Also add _MEIPASS to DLL search path for bundled resources
        kernel32.AddDllDirectory(sys._MEIPASS)
