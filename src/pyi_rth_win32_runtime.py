"""
PyInstaller runtime hook for Windows DLL loading.

This keeps bundled DLLs available while preventing same-named DLLs next to the
exe, such as Downloads/DWrite.dll, from shadowing real System32 DLLs used by Qt.
"""
import os
import sys

# Set environment variable to prevent mixed runtime loading.
os.environ["PYINSTALLER_DISABLE_PYC_ARCHIVE"] = "1"

_PRELOADED_SYSTEM_DLLS = []

if sys.platform == "win32" and hasattr(sys, "_MEIPASS"):
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # Correct values:
        #   0x00000400 = LOAD_LIBRARY_SEARCH_USER_DIRS
        #   0x00000800 = LOAD_LIBRARY_SEARCH_SYSTEM32
        # Do not include LOAD_LIBRARY_SEARCH_APPLICATION_DIR; for one-file exe
        # launches, that is the folder containing the exe and any poison DLLs.
        set_default_dirs = kernel32.SetDefaultDllDirectories
        set_default_dirs.argtypes = [ctypes.c_uint32]
        set_default_dirs.restype = wintypes.BOOL
        set_default_dirs(0x00000400 | 0x00000800)

        add_dll_directory = kernel32.AddDllDirectory
        add_dll_directory.argtypes = [wintypes.LPCWSTR]
        add_dll_directory.restype = wintypes.HANDLE
        add_dll_directory(sys._MEIPASS)

        # Keep PATH behavior for older/native loaders that still consult it.
        os.environ["PATH"] = sys._MEIPASS + os.pathsep + os.environ.get("PATH", "")

        # QtGui may load these OS graphics/text DLLs. Pin them to System32 before
        # PySide6 imports so same-named files beside the exe cannot satisfy them.
        system32_buf = ctypes.create_unicode_buffer(32768)
        if kernel32.GetSystemDirectoryW(system32_buf, len(system32_buf)):
            system32 = system32_buf.value
        else:
            system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")

        for dll_name in ("DWrite.dll", "d3d11.dll", "dxgi.dll", "dcomp.dll"):
            dll_path = os.path.join(system32, dll_name)
            if os.path.exists(dll_path):
                _PRELOADED_SYSTEM_DLLS.append(ctypes.WinDLL(dll_path))
    except Exception:
        pass
