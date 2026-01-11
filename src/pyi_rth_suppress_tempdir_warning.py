# PyInstaller Runtime Hook
# This hook suppresses the temporary directory cleanup warning on Windows
# PyInstaller creates a _MEIPASS temporary directory that sometimes can't be 
# removed immediately on exit due to lingering file handles. This is harmless
# and Windows will clean it up eventually.

import sys
import os

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # We're running as a PyInstaller bundle
    # Store the _MEIPASS path for cleanup at exit only
    _meipass_path = sys._MEIPASS
    
    # IMPORTANT: Only register cleanup at exit, don't touch _MEIPASS during runtime
    # The directory must remain accessible throughout the application's lifetime
    # for SSL certificates, DLLs, and other bundled resources
    import atexit
    
    @atexit.register
    def _silent_cleanup_meipass():
        """Silently attempt to clean up PyInstaller's temporary directory at exit.
        If cleanup fails (files still in use), it's ignored - Windows will clean up later.
        This prevents the "Failed to remove temporary directory" warning dialog.
        """
        try:
            import shutil
            # Only try cleanup if directory still exists
            if _meipass_path and os.path.exists(_meipass_path):
                # Use ignore_errors=True to suppress any cleanup failures
                shutil.rmtree(_meipass_path, ignore_errors=True)
        except Exception:
            # Silently ignore any errors - Windows will clean up eventually
            pass
