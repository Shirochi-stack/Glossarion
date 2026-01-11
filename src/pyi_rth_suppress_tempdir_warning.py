# PyInstaller Runtime Hook
# This hook suppresses the temporary directory cleanup warning on Windows
# PyInstaller creates a _MEIPASS temporary directory that sometimes can't be 
# removed immediately on exit due to lingering file handles. This is harmless
# and Windows will clean it up eventually.

import sys
import os

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # We're running as a PyInstaller bundle
    
    # Override the cleanup function to suppress error dialogs
    # PyInstaller's bootloader tries to remove _MEIPASS on exit
    # but Windows may fail if files are still in use
    
    # Set environment variable to tell PyInstaller bootloader to ignore cleanup errors
    # Note: This may not work with all PyInstaller versions
    os.environ['_MEIPASS2'] = sys._MEIPASS
    
    # Store the _MEIPASS path for potential manual cleanup
    _meipass_path = sys._MEIPASS
    
    # Register a cleanup function that silently attempts to remove the temp directory
    # If it fails, that's okay - Windows will clean it up eventually
    import atexit
    import shutil
    
    @atexit.register
    def _silent_cleanup_meipass():
        """Silently attempt to clean up PyInstaller's temporary directory.
        If cleanup fails (files still in use), it's ignored - Windows will clean up later.
        """
        try:
            # Try to remove the temp directory
            if os.path.exists(_meipass_path):
                shutil.rmtree(_meipass_path, ignore_errors=True)
        except Exception:
            # Silently ignore any errors
            # Windows will eventually clean up the temp directory
            pass
