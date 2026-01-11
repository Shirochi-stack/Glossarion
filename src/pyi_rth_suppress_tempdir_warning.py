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
    
    # Fix SSL certificate paths BEFORE any SSL imports
    try:
        import ssl
        import certifi
        
        # Get certifi CA bundle path and verify it exists
        cafile = certifi.where()
        
        # If certifi bundle doesn't exist in PyInstaller bundle, look in _MEIPASS
        if not os.path.exists(cafile):
            # Try to find it in _MEIPASS
            meipass_cafile = os.path.join(sys._MEIPASS, 'certifi', 'cacert.pem')
            if os.path.exists(meipass_cafile):
                cafile = meipass_cafile
            else:
                # Try alternate location
                meipass_cafile = os.path.join(sys._MEIPASS, 'cacert.pem')
                if os.path.exists(meipass_cafile):
                    cafile = meipass_cafile
                else:
                    cafile = None
        
        if cafile and os.path.exists(cafile):
            # Set environment variables
            os.environ['SSL_CERT_FILE'] = cafile
            os.environ['REQUESTS_CA_BUNDLE'] = cafile
            os.environ['CURL_CA_BUNDLE'] = cafile
            
            # Monkey-patch ssl.create_default_context to use certifi
            _original_create_default_context = ssl.create_default_context
            
            def _patched_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=None, capath=None, cadata=None):
                # If no cafile is specified, use certifi's bundle
                if cafile is None and capath is None and cadata is None:
                    cafile = os.environ.get('SSL_CERT_FILE')
                    if cafile and os.path.exists(cafile):
                        return _original_create_default_context(purpose=purpose, cafile=cafile, capath=capath, cadata=cadata)
                return _original_create_default_context(purpose=purpose, cafile=cafile, capath=capath, cadata=cadata)
            
            ssl.create_default_context = _patched_create_default_context
        else:
            # CA bundle not found - use unverified context as fallback
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
        
    except Exception:
        # Last resort - disable SSL verification
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
        except:
            pass
    
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
