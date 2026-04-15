# android_file_utils.py
"""
Android-safe file path utilities for Glossarion.
Handles path resolution across Android, Windows, and macOS.

On Android, file picking uses the native Storage Access Framework
(ACTION_OPEN_DOCUMENT) which properly shows all accessible .epub,
.txt, and .pdf files.  On desktop, falls back to KivyMD's MDFileManager.
"""

import os
import sys
import shutil
from pathlib import Path


def is_android():
    """Check if running on Android."""
    try:
        from kivy.utils import platform
        return platform == 'android'
    except ImportError:
        return False


def get_app_data_dir():
    """Get the app's private data directory.
    
    Android: /data/data/com.glossarion.app/files/
    Desktop: ~/.glossarion/
    """
    if is_android():
        try:
            from android.storage import app_storage_path  # noqa: F811
            return app_storage_path()
        except ImportError:
            # Fallback for older python-for-android
            return '/data/data/com.glossarion.app/files'
    else:
        return os.path.join(os.path.expanduser('~'), '.glossarion')


def get_documents_dir():
    """Get the shared Documents directory for EPUB/TXT files.
    
    Android: /storage/emulated/0/Documents/Glossarion/
    Windows: ~/Documents/Glossarion/
    macOS:   ~/Documents/Glossarion/
    """
    if is_android():
        try:
            from android.storage import primary_external_storage_path
            base = primary_external_storage_path()
        except ImportError:
            base = '/storage/emulated/0'
        docs = os.path.join(base, 'Documents', 'Glossarion')
    elif sys.platform == 'win32':
        docs = os.path.join(os.path.expanduser('~'), 'Documents', 'Glossarion')
    elif sys.platform == 'darwin':
        docs = os.path.join(os.path.expanduser('~'), 'Documents', 'Glossarion')
    else:
        docs = os.path.join(os.path.expanduser('~'), 'Documents', 'Glossarion')
    
    os.makedirs(docs, exist_ok=True)
    return docs


def get_downloads_dir():
    """Get the Downloads directory (common place users put EPUBs)."""
    if is_android():
        try:
            from android.storage import primary_external_storage_path
            base = primary_external_storage_path()
        except ImportError:
            base = '/storage/emulated/0'
        return os.path.join(base, 'Download')
    else:
        return os.path.join(os.path.expanduser('~'), 'Downloads')


def get_config_path():
    """Get the config file path."""
    data_dir = get_app_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, 'config_android.json')


def get_library_dir():
    """Get the dedicated library directory for EPUBs.
    
    All translated EPUBs and their output folders live here.
    
    Android: /storage/emulated/0/Documents/Glossarion/Library/
    Windows: ~/Documents/Glossarion/Library/
    macOS:   ~/Documents/Glossarion/Library/
    """
    docs = get_documents_dir()
    lib_dir = os.path.join(docs, 'Library')
    os.makedirs(lib_dir, exist_ok=True)
    return lib_dir


def get_output_dir(input_file):
    """Get/create the output directory for a given input file.
    
    Places the output inside the Glossarion Library folder:
        ~/Documents/Glossarion/Library/{filename}_output/
    
    Falls back to creating next to the input file if the library dir
    is not writable.
    
    Args:
        input_file: Path to the input EPUB/TXT file
        
    Returns:
        str: Path to the output directory (created if needed)
    """
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_name = f"{base_name}_output"
    
    # Primary: use the dedicated Glossarion Library folder
    lib_dir = get_library_dir()
    output_dir = os.path.join(lib_dir, output_name)
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Test write access
        test_file = os.path.join(output_dir, '.write_test')
        with open(test_file, 'w') as f:
            f.write('ok')
        os.remove(test_file)
        return output_dir
    except (OSError, PermissionError):
        pass
    
    # Fallback: try next to the input file
    input_dir = os.path.dirname(os.path.abspath(input_file))
    output_dir = os.path.join(input_dir, output_name)
    try:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    except (OSError, PermissionError):
        pass
    
    # Last resort: app's documents directory directly
    docs = get_documents_dir()
    output_dir = os.path.join(docs, output_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_glossary_dir():
    """Get the glossary storage directory."""
    docs = get_documents_dir()
    glossary_dir = os.path.join(docs, 'Glossaries')
    os.makedirs(glossary_dir, exist_ok=True)
    return glossary_dir


def get_cache_dir():
    """Get the app's cache directory."""
    if is_android():
        try:
            from jnius import autoclass
            Context = autoclass('android.content.Context')
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            cache_dir = PythonActivity.mActivity.getCacheDir().getAbsolutePath()
            return cache_dir
        except Exception:
            return os.path.join(get_app_data_dir(), 'cache')
    else:
        cache = os.path.join(get_app_data_dir(), 'cache')
        os.makedirs(cache, exist_ok=True)
        return cache


def resolve_path(path):
    """Normalize a file path for the current platform.
    
    - Converts backslashes to forward slashes
    - Expands ~ to home directory
    - Resolves relative paths
    - Strips Windows drive letters on Android
    """
    if not path:
        return path
    
    # Expand user home
    path = os.path.expanduser(path)
    
    # On Android, strip Windows-style drive letters (C:\\ etc.)
    if is_android() and len(path) >= 2 and path[1] == ':':
        path = path[2:]
    
    # Normalize separators
    path = path.replace('\\', '/')
    
    # Resolve to absolute
    path = os.path.abspath(path)
    
    return path


def copy_file_to_documents(source_path):
    """Copy a file to the Glossarion Library directory.
    
    Imports files directly into the Library/ subfolder so they
    appear in the library scan.
    
    Args:
        source_path: Path to the source file
        
    Returns:
        str: Path to the copied file in the library, or source_path if copy failed
    """
    try:
        lib_dir = get_library_dir()
        dest = os.path.join(lib_dir, os.path.basename(source_path))
        if os.path.abspath(source_path) == os.path.abspath(dest):
            return dest
        shutil.copy2(source_path, dest)
        return dest
    except Exception:
        return source_path


def request_storage_permissions():
    """Request storage permissions on Android.
    
    On Android 11+, requests MANAGE_EXTERNAL_STORAGE.
    On older Android, requests READ/WRITE_EXTERNAL_STORAGE.
    
    Returns True if permissions are granted, False otherwise.
    """
    if not is_android():
        return True
    
    try:
        from android.permissions import request_permissions, Permission, check_permission
        
        # Check if we already have permissions
        if check_permission(Permission.READ_EXTERNAL_STORAGE):
            return True
        
        # Request permissions
        request_permissions([
            Permission.READ_EXTERNAL_STORAGE,
            Permission.WRITE_EXTERNAL_STORAGE,
        ])
        return True
    except Exception as e:
        print(f"[WARN] Could not request storage permissions: {e}")
        return False


def scan_for_books(directory=None, extensions=None):
    """Scan a directory for EPUB, TXT, and PDF files.
    
    Args:
        directory: Directory to scan (default: documents dir)
        extensions: List of extensions to include (default: ['.epub', '.txt', '.pdf'])
        
    Returns:
        list: List of dicts with 'path', 'name', 'ext', 'size', 'modified' keys
    """
    if directory is None:
        directory = get_documents_dir()
    
    if extensions is None:
        extensions = ['.epub', '.txt', '.pdf']
    
    books = []
    
    if not os.path.isdir(directory):
        return books
    
    try:
        for entry in os.scandir(directory):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in extensions:
                continue
            
            try:
                stat = entry.stat()
                books.append({
                    'path': entry.path,
                    'name': os.path.splitext(entry.name)[0],
                    'ext': ext,
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                })
            except OSError:
                continue
    except PermissionError:
        print(f"[WARN] No permission to scan: {directory}")
    
    # Sort by modification time (newest first)
    books.sort(key=lambda b: b['modified'], reverse=True)
    return books


# ===========================================================================
# Android native file picker (Storage Access Framework)
# ===========================================================================

# MIME types for each supported extension
_MIME_MAP = {
    '.epub': 'application/epub+zip',
    '.txt':  'text/plain',
    '.pdf':  'application/pdf',
    '.csv':  'text/csv',
}

# Request codes for different picker contexts
REQUEST_CODE_OPEN_FILE = 9001
REQUEST_CODE_OPEN_GLOSSARY = 9002

# Pending callbacks — keyed by request code
_pending_callbacks = {}


def _copy_uri_to_local(uri_string):
    """Copy content from a content:// URI to the Glossarion Library folder.
    
    Android's SAF returns content:// URIs which may not be directly readable
    as file paths.  We copy the content to a local file and return that path.
    
    Returns:
        str: Path to the local copy, or None on failure.
    """
    try:
        from jnius import autoclass

        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        Uri = autoclass('android.net.Uri')
        ContentResolver = autoclass('android.content.ContentResolver')

        activity = PythonActivity.mActivity
        resolver = activity.getContentResolver()
        uri = Uri.parse(uri_string)

        # Try to get the display name from the cursor
        filename = None
        cursor = resolver.query(uri, None, None, None, None)
        if cursor is not None:
            try:
                name_index = cursor.getColumnIndex("_display_name")
                if name_index >= 0 and cursor.moveToFirst():
                    filename = cursor.getString(name_index)
            finally:
                cursor.close()

        if not filename:
            # Fallback: extract from URI path
            path_part = uri.getPath()
            if path_part:
                filename = os.path.basename(path_part)
            else:
                filename = 'imported_file'

        # Sanitize filename
        filename = filename.replace('/', '_').replace('\\', '_')

        # Copy to Library dir
        lib_dir = get_library_dir()
        dest_path = os.path.join(lib_dir, filename)

        # Read from content URI and write to local file
        input_stream = resolver.openInputStream(uri)
        if input_stream is None:
            print(f"[WARN] Could not open input stream for URI: {uri_string}")
            return None

        BufferedInputStream = autoclass('java.io.BufferedInputStream')
        bis = BufferedInputStream(input_stream)

        with open(dest_path, 'wb') as f:
            buf = bytearray(8192)
            while True:
                # Read into a Java byte array
                java_buf = autoclass('java.lang.reflect.Array').newInstance(
                    autoclass('java.lang.Byte').TYPE, 8192
                )
                bytes_read = bis.read(java_buf)
                if bytes_read == -1:
                    break
                # Convert Java byte[] to Python bytes
                for i in range(bytes_read):
                    buf[i] = java_buf[i] & 0xFF
                f.write(bytes(buf[:bytes_read]))

        bis.close()
        input_stream.close()

        print(f"[INFO] Copied URI to local: {dest_path}")
        return dest_path

    except Exception as e:
        print(f"[ERR] Failed to copy URI to local: {e}")
        import traceback
        traceback.print_exc()
        return None


def open_native_file_picker(callback, extensions=None, request_code=None):
    """Open the Android native file picker (Storage Access Framework).
    
    Uses ACTION_OPEN_DOCUMENT Intent which respects scoped storage,
    shows the system document picker UI, and lets the user browse all
    accessible locations (Downloads, Drive, file managers, etc.).
    
    On non-Android platforms, falls back to a KivyMD MDFileManager.
    
    Args:
        callback: function(file_path_or_None) called with the selected
                  file's local path, or None if cancelled.
        extensions: list of '.ext' strings (default: ['.epub', '.txt', '.pdf'])
        request_code: int request code (default: REQUEST_CODE_OPEN_FILE)
    """
    if extensions is None:
        extensions = ['.epub', '.txt', '.pdf']
    if request_code is None:
        request_code = REQUEST_CODE_OPEN_FILE

    if not is_android():
        # Desktop fallback: use KivyMD file manager
        _desktop_file_picker(callback, extensions)
        return

    try:
        from jnius import autoclass
        from android import activity as android_activity

        Intent = autoclass('android.content.Intent')
        PythonActivity = autoclass('org.kivy.android.PythonActivity')

        intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
        intent.addCategory(Intent.CATEGORY_OPENABLE)

        # Build MIME types from extensions
        mime_types = []
        for ext in extensions:
            mime = _MIME_MAP.get(ext.lower())
            if mime:
                mime_types.append(mime)

        if not mime_types:
            mime_types = ['*/*']

        if len(mime_types) == 1:
            intent.setType(mime_types[0])
        else:
            # Multiple MIME types → use EXTRA_MIME_TYPES
            intent.setType('*/*')
            String = autoclass('java.lang.String')
            Array = autoclass('java.lang.reflect.Array')
            mime_array = Array.newInstance(String.getClass(), len(mime_types))
            for i, m in enumerate(mime_types):
                Array.set(mime_array, i, String(m))
            intent.putExtra(Intent.EXTRA_MIME_TYPES, mime_array)

        # Register the activity result callback
        _pending_callbacks[request_code] = callback

        def _on_activity_result(request_code_recv, result_code, intent_data):
            cb = _pending_callbacks.pop(request_code_recv, None)
            if cb is None:
                return

            # RESULT_OK == -1
            if result_code != -1 or intent_data is None:
                # User cancelled
                try:
                    from kivy.clock import Clock
                    Clock.schedule_once(lambda dt: cb(None), 0)
                except ImportError:
                    cb(None)
                return

            uri = intent_data.getData()
            if uri is None:
                try:
                    from kivy.clock import Clock
                    Clock.schedule_once(lambda dt: cb(None), 0)
                except ImportError:
                    cb(None)
                return

            uri_string = uri.toString()
            print(f"[INFO] SAF picked URI: {uri_string}")

            # Copy from content:// URI to a local file in a worker thread
            import threading

            def _copy_worker():
                local_path = _copy_uri_to_local(uri_string)
                try:
                    from kivy.clock import Clock
                    Clock.schedule_once(lambda dt: cb(local_path), 0)
                except ImportError:
                    cb(local_path)

            threading.Thread(target=_copy_worker, daemon=True).start()

        android_activity.bind(on_activity_result=_on_activity_result)

        current_activity = PythonActivity.mActivity
        current_activity.startActivityForResult(intent, request_code)
        print(f"[INFO] Opened native file picker (request_code={request_code})")

    except Exception as e:
        print(f"[ERR] Native file picker failed: {e}")
        import traceback
        traceback.print_exc()
        # Fall back to KivyMD file manager
        _desktop_file_picker(callback, extensions)


def _desktop_file_picker(callback, extensions):
    """Fallback file picker for desktop using KivyMD's MDFileManager."""
    try:
        from kivymd.uix.filemanager import MDFileManager

        fm = None

        def _exit(path=None):
            nonlocal fm
            if fm:
                fm.close()

        def _select(path):
            _exit()
            if path and os.path.isfile(path):
                callback(path)
            else:
                callback(None)

        fm = MDFileManager(
            exit_manager=_exit,
            select_path=_select,
            ext=list(extensions),
        )
        start_path = get_documents_dir()
        fm.show(start_path)
    except Exception as e:
        print(f"[WARN] Desktop file picker error: {e}")
        callback(None)
