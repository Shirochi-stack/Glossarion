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


def get_sd_card_dirs():
    """Detect mounted SD card directories on Android.
    
    Scans /storage/ for non-emulated, non-self mount points and also
    uses Context.getExternalFilesDirs() to find secondary storage.
    
    Returns:
        list: Absolute paths to accessible SD card root directories.
    """
    sd_dirs = []
    if not is_android():
        return sd_dirs

    # Method 1: Scan /storage/ for mount points
    # SD cards appear as /storage/XXXX-XXXX/ (hex formatted label)
    storage_root = '/storage'
    try:
        if os.path.isdir(storage_root):
            for entry in os.scandir(storage_root):
                if not entry.is_dir():
                    continue
                name = entry.name
                # Skip internal storage + self
                if name in ('emulated', 'self'):
                    continue
                # Verify actually accessible
                if os.access(entry.path, os.R_OK):
                    sd_dirs.append(entry.path)
    except (PermissionError, OSError):
        pass

    # Method 2: Use Android Context.getExternalFilesDirs()
    # Returns app-private directories on all mounted volumes;
    # we extract the volume root from each path.
    try:
        from jnius import autoclass
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        activity = PythonActivity.mActivity
        # getExternalFilesDirs(null) returns File[] for all volumes
        ext_dirs = activity.getExternalFilesDirs(None)
        if ext_dirs:
            for i in range(ext_dirs.length):
                d = ext_dirs[i]
                if d is None:
                    continue
                abs_path = d.getAbsolutePath()
                # Extract volume root: everything before /Android/data/...
                android_idx = abs_path.find('/Android/')
                if android_idx > 0:
                    volume_root = abs_path[:android_idx]
                    if (volume_root not in sd_dirs
                            and '/emulated/' not in volume_root
                            and os.access(volume_root, os.R_OK)):
                        sd_dirs.append(volume_root)
    except Exception as e:
        print(f"[WARN] getExternalFilesDirs failed: {e}")

    return sd_dirs


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


def scan_for_books(directory=None, extensions=None, include_downloads=False,
                   include_sdcard=False):
    """Scan a directory for EPUB, TXT, and PDF files.
    
    Args:
        directory: Directory to scan (default: documents dir)
        extensions: List of extensions to include (default: ['.epub', '.txt', '.pdf'])
        include_downloads: Also scan the Downloads directory
        include_sdcard: Also scan detected SD card root directories
        
    Returns:
        list: List of dicts with 'path', 'name', 'ext', 'size', 'modified' keys
    """
    if directory is None:
        directory = get_documents_dir()
    
    if extensions is None:
        extensions = ['.epub', '.txt', '.pdf']
    
    books = []
    
    # Scan the primary directory
    books.extend(_scan_single_dir(directory, extensions))
    
    # Optionally scan Downloads
    if include_downloads:
        dl = get_downloads_dir()
        if dl and os.path.isdir(dl) and dl != directory:
            books.extend(_scan_single_dir(dl, extensions))
    
    # Optionally scan SD card root directories
    if include_sdcard and is_android():
        for sd_root in get_sd_card_dirs():
            # Scan root-level files
            books.extend(_scan_single_dir(sd_root, extensions))
            # Scan common subdirectories on SD cards
            for subdir in ('Download', 'Downloads', 'Documents', 'Books',
                           'Glossarion', 'EPUBs'):
                sd_sub = os.path.join(sd_root, subdir)
                if os.path.isdir(sd_sub):
                    books.extend(_scan_single_dir(sd_sub, extensions))
    
    # Deduplicate by absolute path
    seen = set()
    unique = []
    for b in books:
        abs_p = os.path.abspath(b['path'])
        if abs_p not in seen:
            seen.add(abs_p)
            unique.append(b)
    
    # Sort by modification time (newest first)
    unique.sort(key=lambda b: b['modified'], reverse=True)
    return unique


def _scan_single_dir(directory, extensions):
    """Scan a single directory (non-recursive) for matching files."""
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

# Valid extensions for client-side filtering after SAF returns
_VALID_EXTENSIONS = {'.epub', '.txt', '.pdf', '.csv'}

# Request codes for different picker contexts
REQUEST_CODE_OPEN_FILE = 9001
REQUEST_CODE_OPEN_GLOSSARY = 9002

# Pending callbacks — keyed by request code
_pending_callbacks = {}
# Pending extension filters — keyed by request code
_pending_extensions = {}


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

        activity = PythonActivity.mActivity
        resolver = activity.getContentResolver()
        uri = Uri.parse(uri_string)

        # ── Get display name ──
        filename = None
        try:
            cursor = resolver.query(uri, None, None, None, None)
            if cursor is not None:
                try:
                    name_index = cursor.getColumnIndex("_display_name")
                    if name_index >= 0 and cursor.moveToFirst():
                        filename = cursor.getString(name_index)
                finally:
                    cursor.close()
        except Exception as e:
            print(f"[WARN] Cursor query failed: {e}")

        if not filename:
            # Fallback: extract from URI path
            path_part = uri.getPath()
            if path_part:
                filename = os.path.basename(path_part)
            else:
                filename = 'imported_file'

        # Sanitize filename
        filename = filename.replace('/', '_').replace('\\', '_')

        # Determine destination directory
        lib_dir = get_library_dir()
        dest_path = os.path.join(lib_dir, filename)

        # Avoid overwriting — append counter if file exists
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(lib_dir, f"{base}_{counter}{ext}")
                counter += 1

        # ── Copy file using lightning fast java.nio.file API ──
        # Runs entirely in Java space with 0 context switches. Android 8.0+ (API 26+)
        input_stream = resolver.openInputStream(uri)
        if input_stream is None:
            print(f"[WARN] Could not open input stream for URI: {uri_string}")
            return None

        try:
            jFiles = autoclass('java.nio.file.Files')
            jPaths = autoclass('java.nio.file.Paths')
            
            # Create empty arrays to satisfy varargs method signatures 
            JString = autoclass('java.lang.String')
            Array = autoclass('java.lang.reflect.Array')
            str_array = Array.newInstance(JString("").getClass(), 0)
            dest_java_path = jPaths.get(dest_path, str_array)
            
            CopyOption_class = autoclass('java.nio.file.CopyOption')
            opts_array = Array.newInstance(CopyOption_class, 0)

            # Do the copy!
            jFiles.copy(input_stream, dest_java_path, opts_array)
            input_stream.close()
            print(f"[INFO] Fast native Java copy successful.")

        except Exception as e:
            print(f"[WARN] Native Java file copy failed ({e}), trying fallback...")
            import traceback
            traceback.print_exc()

            try:
                input_stream.close()
            except:
                pass
            
            # Fallback: Java String conversion via ISO-8859-1 mapping.
            # Faster than reading byte-by-byte.
            input_stream = resolver.openInputStream(uri)
            ByteArrayOutputStream = autoclass('java.io.ByteArrayOutputStream')
            baos = ByteArrayOutputStream()
            buf_size = 8192
            
            Byte_TYPE = autoclass('java.lang.Byte').TYPE
            java_buf = Array.newInstance(Byte_TYPE, buf_size)
            
            while True:
                bytes_read = input_stream.read(java_buf)
                if bytes_read == -1:
                    break
                baos.write(java_buf, 0, bytes_read)
            
            input_stream.close()
            
            # ISO-8859-1 natively preserves raw byte values directly
            java_string = baos.toString("ISO-8859-1")
            baos.close()
            
            with open(dest_path, 'wb') as f:
                f.write(java_string.encode('iso-8859-1'))
            
            print(f"[INFO] String-encoded copy successful.")

        print(f"[INFO] Copied URI to local: {dest_path} ({os.path.getsize(dest_path)} bytes)")
        return dest_path

    except Exception as e:
        print(f"[ERR] Failed to copy URI to local: {e}")
        import traceback
        traceback.print_exc()
        return None


def open_native_file_picker(callback, extensions=None, request_code=None):
    """Open the Android native file picker.

    Uses the simplest possible ACTION_GET_CONTENT intent with type */*
    to show ALL files on ALL storages (internal + SD card).  No MIME
    filtering — the user picks whatever they want.

    On non-Android platforms, falls back to a KivyMD MDFileManager.

    Args:
        callback: function(file_path_or_None) called with the selected
                  file's local path, or None if cancelled.
        extensions: list of '.ext' strings (unused on Android — kept for
                    desktop fallback compatibility)
        request_code: int request code (default: REQUEST_CODE_OPEN_FILE)
    """
    if extensions is None:
        extensions = ['.epub', '.txt', '.pdf']
    if request_code is None:
        request_code = REQUEST_CODE_OPEN_FILE

    if not is_android():
        _desktop_file_picker(callback, extensions)
        return

    try:
        from jnius import autoclass
        from android import activity as android_activity

        Intent = autoclass('android.content.Intent')
        PythonActivity = autoclass('org.kivy.android.PythonActivity')

        # Simplest possible intent — show ALL files on ALL storages
        intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.addCategory(Intent.CATEGORY_OPENABLE)
        intent.setType('*/*')

        # Store callback
        _pending_callbacks[request_code] = callback

        def _on_activity_result(request_code_recv, result_code, intent_data):
            cb = _pending_callbacks.pop(request_code_recv, None)
            if cb is None:
                return

            # RESULT_OK == -1
            if result_code != -1 or intent_data is None:
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
            print(f"[INFO] Picked URI: {uri_string}")

            # Copy from content:// URI to a local file in a worker thread
            import threading

            def _copy_worker():
                local_path = _copy_uri_to_local(uri_string)
                print(f"[INFO] Copied to local: {local_path}")
                try:
                    from kivy.clock import Clock
                    Clock.schedule_once(lambda dt: cb(local_path), 0)
                except ImportError:
                    cb(local_path)

            threading.Thread(target=_copy_worker, daemon=True).start()

        android_activity.bind(on_activity_result=_on_activity_result)

        current_activity = PythonActivity.mActivity
        current_activity.startActivityForResult(intent, request_code)
        print(f"[INFO] Opened file picker (request_code={request_code})")

    except Exception as e:
        print(f"[ERR] File picker failed: {e}")
        import traceback
        traceback.print_exc()
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


# ===========================================================================
# Handle "Open with" intents (app launched via EPUB file association)
# ===========================================================================

def get_intent_file_path():
    """Check if the app was launched via an "Open with" intent for a file.

    Returns:
        str: Local file path if launched with a file intent, None otherwise.
    """
    if not is_android():
        return None

    try:
        from jnius import autoclass
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        Intent = autoclass('android.content.Intent')

        activity = PythonActivity.mActivity
        intent = activity.getIntent()

        if intent is None:
            return None

        action = intent.getAction()

        # Only process VIEW or SEND actions
        if action not in (Intent.ACTION_VIEW, Intent.ACTION_SEND):
            return None

        uri = intent.getData()
        if uri is None:
            # Try EXTRA_STREAM for ACTION_SEND
            uri = intent.getParcelableExtra(Intent.EXTRA_STREAM)

        if uri is None:
            return None

        uri_string = uri.toString()
        print(f"[INFO] App launched with intent URI: {uri_string}")

        # If it's a file:// URI, extract the path directly
        scheme = uri.getScheme()
        if scheme == 'file':
            file_path = uri.getPath()
            if file_path and os.path.isfile(file_path):
                return file_path

        # For content:// URIs, copy to local Library
        local_path = _copy_uri_to_local(uri_string)
        return local_path

    except Exception as e:
        print(f"[WARN] Could not process intent file: {e}")
        import traceback
        traceback.print_exc()
        return None
