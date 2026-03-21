# android_file_utils.py
"""
Android-safe file path utilities for Glossarion.
Handles path resolution across Android, Windows, and macOS.
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
    """Scan a directory for EPUB and TXT files.
    
    Args:
        directory: Directory to scan (default: documents dir)
        extensions: List of extensions to include (default: ['.epub', '.txt'])
        
    Returns:
        list: List of dicts with 'path', 'name', 'ext', 'size', 'modified' keys
    """
    if directory is None:
        directory = get_documents_dir()
    
    if extensions is None:
        extensions = ['.epub', '.txt']
    
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
