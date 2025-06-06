# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Add all your python files here
added_files = [
    ('translator_gui.py', '.'),
    ('splash_utils.py', '.'),          # NEW - Added splash utilities
    ('TransateKRtoEN.py', '.'),
    ('extract_glossary_from_epub.py', '.'),
    ('epub_converter.py', '.'),
    ('scan_html_folder.py', '.'),
    ('unified_api_client.py', '.'),
    ('chapter_splitter.py', '.'),
    ('history_manager.py', '.'),
    ('check_epub_directory.py', '.'),  # Added missing file
    ('direct_imports.py', '.'),        # Added missing file
    ('Halgakos.ico', '.'),  # Include icon
]

a = Analysis(
    ['translator_gui.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        # Core modules
        'TransateKRtoEN',
        'extract_glossary_from_epub',
        'epub_converter',
        'scan_html_folder',
        'unified_api_client',
        'chapter_splitter',
        'history_manager',
        'check_epub_directory',  # Added
        'direct_imports',        # Added
        'splash_utils',          # NEW - Added splash utilities
        
        # EPUB processing
        'ebooklib',
        'ebooklib.epub',
        'ebooklib.utils',  # Added for completeness
        'bs4',
        'BeautifulSoup',
        'lxml',
        'lxml.etree',
        'lxml._elementpath',
        'lxml.html',  # Added
        'lxml.html.clean',  # Added
        'html5lib',
        'html',  # Added for html.escape
        'html.parser',  # Added
        
        # GUI
        'ttkbootstrap',
        'ttkbootstrap.constants',  # Added
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        'tkinter.simpledialog',
        'tkinter.ttk',
        '_tkinter',  # Added for tkinter backend
        
        # PIL/Pillow - EXPANDED
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'PIL._binary',
        'PIL._imaging',
        'PIL._imagingft',
        'PIL._imagingmath',
        'PIL._imagingtk',
        'PIL.BmpImagePlugin',
        'PIL.GifImagePlugin',
        'PIL.JpegImagePlugin',
        'PIL.PngImagePlugin',
        'PIL.PpmImagePlugin',
        'PIL.TiffImagePlugin',
        'PIL.WebPImagePlugin',
        'PIL.IcoImagePlugin',
        'PIL.MicImagePlugin',
        'olefile',  # FIXES THE ERROR - this is what MicImagePlugin needs
        
        # AI/API clients
        'google',  # Added base google module
        'google.generativeai',
        'google.ai',
        'google.ai.generativelanguage',
        'google.auth',  # Added for authentication
        'google.auth.transport',  # Added
        'google.auth.transport.requests',  # Added
        'openai',
        'tiktoken',
        'tiktoken_ext',
        'tiktoken_ext.openai_public',
        'regex',  # Added - tiktoken dependency
        
        # Text processing
        'langdetect',
        'langdetect.detector',  # Added
        'langdetect.lang_detect_exception',  # Added
        'difflib',
        'unicodedata',
        
        # Progress/UI
        'tqdm',
        'tqdm.auto',  # Added
        'tqdm.std',   # Added
        
        # Network
        'requests',
        'requests.adapters',  # Added
        'requests.models',    # Added
        'requests.sessions',  # Added
        'chardet',
        'certifi',
        'urllib3',
        'urllib',     # Added
        'urllib.parse',  # Added
        'urllib.request',  # Added
        'idna',
        'ssl',  # Added
        'socket',  # Added
        
        # Additional commonly missed modules
        'pkg_resources',
        'pkg_resources._vendor',  # Added
        'encodings',
        'encodings.utf_8',
        'encodings.ascii',
        'encodings.latin_1',
        'encodings.cp1252',  # Added for Windows
        'encodings.utf_16',  # Added
        'encodings.utf_32',  # Added
        'codecs',
        
        # Standard library modules that might be missed
        'json',
        'csv',
        'hashlib',
        'tempfile',
        'shutil',
        'threading',
        'queue',
        're',
        'zipfile',
        'mimetypes',
        'collections',
        'collections.abc',  # Added
        'io',
        'logging',
        'logging.handlers',  # Added
        'time',
        'datetime',  # Added
        'os',
        'os.path',  # Added
        'sys',
        'dataclasses',
        'typing',
        'typing_extensions',  # Added - often needed
        'argparse',
        'subprocess',
        'platform',
        'pathlib',  # Added
        'contextlib',  # Added - used by history_manager
        'functools',  # Added
        'itertools',  # Added
        'warnings',  # Added
        'copy',  # Added
        'weakref',  # Added
        'locale',  # Added
        'struct',  # Added
        'base64',  # Added - might be used by APIs
        'hmac',  # Added - might be used by APIs
        'secrets',  # Added - might be used by APIs
        'uuid',  # Added - might be used by APIs
        'email',  # Added - might be used by requests
        'email.utils',  # Added
        'http',  # Added
        'http.client',  # Added
        'xml',  # Added - used by BeautifulSoup
        'xml.etree',  # Added
        'xml.etree.ElementTree',  # Added
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',  # Exclude if not used
        'numpy',       # Exclude if not used
        'pandas',      # Exclude if not used
        'scipy',       # Exclude if not used
        'pytest',      # Exclude test frameworks
        'nose',        # Exclude test frameworks
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Glossarion v1.6.8',  # Updated version
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want to see console output for debugging
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Halgakos.ico'  # Icon path
)
