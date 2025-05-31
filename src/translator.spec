# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Add all your python files here
added_files = [
    ('translator_gui.py', '.'),
    ('TransateKRtoEN.py', '.'),
    ('extract_glossary_from_epub.py', '.'),
    ('epub_converter.py', '.'),
    ('scan_html_folder.py', '.'),
    ('unified_api_client.py', '.'),
    ('chapter_splitter.py', '.'),  # Added
    ('history_manager.py', '.'),    # Added
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
        
        # EPUB processing
        'ebooklib',
        'ebooklib.epub',
        'bs4',
        'BeautifulSoup',
        'lxml',
        'lxml.etree',
        'lxml._elementpath',
        'html5lib',
        
        # GUI
        'ttkbootstrap',
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        'tkinter.simpledialog',
        'tkinter.ttk',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        
        # AI/API clients
        'google.generativeai',
        'openai',
        'tiktoken',
        'tiktoken_ext',
        'tiktoken_ext.openai_public',
        
        # Text processing
        'langdetect',
        'difflib',
        
        # Progress/UI
        'tqdm',
        
        # Network
        'requests',
        'chardet',
        'certifi',
        'urllib3',
        'idna',
        
        # Standard library modules that might be missed
        'json',
        'csv',
        'hashlib',
        'unicodedata',
        'tempfile',
        'shutil',
        'threading',
        'queue',
        're',
        'zipfile',
        'mimetypes',
        'collections',
        'io',
        'logging',
        'time',
        'os',
        'sys',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='Glossarion v1.4.8',  # Updated version
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
