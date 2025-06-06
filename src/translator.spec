# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Core application files
app_files = [
    ('translator_gui.py', '.'),
    ('splash_utils.py', '.'),
    ('TransateKRtoEN.py', '.'),
    ('extract_glossary_from_epub.py', '.'),
    ('epub_converter.py', '.'),
    ('scan_html_folder.py', '.'),
    ('unified_api_client.py', '.'),
    ('chapter_splitter.py', '.'),
    ('history_manager.py', '.'),
    ('check_epub_directory.py', '.'),
    ('direct_imports.py', '.'),
    ('Halgakos.ico', '.'),
]

a = Analysis(
    ['translator_gui.py'],
    pathex=[],
    binaries=[],
    datas=app_files,
    hiddenimports=[
        # ==================== CORE APPLICATION ====================
        'TransateKRtoEN',
        'extract_glossary_from_epub', 
        'epub_converter',
        'scan_html_folder',
        'unified_api_client',
        'chapter_splitter',
        'history_manager',
        'check_epub_directory',
        'direct_imports',
        'splash_utils',
        
        # ==================== GUI FRAMEWORK ====================
        'ttkbootstrap',
        'ttkbootstrap.constants',
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        'tkinter.simpledialog',
        'tkinter.ttk',
        '_tkinter',
        
        # ==================== WEB & API ====================
        'requests',
        'requests.adapters',
        'urllib3',
        'certifi',
        'chardet',
        'idna',
        
        # ==================== AI CLIENTS ====================
        'openai',
        'google.generativeai',
        'tiktoken',
        'tiktoken_ext',
        
        # ==================== EPUB & HTML PROCESSING ====================
        'ebooklib',
        'ebooklib.epub',
        'bs4',
        'lxml',
        'lxml.etree',
        'html5lib',
        
        # ==================== IMAGE PROCESSING ====================
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'PIL.ImageOps',
        'PIL.IcoImagePlugin',
        
        # ==================== LANGUAGE & TEXT ====================
        'langdetect',
        'unicodedata',
        'difflib',
        
        # ==================== PROGRESS & UI ====================
        'tqdm',
        
        # ==================== CORE PYTHON ====================
        'json',
        'os',
        'sys',
        'time',
        'threading',
        'queue',
        'zipfile',
        'shutil',
        'tempfile',
        'hashlib',
        'logging',
        'collections',
        're',
        'io',
        'pathlib',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
    name='Glossarion v1.6.6',
    debug=False,
    bootloader_ignore_signals=True,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Halgakos.ico'
)
