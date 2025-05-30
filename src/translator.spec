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
    ('Halgakos.ico', '.'),  # Include icon if you have it
]

a = Analysis(
    ['translator_gui.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'TransateKRtoEN',
        'extract_glossary_from_epub',
        'epub_converter',
        'scan_html_folder',
        'unified_api_client',
        'ebooklib',
        'ebooklib.epub',
        'bs4',
        'BeautifulSoup',
        'ttkbootstrap',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'google.generativeai',
        'openai',
        'tiktoken',
        'tiktoken_ext',
        'tiktoken_ext.openai_public',
        'langdetect',
        'difflib',
        'tqdm',
        'requests',
        'chardet',
        'certifi',
        'urllib3',
        'idna',
        'lxml',
        'lxml.etree',
        'lxml._elementpath',
        'html5lib',
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
    name='Glossarion v1.3',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want to see console output
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Halgakos.ico'  # Add icon path if you have one
)