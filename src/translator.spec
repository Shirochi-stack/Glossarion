# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Add all your python files here
added_files = [
    ('translator_gui.py', '.'),
    ('splash_screen.py', '.'),              # NEW - Separate splash screen script
    ('splash_utils.py', '.'),               # NEW - Splash management utilities
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
    datas=added_files,
    hiddenimports=[
        # ==================== CORE PROJECT MODULES ====================
        'TransateKRtoEN',
        'extract_glossary_from_epub',
        'epub_converter',
        'scan_html_folder',
        'unified_api_client',
        'chapter_splitter',
        'history_manager',
        'check_epub_directory',
        'direct_imports',
        'splash_screen',              # NEW - Splash screen module
        'splash_utils',               # NEW - Splash utilities module
        
        # ==================== EPUB PROCESSING ====================
        'ebooklib',
        'ebooklib.epub',
        'ebooklib.utils',
        'ebooklib.plugins',
        'ebooklib.plugins.base',
        'ebooklib.plugins.standard',
        
        # ==================== HTML/XML PARSING ====================
        'bs4',
        'bs4.builder',
        'bs4.builder._html5lib',
        'bs4.builder._lxml',
        'bs4.builder._htmlparser',
        'bs4.dammit',
        'BeautifulSoup',
        'lxml',
        'lxml.etree',
        'lxml._elementpath',
        'lxml.html',
        'lxml.html.clean',
        'html5lib',
        'html',
        'html.parser',
        'cgi',                    # CRITICAL - used in epub_converter.py fallback
        'HTMLParser',             # CRITICAL - used in epub_converter.py fallback
        
        # ==================== XML PROCESSING ====================
        'xml',
        'xml.etree',
        'xml.etree.ElementTree',
        'xml.sax',
        'xml.sax.handler',
        'xml.dom',
        'xml.dom.minidom',
        'xml.parsers',
        'xml.parsers.expat',
        
        # ==================== GUI FRAMEWORK ====================
        'ttkbootstrap',
        'ttkbootstrap.constants',
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        'tkinter.simpledialog',
        'tkinter.ttk',
        'tkinter.font',
        'tkinter.colorchooser',
        'tkinter.commondialog',
        'tkinter.dnd',
        '_tkinter',
        
        # ==================== IMAGE PROCESSING (PIL/Pillow) ====================
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
        'PIL.IcoImagePlugin',        # CRITICAL - for splash screen icon loading
        'PIL.MicImagePlugin',
        'PIL.PcxImagePlugin',
        'PIL.TarImagePlugin',
        'PIL.XpmImagePlugin',
        'PIL.ExifTags',
        'PIL.ImageChops',
        'PIL.ImageColor',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        'PIL.ImageOps',
        'PIL.ImageSequence',
        'PIL.ImageStat',
        'olefile',                # CRITICAL - fixes MicImagePlugin error
        
        # ==================== AI/API CLIENTS ====================
        # OpenAI
        'openai',
        'openai.api_resources',
        'openai.error',
        'openai.util',
        
        # Google AI
        'google',
        'google.generativeai',
        'google.ai',
        'google.ai.generativelanguage',
        'google.auth',
        'google.auth.transport',
        'google.auth.transport.requests',
        'google.auth.exceptions',
        'google.api_core',         # CRITICAL - Google API dependency
        'google.api_core.exceptions', # CRITICAL - Google API dependency
        'google.protobuf',
        'grpcio',                  # CRITICAL - Google API dependency
        'grpcio._channel',
        
        # Tokenization
        'tiktoken',
        'tiktoken_ext',
        'tiktoken_ext.openai_public',
        'tiktoken.load',
        'tiktoken.registry',
        'tiktoken.model',
        'regex',
        
        # ==================== NETWORK/HTTP ====================
        'requests',
        'requests.adapters',
        'requests.models',
        'requests.sessions',
        'chardet',
        'certifi',
        'urllib3',
        'urllib',
        'urllib.parse',
        'urllib.request',
        'idna',
        'ssl',
        'socket',
        'http',
        'http.client',
        'http.cookiejar',
        'http.cookies',
        'email',
        'email.utils',
        'email.mime',
        'email.mime.text',
        'email.mime.multipart',
        'email.encoders',
        
        # ==================== TEXT/LANGUAGE PROCESSING ====================
        'langdetect',
        'langdetect.detector',
        'langdetect.lang_detect_exception',
        'langdetect.detector_factory',
        'langdetect.language',
        'difflib',
        'unicodedata',
        'string',
        'textwrap',
        'gettext',
        
        # ==================== PROGRESS BARS ====================
        'tqdm',
        'tqdm.auto',
        'tqdm.std',
        'tqdm.gui',
        'tqdm.tk',
        'tqdm.notebook',
        'tqdm.asyncio',
        
        # ==================== PROCESS MANAGEMENT ====================
        'subprocess',             # CRITICAL - used by splash_utils.py to launch splash
        'multiprocessing',
        'multiprocessing.pool',
        'threading',              # CRITICAL - used by splash screen
        '_thread',
        'queue',
        'concurrent',             # CRITICAL - used by various libraries
        'concurrent.futures',     # CRITICAL - used by various libraries
        'sched',
        'contextlib',
        'atexit',                 # CRITICAL - used by splash_utils.py for cleanup
        'signal',                 # CRITICAL - for process management
        
        # ==================== FILE SYSTEM/COMPRESSION ====================
        'os',
        'os.path',
        'sys',
        'pathlib',
        'tempfile',
        'shutil',
        'zipfile',
        'gzip',
        'zlib',
        'bz2',
        'lzma',
        'glob',
        'fnmatch',
        'stat',
        'fileinput',
        'mimetypes',
        
        # ==================== DATA STRUCTURES/ALGORITHMS ====================
        'collections',
        'collections.abc',
        'json',
        'csv',
        'hashlib',
        'hmac',
        'random',
        'binascii',
        'base64',
        'secrets',
        'uuid',
        'functools',
        'itertools',
        'operator',
        'copy',
        'weakref',
        
        # ==================== ENCODING/TEXT ====================
        'encodings',
        'encodings.utf_8',
        'encodings.ascii',
        'encodings.latin_1',
        'encodings.cp1252',
        'encodings.utf_16',
        'encodings.utf_32',
        'codecs',
        'locale',
        'struct',
        
        # ==================== STANDARD LIBRARY CORE ====================
        're',
        'io',
        'time',
        'datetime',
        'argparse',
        'platform',
        'warnings',
        'types',
        'typing',
        'typing_extensions',
        'dataclasses',
        'inspect',
        
        # ==================== LOGGING/DEBUGGING ====================
        'logging',
        'logging.handlers',
        'traceback',
        'linecache',
        'gc',
        'resource',
        
        # ==================== PACKAGE MANAGEMENT ====================
        'pkg_resources',
        'pkg_resources._vendor',
        
        # ==================== PLATFORM-SPECIFIC (conditionally imported) ====================
        'msvcrt',     # Windows only
        'termios',    # Unix only
        'fcntl',      # Unix only
        'pwd',        # Unix only
        'grp',        # Unix only
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude large packages not used
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
		'multiprocessing.spawn',
		'multiprocessing.forkserver',
        'pytest',
        'nose',
        'jupyter',
        'IPython',
        'notebook',
        'tornado',
        'django',
        'flask',
        'sqlalchemy',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
	optimize=0,
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
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Keep as False for clean splash screen experience
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Halgakos.ico'
)
