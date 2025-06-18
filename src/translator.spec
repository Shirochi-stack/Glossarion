# -*- mode: python ; coding: utf-8 -*-
"""
Glossarion v2.6.9 - PyInstaller Specification File
Enhanced Translation Tool with QA Scanner
"""

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_NAME = 'Glossarion v2.6.9'
APP_ICON = 'Halgakos.ico'
ENABLE_CONSOLE = False  # Console disabled for production
ENABLE_UPX = False      # Compression (smaller file size but slower startup)
ONE_FILE = True        # Single executable vs folder distribution

# ============================================================================
# BLOCK CIPHER (for code obfuscation - optional)
# ============================================================================

block_cipher = None  # Set to pyi_crypto.PyiBlockCipher() if needed

# ============================================================================
# COLLECT DYNAMIC IMPORTS
# ============================================================================

# Collect all data files from specific packages
datas = []
binaries = []
hiddenimports = []

# Collect data files from packages that need them
for package in ['langdetect', 'certifi', 'tiktoken_ext', 'ttkbootstrap']:
    try:
        data, bins, hidden = collect_all(package)
        datas.extend(data)
        binaries.extend(bins)
        hiddenimports.extend(hidden)
    except:
        pass

# ============================================================================
# APPLICATION FILES
# ============================================================================

# Main application files
app_files = [
    # Core GUI
    ('translator_gui.py', '.'),
    ('splash_utils.py', '.'),
    
    # Translation modules
    ('TransateKRtoEN.py', '.'),
    ('unified_api_client.py', '.'),
    
    # File processors
    ('epub_converter.py', '.'),
    ('txt_processor.py', '.'),
    ('chapter_splitter.py', '.'),
    
    # Glossary extractors
    ('extract_glossary_from_epub.py', '.'),
    ('extract_glossary_from_txt.py', '.'),
    
    # Utilities
    ('scan_html_folder.py', '.'),
    ('history_manager.py', '.'),
    ('image_translator.py', '.'),
    ('check_epub_directory.py', '.'),
    ('direct_imports.py', '.'),
    
    # Resources
    ('Halgakos.ico', '.'),
]

# Add application files to datas
datas.extend(app_files)

# ============================================================================
# HIDDEN IMPORTS (Organized by category)
# ============================================================================

# Application modules
app_modules = [
    'TransateKRtoEN',
    'extract_glossary_from_epub',
    'extract_glossary_from_txt',
    'epub_converter',
    'txt_processor',
    'scan_html_folder',
    'unified_api_client',
    'chapter_splitter',
    'history_manager',
    'image_translator',
    'check_epub_directory',
    'direct_imports',
    'splash_utils',
]

# GUI Framework
gui_modules = [
    # Tkinter core
    'tkinter',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'tkinter.scrolledtext',
    'tkinter.simpledialog',
    'tkinter.ttk',
    '_tkinter',
    
    # TTKBootstrap
    'ttkbootstrap',
    'ttkbootstrap.constants',
    'ttkbootstrap.themes',
    'ttkbootstrap.style',
    'ttkbootstrap.utility',
    'ttkbootstrap.widgets',
    'ttkbootstrap.dialogs',
    'ttkbootstrap.tooltip',
    'ttkbootstrap.validation',
    'ttkbootstrap.scrolled',
    'ttkbootstrap.icons',
]

# EPUB/HTML Processing
epub_modules = [
    # EbookLib
    'ebooklib',
    'ebooklib.epub',
    'ebooklib.utils',
    'ebooklib.plugins',
    
    # BeautifulSoup
    'bs4',
    'soupsieve',
    
    # LXML
    'lxml',
    'lxml.etree',
    'lxml._elementpath',
    'lxml.html',
    'lxml.html.clean',
    'lxml.builder',
    'lxml.cssselect',
    
    # HTML processing
    'html5lib',
    'html',
    'html.parser',
    'html.entities',
    'cgi',
    'xml',
    'xml.etree',
    'xml.etree.ElementTree',
    'xml.dom',
    'xml.dom.minidom',
]

# Image Processing
image_modules = [
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'PIL.ImageEnhance',
    'PIL.ImageFilter',
    'PIL.ImageOps',
    'PIL.ImageChops',
    'PIL.ImageStat',
    'PIL.ImagePalette',
    'PIL.ImageSequence',
    'PIL.ImageGrab',
    'PIL.ImageMath',
    'PIL.ImageMode',
    'PIL.ImageShow',
    'PIL.ImageTransform',
    'PIL.ImageQt',
    'PIL.ImageCms',
    'PIL._binary',
    'PIL._imaging',
    'PIL._imagingft',
    'PIL._imagingmath',
    'PIL._imagingtk',
    'PIL._imagingcms',
    'PIL._webp',
    
    # Image format plugins
    'PIL.BmpImagePlugin',
    'PIL.GifImagePlugin',
    'PIL.JpegImagePlugin',
    'PIL.PngImagePlugin',
    'PIL.PpmImagePlugin',
    'PIL.TiffImagePlugin',
    'PIL.WebPImagePlugin',
    'PIL.IcoImagePlugin',
    'PIL.MicImagePlugin',
    'PIL.Jpeg2KImagePlugin',
    'PIL.IcnsImagePlugin',
    'PIL.DdsImagePlugin',
    'PIL.BlpImagePlugin',
    'PIL.FtexImagePlugin',
    
    'olefile',
]

# AI/API Clients
api_modules = [
    # Google AI
    'google',
    'google.generativeai',
    'google.ai',
    'google.ai.generativelanguage',
    'google.auth',
    'google.auth.transport',
    'google.auth.transport.requests',
    'google.auth.crypt',
    'google.auth.exceptions',
    'google.oauth2',
    'google.oauth2.credentials',
    'google.api_core',
    'google.api_core.gapic_v1',
    'google.api_core.operations_v1',
    'google.api_core.protobuf_helpers',
    'google.protobuf',
    'google.protobuf.message',
    'google.protobuf.descriptor',
    'google.protobuf.json_format',
    'google.rpc',
    'proto',
    'proto.message',
    
    # OpenAI
    'openai',
    'openai.api_resources',
    'openai.error',
    'openai.util',
    'openai.version',
    
    # Anthropic
    'anthropic',
    'httpx',
    'httpcore',
    'anyio',
    'sniffio',
    
    # Token counting
    'tiktoken',
    'tiktoken_ext',
    'tiktoken_ext.openai_public',
]

# Text Processing & Analysis
text_modules = [
    # Language detection
    'langdetect',
    'langdetect.detector',
    'langdetect.lang_detect_exception',
    'langdetect.language',
    'langdetect.detector_factory',
    'langdetect.utils',
    
    # Text analysis
    'difflib',
    'unicodedata',
    'string',
    'textwrap',
    're',
    'regex',
    
    # Datasketch for enhanced QA Scanner performance
    'datasketch',
    'datasketch.minhash',
    'datasketch.lsh',
    'datasketch.lshensemble',
    'datasketch.hashfunc',
    'datasketch.storage',
    'datasketch.lean_minhash',
    'datasketch.weighted_minhash',
    'datasketch.hyperloglog',
    'datasketch.hyperloglogplusplus',
    'datasketch.lshforest',
    'datasketch.b_bit_minhash',
    'datasketch.counting',
    'datasketch.hnsw',
    'datasketch.utils',
    
    # NumPy - Required by datasketch
    'numpy',
    'numpy.core',
    'numpy.core._multiarray_umath',
    'numpy.core.multiarray',
    'numpy.core.numeric',
    'numpy.core.shape_base',
    'numpy.core.fromnumeric',
    'numpy.core.getlimits',
    'numpy.core.arrayprint',
    'numpy.core._dtype',
    'numpy.core._type_aliases',
    'numpy.core._internal',
    'numpy.core._methods',
    'numpy.lib',
    'numpy.lib.type_check',
    'numpy.lib.npyio',
    'numpy.lib.format',
    'numpy.lib.arrayterator',
    'numpy.lib.arraypad',
    'numpy.lib.utils',
    'numpy.lib.stride_tricks',
    'numpy.linalg',
    'numpy.fft',
    'numpy.random',
    'numpy.random._common',
    'numpy.random.mtrand',
    'numpy.ctypeslib',
    'numpy.ma',
    'numpy.matrixlib',
    
    # SciPy - Required by datasketch
    'scipy',
    'scipy.integrate',
    'scipy.integrate._quadpack',
    'scipy.integrate._odepack',
    'scipy.integrate._quad_vec',
    'scipy.special',
    'scipy.special._ufuncs',
    'scipy.special._ufuncs_cxx',
    'scipy.special._basic',
    'scipy.special._logsumexp',
    'scipy.sparse',
    'scipy.sparse.linalg',
    'scipy.sparse.csgraph',
    'scipy._lib',
    'scipy._lib._util',
    'scipy._lib._ccallback',
    'scipy.version',
]

# Networking & HTTP
network_modules = [
    'requests',
    'requests.adapters',
    'requests.models',
    'requests.sessions',
    'requests.structures',
    'requests.utils',
    'requests.cookies',
    'requests.exceptions',
    'requests.auth',
    'requests.api',
    'chardet',
    'charset_normalizer',
    'certifi',
    'urllib3',
    'urllib3.util',
    'urllib3.poolmanager',
    'urllib3.connectionpool',
    'urllib3.connection',
    'urllib',
    'urllib.parse',
    'urllib.request',
    'urllib.error',
    'idna',
    'ssl',
    'socket',
    'select',
    'selectors',
    'http',
    'http.client',
    'http.cookies',
    'http.cookiejar',
    'email',
    'email.utils',
    'email.message',
    'email.header',
    'email.mime',
    'email.mime.text',
    'email.parser',
]

# Data Handling & Serialization
data_modules = [
    'json',
    'csv',
    'configparser',
    'pickle',
    'shelve',
    'sqlite3',
    'hashlib',
    'hmac',
    'secrets',
    'base64',
    'binascii',
    'struct',
    'array',
    'collections',
    'collections.abc',
    'dataclasses',
    'enum',
    'typing',
    'typing_extensions',
    'types',
]

# File & System Operations
system_modules = [
    'os',
    'os.path',
    'sys',
    'pathlib',
    'shutil',
    'tempfile',
    'zipfile',
    'tarfile',
    'gzip',
    'io',
    'mimetypes',
    'glob',
    'fnmatch',
    'filecmp',
    'stat',
    'platform',
    'subprocess',
    'multiprocessing',
    'multiprocessing.freeze_support',
    'multiprocessing.connection',
    'threading',
    'queue',
    'concurrent',
    'concurrent.futures',
    'asyncio',
]

# Date & Time
datetime_modules = [
    'datetime',
    'time',
    'calendar',
    'zoneinfo',
    '_zoneinfo',
    'tzdata',
    'pytz',
    'dateutil',
    'dateutil.parser',
    'dateutil.tz',
]

# Utilities & Helpers
utility_modules = [
    'tqdm',
    'tqdm.auto',
    'tqdm.std',
    'tqdm.gui',
    'tqdm.notebook',
    'logging',
    'logging.handlers',
    'logging.config',
    'warnings',
    'traceback',
    'contextlib',
    'functools',
    'itertools',
    'operator',
    'copy',
    'weakref',
    'gc',
    'atexit',
    'signal',
    'locale',
    'gettext',
    'uuid',
    'random',
    'math',
    'decimal',
    'fractions',
    'numbers',
    'cmath',
    'argparse',
    'getopt',
    'cmd',
    'shlex',
    'pprint',
    'reprlib',
    'dis',
    'inspect',
    'ast',
    'imp',
    'importlib',
    'importlib.util',
    'importlib.machinery',
    'importlib.metadata',
    'pkg_resources',
    'pkg_resources._vendor',
    'pkg_resources.extern',
    'setuptools',
    'distutils',
    'sysconfig',
    'site',
    'sitecustomize',
    'usercustomize',
]

# Encoding support
encoding_modules = [
    'encodings',
    'encodings.utf_8',
    'encodings.ascii',
    'encodings.latin_1',
    'encodings.cp1252',
    'encodings.cp437',
    'encodings.utf_16',
    'encodings.utf_16_le',
    'encodings.utf_16_be',
    'encodings.utf_32',
    'encodings.utf_32_le',
    'encodings.utf_32_be',
    'encodings.unicode_escape',
    'encodings.raw_unicode_escape',
    'encodings.idna',
    'codecs',
]

# Combine all hidden imports
hiddenimports.extend(app_modules)
hiddenimports.extend(gui_modules)
hiddenimports.extend(epub_modules)
hiddenimports.extend(image_modules)
hiddenimports.extend(api_modules)
hiddenimports.extend(text_modules)
hiddenimports.extend(network_modules)
hiddenimports.extend(data_modules)
hiddenimports.extend(system_modules)
hiddenimports.extend(datetime_modules)
hiddenimports.extend(utility_modules)
hiddenimports.extend(encoding_modules)

# Remove duplicates
hiddenimports = list(set(hiddenimports))

# ============================================================================
# EXCLUSIONS (Packages to exclude to reduce size)
# ============================================================================

excludes = [
    # Large scientific packages (unless needed)
    'matplotlib',
    'pandas',
    # 'scipy',  # Required by datasketch - do not exclude
    # 'numpy',  # Required by datasketch - do not exclude
    'sklearn',
    'skimage',
    'cv2',
    
    # Testing frameworks
    'pytest',
    'nose',
    'unittest',
    'doctest',
    'test',
    'tests',
    
    # Development tools
    'IPython',
    'jupyter',
    'notebook',
    'ipykernel',
    'ipywidgets',
    'pylint',
    'black',
    'flake8',
    'mypy',
    'coverage',
    
    # Documentation
    'sphinx',
    'docutils',
    
    # Other unnecessary packages
    'PyQt5',
    'PyQt6',
    'PySide2',
    'PySide6',
    'wx',
    'kivy',
    'pygame',
    'tornado',
    'flask',
    'django',
    'fastapi',
    'uvicorn',
    'colorama',  # Unless you need colored console output
    'win32com',  # Unless you need Windows COM
    'pythoncom', # Unless you need Windows COM
]

# ============================================================================
# ANALYSIS
# ============================================================================

a = Analysis(
    ['translator_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ============================================================================
# PYZ (Python Zip archive)
# ============================================================================

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

# ============================================================================
# EXECUTABLE CONFIGURATION
# ============================================================================

if ONE_FILE:
    # Single file executable
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=ENABLE_UPX,
        upx_exclude=[
            'vcruntime140.dll',  # Don't compress Windows runtime
            'python*.dll',       # Don't compress Python DLLs
            'api-ms-win-*.dll',  # Don't compress Windows API DLLs
        ],
        runtime_tmpdir=None,
        console=ENABLE_CONSOLE,
        disable_windowed_traceback=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=APP_ICON,
        version='version_info.txt' if os.path.exists('version_info.txt') else None,
    )
else:
    # Folder distribution
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=ENABLE_CONSOLE,
        disable_windowed_traceback=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=APP_ICON,
        version='version_info.txt' if os.path.exists('version_info.txt') else None,
    )
    
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=ENABLE_UPX,
        upx_exclude=[
            'vcruntime140.dll',
            'python*.dll',
            'api-ms-win-*.dll',
        ],
        name=APP_NAME.replace(' ', '_'),
    )

# ============================================================================
# NOTES
# ============================================================================

"""
Build Instructions:
1. Install PyInstaller: pip install pyinstaller
2. Install datasketch: pip install datasketch
3. Run: pyinstaller translator.spec

Optimization Tips:
- Set ENABLE_UPX = False for faster startup but larger file
- Set ONE_FILE = False for faster startup but folder distribution

This build includes datasketch for enhanced QA scanning performance.
The executable will be larger (~100-150MB more) but will provide
significantly faster duplicate detection on large datasets (50+ files).

For version information:
Create a version_info.txt file with Windows version resource information
"""
