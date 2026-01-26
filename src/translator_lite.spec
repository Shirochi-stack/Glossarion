# -*- mode: python ; coding: utf-8 -*-
"""
Glossarion Lite v7.3.7 - PyInstaller Specification File
Enhanced Translation Tool with QA Scanner, and AI Hunter
"""

import sys
import os

# Fix DLL search path for WeasyPrint during build
if os.name == 'nt' and os.path.exists(r'C:\msys64\mingw64\bin'):
    os.environ['PATH'] = r'C:\msys64\mingw64\bin' + os.pathsep + os.environ.get('PATH', '')

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_NAME = 'L_Glossarion_Lite v7.3.7'  # CHANGED: Updated version
APP_ICON = 'Halgakos.ico'
ENABLE_CONSOLE = False  # Console disabled for production
ENABLE_UPX = False      # Compression (smaller file size but slower startup)
ONE_FILE = True         # Single executable vs folder distribution

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

# Add custom DLL and CPP files
binaries.extend([])

# Add MSYS2 DLLs for WeasyPrint (PDF generation with formatting and images)
import glob
msys2_bin = r'C:\msys64\mingw64\bin'
if os.path.exists(msys2_bin):
    for dll in glob.glob(os.path.join(msys2_bin, '*.dll')):
        binaries.append((dll, '.'))
    print(f"  Added {len(glob.glob(os.path.join(msys2_bin, '*.dll')))} MSYS2 DLLs for WeasyPrint")

# Collect data files from packages that need them
for package in ['langdetect', 'certifi', 'tiktoken_ext', 'ttkbootstrap', 'chardet', 'charset_normalizer']:
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

# Main application files
app_files = [
    # Core GUI
    ('translator_gui.py', '.'),
    ('splash_utils.py', '.'),
    ('other_settings.py', '.'),
    ('GlossaryManager.py', '.'),
    ('GlossaryManager_GUI.py', '.'),
    ('Retranslation_GUI.py', '.'),
    ('QA_Scanner_GUI.py', '.'),
    ('Chapter_Extractor.py', '.'),
    ('PatternManager.py', '.'),
    
    # Translation modules
    ('TransateKRtoEN.py', '.'),
    ('unified_api_client.py', '.'),
    ('google_free_translate.py', '.'),
    
    # File processors
    ('epub_converter.py', '.'),
    ('txt_processor.py', '.'),
    ('chapter_splitter.py', '.'),
    
    # Glossary extractors
    ('extract_glossary_from_epub.py', '.'),
    ('extract_glossary_from_txt.py', '.'),
    ('glossary_process_worker.py', '.'),  # Glossary subprocess worker
    ('chapter_extraction_worker.py', '.'),  # Chapter extraction subprocess worker
    ('chapter_extraction_manager.py', '.'),  # Chapter extraction manager
    
    # Utilities
    ('scan_html_folder.py', '.'),
    ('history_manager.py', '.'),
    ('image_translator.py', '.'),
    ('check_epub_directory.py', '.'),
    ('direct_imports.py', '.'),
    ('api_key_encryption.py', '.'), 
    ('http_logger.py', '.'),
    ('shutdown_utils.py', '.'),
    
    # AI Hunter Enhanced
    ('ai_hunter_enhanced.py', '.'),
    
    # Dialog animations
    ('dialog_animations.py', '.'),
    
    # Spinning icon helper
    ('spinning.py', '.'),
    
    # Update Manager
    ('update_manager.py', '.'),
	
	# Async Processing
    ('async_api_processor.py', '.'),
	
	# Metadata and header batch translation
    ('metadata_batch_translator.py', '.'),
    ('translate_headers_standalone.py', '.'),
    
    # Resources
    ('Halgakos.ico', '.'),
	
	('enhanced_text_extractor.py', '.'),	
	('pdf_extractor.py', '.'),
	
	('multi_api_key_manager.py', '.'),
	('individual_endpoint_dialog.py', '.'),
	('model_options.py', '.'),
	('hyphen_textwrap.py', '.'),
	
	# Duplicate detection
	('duplicate_detection_config.py', '.'),
	('advanced_duplicate_detection.py', '.'),
	
]
# Add application files to datas
datas.extend(app_files)
datas.append(('memory_usage_reporter.py', '.'))
datas.append(('tqdm_safety.py', '.'))
datas.append(('debug_env_vars.py', '.'))
datas.append(('enable_debug_mode.py', '.'))

# ============================================================================
# HIDDEN IMPORTS (Organized by category)
# ============================================================================

# Application modules
app_modules = [
    'TransateKRtoEN',
    'extract_glossary_from_epub',
    'extract_glossary_from_txt',
    'glossary_process_worker',  # Glossary subprocess worker
    'chapter_extraction_worker',  # Chapter extraction subprocess worker
    'chapter_extraction_manager',  # Chapter extraction manager
    'GlossaryManager',
    'GlossaryManager_GUI',
    'Retranslation_GUI',
    'QA_Scanner_GUI',
    'Chapter_Extractor',
    'PatternManager',
    'epub_converter',
    'txt_processor',
    'scan_html_folder',
    'unified_api_client',
    'google_free_translate',
    'chapter_splitter',
    'history_manager',
    'image_translator',
    'check_epub_directory',
    'direct_imports',
    'splash_utils',
    'other_settings',      # Other Settings module
    'ai_hunter_enhanced',  # AI Hunter Enhanced module
    'dialog_animations',   # Dialog fade animations
    'spinning',            # Spinning icon helper
    'update_manager',
    'api_key_encryption',
	'http_logger',
	'shutdown_utils',
	'async_api_processor',
	'metadata_batch_translator',
	'translate_headers_standalone',
	'enhanced_text_extractor.py',
	'pdf_extractor',
	'multi_api_key_manager.py',
	'individual_endpoint_dialog.py',
	'model_options',
	'hyphen_textwrap',
	'duplicate_detection_config',
	'advanced_duplicate_detection',
	
]
# GUI Framework
gui_modules = [
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
    'ttkbootstrap.colorutils',
    'ttkbootstrap.themes.standard',
    'ttkbootstrap.themes.user',
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
    'bs4.element',
    'bs4.builder',
    'bs4.builder._html5lib',
    'bs4.builder._htmlparser',
    'bs4.builder._lxml',
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
    'html5lib.treebuilders',
    'html5lib.treewalkers',
    'html5lib.serializer',
    'html',
    'html.parser',
    'html.entities',
    'cgi',
    'xml',
    'xml.etree',
    'xml.etree.ElementTree',
    'xml.dom',
    'xml.dom.minidom',
    'xml.parsers',
    'xml.parsers.expat',
]

# Image Processing (Enhanced for Manga)
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
    'PIL.ImImagePlugin',
    'PIL.Jpeg2KImagePlugin',
    'PIL.MspImagePlugin',
    'PIL.PcxImagePlugin',
    'PIL.SgiImagePlugin',
    'PIL.TgaImagePlugin',
    'PIL.XbmImagePlugin',
    'PIL.XpmImagePlugin',
    'PIL.DdsImagePlugin',
    'PIL.BlpImagePlugin',
    'PIL.FtexImagePlugin',
    
    'olefile',
]

# AI/API Clients (Including Google Cloud Vision)
api_modules = [
    # Google AI
    'google',
	'google.genai',
	'google.genai.types',
    'google.auth',
    'google.auth.transport',
    'google.auth.transport.requests',
    'google.auth.transport.grpc',
    'google.auth.crypt',
    'google.auth.exceptions',
    'google.oauth2',
    'google.oauth2.credentials',
    'google.api_core',
    'google.api_core.client_options',
    'google.api_core.exceptions',
    'google.api_core.gapic_v1',
    'google.api_core.operations_v1',
    'google.api_core.protobuf_helpers',
    'google.protobuf',
    'google.protobuf.message',
    'google.protobuf.descriptor',
    'google.protobuf.json_format',
    'google.protobuf.internal',
    'google.protobuf.reflection',
    'google.rpc',
    'google.type',
    
    # Azure (for Azure OpenAI endpoints - NOT Computer Vision)
    'azure',
    'azure.core',
    'azure.core.credentials',
    'azure.core.exceptions',
    'azure.core.pipeline',
    'azure.core.pipeline.transport',
    'azure.core.pipeline.policies',
    'azure.core.rest',
    'azure.core.tracing',
    'azure.core.utils',
    'azure.identity',
    'azure.common',
    
    # Additional Azure dependencies
    'isodate',  # Required by Azure
    'oauthlib',  # May be required for Azure auth
    'requests_oauthlib',  # May be required for Azure auth
	
	# Google Cloud Translate
	'google.cloud.translate',
	'google.cloud.translate_v2',
	'google.cloud.translate_v3',
	'google.cloud.translate_v3.types',
	'google.cloud.translate_v3.services',
	'google.cloud.translate_v3.services.translation_service',
	
	# DeepL
	'deepl',
	'deepl.translator',
	'deepl.exceptions',
	'deepl.api',
	'deepl.http',
	'deepl.util',
	'deepl.auth',
	'deepl.model',	
    
    'proto',
    'proto.message',
    'grpcio',
    'grpcio_status',
    'googleapis_common_protos',
	
	# Google Vertex AI:
    'google.cloud.aiplatform',
    'google.cloud.aiplatform_v1', 
    'google.cloud.aiplatform_v1beta1',
    'vertexai',
    'vertexai.generative_models',
    'vertexai.language_models',
    
    # OpenAI
    'openai',
    'openai.api_resources',
    'openai.error',
    'openai.util',
    'openai.version',
    'openai.api_requestor',
    'openai.openai_response',
    'openai._base_client',
    'openai._constants',
    'openai._models',
    'openai._response',
    'openai._legacy_response',
    'openai._streaming',
    'openai._exceptions',
    'openai.resources',
    'openai.resources.chat',
    'openai.resources.completions',
    'openai.types',
    'openai.types.chat',
    
    # Anthropic
    'anthropic',
    'anthropic._client',
    'anthropic._base_client',
    'anthropic._constants',
    'anthropic._models',
    'anthropic._response',
    'anthropic._streaming',
    'anthropic._exceptions',
    'anthropic.resources',
    'anthropic.resources.messages',
    'anthropic.types',
    'anthropic.types.message',
    'anthropic.types.content_block',
    'anthropic.types.usage',
    
    # HTTP clients
    'httpx',
    'httpx._client',
    'httpx._config',
    'httpx._models',
    'httpx._transports',
    'httpx._types',
    'httpcore',
    'httpcore._sync',
    'httpcore._async',
    'h11',
    'h11._connection',
    'h11._events',
    'h11._state',
    'h11._util',
    'h11._writers',
    'h2',
    'hyperframe',
    'hpack',
    'socksio',
    'sniffio',
    'anyio',
    'anyio._core',
    'anyio._core._eventloop',
    'anyio.streams',
    'anyio.streams.memory',
	
	# POE API Wrapper (add these at the end)
    'poe_api_wrapper',
    'poe_api_wrapper.api',
    'poe_api_wrapper.client',
    'poe_api_wrapper.models',
    'poe_api_wrapper.utils',
    'ballyregan',
    'ballyregan.proxies',
    
    # WebSocket support for POE
    'websocket',
    'websocket._core',
    'websocket._app',
    'websocket._url',
    'websocket._http',
    'websocket._logging',
    'websocket._socket',
    'websocket._ssl_compat',
    'websocket._abnf',
    'websocket._handshake',
    'websocket._exceptions',
    'websockets',
]

# Text Processing & NLP
text_modules = [

    # Language detection
    'langdetect',
    'langdetect.detector',
    'langdetect.lang_detect_exception',
    'langdetect.language',
    'langdetect.detector_factory',
    'langdetect.utils',
	
	# Fuzzy string matching (ADD THIS SECTION)
    'rapidfuzz',
    'rapidfuzz.fuzz',
    'rapidfuzz.process',
    'rapidfuzz.distance',
    'rapidfuzz.utils',
    
    # Token counting
    'tiktoken',
    'tiktoken_ext',
    'tiktoken_ext.openai_public',
    'tiktoken.core',
    'tiktoken.registry',
    'tiktoken.load',
    'tiktoken.model',
	
	# Markdown2
	'markdown2',
    'markdown2.extras',

    # HTML to text conversion
    'html2text',
    'html2text.__init__',
    'html2text.config',
    'html2text.compat',
    'html2text.utils',
	
	# ilru cache
	'functools',
	'lru',
 
    # AI Hunter (Datasketch)
    'datasketch',
    'datasketch.minhash',
    'datasketch.lsh',
    'datasketch.lshensemble',
    'datasketch.weighted_minhash',
    'datasketch.hyperloglog',
    'datasketch.lshforest',
    'datasketch.lean_minhash',
    'datasketch.hashfunc',
    'datasketch.storage',
    'datasketch.experimental',
    'datasketch.version',
    
    # Regex
    'regex',
    'regex._regex',
    'regex._regex_core',
    're',
    '_sre',
    'sre_compile',
    'sre_parse',
    'sre_constants',
    
    # JSON processing
    'json',
    'json.decoder',
    'json.encoder',
    'json.scanner',
    '_json',
    'simplejson',  # fallback
]

# Network & System
network_modules = [
    'requests',
    'requests.models',
    'requests.sessions',
    'requests.auth',
    'requests.cookies',
    'requests.exceptions',
    'requests.packages',
    'requests.packages.urllib3',
    'requests.adapters',
    'requests.api',
    'requests.structures',
    'requests.utils',
    'urllib',
    'urllib.parse',
    'urllib.request',
    'urllib.error',
    'urllib.response',
    'urllib3',
    'urllib3.connection',
    'urllib3.connectionpool',
    'urllib3.poolmanager',
    'urllib3.response',
    'urllib3.util',
    'urllib3.util.ssl_',
    'urllib3.util.retry',
    'urllib3.contrib',
    'certifi',
    'certifi.core',
    'ssl',
    '_ssl',
    'socket',
    '_socket',
    'select',
    'selectors',
    'socketserver',
    'http',
    'http.client',
    'http.server',
    'http.cookies',
    'http.cookiejar',
    'email',
    'email.utils',
    'email.message',
    'email.header',
    'email.charset',
    'email.encoders',
    'email.errors',
    'email.generator',
    'email.iterators',
    'email.mime',
    'email.parser',
    'email.policy',
    'mimetypes',
    'base64',
    'binascii',
    'quopri',
    'uu',
]

# Data Processing
data_modules = [
    'csv',
    '_csv',
    'pickle',
    '_pickle',
    'cPickle',
    'cpickle',
    'shelve',
    'dbm',
    'sqlite3',
    '_sqlite3',
    'gzip',
    'zlib',
    'bz2',
    '_bz2',
    'lzma',
    '_lzma',
    'zipfile',
    'tarfile',
    'shutil',
    'glob',
    'fnmatch',
    'pathlib',
    'tempfile',
    'io',
    '_io',
    'StringIO',
    'BytesIO',
    'hashlib',
    '_hashlib',
    '_blake2',
    '_sha3',
    'hmac',
    'secrets',
    '_random',
    'bisect',
    '_bisect',
    'heapq',
    '_heapq',
    'array',
    'collections',
    'collections.abc',
    '_collections',
    '_collections_abc',
]

# System & OS
system_modules = [
    'os',
    'os.path',
    'ntpath',
    'posixpath',
    'genericpath',
    'stat',
    '_stat',
    'sys',
    'platform',
    'subprocess',
    '_subprocess',
    '_winapi',
    'msvcrt',
    '_msvcrt',
    'errno',
    'signal',
    '_signal',
    'atexit',
    'gc',
    '_gc',
    'multiprocessing',
    'multiprocessing.freeze_support',
    'multiprocessing.connection',
    'multiprocessing.pool',
    'multiprocessing.process',
    'threading',
    'queue',
    'concurrent',
    'concurrent.futures',
    'concurrent.futures._base',
    'asyncio',
    'asyncio.base_events',
    'asyncio.events',
    'asyncio.futures',
    'asyncio.tasks',
    'asyncio.protocols',
    'asyncio.streams',
    'asyncio.subprocess',
    'asyncio.queues',
    'ctypes',
    'ctypes.util',
    'ctypes.wintypes',
    'aiohttp',
    'aiofiles',
	'yarl',  # URL handling for aiohttp
    'multidict',  # Required by aiohttp
    'async_timeout',  # Required by aiohttp
    'attrs',  # Required by aiohttp
    'charset_normalizer',  # Encoding detection
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
    'dateutil.relativedelta',
    'dateutil.rrule',
]

# Utilities & Helpers
utility_modules = [
    'tqdm',
    'tqdm.auto',
	'dataclasses',
    'tqdm.std',
    'tqdm.gui',
	'dataclasses',
    'tqdm.notebook',
	'concurrent.futures',
    'tqdm.utils',
    'tqdm.cli',
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
    'statistics',
    'argparse',
    'getopt',
    'cmd',
    'shlex',
    'pprint',
    'reprlib',
    'dis',
    'inspect',
    'ast',
    'importlib',
    'importlib.util',
    'importlib.machinery',
    'importlib.metadata',
    'importlib.resources',
    'pkg_resources',
    'pkg_resources._vendor',
    'pkg_resources.extern',
    'setuptools',
    'distutils',
    'sysconfig',
    'site',
    'sitecustomize',
    'usercustomize',
    'dotenv',
    'python-dotenv',
    'os.environ',
    'dotenv.main',
    'dotenv.parser',
    'dataclasses',  # For manga TextRegion dataclass
    # ADDED: Version parsing for update manager
    'packaging',
    'packaging.version',
    'packaging.specifiers',
    'packaging.requirements',
	'cryptography',
    'cryptography.fernet',
    'cryptography.hazmat',
    'cryptography.hazmat.primitives',
    'cryptography.hazmat.primitives.kdf',
    'cryptography.hazmat.primitives.kdf.pbkdf2',
    'cryptography.hazmat.primitives.hashes',
    'cryptography.hazmat.backends',
    'cryptography.hazmat.backends.openssl',
	'numpy',
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
    'encodings.aliases',
    'codecs',
]

# Combine all hidden imports
hiddenimports.append('memory_usage_reporter')
hiddenimports.append('tqdm_safety')
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
    # ============================================================================
    # MACHINE LEARNING & AI FRAMEWORKS (MAJOR SIZE REDUCTION)
    # ============================================================================
    
    # PyTorch ecosystem (~800MB)
    'torch', 'torch.*','torch-*',
    'torchvision', 'torchvision.*', 
    'torchaudio', 'torchaudio.*',
    'torch.nn', 'torch.nn.*',
    'torch.cuda', 'torch.cuda.*',
    'torch.jit', 'torch.onnx', 'torch.autograd',
    'torch.optim', 'torch.utils', 'torch.distributed',
    'torch.multiprocessing', 'torch.serialization',
    'torch.nn.modules.*',
    
    # HuggingFace ecosystem (~400MB)
    'transformers', 'transformers.*',
    'tokenizers', 'tokenizers.*',
    'huggingface_hub', 'huggingface_hub.*',
    'safetensors', 'safetensors.*',
    'accelerate', 'accelerate.*',
    
    # Computer Vision & YOLO (~200MB)
    'ultralytics', 'ultralytics.*',
    'ultralytics-thop',
    
    # ONNX Runtime (~300MB)
    'onnx', 'onnx.*',
    'onnxruntime', 'onnxruntime.*',
    'onnxruntime-gpu',
    'onnxruntime.capi', 'onnxruntime.capi.*',
    'onnxruntime.tools', 'onnxruntime.transformers',
    'onnxruntime.backend', 'onnxruntime.backend.*',
    
    # OCR Libraries (~300MB) - MAJOR ADDITION
    'easyocr', 'easyocr.*',
    'manga-ocr', 'manga_ocr.*',
    'paddleocr', 'paddleocr.*',
    'paddlepaddle', 'paddlepaddle.*',
    'paddlex', 'paddlex.*',
    'python-doctr', 'python_doctr.*',
    
    # Multiple OpenCV versions (~150MB) - MAJOR ADDITION
    'opencv-contrib-python',
    'opencv-python',  # Keep opencv-python-headless only
	'cv2',
    'cv2.contrib',
    
    # Scientific Computing (Optional)
    'matplotlib', 'matplotlib.*',
    'pandas', 'pandas.*',
    'scikit-image', 'skimage', 'skimage.*',
    'sklearn', 'sklearn.*',
    
	# Remove AVIF support if not needed (7MB)
    'PIL._avif',
    'pillow.libs',
    
    # Remove PDF support if not needed (5MB)
    'pypdfium2', 'pypdfium2.*',
    'pypdfium2_raw',
	
	# Scientific/Data formats (15MB savings)
    'h5py', 'h5py.*',
    'tables', 'tables.*',
    
    # Geographic/Geometry (2MB)
    'shapely', 'shapely.*',
    'Shapely', 'Shapely.*',
    
    # Audio processing (2MB)
    'soundfile', 'soundfile.*',
    '_soundfile_data',
    'librosa', 'librosa.*',
	
	# Fix numpy source directory issue
    'numpy.f2py',
    'numpy.f2py.*',
    'numpy.testing',
    'numpy.testing.*',
    'numpy.tests',
    'numpy.distutils',
    'numpy.distutils.*',
    'docstring_parser.numpydoc',  # This is pulling in numpy incorrectly
    'numpy.doc',
    'numpy.conftest',
    # ============================================================================
    # CUDA & GPU LIBRARIES
    # ============================================================================
    'nvidia', 'nvidia.*',
    'cuda', 'cudart', 'cublas', 'curand', 'cusparse', 'cufft',
    'cutlass', 'nccl', 'nvtx', 'cudnn',
    
    # ============================================================================
    # OTHER ML FRAMEWORKS
    # ============================================================================
    'tensorflow', 'tensorflow.*',
    'tensorflow_hub', 'tensorboard',
    'keras', 'keras.*',
    'jax', 'jax.*', 'flax', 'flax.*',
    'xformers', 'triton',
    
    # ============================================================================
    # DEVELOPMENT & TESTING TOOLS
    # ============================================================================
    'pytest', 'nose', 'unittest', 'doctest', 'test', 'tests',
    'IPython', 'jupyter', 'notebook', 'ipykernel', 'ipywidgets',
    'pylint', 'black', 'flake8', 'mypy', 'coverage',
    'sphinx', 'docutils',
    
    # ============================================================================
    # ALTERNATIVE GUI FRAMEWORKS
    # ============================================================================
    'PyQt5', 'PyQt6', 'PySide2',
    'wx', 'kivy', 'pygame',
    
    # ============================================================================
    # TKINTER (No longer used)
    # ============================================================================
    'tkinter', 'tkinter.*', '_tkinter',
    
    # ============================================================================
    # WEB FRAMEWORKS
    # ============================================================================
    'tornado', 'flask', 'django', 'fastapi', 'uvicorn',
    
    # ============================================================================
    # OPTIONAL/RARELY USED PACKAGES
    # ============================================================================
    'dask', 'dask.*',
    'cupy', 'sparse',
    'colorama',  # Unless you need colored console output
    'win32com', 'pythoncom',  # Unless you need Windows COM
    
    # ============================================================================
    # ADDITIONAL HEAVY PACKAGES FROM YOUR ENVIRONMENT
    # ============================================================================
    'modelscope', 'modelscope.*',
    # Argos Translate (exclude from lite)
    'argostranslate', 'argostranslate.*',
    'aistudio-sdk',
    'bce-python-sdk',
    'briefcase',
    'cookiecutter',
    'fugashi', 'unidic-lite', 'jaconv',  # Japanese text processing
    'python-bidi',  # BiDi text
	
		
	# MORE AGGRESSIVE EXCLUDES
	'bitsandbytes', 'bitsandbytes.*',
	'polars', 'polars.*',
	'pyarrow', 'pyarrow.*',
    'scipy', 'scipy.*',
    'scipy.libs'
	
	# Force exclude ALL torch variants
	'*torch*',
	'torch*',
	'_torch*',
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
    hookspath=['.'],  # Use custom hooks in current directory (bypasses PySide6.QtNetwork SSL check)
    hooksconfig={},
    runtime_hooks=['pyi_rth_win32_runtime.py'],  # Fix R6034 runtime conflicts
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Simple cleanup after Analysis + Fix R6034 runtime conflicts
a.binaries = [b for b in a.binaries if not any([
    b[0].endswith('opencv_videoio_ffmpeg490_64.dll'),
    'scipy.libs' in b[0],
    '_avif' in b[0],
    'pypdfium' in b[0],
    'metrics_cpp_avx2' in b[0],
    'hdf5.dll' in b[0],
    'libsndfile' in b[0],
    'geos-' in b[0],
    # Remove conflicting MSVC runtime DLLs (keep only system ones)
    b[0].lower() in ['msvcr90.dll', 'msvcp90.dll', 'msvcr100.dll', 'msvcp100.dll',
                     'msvcr110.dll', 'msvcp110.dll', 'msvcr120.dll', 'msvcp120.dll'],
])]

# Remove torch only
a.datas = [d for d in a.datas if not any([
    'torch' in d[0].lower(),
    'torch-' in d[0],
    '.dist-info' in d[0] and 'torch' in d[0].lower(),
])]

a.pure = [p for p in a.pure if not any([
    'torch' in str(p).lower(),
    'pytorch' in str(p).lower(),
    '_torchcodec' in str(p),
])]

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
            'ucrtbase.dll',      # Don't compress Universal CRT
            'msvcp*.dll',        # Don't compress MSVC runtime
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
            'ucrtbase.dll',
            'msvcp*.dll',
        ],
        name=APP_NAME.replace(' ', '_'),
    )

# ============================================================================
# NOTES
# ============================================================================

"""
Build Instructions:
1. Install PyInstaller: pip install pyinstaller
2. Install all dependencies: pip install -r requirements.txt
3. Run: pyinstaller translator.spec

Optimization Tips:
- Set ENABLE_UPX = True for smaller file size (but slower startup)
- Set ONE_FILE = False for faster startup but folder distribution
- Set ENABLE_CONSOLE = True for debugging

This build includes:
- Datasketch for enhanced QA scanning performance
- AI Hunter Enhanced for improved duplicate detection
- Complete API client support (Google, OpenAI, Anthropic)
- DeepL and Google Translate for traditional translation APIs 
- Full text processing and analysis capabilities
- Manga text detection with Google Cloud Vision OCR support
- Manga text translation with API key
- OpenCV for advanced image processing
- Auto-update functionality with GitHub release checking

The executable will be ~160MB due to included ML libraries and OpenCV.

For version information:
Create a version_info.txt file with Windows version resource information

Note: Warnings about missing 'dask', 'torch', 'cupy', etc. are expected
and safe to ignore. These are optional scipy dependencies.

For manga translation:
- Google Cloud Vision API credentials required (JSON file)
- OpenCV (cv2) included for image processing
- Supports manga panel text detection and translation

For auto-update:
- Checks GitHub releases for new versions
- Downloads updates directly from GitHub
- Configurable auto-check on startup
"""
