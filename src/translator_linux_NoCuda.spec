# -*- mode: python ; coding: utf-8 -*-
"""
Glossarion NoCuda - PyInstaller Specification File (Linux)
Enhanced Translation Tool with QA Scanner, AI Hunter, and Manga Translation
"""

import sys
import os

SPEC_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if SPEC_DIR not in sys.path:
    sys.path.insert(0, SPEC_DIR)
from app_version import get_spec_app_name

from PyInstaller.utils.hooks import (
    collect_all,
    collect_submodules,
    collect_dynamic_libs,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_NAME = get_spec_app_name("translator_linux_NoCuda.spec")
ENABLE_CONSOLE = True  # Retain diagnostic output for the Linux executable
ENABLE_UPX = False      # Compression (smaller file size but slower startup)

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

# PyInstaller's torch hook resolves dependencies of these native Linux libraries.
# Install CPU-only torch/torchvision wheels before building this spec.
binaries.extend(collect_dynamic_libs('torch'))

# Collect data files from packages that need them. Keep ONNX out of
# collect_all(); it pulls the backend test suite into hidden imports and makes
# local NoCuda builds crawl near the end of Analysis.
for package in ['langdetect', 'certifi', 'tiktoken_ext', 'ttkbootstrap', 'chardet', 'charset_normalizer', 'rapidocr_onnxruntime', 'onnxruntime']:
    try:
        data, bins, hidden = collect_all(package)
        datas.extend(data)
        binaries.extend(bins)
        hiddenimports.extend(hidden)
    except:
        pass

# Include lazy CPU inference imports used by manga OCR, detection and inpainting.
# In particular, keep torch.cuda's Python API: CPU builds use is_available().
hiddenimports.extend([
    'torch',
    'torchvision',
    'transformers',
    'diffusers',
    'accelerate',
    'safetensors',
    'tokenizers',
    'huggingface_hub',
    'ultralytics',
    'easyocr',
    'manga_ocr',
    'onnx',
    'onnx.checker',
    'onnx.helper',
    'onnx.numpy_helper',
    'onnx.shape_inference',
])

# Google AI protocol modules are imported dynamically by the gRPC Gemini client.
hiddenimports.extend(collect_submodules('google.ai.generativelanguage_v1beta'))

# ============================================================================
# APPLICATION FILES
# ============================================================================

# Main application files
# Add icons and images to data
datas.append(('Halgakos.ico', '.'))
datas.append(('Halgakos_NoChibi.png', '.'))
datas.append(('WhereIsMyOutput.png', '.'))

app_files = [
    ('manga_ocr_io.py', '.'),
    ('gemini_policy.py', '.'),
    ('epub_package.py', '.'),
    ('epub_special_files.py', '.'),
    ('gender_tracking.py', '.'),
    ('title_tag_translation.py', '.'),
    ('chapter_chunk_progress.py', '.'),
    ('chapter_display_numbering.py', '.'),
    # Core GUI
    ('translator_gui.py', '.'),
    ('parallel_epub_glossary.py', '.'),
    ('metadata_translation_worker.py', '.'),
    ('language_options.py', '.'),
    ('metadata_progress.py', '.'),
    ('translation_artifacts.py', '.'),
    ('splash_utils.py', '.'),
    ('dpi_setup.py', '.'),
    ('other_settings.py', '.'),
    ('GlossaryManager.py', '.'),
    ('GlossaryManager_GUI.py', '.'),
    ('glossary_paths.py', '.'),
    ('Retranslation_GUI.py', '.'),
    ('QA_Scanner_GUI.py', '.'),
    ('Chapter_Extractor.py', '.'),
    ('PatternManager.py', '.'),

    # Translation modules
    ('TransateKRtoEN.py', '.'),
    ('subtitle_processor.py', '.'),
    ('refinement_prompts.py', '.'),
    ('unified_api_client.py', '.'),
    ('google_free_translate.py', '.'),
    ('vision_ocr_source_epub.py', '.'),

    # File processors
    ('epub_converter.py', '.'),
    ('image_archive_epub.py', '.'),
    ('html_archive_epub.py', '.'),
    ('html_tag_entities.py', '.'),
    ('emoticon_patterns.py', '.'),
    ('qa_scan_runtime.py', '.'),
    ('txt_processor.py', '.'),
    ('chapter_splitter.py', '.'),

    # Glossary extractors
    ('extract_glossary_from_epub.py', '.'),
    ('glossary_usage.py', '.'),
    ('glossary_refinement.py', '.'),
    ('extract_glossary_from_txt.py', '.'),
    ('glossary_process_worker.py', '.'),  # Glossary subprocess worker
    ('chapter_extraction_worker.py', '.'),  # Chapter extraction subprocess worker
    ('sdlxliff_extraction_worker.py', '.'),
    ('sdlxliff_extraction_manager.py', '.'),
    ('sdlxliff_extractor.py', '.'),
    ('sdlxliff_converter.py', '.'),
    ('sdlxliff_sidecar_writer.py', '.'),
    ('md_txt_sidecar_writer.py', '.'),  # MD/TXT sidecar writer (html2text)
    ('_compress_worker.py', '.'),  # Lightweight image compression worker
    ('_empty_attr_fix.py', '.'),  # Shared LLM Token Fix (empty-attr) helper
    ('html_duplicate_cleanup.py', '.'),
    ('_pdf_worker.py', '.'),  # PDF generation subprocess worker
    ('pdf_generation_manager.py', '.'),  # PDF generation manager
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

    # Manga Translation modules
    ('manga_translator.py', '.'),
    ('manga_integration.py', '.'),
    ('manga_settings_dialog.py', '.'),
    ('manga_image_preview.py', '.'),

    # Dialog animations
    ('dialog_animations.py', '.'),

    # Spinning icon helper
    ('spinning.py', '.'),

    # Rotatable label widget for animations
    ('rotatable_label.py', '.'),

    # Update Manager
    ('update_manager.py', '.'),

    # Async Processing
    ('async_api_processor.py', '.'),

    # Metadata and header batch translation
    ('metadata_batch_translator.py', '.'),
    ('translate_headers_standalone.py', '.'),
    ('review_dialog.py', '.'),
    ('review_generator.py', '.'),

    # Resources

    ('enhanced_text_extractor.py', '.'),
    ('pdf_extractor.py', '.'),
    ('_pdf_extraction_worker.py', '.'),
    ('pdf_extraction_manager.py', '.'),
    ('pdf_bookmarks.py', '.'),
    ('output_workspace.py', '.'),
    ('pdf_fast_extractor.py', '.'),
    ('pdf_workspace_compiler.py', '.'),
    ('workspace_reader.py', '.'),
    ('reader_overlay.py', '.'),
    ('pdf_output_naming.py', '.'),
    ('installer_utils.py', '.'),

    ('multi_api_key_manager.py', '.'),
    ('individual_endpoint_dialog.py', '.'),
    ('bubble_detector.py', '.'),

    ('local_inpainter.py', '.'),

    ('ocr_manager.py', '.'),
    ('model_options.py', '.'),
    ('hyphen_textwrap.py', '.'),

    # Image Rendering
    ('ImageRenderer.py', '.'),

    # Environment variable size limit workaround
    ('large_env.py', '.'),

    # AuthGPT - ChatGPT subscription OAuth
    ('authgpt_auth.py', '.'),
    ('reasoning_compatibility.py', '.'),
    ('authgrok_auth.py', '.'),  # xAI Grok subscription OAuth
    ('authgem_auth.py', '.'),
    ('authcd_auth.py', '.'),  # Claude subscription OAuth
    ('glm_proxy.py', '.'),
    ('authnd_auth.py', '.'),  # NVIDIA Build browser-backed auth
    ('gemini_free.py', '.'),  # Google Search/Gemini browser-backed route
    ('token_encryption.py', '.'),
    ('proxy_token_storage.py', '.'),

    # Antigravity Cloud Code proxy
    ('ocagy_cli.py', '.'),  # OpenCode + opencode-antigravity-auth
    ('autharena_proxy.py', '.'),
    ('antigravity_proxy.py', '.'),

    # gRPC Gemini client
    ('grpc_gemini_client.py', '.'),

    # EPUB Library & Reader
    ('epub_library.py', '.'),

    # RPG Maker handler
    ('rpgmaker_handler.py', '.'),
]
# Add application files to datas
datas.extend(app_files)
datas.append(('memory_usage_reporter.py', '.'))
datas.append(('tqdm_safety.py', '.'))
datas.append(('debug_env_vars.py', '.'))
datas.append(('enable_debug_mode.py', '.'))

# MAT Inpainting Support - Add MAT architecture directories
try:
    # Collect MAT model architecture directories
    from pathlib import Path
    mat_dirs = ['torch_utils', 'dnnlib', 'networks']
    for mat_dir in mat_dirs:
        mat_path = Path(mat_dir)
        if mat_path.exists() and mat_path.is_dir():
            datas.append((str(mat_path), mat_dir))
            print(f"  Added MAT directory: {mat_dir}")
except Exception as e:
    print(f"  Warning: Could not add MAT directories: {e}")

# ============================================================================
# HIDDEN IMPORTS (Organized by category)
# ============================================================================

# Application modules
app_modules = [
    'manga_ocr_io',
    'gemini_policy',
    'epub_package',
    'epub_special_files',
    'gender_tracking',
    'title_tag_translation',
    'chapter_chunk_progress',
    'chapter_display_numbering',
    'TransateKRtoEN',
    'metadata_translation_worker',
    'subtitle_processor',
    'language_options',
    'metadata_progress',
    'translation_artifacts',
    'refinement_prompts',
    'extract_glossary_from_epub',
    'parallel_epub_glossary',
    'glossary_usage',
    'glossary_refinement',
    'extract_glossary_from_txt',
    'glossary_process_worker',  # Glossary subprocess worker
    'chapter_extraction_worker',  # Chapter extraction subprocess worker
    'sdlxliff_extraction_worker',
    'sdlxliff_extraction_manager',
    'sdlxliff_extractor',
    'sdlxliff_converter',
    'sdlxliff_sidecar_writer',
    'md_txt_sidecar_writer',  # MD/TXT sidecar writer (html2text)
    '_compress_worker',  # Lightweight image compression worker
    '_empty_attr_fix',  # Shared LLM Token Fix (empty-attr) helper
    'html_duplicate_cleanup',
    '_pdf_worker',  # PDF generation subprocess worker
    'pdf_generation_manager',  # PDF generation manager
    'chapter_extraction_manager',  # Chapter extraction manager
    'GlossaryManager',
    'GlossaryManager_GUI',
    'glossary_paths',
    'Retranslation_GUI',
    'QA_Scanner_GUI',
    'Chapter_Extractor',
    'PatternManager',
    'epub_converter',
    'image_archive_epub',
    'html_archive_epub',
    'html_tag_entities',
    'emoticon_patterns',
    'qa_scan_runtime',
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
    'dpi_setup',
    'other_settings',      # Other Settings module
    'ai_hunter_enhanced',  # AI Hunter Enhanced module
    'manga_translator',    # Manga translator module
    'manga_integration',   # Manga GUI integration
    'manga_settings_dialog',
    'manga_image_preview', # Manga image preview widget
    'dialog_animations',   # Dialog fade animations
    'spinning',            # Spinning icon helper
    'rotatable_label',     # Rotatable label widget
    'update_manager',
    'api_key_encryption',
    'http_logger',
    'shutdown_utils',
    'async_api_processor',
    'metadata_batch_translator',
    'translate_headers_standalone',
    'review_dialog',
    'review_generator',
    'enhanced_text_extractor',
    'pdf_extractor',
    '_pdf_extraction_worker',
    'pdf_extraction_manager',
    'pdf_bookmarks',
    'output_workspace',
    'pdf_fast_extractor',
    'pdf_workspace_compiler',
    'workspace_reader',
    'reader_overlay',
    'pdf_output_naming',
    'installer_utils',
    'multi_api_key_manager',
    'individual_endpoint_dialog',
    'bubble_detector',
    'local_inpainter',
    'ocr_manager',
    'model_options',
    'hyphen_textwrap',
    'ImageRenderer',
    'large_env',
    'authgpt_auth',  # ChatGPT subscription OAuth
    'reasoning_compatibility',
    'authgrok_auth',  # xAI Grok subscription OAuth
    'authgem_auth',  # Gemini subscription OAuth
    'authcd_auth',  # Claude subscription OAuth
    'glm_proxy',  # Z.AI Coding Plan login proxy
    'authnd_auth',  # NVIDIA Build browser-backed auth
    'gemini_free',  # Google Search/Gemini browser-backed route
    'token_encryption',  # Encrypted token storage
    'proxy_token_storage',  # Encrypted proxy token storage
    'ocagy_cli',  # OpenCode + opencode-antigravity-auth
    'autharena_proxy',
    'antigravity_proxy',  # Antigravity Cloud Code proxy
    'grpc_gemini_client',  # gRPC Gemini client
    'epub_library',  # EPUB Library & Reader
    'rpgmaker_handler',  # RPG Maker game file handler

    # MAT Inpainting Support
    'torch_utils',
    'dnnlib',
    'dnnlib.util',
    'networks',
    'networks.mat',

]

# GUI Framework
gui_modules = [
    # Trigger PyInstaller's Qt6 hooks for the embedded browser and EPUB reader.
    'PySide6.QtWebEngineCore',
    'PySide6.QtWebEngineWidgets',
    'PySide6.QtWebChannel',
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
    'cv2',  # OpenCV for manga processing
    'numpy',  # Required for OpenCV and image processing
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

    # Azure Computer Vision (for manga OCR) - New Image Analysis API
    'azure',
    'azure.ai',
    'azure.ai.vision',
    'azure.ai.vision.imageanalysis',
    'azure.ai.vision.imageanalysis.models',
    'azure.ai.vision.imageanalysis._client',
    'azure.ai.vision.imageanalysis._operations',
    'azure.ai.vision.imageanalysis._version',
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

    # Google Cloud Vision (for manga OCR)
    'google.cloud',
    'google.cloud.vision',
    'google.cloud.vision_v1',
    'google.cloud.vision_v1.types',
    'google.cloud.vision_v1.services',
    'google.cloud.vision_v1.services.image_annotator',

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
    # Preserve generated protocol modules required by the installed SDK.
    'google.cloud.aiplatform',
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
hiddenimports = sorted(set(hiddenimports))

excludes = [
    # POE / websocket-client (no longer needed; imports guarded by try/except)
    'poe_api_wrapper', 'poe_api_wrapper.*',
    'ballyregan', 'ballyregan.*',
    'websocket', 'websocket.*',

    # CUDA-specific distributions (retain CPU torch and its Python CUDA stubs).
    'nvidia', 'nvidia.*',

    # CUDA-specific ONNX
    'onnxruntime-gpu',
    'onnxruntime_gpu',

    # Optional GPU/training accelerators pulled in by transformers/diffusers
    # hooks from a developer's global site-packages. NoCuda keeps CPU torch,
    # transformers, diffusers, and ONNX Runtime, but should not analyze these.
    'bitsandbytes', 'bitsandbytes.*',
    'triton', 'triton.*',
    'xformers', 'xformers.*',
    'flash_attn', 'flash_attn.*',
    'flash_attn_2_cuda',
    'deepspeed', 'deepspeed.*',
    'cupy', 'cupy.*',
    'cupy_backends', 'cupy_backends.*',
    'jax', 'jax.*',
    'jaxlib', 'jaxlib.*',
    'flax', 'flax.*',
    'tensorflow', 'tensorflow.*',
    'tensorflow_hub', 'tensorflow_hub.*',
    'tensorboard', 'tensorboard.*',
    'keras', 'keras.*',

    # Paddle GPU
    'paddle.fluid.core_avx',
    'paddle.fluid.core_noavx',
    'paddlepaddle-gpu',

    # Development & Testing (optional, for size)
    'pytest', 'nose', 'doctest',
    'IPython', 'jupyter', 'notebook',
    'pylint', 'black', 'flake8', 'mypy',
    'sphinx', 'docutils',

    # Alternative GUIs
    'PyQt5', 'PyQt6', 'PySide2',
    'wx', 'kivy', 'pygame',

    # Tkinter (No longer used)
    'tkinter', 'tkinter.*', '_tkinter',

    # Web frameworks
    'tornado', 'flask', 'django', 'fastapi', 'uvicorn',
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
    hookspath=['.'],  # Shared QtNetwork hook plus PyInstaller's native ML hooks
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    cipher=block_cipher,
    noarchive=False,
)

# ============================================================================
# CUDA SHARED LIBRARY REMOVAL
# ============================================================================

# Linux wheels use .so/.so.N names. Match native library names only; substring
# filters over all paths would remove CPU torch's Python CUDA stubs and metadata.
GPU_LIBRARY_PREFIXES = (
    'libcuda.',
    'libcudart.',
    'libcudnn',
    'libcublas',
    'libcufft',
    'libcurand',
    'libcusolver',
    'libcusparse',
    'libnvrtc',
    'libnvjitlink',
    'libnvptxcompiler',
    'libnccl',
    'libnvtx',
    'libnvtoolsext',
    'libtorch_cuda',
    'libc10_cuda',
    'libonnxruntime_providers_cuda',
    'libonnxruntime_providers_tensorrt',
    'onnxruntime_providers_cuda',
    'onnxruntime_providers_tensorrt',
    'libnvinfer',
    'libnvonnxparser',
    'libpaddle_cuda',
)


def is_gpu_library(filepath):
    """Return whether a Linux shared library belongs to a CUDA/GPU backend."""
    filename = os.path.basename(filepath).lower()
    if not (filename.endswith('.so') or '.so.' in filename):
        return False
    return filename.startswith(GPU_LIBRARY_PREFIXES)


# CPU torch, torchvision, ONNX and their supporting shared libraries stay bundled.
# Apply the same narrow check to native libraries collected as package data.
a.binaries = [entry for entry in a.binaries if not is_gpu_library(entry[0])]
a.datas = [entry for entry in a.datas if not is_gpu_library(entry[0])]
# Keep all pure Python modules, including torch.cuda and Google Cloud Vision.

# ============================================================================
# PYZ (Python Zip archive)
# ============================================================================

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

# ============================================================================
# EXECUTABLE CONFIGURATION (Linux single file)
# ============================================================================

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
    runtime_tmpdir=None,
    console=ENABLE_CONSOLE,
    disable_windowed_traceback=False,
)
