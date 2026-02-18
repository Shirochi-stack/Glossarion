# -*- mode: python ; coding: utf-8 -*-
"""
Glossarion v7.5.6 - PyInstaller Specification File
Enhanced Translation Tool with QA Scanner, AI Hunter, and Manga Translation
"""

import sys
import os

# Fix DLL search path for WeasyPrint during build
# Check GTK_FOLDER env var first (set by CI), then fallback to common locations
gtk_folder = os.environ.get('GTK_FOLDER', '')
msys2_paths = [
    os.path.join(gtk_folder, 'bin') if gtk_folder else '',
    r'C:\msys64\mingw64\bin',
    r'D:\a\_temp\msys64\mingw64\bin',
]
for msys_path in msys2_paths:
    if msys_path and os.path.exists(msys_path):
        os.environ['PATH'] = msys_path + os.pathsep + os.environ.get('PATH', '')
        print(f"  Added {msys_path} to PATH for WeasyPrint")
        break

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_NAME = 'N_Glossarion_NoCuda v7.5.6'  # CHANGED: Updated version
APP_ICON = 'Halgakos.ico'
ENABLE_CONSOLE = True  # Console disabled for production
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
binaries.extend([
    ('libgcc_s_seh-1.dll', '.'),
    ('libonnx_inpainter.dll', '.'),
    ('onnx_inpainter.dll', '.'),
    ('libstdc++-6.dll', '.'),
    ('libwinpthread-1.dll', '.'),
    ('onnxruntime.dll', '.'),
    ('onnxruntime_providers_shared.dll', '.'),
    ('onnx_inpainter.cpp', '.'),
    ('vfcompat.dll', '.'),
    ('appverifUI.dll', '.')
])

# Add MSYS2 DLLs for WeasyPrint (PDF generation with formatting and images)
import glob

# Find MSYS2 bin directory - check env var first, then common locations
gtk_folder = os.environ.get('GTK_FOLDER', '')
msys2_bin_candidates = [
    os.path.join(gtk_folder, 'bin') if gtk_folder else '',
    r'C:\msys64\mingw64\bin',
    r'D:\a\_temp\msys64\mingw64\bin',
]

msys2_bin = None
for candidate in msys2_bin_candidates:
    if candidate and os.path.exists(candidate):
        msys2_bin = candidate
        break

print(f"  GTK_FOLDER env var: {gtk_folder}")
print(f"  Checking candidates: {msys2_bin_candidates}")
print(f"  Selected MSYS2 bin: {msys2_bin}")

if msys2_bin and os.path.exists(msys2_bin):
    dll_list = glob.glob(os.path.join(msys2_bin, '*.dll'))
    print(f"  Found {len(dll_list)} DLL files in {msys2_bin}")
    if dll_list:
        # Print first few DLLs for verification
        print(f"  Sample DLLs: {[os.path.basename(d) for d in dll_list[:5]]}")
        for dll in dll_list:
            binaries.append((dll, '.'))
        print(f"  Added {len(dll_list)} MSYS2 DLLs for WeasyPrint")
    else:
        print(f"  WARNING: No DLLs found in {msys2_bin}")
else:
    print(f"  WARNING: No MSYS2 directory found in any candidate location")

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
# Add icons and images to data
datas.append(('Halgakos.ico', '.'))
datas.append(('Halgakos_NoChibi.png', '.'))
datas.append(('WhereIsMyOutput.png', '.'))

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
    
    # Resources
	
	('enhanced_text_extractor.py', '.'),	
	('pdf_extractor.py', '.'),
	
	('multi_api_key_manager.py', '.'),
	('individual_endpoint_dialog.py', '.'),
	('bubble_detector.py', '.'),

	('local_inpainter.py', '.'),	
	
	('ocr_manager.py', '.'),
	('model_options.py', '.'),
	('hyphen_textwrap.py', '.'),
	
	# Image Rendering
	('ImageRenderer.py', '.'),
]
# Add application files to datas
datas.extend(app_files)
datas.append(('memory_usage_reporter.py', '.'))
datas.append(('tqdm_safety.py', '.'))
datas.append(('debug_env_vars.py', '.'))
datas.append(('enable_debug_mode.py', '.'))

# MAT Inpainting Support - Add MAT architecture directories
from PyInstaller.utils.hooks import collect_all
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
# ADD WINDOWS RUNTIME DEPENDENCIES
# ============================================================================

# Add Windows Visual C++ Runtime DLLs that PyTorch needs
import platform
if platform.system() == 'Windows':
    # Find and add Visual C++ runtime DLLs
    import ctypes.util
    
    # List of runtime DLLs that PyTorch might need
    runtime_dlls = [
        'msvcp140.dll',
        'msvcp140_1.dll',
        'msvcp140_2.dll',
        'vcruntime140.dll',
        'vcruntime140_1.dll',
        'vcomp140.dll',
        'concrt140.dll',
        'api-ms-win-crt-runtime-l1-1-0.dll',
        'api-ms-win-crt-heap-l1-1-0.dll',
        'api-ms-win-crt-math-l1-1-0.dll',
        'api-ms-win-crt-stdio-l1-1-0.dll',
        'api-ms-win-crt-locale-l1-1-0.dll',
        'api-ms-win-crt-string-l1-1-0.dll',
        'api-ms-win-crt-time-l1-1-0.dll',
        'api-ms-win-crt-convert-l1-1-0.dll',
        'api-ms-win-crt-environment-l1-1-0.dll',
        'api-ms-win-crt-process-l1-1-0.dll',
        'api-ms-win-crt-filesystem-l1-1-0.dll',
        'api-ms-win-crt-utility-l1-1-0.dll',
        'api-ms-win-core-console-l1-1-0.dll',
        'api-ms-win-core-datetime-l1-1-0.dll',
        'api-ms-win-core-debug-l1-1-0.dll',
        'api-ms-win-core-errorhandling-l1-1-0.dll',
        'api-ms-win-core-file-l1-1-0.dll',
        'api-ms-win-core-file-l1-2-0.dll',
        'api-ms-win-core-file-l2-1-0.dll',
        'api-ms-win-core-handle-l1-1-0.dll',
        'api-ms-win-core-heap-l1-1-0.dll',
        'api-ms-win-core-interlocked-l1-1-0.dll',
        'api-ms-win-core-libraryloader-l1-1-0.dll',
        'api-ms-win-core-localization-l1-2-0.dll',
        'api-ms-win-core-memory-l1-1-0.dll',
        'api-ms-win-core-namedpipe-l1-1-0.dll',
        'api-ms-win-core-processenvironment-l1-1-0.dll',
        'api-ms-win-core-processthreads-l1-1-0.dll',
        'api-ms-win-core-processthreads-l1-1-1.dll',
        'api-ms-win-core-profile-l1-1-0.dll',
        'api-ms-win-core-rtlsupport-l1-1-0.dll',
        'api-ms-win-core-string-l1-1-0.dll',
        'api-ms-win-core-synch-l1-1-0.dll',
        'api-ms-win-core-synch-l1-2-0.dll',
        'api-ms-win-core-sysinfo-l1-1-0.dll',
        'api-ms-win-core-timezone-l1-1-0.dll',
        'api-ms-win-core-util-l1-1-0.dll',
        'ucrtbase.dll',
    ]
    
    for dll_name in runtime_dlls:
        dll_path = ctypes.util.find_library(dll_name)
        if dll_path and os.path.exists(dll_path):
            # Check if not already in binaries
            if not any(b[0] == dll_path or os.path.basename(b[0]) == dll_name for b in binaries):
                binaries.append((dll_path, '.'))
                print(f"  Added runtime DLL: {dll_name}")
    
    # Also try to find Intel MKL and OpenMP libraries if they exist
    mkl_dlls = ['mkl_core.dll', 'mkl_intel_thread.dll', 'mkl_rt.dll', 'libiomp5md.dll', 'libomp140.x86_64.dll']
    
    # CRITICAL: Find and add dependencies for shm.dll
    # Search for OpenMP and other runtime libraries in multiple locations
    search_paths = [
        os.path.join(os.environ.get('CONDA_PREFIX', ''), 'Library', 'bin') if os.environ.get('CONDA_PREFIX') else None,
        os.path.join(sys.prefix, 'Library', 'bin'),
        os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib'),
        os.path.dirname(sys.executable),  # Python installation directory
        os.path.join(os.path.dirname(sys.executable), 'DLLs'),
        os.path.join(os.path.dirname(sys.executable), 'Library', 'bin'),
    ]
    
    # Add PATH directories
    if 'PATH' in os.environ:
        search_paths.extend(os.environ['PATH'].split(';'))
    
    for dll_name in mkl_dlls:
        found = False
        for search_path in search_paths:
            if not search_path:
                continue
            if isinstance(search_path, list):
                for p in search_path:
                    dll_path = os.path.join(p, dll_name)
                    if os.path.exists(dll_path):
                        binaries.append((dll_path, '.'))
                        print(f"  Added MKL/OpenMP DLL: {dll_name}")
                        found = True
                        break
            else:
                dll_path = os.path.join(search_path, dll_name)
                if os.path.exists(dll_path):
                    binaries.append((dll_path, '.'))
                    print(f"  Added MKL/OpenMP DLL: {dll_name}")
                    found = True
                    break
            if found:
                break
    
    # CRITICAL FIX: Collect ALL DLLs from torch/lib directory
    # This ensures shm.dll, c10.dll, and all their dependencies are included
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        
        if os.path.exists(torch_lib):
            print(f"\n  Collecting ALL torch DLLs from: {torch_lib}")
            dll_count = 0
            for filename in os.listdir(torch_lib):
                if filename.lower().endswith('.dll'):
                    full_path = os.path.join(torch_lib, filename)
                    # Check if not already added
                    if not any(b[0] == full_path for b in binaries):
                        binaries.append((full_path, 'torch/lib'))
                        dll_count += 1
                        print(f"    + {filename}")
            print(f"  Total torch DLLs added: {dll_count}\n")
    except Exception as e:
        print(f"  Warning: Could not collect torch DLLs: {e}")
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
	'enhanced_text_extractor.py',
	'pdf_extractor',
	'multi_api_key_manager.py',
	'individual_endpoint_dialog.py',
	'bubble_detector', 
	'local_inpainter',  	
	'ocr_manager',
	'model_options',
	'hyphen_textwrap',
	'ImageRenderer',
	
	# MAT Inpainting Support
	'torch_utils',
	'dnnlib',
	'dnnlib.util',
	'networks',
	'networks.mat',
	
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

excludes = [
    
    # CUDA-specific ONNX
    'onnxruntime-gpu',
    'onnxruntime_gpu',
	'scipy',
    
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

# Create hooks directory path
import os
hooks_dir = os.path.join(os.getcwd(), 'hooks')
if not os.path.exists(hooks_dir):
    os.makedirs(hooks_dir)
    print(f"Created hooks directory: {hooks_dir}")

a = Analysis(
    ['translator_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['.'],  # Custom hooks: R6034 fix + torch hooks
    hooksconfig={},
    runtime_hooks=['pyi_rth_win32_runtime.py'],  # Fix R6034 + torch env
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ============================================================================
# SURGICAL GPU/CUDA BINARY REMOVAL
# ============================================================================

# GPU/CUDA specific files to remove (keeping CPU versions)
GPU_CUDA_DLLS = [
    # CUDA Runtime & Libraries
    'cudart64_',
    'cudart32_',
    'cudnn64_',
    'cudnn32_',
    'cudnn_ops',
    'cudnn_adv',
    'cudnn_cnn',
    'cudnn_engines',
    'cudnn_heuristic',
    'cublas64_',
    'cublasLt64_',
    'cufft64_',
    'curand64_',
    'cusolver64_',
    'cusolverMg64_',
    'cusparse64_',
    'nvrtc64_',
    'nvrtc-builtins64_',
    'nvJitLink_',
    'nccl64_',
    'nvToolsExt64_',
    
    # PyTorch CUDA specific
    'torch_cuda.dll',
    'torch_cuda_cu.dll',
    'torch_cuda_cpp.dll',
    'c10_cuda.dll',
    'caffe2_nvrtx.dll',
    
    # ONNX GPU providers
    'onnxruntime_providers_cuda.dll',
    'onnxruntime_providers_tensorrt.dll',
    'onnxruntime_providers_dml.dll',
    
    # Paddle GPU
    'paddle_cuda',
    'libpaddle_cuda',
    
    # Keep torch_cpu.dll! It's needed for CPU operations
]

# Additional patterns for files that are definitely GPU-only
GPU_PATTERNS = [
    'cuda',
    'cudnn',
    'cublas',
    'cufft',
    'curand',
    'cusolver',
    'cusparse',
    'nvrtc',
    'nvjitlink',
    'nccl',
    'nvtx',
    'nvtools',
]

def is_gpu_related(filepath):
    """Check if file is GPU/CUDA related"""
    filename = os.path.basename(filepath).lower()
    
    # Check specific DLL names
    for dll_pattern in GPU_CUDA_DLLS:
        if dll_pattern.lower() in filename:
            return True
    
    # Check if it's a CUDA library (but not torch_cpu!)
    if 'cuda' in filename and 'torch_cpu' not in filename:
        return True
    
    # Check patterns (but preserve CPU versions)
    if filename.endswith('.dll') or filename.endswith('.pyd'):
        # Don't remove torch_cpu, torch_python, or other CPU components
        if 'torch_cpu' in filename or 'torch_python' in filename:
            return False
        
        # Check GPU patterns
        for pattern in GPU_PATTERNS:
            if pattern in filename and 'cpu' not in filename:
                return True
    
    return False

# Clean binaries - ONLY GPU stuff
print("\n" + "="*60)
print("REMOVING GPU/CUDA BINARIES ONLY...")
print("="*60)
original_count = len(a.binaries)
cleaned_binaries = []
removed_size = 0

for binary in a.binaries:
    binary_path = binary[0]
    if is_gpu_related(binary_path):
        print(f"  - Removing GPU/CUDA: {os.path.basename(binary_path)}")
        # Try to estimate size if possible
        try:
            if os.path.exists(binary[1]):
                size_mb = os.path.getsize(binary[1]) / (1024*1024)
                removed_size += size_mb
                print(f"    Size: {size_mb:.1f} MB")
        except:
            pass
    else:
        cleaned_binaries.append(binary)

a.binaries = cleaned_binaries
print(f"\nBinaries: {original_count} -> {len(a.binaries)} (removed {original_count - len(a.binaries)})")
print(f"Estimated space saved: {removed_size:.1f} MB")

# Clean data files - remove CUDA related data
print("\n" + "="*60)
print("REMOVING GPU/CUDA DATA FILES...")
print("="*60)
original_count = len(a.datas)
cleaned_datas = []

for data in a.datas:
    data_path = data[0]
    # Only remove CUDA specific data files
    if any(x in data_path.lower() for x in ['cuda', 'cudnn', 'nvidia-cuda', 'nvidia-ml']):
        if 'cpu' not in data_path.lower():  # Keep CPU variants
            print(f"  - Removing: {data_path}")
            continue
    cleaned_datas.append(data)

a.datas = cleaned_datas
print(f"\nData files: {original_count} -> {len(a.datas)} (removed {original_count - len(a.datas)})")

# DON'T remove torch or transformers from pure modules - we want to keep them!
print("\n" + "="*60)
print("KEEPING CPU ML MODULES (torch, transformers, etc.)...")
print("="*60)
print("  + torch CPU modules preserved")
print("  + Transformers preserved")
print("  + Ultralytics preserved")
print("  + ONNX Runtime CPU preserved")
print("  + All other ML libraries preserved")

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
            'vcruntime140.dll',
            'python*.dll',
            'api-ms-win-*.dll',
            'ucrtbase.dll',
            'msvcp*.dll',
            'torch_cpu.dll',  # Don't compress torch CPU
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
            'torch_cpu.dll',
        ],
        name=APP_NAME.replace(' ', '_'),
    )

# ============================================================================
# BUILD NOTES
# ============================================================================

"""
GPU/CUDA-FREE BUILD - CPU ML FUNCTIONALITY PRESERVED

This spec file removes ONLY GPU/CUDA components while keeping:
+ PyTorch CPU (torch_cpu.dll)
+ Transformers library
+ ONNX Runtime CPU
+ Ultralytics YOLO
+ All text processing
+ All API clients

REMOVED (GPU/CUDA only):
- torch_cuda.dll (884MB)
- cudnn_* libraries (870MB+)
- cublasLt64_12.dll (513MB)
- cusparse64_12.dll (250MB)
- CUDA runtime libraries
- onnxruntime_providers_cuda.dll (306MB)

KEPT (CPU ML):
+ torch_cpu.dll (237MB) - REQUIRED for PyTorch CPU
+ transformers library
+ ONNX Runtime CPU
+ All model inference capabilities

Expected size: ~3.9GB -> ~600-800MB
All ML features work on CPU!

Build command: pyinstaller translator.spec
"""
