import sys
import os

# Add the current directory to Python path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# When running as executable, modules might be in _MEIPASS
if hasattr(sys, '_MEIPASS'):
    meipass_dir = sys._MEIPASS
    if meipass_dir not in sys.path:
        sys.path.insert(0, meipass_dir)

# Now we can safely import our modules
try:
    from extract_glossary_from_epub import main as glossary_main
except ImportError as e:
    print(f"Failed to import glossary module: {e}")
    glossary_main = None

try:
    from TransateKRtoEN import main as translation_main
except ImportError as e:
    print(f"Failed to import translation module: {e}")
    translation_main = None

try:
    from epub_converter import fallback_compile_epub
except ImportError as e:
    print(f"Failed to import epub converter: {e}")
    fallback_compile_epub = None

try:
    from scan_html_folder import scan_html_folder
except ImportError as e:
    print(f"Failed to import scanner: {e}")
    scan_html_folder = None
