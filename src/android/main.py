# main.py
"""
Glossarion Android — Main entry point.
Kivy/KivyMD app with Material Design dark theme.
Compatible with KivyMD 1.2.0 (stable PyPI release).
"""

import os
import sys

# Add parent directory (src/) to Python path so we can import shared modules
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Add _backend/ to sys.path — contains shared translation engine + stubs
_backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_backend')
if os.path.isdir(_backend_dir) and _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# Pre-register Android stubs for modules that don't exist on Android
# (tiktoken, ebooklib, httpx, rapidfuzz, langdetect)
# This MUST run before any _backend module is imported, because e.g.
# TransateKRtoEN.py does `import tiktoken` at module level.
try:
    import importlib
    _stub_map = {
        'tiktoken': 'tiktoken_stub',
        'ebooklib': 'ebooklib_stub',
        'ebooklib.epub': 'ebooklib_stub',
        'httpx': 'httpx_stub',
        'rapidfuzz': 'rapidfuzz_stub',
        'rapidfuzz.fuzz': 'rapidfuzz_stub',
        'rapidfuzz.process': 'rapidfuzz_stub',
        'langdetect': 'langdetect_stub',
    }
    for _mod_name, _stub_name in _stub_map.items():
        if _mod_name not in sys.modules:
            try:
                importlib.import_module(_mod_name)
            except ImportError:
                try:
                    _stub = importlib.import_module(_stub_name)
                    sys.modules[_mod_name] = _stub
                except Exception:
                    pass
except Exception:
    pass  # stubs dir may not exist on desktop

# Kivy config must be set BEFORE importing kivy
os.environ['KIVY_LOG_LEVEL'] = 'info'

from kivy.config import Config
Config.set('graphics', 'multisamples', '0')

from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager, SlideTransition
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.utils import platform
from kivy.lang import Builder
from kivy.clock import Clock

# ── CJK font helper ──
def _get_cjk_font_paths():
    """Find a CJK-capable font that supports Korean/Japanese/Chinese.
    
    Prefers the bundled Noto Sans CJK SC font which covers ALL CJK scripts.
    Falls back to system fonts if the bundled font is not present.
    
    Returns (fn_regular, fn_bold) or (None, None).
    """
    import sys as _sys

    # 1. Bundled Noto Sans CJK SC (covers Chinese + Japanese + Korean + Latin)
    bundled = os.path.join(os.path.dirname(__file__), 'fonts', 'NotoSansCJKsc-Regular.otf')
    if os.path.isfile(bundled):
        return bundled, bundled  # same for bold (no bold variant bundled)

    # 2. System fonts as fallback
    if platform == 'android':
        pairs = [
            ('/system/fonts/NotoSansCJK-Regular.ttc', '/system/fonts/NotoSansCJK-Bold.ttc'),
            ('/system/fonts/NotoSansCJKsc-Regular.otf', '/system/fonts/NotoSansCJKsc-Bold.otf'),
            ('/system/fonts/DroidSansFallback.ttf', '/system/fonts/DroidSansFallback.ttf'),
        ]
    elif _sys.platform == 'win32':
        windir = os.environ.get('WINDIR', r'C:\Windows')
        fd = os.path.join(windir, 'Fonts')
        pairs = [
            # malgun covers Korean well
            (os.path.join(fd, 'malgun.ttf'), os.path.join(fd, 'malgunbd.ttf')),
            # msyh covers Chinese well
            (os.path.join(fd, 'msyh.ttc'), os.path.join(fd, 'msyhbd.ttc')),
            (os.path.join(fd, 'meiryo.ttc'), os.path.join(fd, 'meiryob.ttc')),
            (os.path.join(fd, 'YuGothR.ttc'), os.path.join(fd, 'YuGothB.ttc')),
        ]
    elif _sys.platform == 'darwin':
        pairs = [
            ('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc', '/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc'),
            ('/Library/Fonts/Arial Unicode.ttf', '/Library/Fonts/Arial Unicode.ttf'),
        ]
    else:
        pairs = []

    for fn_r, fn_b in pairs:
        if os.path.isfile(fn_r):
            return fn_r, fn_b if os.path.isfile(fn_b) else fn_r
    return None, None

# Import screens
from library_screen import LibraryScreen
from reader_screen import ReaderScreen
from multikey_screen import MultiKeyScreen
from translation_screen import TranslationScreen
from progress_screen import ProgressScreen

# Import utilities
from android_config import load_config, save_config
from android_file_utils import request_storage_permissions
from android_notification import create_notification_channels

# KV layout — KivyMD 1.2.0 compatible
KV = '''
#:import SlideTransition kivy.uix.screenmanager.SlideTransition

BoxLayout:
    orientation: 'vertical'

    ScreenManager:
        id: screen_manager
        transition: SlideTransition(duration=0.25)

        LibraryScreen:
            name: 'library'

        ReaderScreen:
            name: 'reader'

        MultiKeyScreen:
            name: 'multikey'

        ProgressScreen:
            name: 'progress'

        TranslationScreen:
            name: 'translation'

    # Custom bottom nav bar (MDBottomNavigation reserves its own content
    # panel area which clips the ScreenManager to ~50% height)
    BoxLayout:
        id: bottom_nav
        size_hint_y: None
        height: dp(56)
        padding: [dp(8), 0]

        canvas.before:
            Color:
                rgba: app.theme_cls.bg_dark
            Rectangle:
                pos: self.pos
                size: self.size
            # Top border line
            Color:
                rgba: 0.3, 0.3, 0.3, 0.5
            Rectangle:
                pos: self.x, self.top - 1
                size: self.width, 1

        MDIconButton:
            icon: "bookshelf"
            theme_text_color: "Custom"
            text_color: (1,1,1,1) if app.root and app.root.ids.screen_manager.current == 'library' else (0.5,0.5,0.5,1)
            pos_hint: {"center_y": 0.5}
            on_release: app.switch_screen('library')

        Widget:

        MDIconButton:
            icon: "key-chain-variant"
            theme_text_color: "Custom"
            text_color: (1,1,1,1) if app.root and app.root.ids.screen_manager.current == 'multikey' else (0.5,0.5,0.5,1)
            pos_hint: {"center_y": 0.5}
            on_release: app.switch_screen('multikey')

        Widget:

        MDIconButton:
            icon: "chart-bar-stacked"
            theme_text_color: "Custom"
            text_color: (1,1,1,1) if app.root and app.root.ids.screen_manager.current == 'progress' else (0.5,0.5,0.5,1)
            pos_hint: {"center_y": 0.5}
            on_release: app.switch_screen('progress')

        Widget:

        MDIconButton:
            icon: "translate"
            theme_text_color: "Custom"
            text_color: (1,1,1,1) if app.root and app.root.ids.screen_manager.current == 'translation' else (0.5,0.5,0.5,1)
            pos_hint: {"center_y": 0.5}
            on_release: app.switch_screen('translation')
'''


class GlossarionApp(MDApp):
    """Main application class for Glossarion Android."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config_data = {}
        self.title = 'Glossarion'

    def build(self):
        # Set dark theme with more vibrant premium colors
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "DeepPurple"
        self.theme_cls.accent_palette = "Teal"
        self.theme_cls.primary_hue = "500"
        self.theme_cls.material_style = "M3"

        # Set window background for desktop testing
        if platform != 'android':
            Window.size = (412, 915)  # ~6" phone
            Window.clearcolor = (0.07, 0.07, 0.09, 1)

        # Load config
        self.config_data = load_config()

        # Build UI
        root = Builder.load_string(KV)
        return root

    def _register_cjk_font(self):
        """Register a CJK-capable font to replace Kivy's built-in Roboto.
        
        Must be called AFTER KivyMD sets up its theme, because KivyMD's
        _run_prepare() re-registers the bundled Roboto font which overrides
        any earlier registration.
        """
        from kivy import resources as kivy_resources
        from kivy.logger import Logger
        fn_regular, fn_bold = _get_cjk_font_paths()
        if fn_regular:
            # Add the font directory to Kivy's resource search path
            font_dir = os.path.dirname(fn_regular)
            kivy_resources.resource_add_path(font_dir)
            try:
                LabelBase.register(
                    name='Roboto',
                    fn_regular=fn_regular,
                    fn_bold=fn_bold,
                    fn_italic=fn_regular,
                    fn_bolditalic=fn_bold,
                )
                Logger.info(f"CJK Font: Registered {os.path.basename(fn_regular)}")
            except Exception as exc:
                Logger.warning(f"CJK Font: Registration failed: {exc}")
        else:
            Logger.warning("CJK Font: No CJK font found, Korean/Japanese/Chinese may show as boxes")

    def on_start(self):
        """Called when the app starts."""
        # Register CJK font — MUST happen here (after _run_prepare finishes)
        # because KivyMD's _run_prepare re-registers the bundled Roboto AFTER build().
        self._register_cjk_font()

        if platform == 'android':
            request_storage_permissions()
            create_notification_channels()

        # Set app icon
        try:
            icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'Halgakos.png')
            if not os.path.exists(icon_path):
                icon_path = os.path.join(_src_dir, 'Halgakos.png')
            if os.path.exists(icon_path):
                self.icon = icon_path
        except Exception:
            pass

        # Initialize screens
        for screen_name in ['library', 'reader', 'multikey', 'progress', 'translation']:
            try:
                screen = self.root.ids.screen_manager.get_screen(screen_name)
                screen.app = self
            except Exception:
                pass

        try:
            lib_screen = self.root.ids.screen_manager.get_screen('library')
            lib_screen.load_books()
        except Exception as e:
            print(f"Library init error: {e}")

        # Handle "Open with" intent — if user tapped an EPUB in a file manager
        # and chose Glossarion, open that file after UI init completes
        if platform == 'android':
            Clock.schedule_once(self._handle_open_with_intent, 0.5)

    def switch_screen(self, screen_name, **kwargs):
        """Switch to a screen by name."""
        sm = self.root.ids.screen_manager
        sm.current = screen_name

        # Show/hide bottom nav for reader
        bottom_nav = self.root.ids.bottom_nav
        if screen_name == 'reader':
            bottom_nav.opacity = 0
            bottom_nav.disabled = True
            bottom_nav.height = 0
        else:
            bottom_nav.opacity = 1
            bottom_nav.disabled = False
            bottom_nav.height = 56  # dp(56) handled by KV

        # Pass data to target screen
        screen = sm.get_screen(screen_name)
        if hasattr(screen, 'on_enter_data') and kwargs:
            screen.on_enter_data(**kwargs)

    def open_reader(self, file_path):
        self.switch_screen('reader', file_path=file_path)

    def open_translation(self, file_path=None):
        self.switch_screen('translation', file_path=file_path)

    def on_pause(self):
        save_config(self.config_data)
        return True

    def on_resume(self):
        pass

    def on_stop(self):
        save_config(self.config_data)

    def _handle_open_with_intent(self, dt):
        """Check if the app was launched via 'Open with' and handle the file."""
        try:
            from android_file_utils import get_intent_file_path
            file_path = get_intent_file_path()
            if file_path and os.path.isfile(file_path):
                print(f"[INFO] Opening file from intent: {file_path}")
                # Refresh library so the imported file shows up
                try:
                    lib_screen = self.root.ids.screen_manager.get_screen('library')
                    lib_screen.load_books()
                except Exception:
                    pass
                # Open the file in the reader
                self.open_reader(file_path)
        except Exception as e:
            print(f"[WARN] Intent handling error: {e}")


if __name__ == '__main__':
    GlossarionApp().run()
