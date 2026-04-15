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
from kivy.animation import Animation

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
from extract_glossary_screen import ExtractGlossaryScreen
from translation_screen import TranslationScreen
from progress_screen import ProgressScreen
from other_settings_screen import OtherSettingsScreen

# Import utilities
from android_config import load_config, save_config
from android_file_utils import request_storage_permissions
from android_notification import create_notification_channels

# KV layout — KivyMD 1.2.0 compatible
KV = '''
#:import SlideTransition kivy.uix.screenmanager.SlideTransition

FloatLayout:
    BoxLayout:
        orientation: 'vertical'
        size_hint: 1, 1

        ScreenManager:
            id: screen_manager
            transition: SlideTransition(duration=0.25)

            LibraryScreen:
                name: 'library'

            ReaderScreen:
                name: 'reader'

            MultiKeyScreen:
                name: 'multikey'

            ExtractGlossaryScreen:
                name: 'extract_glossary'

            ProgressScreen:
                name: 'progress'

            OtherSettingsScreen:
                name: 'other_settings'

            TranslationScreen:
                name: 'translation'

    Button:
        id: tab_menu_scrim
        size_hint: 1, 1
        background_normal: ''
        background_color: 0, 0, 0, 0.48
        opacity: 0
        disabled: True
        on_release: app.close_tab_menu()

    MDCard:
        id: tab_menu_panel
        size_hint: None, 1
        width: dp(260)
        x: -self.width
        y: 0
        elevation: 8
        radius: [0, dp(18), dp(18), 0]
        md_bg_color: app.theme_cls.bg_dark
        padding: [dp(10), dp(12), dp(10), dp(12)]

        BoxLayout:
            orientation: 'vertical'
            spacing: dp(8)

            BoxLayout:
                size_hint_y: None
                height: dp(40)
                spacing: dp(8)

                MDLabel:
                    text: "Navigation"
                    font_style: "Subtitle1"

                MDIconButton:
                    icon: "close"
                    on_release: app.close_tab_menu()

            ScrollView:
                do_scroll_x: False

                BoxLayout:
                    orientation: 'vertical'
                    size_hint_y: None
                    height: self.minimum_height
                    spacing: dp(8)
                    padding: [0, dp(4), 0, dp(4)]

                    MDRaisedButton:
                        text: "Library"
                        size_hint_y: None
                        height: dp(40)
                        md_bg_color: (0.20, 0.55, 0.90, 1) if app.root and app.root.ids.screen_manager.current == 'library' else (0.30, 0.30, 0.34, 1)
                        on_release: app.switch_screen_from_menu('library')

                    MDRaisedButton:
                        text: "Multi-Key"
                        size_hint_y: None
                        height: dp(40)
                        md_bg_color: (0.20, 0.55, 0.90, 1) if app.root and app.root.ids.screen_manager.current == 'multikey' else (0.30, 0.30, 0.34, 1)
                        on_release: app.switch_screen_from_menu('multikey')

                    MDRaisedButton:
                        text: "Extract Glossary"
                        size_hint_y: None
                        height: dp(40)
                        md_bg_color: (0.20, 0.55, 0.90, 1) if app.root and app.root.ids.screen_manager.current == 'extract_glossary' else (0.30, 0.30, 0.34, 1)
                        on_release: app.switch_screen_from_menu('extract_glossary')

                    MDRaisedButton:
                        text: "Progress"
                        size_hint_y: None
                        height: dp(40)
                        md_bg_color: (0.20, 0.55, 0.90, 1) if app.root and app.root.ids.screen_manager.current == 'progress' else (0.30, 0.30, 0.34, 1)
                        on_release: app.switch_screen_from_menu('progress')

                    MDRaisedButton:
                        text: "Translate"
                        size_hint_y: None
                        height: dp(40)
                        md_bg_color: (0.20, 0.55, 0.90, 1) if app.root and app.root.ids.screen_manager.current == 'translation' else (0.30, 0.30, 0.34, 1)
                        on_release: app.switch_screen_from_menu('translation')

                    MDRaisedButton:
                        text: "Other Settings"
                        size_hint_y: None
                        height: dp(40)
                        md_bg_color: (0.20, 0.55, 0.90, 1) if app.root and app.root.ids.screen_manager.current == 'other_settings' else (0.30, 0.30, 0.34, 1)
                        on_release: app.switch_screen_from_menu('other_settings')

    MDIconButton:
        id: tab_menu_button
        icon: "menu"
        pos_hint: {"x": 0.015, "top": 0.99}
        theme_text_color: "Custom"
        text_color: 1, 1, 1, 1
        on_release: app.toggle_tab_menu()
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
        for screen_name in ['library', 'reader', 'multikey', 'extract_glossary', 'progress', 'other_settings', 'translation']:
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

        # Show/hide left-menu trigger for reader
        menu_btn = self.root.ids.tab_menu_button
        if screen_name == 'reader':
            self.close_tab_menu()
            menu_btn.opacity = 0
            menu_btn.disabled = True
        else:
            menu_btn.opacity = 1
            menu_btn.disabled = False

        # Pass data to target screen
        screen = sm.get_screen(screen_name)
        if hasattr(screen, 'on_enter_data') and kwargs:
            screen.on_enter_data(**kwargs)

    def switch_screen_from_menu(self, screen_name):
        """Switch screen from side menu and close it."""
        self.switch_screen(screen_name)
        self.close_tab_menu()

    def toggle_tab_menu(self):
        if not self.root:
            return
        panel = self.root.ids.tab_menu_panel
        if panel.x < -1:
            self.open_tab_menu()
        else:
            self.close_tab_menu()

    def open_tab_menu(self):
        if not self.root:
            return
        panel = self.root.ids.tab_menu_panel
        scrim = self.root.ids.tab_menu_scrim
        menu_btn = self.root.ids.tab_menu_button
        menu_btn.disabled = True
        menu_btn.opacity = 0
        scrim.disabled = False
        scrim.opacity = 1
        Animation.cancel_all(panel)
        Animation.cancel_all(scrim)
        Animation(x=0, d=0.18, t='out_quad').start(panel)

    def close_tab_menu(self, *_args):
        if not self.root:
            return
        panel = self.root.ids.tab_menu_panel
        scrim = self.root.ids.tab_menu_scrim
        menu_btn = self.root.ids.tab_menu_button
        current = self.root.ids.screen_manager.current
        Animation.cancel_all(panel)
        Animation.cancel_all(scrim)
        # Force-release touch capture immediately; scrim animation race could
        # leave an invisible full-screen button blocking scroll/input.
        scrim.disabled = True
        scrim.opacity = 0
        Animation(x=-panel.width, d=0.18, t='out_quad').start(panel)
        if current == 'reader':
            menu_btn.disabled = True
            menu_btn.opacity = 0
        else:
            menu_btn.disabled = False
            menu_btn.opacity = 1

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
        import threading

        def _process_intent():
            try:
                from android_file_utils import get_intent_file_path
                file_path = get_intent_file_path()
                if file_path and os.path.isfile(file_path):
                    print(f"[INFO] Intent file copied to: {file_path}")
                    # Schedule UI updates on main thread
                    Clock.schedule_once(lambda dt: self._on_intent_file_ready(file_path), 0)
                else:
                    if file_path:
                        print(f"[WARN] Intent file path resolved but file missing: {file_path}")
            except Exception as e:
                print(f"[WARN] Intent handling error: {e}")
                import traceback
                traceback.print_exc()

        threading.Thread(target=_process_intent, daemon=True).start()

    def _on_intent_file_ready(self, file_path):
        """Called on the main thread after an intent file has been copied."""
        from kivymd.toast import toast
        fname = os.path.basename(file_path)
        toast(f"Imported: {fname}")

        # Refresh library so the imported file shows up
        try:
            lib_screen = self.root.ids.screen_manager.get_screen('library')
            lib_screen.load_books()
        except Exception:
            pass

        # Open the file in the reader
        self.open_reader(file_path)


if __name__ == '__main__':
    GlossarionApp().run()
