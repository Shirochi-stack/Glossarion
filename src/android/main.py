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

# Kivy config must be set BEFORE importing kivy
os.environ['KIVY_LOG_LEVEL'] = 'info'

from kivy.config import Config
Config.set('graphics', 'multisamples', '0')

from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager, SlideTransition
from kivy.core.window import Window
from kivy.utils import platform
from kivy.lang import Builder

# Import screens
from library_screen import LibraryScreen
from reader_screen import ReaderScreen
from multikey_screen import MultiKeyScreen
from translation_screen import TranslationScreen

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

        TranslationScreen:
            name: 'translation'

    MDBottomNavigation:
        id: bottom_nav
        panel_color: app.theme_cls.bg_dark
        text_color_active: 1, 1, 1, 1

        MDBottomNavigationItem:
            name: 'nav_library'
            text: 'Library'
            icon: 'bookshelf'
            on_tab_press: app.switch_screen('library')

        MDBottomNavigationItem:
            name: 'nav_keys'
            text: 'API Keys'
            icon: 'key-chain-variant'
            on_tab_press: app.switch_screen('multikey')

        MDBottomNavigationItem:
            name: 'nav_translate'
            text: 'Translate'
            icon: 'translate'
            on_tab_press: app.switch_screen('translation')
'''


class GlossarionApp(MDApp):
    """Main application class for Glossarion Android."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config_data = {}
        self.title = 'Glossarion'

    def build(self):
        # Set dark theme
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"
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

    def on_start(self):
        """Called when the app starts."""
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
        for screen_name in ['library', 'reader', 'multikey', 'translation']:
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
            bottom_nav.height = bottom_nav.minimum_height or 56

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


if __name__ == '__main__':
    GlossarionApp().run()
