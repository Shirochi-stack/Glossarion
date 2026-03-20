# main.py
"""
Glossarion Android — Main entry point.
Kivy/KivyMD app with Material Design 3 dark theme.
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
Config.set('kivy', 'window_icon', '')
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

# KV layout for the root screen manager and bottom navigation
KV = '''
#:import SlideTransition kivy.uix.screenmanager.SlideTransition

MDBoxLayout:
    orientation: 'vertical'

    # Main content area
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

    # Bottom navigation bar (hidden when in reader)
    MDBottomNavigation:
        id: bottom_nav
        #panel_color: app.theme_cls.surfaceColor
        #text_color_active: app.theme_cls.primaryColor
        selected_color_background: app.theme_cls.primaryColor
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
        # Set Material Design 3 dark theme
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"

        # Set window background for desktop testing
        if platform != 'android':
            Window.size = (412, 915)  # ~6" phone at 1080p
            Window.clearcolor = (0.07, 0.07, 0.09, 1)  # #121217

        # Load config
        self.config_data = load_config()

        # Build UI
        root = Builder.load_string(KV)
        return root

    def on_start(self):
        """Called when the app starts."""
        # Request storage permissions on Android
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

        # Initialize library screen
        try:
            lib_screen = self.root.ids.screen_manager.get_screen('library')
            lib_screen.app = self
            lib_screen.load_books()
        except Exception as e:
            print(f"Library init error: {e}")

        # Pass app reference to all screens
        for screen_name in ['reader', 'multikey', 'translation']:
            try:
                screen = self.root.ids.screen_manager.get_screen(screen_name)
                screen.app = self
            except Exception:
                pass

    def switch_screen(self, screen_name, **kwargs):
        """Switch to a screen by name.
        
        Args:
            screen_name: Name of the screen to switch to
            **kwargs: Optional data to pass to the screen
        """
        sm = self.root.ids.screen_manager
        sm.current = screen_name

        # Show/hide bottom nav based on screen
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
        """Navigate to the reader screen with the given file."""
        self.switch_screen('reader', file_path=file_path)

    def open_translation(self, file_path=None):
        """Navigate to the translation screen, optionally with a pre-selected file."""
        self.switch_screen('translation', file_path=file_path)

    def on_pause(self):
        """Handle app pause (Android background)."""
        # Save config
        save_config(self.config_data)
        return True

    def on_resume(self):
        """Handle app resume (Android foreground)."""
        pass

    def on_stop(self):
        """Handle app stop."""
        save_config(self.config_data)


if __name__ == '__main__':
    GlossarionApp().run()
