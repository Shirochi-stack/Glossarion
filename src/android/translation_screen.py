# translation_screen.py
"""
AndroidTranslationGUI — Translation settings and execution screen.
Mirrors translator_gui.py's settings and app.py's set_all_environment_variables().
"""

import os
import sys
import logging
import threading
import time

from kivy.properties import ObjectProperty, StringProperty, BooleanProperty, NumericProperty
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp, sp

from kivymd.uix.screen import MDScreen
from kivymd.uix.card import MDCard
from kivymd.uix.button import MDButton, MDButtonText, MDFabButton, MDIconButton, MDExtendedFabButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.selectioncontrol import MDSwitch
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.slider import MDSlider
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.snackbar import MDSnackbar, MDSnackbarText
from kivymd.uix.dialog import MDDialog, MDDialogHeadlineText, MDDialogContentContainer, MDDialogButtonContainer
from kivymd.uix.list import MDList, MDListItem, MDListItemHeadlineText, MDListItemSupportingText
from kivy.uix.scrollview import ScrollView

logger = logging.getLogger(__name__)

# Add parent dir to import shared modules
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Built-in model suggestions
DEFAULT_MODELS = [
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'gemini-2.0-flash',
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
    'gpt-4o',
    'gpt-4o-mini',
    'claude-sonnet-4-20250514',
    'claude-3-5-sonnet-20241022',
    'claude-3-5-haiku-20241022',
    'deepseek/deepseek-chat',
    'qwen/qwen3-235b-a22b',
    'authgpt/gpt-5.2',
    'authgpt/gemini-2.5-flash',
]

KV = '''
<TranslationScreen>:
    MDBoxLayout:
        orientation: 'vertical'

        MDTopAppBar:
            MDTopAppBarTitle:
                text: "Translation Settings"

        ScrollView:
            id: settings_scroll

            MDBoxLayout:
                orientation: 'vertical'
                adaptive_height: True
                padding: [dp(12), dp(8)]
                spacing: dp(8)

                # ── File Selection ──
                MDCard:
                    style: "outlined"
                    size_hint: 1, None
                    height: dp(80)
                    padding: dp(16)

                    MDBoxLayout:
                        spacing: dp(8)

                        MDBoxLayout:
                            orientation: 'vertical'
                            size_hint_x: 0.75

                            MDLabel:
                                text: "Input File"
                                font_style: "Label"
                                role: "large"
                                theme_text_color: "Secondary"

                            MDLabel:
                                id: file_label
                                text: root.selected_file_display
                                font_style: "Body"
                                role: "medium"
                                shorten: True
                                shorten_from: "center"

                        MDIconButton:
                            icon: "folder-open-outline"
                            pos_hint: {"center_y": 0.5}
                            on_release: root.pick_file()

                # ── Model ──
                MDCard:
                    style: "outlined"
                    size_hint: 1, None
                    height: dp(80)
                    padding: dp(16)

                    MDBoxLayout:
                        spacing: dp(8)

                        MDTextField:
                            id: model_field
                            hint_text: "Model"
                            text: root.model_name
                            on_text: root.model_name = self.text
                            size_hint_x: 0.8

                        MDIconButton:
                            icon: "chevron-down"
                            pos_hint: {"center_y": 0.5}
                            on_release: root.show_model_menu(self)

                # ── API Key ──
                MDCard:
                    style: "outlined"
                    size_hint: 1, None
                    height: dp(120)
                    padding: dp(16)

                    MDBoxLayout:
                        orientation: 'vertical'
                        spacing: dp(4)

                        MDTextField:
                            id: api_key_field
                            hint_text: "API Key"
                            text: root.api_key
                            on_text: root.api_key = self.text
                            password: True

                        MDBoxLayout:
                            size_hint_y: None
                            height: dp(36)
                            spacing: dp(8)

                            MDLabel:
                                text: "Use Multi-Key Rotation"
                                size_hint_x: 0.6

                            MDSwitch:
                                id: multikey_switch
                                active: root.use_multi_keys
                                on_active: root.use_multi_keys = self.active
                                size_hint_x: 0.2

                            MDIconButton:
                                icon: "key-chain-variant"
                                on_release: root.open_multikey_manager()
                                size_hint_x: 0.2

                # ── Temperature ──
                MDCard:
                    style: "outlined"
                    size_hint: 1, None
                    height: dp(72)
                    padding: dp(16)

                    MDBoxLayout:
                        spacing: dp(8)

                        MDLabel:
                            text: "Temperature"
                            size_hint_x: 0.3

                        MDSlider:
                            id: temp_slider
                            min: 0.0
                            max: 2.0
                            step: 0.05
                            value: root.temperature
                            on_value: root.temperature = round(self.value, 2)
                            size_hint_x: 0.55

                        MDLabel:
                            id: temp_label
                            text: str(root.temperature)
                            halign: "right"
                            size_hint_x: 0.15

                # ── Profile ──
                MDCard:
                    style: "outlined"
                    size_hint: 1, None
                    height: dp(72)
                    padding: dp(16)

                    MDBoxLayout:
                        spacing: dp(8)

                        MDLabel:
                            text: "Profile"
                            size_hint_x: 0.25

                        MDButton:
                            id: profile_btn
                            on_release: root.show_profile_menu(self)
                            size_hint_x: 0.75

                            MDButtonText:
                                id: profile_btn_text
                                text: root.active_profile

                # ── System Prompt (expandable) ──
                MDCard:
                    style: "outlined"
                    size_hint: 1, None
                    height: dp(56) if not root.prompt_expanded else dp(240)
                    padding: dp(16)

                    MDBoxLayout:
                        orientation: 'vertical'

                        MDBoxLayout:
                            size_hint_y: None
                            height: dp(32)

                            MDLabel:
                                text: "System Prompt"
                                size_hint_x: 0.7

                            MDIconButton:
                                icon: "chevron-down" if not root.prompt_expanded else "chevron-up"
                                on_release: root.toggle_prompt()

                        MDTextField:
                            id: prompt_field
                            text: root.system_prompt
                            on_text: root.system_prompt = self.text
                            multiline: True
                            max_height: dp(160)
                            opacity: 1 if root.prompt_expanded else 0
                            disabled: not root.prompt_expanded

                # ── Output Tokens ──
                MDCard:
                    style: "outlined"
                    size_hint: 1, None
                    height: dp(72)
                    padding: dp(16)

                    MDBoxLayout:
                        spacing: dp(8)

                        MDLabel:
                            text: "Max Output Tokens"
                            size_hint_x: 0.5

                        MDTextField:
                            id: tokens_field
                            text: str(root.max_output_tokens)
                            on_text: root._parse_tokens(self.text)
                            input_filter: "int"
                            size_hint_x: 0.5

                # ── Batch Translation ──
                MDCard:
                    style: "outlined"
                    size_hint: 1, None
                    height: dp(72)
                    padding: dp(16)

                    MDBoxLayout:
                        spacing: dp(8)

                        MDLabel:
                            text: "Batch Translation"
                            size_hint_x: 0.4

                        MDSwitch:
                            id: batch_switch
                            active: root.batch_enabled
                            on_active: root.batch_enabled = self.active
                            size_hint_x: 0.2

                        MDLabel:
                            text: "Size:"
                            halign: "right"
                            size_hint_x: 0.15

                        MDTextField:
                            id: batch_size_field
                            text: str(root.batch_size)
                            on_text: root._parse_batch_size(self.text)
                            input_filter: "int"
                            size_hint_x: 0.25

                # ── Auto Glossary ──
                MDCard:
                    style: "outlined"
                    size_hint: 1, None
                    height: dp(56)
                    padding: dp(16)

                    MDBoxLayout:
                        MDLabel:
                            text: "Auto Glossary"
                            size_hint_x: 0.7

                        MDSwitch:
                            id: glossary_switch
                            active: root.auto_glossary
                            on_active: root.auto_glossary = self.active
                            size_hint_x: 0.3

                # ── Output Language ──
                MDCard:
                    style: "outlined"
                    size_hint: 1, None
                    height: dp(72)
                    padding: dp(16)

                    MDBoxLayout:
                        spacing: dp(8)

                        MDLabel:
                            text: "Output Language"
                            size_hint_x: 0.4

                        MDTextField:
                            id: language_field
                            text: root.output_language
                            on_text: root.output_language = self.text
                            size_hint_x: 0.6

                # ── Spacer for FAB ──
                Widget:
                    size_hint_y: None
                    height: dp(80)

        # Log output panel (slides up during translation)
        MDBoxLayout:
            id: log_panel
            orientation: 'vertical'
            size_hint_y: None
            height: 0
            opacity: 0
            md_bg_color: app.theme_cls.surfaceContainerColor

            MDBoxLayout:
                size_hint_y: None
                height: dp(40)
                padding: [dp(16), 0]

                MDLabel:
                    text: "Translation Log"
                    font_style: "Title"
                    role: "small"
                    size_hint_x: 0.7

                MDIconButton:
                    icon: "chevron-down"
                    on_release: root._hide_log_panel()

            ScrollView:
                id: log_scroll
                do_scroll_x: False

                MDLabel:
                    id: log_label
                    text: root.log_text
                    markup: True
                    size_hint_y: None
                    height: self.texture_size[1] + dp(20)
                    padding: [dp(12), dp(4)]
                    font_size: sp(12)
                    font_name: "RobotoMono"

    # Run button (FAB)
    MDExtendedFabButton:
        id: run_fab
        icon: "play"
        pos_hint: {"right": 0.95, "y": 0.02}
        on_release: root.run_translation()

        MDExtendedFabButtonText:
            id: run_fab_text
            text: "Translate"
'''


class TranslationScreen(MDScreen):
    """Translation settings and execution screen."""

    app = ObjectProperty(None, allownone=True)

    # Settings properties
    selected_file = StringProperty('')
    selected_file_display = StringProperty('No file selected')
    model_name = StringProperty('authgpt/gpt-5.2')
    api_key = StringProperty('')
    use_multi_keys = BooleanProperty(False)
    temperature = NumericProperty(0.3)
    active_profile = StringProperty('Universal')
    system_prompt = StringProperty('')
    max_output_tokens = NumericProperty(128000)
    batch_enabled = BooleanProperty(False)
    batch_size = NumericProperty(10)
    auto_glossary = BooleanProperty(True)
    output_language = StringProperty('English')
    prompt_expanded = BooleanProperty(False)

    # Translation state
    is_translating = BooleanProperty(False)
    log_text = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Builder.load_string(KV)
        self._translation_thread = None
        self._model_menu = None
        self._profile_menu = None
        self._stop_requested = False
        self._log_lines = []

    def on_enter(self, *args):
        """Called when screen is displayed."""
        self._load_settings()

    def on_enter_data(self, file_path=None, **kwargs):
        """Called when navigating with data."""
        if file_path:
            self.selected_file = file_path
            self.selected_file_display = os.path.basename(file_path)

    def _load_settings(self):
        """Load settings from config."""
        if not self.app:
            return
        cfg = self.app.config_data
        self.model_name = cfg.get('model', 'authgpt/gpt-5.2')
        self.api_key = cfg.get('api_key', '')
        self.use_multi_keys = cfg.get('use_multi_api_keys', False)
        self.temperature = cfg.get('translation_temperature', 0.3)
        self.active_profile = cfg.get('active_profile', 'Universal')
        self.max_output_tokens = cfg.get('max_output_tokens', 128000)
        self.batch_enabled = cfg.get('batch_translation', False)
        self.batch_size = cfg.get('batch_size', 10)
        self.auto_glossary = cfg.get('enable_auto_glossary', True)
        self.output_language = cfg.get('output_language', 'English')

        # Load system prompt from profile
        profiles = cfg.get('prompt_profiles', {})
        self.system_prompt = profiles.get(self.active_profile, '')

    def _save_settings(self):
        """Save current settings to config."""
        if not self.app:
            return
        cfg = self.app.config_data
        cfg['model'] = self.model_name
        cfg['api_key'] = self.api_key
        cfg['use_multi_api_keys'] = self.use_multi_keys
        cfg['translation_temperature'] = self.temperature
        cfg['active_profile'] = self.active_profile
        cfg['max_output_tokens'] = self.max_output_tokens
        cfg['batch_translation'] = self.batch_enabled
        cfg['batch_size'] = self.batch_size
        cfg['enable_auto_glossary'] = self.auto_glossary
        cfg['output_language'] = self.output_language
        cfg['active_system_prompt'] = self.system_prompt

        from android_config import save_config
        save_config(cfg)

    # ── UI Helpers ──

    def pick_file(self):
        """Open file picker for EPUB/TXT."""
        try:
            from plyer import filechooser
            filechooser.open_file(
                title="Select EPUB or TXT file",
                filters=[("Books", "*.epub", "*.txt")],
                on_selection=self._on_file_selected,
                multiple=False,
            )
        except Exception as e:
            self._append_log(f"⚠️ File picker error: {e}")

    def _on_file_selected(self, selection):
        if selection:
            path = selection[0] if isinstance(selection, list) else selection
            self.selected_file = path
            self.selected_file_display = os.path.basename(path)

    def show_model_menu(self, caller):
        """Show model selection dropdown."""
        items = [
            {
                "text": model,
                "on_release": lambda x=model: self._select_model(x),
            }
            for model in DEFAULT_MODELS
        ]
        self._model_menu = MDDropdownMenu(caller=caller, items=items)
        self._model_menu.open()

    def _select_model(self, model):
        self.model_name = model
        if self._model_menu:
            self._model_menu.dismiss()

    def show_profile_menu(self, caller):
        """Show profile selection dropdown."""
        # Built-in profiles
        profiles = [
            'Universal', 'Korean_BS', 'Korean_html2text',
            'Japanese_BS', 'Japanese_html2text',
            'Chinese_BS', 'Chinese_html2text',
        ]
        # Add custom profiles from config
        if self.app:
            custom = list(self.app.config_data.get('prompt_profiles', {}).keys())
            for p in custom:
                if p not in profiles:
                    profiles.append(p)

        items = [
            {
                "text": p,
                "on_release": lambda x=p: self._select_profile(x),
            }
            for p in profiles
        ]
        self._profile_menu = MDDropdownMenu(caller=caller, items=items)
        self._profile_menu.open()

    def _select_profile(self, profile):
        self.active_profile = profile
        if self._profile_menu:
            self._profile_menu.dismiss()
        # Load prompt for this profile
        if self.app:
            profiles = self.app.config_data.get('prompt_profiles', {})
            self.system_prompt = profiles.get(profile, '')

    def toggle_prompt(self):
        self.prompt_expanded = not self.prompt_expanded

    def open_multikey_manager(self):
        """Navigate to multi-key manager screen."""
        if self.app:
            self.app.switch_screen('multikey')

    def _parse_tokens(self, text):
        try:
            self.max_output_tokens = int(text)
        except (ValueError, TypeError):
            pass

    def _parse_batch_size(self, text):
        try:
            self.batch_size = int(text)
        except (ValueError, TypeError):
            pass

    # ── Translation Execution ──

    def run_translation(self):
        """Start or stop translation."""
        if self.is_translating:
            self._stop_translation()
            return

        # Validate
        if not self.selected_file or not os.path.isfile(self.selected_file):
            MDSnackbar(
                MDSnackbarText(text="Please select a file first"),
                y=dp(24), pos_hint={"center_x": 0.5},
            ).open()
            return

        if not self.api_key and not self.use_multi_keys:
            MDSnackbar(
                MDSnackbarText(text="Please enter an API key"),
                y=dp(24), pos_hint={"center_x": 0.5},
            ).open()
            return

        # Save settings
        self._save_settings()

        # Start translation
        self.is_translating = True
        self._stop_requested = False
        self._log_lines = []
        self.log_text = ''

        # Update FAB
        try:
            self.ids.run_fab.icon = "stop"
            self.ids.run_fab_text.text = "Stop"
        except Exception:
            pass

        # Show log panel
        self._show_log_panel()

        # Set environment variables
        self._append_log("🚀 Setting up environment...")
        self._setup_environment()

        # Start translation thread
        self._translation_thread = threading.Thread(
            target=self._translation_worker,
            daemon=True,
        )
        self._translation_thread.start()

    def _stop_translation(self):
        """Request translation stop."""
        self._stop_requested = True
        os.environ['GRACEFUL_STOP'] = '1'
        self._append_log("⏹ Stop requested — finishing current chapter...")

        try:
            self.ids.run_fab.icon = "clock-outline"
            self.ids.run_fab_text.text = "Stopping..."
        except Exception:
            pass

        # Try to set the stop flag in the translation engine
        try:
            from TransateKRtoEN import set_stop_flag
            set_stop_flag(True)
        except Exception:
            pass

    def _setup_environment(self):
        """Set all environment variables for the translation engine."""
        from android_env_propagator import set_all_env_vars, set_per_run_env_vars, set_input_file_env
        from android_file_utils import get_output_dir

        # Build config dict from current settings
        cfg = {}
        if self.app:
            cfg = self.app.config_data.copy()

        # Override with current UI values
        cfg['model'] = self.model_name
        cfg['api_key'] = self.api_key
        cfg['use_multi_api_keys'] = self.use_multi_keys
        cfg['translation_temperature'] = self.temperature
        cfg['active_profile'] = self.active_profile
        cfg['active_system_prompt'] = self.system_prompt
        cfg['max_output_tokens'] = self.max_output_tokens
        cfg['batch_translation'] = self.batch_enabled
        cfg['batch_size'] = self.batch_size
        cfg['enable_auto_glossary'] = self.auto_glossary
        cfg['output_language'] = self.output_language

        set_all_env_vars(cfg)
        set_per_run_env_vars()

        output_dir = get_output_dir(self.selected_file)
        set_input_file_env(self.selected_file, output_dir)

        self._append_log(f"📁 Output: {output_dir}")
        self._append_log(f"🤖 Model: {self.model_name}")
        self._append_log(f"🌡️ Temperature: {self.temperature}")
        self._append_log(f"📦 Batch: {'On' if self.batch_enabled else 'Off'} (size {self.batch_size})")

    def _translation_worker(self):
        """Background thread that runs the translation engine."""
        from android_notification import notify_translation_progress, notify_translation_complete, cancel_progress_notification

        Clock.schedule_once(lambda dt: self._append_log("🟢 Translation started"))

        try:
            # Redirect print to capture logs
            import builtins
            original_print = builtins.print

            def captured_print(*args, **kwargs):
                text = ' '.join(str(a) for a in args)
                Clock.schedule_once(lambda dt, t=text: self._append_log(t))
                original_print(*args, **kwargs)

            builtins.print = captured_print

            try:
                # Import and run the translation engine
                import importlib
                if 'TransateKRtoEN' in sys.modules:
                    importlib.reload(sys.modules['TransateKRtoEN'])
                from TransateKRtoEN import TranslationConfig, translate_epub

                config = TranslationConfig()

                input_path = self.selected_file
                ext = os.path.splitext(input_path)[1].lower()

                if ext == '.epub':
                    notify_translation_progress("Translating", 0, 100, os.path.basename(input_path))
                    result = translate_epub(input_path, config)
                    self._append_log(f"✅ Translation complete: {result}")
                elif ext == '.txt':
                    notify_translation_progress("Translating", 0, 100, os.path.basename(input_path))
                    # TXT files use the same pipeline but via TextFileProcessor
                    os.environ['input_path'] = input_path
                    result = translate_epub(input_path, config)
                    self._append_log(f"✅ Translation complete: {result}")
                else:
                    self._append_log(f"❌ Unsupported format: {ext}")
                    return

                # Notify completion
                cancel_progress_notification()
                notify_translation_complete(
                    "Translation Complete",
                    f"{os.path.basename(input_path)} has been translated"
                )

            finally:
                builtins.print = original_print

        except Exception as e:
            import traceback
            error_text = traceback.format_exc()
            Clock.schedule_once(lambda dt: self._append_log(f"❌ Error: {e}\n{error_text}"))

            try:
                from android_notification import cancel_progress_notification
                cancel_progress_notification()
            except Exception:
                pass

        finally:
            Clock.schedule_once(lambda dt: self._translation_finished())

    def _translation_finished(self):
        """Called on main thread when translation completes."""
        self.is_translating = False
        self._stop_requested = False

        try:
            self.ids.run_fab.icon = "play"
            self.ids.run_fab_text.text = "Translate"
        except Exception:
            pass

    # ── Log Panel ──

    def _append_log(self, text):
        """Append text to the log panel."""
        self._log_lines.append(text)
        # Keep last 200 lines
        if len(self._log_lines) > 200:
            self._log_lines = self._log_lines[-200:]
        self.log_text = '\n'.join(self._log_lines)

        # Auto-scroll to bottom
        try:
            Clock.schedule_once(lambda dt: setattr(self.ids.log_scroll, 'scroll_y', 0), 0.1)
        except Exception:
            pass

    def _show_log_panel(self):
        """Show the log panel."""
        from kivy.animation import Animation
        panel = self.ids.log_panel
        anim = Animation(height=dp(250), opacity=1, duration=0.3)
        anim.start(panel)

    def _hide_log_panel(self):
        """Hide the log panel."""
        from kivy.animation import Animation
        panel = self.ids.log_panel
        anim = Animation(height=0, opacity=0, duration=0.2)
        anim.start(panel)
