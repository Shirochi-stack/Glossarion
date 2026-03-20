# translation_screen.py
"""
AndroidTranslationGUI — Translation settings and execution screen.
KivyMD 1.2.0 compatible.
"""

import os
import sys
import logging
import threading

from kivy.properties import (ObjectProperty, StringProperty, BooleanProperty,
                              NumericProperty)
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.animation import Animation

from kivymd.uix.screen import MDScreen
from kivymd.uix.button import MDFlatButton, MDRaisedButton, MDFloatingActionButton, MDIconButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.selectioncontrol import MDSwitch
from kivymd.uix.label import MDLabel
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.card import MDCard
from kivymd.uix.slider import MDSlider
from kivymd.uix.menu import MDDropdownMenu
from kivymd.toast import toast
from kivymd.uix.dialog import MDDialog

logger = logging.getLogger(__name__)

_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

DEFAULT_MODELS = [
    'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash',
    'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
    'gpt-4o', 'gpt-4o-mini',
    'claude-sonnet-4-20250514', 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022',
    'deepseek/deepseek-chat', 'qwen/qwen3-235b-a22b',
    'authgpt/gpt-5.2', 'authgpt/gemini-2.5-flash',
]

GLOSSARY_MODES = ['Off', 'Minimal', 'Balanced', 'Full', 'No Glossary']

# These match translator_gui.py's _get_protected_prompt_profiles()
DEFAULT_PROFILE_NAMES = [
    'Universal',
    'Korean_BeautifulSoup',
    'Japanese_BeautifulSoup',
    'Chinese_BeautifulSoup',
    'Korean_html2text',
    'Japanese_html2text',
    'Chinese_html2text',
    'korean_OCR',
    'japanese_OCR',
    'chinese_OCR',
]

KV = '''
<TranslationScreen>:
    BoxLayout:
        orientation: 'vertical'

        MDTopAppBar:
            title: "Translation Settings"
            md_bg_color: app.theme_cls.primary_color
            elevation: 2

        ScrollView:
            id: settings_scroll

            BoxLayout:
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                padding: [dp(12), dp(8)]
                spacing: dp(10)

                # ── File selection ──
                MDCard:
                    size_hint: 1, None
                    height: dp(80)
                    padding: dp(16)
                    elevation: 1

                    BoxLayout:
                        spacing: dp(8)

                        BoxLayout:
                            orientation: 'vertical'
                            size_hint_x: 0.8

                            MDLabel:
                                text: "Input File"
                                font_style: "Caption"
                                theme_text_color: "Secondary"
                                size_hint_y: None
                                height: dp(20)

                            MDLabel:
                                id: file_label
                                text: root.selected_file_display
                                font_style: "Body1"
                                shorten: True
                                shorten_from: "center"

                        MDIconButton:
                            icon: "folder-open-outline"
                            pos_hint: {"center_y": 0.5}
                            on_release: root.pick_file()

                # ── Model ──
                MDCard:
                    size_hint: 1, None
                    height: dp(80)
                    padding: dp(16)
                    elevation: 1

                    BoxLayout:
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
                    size_hint: 1, None
                    height: dp(130)
                    padding: dp(16)
                    elevation: 1

                    BoxLayout:
                        orientation: 'vertical'
                        spacing: dp(8)

                        MDTextField:
                            id: api_key_field
                            hint_text: "API Key"
                            text: root.api_key
                            on_text: root.api_key = self.text
                            password: True

                        BoxLayout:
                            size_hint_y: None
                            height: dp(40)
                            spacing: dp(12)

                            MDLabel:
                                text: "Multi-Key Rotation"
                                size_hint_x: 0.55

                            MDSwitch:
                                active: root.use_multi_keys
                                on_active: root.use_multi_keys = self.active
                                size_hint_x: 0.2
                                pos_hint: {"center_y": 0.5}

                            MDIconButton:
                                icon: "key-chain-variant"
                                on_release: root.open_multikey_manager()
                                size_hint_x: 0.25

                # ── Temperature ──
                MDCard:
                    size_hint: 1, None
                    height: dp(72)
                    padding: dp(16)
                    elevation: 1

                    BoxLayout:
                        spacing: dp(8)

                        MDLabel:
                            text: "Temp"
                            size_hint_x: 0.15

                        MDSlider:
                            min: 0.0
                            max: 2.0
                            step: 0.05
                            value: root.temperature
                            on_value: root.temperature = round(self.value, 2)
                            size_hint_x: 0.6

                        MDLabel:
                            text: str(root.temperature)
                            halign: "right"
                            size_hint_x: 0.25

                # ── Profile ──
                MDCard:
                    size_hint: 1, None
                    height: dp(72)
                    padding: dp(16)
                    elevation: 1

                    BoxLayout:
                        spacing: dp(12)

                        MDLabel:
                            text: "Profile"
                            size_hint_x: 0.2

                        MDRaisedButton:
                            id: profile_btn
                            text: root.active_profile
                            on_release: root.show_profile_menu(self)
                            size_hint_x: 0.8

                # ── System Prompt (expandable) ──
                MDCard:
                    size_hint: 1, None
                    height: dp(56) if not root.prompt_expanded else dp(260)
                    padding: dp(16)
                    elevation: 1

                    BoxLayout:
                        orientation: 'vertical'
                        spacing: dp(4)

                        BoxLayout:
                            size_hint_y: None
                            height: dp(32)

                            MDLabel:
                                text: "System Prompt"

                            MDIconButton:
                                icon: "chevron-down" if not root.prompt_expanded else "chevron-up"
                                on_release: root.toggle_prompt()

                        ScrollView:
                            size_hint_y: None
                            height: dp(180) if root.prompt_expanded else 0
                            opacity: 1 if root.prompt_expanded else 0

                            MDTextField:
                                id: prompt_field
                                text: root.system_prompt
                                on_text: root.system_prompt = self.text
                                multiline: True
                                size_hint_y: None
                                height: max(dp(180), self.minimum_height)

                # ── Output tokens ──
                MDCard:
                    size_hint: 1, None
                    height: dp(72)
                    padding: dp(16)
                    elevation: 1

                    BoxLayout:
                        spacing: dp(8)

                        MDLabel:
                            text: "Max Output Tokens"
                            size_hint_x: 0.5

                        MDTextField:
                            text: str(root.max_output_tokens)
                            on_text: root._parse_tokens(self.text)
                            input_filter: "int"
                            size_hint_x: 0.5

                # ── Batch translation ──
                MDCard:
                    size_hint: 1, None
                    height: dp(72)
                    padding: dp(16)
                    elevation: 1

                    BoxLayout:
                        spacing: dp(8)

                        MDLabel:
                            text: "Batch"
                            size_hint_x: 0.2

                        MDSwitch:
                            active: root.batch_enabled
                            on_active: root.batch_enabled = self.active
                            size_hint_x: 0.2
                            pos_hint: {"center_y": 0.5}

                        MDLabel:
                            text: "Size:"
                            halign: "right"
                            size_hint_x: 0.25

                        MDTextField:
                            text: str(root.batch_size)
                            on_text: root._parse_batch_size(self.text)
                            input_filter: "int"
                            size_hint_x: 0.35

                # ── Glossary mode ──
                MDCard:
                    size_hint: 1, None
                    height: dp(72)
                    padding: dp(16)
                    elevation: 1

                    BoxLayout:
                        spacing: dp(12)

                        MDLabel:
                            text: "Glossary Mode"
                            size_hint_x: 0.35

                        MDRaisedButton:
                            id: glossary_btn
                            text: root.glossary_mode
                            on_release: root.show_glossary_menu(self)
                            size_hint_x: 0.65

                # ── Output language ──
                MDCard:
                    size_hint: 1, None
                    height: dp(72)
                    padding: dp(16)
                    elevation: 1

                    BoxLayout:
                        spacing: dp(8)

                        MDLabel:
                            text: "Output Language"
                            size_hint_x: 0.4

                        MDTextField:
                            text: root.output_language
                            on_text: root.output_language = self.text
                            size_hint_x: 0.6

                # Spacer for FAB
                Widget:
                    size_hint_y: None
                    height: dp(80)

        # Log panel
        BoxLayout:
            id: log_panel
            orientation: 'vertical'
            size_hint_y: None
            height: 0
            opacity: 0
            canvas.before:
                Color:
                    rgba: app.theme_cls.bg_dark
                Rectangle:
                    pos: self.pos
                    size: self.size

            BoxLayout:
                size_hint_y: None
                height: dp(40)
                padding: [dp(16), 0]

                MDLabel:
                    text: "Translation Log"
                    font_style: "Subtitle1"
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

    # Run FAB
    MDFloatingActionButton:
        id: run_fab
        icon: "play"
        pos_hint: {"right": 0.95, "y": 0.02}
        elevation: 4
        on_release: root.run_translation()
'''


class TranslationScreen(MDScreen):
    """Translation settings and execution."""

    app = ObjectProperty(None, allownone=True)

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
    auto_glossary = BooleanProperty(True)  # legacy compat
    glossary_mode = StringProperty('Balanced')
    output_language = StringProperty('English')
    prompt_expanded = BooleanProperty(False)

    is_translating = BooleanProperty(False)
    log_text = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Builder.load_string(KV)
        self._translation_thread = None
        self._model_menu = None
        self._profile_menu = None
        self._glossary_menu = None
        self._stop_requested = False
        self._log_lines = []

    def on_enter(self, *args):
        self._load_settings()

    def on_enter_data(self, file_path=None, **kwargs):
        if file_path:
            self.selected_file = file_path
            self.selected_file_display = os.path.basename(file_path)

    def _load_settings(self):
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
        self.glossary_mode = cfg.get('auto_glossary_mode', 'balanced').capitalize()
        if self.glossary_mode == 'No_glossary':
            self.glossary_mode = 'No Glossary'
        self.output_language = cfg.get('output_language', 'English')

        # Load system prompt for active profile
        self.system_prompt = self._get_prompt_for_profile(self.active_profile)

    def _get_prompt_for_profile(self, profile_name):
        """Get system prompt text for a profile.
        
        Handles both old format (string) and new format (dict with 'prompt' key).
        Falls back to embedded default prompts with {target_lang} replacement.
        """
        import re
        target_lang = self.output_language or 'English'

        def _resolve(raw):
            """Replace placeholders in a raw prompt string."""
            if not raw:
                return ''
            raw = raw.replace('{target_lang}', target_lang)
            raw = re.sub(r'\s*\{split_marker_instruction\}\s*', '\n', raw)
            while '\n\n\n' in raw:
                raw = raw.replace('\n\n\n', '\n\n')
            return raw.strip()

        if not self.app:
            # No app context — use embedded defaults directly
            from default_prompts import get_prompt
            return get_prompt(profile_name, target_lang)

        # 1. Try the android config's prompt_profiles
        profiles = self.app.config_data.get('prompt_profiles', {})
        if profiles and profile_name in profiles:
            data = profiles[profile_name]
            if isinstance(data, str):
                return _resolve(data)
            elif isinstance(data, dict):
                return _resolve(data.get('prompt', ''))

        # 2. Try loading from desktop config
        try:
            desktop_config = self._load_desktop_config()
            if desktop_config:
                dt_profiles = desktop_config.get('prompt_profiles', {})
                if profile_name in dt_profiles:
                    data = dt_profiles[profile_name]
                    if isinstance(data, str):
                        return _resolve(data)
                    elif isinstance(data, dict):
                        return _resolve(data.get('prompt', ''))
        except Exception:
            pass

        # 3. Use embedded defaults (always available)
        from default_prompts import get_prompt
        return get_prompt(profile_name, target_lang)

    def _load_desktop_config(self):
        """Try to load the main desktop config.json."""
        import json
        # Check common locations
        paths = []
        if sys.platform == 'win32':
            appdata = os.environ.get('APPDATA', '')
            localappdata = os.environ.get('LOCALAPPDATA', '')
            home = os.path.expanduser('~')
            paths = [
                os.path.join(home, '.glossarion', 'config.json'),
                os.path.join(appdata, 'Glossarion', 'config.json'),
                os.path.join(localappdata, 'Glossarion', 'config.json'),
            ]
        else:
            home = os.path.expanduser('~')
            paths = [
                os.path.join(home, '.glossarion', 'config.json'),
                os.path.join(home, '.config', 'glossarion', 'config.json'),
            ]

        for p in paths:
            if os.path.isfile(p):
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception:
                    continue
        return None

    def _save_settings(self):
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
        cfg['auto_glossary_mode'] = self.glossary_mode.lower().replace(' ', '_')
        cfg['output_language'] = self.output_language
        cfg['active_system_prompt'] = self.system_prompt

        # Save prompt to profile
        profiles = cfg.get('prompt_profiles', {})
        profiles[self.active_profile] = self.system_prompt
        cfg['prompt_profiles'] = profiles

        from android_config import save_config
        save_config(cfg)

    # ── UI ──

    def pick_file(self):
        try:
            from plyer import filechooser
            filechooser.open_file(
                title="Select EPUB or TXT",
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
        items = [{"text": m, "viewclass": "OneLineListItem",
                  "on_release": lambda x=m: self._select_model(x)} for m in DEFAULT_MODELS]
        self._model_menu = MDDropdownMenu(caller=caller, items=items, width_mult=5)
        self._model_menu.open()

    def _select_model(self, model):
        self.model_name = model
        if self._model_menu:
            self._model_menu.dismiss()

    def show_profile_menu(self, caller):
        # Start with built-in profiles (full names)
        profiles = list(DEFAULT_PROFILE_NAMES)
        # Add custom profiles from config
        if self.app:
            for p in self.app.config_data.get('prompt_profiles', {}).keys():
                if p not in profiles:
                    profiles.append(p)
        items = [{"text": p, "viewclass": "OneLineListItem",
                  "on_release": lambda x=p: self._select_profile(x)} for p in profiles]
        self._profile_menu = MDDropdownMenu(caller=caller, items=items, width_mult=5)
        self._profile_menu.open()

    def _select_profile(self, profile):
        self.active_profile = profile
        if self._profile_menu:
            self._profile_menu.dismiss()
        # Load the system prompt for this profile
        self.system_prompt = self._get_prompt_for_profile(profile)

    def toggle_prompt(self):
        self.prompt_expanded = not self.prompt_expanded

    def show_glossary_menu(self, caller):
        items = [{"text": m, "viewclass": "OneLineListItem",
                  "on_release": lambda x=m: self._select_glossary_mode(x)} for m in GLOSSARY_MODES]
        self._glossary_menu = MDDropdownMenu(caller=caller, items=items, width_mult=4)
        self._glossary_menu.open()

    def _select_glossary_mode(self, mode):
        self.glossary_mode = mode
        # Update legacy compat
        self.auto_glossary = mode.lower() not in ('off', 'no_glossary', 'no glossary')
        if self._glossary_menu:
            self._glossary_menu.dismiss()

    def open_multikey_manager(self):
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

    # ── Translation ──

    def run_translation(self):
        if self.is_translating:
            self._stop_translation()
            return

        if not self.selected_file or not os.path.isfile(self.selected_file):
            toast("Please select a file first")
            return
        if not self.api_key and not self.use_multi_keys:
            toast("Please enter an API key")
            return

        self._save_settings()
        self.is_translating = True
        self._stop_requested = False
        self._log_lines = []
        self.log_text = ''

        try:
            self.ids.run_fab.icon = "stop"
        except Exception:
            pass

        self._show_log_panel()
        self._append_log("[>>] Setting up environment...")
        self._setup_environment()

        self._translation_thread = threading.Thread(target=self._translation_worker, daemon=True)
        self._translation_thread.start()

    def _stop_translation(self):
        self._stop_requested = True
        os.environ['GRACEFUL_STOP'] = '1'
        self._append_log("[STOP] Stop requested...")
        try:
            self.ids.run_fab.icon = "clock-outline"
        except Exception:
            pass

    def _setup_environment(self):
        from android_env_propagator import set_all_env_vars, set_per_run_env_vars, set_input_file_env
        from android_file_utils import get_output_dir

        cfg = self.app.config_data.copy() if self.app else {}
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
        cfg['auto_glossary_mode'] = self.glossary_mode.lower().replace(' ', '_')
        cfg['output_language'] = self.output_language

        set_all_env_vars(cfg)
        set_per_run_env_vars()
        output_dir = get_output_dir(self.selected_file)
        set_input_file_env(self.selected_file, output_dir)

        self._append_log(f"[DIR] Output: {output_dir}")
        self._append_log(f"[AI] Model: {self.model_name}")
        self._append_log(f"[T] Temp: {self.temperature}")

    def _translation_worker(self):
        from android_notification import notify_translation_progress, notify_translation_complete, cancel_progress_notification
        Clock.schedule_once(lambda dt: self._append_log("[OK] Translation started"))

        try:
            import builtins
            original_print = builtins.print

            def captured_print(*args, **kwargs):
                text = ' '.join(str(a) for a in args)
                Clock.schedule_once(lambda dt, t=text: self._append_log(t))
                original_print(*args, **kwargs)

            builtins.print = captured_print
            try:
                import importlib
                if 'TransateKRtoEN' in sys.modules:
                    importlib.reload(sys.modules['TransateKRtoEN'])
                from TransateKRtoEN import main as translate_main

                notify_translation_progress("Translating", 0, 100, os.path.basename(self.selected_file))

                def _stop_check():
                    return self._stop_requested

                result = translate_main(
                    log_callback=lambda msg: Clock.schedule_once(lambda dt, m=msg: self._append_log(str(m))),
                    stop_callback=_stop_check,
                )
                self._append_log(f"[DONE] Complete: {result}")
                cancel_progress_notification()
                notify_translation_complete("Translation Complete", f"{os.path.basename(self.selected_file)} translated")
            finally:
                builtins.print = original_print

        except Exception as e:
            import traceback
            # Capture eagerly — Python deletes `e` after the except block exits
            err_msg = f"[ERR] Error: {e}\n{traceback.format_exc()}"
            Clock.schedule_once(lambda dt, m=err_msg: self._append_log(m))
            try:
                from android_notification import cancel_progress_notification
                cancel_progress_notification()
            except Exception:
                pass
        finally:
            Clock.schedule_once(lambda dt: self._translation_finished())

    def _translation_finished(self):
        self.is_translating = False
        self._stop_requested = False
        try:
            self.ids.run_fab.icon = "play"
        except Exception:
            pass

    # ── Log ──

    def _append_log(self, text):
        self._log_lines.append(text)
        if len(self._log_lines) > 200:
            self._log_lines = self._log_lines[-200:]
        self.log_text = '\n'.join(self._log_lines)
        try:
            Clock.schedule_once(lambda dt: setattr(self.ids.log_scroll, 'scroll_y', 0), 0.1)
        except Exception:
            pass

    def _show_log_panel(self):
        panel = self.ids.log_panel
        Animation(height=dp(250), opacity=1, d=0.3).start(panel)

    def _hide_log_panel(self):
        panel = self.ids.log_panel
        Animation(height=0, opacity=0, d=0.2).start(panel)
