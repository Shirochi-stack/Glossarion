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
from halgakos_fab import HalgakosFAB
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

# Import model catalog — fall back to a minimal list if unavailable
try:
    from model_options import get_model_options
    DEFAULT_MODELS = get_model_options()
except ImportError:
    DEFAULT_MODELS = [
        'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash',
        'gpt-5.4', 'gpt-5.2', 'gpt-5-mini',
        'claude-sonnet-4-20250514', 'claude-haiku-4-5-20251001',
        'deepseek-chat',
        'authgpt/gpt-5.2', 'authgpt/gemini-2.5-flash',
        'antigravity/claude-sonnet-4-6', 'antigravity/gemini-2.5-flash',
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
                    height: dp(170)
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
                                size_hint_x: 0.5

                            MDLabel:
                                id: multi_key_status
                                text: root.multi_key_status_text
                                halign: "right"
                                theme_text_color: "Secondary"
                                size_hint_x: 0.2

                            MDRaisedButton:
                                text: "Manage"
                                on_release: root.open_multikey_manager()
                                size_hint_x: 0.3

                        BoxLayout:
                            size_hint_y: None
                            height: dp(36)
                            spacing: dp(8)

                            MDRaisedButton:
                                id: authgpt_btn
                                text: root.authgpt_status_text
                                font_size: sp(12)
                                on_release: root.authgpt_login_toggle()
                                size_hint_x: 0.75

                            MDIconButton:
                                icon: "logout"
                                on_release: root.authgpt_logout()
                                size_hint_x: 0.25
                                disabled: not root.authgpt_logged_in

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

                        MDRaisedButton:
                            id: batch_btn
                            text: "No"
                            on_release: root._toggle_batch()
                            size_hint_x: 0.2
                            md_bg_color: (0.3, 0.3, 0.3, 1)

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

                # ── Reader: Enable Thinking ──
                MDCard:
                    size_hint: 1, None
                    height: dp(72)
                    padding: dp(16)
                    elevation: 1

                    BoxLayout:
                        spacing: dp(12)

                        MDLabel:
                            text: "Thinking"
                            size_hint_x: 0.6

                        MDRaisedButton:
                            id: thinking_btn
                            text: "No"
                            on_release: root._toggle_thinking()
                            size_hint_x: 0.4
                            md_bg_color: (0.3, 0.3, 0.3, 1)

                # ── Reader: Load Glossary ──
                MDCard:
                    size_hint: 1, None
                    height: dp(90)
                    padding: dp(16)
                    elevation: 1

                    BoxLayout:
                        orientation: 'vertical'
                        spacing: dp(4)

                        BoxLayout:
                            spacing: dp(12)

                            MDLabel:
                                text: "Glossary (CSV)"
                                size_hint_x: 0.6

                            MDRaisedButton:
                                text: "Browse"
                                size_hint_x: 0.4
                                on_release: root._pick_glossary()

                        MDLabel:
                            id: glossary_path_label
                            text: root.reader_glossary_path if root.reader_glossary_path else "No glossary loaded"
                            font_style: "Caption"
                            theme_text_color: "Hint"

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

    # Run FAB with spinning Halgakos
    HalgakosFAB:
        id: run_fab
        pos_hint: {"right": 0.95, "y": 0.02}
        on_release: root.run_translation()
'''


class TranslationScreen(MDScreen):
    """Translation settings and execution."""

    app = ObjectProperty(None, allownone=True)

    selected_file = StringProperty('')
    selected_file_display = StringProperty('No file selected')
    model_name = StringProperty('authgpt/gpt-5.2')
    api_key = StringProperty('')
    authgpt_logged_in = BooleanProperty(False)
    authgpt_status_text = StringProperty('AuthGPT: Login')
    use_multi_keys = BooleanProperty(False)
    multi_key_status_text = StringProperty('Off')
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
    reader_enable_thinking = BooleanProperty(False)
    reader_glossary_path = StringProperty('')

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
        self.reader_enable_thinking = cfg.get('reader_enable_thinking', False)
        self.reader_glossary_path = cfg.get('reader_glossary_path', '')
        self._refresh_multi_key_status()

        # Load system prompt for active profile
        self.system_prompt = self._get_prompt_for_profile(self.active_profile)

        # Check if AuthGPT token already exists
        self._check_authgpt_status()

        # Sync toggle buttons AFTER KV tree is ready
        from kivy.clock import Clock
        Clock.schedule_once(lambda dt: self._sync_buttons(), 0)

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
        cfg['reader_enable_thinking'] = self.reader_enable_thinking
        cfg['reader_glossary_path'] = self.reader_glossary_path

        # Save prompt to profile
        profiles = cfg.get('prompt_profiles', {})
        profiles[self.active_profile] = self.system_prompt
        cfg['prompt_profiles'] = profiles

        from android_config import save_config
        save_config(cfg)

    # ── Glossary File Picker ──

    def _pick_glossary(self):
        """Open the native file picker to select a glossary CSV file."""
        try:
            from android_file_utils import open_native_file_picker, REQUEST_CODE_OPEN_GLOSSARY
            open_native_file_picker(
                callback=self._on_glossary_picked,
                extensions=['.csv'],
                request_code=REQUEST_CODE_OPEN_GLOSSARY,
            )
        except Exception as e:
            logger.error(f"Glossary file picker error: {e}")
            self._append_log(f"[WARN] Glossary picker error: {e}")

    def _on_glossary_picked(self, path):
        """Handle glossary file selection from native picker."""
        if path and os.path.isfile(path):
            self.reader_glossary_path = path
            try:
                from kivymd.toast import toast
                toast(f"Loaded: {os.path.basename(path)}")
            except Exception:
                pass

    # ── AuthGPT / AuthGem OAuth ──

    def _get_auth_account_id(self):
        """Extract multi-account ID from model (e.g. authgpt2/ -> 2)"""
        import re
        m = re.match(r'^authgpt(\d{1,4})/', self.model_name.lower())
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        return 0

    def _get_auth_system_name(self):
        lower = self.model_name.lower()
        if lower.startswith('authgem'):
            return "AuthGem"
        acct_id = self._get_auth_account_id()
        if acct_id > 0:
            return f"ChatGPT {acct_id}"
        return "ChatGPT"

    def _update_auth_btn_label(self):
        sys_name = self._get_auth_system_name()
        system = "authgem" if "AuthGem" in sys_name else "authgpt"
        if self.authgpt_logged_in:
            from android_oauth import get_account_email
            email = get_account_email(self._get_auth_account_id(), system=system)
            self.authgpt_status_text = f"🔓 {sys_name}: {email}" if email else f"🔓 {sys_name}: Logged In"
        else:
            self.authgpt_status_text = f"🔒 {sys_name}: Login"

    def _check_authgpt_status(self):
        """Check if Auth token exists and update UI accordingly."""
        sys_name = self._get_auth_system_name()
        system = "authgem" if "AuthGem" in sys_name else "authgpt"
        try:
            from android_oauth import has_valid_token
            if has_valid_token(self._get_auth_account_id(), system=system):
                self.authgpt_logged_in = True
            else:
                self.authgpt_logged_in = False
            self._update_auth_btn_label()
        except Exception:
            self.authgpt_logged_in = False
            self._update_auth_btn_label()

    def authgpt_login_toggle(self):
        """Start the OAuth flow."""
        sys_name = self._get_auth_system_name()
        if self.authgpt_logged_in:
            from kivymd.toast import toast
            toast(f"Already logged in as {self.authgpt_status_text.replace('🔓 ', '')}")
            return

        sys_name = self._get_auth_system_name()
        system = "authgem" if "AuthGem" in sys_name else "authgpt"
        try:
            from android_oauth import start_oauth_flow
            start_oauth_flow(
                on_success=self._on_authgpt_success,
                on_error=self._on_authgpt_error,
                account_id=self._get_auth_account_id(),
                system=system
            )
        except Exception as e:
            self._update_auth_btn_label()
            self._append_log(f"[ERR] {sys_name} login failed: {e}")

    def _on_authgpt_success(self, tokens):
        self.authgpt_logged_in = True
        self._update_auth_btn_label()
        self._append_log(f"[OK] {self._get_auth_system_name()} login successful!")
        from kivymd.toast import toast
        toast(f"{self._get_auth_system_name()}: Logged in!")

    def _on_authgpt_error(self, message):
        self.authgpt_logged_in = False
        self._update_auth_btn_label()
        self._append_log(f"[ERR] {self._get_auth_system_name()}: {message}")
        from kivymd.toast import toast
        # Show the actual error message, not just "login failed"
        short_msg = str(message)[:120] if message else "Unknown error"
        toast(f"{self._get_auth_system_name()}: {short_msg}")

    def authgpt_logout(self):
        """Clear stored OAuth tokens."""
        sys_name = self._get_auth_system_name()
        system = "authgem" if "AuthGem" in sys_name else "authgpt"
        try:
            from android_oauth import logout
            logout(self._get_auth_account_id(), system=system)
        except Exception:
            pass
        self.authgpt_logged_in = False
        self._update_auth_btn_label()
        from kivymd.toast import toast
        toast(f"{self._get_auth_system_name()}: Logged out")

    # ── UI ──

    def pick_file(self):
        """Open the native Android file picker (SAF) for input file selection."""
        try:
            from android_file_utils import open_native_file_picker, REQUEST_CODE_OPEN_FILE
            open_native_file_picker(
                callback=self._on_native_file_picked,
                extensions=['.epub', '.txt', '.pdf'],
                request_code=REQUEST_CODE_OPEN_FILE,
            )
        except Exception as e:
            self._append_log(f"[WARN] File picker error: {e}")

    def _on_native_file_picked(self, path):
        """Handle file selection from native picker."""
        if path and os.path.isfile(path):
            self.selected_file = path
            self.selected_file_display = os.path.basename(path)
            try:
                from kivymd.toast import toast
                toast(f"Selected: {os.path.basename(path)}")
            except Exception:
                pass
        elif path is None:
            # User cancelled — no action needed
            pass
        else:
            self._append_log(f"[WARN] Selected file not found: {path}")

    def show_model_menu(self, caller):
        items = [{"text": m, "viewclass": "OneLineListItem",
                  "height": dp(48), "on_release": lambda x=m: self._select_model(x)} for m in DEFAULT_MODELS]
        self._model_menu = MDDropdownMenu(caller=caller, items=items, width_mult=5, position="center", max_height=dp(240))
        self._model_menu.open()

    def _select_model(self, model):
        self.model_name = model
        self._update_auth_btn_label()
        if self._model_menu:
            self._model_menu.dismiss()

    def show_profile_menu(self, caller):
        profiles = list(DEFAULT_PROFILE_NAMES)
        if self.app:
            for p in self.app.config_data.get('prompt_profiles', {}).keys():
                if p not in profiles:
                    profiles.append(p)
        items = [{"text": p, "viewclass": "OneLineListItem",
                  "height": dp(48), "on_release": lambda x=p: self._select_profile(x)} for p in profiles]
        self._profile_menu = MDDropdownMenu(caller=caller, items=items, width_mult=5, position="center", max_height=dp(240))
        self._profile_menu.open()

    def _select_profile(self, profile):
        self.active_profile = profile
        if self._profile_menu:
            self._profile_menu.dismiss()
        self.system_prompt = self._get_prompt_for_profile(profile)

    def toggle_prompt(self):
        self.prompt_expanded = not self.prompt_expanded

    def show_glossary_menu(self, caller):
        items = [{"text": m, "viewclass": "OneLineListItem",
                  "height": dp(48), "on_release": lambda x=m: self._select_glossary_mode(x)} for m in GLOSSARY_MODES]
        self._glossary_menu = MDDropdownMenu(caller=caller, items=items, width_mult=4, position="center", max_height=dp(200))
        self._glossary_menu.open()

    def _select_glossary_mode(self, mode):
        self.glossary_mode = mode
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

    # ── Yes/No button toggle handlers ──

    def _sync_buttons(self):
        """Push property values into Yes/No buttons (called once after KV build)."""
        self._update_toggle_btn('batch_btn', self.batch_enabled)
        self._update_toggle_btn('thinking_btn', self.reader_enable_thinking)
        self._update_toggle_btn('multi_key_btn', self.use_multi_keys)

    def _update_toggle_btn(self, btn_id, value):
        """Update a toggle button's text and color."""
        try:
            btn = self.ids[btn_id]
            btn.text = "Yes" if value else "No"
            btn.md_bg_color = (0.2, 0.6, 0.3, 1) if value else (0.3, 0.3, 0.3, 1)
        except (KeyError, AttributeError):
            pass

    def _toggle_batch(self):
        """Toggle batch on/off."""
        self.batch_enabled = not self.batch_enabled
        self._update_toggle_btn('batch_btn', self.batch_enabled)

    def _toggle_thinking(self):
        """Toggle thinking on/off."""
        self.reader_enable_thinking = not self.reader_enable_thinking
        self._update_toggle_btn('thinking_btn', self.reader_enable_thinking)

    def _toggle_multi_key(self):
        """Toggle multi-key mode on/off."""
        self.use_multi_keys = not self.use_multi_keys
        self._update_toggle_btn('multi_key_btn', self.use_multi_keys)
        
        # Keep Multi-Key manager screen toggle in sync if it exists
        try:
            mkm = self.app.root.ids.screen_manager.get_screen('multikey')
            mkm.multi_key_enabled = self.use_multi_keys
            if hasattr(mkm, '_update_toggle_btn'):
                mkm._update_toggle_btn(self.use_multi_keys)
        except:
            pass

    # ── Translation ──

    def run_translation(self):
        if self.is_translating:
            self._stop_translation()
            return

        if not self.selected_file or not os.path.isfile(self.selected_file):
            toast("Please select a file first")
            return
        # authgpt/ and antigravity/ prefixes don't need an API key
        model_lower = self.model_name.lower()
        needs_api_key = not (model_lower.startswith('authgpt/') or
                             model_lower.startswith('antigravity/') or
                             model_lower == 'google-translate-free')
        if needs_api_key and not self.api_key and not self.use_multi_keys:
            toast("Please enter an API key")
            return

        self._save_settings()
        self.is_translating = True
        self._reset_stop_flags()
        self._log_lines = []
        self.log_text = ''

        try:
            self.ids.run_fab.start_spinning()
        except Exception:
            pass

        self._show_log_panel()
        self._append_log("[>>] Setting up environment...")
        self._setup_environment()

        self._translation_thread = threading.Thread(target=self._translation_worker, daemon=True)
        self._translation_thread.start()

    def _stop_translation(self):
        self._stop_requested = True

        # Match desktop translator_gui.py stop flag propagation
        os.environ['TRANSLATION_CANCELLED'] = '1'
        os.environ['GRACEFUL_STOP'] = '1'
        os.environ['WAIT_FOR_CHUNKS'] = '0'

        # Set stop flag on TransateKRtoEN module
        try:
            import TransateKRtoEN
            if hasattr(TransateKRtoEN, 'set_stop_flag'):
                TransateKRtoEN.set_stop_flag(True)
        except Exception:
            pass

        # Set stop flag on unified_api_client
        try:
            import unified_api_client
            if hasattr(unified_api_client, 'set_stop_flag'):
                unified_api_client.set_stop_flag(True)
            if hasattr(unified_api_client, 'UnifiedClient'):
                unified_api_client.UnifiedClient._global_cancelled = True
            if hasattr(unified_api_client, 'hard_cancel_all'):
                unified_api_client.hard_cancel_all()
        except Exception:
            pass

        self._append_log("[STOP] Stop requested — waiting for in-flight calls to finish...")
        try:
            self.ids.run_fab.show_stopping()
        except Exception:
            pass

    def _reset_stop_flags(self):
        """Reset all stop flags before starting a new translation."""
        self._stop_requested = False
        os.environ.pop('TRANSLATION_CANCELLED', None)
        os.environ.pop('GRACEFUL_STOP', None)
        os.environ['WAIT_FOR_CHUNKS'] = '0'

        try:
            import TransateKRtoEN
            if hasattr(TransateKRtoEN, 'set_stop_flag'):
                TransateKRtoEN.set_stop_flag(False)
            if hasattr(TransateKRtoEN, '_stop_requested'):
                TransateKRtoEN._stop_requested = False
            if hasattr(TransateKRtoEN, 'STOP_LOGGED'):
                TransateKRtoEN.STOP_LOGGED = False
        except Exception:
            pass

        try:
            import unified_api_client
            if hasattr(unified_api_client, 'set_stop_flag'):
                unified_api_client.set_stop_flag(False)
            if hasattr(unified_api_client, 'UnifiedClient'):
                unified_api_client.UnifiedClient._global_cancelled = False
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
            import importlib

            # Pre-register Android stubs into sys.modules BEFORE importing
            # TransateKRtoEN, which does top-level `import tiktoken`, etc.
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
            for mod_name, stub_name in _stub_map.items():
                if mod_name not in sys.modules:
                    try:
                        importlib.import_module(mod_name)
                    except ImportError:
                        try:
                            stub = importlib.import_module(stub_name)
                            sys.modules[mod_name] = stub
                        except Exception:
                            pass

            if 'TransateKRtoEN' in sys.modules:
                importlib.reload(sys.modules['TransateKRtoEN'])

            try:
                from TransateKRtoEN import main as translate_main
            except ImportError as ie:
                Clock.schedule_once(lambda dt, m=str(ie): self._append_log(f"[ERR] Cannot import translation engine: {m}"))
                return

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
            self.ids.run_fab.stop_spinning()
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
