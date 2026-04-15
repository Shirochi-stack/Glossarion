# other_settings_screen.py
"""
Android Other Settings screen.

Shows endpoint controls plus a dynamic list of config keys discovered from
other_settings.py so Android can expose the same configurable surface area.
"""

import copy
import json
import logging
import os
import re

from kivy.core.clipboard import Clipboard
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout

from kivymd.toast import toast
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.screen import MDScreen
from kivymd.uix.textfield import MDTextField

from android_config import save_config

logger = logging.getLogger(__name__)

KV = '''
<OtherSettingsScreen>:
    BoxLayout:
        orientation: 'vertical'

        MDTopAppBar:
            title: "Other Settings"
            md_bg_color: app.theme_cls.primary_color
            elevation: 2

        BoxLayout:
            size_hint_y: None
            height: dp(56)
            padding: [dp(12), dp(8)]
            spacing: dp(8)

            MDTextField:
                id: search_field
                hint_text: "Search settings key..."
                on_text: root.apply_filter(self.text)
                size_hint_x: 0.58

            MDRaisedButton:
                text: "Reload"
                size_hint_x: 0.21
                on_release: root.reload_settings()

            MDRaisedButton:
                text: "Save"
                size_hint_x: 0.21
                md_bg_color: 0.2, 0.6, 0.3, 1
                on_release: root.save_all()

        ScrollView:
            do_scroll_x: False

            BoxLayout:
                id: settings_box
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(8)
                padding: [dp(12), dp(8), dp(12), dp(16)]
'''


class OtherSettingsScreen(MDScreen):
    """Dynamic settings page with endpoint shortcuts and full config key list."""

    app = ObjectProperty(None, allownone=True)
    search_text = StringProperty('')

    _ENDPOINT_KEYS = {
        'use_custom_openai_endpoint',
        'openai_base_url',
        'use_gemini_openai_endpoint',
        'gemini_openai_endpoint',
        'force_native_anthropic',
        'anthropic_base_url',
    }

    _ENDPOINT_DEFAULTS = {
        'use_custom_openai_endpoint': False,
        'openai_base_url': '',
        'use_gemini_openai_endpoint': False,
        'gemini_openai_endpoint': 'generativelanguage.googleapis.com',
        'force_native_anthropic': False,
        'anthropic_base_url': '',
    }

    _ENDPOINT_PRESETS = {
        'openai_base_url': [
            'http://127.0.0.1:11434/v1',
            'http://127.0.0.1:1234/v1',
        ],
        'gemini_openai_endpoint': [
            'https://generativelanguage.googleapis.com/v1beta/openai/',
            'generativelanguage.googleapis.com',
        ],
        'anthropic_base_url': [
            'https://api.anthropic.com',
            'https://api.electronhub.top/anthropic',
            'https://openrouter.ai/api',
        ],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Builder.load_string(KV)
        self._initialized = False
        self._all_keys = []
        self._pending_config = {}
        self._raw_values = {}
        self._types = {}
        self._bool_values = {}
        self._endpoint_fields = {}
        self._dynamic_box = None

    def on_enter(self, *args):
        if not self.app:
            return
        if not self._initialized:
            self._init_state()
            self._build_ui()
            self._initialized = True
        else:
            self._refresh_from_app_config()
            self._sync_buffers_from_pending()
            self._build_ui()

    def reload_settings(self):
        if not self.app:
            return
        self._refresh_from_app_config()
        self._sync_buffers_from_pending()
        self._build_ui()
        toast("Reloaded settings")

    def apply_filter(self, text):
        self.search_text = (text or '').strip().lower()
        self._render_dynamic_settings()

    def _init_state(self):
        self._refresh_from_app_config()
        self._all_keys = sorted(
            set(self._discover_config_keys())
            | set(self._pending_config.keys())
            | self._ENDPOINT_KEYS
        )
        self._sync_buffers_from_pending()

    def _sync_buffers_from_pending(self):
        for key in self._all_keys:
            if key not in self._pending_config:
                self._pending_config[key] = self._default_for_key(key)
            self._types.setdefault(key, self._infer_type(key, self._pending_config.get(key)))
            if self._types[key] is bool:
                self._bool_values[key] = bool(self._pending_config.get(key, False))
            self._raw_values[key] = self._value_to_text(self._pending_config.get(key))

    def _refresh_from_app_config(self):
        cfg = copy.deepcopy(self.app.config_data if self.app else {})
        for key, value in self._ENDPOINT_DEFAULTS.items():
            cfg.setdefault(key, value)
        self._pending_config = cfg

    def _build_ui(self):
        box = self.ids.settings_box
        box.clear_widgets()
        box.add_widget(self._build_endpoint_card())

        title = MDLabel(
            text="All Other Config Keys (from other_settings.py)",
            theme_text_color="Secondary",
            size_hint_y=None,
            height=dp(28),
        )
        box.add_widget(title)

        self._dynamic_box = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=0,
            spacing=dp(8),
        )
        self._dynamic_box.bind(minimum_height=self._dynamic_box.setter('height'))
        box.add_widget(self._dynamic_box)
        self._render_dynamic_settings()

    def _build_endpoint_card(self):
        card = MDCard(
            size_hint_y=None,
            elevation=2,
            padding=dp(12),
            radius=[12, 12, 12, 12],
        )
        content = BoxLayout(
            orientation='vertical',
            spacing=dp(8),
            size_hint_y=None,
        )
        content.bind(minimum_height=content.setter('height'))
        card.add_widget(content)
        card.bind(height=lambda *_: setattr(card, 'height', content.height + dp(24)))

        content.add_widget(MDLabel(
            text="Custom API Endpoints",
            bold=True,
            size_hint_y=None,
            height=dp(26),
        ))

        self._add_endpoint_block(
            content,
            "OpenAI Custom Endpoint",
            'use_custom_openai_endpoint',
            'openai_base_url',
        )
        self._add_endpoint_block(
            content,
            "Gemini Custom Endpoint",
            'use_gemini_openai_endpoint',
            'gemini_openai_endpoint',
        )
        self._add_endpoint_block(
            content,
            "Anthropic Custom Endpoint",
            'force_native_anthropic',
            'anthropic_base_url',
        )
        return card

    def _add_endpoint_block(self, parent, title, toggle_key, url_key):
        header = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8))
        header.add_widget(MDLabel(text=title, size_hint_x=0.7))
        btn = MDRaisedButton(size_hint_x=0.3)
        self._set_toggle_button_style(btn, self._bool_values.get(toggle_key, False))
        btn.bind(on_release=lambda *_a, k=toggle_key, b=btn: self._toggle_bool_key(k, b))
        header.add_widget(btn)
        parent.add_widget(header)

        field = MDTextField(
            hint_text="Endpoint URL / host",
            text=self._raw_values.get(url_key, ''),
            multiline=False,
            size_hint_y=None,
            height=dp(56),
        )
        field.bind(text=lambda _i, text, k=url_key: self._on_text_changed(k, text))
        field.bind(on_touch_down=lambda inst, touch, k=url_key: self._endpoint_field_touch(inst, touch, k))
        parent.add_widget(field)
        self._endpoint_fields[url_key] = field

        hint = MDLabel(
            text="Double-tap field: copy current value (or paste clipboard if empty)",
            theme_text_color="Secondary",
            font_style="Caption",
            size_hint_y=None,
            height=dp(20),
        )
        parent.add_widget(hint)

        shortcuts = BoxLayout(size_hint_y=None, height=dp(28), spacing=dp(6))
        shortcuts.add_widget(MDLabel(text="Shortcuts (double-tap):", size_hint_x=0.45))
        for preset in self._ENDPOINT_PRESETS.get(url_key, []):
            lbl = MDLabel(
                text=f"[u]{preset}[/u]",
                markup=True,
                theme_text_color="Custom",
                text_color=[0.2, 0.75, 1, 1],
                size_hint_x=0.9,
            )
            lbl.bind(on_touch_down=lambda inst, touch, p=preset, k=url_key: self._shortcut_touch(inst, touch, p, k))
            shortcuts.add_widget(lbl)
        parent.add_widget(shortcuts)

    def _endpoint_field_touch(self, field, touch, key):
        if not field.collide_point(*touch.pos) or not touch.is_double_tap:
            return False
        value = (field.text or '').strip()
        if value:
            Clipboard.copy(value)
            toast(f"Copied {key}")
        else:
            pasted = (Clipboard.paste() or '').strip()
            if pasted:
                field.text = pasted
                self._on_text_changed(key, pasted)
                toast(f"Pasted into {key}")
        return True

    def _shortcut_touch(self, label, touch, preset_value, key):
        if not label.collide_point(*touch.pos) or not touch.is_double_tap:
            return False
        field = self._endpoint_fields.get(key)
        if field is not None:
            field.text = preset_value
        self._on_text_changed(key, preset_value)
        Clipboard.copy(preset_value)
        toast(f"Pasted + copied {key}")
        return True

    def _render_dynamic_settings(self):
        if self._dynamic_box is None:
            return
        self._dynamic_box.clear_widgets()
        needle = self.search_text

        for key in self._all_keys:
            if key in self._ENDPOINT_KEYS:
                continue
            if needle and needle not in key.lower():
                continue
            self._dynamic_box.add_widget(self._build_setting_row(key))

    def _build_setting_row(self, key):
        t = self._types.get(key, str)
        is_complex = t in (dict, list)

        card = MDCard(
            size_hint_y=None,
            elevation=1,
            padding=dp(10),
            radius=[10, 10, 10, 10],
        )
        inner = BoxLayout(
            orientation='vertical',
            spacing=dp(6),
            size_hint_y=None,
        )
        inner.bind(minimum_height=inner.setter('height'))
        card.add_widget(inner)
        card.bind(height=lambda *_: setattr(card, 'height', inner.height + dp(18)))

        inner.add_widget(MDLabel(
            text=key,
            bold=True,
            size_hint_y=None,
            height=dp(22),
        ))

        if t is bool:
            btn = MDRaisedButton(size_hint_y=None, height=dp(38))
            self._set_toggle_button_style(btn, self._bool_values.get(key, False))
            btn.bind(on_release=lambda *_a, k=key, b=btn: self._toggle_bool_key(k, b))
            inner.add_widget(btn)
            return card

        field = MDTextField(
            text=self._raw_values.get(key, ''),
            multiline=is_complex,
            size_hint_y=None,
            height=dp(96) if is_complex else dp(56),
        )
        field.bind(text=lambda _i, text, k=key: self._on_text_changed(k, text))
        inner.add_widget(field)
        return card

    def _toggle_bool_key(self, key, button):
        current = bool(self._bool_values.get(key, False))
        new_value = not current
        self._bool_values[key] = new_value
        self._pending_config[key] = new_value
        self._raw_values[key] = 'true' if new_value else 'false'
        self._set_toggle_button_style(button, new_value)

    @staticmethod
    def _set_toggle_button_style(button, enabled):
        button.text = "Enabled" if enabled else "Disabled"
        button.md_bg_color = (0.2, 0.6, 0.3, 1) if enabled else (0.3, 0.3, 0.3, 1)

    def _on_text_changed(self, key, text):
        self._raw_values[key] = text

    def save_all(self):
        if not self.app:
            return
        new_cfg = copy.deepcopy(self._pending_config)
        errors = []

        for key in self._all_keys:
            t = self._types.get(key, str)
            if t is bool:
                new_cfg[key] = bool(self._bool_values.get(key, False))
                continue
            raw = self._raw_values.get(key, '')
            try:
                new_cfg[key] = self._coerce_value(raw, t)
            except Exception as exc:
                errors.append(f"{key}: {exc}")

        if errors:
            toast(f"Save blocked ({len(errors)} parse errors). Fix highlighted values.")
            logger.warning("Other settings parse errors: %s", "; ".join(errors[:8]))
            return

        self.app.config_data.update(new_cfg)
        save_config(self.app.config_data)
        self._pending_config = copy.deepcopy(self.app.config_data)
        toast("Saved all settings")

    @staticmethod
    def _coerce_value(raw_text, value_type):
        text = (raw_text or '').strip()
        if value_type is int:
            return int(text) if text else 0
        if value_type is float:
            return float(text) if text else 0.0
        if value_type is dict:
            if not text:
                return {}
            loaded = json.loads(text)
            if not isinstance(loaded, dict):
                raise ValueError("must be a JSON object")
            return loaded
        if value_type is list:
            if not text:
                return []
            loaded = json.loads(text)
            if not isinstance(loaded, list):
                raise ValueError("must be a JSON array")
            return loaded
        return text

    def _infer_type(self, key, value):
        if isinstance(value, bool):
            return bool
        if isinstance(value, int) and not isinstance(value, bool):
            return int
        if isinstance(value, float):
            return float
        if isinstance(value, dict):
            return dict
        if isinstance(value, list):
            return list
        if self._looks_bool_key(key):
            return bool
        return str

    def _default_for_key(self, key):
        if key in self._ENDPOINT_DEFAULTS:
            return self._ENDPOINT_DEFAULTS[key]
        if self._looks_bool_key(key):
            return False
        return ''

    @staticmethod
    def _looks_bool_key(key):
        lk = key.lower()
        prefixes = (
            'enable_', 'disable_', 'use_', 'force_', 'skip_', 'show_',
            'auto_', 'retain_', 'append_', 'deduplicate_', 'allow_',
            'save_', 'preserve_', 'optimize_', 'ignore_',
        )
        exact = {
            'contextual',
            'translation_history_rolling',
            'batch_translation',
            'token_limit_disabled',
        }
        return lk.startswith(prefixes) or lk in exact

    @staticmethod
    def _value_to_text(value):
        if value is None:
            return ''
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return ''
        return str(value)

    @staticmethod
    def _discover_config_keys():
        """Extract config keys referenced by other_settings.py."""
        here = os.path.dirname(os.path.abspath(__file__))
        src_root = os.path.dirname(here)
        target = os.path.join(src_root, 'other_settings.py')
        if not os.path.isfile(target):
            return []

        try:
            with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception:
            return []

        keys = set()
        patterns = [
            r"(?:[A-Za-z_][\w\.]*)?config\.get\(\s*['\"]([^'\"]+)['\"]",
            r"(?:[A-Za-z_][\w\.]*)?config\[\s*['\"]([^'\"]+)['\"]\s*\]",
            r"(?:[A-Za-z_][\w\.]*)?config\[\s*['\"]([^'\"]+)['\"]\s*\]\s*=",
        ]
        for pattern in patterns:
            keys.update(re.findall(pattern, text))
        return sorted(k for k in keys if k and not k.startswith('_') and len(k) < 160)
