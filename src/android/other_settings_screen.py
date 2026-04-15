# other_settings_screen.py
"""
Android Other Settings screen.

Design goals:
- Only show keys discovered from other_settings.py (+ endpoint keys)
- Avoid UI lag by debounced filtering + paged rendering
- Stable card heights to prevent label/field overlap
"""

import ast
import copy
import json
import logging
import os
import re

from kivy.clock import Clock
from kivy.core.clipboard import Clipboard
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout

from kivymd.toast import toast
from kivymd.uix.button import MDFlatButton, MDRaisedButton
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
            height: dp(64)
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
    """Dynamic settings page for other_settings.py keys."""

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

    _CARD_RADIUS = [12, 12, 12, 12]
    _CARD_BG = (0.13, 0.13, 0.17, 1)
    _SUBCARD_BG = (0.16, 0.16, 0.20, 1)

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
        self._discovered_defaults = None

        self._render_step = 36
        self._render_limit = self._render_step
        self._filtered_keys = []
        self._filter_event = None

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
        self._render_limit = self._render_step
        self._build_ui()
        toast("Reloaded settings")

    def apply_filter(self, text):
        self.search_text = (text or '').strip().lower()
        self._render_limit = self._render_step
        if self._filter_event is not None:
            self._filter_event.cancel()
        self._filter_event = Clock.schedule_once(lambda _dt: self._render_dynamic_settings(), 0.12)

    def _init_state(self):
        self._refresh_from_app_config()
        if self._discovered_defaults is None:
            self._discovered_defaults = self._discover_config_defaults()
        self._all_keys = sorted(set(self._discovered_defaults.keys()) | self._ENDPOINT_KEYS)
        self._sync_buffers_from_pending()

    def _sync_buffers_from_pending(self):
        for key in self._all_keys:
            default = self._discovered_defaults.get(key, self._default_for_key(key))
            if key not in self._pending_config:
                self._pending_config[key] = default

            value = self._pending_config.get(key, default)
            self._types[key] = self._infer_type(key, value, default)
            if self._types[key] is bool:
                b = bool(value)
                self._bool_values[key] = b
                self._raw_values[key] = 'true' if b else 'false'
            else:
                self._raw_values[key] = self._value_to_text(value)

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
            text="Other Settings Keys (from other_settings.py only)",
            theme_text_color="Secondary",
            size_hint_y=None,
            height=dp(28),
            font_style="Subtitle2",
            shorten=True,
            shorten_from="right",
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
            radius=self._CARD_RADIUS,
            md_bg_color=self._CARD_BG,
        )
        content = BoxLayout(
            orientation='vertical',
            spacing=dp(10),
            size_hint_y=None,
        )
        content.bind(minimum_height=content.setter('height'))
        card.add_widget(content)
        card.bind(height=lambda *_: setattr(card, 'height', content.height + dp(22)))

        content.add_widget(MDLabel(
            text="Custom API Endpoints",
            font_style="Subtitle2",
            size_hint_y=None,
            height=dp(24),
        ))
        content.add_widget(MDLabel(
            text="Double-tap field to copy, or use Copy/Paste buttons.",
            theme_text_color="Secondary",
            font_style="Caption",
            size_hint_y=None,
            height=dp(18),
        ))

        content.add_widget(self._build_endpoint_block(
            "OpenAI endpoint",
            'use_custom_openai_endpoint',
            'openai_base_url',
        ))
        content.add_widget(self._build_endpoint_block(
            "Gemini endpoint",
            'use_gemini_openai_endpoint',
            'gemini_openai_endpoint',
        ))
        content.add_widget(self._build_endpoint_block(
            "Anthropic endpoint",
            'force_native_anthropic',
            'anthropic_base_url',
        ))
        return card

    def _build_endpoint_block(self, title, toggle_key, url_key):
        block = MDCard(
            size_hint_y=None,
            elevation=0,
            padding=dp(10),
            radius=self._CARD_RADIUS,
            md_bg_color=self._SUBCARD_BG,
        )
        inner = BoxLayout(orientation='vertical', spacing=dp(8), size_hint_y=None)
        inner.bind(minimum_height=inner.setter('height'))
        block.add_widget(inner)
        block.bind(height=lambda *_: setattr(block, 'height', inner.height + dp(16)))

        header = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8))
        header.add_widget(MDLabel(
            text=title,
            size_hint_x=0.64,
            font_style="Subtitle2",
            shorten=True,
            shorten_from="right",
        ))
        btn = MDRaisedButton(size_hint_x=0.36)
        self._set_toggle_button_style(btn, self._bool_values.get(toggle_key, False))
        btn.bind(on_release=lambda *_a, k=toggle_key, b=btn: self._toggle_bool_key(k, b))
        header.add_widget(btn)
        inner.add_widget(header)

        field = MDTextField(
            hint_text="Endpoint URL / host",
            text=self._raw_values.get(url_key, ''),
            multiline=False,
            size_hint_y=None,
            height=dp(56),
        )
        field.bind(text=lambda _i, text, k=url_key: self._on_text_changed(k, text))
        field.bind(on_touch_down=lambda inst, touch, k=url_key: self._endpoint_field_touch(inst, touch, k))
        inner.add_widget(field)
        self._endpoint_fields[url_key] = field

        actions = BoxLayout(size_hint_y=None, height=dp(32), spacing=dp(6))
        actions.add_widget(MDFlatButton(text="Copy", on_release=lambda *_a, k=url_key: self._copy_endpoint_value(k), size_hint_x=0.18))
        actions.add_widget(MDFlatButton(text="Paste", on_release=lambda *_a, k=url_key: self._paste_endpoint_value(k), size_hint_x=0.20))
        for idx, preset in enumerate(self._ENDPOINT_PRESETS.get(url_key, []), start=1):
            b = MDFlatButton(text=f"Preset {idx}", size_hint_x=0.26)
            b.bind(on_release=lambda *_a, k=url_key, p=preset: self._apply_shortcut(k, p))
            actions.add_widget(b)
        inner.add_widget(actions)

        return block

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

    def _copy_endpoint_value(self, key):
        field = self._endpoint_fields.get(key)
        if field is None:
            return
        value = (field.text or '').strip()
        if not value:
            toast("Field is empty")
            return
        Clipboard.copy(value)
        toast(f"Copied {key}")

    def _paste_endpoint_value(self, key):
        pasted = (Clipboard.paste() or '').strip()
        if not pasted:
            toast("Clipboard is empty")
            return
        field = self._endpoint_fields.get(key)
        if field is not None:
            field.text = pasted
        self._on_text_changed(key, pasted)
        toast(f"Pasted into {key}")

    def _apply_shortcut(self, key, preset_value):
        field = self._endpoint_fields.get(key)
        if field is not None:
            field.text = preset_value
        self._on_text_changed(key, preset_value)
        Clipboard.copy(preset_value)
        toast(f"Applied {key} preset")

    def _render_dynamic_settings(self):
        if self._dynamic_box is None:
            return
        self._dynamic_box.clear_widgets()

        needle = self.search_text
        filtered = []
        for key in self._all_keys:
            if key in self._ENDPOINT_KEYS:
                continue
            if needle and needle not in key.lower():
                continue
            filtered.append(key)
        self._filtered_keys = filtered

        visible = filtered[:self._render_limit]
        for key in visible:
            self._dynamic_box.add_widget(self._build_setting_row(key))

        remaining = len(filtered) - len(visible)
        if remaining > 0:
            load_more_btn = MDRaisedButton(
                text=f"Load more ({remaining} left)",
                size_hint_y=None,
                height=dp(40),
                md_bg_color=(0.30, 0.30, 0.34, 1),
                on_release=lambda *_a: self._load_more_rows(),
            )
            self._dynamic_box.add_widget(load_more_btn)

    def _load_more_rows(self):
        self._render_limit += self._render_step
        self._render_dynamic_settings()

    def _build_setting_row(self, key):
        t = self._types.get(key, str)
        is_bool = (t is bool)
        is_complex = t in (dict, list)
        card_h = dp(100) if is_bool else (dp(170) if is_complex else dp(122))

        card = MDCard(
            size_hint_y=None,
            height=card_h,
            elevation=1,
            padding=dp(10),
            radius=self._CARD_RADIUS,
            md_bg_color=self._CARD_BG,
        )
        inner = BoxLayout(orientation='vertical', spacing=dp(8))
        card.add_widget(inner)

        inner.add_widget(MDLabel(
            text=key,
            font_style="Body2",
            size_hint_y=None,
            height=dp(24),
            shorten=True,
            shorten_from="right",
        ))

        if is_bool:
            row = BoxLayout(size_hint_y=None, height=dp(40))
            btn = MDRaisedButton(size_hint_x=None, width=dp(140))
            self._set_toggle_button_style(btn, self._bool_values.get(key, False))
            btn.bind(on_release=lambda *_a, k=key, b=btn: self._toggle_bool_key(k, b))
            row.add_widget(btn)
            row.add_widget(BoxLayout())
            inner.add_widget(row)
            return card

        field = MDTextField(
            text=self._raw_values.get(key, ''),
            multiline=is_complex,
            size_hint_y=None,
            height=dp(104) if is_complex else dp(56),
        )
        field.bind(text=lambda _i, text, k=key: self._on_text_changed(k, text))
        inner.add_widget(field)
        return card

    def _toggle_bool_key(self, key, button):
        new_value = not bool(self._bool_values.get(key, False))
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
        new_cfg = copy.deepcopy(self.app.config_data)
        errors = []

        for key in self._all_keys:
            t = self._types.get(key, str)
            if t is bool:
                new_cfg[key] = bool(self._bool_values.get(key, False))
                continue
            raw = self._raw_values.get(key, '')
            default = self._discovered_defaults.get(key, self._default_for_key(key))
            try:
                new_cfg[key] = self._coerce_value(raw, t, default)
            except Exception as exc:
                errors.append(f"{key}: {exc}")

        if errors:
            toast(f"Save blocked ({len(errors)} parse errors)")
            logger.warning("Other settings parse errors: %s", "; ".join(errors[:8]))
            return

        self.app.config_data.update(new_cfg)
        save_config(self.app.config_data)
        self._pending_config = copy.deepcopy(self.app.config_data)
        toast("Saved all settings")

    @staticmethod
    def _coerce_value(raw_text, value_type, default):
        text = (raw_text or '').strip()
        if value_type is int:
            return int(text) if text else int(default or 0)
        if value_type is float:
            return float(text) if text else float(default or 0.0)
        if value_type is dict:
            if not text:
                return {} if not isinstance(default, dict) else default
            loaded = json.loads(text)
            if not isinstance(loaded, dict):
                raise ValueError("must be a JSON object")
            return loaded
        if value_type is list:
            if not text:
                return [] if not isinstance(default, list) else default
            loaded = json.loads(text)
            if not isinstance(loaded, list):
                raise ValueError("must be a JSON array")
            return loaded
        return text if text else ('' if default is None else str(default))

    def _infer_type(self, key, value, default):
        if key in {'use_custom_openai_endpoint', 'use_gemini_openai_endpoint', 'force_native_anthropic'}:
            return bool
        if isinstance(value, bool) or isinstance(default, bool):
            return bool
        if isinstance(value, int) and not isinstance(value, bool):
            return int
        if isinstance(default, int) and not isinstance(default, bool):
            return int
        if isinstance(value, float) or isinstance(default, float):
            return float
        if isinstance(value, dict) or isinstance(default, dict):
            return dict
        if isinstance(value, list) or isinstance(default, list):
            return list
        return str

    def _default_for_key(self, key):
        if key in self._ENDPOINT_DEFAULTS:
            return self._ENDPOINT_DEFAULTS[key]
        return ''

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

    def _discover_config_defaults(self):
        """Extract config keys + default literals from other_settings.py."""
        here = os.path.dirname(os.path.abspath(__file__))
        src_root = os.path.dirname(here)
        target = os.path.join(src_root, 'other_settings.py')
        if not os.path.isfile(target):
            return {}

        try:
            with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception:
            return {}

        defaults = {}
        get_pattern = r"(?:[A-Za-z_][\w\.]*)?config\.get\(\s*['\"]([^'\"]+)['\"]\s*(?:,\s*([^\)]*?))?\)"
        for match in re.finditer(get_pattern, text, flags=re.MULTILINE):
            key = (match.group(1) or '').strip()
            if not key or key.startswith('_'):
                continue
            default_expr = (match.group(2) or '').strip()
            if default_expr.endswith(','):
                default_expr = default_expr[:-1].strip()
            defaults.setdefault(key, self._parse_default_literal(default_expr))

        for key in re.findall(r"(?:[A-Za-z_][\w\.]*)?config\[\s*['\"]([^'\"]+)['\"]\s*\]", text):
            if key and not key.startswith('_'):
                defaults.setdefault(key, '')

        for key, value in self._ENDPOINT_DEFAULTS.items():
            defaults.setdefault(key, value)
        return defaults

    @staticmethod
    def _parse_default_literal(expr):
        if not expr:
            return ''
        simple_tokens = {'True': True, 'False': False, 'None': ''}
        if expr in simple_tokens:
            return simple_tokens[expr]
        try:
            return ast.literal_eval(expr)
        except Exception:
            if expr.startswith(("'", '"')) and expr.endswith(("'", '"')):
                return expr[1:-1]
            return ''
