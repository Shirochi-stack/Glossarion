# multikey_screen.py
"""
AndroidMultiKeyGUI — Multi-API key manager with rotation support.
KivyMD 1.2.0 compatible.
"""

import os
import json
import logging

from kivy.properties import ObjectProperty, ListProperty, NumericProperty, BooleanProperty, StringProperty
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout

from kivymd.uix.screen import MDScreen
from kivymd.uix.list import TwoLineListItem, ThreeLineListItem
from kivymd.uix.card import MDCard
from kivymd.uix.button import MDFlatButton, MDRaisedButton, MDFloatingActionButton, MDIconButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.textfield import MDTextField
from kivymd.uix.selectioncontrol import MDSwitch
from kivymd.uix.label import MDLabel
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.menu import MDDropdownMenu

logger = logging.getLogger(__name__)

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

KV = '''
<MultiKeyScreen>:
    BoxLayout:
        orientation: 'vertical'

        MDTopAppBar:
            title: "API Key Manager"
            right_action_items: [["help-circle-outline", lambda x: root.show_help()]]
            md_bg_color: app.theme_cls.primary_color
            elevation: 2

        # Rotation card
        MDCard:
            size_hint: 1, None
            height: dp(80)
            padding: dp(16)
            elevation: 2
            md_bg_color: app.theme_cls.bg_dark

            BoxLayout:
                spacing: dp(16)

                BoxLayout:
                    orientation: 'vertical'
                    size_hint_x: 0.5

                    MDLabel:
                        text: "Key Rotation"
                        font_style: "Subtitle1"

                    MDLabel:
                        id: rotation_info
                        text: root.rotation_info_text
                        font_style: "Caption"
                        theme_text_color: "Secondary"

                BoxLayout:
                    size_hint_x: 0.5
                    spacing: dp(4)

                    MDLabel:
                        text: "Per key:"
                        halign: "right"
                        size_hint_x: 0.6

                    MDIconButton:
                        icon: "minus"
                        on_release: root.decrease_rotation()

                    MDLabel:
                        text: str(root.rotation_frequency)
                        halign: "center"
                        size_hint_x: 0.2

                    MDIconButton:
                        icon: "plus"
                        on_release: root.increase_rotation()

        # Multi-key toggle
        BoxLayout:
            size_hint_y: None
            height: dp(48)
            padding: [dp(16), 0]

            MDLabel:
                text: "Enable Multi-Key Rotation"
                size_hint_x: 0.7

            MDRaisedButton:
                id: multi_key_btn
                text: "No"
                on_release: root.toggle_multi_key()
                size_hint_x: 0.3
                md_bg_color: (0.3, 0.3, 0.3, 1)

        ScrollView:
            MDList:
                id: key_list
                spacing: dp(4)
                padding: dp(8)

        MDLabel:
            id: empty_label
            text: "No API keys added.\\nTap + to add your first key."
            halign: "center"
            theme_text_color: "Hint"
            font_style: "Body1"
            opacity: 0

    MDFloatingActionButton:
        icon: "plus"
        pos_hint: {"right": 0.95, "y": 0.02}
        elevation: 4
        on_release: root.show_add_key_dialog()
'''


class MultiKeyScreen(MDScreen):
    """Multi-API key manager."""

    app = ObjectProperty(None, allownone=True)
    api_keys = ListProperty([])
    rotation_frequency = NumericProperty(1)
    multi_key_enabled = BooleanProperty(False)
    rotation_info_text = StringProperty("0 keys · 0 active")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Builder.load_string(KV)
        self._add_dialog = None

    def on_enter(self, *args):
        self._load_keys()
        self._refresh_list()

    def _load_keys(self):
        if not self.app:
            return
        cfg = self.app.config_data
        self.api_keys = cfg.get('multi_api_keys', [])
        self.rotation_frequency = cfg.get('rotation_frequency', 1)
        self.multi_key_enabled = cfg.get('use_multi_api_keys', False)
        
        from kivy.clock import Clock
        Clock.schedule_once(lambda dt: self._update_toggle_btn(self.multi_key_enabled), 0)
        self._update_info()

    def _save_keys(self):
        if not self.app:
            return
        self.app.config_data['multi_api_keys'] = self.api_keys
        self.app.config_data['rotation_frequency'] = self.rotation_frequency
        self.app.config_data['use_multi_api_keys'] = self.multi_key_enabled
        from android_config import save_config
        save_config(self.app.config_data)
        self._sync_translation_state()

    def _sync_translation_state(self):
        """Push multi-key state to Translation screen when it is available."""
        if not self.app:
            return
        try:
            ts = self.app.root.ids.screen_manager.get_screen('translation')
            ts.use_multi_keys = self.multi_key_enabled
            if hasattr(ts, '_refresh_multi_key_status'):
                ts._refresh_multi_key_status()
        except Exception:
            pass

    def _update_info(self):
        total = len(self.api_keys)
        active = sum(1 for k in self.api_keys if k.get('enabled', True))
        self.rotation_info_text = f"{total} keys · {active} active"

    def _refresh_list(self):
        try:
            key_list = self.ids.key_list
            key_list.clear_widgets()
            empty_label = self.ids.empty_label
            if not self.api_keys:
                empty_label.opacity = 1
                return
            empty_label.opacity = 0
            for i, key_data in enumerate(self.api_keys):
                item = self._create_key_item(i, key_data)
                key_list.add_widget(item)
        except Exception as e:
            logger.error(f"Key list refresh error: {e}")

    def _create_key_item(self, index, key_data):
        api_key = key_data.get('api_key', '')
        masked = f"···{api_key[-4:]}" if len(api_key) >= 4 else "···"
        enabled = key_data.get('enabled', True)
        model = key_data.get('model', '')
        usage = key_data.get('usage_count', 0)
        cooldown = key_data.get('cooldown', 0)

        line2 = f"Model: {model}" if model else "No model set"
        line3 = f"{'[ON] Enabled' if enabled else '[OFF] Disabled'} · Used: {usage}x · Cooldown: {cooldown}s"

        item = ThreeLineListItem(
            text=f"Key #{index + 1}: {masked}",
            secondary_text=line2,
            tertiary_text=line3,
            on_release=lambda x, idx=index: self._edit_key(idx),
        )
        return item

    def show_add_key_dialog(self):
        if len(self.api_keys) >= 20:
            Snackbar(text="Maximum 20 keys allowed").open()
            return

        content = BoxLayout(orientation='vertical', spacing=dp(12), size_hint_y=None, height=dp(270))
        api_field = MDTextField(hint_text="API Key")
        model_field = MDTextField(hint_text="Model (optional)", size_hint_x=0.82)
        model_btn = MDIconButton(icon="chevron-down", size_hint_x=0.18)
        model_row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        model_row.add_widget(model_field)
        model_row.add_widget(model_btn)
        endpoint_field = MDTextField(hint_text="Custom Endpoint (optional)")
        cooldown_field = MDTextField(hint_text="Cooldown seconds (0)", input_filter='int')
        content.add_widget(api_field)
        content.add_widget(model_row)
        content.add_widget(endpoint_field)
        content.add_widget(cooldown_field)

        model_menu = None

        def _select_model(model_name):
            nonlocal model_menu
            model_field.text = model_name
            if model_menu:
                model_menu.dismiss()
                model_menu = None

        def _show_model_menu(*args):
            nonlocal model_menu
            items = [{
                "text": m,
                "viewclass": "OneLineListItem",
                "height": dp(48),
                "on_release": (lambda x=m: _select_model(x)),
            } for m in DEFAULT_MODELS]
            model_menu = MDDropdownMenu(
                caller=model_btn,
                items=items,
                width_mult=5,
                position="center",
                max_height=dp(240),
            )
            model_menu.open()

        model_btn.bind(on_release=_show_model_menu)

        def add_key(*args):
            key = api_field.text.strip()
            if not key:
                return
            self.api_keys.append({
                'api_key': key,
                'model': model_field.text.strip(),
                'endpoint': endpoint_field.text.strip(),
                'cooldown': int(cooldown_field.text or '0'),
                'enabled': True,
                'usage_count': 0,
            })
            self._save_keys()
            self._refresh_list()
            self._update_info()
            self._add_dialog.dismiss()

        self._add_dialog = MDDialog(
            title="Add API Key",
            type="custom",
            content_cls=content,
            buttons=[
                MDFlatButton(text="Cancel", on_release=lambda x: self._add_dialog.dismiss()),
                MDRaisedButton(text="Add", on_release=add_key),
            ],
        )
        self._add_dialog.open()

    def _edit_key(self, index):
        if not (0 <= index < len(self.api_keys)):
            return
        kd = self.api_keys[index]

        content = BoxLayout(orientation='vertical', spacing=dp(12), size_hint_y=None, height=dp(308))
        api_field = MDTextField(hint_text="API Key", text=kd.get('api_key', ''))
        model_field = MDTextField(hint_text="Model", text=kd.get('model', ''), size_hint_x=0.82)
        model_btn = MDIconButton(icon="chevron-down", size_hint_x=0.18)
        model_row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        model_row.add_widget(model_field)
        model_row.add_widget(model_btn)
        endpoint_field = MDTextField(hint_text="Endpoint", text=kd.get('endpoint', ''))
        cooldown_field = MDTextField(hint_text="Cooldown", text=str(kd.get('cooldown', 0)), input_filter='int')

        enabled_row = BoxLayout(size_hint_y=None, height=dp(40))
        enabled_row.add_widget(MDLabel(text="Enabled", size_hint_x=0.7))
        switch = MDSwitch(active=kd.get('enabled', True))
        enabled_row.add_widget(switch)

        content.add_widget(api_field)
        content.add_widget(model_row)
        content.add_widget(endpoint_field)
        content.add_widget(cooldown_field)
        content.add_widget(enabled_row)

        model_menu = None

        def _select_model(model_name):
            nonlocal model_menu
            model_field.text = model_name
            if model_menu:
                model_menu.dismiss()
                model_menu = None

        def _show_model_menu(*args):
            nonlocal model_menu
            items = [{
                "text": m,
                "viewclass": "OneLineListItem",
                "height": dp(48),
                "on_release": (lambda x=m: _select_model(x)),
            } for m in DEFAULT_MODELS]
            model_menu = MDDropdownMenu(
                caller=model_btn,
                items=items,
                width_mult=5,
                position="center",
                max_height=dp(240),
            )
            model_menu.open()

        model_btn.bind(on_release=_show_model_menu)

        def save_edit(*args):
            self.api_keys[index]['api_key'] = api_field.text.strip()
            self.api_keys[index]['model'] = model_field.text.strip()
            self.api_keys[index]['endpoint'] = endpoint_field.text.strip()
            self.api_keys[index]['cooldown'] = int(cooldown_field.text or '0')
            self.api_keys[index]['enabled'] = switch.active
            self._save_keys()
            self._refresh_list()
            self._update_info()
            edit_dialog.dismiss()

        def delete_key(*args):
            self.api_keys.pop(index)
            self._save_keys()
            self._refresh_list()
            self._update_info()
            edit_dialog.dismiss()

        edit_dialog = MDDialog(
            title=f"Edit Key #{index + 1}",
            type="custom",
            content_cls=content,
            buttons=[
                MDFlatButton(text="Delete", text_color=(1, 0.3, 0.3, 1), on_release=delete_key),
                MDFlatButton(text="Cancel", on_release=lambda x: edit_dialog.dismiss()),
                MDRaisedButton(text="Save", on_release=save_edit),
            ],
        )
        edit_dialog.open()

    def toggle_multi_key(self):
        self.multi_key_enabled = not self.multi_key_enabled
        self._update_toggle_btn(self.multi_key_enabled)
        self._save_keys()

    def _update_toggle_btn(self, value):
        try:
            btn = self.ids.multi_key_btn
            btn.text = "Yes" if value else "No"
            btn.md_bg_color = (0.2, 0.6, 0.3, 1) if value else (0.3, 0.3, 0.3, 1)
        except (KeyError, AttributeError):
            pass

    def increase_rotation(self):
        if self.rotation_frequency < 100:
            self.rotation_frequency += 1
            self._save_keys()

    def decrease_rotation(self):
        if self.rotation_frequency > 1:
            self.rotation_frequency -= 1
            self._save_keys()

    def show_help(self):
        dialog = MDDialog(
            title="How Key Rotation Works",
            text=(
                "API Key Rotation distributes calls across multiple keys "
                "to avoid rate limits.\n\n"
                "• Add up to 20 API keys\n"
                "• 'Uses per key' controls rotations\n"
                "• Disable keys temporarily\n"
                "• Set cooldown periods\n\n"
                "Keys cycle top to bottom, then restart."
            ),
            buttons=[
                MDFlatButton(text="Got it", on_release=lambda x: dialog.dismiss()),
            ],
        )
        dialog.open()
