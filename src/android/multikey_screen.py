# multikey_screen.py
"""
AndroidMultiKeyGUI — Multi-API key manager with rotation support.
Port of multi_api_key_manager.py's GUI logic, adapted for touch.
"""

import os
import json
import logging
import time

from kivy.properties import ObjectProperty, ListProperty, NumericProperty
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp

from kivymd.uix.screen import MDScreen
from kivymd.uix.list import (
    MDList, MDListItem, MDListItemLeadingIcon,
    MDListItemHeadlineText, MDListItemSupportingText,
    MDListItemTrailingCheckbox,
)
from kivymd.uix.card import MDCard
from kivymd.uix.button import MDButton, MDButtonText, MDFabButton, MDIconButton
from kivymd.uix.dialog import MDDialog, MDDialogHeadlineText, MDDialogContentContainer, MDDialogButtonContainer
from kivymd.uix.textfield import MDTextField
from kivymd.uix.selectioncontrol import MDSwitch
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.slider import MDSlider
from kivymd.uix.snackbar import MDSnackbar, MDSnackbarText
from kivy.uix.scrollview import ScrollView

logger = logging.getLogger(__name__)

KV = '''
<MultiKeyScreen>:
    MDBoxLayout:
        orientation: 'vertical'

        MDTopAppBar:
            MDTopAppBarTitle:
                text: "API Key Manager"

            MDTopAppBarTrailingButtonContainer:
                MDActionButton:
                    icon: "help-circle-outline"
                    on_release: root.show_help()

        # Rotation settings card
        MDCard:
            style: "elevated"
            size_hint: 1, None
            height: dp(80)
            padding: dp(16)

            MDBoxLayout:
                spacing: dp(16)

                MDBoxLayout:
                    orientation: 'vertical'
                    size_hint_x: 0.5

                    MDLabel:
                        text: "Key Rotation"
                        font_style: "Title"
                        role: "small"

                    MDLabel:
                        id: rotation_info
                        text: root.rotation_info_text
                        font_style: "Body"
                        role: "small"
                        theme_text_color: "Secondary"

                MDBoxLayout:
                    size_hint_x: 0.5
                    spacing: dp(8)

                    MDLabel:
                        text: "Uses per key:"
                        halign: "right"
                        size_hint_x: 0.6

                    MDIconButton:
                        icon: "minus"
                        on_release: root.decrease_rotation()

                    MDLabel:
                        id: rotation_count_label
                        text: str(root.rotation_frequency)
                        halign: "center"
                        size_hint_x: 0.2

                    MDIconButton:
                        icon: "plus"
                        on_release: root.increase_rotation()

        # Multi-key toggle
        MDBoxLayout:
            size_hint_y: None
            height: dp(48)
            padding: [dp(16), 0]

            MDLabel:
                text: "Enable Multi-Key Rotation"
                size_hint_x: 0.7

            MDSwitch:
                id: multi_key_switch
                active: root.multi_key_enabled
                on_active: root.toggle_multi_key(self.active)
                size_hint_x: 0.3

        # Key list
        ScrollView:
            id: key_scroll

            MDList:
                id: key_list
                spacing: "4dp"
                padding: "8dp"

        # Empty state
        MDLabel:
            id: empty_label
            text: "No API keys added.\\nTap + to add your first key."
            halign: "center"
            theme_text_color: "Hint"
            font_style: "Body"
            role: "large"
            opacity: 0

    # FAB
    MDFabButton:
        icon: "plus"
        pos_hint: {"right": 0.95, "y": 0.02}
        style: "standard"
        on_release: root.show_add_key_dialog()
'''


class MultiKeyScreen(MDScreen):
    """Multi-API key manager with rotation support."""

    app = ObjectProperty(None, allownone=True)
    api_keys = ListProperty([])
    rotation_frequency = NumericProperty(1)
    multi_key_enabled = False
    rotation_info_text = f"0 keys · 0 active"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Builder.load_string(KV)
        self._add_dialog = None

    def on_enter(self, *args):
        """Called when screen is displayed."""
        self._load_keys()
        self._refresh_list()

    def _load_keys(self):
        """Load API keys from config."""
        if not self.app:
            return
        cfg = self.app.config_data
        self.api_keys = cfg.get('multi_api_keys', [])
        self.rotation_frequency = cfg.get('rotation_frequency', 1)
        self.multi_key_enabled = cfg.get('use_multi_api_keys', False)
        self._update_info()

    def _save_keys(self):
        """Save API keys to config."""
        if not self.app:
            return
        self.app.config_data['multi_api_keys'] = self.api_keys
        self.app.config_data['rotation_frequency'] = self.rotation_frequency
        self.app.config_data['use_multi_api_keys'] = self.multi_key_enabled
        from android_config import save_config
        save_config(self.app.config_data)

    def _update_info(self):
        """Update the rotation info text."""
        total = len(self.api_keys)
        active = sum(1 for k in self.api_keys if k.get('enabled', True))
        self.rotation_info_text = f"{total} keys · {active} active"
        try:
            self.ids.rotation_info.text = self.rotation_info_text
        except Exception:
            pass

    def _refresh_list(self):
        """Refresh the key list UI."""
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
            logger.error(f"Error refreshing key list: {e}")

    def _create_key_item(self, index, key_data):
        """Create a list item for an API key."""
        api_key = key_data.get('api_key', '')
        masked = f"···{api_key[-4:]}" if len(api_key) >= 4 else "···"

        model = key_data.get('model', '')
        endpoint = key_data.get('endpoint', '')
        enabled = key_data.get('enabled', True)
        usage = key_data.get('usage_count', 0)
        cooldown = key_data.get('cooldown', 0)

        subtitle_parts = []
        if model:
            subtitle_parts.append(f"Model: {model}")
        if endpoint:
            subtitle_parts.append(f"Endpoint: {endpoint}")
        subtitle_parts.append(f"Used: {usage}×")
        if cooldown > 0:
            subtitle_parts.append(f"Cooldown: {cooldown}s")
        subtitle = " · ".join(subtitle_parts)

        # Main item card
        card = MDCard(
            style="outlined",
            size_hint=(1, None),
            height=dp(100),
            padding=dp(12),
            line_color=(0.3, 0.5, 0.9, 0.5) if enabled else (0.3, 0.3, 0.3, 0.3),
        )

        layout = MDBoxLayout(orientation='horizontal', spacing=dp(12))

        # Key info
        info_layout = MDBoxLayout(orientation='vertical', size_hint_x=0.6)
        info_layout.add_widget(MDLabel(
            text=f"Key #{index + 1}: {masked}",
            font_style="Title",
            role="small",
            bold=True if index == 0 and enabled else False,
        ))
        info_layout.add_widget(MDLabel(
            text=subtitle,
            font_style="Body",
            role="small",
            theme_text_color="Secondary",
        ))
        layout.add_widget(info_layout)

        # Controls
        controls = MDBoxLayout(
            orientation='vertical',
            size_hint_x=0.4,
            spacing=dp(4),
        )

        # Enable/disable switch
        switch_row = MDBoxLayout(size_hint_y=0.5, spacing=dp(4))
        switch_row.add_widget(MDLabel(
            text="On" if enabled else "Off",
            halign="right",
            size_hint_x=0.5,
        ))
        switch = MDSwitch(active=enabled, size_hint_x=0.5)
        switch.bind(active=lambda s, val, idx=index: self._toggle_key(idx, val))
        switch_row.add_widget(switch)
        controls.add_widget(switch_row)

        # Delete button
        btn_row = MDBoxLayout(size_hint_y=0.5)
        edit_btn = MDIconButton(icon="pencil", on_release=lambda x, idx=index: self._edit_key(idx))
        del_btn = MDIconButton(icon="delete-outline", on_release=lambda x, idx=index: self._delete_key(idx))
        btn_row.add_widget(edit_btn)
        btn_row.add_widget(del_btn)
        controls.add_widget(btn_row)

        layout.add_widget(controls)
        card.add_widget(layout)
        return card

    def show_add_key_dialog(self):
        """Show dialog to add a new API key."""
        if len(self.api_keys) >= 20:
            MDSnackbar(
                MDSnackbarText(text="Maximum 20 keys allowed"),
                y=dp(24), pos_hint={"center_x": 0.5},
            ).open()
            return

        api_field = MDTextField(hint_text="API Key")
        model_field = MDTextField(hint_text="Model (optional, e.g. gpt-4)")
        endpoint_field = MDTextField(hint_text="Custom Endpoint (optional)")
        cooldown_field = MDTextField(hint_text="Cooldown seconds (default: 0)")
        cooldown_field.input_filter = 'int'

        content = MDBoxLayout(
            orientation='vertical',
            spacing=dp(12),
            size_hint_y=None,
            height=dp(260),
            padding=[dp(16), 0],
        )
        content.add_widget(api_field)
        content.add_widget(model_field)
        content.add_widget(endpoint_field)
        content.add_widget(cooldown_field)

        def add_key(*args):
            key = api_field.text.strip()
            if not key:
                return
            new_entry = {
                'api_key': key,
                'model': model_field.text.strip(),
                'endpoint': endpoint_field.text.strip(),
                'cooldown': int(cooldown_field.text or '0'),
                'enabled': True,
                'usage_count': 0,
            }
            self.api_keys.append(new_entry)
            self._save_keys()
            self._refresh_list()
            self._update_info()
            self._add_dialog.dismiss()

        self._add_dialog = MDDialog(
            MDDialogHeadlineText(text="Add API Key"),
            MDDialogContentContainer(content),
            MDDialogButtonContainer(
                MDButton(MDButtonText(text="Cancel"), on_release=lambda x: self._add_dialog.dismiss()),
                MDButton(MDButtonText(text="Add"), on_release=add_key),
            ),
        )
        self._add_dialog.open()

    def _toggle_key(self, index, enabled):
        """Toggle a key's enabled state."""
        if 0 <= index < len(self.api_keys):
            self.api_keys[index]['enabled'] = enabled
            self._save_keys()
            self._update_info()

    def _delete_key(self, index):
        """Delete a key after confirmation."""
        if 0 <= index < len(self.api_keys):
            self.api_keys.pop(index)
            self._save_keys()
            self._refresh_list()
            self._update_info()

    def _edit_key(self, index):
        """Edit an existing key."""
        if not (0 <= index < len(self.api_keys)):
            return

        key_data = self.api_keys[index]
        api_field = MDTextField(hint_text="API Key", text=key_data.get('api_key', ''))
        model_field = MDTextField(hint_text="Model", text=key_data.get('model', ''))
        endpoint_field = MDTextField(hint_text="Custom Endpoint", text=key_data.get('endpoint', ''))
        cooldown_field = MDTextField(hint_text="Cooldown (s)", text=str(key_data.get('cooldown', 0)))
        cooldown_field.input_filter = 'int'

        content = MDBoxLayout(
            orientation='vertical',
            spacing=dp(12),
            size_hint_y=None,
            height=dp(260),
            padding=[dp(16), 0],
        )
        content.add_widget(api_field)
        content.add_widget(model_field)
        content.add_widget(endpoint_field)
        content.add_widget(cooldown_field)

        def save_edit(*args):
            key = api_field.text.strip()
            if not key:
                return
            self.api_keys[index]['api_key'] = key
            self.api_keys[index]['model'] = model_field.text.strip()
            self.api_keys[index]['endpoint'] = endpoint_field.text.strip()
            self.api_keys[index]['cooldown'] = int(cooldown_field.text or '0')
            self._save_keys()
            self._refresh_list()
            edit_dialog.dismiss()

        edit_dialog = MDDialog(
            MDDialogHeadlineText(text=f"Edit Key #{index + 1}"),
            MDDialogContentContainer(content),
            MDDialogButtonContainer(
                MDButton(MDButtonText(text="Cancel"), on_release=lambda x: edit_dialog.dismiss()),
                MDButton(MDButtonText(text="Save"), on_release=save_edit),
            ),
        )
        edit_dialog.open()

    def toggle_multi_key(self, enabled):
        """Toggle multi-key mode."""
        self.multi_key_enabled = enabled
        self._save_keys()

    def increase_rotation(self):
        """Increase rotation frequency."""
        if self.rotation_frequency < 100:
            self.rotation_frequency += 1
            self._save_keys()

    def decrease_rotation(self):
        """Decrease rotation frequency."""
        if self.rotation_frequency > 1:
            self.rotation_frequency -= 1
            self._save_keys()

    def show_help(self):
        """Show help dialog explaining multi-key rotation."""
        help_text = (
            "API Key Rotation allows you to distribute API calls across "
            "multiple keys to avoid rate limits.\n\n"
            "• Add up to 20 API keys\n"
            "• Set 'Uses per key' to control how many requests use one key "
            "before rotating to the next\n"
            "• Disable keys temporarily without deleting them\n"
            "• Set cooldown periods to prevent overuse\n\n"
            "Keys are used in order from top to bottom, then cycle back."
        )
        dialog = MDDialog(
            MDDialogHeadlineText(text="How Key Rotation Works"),
            MDDialogContentContainer(
                MDLabel(
                    text=help_text,
                    padding=[dp(24), dp(16)],
                    size_hint_y=None,
                    height=dp(300),
                ),
            ),
            MDDialogButtonContainer(
                MDButton(MDButtonText(text="Got it"), on_release=lambda x: dialog.dismiss()),
            ),
        )
        dialog.open()
