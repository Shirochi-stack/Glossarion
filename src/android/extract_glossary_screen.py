# extract_glossary_screen.py
"""
Extract Glossary screen for Android.

Runs extract_glossary_from_epub.py with Android-friendly controls and exposes
the glossary configuration surface using local/default keys (no heavy runtime
parsing of desktop GUI scripts).
"""

import copy
import json
import logging
import os
import threading
import traceback

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout

from kivymd.toast import toast
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.dialog import MDDialog
from kivymd.uix.label import MDLabel
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.screen import MDScreen
from kivymd.uix.textfield import MDTextField

from android_config import DEFAULT_CONFIG, save_config

logger = logging.getLogger(__name__)

KV = """
<ExtractGlossaryScreen>:
    BoxLayout:
        orientation: "vertical"

        MDTopAppBar:
            title: "Extract Glossary"
            md_bg_color: app.theme_cls.primary_color
            elevation: 2

        ScrollView:
            do_scroll_x: False

            BoxLayout:
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                padding: [dp(12), dp(8), dp(12), dp(16)]
                spacing: dp(8)

                MDCard:
                    size_hint_y: None
                    height: dp(142)
                    padding: dp(12)
                    elevation: 2

                    BoxLayout:
                        orientation: "vertical"
                        spacing: dp(6)

                        BoxLayout:
                            size_hint_y: None
                            height: dp(36)
                            spacing: dp(8)

                            MDLabel:
                                text: "Input file"
                                size_hint_x: 0.6

                            MDRaisedButton:
                                text: "Pick File"
                                size_hint_x: 0.4
                                on_release: root.pick_file()

                        MDLabel:
                            text: root.selected_file_display
                            theme_text_color: "Secondary"
                            shorten: True
                            shorten_from: "center"

                        MDLabel:
                            text: "Output: " + root.output_path_display
                            theme_text_color: "Hint"
                            font_style: "Caption"
                            shorten: True
                            shorten_from: "center"

                MDCard:
                    size_hint_y: None
                    height: dp(92)
                    padding: dp(12)
                    elevation: 2

                    BoxLayout:
                        spacing: dp(8)

                        MDRaisedButton:
                            id: run_btn
                            text: root.run_button_text
                            md_bg_color: root.run_button_color
                            size_hint_x: 0.36
                            on_release: root.toggle_run_stop()

                        MDRaisedButton:
                            text: "Save Settings"
                            size_hint_x: 0.33
                            on_release: root.save_glossary_settings()

                        MDRaisedButton:
                            text: "Reload"
                            size_hint_x: 0.31
                            on_release: root.reload_settings()

                MDCard:
                    size_hint_y: None
                    height: dp(94)
                    padding: dp(12)
                    elevation: 1

                    BoxLayout:
                        orientation: "vertical"
                        spacing: dp(4)

                        MDTextField:
                            id: search_field
                            hint_text: "Search glossary settings key..."
                            on_text: root.apply_filter(self.text)

                        MDLabel:
                            text: root.status_text
                            font_style: "Caption"
                            theme_text_color: "Secondary"

                MDLabel:
                    text: "Glossary Settings"
                    size_hint_y: None
                    height: dp(40)
                    font_style: "Subtitle2"
                    theme_text_color: "Secondary"
                    shorten: True
                    shorten_from: "right"

                BoxLayout:
                    id: settings_box
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height
                    spacing: dp(8)

                MDCard:
                    size_hint_y: None
                    height: dp(260)
                    padding: dp(12)
                    elevation: 1

                    BoxLayout:
                        orientation: "vertical"
                        spacing: dp(6)

                        MDLabel:
                            text: "Extraction Log"
                            size_hint_y: None
                            height: dp(24)
                            bold: True

                        ScrollView:
                            do_scroll_x: False

                            MDLabel:
                                id: log_label
                                text: root.log_text
                                font_style: "Caption"
                                theme_text_color: "Secondary"
                                size_hint_y: None
                                height: self.texture_size[1] + dp(24)
                                text_size: self.width - dp(12), None
"""


class ExtractGlossaryScreen(MDScreen):
    """Glossary extraction UI + full glossary settings editor."""

    app = ObjectProperty(None, allownone=True)

    selected_file = StringProperty("")
    selected_file_display = StringProperty("No file selected")
    output_path_display = StringProperty("not set")
    status_text = StringProperty("Ready")
    log_text = StringProperty("")

    run_button_text = StringProperty("Run Extraction")
    run_button_color = ListProperty([0.2, 0.6, 0.3, 1])
    is_running = BooleanProperty(False)

    _ENUM_OPTIONS = {
        "auto_glossary_mode": ["off", "minimal", "balanced", "full", "no_glossary"],
        "glossary_filter_mode": ["all", "strict", "loose", "smart"],
        "glossary_entry_type_filter_mode": ["none", "Loose", "Strict", "All"],
        "glossary_duplicate_algorithm": ["auto", "rapidfuzz", "difflib", "simple"],
        "glossary_duplicate_key_mode": ["auto", "raw_name", "translated_name", "skip"],
        "emergency_glossary_compliance_mode": ["characters", "all_fields", "custom"],
    }

    _FORCED_KEYS = {
        "enable_auto_glossary",
        "auto_glossary_mode",
        "append_glossary",
        "append_glossary_auto_load",
        "add_additional_glossary",
        "additional_glossary_path",
        "manual_glossary_temperature",
        "manual_context_limit",
        "manual_glossary_prompt",
        "append_glossary_prompt",
        "glossary_translation_prompt",
        "glossary_format_instructions",
        "glossary_min_frequency",
        "glossary_max_names",
        "glossary_max_titles",
        "glossary_max_text_size",
        "glossary_max_sentences",
        "glossary_chapter_split_threshold",
        "glossary_fuzzy_threshold",
        "glossary_filter_mode",
        "glossary_duplicate_algorithm",
        "glossary_entry_type_filter_mode",
        "glossary_enable_chapter_split",
        "glossary_compression_factor",
        "glossary_max_output_tokens",
        "compress_glossary_prompt",
        "strip_honorifics",
        "glossary_disable_honorifics_filter",
        "glossary_history_rolling",
        "glossary_request_merging_enabled",
        "glossary_request_merge_count",
        "glossary_include_all_characters",
        "glossary_skip_identical_entries",
        "glossary_use_legacy_csv",
        "glossary_output_legacy_json",
        "use_glossary_keys",
        "glossary_keys",
        "glossary_target_language",
        "glossary_enable_anti_duplicate",
        "glossary_top_p",
        "glossary_top_k",
        "glossary_frequency_penalty",
        "glossary_presence_penalty",
        "glossary_repetition_penalty",
        "glossary_candidate_count",
        "glossary_custom_stop_sequences",
        "glossary_logit_bias_enabled",
        "glossary_logit_bias_strength",
        "glossary_bias_common_words",
        "glossary_bias_repetitive_phrases",
        "glossary_custom_fields",
        "glossary_custom_entry_types",
        "include_book_title_glossary",
        "auto_inject_book_title",
    }

    _BOOL_KEYS = {
        "enable_auto_glossary",
        "append_glossary",
        "append_glossary_auto_load",
        "add_additional_glossary",
        "glossary_enable_chapter_split",
        "compress_glossary_prompt",
        "strip_honorifics",
        "glossary_disable_honorifics_filter",
        "glossary_history_rolling",
        "glossary_request_merging_enabled",
        "glossary_include_all_characters",
        "glossary_skip_identical_entries",
        "glossary_use_legacy_csv",
        "glossary_output_legacy_json",
        "use_glossary_keys",
        "glossary_enable_anti_duplicate",
        "glossary_logit_bias_enabled",
        "glossary_bias_common_words",
        "glossary_bias_repetitive_phrases",
        "include_book_title_glossary",
        "auto_inject_book_title",
    }

    _INT_KEYS = {
        "manual_context_limit",
        "glossary_min_frequency",
        "glossary_max_names",
        "glossary_max_titles",
        "glossary_max_text_size",
        "glossary_max_sentences",
        "glossary_chapter_split_threshold",
        "glossary_request_merge_count",
        "glossary_max_output_tokens",
        "glossary_top_k",
        "glossary_candidate_count",
    }

    _FLOAT_KEYS = {
        "manual_glossary_temperature",
        "glossary_fuzzy_threshold",
        "glossary_compression_factor",
        "glossary_top_p",
        "glossary_frequency_penalty",
        "glossary_presence_penalty",
        "glossary_repetition_penalty",
        "glossary_logit_bias_strength",
    }

    _LIST_KEYS = {
        "glossary_keys",
        "glossary_custom_fields",
        "emergency_glossary_compliance_custom_types",
    }

    _DICT_KEYS = {
        "glossary_custom_entry_types",
    }

    _CARD_RADIUS = [12, 12, 12, 12]
    _CARD_BG = (0.13, 0.13, 0.17, 1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Builder.load_string(KV)
        self._initialized = False
        self._stop_requested = False
        self._worker_thread = None
        self._menu = None
        self._settings_defaults = {}
        self._settings_keys = []
        self._types = {}
        self._raw_values = {}
        self._bool_values = {}
        self._cached_discovered_defaults = None
        self._render_step = 16
        self._render_limit = self._render_step
        self._filter_event = None
        self._build_event = None

    def on_enter(self, *args):
        if not self.app:
            return
        if self._build_event is not None:
            self._build_event.cancel()
            self._build_event = None
        self._load_state()
        self._build_event = Clock.schedule_once(lambda _dt: self._build_settings_ui(), 0)
        self._hydrate_selected_file_from_other_screens()
        self._initialized = True

    def on_enter_data(self, file_path=None, **kwargs):
        if file_path and os.path.isfile(file_path):
            self._set_selected_file(file_path)

    def _hydrate_selected_file_from_other_screens(self):
        if self.selected_file and os.path.isfile(self.selected_file):
            return
        if not self.app or not getattr(self.app, "root", None):
            return
        try:
            ts = self.app.root.ids.screen_manager.get_screen("translation")
            fp = getattr(ts, "selected_file", "")
            if fp and os.path.isfile(fp):
                self._set_selected_file(fp)
                return
        except Exception:
            pass
        try:
            rs = self.app.root.ids.screen_manager.get_screen("reader")
            fp = getattr(rs, "file_path", "")
            if fp and os.path.isfile(fp):
                self._set_selected_file(fp)
        except Exception:
            pass

    def _load_state(self):
        cfg = dict(self.app.config_data if self.app else {})
        if self._cached_discovered_defaults is None:
            self._cached_discovered_defaults = self._build_local_defaults()
        discovered_defaults = self._cached_discovered_defaults
        keys = set(discovered_defaults.keys()) | set(self._FORCED_KEYS)
        keys |= {k for k in cfg.keys() if self._is_glossary_key(k)}

        self._settings_defaults = discovered_defaults
        self._settings_keys = sorted(keys)
        self._types.clear()
        self._raw_values.clear()
        self._bool_values.clear()

        for key in self._settings_keys:
            default = discovered_defaults.get(key, self._default_for_key(key))
            value = cfg.get(key, default)
            val_type = self._infer_type(key, value, default)
            self._types[key] = val_type
            if val_type is bool:
                bool_val = bool(value)
                self._bool_values[key] = bool_val
                self._raw_values[key] = "true" if bool_val else "false"

        self.status_text = f"{len(self._settings_keys)} glossary settings loaded"
        self._render_limit = self._render_step

    def reload_settings(self):
        if self.is_running:
            toast("Cannot reload while extraction is running")
            return
        self._load_state()
        self._build_settings_ui()
        toast("Reloaded glossary settings")

    def apply_filter(self, text):
        self.search_text = (text or "").strip().lower()
        self._render_limit = self._render_step
        if self._filter_event is not None:
            self._filter_event.cancel()
        self._filter_event = Clock.schedule_once(lambda _dt: self._render_settings_rows(), 0.12)

    def _build_settings_ui(self):
        box = self.ids.settings_box
        box.clear_widgets()
        Clock.schedule_once(lambda _dt: self._render_settings_rows(), 0)

    def _render_settings_rows(self):
        box = self.ids.settings_box
        box.clear_widgets()
        needle = getattr(self, "search_text", "")
        filtered = [k for k in self._settings_keys if (not needle or needle in k.lower())]
        visible = filtered[:self._render_limit]
        for key in visible:
            box.add_widget(self._build_setting_row(key))
        remaining = len(filtered) - len(visible)
        if remaining > 0:
            box.add_widget(MDRaisedButton(
                text=f"Load more ({remaining} left)",
                size_hint_y=None,
                height=dp(40),
                md_bg_color=(0.30, 0.30, 0.34, 1),
                on_release=lambda *_a: self._load_more_rows(),
            ))

    def _load_more_rows(self):
        self._render_limit += self._render_step
        self._render_settings_rows()

    def _build_setting_row(self, key):
        t = self._types.get(key, str)
        is_bool = (t is bool)
        is_enum = key in self._ENUM_OPTIONS
        is_multiline = t in (list, dict)
        card_height = dp(100) if (is_bool or is_enum) else (dp(170) if is_multiline else dp(122))

        card = MDCard(
            size_hint_y=None,
            height=card_height,
            elevation=1,
            padding=dp(10),
            radius=self._CARD_RADIUS,
            md_bg_color=self._CARD_BG,
        )
        inner = BoxLayout(
            orientation="vertical",
            spacing=dp(8),
        )
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
            self._sync_toggle_btn(btn, self._bool_values.get(key, False))
            btn.bind(on_release=lambda *_a, k=key, b=btn: self._toggle_bool(k, b))
            row.add_widget(btn)
            row.add_widget(BoxLayout())
            inner.add_widget(row)
            return card

        if is_enum:
            btn = MDRaisedButton(
                text=(self._get_raw_value(key) or self._ENUM_OPTIONS[key][0]),
                size_hint_y=None,
                height=dp(40),
            )
            btn.bind(on_release=lambda *_a, k=key, b=btn: self._open_enum_menu(k, b))
            inner.add_widget(btn)
            return card

        field = MDTextField(
            text=self._get_raw_value(key),
            multiline=is_multiline,
            size_hint_y=None,
            height=dp(104) if is_multiline else dp(56),
        )
        field.bind(text=lambda _i, text, k=key: self._on_text_changed(k, text))
        inner.add_widget(field)
        return card

    def _toggle_bool(self, key, button):
        new_val = not bool(self._bool_values.get(key, False))
        self._bool_values[key] = new_val
        self._raw_values[key] = "true" if new_val else "false"
        self._sync_toggle_btn(button, new_val)

    @staticmethod
    def _sync_toggle_btn(button, enabled):
        button.text = "Enabled" if enabled else "Disabled"
        button.md_bg_color = (0.2, 0.6, 0.3, 1) if enabled else (0.3, 0.3, 0.3, 1)

    def _open_enum_menu(self, key, caller):
        if self._menu:
            try:
                self._menu.dismiss()
            except Exception:
                pass
        options = self._ENUM_OPTIONS.get(key, [])
        items = [{
            "text": str(opt),
            "viewclass": "OneLineListItem",
            "height": dp(44),
            "on_release": (lambda choice=opt: self._select_enum(key, str(choice), caller)),
        } for opt in options]
        self._menu = MDDropdownMenu(
            caller=caller,
            items=items,
            width_mult=4,
            position="center",
            max_height=dp(260),
        )
        self._menu.open()

    def _select_enum(self, key, value, caller):
        self._raw_values[key] = value
        caller.text = value
        if self._menu:
            self._menu.dismiss()
            self._menu = None

    def _on_text_changed(self, key, text):
        self._raw_values[key] = text

    def pick_file(self):
        try:
            from android_file_utils import REQUEST_CODE_OPEN_FILE, open_native_file_picker
            open_native_file_picker(
                callback=self._on_file_picked,
                extensions=[".epub", ".txt", ".pdf"],
                request_code=REQUEST_CODE_OPEN_FILE,
            )
        except Exception as exc:
            self._append_log(f"[ERR] File picker failed: {exc}")

    def _on_file_picked(self, path):
        if path and os.path.isfile(path):
            self._set_selected_file(path)
            toast(f"Selected: {os.path.basename(path)}")

    def _set_selected_file(self, file_path):
        self.selected_file = file_path
        self.selected_file_display = os.path.basename(file_path)
        self.output_path_display = self._compute_output_json(file_path)

    def _compute_output_json(self, file_path):
        try:
            from android_file_utils import get_output_dir
            base = os.path.splitext(os.path.basename(file_path))[0]
            out_dir = get_output_dir(file_path)
            glossary_dir = os.path.join(out_dir, "Glossary")
            return os.path.join(glossary_dir, f"{base}_glossary.json")
        except Exception:
            base = os.path.splitext(os.path.basename(file_path))[0]
            return os.path.join(".", "Glossary", f"{base}_glossary.json")

    def toggle_run_stop(self):
        if self.is_running:
            self._request_stop()
            return
        self._start_extraction()

    def _start_extraction(self):
        if not self.selected_file or not os.path.isfile(self.selected_file):
            toast("Pick a file first")
            return

        if not self.save_glossary_settings(show_toast=False):
            return

        self._stop_requested = False
        self._set_running_state(True)
        self.status_text = "Starting glossary extraction..."
        self._append_log(f"[START] {self.selected_file}")

        self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self._worker_thread.start()

    def _request_stop(self):
        self._stop_requested = True
        os.environ["GRACEFUL_STOP"] = "1"
        os.environ["GRACEFUL_STOP_COMPLETED"] = "0"
        try:
            import extract_glossary_from_epub
            if hasattr(extract_glossary_from_epub, "set_stop_flag"):
                extract_glossary_from_epub.set_stop_flag(True)
        except Exception:
            pass
        self.status_text = "Stopping after current step..."
        self._append_log("[STOP] Stop requested")

    def _set_running_state(self, running):
        self.is_running = bool(running)
        if running:
            self.run_button_text = "Stop"
            self.run_button_color = [0.75, 0.2, 0.2, 1]
        else:
            self.run_button_text = "Run Extraction"
            self.run_button_color = [0.2, 0.6, 0.3, 1]

    def _run_worker(self):
        try:
            cfg = self.app.config_data.copy() if self.app else {}
            output_json = self._compute_output_json(self.selected_file)
            os.makedirs(os.path.dirname(output_json), exist_ok=True)

            self._append_log(f"[OUT] {output_json}")
            self._apply_runtime_env(cfg, output_json)

            import extract_glossary_from_epub
            if hasattr(extract_glossary_from_epub, "set_stop_flag"):
                extract_glossary_from_epub.set_stop_flag(False)

            extract_glossary_from_epub.main(
                log_callback=lambda msg: self._append_log(str(msg)),
                stop_callback=lambda: self._stop_requested,
            )

            if self._stop_requested:
                self._set_status("Stopped")
                self._append_log("[DONE] Stopped by user")
            else:
                self._set_status("Extraction complete")
                self._append_log("[DONE] Extraction finished")
        except Exception as exc:
            self._set_status("Extraction failed")
            self._append_log(f"[ERR] {exc}")
            self._append_log(traceback.format_exc())
        finally:
            Clock.schedule_once(lambda dt: self._on_worker_finished(), 0)

    def _on_worker_finished(self):
        self._set_running_state(False)
        try:
            ps = self.app.root.ids.screen_manager.get_screen("progress")
            if hasattr(ps, "refresh_all"):
                ps.refresh_all()
        except Exception:
            pass

    def _apply_runtime_env(self, cfg, output_json):
        from android_env_propagator import set_all_env_vars, set_input_file_env, set_per_run_env_vars
        from android_file_utils import get_config_path, get_output_dir

        output_dir = get_output_dir(self.selected_file)
        set_all_env_vars(cfg)
        set_per_run_env_vars()
        set_input_file_env(self.selected_file, output_dir)

        os.environ["EPUB_PATH"] = self.selected_file
        os.environ["OUTPUT_PATH"] = output_json
        os.environ["CONFIG_PATH"] = get_config_path()
        os.environ["GLOSSARY_TEMPERATURE"] = str(cfg.get("manual_glossary_temperature", 0.3))
        os.environ["GLOSSARY_CONTEXT_LIMIT"] = str(cfg.get("manual_context_limit", 2))
        os.environ["GLOSSARY_SYSTEM_PROMPT"] = str(cfg.get("manual_glossary_prompt", ""))
        os.environ["AUTO_GLOSSARY_PROMPT"] = str(cfg.get("unified_auto_glosary_prompt3", ""))
        os.environ["AUTO_GLOSSARY_MODE"] = str(cfg.get("auto_glossary_mode", "off"))
        os.environ["APPEND_GLOSSARY_PROMPT"] = str(
            cfg.get(
                "append_glossary_prompt",
                "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n",
            )
        )
        os.environ["GLOSSARY_TRANSLATION_PROMPT"] = str(cfg.get("glossary_translation_prompt", ""))
        os.environ["GLOSSARY_FORMAT_INSTRUCTIONS"] = str(cfg.get("glossary_format_instructions", ""))
        os.environ["GLOSSARY_DISABLE_HONORIFICS_FILTER"] = "1" if cfg.get("glossary_disable_honorifics_filter", False) else "0"
        os.environ["GLOSSARY_HISTORY_ROLLING"] = "1" if cfg.get("glossary_history_rolling", False) else "0"
        os.environ["GLOSSARY_REQUEST_MERGING_ENABLED"] = "1" if cfg.get("glossary_request_merging_enabled", False) else "0"
        os.environ["GLOSSARY_REQUEST_MERGE_COUNT"] = str(cfg.get("glossary_request_merge_count", 3))
        os.environ["GLOSSARY_ENABLE_CHAPTER_SPLIT"] = "1" if cfg.get("glossary_enable_chapter_split", True) else "0"
        os.environ["GLOSSARY_COMPRESSION_FACTOR"] = str(cfg.get("glossary_compression_factor", 1.0))
        os.environ["GLOSSARY_MAX_OUTPUT_TOKENS"] = str(cfg.get("glossary_max_output_tokens", -1))
        os.environ["USE_GLOSSARY_KEYS"] = "1" if cfg.get("use_glossary_keys", False) else "0"
        os.environ["GLOSSARY_API_KEYS"] = json.dumps(cfg.get("glossary_keys", []))
        os.environ["GLOSSARY_CUSTOM_FIELDS"] = json.dumps(cfg.get("glossary_custom_fields", []))
        os.environ["GLOSSARY_CUSTOM_ENTRY_TYPES"] = json.dumps(cfg.get("glossary_custom_entry_types", {}))
        os.environ["GLOSSARY_INCLUDE_BOOK_TITLE"] = "1" if cfg.get("include_book_title_glossary", True) else "0"
        os.environ["GLOSSARY_AUTO_INJECT_BOOK_TITLE"] = "1" if cfg.get("auto_inject_book_title", False) else "0"

    def save_glossary_settings(self, show_toast=True):
        if not self.app:
            return False

        new_cfg = dict(self.app.config_data)
        parse_errors = []

        for key in self._settings_keys:
            target_type = self._types.get(key, str)
            if target_type is bool:
                new_cfg[key] = bool(self._bool_values.get(key, False))
                continue
            default = self._settings_defaults.get(key, self._default_for_key(key))
            if key not in self._raw_values:
                if key not in new_cfg:
                    new_cfg[key] = copy.deepcopy(default) if isinstance(default, (dict, list)) else default
                continue
            raw = self._raw_values.get(key, "")
            try:
                new_cfg[key] = self._coerce_value(raw, target_type, default)
            except Exception as exc:
                parse_errors.append(f"{key}: {exc}")

        if parse_errors:
            self.status_text = f"Parse errors: {len(parse_errors)}"
            self._show_parse_errors_dialog(parse_errors)
            return False

        self.app.config_data.update(new_cfg)
        save_config(self.app.config_data)
        self.status_text = "Settings saved"
        if show_toast:
            toast("Saved glossary settings")
        return True

    def _show_parse_errors_dialog(self, parse_errors):
        preview = "\n".join(parse_errors[:8])
        if len(parse_errors) > 8:
            preview += f"\n... and {len(parse_errors) - 8} more"
        dialog = MDDialog(
            title="Invalid setting values",
            text=preview,
            buttons=[MDFlatButton(text="OK", on_release=lambda x: dialog.dismiss())],
        )
        dialog.open()

    def _append_log(self, text):
        def _do_append(_dt):
            lines = self.log_text.splitlines() if self.log_text else []
            lines.append(str(text))
            if len(lines) > 400:
                lines = lines[-400:]
            self.log_text = "\n".join(lines)

        Clock.schedule_once(_do_append, 0)

    def _set_status(self, text):
        Clock.schedule_once(lambda _dt: setattr(self, "status_text", str(text)), 0)

    @staticmethod
    def _coerce_value(raw_text, value_type, default):
        text = (raw_text or "").strip()
        if value_type is int:
            return int(text) if text else int(default or 0)
        if value_type is float:
            return float(text) if text else float(default or 0.0)
        if value_type is list:
            if not text:
                return [] if default is None else list(default) if isinstance(default, list) else []
            loaded = json.loads(text)
            if not isinstance(loaded, list):
                raise ValueError("must be a JSON array")
            return loaded
        if value_type is dict:
            if not text:
                return {} if default is None else dict(default) if isinstance(default, dict) else {}
            loaded = json.loads(text)
            if not isinstance(loaded, dict):
                raise ValueError("must be a JSON object")
            return loaded
        return text if text or default is None else str(default)

    @staticmethod
    def _value_to_text(value):
        if value is None:
            return ""
        if isinstance(value, (list, dict)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return ""
        return str(value)

    def _get_raw_value(self, key):
        if key in self._raw_values:
            return self._raw_values[key]
        default = self._settings_defaults.get(key, self._default_for_key(key))
        source = self.app.config_data if self.app else {}
        value = source.get(key, default)
        text = self._value_to_text(value)
        self._raw_values[key] = text
        return text

    def _infer_type(self, key, value, default):
        if isinstance(value, bool) or isinstance(default, bool):
            return bool
        if isinstance(value, int) and not isinstance(value, bool):
            return int
        if isinstance(default, int) and not isinstance(default, bool):
            return int
        if isinstance(value, float) or isinstance(default, float):
            return float
        if isinstance(value, list) or isinstance(default, list):
            return list
        if isinstance(value, dict) or isinstance(default, dict):
            return dict
        if key in self._LIST_KEYS:
            return list
        if key in self._DICT_KEYS:
            return dict
        if key in self._BOOL_KEYS:
            return bool
        if key in self._INT_KEYS:
            return int
        if key in self._FLOAT_KEYS:
            return float
        return str

    def _default_for_key(self, key):
        if key in self._ENUM_OPTIONS:
            return self._ENUM_OPTIONS[key][0]
        if key in self._BOOL_KEYS:
            return False
        if key in self._INT_KEYS:
            return 0
        if key in self._FLOAT_KEYS:
            return 0.0
        if key in self._LIST_KEYS:
            return []
        if key in self._DICT_KEYS:
            return {}
        return ""

    def _build_local_defaults(self):
        """
        Build defaults without runtime parsing of huge desktop GUI scripts.
        """
        discovered = {}
        for key in self._FORCED_KEYS:
            if key in DEFAULT_CONFIG:
                discovered[key] = DEFAULT_CONFIG[key]
            else:
                discovered[key] = self._default_for_key(key)
        for key, value in DEFAULT_CONFIG.items():
            if self._is_glossary_key(key):
                discovered.setdefault(key, value)
        return discovered

    @staticmethod
    def _is_glossary_key(key):
        if not key or key.startswith("_"):
            return False
        lk = key.lower()
        if "glossary" in lk:
            return True
        return lk in {
            "enable_auto_glossary",
            "append_glossary",
            "append_glossary_auto_load",
            "add_additional_glossary",
            "additional_glossary_path",
            "strip_honorifics",
            "auto_inject_book_title",
            "include_book_title_glossary",
            "manual_context_limit",
            "manual_glossary_temperature",
            "auto_glossary_mode",
            "use_glossary_keys",
        }
