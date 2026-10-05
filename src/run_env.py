"""RunEnvMixin: the desktop run environment builders, moved verbatim out of TranslatorGUI.

Shared GUI-free core (Glossarion mobile rewrite, milestone U2). ``TranslatorGUI``
inherits ``RunEnvMixin`` (listed before its Qt mixins, so precedence is unchanged)
and the mobile ``HeadlessOwner`` inherits it too. Every method body is a verbatim
move of the translator_gui.py code (git show 96af9adb:src/translator_gui.py); the
only edits are:

* ``QApplication``/``QThread`` "can read widgets" checks call the
  ``_can_read_widgets()`` hook (default ``True``; TranslatorGUI overrides it with
  the original Qt check);
* ``TranslatorGUI.<name>(self, ...)`` explicit calls name ``RunEnvMixin`` (the
  same function: TranslatorGUI does not override them);
* ``_InputOutputDialog._FORCED_STREAM_ENV_KEYS`` is ``FORCED_STREAM_ENV_KEYS``
  (the dialog keeps an alias);
* split-outs that the job runners call at the original position:
  ``_glossary_extraction_paths`` / ``_build_glossary_extraction_env``
  (``_extract_glossary_from_text_file``), ``_build_epub_compile_env``
  (``run_epub_converter_direct``) and ``_build_pdf_compile_env``
  (``run_pdf_converter_direct``).

The owner is duck-typed: methods read ``self.config``, ``self.*_var`` attributes
and a few widget-like objects (``.text()``/``.isChecked()``/``.toPlainText()``),
exactly as the desktop does. ``build_*`` module functions are conveniences for
mobile, previews and tests.

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup;
backend modules are imported lazily inside the methods (as before).
"""

import json
import os
import platform
import re
import sys
from collections import namedtuple
from decimal import Decimal, InvalidOperation

from app_paths import _get_app_dir
from emoticon_patterns import DEFAULT_EMOTICON_PATTERNS
from key_pools import apply_key_pools_to_runtime
from title_tag_translation import DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT

# ---- module helpers (moved verbatim from translator_gui module level) ----

def _authnd_auto_token_limits():
    cores = max(1, int(os.cpu_count() or 1))
    token_limit = min(4, max(1, cores // 2))
    subprocess_limit = min(8, max(token_limit, cores))
    return token_limit, subprocess_limit, cores


def _format_plain_decimal_setting(value, default="0.0001"):
    """Return a validated decimal string without scientific notation."""
    raw = "" if value is None else str(value).strip()
    if not raw:
        raw = str(default)
    try:
        number = Decimal(raw)
    except (InvalidOperation, ValueError):
        number = Decimal(str(default))
    if not number.is_finite():
        number = Decimal(str(default))

    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


MULTIPASS_REFINEMENT_MODES = ("full", "full_with_raw", "failed", "partial", "partial.b", "partial.b2")
REFINEMENT_RAW_PROMPT_ROLES = ("assistant", "system", "user")

# Interactive (Direct Text / single chapter) runs force every streaming/logging switch on.
# Was _InputOutputDialog._FORCED_STREAM_ENV_KEYS (the dialog keeps that name as an alias).
FORCED_STREAM_ENV_KEYS = (
    'ENABLE_STREAMING',
    'ALLOW_BATCH_STREAM_LOGS',
    'ALLOW_AUTHGPT_BATCH_STREAM_LOGS',
    'ALLOW_GLM_PROXY_BATCH_STREAM_LOGS',
    'STREAM_THINKING_LOGS',
    'LOG_STREAM_CHUNKS',
    'RESPONSE_STREAMING',
    'ENABLE_THOUGHTS',
    'AUTHND_STREAM',
    'AUTHND_LOG_STREAM_CHUNKS',
    'AUTHND_STREAM_THINKING_LOGS',
)


class RunEnvMixin:
    """Run environment builders shared by TranslatorGUI and HeadlessOwner (see module docstring)."""

    # ---- hooks (GUI-free defaults; TranslatorGUI overrides) ----
    def _can_read_widgets(self):
        """True when live widgets may be read on this thread (desktop: the Qt GUI thread)."""
        return True

    _CUSTOM_PREFIX_ENDPOINT_TYPES = (
        "/chat/completions",
        "/images/generations",
        "/v1/messages",
        "/v1/ocr",
        "/{model_id}",
    )

    # Named model aliases that should be treated as image / video generators
    # even though the word 'image'/'video' doesn't appear in the model name.
    _IMAGE_MODEL_ALIASES = ('nano-banana',)

    _VIDEO_MODEL_ALIASES = ('seedance', 'kling', 'pixverse')

    @classmethod
    def _model_is_image_gen(cls, model_name: str) -> bool:
        """Return True when model_name indicates an image-generation model."""
        m = model_name.lower()
        return 'image' in m or any(a in m for a in cls._IMAGE_MODEL_ALIASES)

    @classmethod
    def _model_is_video_gen(cls, model_name: str) -> bool:
        """Return True when model_name indicates a video-generation model."""
        m = model_name.lower()
        return 'video' in m or any(a in m for a in cls._VIDEO_MODEL_ALIASES)

    def _is_generative_output_mode(self) -> bool:
        """Return True when an image/video output-only toggle is active.

        This complements the model-name heuristics so that generative
        routing also triggers when the user manually enables an output
        mode toggle for a model whose name doesn't contain 'image'/'video'.
        """
        return (
            getattr(self, 'enable_image_output_mode_var', False)
            or getattr(self, 'enable_video_output_mode_var', False)
        )

    def _get_output_mode(self) -> str:
        """Return the normalized output mode selected in the UI/config."""
        try:
            mode = str(getattr(self, 'output_mode_var', None) or self.config.get('output_mode', 'text')).lower().strip()
        except Exception:
            mode = 'text'
        if mode == 'refine':
            mode = 'refinement'
        # Older config fields are still saved for compatibility. If they ever
        # become inconsistent, prefer explicit output-mode toggles over the
        # generic image-translation flag, which is also true for Image mode.
        if mode == 'vision':
            try:
                if bool(getattr(self, 'enable_image_output_mode_var', False) or self.config.get('enable_image_output_mode', False)):
                    return 'image'
                if bool(getattr(self, 'enable_video_output_mode_var', False) or self.config.get('enable_video_output_mode', False)):
                    return 'video'
                if bool(getattr(self, 'enable_audio_output_mode_var', False) or self.config.get('enable_audio_output_mode', False)):
                    return 'audio'
                if bool(getattr(self, 'enable_refinement_output_mode_var', False) or self.config.get('enable_refinement_output_mode', False)):
                    return 'refinement'
            except Exception:
                pass
        return mode if mode in ('text', 'vision', 'image', 'video', 'audio', 'refinement') else 'text'

    def _get_allowed_image_output_mode(self):
        """Check if image output mode should be enabled based on dependencies.
        Returns '1' if allowed and enabled, '0' otherwise.
        The model name never changes this setting; the dropdown/radio choice does."""
        try:
            # Video mode takes priority – never both at once
            if getattr(self, 'enable_video_output_mode_var', False):
                return '0'
            # Check if user wants it enabled
            if not getattr(self, 'enable_image_output_mode_var', False):
                return '0'
            # Otherwise requires image translation to be enabled
            if getattr(self, 'enable_image_translation_var', False):
                return '1'
            return '0'
        except Exception:
            return '0'

    def _get_allowed_video_output_mode(self):
        """Check if video output mode should be enabled based on dependencies.
        Returns '1' if allowed and enabled, '0' otherwise.
        The model name never changes this setting; the dropdown/radio choice does."""
        try:
            # Image mode takes priority if somehow both are on
            if getattr(self, 'enable_image_output_mode_var', False):
                return '0'
            if not getattr(self, 'enable_video_output_mode_var', False):
                return '0'
            return '1'
        except Exception:
            return '0'

    _CHUNK_BUDGET_SAFETY_MARGIN = 500

    def _compression_chunk_budget(self):
        """Max input chunk budget for the current compression factor (same formula as Other Settings)."""
        try:
            output_tokens = int(getattr(self, 'max_output_tokens', self.config.get('max_output_tokens', 65536)))
            factor = float(getattr(self, 'compression_factor_var', self.config.get('compression_factor', 3.0)))
        except (TypeError, ValueError):
            return None
        if factor <= 0:
            return None
        return max(1000, int((output_tokens - self._CHUNK_BUDGET_SAFETY_MARGIN) / factor))

    def _resolve_max_retry_tokens(self, current_max_tokens: int) -> int:
        """
        Interpret max_retry_tokens_var, treating -1 (or any non-positive value) as
        'use the main output token limit'.
        """
        try:
            val = int(getattr(self, 'max_retry_tokens_var', current_max_tokens))
        except Exception:
            return int(current_max_tokens)
        if val is None:
            return int(current_max_tokens)
        if val <= 0:
            return int(current_max_tokens)
        return val

    def _resolve_max_retries(self) -> int:
        """Return the live Other Settings retry count, falling back to config/default."""
        raw = getattr(self, 'max_retries_var', None)
        if raw is None:
            raw = self.config.get('max_retries', os.environ.get('MAX_RETRIES', '7'))
        try:
            value = int(str(raw).strip())
        except Exception:
            value = 7
        return max(1, value)

    def _get_multipass_refinement_mode(self):
        """Return the live multipass refinement mode from the combobox/config."""
        mode = None
        can_read_widgets = self._can_read_widgets()

        combo = getattr(self, 'multipass_refinement_mode_combo', None) if can_read_widgets else None
        if combo is not None:
            try:
                mode = combo.currentData()
            except Exception:
                mode = None
            if not mode:
                try:
                    mode = combo.currentText()
                except Exception:
                    mode = None
        if not mode:
            mode = getattr(self, 'multipass_refinement_mode_var', None)
        if not mode:
            mode = self.config.get('multipass_refinement_mode', 'full')
        mode = str(mode or 'full').strip().lower()
        return mode if mode in MULTIPASS_REFINEMENT_MODES else 'full'

    def _sync_multipass_refinement_mode_from_combo(self, _idx=None):
        mode = self._get_multipass_refinement_mode()
        self.multipass_refinement_mode_var = mode
        self.config['multipass_refinement_mode'] = mode

    def _export_multipass_runtime_env(self):
        """Capture the live multipass controls and export them for the worker."""
        forced_mode = str(
            getattr(self, '_translation_run_forced_multipass_mode', '') or ''
        ).strip().lower()
        if forced_mode in MULTIPASS_REFINEMENT_MODES:
            os.environ['MULTIPASS_MODE'] = '1'
            os.environ['MULTIPASS_REFINEMENT_MODE'] = forced_mode
            try:
                os.environ['SCAN_PHASE_MODE'] = self._get_scan_phase_mode()
                os.environ['QA_SCANNER_SETTINGS_JSON'] = (
                    self._get_qa_scanner_settings_json()
                )
            except Exception:
                pass
            return True, forced_mode

        if (
            getattr(self, '_input_output_run_active', False)
            and getattr(self, '_direct_text_force_multipass_off', True)
        ):
            mode = self._get_multipass_refinement_mode()
            os.environ['MULTIPASS_MODE'] = '0'
            os.environ['MULTIPASS_REFINEMENT_MODE'] = mode
            return False, mode

        can_read_widgets = self._can_read_widgets()

        if can_read_widgets:
            checkbox = getattr(self, 'multipass_checkbox', None)
            if checkbox is not None:
                try:
                    self.multipass_mode_var = bool(checkbox.isChecked())
                except Exception:
                    pass
            self._sync_multipass_refinement_mode_from_combo()

        enabled = bool(getattr(self, 'multipass_mode_var', self.config.get('multipass_mode', False)))
        mode = self._get_multipass_refinement_mode()

        self.multipass_mode_var = enabled
        self.multipass_refinement_mode_var = mode
        self.config['multipass_mode'] = enabled
        self.config['multipass_refinement_mode'] = mode

        os.environ['MULTIPASS_MODE'] = '1' if enabled else '0'
        os.environ['MULTIPASS_REFINEMENT_MODE'] = mode
        try:
            os.environ['SCAN_PHASE_MODE'] = self._get_scan_phase_mode()
            os.environ['QA_SCANNER_SETTINGS_JSON'] = self._get_qa_scanner_settings_json()
        except Exception:
            pass
        return enabled, mode

    def _live_chapter_range_settings(self):
        """Return the live range text, parsed bounds, and spine-order state."""
        range_text = ""
        entry = getattr(self, 'chapter_range_entry', None)
        read_live_entry = False
        if entry is not None:
            try:
                range_text = str(entry.text() or '').strip()
                read_live_entry = True
            except Exception:
                range_text = ""
        if not read_live_entry:
            try:
                range_text = str(self.config.get('chapter_range', '') or '').strip()
            except Exception:
                range_text = ""

        parsed_range = (
            RunEnvMixin._parse_chapter_range_text(self, range_text)
            if range_text else None
        )
        spine_order = False
        spine_checkbox = getattr(self, 'use_spine_order_checkbox', None)
        if spine_checkbox is not None:
            try:
                spine_order = bool(spine_checkbox.isChecked())
            except Exception:
                spine_order = False
        else:
            try:
                spine_order = bool(self.config.get('use_spine_order', False))
            except Exception:
                spine_order = False
        return range_text, parsed_range, spine_order

    def _export_chapter_range_runtime_env(self):
        """Export one authoritative live chapter scope for every run phase."""
        range_text, parsed_range, spine_order = RunEnvMixin._live_chapter_range_settings(self)
        try:
            self.config['chapter_range'] = range_text
            self.config['use_spine_order'] = spine_order
        except Exception:
            pass

        if range_text:
            os.environ['CHAPTER_RANGE'] = range_text
            os.environ['USE_SPINE_ORDER'] = '1' if spine_order else '0'
        else:
            # Do not let a range from a previous run leak into a blank range.
            os.environ.pop('CHAPTER_RANGE', None)
            os.environ.pop('USE_SPINE_ORDER', None)
        return range_text, parsed_range, spine_order

    def _get_scan_phase_mode(self):
        """Return the live QA scan mode used by post-translation scanning."""
        mode = getattr(self, 'scan_phase_mode_var', None) or self.config.get('scan_phase_mode', 'quick-scan')
        mode = str(mode or 'quick-scan').strip().lower()
        if mode not in ('quick-scan', 'aggressive', 'ai-hunter', 'custom'):
            mode = 'quick-scan'
        return mode

    def _get_qa_scanner_settings_json(self):
        """Serialize QA scanner settings so worker-side scans match GUI scans."""
        settings = self.config.get('qa_scanner_settings', {})
        if not isinstance(settings, dict):
            settings = {}
        settings = dict(settings)
        try:
            main_lang = self.config.get('output_language') or getattr(self, 'target_lang_var', '') or os.getenv('OUTPUT_LANGUAGE', '')
            if main_lang:
                settings['target_language'] = str(main_lang).strip().lower()
        except Exception:
            pass
        try:
            return json.dumps(settings, ensure_ascii=False)
        except Exception:
            return "{}"

    @staticmethod
    def _normalize_custom_prefix_endpoint_type(endpoint_type):
        """Return a custom prefix endpoint path, preserving user-defined paths."""
        value = str(endpoint_type or '').strip().lower().replace('-', '_').replace(' ', '_')
        legacy = {
            '': '/chat/completions',
            'openai_chat': '/chat/completions',
            'openai_images': '/images/generations',
            'anthropic_messages': '/v1/messages',
            'mistral_ocr': '/v1/ocr',
            '{base_url}/chat/completions': '/chat/completions',
            '{base_url}/images/generations': '/images/generations',
            '{base_url}/v1/messages': '/v1/messages',
            '{base_url}/v1/ocr': '/v1/ocr',
            '{base_url}/{model_id}': '/{model_id}',
        }
        if value in legacy:
            return legacy[value]
        raw = str(endpoint_type or '').strip()
        if raw.startswith('{base_url}/'):
            raw = raw[len('{base_url}'):]
        if raw.startswith('/') and not any(ch.isspace() for ch in raw):
            return raw
        return '/chat/completions'

    @staticmethod
    def _is_valid_custom_prefix_endpoint_type(endpoint_type):
        """Return True when endpoint_type is a preset or custom absolute endpoint path."""
        raw = str(endpoint_type or '').strip()
        if raw in RunEnvMixin._CUSTOM_PREFIX_ENDPOINT_TYPES:
            return True
        if raw.startswith('{base_url}/'):
            raw = raw[len('{base_url}'):]
        return bool(raw.startswith('/') and not any(ch.isspace() for ch in raw))

    def _normalize_custom_prefix_routes(self, routes):
        """Return validated custom prefix route dictionaries for custom endpoints."""
        normalized = []
        seen = set()

        if isinstance(routes, dict):
            iterable = [{'prefix': k, 'routing': v} for k, v in routes.items()]
        elif isinstance(routes, list):
            iterable = routes
        else:
            iterable = []

        for entry in iterable:
            if not isinstance(entry, dict):
                continue
            prefix = str(entry.get('prefix', '') or '').strip()
            routing = str(entry.get('routing', entry.get('base_url', '')) or '').strip()
            endpoint_type = self._normalize_custom_prefix_endpoint_type(
                entry.get('endpoint_type', entry.get('type', '/chat/completions'))
            )
            if not prefix or not routing:
                continue
            prefix = prefix.replace('\\', '/').lstrip('/')
            if not prefix.endswith('/'):
                prefix = f"{prefix}/"
            routing = routing.rstrip('/')
            if not routing.lower().startswith(('http://', 'https://')):
                continue
            key = prefix.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append({
                'prefix': prefix,
                'routing': routing,
                'endpoint_type': endpoint_type,
            })

        return normalized

    def _custom_prefix_routes_env_json(self):
        """Serialize custom prefix routes for UnifiedClient subprocess/thread routing."""
        try:
            routes = self._normalize_custom_prefix_routes(
                getattr(self, 'custom_prefix_routes', self.config.get('custom_prefix_routes', []))
            )
            return json.dumps(routes, ensure_ascii=False)
        except Exception:
            return "[]"

    def _ollama_settings_env_json(self):
        """Serialize the shared local Ollama settings for every translation path."""
        from ollama_settings import ollama_settings_json
        return ollama_settings_json(self.config)

    def _sync_custom_prefix_routes_env(self):
        """Expose custom prefix routes to UnifiedClient."""
        try:
            os.environ['CUSTOM_OPENAI_PREFIX_ROUTES'] = self._custom_prefix_routes_env_json()
            os.environ['OLLAMA_SETTINGS_JSON'] = self._ollama_settings_env_json()
        except Exception:
            pass

    def _context_mode_from_flags(self):
        """Return the UI context mode represented by legacy config flags."""
        if bool(getattr(self, 'rolling_summary_var', False)):
            mode = str(getattr(self, 'rolling_summary_mode_var', 'replace') or 'replace').strip().lower()
            return 'rolling_summary_append' if mode == 'append' else 'rolling_summary_replace'
        if bool(getattr(self, 'contextual_var', False)):
            return 'contextual_history'
        return 'off'

    def _translation_batching_mode_for_env(self):
        """Return the effective batching mode for main translation subprocesses."""
        context_mode = getattr(self, 'context_mode_var', None) or self._context_mode_from_flags()
        if context_mode == 'off':
            return 'aggressive'
        mode = str(getattr(self, 'batch_mode_var', 'aggressive') or 'aggressive').strip().lower()
        if mode not in ('direct', 'conservative', 'aggressive'):
            mode = 'direct'
        if mode == 'aggressive':
            return 'direct'
        return mode

    def _glossary_batching_mode_for_env(self):
        """Glossary extraction uses no batching except when Contextual History is active."""
        context_mode = getattr(self, 'context_mode_var', None) or self._context_mode_from_flags()
        if context_mode != 'contextual_history':
            return 'aggressive'
        mode = str(getattr(self, 'batch_mode_var', 'direct') or 'direct').strip().lower()
        if mode not in ('direct', 'conservative', 'aggressive'):
            mode = 'direct'
        return 'direct' if mode == 'aggressive' else mode

    def _live_bool_setting(self, checkbox_attr, var_attr, config_key, default=False):
        widget = getattr(self, checkbox_attr, None)
        if widget is not None and hasattr(widget, 'isChecked'):
            try:
                return bool(widget.isChecked())
            except Exception:
                pass
        if hasattr(self, var_attr):
            return self._coerce_live_bool(getattr(self, var_attr), default)
        return self._coerce_live_bool(self.config.get(config_key, default), default)

    def _live_text_setting(self, widget_attr, var_attr, config_key, default=''):
        widget = getattr(self, widget_attr, None)
        if widget is not None and hasattr(widget, 'text'):
            try:
                value = str(widget.text()).strip()
                if value:
                    return value
            except Exception:
                pass
        if hasattr(self, var_attr):
            value = getattr(self, var_attr)
            if value is not None:
                value = str(value).strip()
                if value:
                    return value
        value = self.config.get(config_key, default)
        return str(value if value is not None else default).strip()

    def _current_auto_glossary_mode(self):
        if (
            getattr(self, '_input_output_run_active', False)
            and getattr(self, '_direct_text_use_manual_glossary', False)
        ):
            return 'off_no_automap'
        if (
            getattr(self, '_input_output_run_active', False)
            and getattr(self, '_direct_text_force_no_glossary', True)
        ):
            return 'no_glossary'

        combo = getattr(self, 'auto_glossary_mode_combo', None)
        if combo is not None and hasattr(combo, 'currentText'):
            try:
                mode_raw = str(combo.currentText() or '').strip()
            except Exception:
                mode_raw = ''
        else:
            mode_raw = str(
                getattr(self, 'auto_glossary_mode_var', None)
                or self.config.get('auto_glossary_mode')
                or ''
            ).strip()
        if not mode_raw:
            return 'minimal' if self.config.get('enable_auto_glossary', False) else 'off'
        display_to_mode = {
            'Off': 'off',
            'Off (Fuzzy Mapping)': 'off_fuzzy_automap',
            'Manual Glossary Only': 'off_no_automap',
            'Off (No Auto-Mapping)': 'off_no_automap',
            'No Glossary': 'no_glossary',
            'Minimal': 'minimal',
            'Balanced': 'balanced',
            'Full': 'full',
            'Single Pass': 'single_pass',
        }
        return display_to_mode.get(mode_raw, mode_raw.lower().replace(' ', '_'))

    def _current_glossary_request_env(self, force_balanced_request_merging=False):
        merging_enabled = self._live_bool_setting(
            'glossary_request_merging_checkbox',
            'glossary_request_merging_enabled_var',
            'glossary_request_merging_enabled',
            False,
        )
        merge_count = self._live_text_setting(
            'glossary_request_merge_count_entry',
            'glossary_request_merge_count_var',
            'glossary_request_merge_count',
            '10',
        ) or '10'
        chapter_split = self._live_bool_setting(
            'glossary_enable_chapter_split_checkbox',
            'glossary_enable_chapter_split_var',
            'glossary_enable_chapter_split',
            False,
        )
        if force_balanced_request_merging:
            return '1', '99', '1' if chapter_split else '0'
        return '1' if merging_enabled else '0', str(merge_count), '1' if chapter_split else '0'

    def _glossary_contextual_env_value(self):
        disable_history = self._live_bool_setting(
            'disable_glossary_history_checkbox',
            'disable_glossary_history_var',
            'disable_glossary_history',
            True,
        )
        if disable_history:
            return '0'
        return '1' if getattr(self, 'contextual_var', False) else '0'

    def _glossary_skip_title_header_only_env_value(self):
        enabled = self._live_bool_setting(
            'glossary_skip_title_header_only_checkbox',
            'glossary_skip_title_header_only_var',
            'glossary_skip_title_header_only',
            True,
        )
        return '1' if enabled else '0'

    def _glossary_add_minimal_pass_env_value(self):
        enabled = self._live_bool_setting(
            'glossary_add_minimal_pass_checkbox',
            'glossary_add_minimal_pass_var',
            'glossary_add_minimal_pass',
            False,
        )
        return '1' if enabled else '0'

    def _glossary_match_engine_env_value(self):
        """Resolve Precise Term Matching / Log Match Differences into the engine.

        Must be exported with every run. It used to be set only by the
        Glossary Settings toggle and Save handlers, so after a restart a
        ticked Precise Term Matching box did nothing until that dialog was
        opened and saved again: compression silently ran the legacy matcher.
        """
        precise = self._live_bool_setting(
            'precise_matching_checkbox',
            'compress_glossary_precise_matching_var',
            'compress_glossary_precise_matching',
            True,
        )
        shadow = self._live_bool_setting(
            'shadow_log_matching_checkbox',
            'compress_glossary_shadow_log_var',
            'compress_glossary_shadow_log',
            False,
        )
        return 'new' if precise else ('shadow' if shadow else 'legacy')

    def _strict_matching_env_dict(self):
        """Precise Term Matching's whole-term scope (all / gender / custom / none).

        Exported beside GLOSSARY_MATCH_ENGINE at every site for the same
        reason: a setting only the Glossary Settings dialog writes is lost
        on restart until that dialog is opened again.
        """
        self._migrate_strict_matching_config()
        mode = str(
            getattr(self, 'compress_glossary_strict_matching_mode_var', None)
            or self.config.get('compress_glossary_strict_matching_mode', 'all')
            or 'all'
        ).strip().lower()
        if mode in ('characters', 'character'):
            mode = 'gender'
        if mode not in ('all', 'gender', 'custom', 'none'):
            mode = 'all'
        custom_types = getattr(self, 'compress_glossary_strict_matching_custom_types_var', None)
        if custom_types is None:
            custom_types = self.config.get('compress_glossary_strict_matching_custom_types', [])
        return {
            'COMPRESS_GLOSSARY_STRICT_MATCHING_MODE': mode,
            'COMPRESS_GLOSSARY_STRICT_MATCHING_CUSTOM_TYPES': json.dumps(
                [str(t) for t in (custom_types or [])], ensure_ascii=False
            ),
        }

    def _unified_glossary_env_dict(self):
        """The five unified-glossary variables, for every env export site.

        One helper rather than four literals per site, so the three export
        sites (two dicts and the tuple list) cannot drift apart.
        """
        enabled = self._live_bool_setting(
            'enable_unified_glossary_checkbox',
            'enable_unified_glossary_var',
            'enable_unified_glossary',
            False,
        )
        generate = self._live_bool_setting(
            'generate_unified_glossary_checkbox',
            'generate_unified_glossary_var',
            'generate_unified_glossary',
            False,
        )
        combine_all = self._live_bool_setting(
            'unified_combine_all_languages_checkbox',
            'unified_glossary_combine_all_languages_var',
            'unified_glossary_combine_all_languages',
            False,
        )
        exclude_gender = self._live_bool_setting(
            'unified_exclude_gender_entries_checkbox',
            'unified_glossary_exclude_gender_entries_var',
            'unified_glossary_exclude_gender_entries',
            True,
        )
        source_language = str(
            getattr(self, 'unified_glossary_source_language_var', None)
            or self.config.get('unified_glossary_source_language', 'auto')
            or 'auto'
        ).strip().lower() or 'auto'
        return {
            'ENABLE_UNIFIED_GLOSSARY': '1' if enabled else '0',
            'GENERATE_UNIFIED_GLOSSARY': '1' if generate else '0',
            'UNIFIED_GLOSSARY_SOURCE_LANGUAGE': source_language,
            'UNIFIED_GLOSSARY_COMBINE_ALL_LANGUAGES': '1' if combine_all else '0',
            'UNIFIED_GLOSSARY_EXCLUDE_GENDER_ENTRIES': '1' if exclude_gender else '0',
        }

    def _get_output_base_dir(self, input_file: str = "") -> str:
        """Return the base directory where translation output folders are created.

        On Windows the translation engine writes output relative to the
        script / exe directory, NOT the input file's location.  On macOS
        CWD can be '/' due to App Translocation so we prefer the input
        file's parent there, with the script dir as fallback.
        """
        # Candidate 1: the script / exe directory (where the engine outputs)
        if getattr(sys, 'frozen', False):
            script_dir = os.path.dirname(sys.executable)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))

        if platform.system() == 'Windows':
            return script_dir

        # macOS / Linux — prefer input file dir, fall back to script dir
        if input_file:
            input_dir = os.path.dirname(os.path.abspath(input_file))
            if input_dir and input_dir != '/':
                return input_dir
        return script_dir

    def _subtitle_zip_output_info(self, input_file: str):
        """Return the session-only archive output mapping for an extracted subtitle."""
        mappings = getattr(self, '_subtitle_zip_output_groups', None)
        if not isinstance(mappings, dict) or not input_file:
            return None
        key = os.path.normcase(os.path.abspath(str(input_file)))
        info = mappings.get(key)
        return info if isinstance(info, dict) else None

    def _resolve_translation_output_dir(self, input_file: str) -> str:
        """Return the expected translation output directory for one input file."""
        subtitle_group = self._subtitle_zip_output_info(input_file)
        if subtitle_group and subtitle_group.get('output_dir'):
            return os.path.abspath(str(subtitle_group['output_dir']))

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')
        if override_dir:
            default_output = os.path.join(os.path.abspath(override_dir), base_name)
        else:
            relative_output = base_name
            helper_output = os.path.join(self._get_output_base_dir(input_file), base_name)
            default_output = (
                relative_output
                if os.path.exists(relative_output) or not os.path.exists(helper_output)
                else helper_output
            )
        return default_output

    def _active_translation_output_mode(self) -> str:
        override = getattr(self, '_translation_run_output_mode_override', None)
        if override:
            return str(override).lower().strip()
        if getattr(self, '_input_output_run_active', False):
            direct_mode = getattr(self, '_direct_text_output_mode', None)
            if direct_mode:
                return str(direct_mode).lower().strip()
        return self._get_output_mode()

    def _parse_chapter_range_text(self, value):
        """Parse chapter range text. A single number means that one chapter."""
        value = str(value or '').strip()
        if re.match(r"^\d+$", value):
            num = int(value)
            return num, num
        if re.match(r"^\d+\s*-\s*\d+$", value):
            return tuple(map(int, re.split(r"\s*-\s*", value, 1)))
        return None

    def _metadata_only_environment_for_file(
        self,
        file_path: str,
        api_key: str,
        base_env: dict | None = None,
    ) -> dict:
        """Build one isolated metadata-only backend environment."""
        env_vars = dict(
            base_env
            if base_env is not None
            else self._get_environment_variables(file_path, api_key)
        )
        if base_env is not None:
            # The expensive GUI settings snapshot is shared by the batch, but
            # retain the correct source identity in every individual job.
            env_vars['EPUB_PATH'] = file_path
            env_vars['GLOSSARY_SOURCE_PATH'] = file_path
        try:
            metadata_fields = json.loads(
                env_vars.get('TRANSLATE_METADATA_FIELDS', '{}')
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata_fields = {}
        if not isinstance(metadata_fields, dict):
            metadata_fields = {}
        if not any(
            bool(enabled)
            for field, enabled in metadata_fields.items()
            if field != '_per_epub'
        ):
            metadata_fields['title'] = True

        env_vars.update({
            'METADATA_ONLY': '1',
            'TRANSLATE_BOOK_TITLE': '1',
            'TRANSLATE_METADATA_FIELDS': json.dumps(
                metadata_fields, ensure_ascii=False
            ),
            'OUTPUT_MODE': 'text',
            'VISION_OCR_FIRST': '0',
            'ENABLE_IMAGE_TRANSLATION': '0',
            'ENABLE_IMAGE_OUTPUT_MODE': '0',
            'ENABLE_VIDEO_OUTPUT_MODE': '0',
            'ENABLE_AUDIO_OUTPUT_MODE': '0',
            'ENABLE_REFINEMENT_OUTPUT_MODE': '0',
            'MULTIPASS_MODE': '0',
            'USE_ASYNC_CHAPTER_EXTRACTION': '0',
        })

        output_roots = getattr(self, '_metadata_output_roots', {}) or {}
        source_key = os.path.normcase(os.path.abspath(file_path))
        output_root = output_roots.get(source_key)
        if output_root:
            output_root = os.path.abspath(output_root)
            env_vars['OUTPUT_DIRECTORY'] = output_root
            env_vars['OUTPUT_DIR'] = output_root
        return env_vars

    def _apply_forced_streaming_environment(self):
        """Force every streaming/logging switch on for an interactive run."""
        for key in FORCED_STREAM_ENV_KEYS:
            os.environ[key] = '1'

    def _apply_direct_text_runtime_environment(self):
        """Apply the scoped Direct Text overrides after normal GUI env export."""
        if not getattr(self, '_input_output_run_active', False):
            return

        try:
            import large_env
        except Exception:
            large_env = None

        attachment_prompt = str(
            getattr(self, '_direct_text_attachment_prompt', '') or ''
        ).strip()
        attachment_prompt_role = str(
            getattr(self, '_direct_text_attachment_prompt_role', 'user') or 'user'
        ).strip().lower()
        if attachment_prompt_role not in {'system', 'assistant', 'user'}:
            attachment_prompt_role = 'user'
        skip_prompt_profile = bool(
            getattr(self, '_direct_text_skip_prompt_profile', False)
        )
        os.environ['DIRECT_TEXT_ATTACHMENT_PROMPT_ROLE'] = attachment_prompt_role
        os.environ['DIRECT_TEXT_SKIP_PROMPT_PROFILE'] = (
            '1' if skip_prompt_profile else '0'
        )
        if large_env is not None:
            large_env.set_env('DIRECT_TEXT_ATTACHMENT_PROMPT', attachment_prompt)
        else:
            os.environ['DIRECT_TEXT_ATTACHMENT_PROMPT'] = attachment_prompt

        # Keep the typed attachment instruction's selected role exact. When
        # the selected profile is configured as a user prompt, route only that
        # profile through the scoped user-prefix helper instead of applying the
        # global transform to every system message in the request.
        profile_as_user = bool(getattr(self, 'system_prompt_to_user_var', False))
        profile_prompt = ''
        try:
            profile_prompt = self.prompt_text.toPlainText().strip()
        except Exception:
            profile_prompt = str(self.config.get('system_prompt', '') or '').strip()
        direct_profile_user_prompt = (
            profile_prompt if profile_as_user and not skip_prompt_profile else ''
        )
        if large_env is not None:
            large_env.set_env(
                'DIRECT_TEXT_PROFILE_USER_PROMPT', direct_profile_user_prompt
            )
        else:
            os.environ['DIRECT_TEXT_PROFILE_USER_PROMPT'] = direct_profile_user_prompt
        os.environ['SYSTEM_PROMPT_TO_USER'] = '0'

        output_mode = str(
            getattr(self, '_direct_text_output_mode', 'text') or 'text'
        ).strip().lower()
        if output_mode == 'refine':
            output_mode = 'refinement'
        if output_mode not in {
            'text', 'vision', 'image', 'video', 'audio', 'refinement'
        }:
            output_mode = 'text'
        os.environ['OUTPUT_MODE'] = output_mode
        os.environ['VISION_OCR_FIRST'] = '1' if output_mode == 'vision' else '0'
        os.environ['ENABLE_IMAGE_TRANSLATION'] = (
            '1' if output_mode in {'vision', 'image', 'video'} else '0'
        )
        os.environ['ENABLE_IMAGE_OUTPUT_MODE'] = (
            '1' if output_mode == 'image' else '0'
        )
        os.environ['ENABLE_VIDEO_OUTPUT_MODE'] = (
            '1' if output_mode == 'video' else '0'
        )
        os.environ['ENABLE_AUDIO_OUTPUT_MODE'] = (
            '1' if output_mode == 'audio' else '0'
        )
        os.environ['ENABLE_REFINEMENT_OUTPUT_MODE'] = (
            '1' if output_mode == 'refinement' else '0'
        )

        if getattr(self, '_direct_text_force_multipass_off', True):
            os.environ['MULTIPASS_MODE'] = '0'

        manual_glossary_path = str(
            getattr(self, '_direct_text_manual_glossary_path', '') or ''
        ).strip()
        use_manual_glossary = bool(
            getattr(self, '_direct_text_use_manual_glossary', False)
            and manual_glossary_path
            and os.path.isfile(manual_glossary_path)
        )
        if use_manual_glossary:
            os.environ['AUTO_GLOSSARY_MODE'] = 'off_no_automap'
            os.environ['ENABLE_AUTO_GLOSSARY'] = '0'
            os.environ['SINGLE_PASS_GLOSSARY_MODE'] = '0'
            os.environ['FUZZY_AUTO_MAPPING'] = '0'
            os.environ['APPEND_GLOSSARY'] = '1'
            os.environ['MANUAL_GLOSSARY'] = manual_glossary_path
            os.environ['DEFER_GLOSSARY_APPEND'] = '0'
        elif getattr(self, '_direct_text_force_no_glossary', True):
            os.environ['AUTO_GLOSSARY_MODE'] = 'no_glossary'
            os.environ['ENABLE_AUTO_GLOSSARY'] = '0'
            os.environ['SINGLE_PASS_GLOSSARY_MODE'] = '0'
            os.environ['FUZZY_AUTO_MAPPING'] = '0'
            os.environ['APPEND_GLOSSARY'] = '0'
            os.environ['MANUAL_GLOSSARY'] = ''
            os.environ['DEFER_GLOSSARY_APPEND'] = '0'

        if getattr(self, '_direct_text_skip_thinking', False):
            # Match the Manga custom-API disable-thinking override, and cover
            # GPT/OpenAI-compatible routes as well.
            os.environ['ENABLE_ANTHROPIC_THINKING'] = '0'
            os.environ['ANTHROPIC_THINKING_BUDGET'] = '0'
            os.environ['ANTHROPIC_FORCE_ADAPTIVE'] = '0'
            os.environ['ENABLE_GEMINI_THINKING'] = '0'
            os.environ['ENABLE_DEEPSEEK_THINKING'] = '0'
            os.environ['ENABLE_GPT_THINKING'] = '0'
            os.environ['GPT_REASONING_TOKENS'] = ''
            os.environ['GPT_EFFORT'] = 'none'
            os.environ['PASS_THINKING_TO_OPENAI_COMPATIBLE'] = '0'
            os.environ['GEMINI_THINKING_LEVEL'] = 'minimal'
            os.environ.pop('THINKING_BUDGET', None)
            os.environ['STREAM_THINKING_LOGS'] = '0'
            os.environ['AUTHND_STREAM_THINKING_LOGS'] = '0'
            os.environ['ENABLE_THOUGHTS'] = '0'

    def _format_translation_anti_duplicate_settings(self, env_vars=None):
        """Return the translation anti-duplicate startup log line, or empty if disabled."""
        env = env_vars or os.environ
        try:
            enabled = str(env.get('ENABLE_ANTI_DUPLICATE', '0')).strip() == '1'
        except Exception:
            enabled = False
        if not enabled:
            return ''
        top_p = f"{float(env.get('TOP_P', '1.0')):g}"
        min_p = str(env.get('MIN_P', '0.0'))
        top_k = str(env.get('TOP_K', '0'))
        freq = str(env.get('FREQUENCY_PENALTY', '0.0'))
        pres = str(env.get('PRESENCE_PENALTY', '0.0'))
        rep = str(env.get('REPETITION_PENALTY', '1.0'))
        return (
            f"🎯 Anti-duplicate enabled for translation "
            f"(top_p={top_p}, min_p={min_p}, top_k={top_k}, "
            f"freq_penalty={freq}, presence_penalty={pres}, repetition_penalty={rep})"
        )

    def _log_translation_anti_duplicate_settings(self, env_vars=None):
        """Log translation anti-duplicate settings once per GUI translation run."""
        if getattr(self, '_translation_anti_duplicate_logged', False):
            return
        message = self._format_translation_anti_duplicate_settings(env_vars)
        if not message:
            return
        self.append_log(message)
        self._translation_anti_duplicate_logged = True
        try:
            os.environ['TRANSLATION_ANTI_DUPLICATE_LOGGED'] = '1'
        except Exception:
            pass

    def _resolve_glossary_for_env(self, file_path: str) -> str:
        """Return the glossary path to use for a given input file.

        Priority:
        1) Per-EPUB mapping (manual_glossary_map) — used when multiple EPUBs are loaded
        2) Global manual_glossary_path — used for single-EPUB or manual load
        3) Empty string — no glossary
        """
        try:
            if self._current_auto_glossary_mode() == 'no_glossary':
                return ''
        except Exception:
            pass

        try:
            mgm = getattr(self, 'manual_glossary_map', None)
            if isinstance(mgm, dict) and mgm and file_path:
                key = os.path.normpath(os.path.abspath(file_path))
                gp = mgm.get(file_path) or mgm.get(key) or mgm.get(os.path.normpath(file_path))
                if gp and os.path.exists(gp):
                    return gp
        except Exception:
            pass

        try:
            if hasattr(self, 'manual_glossary_path') and self.manual_glossary_path:
                return self.manual_glossary_path
        except Exception:
            pass

        return ''

    def _get_environment_variables(self, epub_path, api_key):
        """Get all environment variables for translation/glossary"""

        # Get Google Cloud project ID if using Vertex AI
        google_cloud_project = ''
        model = self.model_var
        if '@' in model or model.startswith('vertex/'):
            google_creds = self.config.get('google_cloud_credentials')
            if google_creds and os.path.exists(google_creds):
                try:
                    with open(google_creds, 'r') as f:
                        creds_data = json.load(f)
                        google_cloud_project = creds_data.get('project_id', '')
                except:
                    pass
                    
        output_mode = self._active_translation_output_mode()

        # Handle extraction mode - check which variables exist
        if hasattr(self, 'text_extraction_method_var'):
            # New cleaner UI variables
            extraction_method = self.text_extraction_method_var
            filtering_level = self.file_filtering_level_var
            
            if extraction_method == 'enhanced':
                extraction_mode = 'enhanced'
                enhanced_filtering = filtering_level
            else:
                extraction_mode = filtering_level
                enhanced_filtering = 'smart'  # default
        else:
            # Old UI variables
            extraction_mode = self.extraction_mode_var
            extraction_method = 'enhanced' if extraction_mode == 'enhanced' else 'standard'
            if extraction_mode == 'enhanced':
                enhanced_filtering = getattr(self, 'enhanced_filtering_var', 'smart')
            else:
                enhanced_filtering = 'smart'

        if output_mode == 'vision':
            extraction_method = 'enhanced'
            extraction_mode = 'enhanced'
            enhanced_filtering = filtering_level if hasattr(self, 'file_filtering_level_var') else getattr(self, 'enhanced_filtering_var', 'smart')
                    
        # Ensure multi-key env toggles are set early for the main translation path as well,
        # then configure the in-memory key pools (key_pools.apply_key_pools_to_runtime).
        apply_key_pools_to_runtime(self.config)

        # CRITICAL: Use current GUI value for max_output_tokens, not the initial value
        # This ensures user changes via the button are reflected in image translation
        current_max_tokens = self.max_output_tokens
        resolved_max_retry_tokens = self._resolve_max_retry_tokens(current_max_tokens)
        resolved_max_retries = self._resolve_max_retries()
        
        auto_inject_book_title = bool(getattr(self, 'auto_inject_book_title_var', self.config.get('auto_inject_book_title', False)))
        auto_glossary_mode = self._current_auto_glossary_mode()
        (
            glossary_request_merging_enabled,
            glossary_request_merge_count,
            glossary_enable_chapter_split,
        ) = self._current_glossary_request_env(
            force_balanced_request_merging=(output_mode == 'vision' and auto_glossary_mode == 'balanced')
        )
        output_override_for_env = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')
        subtitle_output_info = self._subtitle_zip_output_info(epub_path)
        glossary_source_path = str(epub_path or '')
        if subtitle_output_info:
            archive_path = str(
                subtitle_output_info.get('archive_path') or ''
            ).strip()
            if archive_path:
                glossary_source_path = os.path.abspath(archive_path)
        subtitle_bundle_files = (
            list(subtitle_output_info.get('bundle_files') or [])
            if subtitle_output_info
            else []
        )
        subtitle_bundle_outputs = {}
        for subtitle_bundle_file in subtitle_bundle_files:
            bundle_member_info = self._subtitle_zip_output_info(
                subtitle_bundle_file
            )
            if bundle_member_info and bundle_member_info.get('output_path'):
                subtitle_bundle_outputs[
                    os.path.abspath(str(subtitle_bundle_file))
                ] = os.path.abspath(str(bundle_member_info['output_path']))
        if output_override_for_env:
            glossary_shared_dir = os.path.join(os.path.abspath(output_override_for_env), 'Glossary')
        else:
            glossary_shared_dir = os.path.join(_get_app_dir(), 'Glossary')
        try:
            from glossary_paths import migrate_all_legacy_glossary_files
            migrate_all_legacy_glossary_files(
                glossary_shared_dir,
                logger=lambda msg: self.append_log(msg) if hasattr(self, 'append_log') else None,
            )
        except Exception:
            pass

        def _bool_setting(value, default=False):
            if value is None:
                return bool(default)
            if hasattr(value, 'isChecked'):
                return bool(value.isChecked())
            if hasattr(value, 'get'):
                try:
                    return _bool_setting(value.get(), default)
                except Exception:
                    return bool(default)
            if isinstance(value, str):
                return value.strip().lower() in ('1', 'true', 'yes', 'on')
            return bool(value)

        def _positive_config_int(attr_name, config_key, default):
            try:
                value = getattr(self, attr_name, self.config.get(config_key, default))
                return str(max(1, int(value or default)))
            except (TypeError, ValueError):
                return str(default)

        def _bounded_config_int(attr_name, config_key, default, minimum=0, maximum=None):
            try:
                value = getattr(self, attr_name, self.config.get(config_key, default))
                numeric_value = int(float(value if value is not None else default))
                numeric_value = max(int(minimum), numeric_value)
                if maximum is not None:
                    numeric_value = min(int(maximum), numeric_value)
                return str(numeric_value)
            except (TypeError, ValueError):
                return str(default)

        def _bounded_config_float(attr_name, config_key, default, minimum=0.0, maximum=None):
            try:
                value = getattr(self, attr_name, self.config.get(config_key, default))
                numeric_value = float(value if value is not None else default)
                numeric_value = max(float(minimum), numeric_value)
                if maximum is not None:
                    numeric_value = min(float(maximum), numeric_value)
                return f"{numeric_value:g}"
            except (TypeError, ValueError):
                return f"{float(default):g}"

        def _bool_config_value(attr_name, config_key, default=False):
            value = getattr(self, attr_name, self.config.get(config_key, default))
            return _bool_setting(value, default)

        def _choice_config_value(attr_name, config_key, default, allowed):
            value = str(getattr(self, attr_name, self.config.get(config_key, default)) or default).strip().lower()
            return value if value in allowed else default

        def _authnd_effective_token_limits():
            if _bool_config_value('authnd_token_concurrency_auto_var', 'authnd_token_concurrency_auto', True):
                token_limit, subprocess_limit, _cores = _authnd_auto_token_limits()
                return str(token_limit), str(subprocess_limit), '1'
            return (
                _positive_config_int('authnd_token_concurrency_var', 'authnd_token_concurrency', 1),
                _positive_config_int('authnd_token_subprocess_concurrency_var', 'authnd_token_subprocess_concurrency', 1),
                '0',
            )

        authnd_token_limit, authnd_subprocess_limit, authnd_auto_flag = _authnd_effective_token_limits()

        def _explicit_blank_prompt_value(attr_name, config_key, default_attr):
            if hasattr(self, attr_name):
                value = getattr(self, attr_name)
                return str(value if value is not None else '')
            if config_key in self.config:
                value = self.config.get(config_key)
                return str(value if value is not None else '')
            return str(getattr(self, default_attr, '') or '')

        resolved_glossary = self._resolve_glossary_for_env(epub_path)
        glossary_mapping_source = (
            'manual'
            if resolved_glossary and getattr(self, 'manual_glossary_manually_loaded', False)
            else ('auto' if resolved_glossary else 'none')
        )

        env_vars = {
            'EPUB_PATH': epub_path,
            # Keep the archive identity while EPUB_PATH changes to an extracted
            # SRT/ASS/LRC member inside the translation engine.
            'GLOSSARY_SOURCE_PATH': glossary_source_path,
            'MODEL': self.model_var,
            'CONTEXTUAL': '1' if self.contextual_var else '0',
            'SEND_INTERVAL_SECONDS': str(self.delay_entry.text()),
            'API_QUEUE_SIZE': self.api_queue_entry.text().strip() or '4',
            'THREAD_SUBMISSION_DELAY_SECONDS': self.thread_delay_entry.text().strip() or '0.0001',
            'MAX_OUTPUT_TOKENS': str(current_max_tokens),
            'API_KEY': api_key,
            'OPENAI_API_KEY': api_key,
            'OPENAI_OR_Gemini_API_KEY': api_key,
            'GEMINI_API_KEY': api_key,
            'SYSTEM_PROMPT': self.prompt_text.toPlainText().strip(),
            'ASSISTANT_PROMPT': getattr(self, 'assistant_prompt', '') or '',  # Optional assistant prefill
            'ENABLE_TRANSLATION_CHUNK_PROMPT': '1' if getattr(self, 'enable_translation_chunk_prompt_var', self.config.get('enable_translation_chunk_prompt', False)) else '0',
            'ENABLE_CHUNK_PROGRESS': '1' if getattr(self, 'enable_chunk_progress_var', self.config.get('enable_chunk_progress', True)) else '0',
            'INCLUDE_PREVIOUS_CHUNK': '1' if getattr(self, 'include_previous_chunk_var', self.config.get('include_previous_chunk', False)) else '0',
            'PREVIOUS_CHUNK_CONTEXT_LIMIT': str(getattr(self, 'previous_chunk_context_limit_var', self.config.get('previous_chunk_context_limit', 3))),
            'TRANSLATION_CHUNK_PROMPT_ROLE': str(getattr(self, 'translation_chunk_prompt_role_var', self.config.get('translation_chunk_prompt_role', 'assistant')) or 'assistant').strip().lower(),
            'TRANSLATION_CHUNK_PROMPT': str(getattr(self, 'translation_chunk_prompt', self.config.get('translation_chunk_prompt', ''))),
            'TRANSLATE_BOOK_TITLE': "1" if self.translate_book_title_var else "0",
            'SKIP_TXT_TITLE_TRANSLATION': "1" if getattr(self, 'skip_txt_title_translation_var', True) else "0",
            'SKIP_PDF_TITLE_TRANSLATION': "1" if getattr(self, 'skip_pdf_title_translation_var', False) else "0",
            'SKIP_IMAGE_TITLE_TRANSLATION': "1" if getattr(self, 'skip_image_title_translation_var', True) else "0",
            'SKIP_TITLE_TAG_TRANSLATION': "1" if getattr(self, 'skip_title_tag_translation_var', False) else "0",
            'USE_TITLE': "0" if getattr(self, 'skip_title_tag_translation_var', False) else "1",
            'IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT': str(getattr(
                self,
                'image_only_title_tag_system_prompt',
                self.config.get(
                    'image_only_title_tag_system_prompt',
                    DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT,
                ),
            ) or DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT),
            'BOOK_TITLE_PROMPT': self.book_title_prompt,
            'BOOK_TITLE_SYSTEM_PROMPT': self.config.get('book_title_system_prompt', 
                "Translate this book title to {target_lang} while retaining any acronyms. Do not output anything other than the translated text."),
            'REMOVE_AI_ARTIFACTS': self.REMOVE_AI_ARTIFACTS_var if isinstance(self.REMOVE_AI_ARTIFACTS_var, str) else "off",
            'USE_ROLLING_SUMMARY': "1" if (hasattr(self, 'rolling_summary_var') and self.rolling_summary_var) else ("1" if self.config.get('use_rolling_summary') else "0"),
            'SUMMARY_ROLE': self.config.get('summary_role', 'system'),
            'ROLLING_SUMMARY_EXCHANGES': str(self.rolling_summary_exchanges_var),
            'ROLLING_SUMMARY_MODE': str(self.rolling_summary_mode_var),
            'ROLLING_SUMMARY_SYSTEM_PROMPT': str(self.rolling_summary_system_prompt),
            'ROLLING_SUMMARY_USER_PROMPT': str(self.rolling_summary_user_prompt),
            'ROLLING_SUMMARY_MAX_ENTRIES': str(self.rolling_summary_max_entries_var),
            'ROLLING_SUMMARY_MAX_TOKENS': str(self.rolling_summary_max_tokens_var),
            'PROFILE_NAME': self.lang_var.lower(),
            'TRANSLATION_TEMPERATURE': str(self.trans_temp.text()),
            'DISABLE_TEMPERATURE': '1' if self.disable_temperature_var else '0',
            'TRANSLATION_HISTORY_LIMIT': str(self.trans_history.text()),
            'EPUB_OUTPUT_DIR': _get_app_dir(),
            'OUTPUT_DIRECTORY': os.path.abspath(output_override_for_env) if output_override_for_env else '',
            'OUTPUT_DIR': os.path.abspath(output_override_for_env) if output_override_for_env else '',
            'SUBTITLE_OUTPUT_GROUP_DIR': (
                os.path.abspath(str(subtitle_output_info.get('output_dir')))
                if subtitle_output_info and subtitle_output_info.get('output_dir')
                else ''
            ),
            'SUBTITLE_OUTPUT_FILE': (
                os.path.abspath(str(subtitle_output_info.get('output_path')))
                if subtitle_output_info and subtitle_output_info.get('output_path')
                else ''
            ),
            'SUBTITLE_WORK_DIR': (
                os.path.abspath(str(subtitle_output_info.get('work_dir')))
                if subtitle_output_info and subtitle_output_info.get('work_dir')
                else ''
            ),
            'SUBTITLE_BUNDLE_FILES_JSON': (
                json.dumps(subtitle_bundle_files, ensure_ascii=False)
                if len(subtitle_bundle_files) > 1
                else ''
            ),
            'SUBTITLE_BUNDLE_OUTPUTS_JSON': (
                json.dumps(subtitle_bundle_outputs, ensure_ascii=False)
                if len(subtitle_bundle_files) > 1
                else ''
            ),
            'SUBTITLE_BUNDLE_WORK_DIR': (
                os.path.abspath(str(subtitle_output_info.get('bundle_work_dir')))
                if (
                    subtitle_output_info
                    and len(subtitle_bundle_files) > 1
                    and subtitle_output_info.get('bundle_work_dir')
                )
                else ''
            ),
            'SUBTITLE_PROGRESS_MIRROR_FILE': (
                os.path.join(
                    os.path.abspath(str(subtitle_output_info.get('output_dir'))),
                    'translation_progress.json',
                )
                if subtitle_output_info and subtitle_output_info.get('output_dir')
                else ''
            ),
            'GLOSSARY_SHARED_DIR': glossary_shared_dir,
            'SAVE_GLOSSARY_IN_OUTPUT': '1' if self.config.get('save_glossary_in_output', False) else '0',
            'GLOSSARY_OUTPUT_BACKUP_DIR': self._output_side_glossary_backup_dir_for_source(epub_path) if self.config.get('save_glossary_in_output', False) else '',
            # Whether to include previous source text as memory context
            'INCLUDE_SOURCE_IN_HISTORY': "1" if getattr(self, 'include_source_in_history_var', False) else "0",
            'APPEND_GLOSSARY': "0" if auto_glossary_mode == 'no_glossary' else ("1" if self.append_glossary_var else "0"),
            'APPEND_GLOSSARY_PROMPT': self.append_glossary_prompt if hasattr(self, 'append_glossary_prompt') and self.append_glossary_prompt else '- Follow this reference glossary for consistent translation (Do not output any raw entries):\n',
            'ADD_ADDITIONAL_GLOSSARY': "1" if self.config.get('add_additional_glossary', False) else "0",
            'ADDITIONAL_GLOSSARY_PATH': self.config.get('additional_glossary_path', ''),
            'GLOSSARY_MATCH_ENGINE': self._glossary_match_engine_env_value(),
            **self._strict_matching_env_dict(),
            **self._unified_glossary_env_dict(),
            'EMERGENCY_PARAGRAPH_RESTORE': "1" if self.emergency_restore_var else "0",

            'BREAK_SPLIT_COUNT': str(self.break_split_count_var) if hasattr(self, 'break_split_count_var') and self.break_split_count_var else '',
            'RETRY_TRUNCATED': "1" if self.retry_truncated_var else "0",
            'MAX_RETRY_TOKENS': str(resolved_max_retry_tokens),
            'TRUNCATION_RETRY_ATTEMPTS': str(self.truncation_retry_attempts_var),
            'RETRY_SPLIT_FAILED': "1" if bool(getattr(self, 'retry_split_failed_var', self.config.get('retry_split_failed', False))) else "0",
            'SPLIT_FAILED_RETRY_ATTEMPTS': str(max(1, int(str(getattr(self, 'split_failed_retry_attempts_var', self.config.get('split_failed_retry_attempts', '1'))).strip() or "1"))),
            'RETRY_DUPLICATE_BODIES': "1" if self.retry_duplicate_var else "0",
            'PRESERVE_ORIGINAL_TEXT_ON_FAILURE': "1" if self.preserve_original_text_var else "0",
            'SAVE_PARTIAL_RESULTS': "1" if getattr(self, 'save_partial_results_var', True) else "0",
            'SAVE_PROHIBITED_RESULTS': "1" if getattr(self, 'save_prohibited_results_var', False) else "0",
            'DISABLE_EMPTY_SAFETY_HEURISTIC': "1" if getattr(self, 'disable_empty_safety_heuristic_var', True) else "0",
            'DISABLE_QA_MARKER_CHECKS': "1" if getattr(self, 'disable_qa_marker_checks_var', True) else "0",
            'QA_MARKER_LENGTH_LIMIT': str(getattr(self, 'qa_marker_length_limit_var', '500')),
            'DISABLE_REFUSAL_CHECKS': "1" if getattr(self, 'disable_refusal_checks_var', False) else "0",
            'REFUSAL_PATTERN_LENGTH_LIMIT': str(getattr(self, 'refusal_pattern_length_limit_var', '1000')),
            'MISSING_FINISH_AS_PROHIBITED': "1" if getattr(self, 'unknown_finish_as_prohibited_var', False) else "0",
            'UNKNOWN_FINISH_AS_PROHIBITED': "1" if getattr(self, 'unknown_finish_as_prohibited_var', False) else "0",
            'DUPLICATE_LOOKBACK_CHAPTERS': str(self.duplicate_lookback_var),
            'GLOSSARY_MIN_FREQUENCY': str(self.glossary_min_frequency_var),
            'GLOSSARY_MAX_NAMES': str(self.glossary_max_names_var),
            'GLOSSARY_MAX_TITLES': str(self.glossary_max_titles_var),
            'CONTEXT_WINDOW_SIZE': str(self.context_window_size_var),
            'GLOSSARY_STRIP_HONORIFICS': "1" if self.strip_honorifics_var else "0",
            'GLOSSARY_CHAPTER_SPLIT_THRESHOLD': str(self.glossary_chapter_split_threshold_var),
            'GLOSSARY_FILTER_MODE': self.glossary_filter_mode_var,
            'GLOSSARY_SKIP_FREQUENCY_CHECK': "1" if self.config.get('glossary_skip_frequency_check', False) else "0",
            'GLOSSARY_INCLUDE_BOOK_TITLE': "1" if getattr(self, 'include_book_title_glossary_var', False) else "0",
            'GLOSSARY_AUTO_INJECT_BOOK_TITLE': "1" if auto_inject_book_title else "0",
            'GLOSSARY_REQUEST_MERGING_ENABLED': glossary_request_merging_enabled,
            'GLOSSARY_REQUEST_MERGE_COUNT': glossary_request_merge_count,
            'GLOSSARY_ENABLE_CHAPTER_SPLIT': glossary_enable_chapter_split,
            'GLOSSARY_SKIP_TITLE_HEADER_ONLY': self._glossary_skip_title_header_only_env_value(),
            'GLOSSARY_ADD_MINIMAL_PASS': self._glossary_add_minimal_pass_env_value(),
            'GLOSSARY_REQUIRE_COMPLETE_BEFORE_TRANSLATION': '1' if self._live_bool_setting(
                'glossary_require_complete_checkbox',
                'glossary_require_complete_before_translation_var',
                'glossary_require_complete_before_translation',
                False,
            ) else '0',
            'GLOSSARY_SKIP_API_ERROR_RETRIES': '1' if self._live_bool_setting(
                'glossary_skip_api_error_retries_checkbox',
                'glossary_skip_api_error_retries_var',
                'glossary_skip_api_error_retries',
                False,
            ) else '0',
            'GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL': '1' if getattr(self, 'never_consider_in_between_files_as_special_var', self.config.get('never_consider_in_between_files_as_special', True)) else '0',
            'ENABLE_AUTO_GLOSSARY': "1" if auto_glossary_mode == 'minimal' else "0",
            'AUTO_GLOSSARY_MODE': auto_glossary_mode,
            'SINGLE_PASS_GLOSSARY_MODE': '1' if auto_glossary_mode == 'single_pass' else '',
            'SINGLE_PASS_GLOSSARY_HEADER_PROMPT': self.config.get('single_pass_glossary_header_prompt', ''),
            'AUTO_GLOSSARY_PROMPT': self.unified_auto_glosary_prompt3 if hasattr(self, 'unified_auto_glosary_prompt3') else '',
            'GLOSSARY_REFINEMENT_ENABLED': '1' if self.config.get('glossary_refinement_enabled', False) else '0',
            'GLOSSARY_REFINEMENT_SYSTEM_PROMPT': self.config.get('glossary_refinement_system_prompt') or getattr(self, 'glossary_refinement_system_prompt', ''),
            'GLOSSARY_REFINEMENT_USER_PROMPT': self.config.get('glossary_refinement_user_prompt', getattr(self, 'glossary_refinement_user_prompt', '')),
            'GLOSSARY_REFINEMENT_TYPE_MODE': self.config.get('glossary_refinement_type_mode', 'all'),
            'GLOSSARY_REFINEMENT_SELECTED_TYPES': ','.join(self.config.get('glossary_refinement_selected_types', [])),
            'GLOSSARY_REFINEMENT_CHUNKING_MODE': self.config.get('glossary_refinement_chunking_mode', 'all'),
            'GLOSSARY_REFINEMENT_SKIP_DEDUPE': '1' if self.config.get('glossary_refinement_skip_dedupe', False) else '0',
            'GLOSSARY_REFINEMENT_WAIT_FOR_COMPLETION': '1' if self.config.get('glossary_refinement_wait_for_completion', False) else '0',
            'GLOSSARY_REFINEMENT_REOPEN_ON_SOURCE_CHANGE': '1' if self.config.get('glossary_refinement_reopen_on_source_change', False) else '0',
            'APPEND_GLOSSARY_PROMPT': self.append_glossary_prompt if hasattr(self, 'append_glossary_prompt') and self.append_glossary_prompt else '- Follow this reference glossary for consistent translation (Do not output any raw entries):\n',
            'GLOSSARY_TRANSLATION_PROMPT': self.glossary_translation_prompt if hasattr(self, 'glossary_translation_prompt') else '',
            'GLOSSARY_FORMAT_INSTRUCTIONS': self.glossary_format_instructions if hasattr(self, 'glossary_format_instructions') else '',
            'GLOSSARY_USE_LEGACY_CSV': '1' if self.use_legacy_csv_var else '0',
            'GLOSSARY_OUTPUT_LEGACY_JSON': '1' if getattr(self, 'glossary_output_legacy_json_var', False) else '0',
            'GLOSSARY_SKIP_GENDER_TRACKING': '1' if getattr(self, 'glossary_skip_gender_tracking_var', False) else '0',
            'GLOSSARY_GENDER_NOISE_THRESHOLD': str(getattr(self, 'glossary_gender_noise_threshold_var', self.config.get('glossary_gender_noise_threshold', 10))),
            'GLOSSARY_GENDER_TRACKING_BIAS': str(getattr(self, 'glossary_gender_tracking_bias_var', self.config.get('glossary_gender_tracking_bias', 'none'))),
            'GLOSSARY_PARTIAL_RATIO_GENDER_ONLY': '1' if getattr(self, 'glossary_partial_ratio_gender_only_var', False) else '0',
            'GLOSSARY_ALIAS_AWARE_NAME_MATCHING': '1' if getattr(self, 'glossary_alias_aware_name_matching_var', False) else '0',
            'GLOSSARY_ALIAS_AWARE_GENDER_ONLY': '1' if getattr(self, 'glossary_alias_aware_gender_only_var', True) else '0',
            'COMPRESS_GLOSSARY_CONSIDER_TRANSLATED_COLUMN': '1' if getattr(self, 'compress_glossary_consider_translated_column_var', self.config.get('compress_glossary_consider_translated_column', False)) else '0',
            'COMPRESS_GLOSSARY_MULTIPASS_EXCLUDE_MATCHING': '1' if getattr(self, 'compress_glossary_multipass_exclude_matching_var', self.config.get('compress_glossary_multipass_exclude_matching', True)) else '0',
            'GLOSSARY_CUSTOM_ENTRY_TYPES': json.dumps(getattr(self, 'custom_entry_types', self.config.get('custom_entry_types', {}))),
            'GLOSSARY_CUSTOM_FIELDS': json.dumps(getattr(self, 'custom_glossary_fields', self.config.get('custom_glossary_fields', []))),
            'OUTPUT_MODE': output_mode,
            'VISION_OCR_FIRST': '1' if output_mode == 'vision' else '0',
            'ENABLE_STREAMING': '1' if bool(getattr(self, 'enable_streaming_var', self.config.get('enable_streaming', False))) else '0',
            'ALLOW_BATCH_STREAM_LOGS': '1' if bool(getattr(self, 'allow_batch_stream_logs_var', self.config.get('allow_batch_stream_logs', False))) else '0',
            'ALLOW_AUTHGPT_BATCH_STREAM_LOGS': '1' if bool(getattr(self, 'allow_authgpt_batch_stream_logs_var', self.config.get('allow_authgpt_batch_stream_logs', False))) else '0',
            'STREAM_THINKING_LOGS': '1' if bool(getattr(self, 'stream_thinking_logs_var', self.config.get('stream_thinking_logs', False))) else '0',
            'AUTHZA_USE_GENERAL_API': '1' if bool(getattr(self, 'authza_use_general_api_var', self.config.get('authza_use_general_api', False))) else '0',
            'AUTHND_TOKEN_CONCURRENCY_AUTO': authnd_auto_flag,
            'AUTHND_TOKEN_CONCURRENCY': authnd_token_limit,
            'AUTHND_TOKEN_SUBPROCESS_CONCURRENCY': authnd_subprocess_limit,
            'AUTHND_TOKEN_TIMEOUT': _bounded_config_int('authnd_token_timeout_var', 'authnd_token_timeout', 180, 30, 600),
            'ORDERED_BATCH_DISPATCH_TIMEOUT': _bounded_config_int('dispatch_order_timeout_var', 'dispatch_order_timeout', 3, 0, 120),
            'GEMINI_FREE_ADAPTIVE_SPLIT': '1' if _bool_config_value('gemini_free_adaptive_split_var', 'gemini_free_adaptive_split', True) else '0',
            'GLOSSARION_TOR_ENABLED': '1' if _bool_config_value('tor_proxy_enabled_var', 'tor_proxy_enabled', False) else '0',
            'GEMINI_FREE_HTML_TEXT_NODE_TRANSPORT': '1' if _bool_config_value('gemini_free_html_text_node_transport_var', 'gemini_free_html_text_node_transport', True) else '0',
            'GEMINI_FREE_SUBCHUNK_PROMPT_CHARS': _bounded_config_int('gemini_free_subchunk_prompt_chars_var', 'gemini_free_subchunk_prompt_chars', 7000, 300, 7000),
            'GEMINI_FREE_SUBCHUNK_URL_CHARS': _bounded_config_int('gemini_free_subchunk_url_chars_var', 'gemini_free_subchunk_url_chars', 14500, 1000, 200000),
            'GEMINI_FREE_SUBCHUNK_SAFETY_CHARS': _bounded_config_int('gemini_free_subchunk_safety_chars_var', 'gemini_free_subchunk_safety_chars', 600, 0, 20000),
            'GEMINI_FREE_MIN_SUBCHUNK_BODY_CHARS': _bounded_config_int('gemini_free_min_subchunk_body_chars_var', 'gemini_free_min_subchunk_body_chars', 80, 1, 50000),
            'GEMINI_FREE_SUBCHUNK_CONCURRENCY': _bounded_config_int('gemini_free_subchunk_concurrency_var', 'gemini_free_subchunk_concurrency', 0, 0, 64),
            'GEMINI_FREE_SUBCHUNK_START_DELAY': _bounded_config_float('gemini_free_subchunk_start_delay_var', 'gemini_free_subchunk_start_delay', 5.0, 0.0, 60.0),
            'GEMINI_FREE_SUBCHUNK_TIMEOUT': _bounded_config_int('gemini_free_subchunk_timeout_var', 'gemini_free_subchunk_timeout', 0, 0, 7200),
            'GEMINI_FREE_SUBCHUNK_PAYLOAD_FORMAT': _choice_config_value('gemini_free_subchunk_payload_format_var', 'gemini_free_subchunk_payload_format', 'auto', {'auto', 'html', 'text'}),
            'GEMINI_FREE_HTML_SPLITTER': _choice_config_value('gemini_free_html_splitter_var', 'gemini_free_html_splitter', 'beautifulsoup4', {'beautifulsoup4', 'regex'}),
            'GEMINI_FREE_SUBCHUNK_BALANCER': _choice_config_value('gemini_free_subchunk_balancer_var', 'gemini_free_subchunk_balancer', 'balanced', {'balanced', 'greedy'}),
            'VISION_OCR_PROMPT': str(getattr(self, 'vision_ocr_prompt', self.config.get('vision_ocr_prompt', ''))),
            'VISION_OCR_USER_PROMPT': str(getattr(self, 'vision_ocr_user_prompt', self.config.get('vision_ocr_user_prompt', ''))),
            'VISION_OCR_COMBINED_CONTEXT_PROMPT': str(getattr(self, 'vision_ocr_combined_context_prompt', self.config.get('vision_ocr_combined_context_prompt', ''))),
            'VISION_OCR_TRANSLATION_USER_PROMPT': str(getattr(self, 'vision_ocr_translation_user_prompt', self.config.get('vision_ocr_translation_user_prompt', ''))),
            'VISION_OCR_BATCH_TRANSLATION': '1' if _bool_setting(getattr(self, 'vision_ocr_batch_translation_var', None), self.config.get('vision_ocr_batch_translation', True)) else '0',
            'VISION_OCR_BATCH_SIZE': str(getattr(self, 'vision_ocr_batch_size_var', self.config.get('vision_ocr_batch_size', '-1'))),
            'VISION_OCR_SKIP_TRANSLATION': '1' if _bool_setting(getattr(self, 'vision_ocr_skip_translation_var', None), self.config.get('vision_ocr_skip_translation', False)) else '0',
            'VISION_OCR_KEEP_IMAGES': '1' if _bool_setting(getattr(self, 'vision_ocr_keep_images_var', None), self.config.get('vision_ocr_keep_images', False)) else '0',
            'VISION_OCR_SOURCE_PREPASS': str(getattr(self, 'vision_ocr_source_prepass_var', self.config.get('vision_ocr_source_prepass', 'auto')) or 'auto'),
            'ENABLE_REFINEMENT_OUTPUT_MODE': "1" if output_mode == 'refinement' else "0",
            'REFINEMENT_SYSTEM_PROMPT': self.config.get('refinement_system_prompt') or getattr(self, 'refinement_system_prompt', getattr(self, 'default_refinement_system_prompt', '')),
            'REFINEMENT_USER_PROMPT': self.config.get('refinement_user_prompt', getattr(self, 'refinement_user_prompt', '')),
            'REFINEMENT_FULL_WITH_RAW_SYSTEM_PROMPT': getattr(self, 'refinement_full_with_raw_system_prompt', None) or self.config.get('refinement_full_with_raw_system_prompt') or getattr(self, 'default_refinement_full_with_raw_system_prompt', ''),
            'REFINEMENT_FULL_WITH_RAW_USER_PROMPT': getattr(self, 'refinement_full_with_raw_user_prompt', self.config.get('refinement_full_with_raw_user_prompt', getattr(self, 'default_refinement_full_with_raw_user_prompt', ''))),
            'REFINEMENT_FULL_WITH_RAW_RAW_ROLE': str(getattr(self, 'refinement_full_with_raw_raw_role_var', self.config.get('refinement_full_with_raw_raw_role', 'assistant')) or 'assistant').strip().lower(),
            'REFINEMENT_FULL_WITH_RAW_RAW_HEADER': _explicit_blank_prompt_value('refinement_full_with_raw_raw_header', 'refinement_full_with_raw_raw_header', 'default_refinement_full_with_raw_raw_header'),
            'REFINEMENT_FULL_WITH_RAW_RAW_FOOTER': _explicit_blank_prompt_value('refinement_full_with_raw_raw_footer', 'refinement_full_with_raw_raw_footer', 'default_refinement_full_with_raw_raw_footer'),
            'REFINEMENT_FAILED_SYSTEM_PROMPT': getattr(self, 'refinement_failed_system_prompt', None) or self.config.get('refinement_failed_system_prompt') or getattr(self, 'default_refinement_failed_system_prompt', ''),
            'REFINEMENT_FAILED_USER_PROMPT': getattr(self, 'refinement_failed_user_prompt', self.config.get('refinement_failed_user_prompt', getattr(self, 'default_refinement_failed_user_prompt', ''))),
            'REFINEMENT_PARTIAL_SYSTEM_PROMPT': getattr(self, 'refinement_partial_system_prompt', None) or self.config.get('refinement_partial_system_prompt') or getattr(self, 'default_refinement_partial_system_prompt', ''),
            'REFINEMENT_PARTIAL_USER_PROMPT': getattr(self, 'refinement_partial_user_prompt', self.config.get('refinement_partial_user_prompt', getattr(self, 'default_refinement_partial_user_prompt', ''))),
            'REFINEMENT_PARTIAL_B_SYSTEM_PROMPT': _explicit_blank_prompt_value('refinement_partial_b_system_prompt', 'refinement_partial_b_system_prompt', 'default_refinement_partial_b_system_prompt'),
            'REFINEMENT_PARTIAL_B_USER_PROMPT': _explicit_blank_prompt_value('refinement_partial_b_user_prompt', 'refinement_partial_b_user_prompt', 'default_refinement_partial_b_user_prompt'),
            'REFINEMENT_PARTIAL_B2_SYSTEM_PROMPT': _explicit_blank_prompt_value('refinement_partial_b2_system_prompt', 'refinement_partial_b2_system_prompt', 'default_refinement_partial_b2_system_prompt'),
            'REFINEMENT_PARTIAL_B2_USER_PROMPT': _explicit_blank_prompt_value('refinement_partial_b2_user_prompt', 'refinement_partial_b2_user_prompt', 'default_refinement_partial_b2_user_prompt'),
            'PARTIAL_B2_ENTRIES_PER_REQUEST': str(getattr(self, 'partial_b2_entries_per_request_var', self.config.get('partial_b2_entries_per_request', '-1'))),
            'ENABLE_IMAGE_TRANSLATION': "1" if (self.enable_image_translation_var and output_mode not in ('refinement', 'audio')) else "0",
            'PROCESS_WEBNOVEL_IMAGES': "1" if self.process_webnovel_images_var else "0",
            'WEBNOVEL_MIN_HEIGHT': str(self.webnovel_min_height_var),
            'MAX_IMAGES_PER_CHAPTER': str(self.max_images_per_chapter_var),
            'IMAGE_API_DELAY': '1.0',
            'SAVE_IMAGE_TRANSLATIONS': '1',
            'IMAGE_CHUNK_HEIGHT': str(self.image_chunk_height_var),
            'IMAGE_CHUNK_OVERLAP_PERCENT': str(getattr(self, 'image_chunk_overlap_var', '3')),
            'IMAGE_CHUNK_MIN_OVERLAP_PIXELS': str(getattr(self, 'image_chunk_min_overlap_pixels_var', '80')),
            'IMAGE_SMART_CHUNKING': '1' if getattr(self, 'image_smart_chunking_var', True) else '0',
            'VISION_OCR_FUZZY_CHUNK_DEDUPE': '1' if getattr(self, 'vision_ocr_fuzzy_chunk_dedupe_var', False) else '0',
            'HIDE_IMAGE_TRANSLATION_LABEL': "1" if self.hide_image_translation_label_var else "0",
            'RETRY_TIMEOUT': "1" if getattr(self, 'retry_timeout_var', self.config.get('retry_timeout', False)) else "0",
            'CHUNK_TIMEOUT': str(self.chunk_timeout_var),
            'TIMEOUT_RETRY_ATTEMPTS': str(getattr(self, 'timeout_retry_attempts_var', self.config.get('timeout_retry_attempts', 2))),
            # New network/HTTP controls
            'ENABLE_HTTP_TUNING': '1' if self.config.get('enable_http_tuning', False) else '0',
            'CONNECT_TIMEOUT': str(self.config.get('connect_timeout', os.environ.get('CONNECT_TIMEOUT', '10'))),
            'READ_TIMEOUT': str(self.config.get('read_timeout', os.environ.get('READ_TIMEOUT', os.environ.get('CHUNK_TIMEOUT', '1800')))),
            'HTTP_POOL_CONNECTIONS': str(self.config.get('http_pool_connections', os.environ.get('HTTP_POOL_CONNECTIONS', '20'))),
            'HTTP_POOL_MAXSIZE': str(self.config.get('http_pool_maxsize', os.environ.get('HTTP_POOL_MAXSIZE', '50'))),
            'IGNORE_RETRY_AFTER': '1' if (hasattr(self, 'ignore_retry_after_var') and self.ignore_retry_after_var) else '0',
            'MAX_RETRIES': str(resolved_max_retries),
            'INDEFINITE_RATE_LIMIT_RETRY': '1' if self.config.get('indefinite_rate_limit_retry', False) else '0',
            # Scanning/QA settings
            'SCAN_PHASE_ENABLED': '1' if getattr(self, 'scan_phase_enabled_var', self.config.get('scan_phase_enabled', False)) else '0',
            'SCAN_PHASE_MODE': self._get_scan_phase_mode(),
            'QA_SCANNER_SETTINGS_JSON': self._get_qa_scanner_settings_json(),
            'QA_AUTO_SEARCH_OUTPUT': '1' if self.config.get('qa_auto_search_output', True) else '0',
            'BATCH_TRANSLATION': "1" if self.batch_translation_var else "0",
            'BATCH_SIZE': str(self.batch_size_var),
            'MULTIPASS_MODE': "1" if getattr(self, 'multipass_mode_var', self.config.get('multipass_mode', False)) else "0",
            'MULTIPASS_REFINEMENT_MODE': self._get_multipass_refinement_mode(),
            'CHAPTER_RANGE': self.chapter_range_entry.text().strip(),
            'USE_SPINE_ORDER': '1' if self.use_spine_order_checkbox.isChecked() else '0',
            'BATCHING_MODE': self._translation_batching_mode_for_env(),
            'BATCH_GROUP_SIZE': str(getattr(self, 'batch_group_size_var', '3')),
            # Backward compatibility for older scripts expecting CONSERVATIVE_BATCHING
            'CONSERVATIVE_BATCHING': "1" if self._translation_batching_mode_for_env() == 'conservative' else "0",
            'DISABLE_ZERO_DETECTION': "1" if self.disable_zero_detection_var else "0",
            'TRANSLATION_HISTORY_ROLLING': "1",
            'USE_GEMINI_OPENAI_ENDPOINT': '1' if self.use_gemini_openai_endpoint_var else '0',
            'GEMINI_OPENAI_ENDPOINT': self.gemini_openai_endpoint_var if self.gemini_openai_endpoint_var else 'generativelanguage.googleapis.com',
            'OVERRIDE_GEMMA_FOR_CUSTOM_ENDPOINT': '1' if getattr(self, 'override_gemma_for_custom_endpoint_var', True) else '0',
            'FORCE_NATIVE_ANTHROPIC': '1' if getattr(self, 'force_native_anthropic_var', False) else '0',
            'ANTHROPIC_BASE_URL': getattr(self, 'anthropic_base_url_var', '') or '',
            'FUZZY_AUTO_MAPPING': '1' if getattr(self, 'fuzzy_auto_mapping_var', False) else '0',
            'FUZZY_AUTO_MAPPING_THRESHOLD': str(getattr(self, 'fuzzy_auto_mapping_threshold_var', 80)),
            "ATTACH_CSS_TO_CHAPTERS": "1" if self.attach_css_to_chapters_var else "0",
            "EPUB_USE_HTML_METHOD": "1" if self.epub_use_html_method_var else "0",
            "EPUB_CSS_OVERRIDE_PATH": getattr(self, 'epub_css_override_path_var', self.config.get('epub_css_override_path', '')) or '',
            'GLOSSARY_FUZZY_THRESHOLD': str(self.config.get('glossary_fuzzy_threshold', 0.90)),
            'GLOSSARY_ENTRY_TYPE_FILTER_MODE': self.config.get('glossary_entry_type_filter_mode', 'Loose'),
            'GLOSSARY_MAX_TEXT_SIZE': str(self.config.get('glossary_max_text_size', 0)),
            'GLOSSARY_MAX_SENTENCES': str(self.config.get('glossary_max_sentences', 200)),
            'USE_FALLBACK_KEYS': '1' if self.config.get('use_fallback_keys', False) else '0',
            'USE_MAIN_KEY_FALLBACK': '1' if self.config.get('use_main_key_fallback', True) else '0',
            'FALLBACK_KEY_SHUFFLE': '1' if self.config.get('fallback_key_shuffle', False) else '0',
            'FALLBACK_KEYS': json.dumps(self.config.get('fallback_keys', [])),
            'USE_GLOSSARY_KEYS': '1' if self.config.get('use_glossary_keys', False) else '0',
            'GLOSSARY_API_KEYS': json.dumps(self.config.get('glossary_keys', [])),
            'USE_GLOSSARY_REFINEMENT_KEYS': '1' if self.config.get('use_glossary_refinement_keys', False) else '0',
            'GLOSSARY_REFINEMENT_API_KEYS': json.dumps(self.config.get('glossary_refinement_keys', [])),
            'USE_METADATA_KEYS': '1' if self.config.get('use_metadata_keys', False) else '0',
            'METADATA_API_KEYS': json.dumps(self.config.get('metadata_keys', [])),
            'USE_VISION_KEYS': '1' if self.config.get('use_qa_scan_keys', False) else '0',
            'VISION_API_KEYS': json.dumps(self.config.get('qa_scan_keys', [])),
            'USE_QA_SCAN_KEYS': '1' if self.config.get('use_qa_scan_keys', False) else '0',
            'QA_SCAN_API_KEYS': json.dumps(self.config.get('qa_scan_keys', [])),
            'USE_AI_TRUNCATION_DETECTION_KEYS': '1' if self.config.get('use_ai_truncation_detection_keys', False) else '0',
            'AI_TRUNCATION_DETECTION_API_KEYS': json.dumps(self.config.get('ai_truncation_detection_keys', [])),
            'USE_ROLLING_SUMMARY_KEYS': '1' if self.config.get('use_rolling_summary_keys', False) else '0',
            'ROLLING_SUMMARY_API_KEYS': json.dumps(self.config.get('rolling_summary_keys', [])),
            'USE_TRUNCATION_RETRY_KEYS': '1' if self.config.get('use_truncation_retry_keys', False) else '0',
            'TRUNCATION_RETRY_API_KEYS': json.dumps(self.config.get('truncation_retry_keys', [])),

            # Extraction settings
            "EXTRACTION_MODE": extraction_mode,
            "ENHANCED_FILTERING": enhanced_filtering,
            "ENHANCED_PRESERVE_STRUCTURE": "1" if getattr(self, 'enhanced_preserve_structure_var', True) else "0",
            "ENHANCED_SINGLE_LINE_BREAK": "1" if getattr(self, 'enhanced_single_line_break_var', False) else "0",
            "CONVERT_BR_TO_PARAGRAPHS": "1" if getattr(self, 'convert_br_to_paragraphs_var', True) else "0",
            "PRESERVE_ASTERISK_SEPARATOR_LINES": "1" if getattr(self, 'preserve_asterisk_separator_lines_var', True) else "0",
            "SKIP_MARKDOWN_TO_HTML": "1" if getattr(self, 'skip_markdown_to_html_var', False) else "0",
            "USE_MARKDOWN2_CONVERTER": "1" if getattr(self, 'use_markdown2_converter_var', False) else "0",
            'FORCE_BS_FOR_TRADITIONAL': '1' if getattr(self, 'force_bs_for_traditional_var', False) else '0',
            'OUTPUT_SDLXLIFF': '1' if getattr(self, 'output_sdlxliff_var', True) else '0',
            'OUTPUT_MD': '1' if getattr(self, 'output_md_var', False) else '0',
            'OUTPUT_TXT': '1' if getattr(self, 'output_txt_var', False) else '0',
            'FIX_STRAY_P_GT_EPUB': '1' if getattr(self, 'fix_stray_p_gt_epub_var', self.config.get('fix_stray_p_gt_epub', False)) else '0',
            'FIX_STRAY_P_GT_BS': '1' if getattr(self, 'fix_stray_p_gt_bs_var', self.config.get('fix_stray_p_gt_bs', False)) else '0',

            # For new UI
            "TEXT_EXTRACTION_METHOD": extraction_method if hasattr(self, 'text_extraction_method_var') or output_mode == 'vision' else ('enhanced' if extraction_mode == 'enhanced' else 'standard'),
            "FILE_FILTERING_LEVEL": filtering_level if hasattr(self, 'file_filtering_level_var') else extraction_mode,
            "USE_HTML2TEXT": "1" if output_mode == 'vision' or extraction_method in ('enhanced', 'html2text', 'markdown') or extraction_mode == 'enhanced' else "0",
            'DISABLE_CHAPTER_MERGING': '1' if self.disable_chapter_merging_var else '0',
            # Request merging (combine multiple chapters into single API request)
            'REQUEST_MERGING_ENABLED': "1" if getattr(self, 'request_merging_enabled_var', False) else "0",
            'REQUEST_MERGE_COUNT': str(getattr(self, 'request_merge_count_var', '3')),
            'SPLIT_THE_MERGE': "1" if getattr(self, 'split_the_merge_var', False) else "0",
            'DISABLE_MERGE_FALLBACK': "1" if getattr(self, 'disable_merge_fallback_var', False) else "0",
            'SYNTHETIC_MERGE_HEADERS': "1" if getattr(self, 'synthetic_merge_headers_var', True) else "0",
            'DISABLE_EPUB_GALLERY': "1" if self.disable_epub_gallery_var else "0",
            'SKIP_NON_SPINE_SPECIAL_FILES': "1" if getattr(self, 'skip_non_spine_special_files_var', False) else "0",
            'SKIP_UNREFERENCED_EPUB_IMAGES': "1" if getattr(self, 'skip_unreferenced_epub_images_var', False) else "0",
            'DISABLE_AUTOMATIC_COVER_CREATION': "1" if getattr(self, 'disable_automatic_cover_creation_var', True) else "0",
            'TRANSLATE_SPECIAL_FILES': "1" if getattr(self, 'translate_special_files_var', False) else "0",
            'TRANSLATE_ALL_NUMBERED_HTML': "1" if getattr(self, 'translate_all_numbered_html_var', True) else "0",
            'USE_P_TAG_TOC_FALLBACK': "1" if getattr(self, 'use_p_tag_toc_fallback_var', False) else "0",
            'DEDUPLICATE_TOC': "1" if getattr(self, 'deduplicate_toc_var', False) else "0",
            'DEDUPLICATE_TOC_USE_TRANSLATED': "1" if getattr(self, 'deduplicate_toc_use_translated_var', False) else "0",
            'SKIP_DUPLICATE_TOC_TRANSLATION': "1" if getattr(self, 'skip_duplicate_toc_translation_var', False) else "0",
            'DUPLICATE_DETECTION_MODE': str(self.duplicate_detection_mode_var),
            'CHAPTER_NUMBER_OFFSET': str(self.chapter_number_offset_var), 
            'USE_HEADER_AS_OUTPUT': "0",
            'ENABLE_DECIMAL_CHAPTERS': "1" if self.enable_decimal_chapters_var else "0",
            'ENABLE_WATERMARK_REMOVAL': "1" if self.enable_watermark_removal_var else "0",
            'ADVANCED_WATERMARK_REMOVAL': "1" if self.advanced_watermark_removal_var else "0",
            'SAVE_CLEANED_IMAGES': "1" if self.save_cleaned_images_var else "0",
            'EMERGENCY_IMAGE_RESTORE': "1" if getattr(self, 'emergency_image_restore_var', False) else "0",
            'EMERGENCY_GLOSSARY_COMPLIANCE': "1" if (getattr(self, 'emergency_glossary_compliance_var', False) and auto_glossary_mode != 'no_glossary') else "0",
            'EMERGENCY_GLOSSARY_COMPLIANCE_MODE': str(getattr(self, 'emergency_glossary_compliance_mode_var', 'characters')),
            'EMERGENCY_GLOSSARY_COMPLIANCE_CUSTOM_TYPES': json.dumps(getattr(self, 'emergency_glossary_compliance_custom_types_var', [])),
            'EMERGENCY_GLOSSARY_COMPLIANCE_MIN_CHARS': str(getattr(self, 'emergency_glossary_compliance_min_chars_var', 3)),
            'COMPRESS_GLOSSARY_PROMPT': '1' if self.config.get('compress_glossary_prompt') else '0',
            'COMPRESSION_FACTOR': str(self.compression_factor_var),
            'DISABLE_GEMINI_SAFETY': str(self.config.get('disable_gemini_safety', False)).lower(),
            'GEMINI_SAFETY_THRESHOLD': str(self.config.get('gemini_safety_threshold', 'BLOCK_NONE')),
            'GLOSSARY_DUPLICATE_KEY_MODE': self.config.get('glossary_duplicate_key_mode', 'auto'),
            'GLOSSARY_DUPLICATE_CUSTOM_FIELD': self.config.get('glossary_duplicate_custom_field', ''),
            'MANUAL_GLOSSARY': resolved_glossary,
            'GLOSSARY_MAPPING_SOURCE': glossary_mapping_source,
            'FORCE_NCX_ONLY': '1' if self.force_ncx_only_var else '0',
            'SINGLE_API_IMAGE_CHUNKS': "1" if self.single_api_image_chunks_var else "0",
            'ENABLE_GEMINI_THINKING': "1" if self.enable_gemini_thinking_var else "0",
            'THINKING_BUDGET': self.thinking_budget_var if self.enable_gemini_thinking_var else '0',
            'GEMINI_THINKING_LEVEL': getattr(self, 'thinking_level_var', 'high'),
            'GEMINI_SERVICE_TIER': getattr(self, 'gemini_service_tier_var', 'off'),
            'FORCE_SERVICE_TIER_UNKNOWN_ROUTES': '1' if self.force_service_tier_unknown_routes_var else '0',
            # GPT/OpenRouter reasoning
            'ENABLE_GPT_THINKING': "1" if self.enable_gpt_thinking_var else "0",
            'GPT_REASONING_TOKENS': self.gpt_reasoning_tokens_var if self.enable_gpt_thinking_var else '',
            'GPT_EFFORT': self.gpt_effort_var,
            'OPENROUTER_USE_REASONING_TOKENS': '1' if self.openrouter_use_reasoning_tokens_var else '0',
            'PASS_THINKING_TO_OPENAI_COMPATIBLE': "1" if getattr(self, 'pass_thinking_all_openai_var', False) else "0",
            # DeepSeek thinking (DeepSeek OpenAI-compatible API)
            'ENABLE_DEEPSEEK_THINKING': "1" if getattr(self, 'enable_deepseek_thinking_var', True) else "0",
            'DEEPSEEK_EFFORT': getattr(self, 'deepseek_effort_var', 'high'),
            'DEEPSEEK_USE_RESPONSES_API': "1" if getattr(self, 'deepseek_use_responses_api_var', False) else "0",
            # Anthropic extended/adaptive thinking
            'ENABLE_ANTHROPIC_THINKING': "1" if getattr(self, 'enable_anthropic_thinking_var', False) else "0",
            'ANTHROPIC_THINKING_BUDGET': str(self.anthropic_thinking_budget_var) if getattr(self, 'enable_anthropic_thinking_var', False) else '0',
            'ANTHROPIC_FORCE_ADAPTIVE': "1" if getattr(self, 'anthropic_force_adaptive_var', False) else "0",
            'ANTHROPIC_EFFORT': getattr(self, 'anthropic_effort_var', 'medium'),
            # Skip thinking for lightweight tasks
            'SKIP_BOOK_TITLE_THINKING': "1" if getattr(self, 'skip_book_title_thinking_var', True) else "0",
            'SKIP_METADATA_THINKING': "1" if getattr(self, 'skip_metadata_thinking_var', True) else "0",
            'SKIP_TOC_THINKING': "1" if getattr(self, 'skip_toc_thinking_var', False) else "0",
            'MANGA_OCR_DISABLE_THINKING': "1" if _bool_setting(((self.config.get('manga_settings') or {}).get('ocr') or {}).get('manga_ocr_disable_thinking', True), True) else "0",
            'LIGHTWEIGHT_THINKING_LEVEL': str(getattr(self, 'lightweight_thinking_level_var', 1)),
            'OPENROUTER_EXCLUDE': '1',
            'OPENROUTER_PREFERRED_PROVIDER': self.config.get('openrouter_preferred_provider', 'Auto'),
            # Custom API endpoints
            'OPENAI_CUSTOM_BASE_URL': self.openai_base_url_var if self.openai_base_url_var else '',
            'USE_CUSTOM_IMAGE_EDIT_ENDPOINT': '1' if getattr(self, 'use_custom_image_edit_endpoint_var', False) else '0',
            'CUSTOM_IMAGE_EDIT_BASE_URL': getattr(self, 'custom_image_edit_endpoint_var', '') if getattr(self, 'use_custom_image_edit_endpoint_var', False) else '',
            'OPENAI_IMAGE_EDIT_BASE_URL': getattr(self, 'custom_image_edit_endpoint_var', '') if getattr(self, 'use_custom_image_edit_endpoint_var', False) else '',
            'CUSTOM_OPENAI_PREFIX_ROUTES': self._custom_prefix_routes_env_json(),
            'OLLAMA_SETTINGS_JSON': self._ollama_settings_env_json(),
            'OPENAI_TTS_ENDPOINT': getattr(self, 'openai_tts_endpoint_var', '') or (self.openai_base_url_var if str(self.openai_base_url_var).rstrip('/').endswith('/audio/speech') else ''),
            'TTS_VOICE': getattr(self, 'tts_voice_var', '') or '',
            'GROQ_API_URL': self.groq_base_url_var if self.groq_base_url_var else '',
            'FIREWORKS_API_URL': self.fireworks_base_url_var if hasattr(self, 'fireworks_base_url_var') and self.fireworks_base_url_var else '',
            'USE_CUSTOM_OPENAI_ENDPOINT': '1' if self.use_custom_openai_endpoint_var else '0',

            # Image compression settings
            'ENABLE_IMAGE_COMPRESSION': "1" if self.config.get('enable_image_compression', False) else "0",
            'AUTO_COMPRESS_ENABLED': "1" if self.config.get('auto_compress_enabled', True) else "0",
            'TARGET_IMAGE_TOKENS': str(self.config.get('target_image_tokens', 1000)),
            'IMAGE_COMPRESSION_FORMAT': self.config.get('image_compression_format', 'auto'),
            'WEBP_QUALITY': str(self.config.get('webp_quality', 85)),
            'JPEG_QUALITY': str(self.config.get('jpeg_quality', 85)),
            'PNG_COMPRESSION': str(self.config.get('png_compression', 6)),
            'MAX_IMAGE_DIMENSION': str(self.config.get('max_image_dimension', 2048)),
            'MAX_IMAGE_SIZE_MB': str(self.config.get('max_image_size_mb', 10)),
            'PRESERVE_TRANSPARENCY': "1" if self.config.get('preserve_transparency', False) else "0",
            
            # PDF settings
            'ENABLE_PDF_OUTPUT': '1' if self.config.get('enable_pdf_output', False) else '0',
            'PDF_GENERATE_TOC': '1' if self.config.get('pdf_generate_toc', False) else '0',
            'PDF_TOC_PAGE_NUMBERS': '1' if self.config.get('pdf_toc_page_numbers', True) else '0',
            'PDF_PAGE_NUMBERS': '1' if self.config.get('pdf_page_numbers', True) else '0',
            'PDF_PAGE_NUMBER_ALIGNMENT': self.config.get('pdf_page_number_alignment', 'center'),
            'PDF_OUTPUT_FORMAT': self.pdf_output_format_var if hasattr(self, 'pdf_output_format_var') else 'pdf',
            'PDF_RENDER_MODE': self.pdf_render_mode_var if hasattr(self, 'pdf_render_mode_var') else 'fast_semantic',
            'PDF_USE_TOC_SECTIONS': '1' if getattr(self, 'pdf_use_toc_sections_var', True) else '0',
            'PDF_ASYNC_PAGE_THRESHOLD': str(self.pdf_async_page_threshold_var) if hasattr(self, 'pdf_async_page_threshold_var') else '100',
            'PDF_EXTRACTION_WORKERS': str(getattr(self, 'pdf_extraction_workers_var', 'auto') or 'auto'),
            'PDF_PARAGRAPH_ALIGNMENT': str(getattr(self, 'pdf_paragraph_alignment_var', 'source') or 'source'),
            'PDF_HEADER_ALIGNMENT': str(getattr(self, 'pdf_header_alignment_var', 'source') or 'source'),
            'PDF_PARAGRAPH_JUSTIFICATION': str(getattr(self, 'pdf_paragraph_justification_var', 'source') or 'source'),
            'PDF_RTL_PARAGRAPH_LAYOUT': '1' if getattr(self, 'pdf_rtl_paragraph_layout_var', False) else '0',
            'PDF_RENDER_BATCH_SIZE': str(self.config.get('pdf_render_batch_size', 50)),
            'PDF_FAST_RENDERING': '1' if self.config.get('pdf_fast_rendering', True) else '0',
            'PDF_USE_RAPID_WORKSPACE_COMPILER': '1' if self.config.get('pdf_use_rapid_workspace_compiler', True) else '0',
            # Image compression quality sub-settings
            'IMAGE_COMPRESSION_QUALITY': str(self.config.get('image_compression_quality', 80)),
            'EXCLUDE_COVER_COMPRESSION': '1' if self.config.get('exclude_cover_compression', True) else '0',
            'EXCLUDE_GIF_COMPRESSION': '1' if self.config.get('exclude_gif_compression', True) else '0',

            'PRESERVE_ORIGINAL_FORMAT'
            'OPTIMIZE_FOR_OCR': "1" if self.config.get('optimize_for_ocr', True) else "0",
            'PROGRESSIVE_ENCODING': "1" if self.config.get('progressive_encoding', True) else "0",
            'SAVE_COMPRESSED_IMAGES': "1" if self.config.get('save_compressed_images', False) else "0",
            'IMAGE_CHUNK_OVERLAP_PERCENT': str(getattr(self, 'image_chunk_overlap_var', '3')),
            'IMAGE_CHUNK_MIN_OVERLAP_PIXELS': str(getattr(self, 'image_chunk_min_overlap_pixels_var', '80')),
            'IMAGE_SMART_CHUNKING': '1' if getattr(self, 'image_smart_chunking_var', True) else '0',
            'VISION_OCR_FUZZY_CHUNK_DEDUPE': '1' if getattr(self, 'vision_ocr_fuzzy_chunk_dedupe_var', False) else '0',


            # Metadata and batch header translation settings
            'TRANSLATE_METADATA_FIELDS': json.dumps(self.translate_metadata_fields),
            'METADATA_TRANSLATION_MODE': self.config.get('metadata_translation_mode', 'together'),
            'BATCH_TRANSLATE_HEADERS': "1" if self.batch_translate_headers_var else "0",
            'HEADERS_PER_BATCH': str(self.headers_per_batch_var),
            'FAILED_TRANSLATION_RETRY_ATTEMPTS': str(
                self.failed_translation_retry_attempts_var
            ),
            'UPDATE_HTML_HEADERS': "1" if self.update_html_headers_var else "0",
            'SAVE_HEADER_TRANSLATIONS': "1" if self.save_header_translations_var else "0",
            'ALLOW_AI_MARKDOWN_HEADERS': "1" if getattr(self, 'allow_ai_markdown_headers_var', False) else "0",
            'METADATA_SYSTEM_PROMPT': self.config.get('metadata_system_prompt', '').replace('{target_lang}', self.config.get('output_language', 'English')),
            'METADATA_FIELD_PROMPTS': json.dumps(self.config.get('metadata_field_prompts', {})),
            'LANG_PROMPT_BEHAVIOR': self.config.get('lang_prompt_behavior', 'auto'),
            'FORCED_SOURCE_LANG': self.config.get('forced_source_lang', 'Korean'),
            'OUTPUT_LANGUAGE': self.config.get('output_language', 'English'),
            'METADATA_BATCH_PROMPT': self.config.get('metadata_batch_prompt', ''),
            'BATCH_HEADER_SYSTEM_PROMPT': self.config.get('batch_header_system_prompt', ''),
            'BATCH_HEADER_PROMPT': self.config.get('batch_header_prompt', ''),
            'BATCH_HEADER_PREPEND_NUMBER_PATTERN': str(self.config.get('batch_header_prepend_number_pattern', '') or ''),
            
            # AI Hunter configuration
            'AI_HUNTER_CONFIG': json.dumps(self.config.get('ai_hunter_config', {})),

            # Anti-duplicate parameters
            'ENABLE_ANTI_DUPLICATE': '1' if hasattr(self, 'enable_anti_duplicate_var') and self.enable_anti_duplicate_var else '0',
            'TOP_P': str(self.top_p_var) if hasattr(self, 'top_p_var') else '1.0',
            'MIN_P': str(self.min_p_var) if hasattr(self, 'min_p_var') else '0.0',
            'BYPASS_MIN_P_ALLOWLIST': '1' if hasattr(self, 'bypass_min_p_allowlist_var') and self.bypass_min_p_allowlist_var else '0',
            'TOP_K': str(self.top_k_var) if hasattr(self, 'top_k_var') else '0',
            'FREQUENCY_PENALTY': str(self.frequency_penalty_var) if hasattr(self, 'frequency_penalty_var') else '0.0',
            'PRESENCE_PENALTY': str(self.presence_penalty_var) if hasattr(self, 'presence_penalty_var') else '0.0',
            'REPETITION_PENALTY': str(self.repetition_penalty_var) if hasattr(self, 'repetition_penalty_var') else '1.0',
            'CANDIDATE_COUNT': str(self.candidate_count_var) if hasattr(self, 'candidate_count_var') else '1',
            'CUSTOM_STOP_SEQUENCES': self.custom_stop_sequences_var if hasattr(self, 'custom_stop_sequences_var') else '',
            'LOGIT_BIAS_ENABLED': '1' if hasattr(self, 'logit_bias_enabled_var') and self.logit_bias_enabled_var else '0',
            'LOGIT_BIAS_STRENGTH': str(self.logit_bias_strength_var) if hasattr(self, 'logit_bias_strength_var') else '-0.5',
            'BIAS_COMMON_WORDS': '1' if hasattr(self, 'bias_common_words_var') and self.bias_common_words_var else '0',
            'BIAS_REPETITIVE_PHRASES': '1' if hasattr(self, 'bias_repetitive_phrases_var') and self.bias_repetitive_phrases_var else '0',
            'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),
            'GOOGLE_CLOUD_PROJECT': google_cloud_project,  # Now properly set from credentials
            'VERTEX_AI_LOCATION': self.vertex_location_var if hasattr(self, 'vertex_location_var') and isinstance(self.vertex_location_var, str) else (self.vertex_location_var.text() if hasattr(self, 'vertex_location_var') and hasattr(self.vertex_location_var, 'text') else 'us-east5'),
            'IS_AZURE_ENDPOINT': '1' if (self.use_custom_openai_endpoint_var and 
                                  ('.azure.com' in self.openai_base_url_var or 
                                   '.cognitiveservices' in self.openai_base_url_var)) else '0',
            'AZURE_API_VERSION': str(self.config.get('azure_api_version', '2024-08-01-preview')),
            
            # Multi API Key support
            # NOTE: Do NOT put the full multi-key JSON into MULTI_API_KEYS env var (Windows 32767-char limit).
            'USE_MULTI_API_KEYS': "1" if self.config.get('use_multi_api_keys', False) else "0",
            'FORCE_KEY_ROTATION': '1' if self.config.get('force_key_rotation', True) else '0',
            'ROTATION_FREQUENCY': str(self.config.get('rotation_frequency', 1)),
            'USE_INPAINTER_KEYS': "1" if self.config.get('use_inpainter_keys', False) else "0",
            'INPAINTER_API_KEYS': json.dumps(self.config.get('inpainter_keys', []) or []),
            'USE_ROLLING_SUMMARY_KEYS': "1" if self.config.get('use_rolling_summary_keys', False) else "0",
            'ROLLING_SUMMARY_API_KEYS': json.dumps(self.config.get('rolling_summary_keys', []) or []),
            'USE_TRUNCATION_RETRY_KEYS': "1" if self.config.get('use_truncation_retry_keys', False) else "0",
            'TRUNCATION_RETRY_API_KEYS': json.dumps(self.config.get('truncation_retry_keys', []) or []),
            'USE_METADATA_KEYS': "1" if self.config.get('use_metadata_keys', False) else "0",
            'METADATA_API_KEYS': json.dumps(self.config.get('metadata_keys', []) or []),
           
            # Glossary-specific overrides
            'GLOSSARY_COMPRESSION_FACTOR': str(self.config.get('glossary_compression_factor', self.compression_factor_var)),
            'GLOSSARY_REFINEMENT_COMPRESSION_FACTOR': str(self.compression_factor_var),
            'GLOSSARY_MAX_OUTPUT_TOKENS': str(current_max_tokens) if str(self.config.get('glossary_max_output_tokens', '-1')) == '-1' else str(self.config.get('glossary_max_output_tokens')),
            'GLOSSARY_TEMPERATURE': str(self.config.get('manual_glossary_temperature', self.trans_temp.text())),
       }

        # When VISION_OCR_BATCH_SIZE is explicitly set (> 0) in vision mode,
        # override BATCH_SIZE to 1 so chapters process sequentially.
        # This prevents the multiplication bug where chapter-level batch workers
        # and per-chapter vision OCR workers both create parallel API requests
        # (e.g. BATCH_SIZE=3 × VISION_OCR_BATCH_SIZE=3 = 9 requests instead of 3).
        # The vision OCR batch size becomes the sole parallelism controller.
        try:
            vision_batch_raw = int(str(env_vars.get('VISION_OCR_BATCH_SIZE', '-1')).strip() or '-1')
        except (ValueError, TypeError):
            vision_batch_raw = -1
        if vision_batch_raw > 0 and env_vars.get('OUTPUT_MODE', '').strip().lower() == 'vision':
            env_vars['BATCH_SIZE'] = '1'

        print(f"[DEBUG] DISABLE_CHAPTER_MERGING = '{os.getenv('DISABLE_CHAPTER_MERGING', '0')}'")
        return env_vars

    def _output_side_glossary_backup_dir_for_source(self, source_path):
        """Return the distributable Glossary_Backup folder beside this source's output."""
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')
        if override_dir:
            output_dir = os.path.join(os.path.abspath(override_dir), base_name)
        else:
            output_dir = os.path.join(_get_app_dir(), base_name)
        return os.path.join(output_dir, "Glossary_Backup")

    def _current_glossary_cjk_script_filter_enabled(self):
        """Resolve the CJK glossary filter from live UI/config/env state."""
        try:
            if hasattr(self, 'cjk_script_filter_checkbox'):
                value = bool(self.cjk_script_filter_checkbox.isChecked())
                self.config['glossary_cjk_script_filter'] = value
                self.glossary_cjk_script_filter_var = value
                return value
        except Exception:
            pass

        try:
            value = bool(self.config.get(
                'glossary_cjk_script_filter',
                getattr(self, 'glossary_cjk_script_filter_var', False),
            ))
            self.glossary_cjk_script_filter_var = value
            return value
        except Exception:
            pass

        return os.environ.get('GLOSSARY_CJK_SCRIPT_FILTER', '0') == '1'

    def _parallel_epub_system_prompt_for_file(self, file_path):
        state = getattr(self, "_parallel_epub_pair_source", None)
        if not isinstance(state, dict):
            return ""
        generated_path = str(state.get("generated_path") or "")
        try:
            matches = os.path.normcase(os.path.abspath(file_path)) == os.path.normcase(
                os.path.abspath(generated_path)
            )
        except Exception:
            matches = str(file_path) == generated_path
        return str(state.get("system_prompt") or "") if matches else ""

    def _glossary_env_mappings(self):
        """(env key, value) pairs for the glossary settings in ``self.config``.

        save_config exports them, and so does the EPUB converter: its chapter
        header and TOC requests read the glossary settings from os.environ,
        where an earlier run may have left them changed (an Input/Output
        run forces APPEND_GLOSSARY=0).
        """
        # Normalize and align glossary prompts
        prompt_keys = ['manual_glossary_prompt', 'append_glossary_prompt', 'single_pass_glossary_header_prompt', 'unified_auto_glosary_prompt3', 'glossary_refinement_system_prompt', 'glossary_refinement_user_prompt', 'glossary_translation_prompt', 'glossary_format_instructions']
        for key in prompt_keys:
            self.config[key] = self.config.get(key, '') or ''

        return [
            ('GLOSSARY_SYSTEM_PROMPT', self.config.get('manual_glossary_prompt', '')),
            ('AUTO_GLOSSARY_PROMPT', self.config.get('unified_auto_glosary_prompt3', '')),
            ('GLOSSARY_REFINEMENT_ENABLED', '1' if self.config.get('glossary_refinement_enabled', False) else '0'),
            ('GLOSSARY_REFINEMENT_SYSTEM_PROMPT', self.config.get('glossary_refinement_system_prompt') or getattr(self, 'glossary_refinement_system_prompt', '')),
            ('GLOSSARY_REFINEMENT_USER_PROMPT', self.config.get('glossary_refinement_user_prompt', '')),
            ('GLOSSARY_REFINEMENT_TYPE_MODE', self.config.get('glossary_refinement_type_mode', 'all')),
            ('GLOSSARY_REFINEMENT_SELECTED_TYPES', ','.join(self.config.get('glossary_refinement_selected_types', []))),
            ('GLOSSARY_REFINEMENT_CHUNKING_MODE', self.config.get('glossary_refinement_chunking_mode', 'all')),
            ('GLOSSARY_REFINEMENT_SKIP_DEDUPE', '1' if self.config.get('glossary_refinement_skip_dedupe', False) else '0'),
            ('GLOSSARY_REFINEMENT_WAIT_FOR_COMPLETION', '1' if self.config.get('glossary_refinement_wait_for_completion', False) else '0'),
            ('GLOSSARY_REFINEMENT_REOPEN_ON_SOURCE_CHANGE', '1' if self.config.get('glossary_refinement_reopen_on_source_change', False) else '0'),
            ('APPEND_GLOSSARY_PROMPT', self.config.get('append_glossary_prompt', '') or '- Follow this reference glossary for consistent translation (Do not output any raw entries):\n'),
            ('APPEND_GLOSSARY', '0' if self.config.get('auto_glossary_mode', 'off') == 'no_glossary' else ('1' if self.config.get('append_glossary') else '0')),
            ('ADD_ADDITIONAL_GLOSSARY', '1' if self.config.get('add_additional_glossary') else '0'),
            ('ADDITIONAL_GLOSSARY_PATH', self.config.get('additional_glossary_path', '')),
            ('GLOSSARY_SHARED_DIR', os.path.join(_get_app_dir(), 'Glossary')),
            ('SAVE_GLOSSARY_IN_OUTPUT', '1' if self.config.get('save_glossary_in_output', False) else '0'),
            ('VISION_OCR_SOURCE_PREPASS', str(self.config.get('vision_ocr_source_prepass', 'auto') or 'auto')),
            ('ENABLE_AUTO_GLOSSARY', '1' if self.config.get('auto_glossary_mode', 'off') == 'minimal' else '0'),
            ('AUTO_GLOSSARY_MODE', self.config.get('auto_glossary_mode', 'off')),
            ('SINGLE_PASS_GLOSSARY_MODE', '1' if self.config.get('auto_glossary_mode', 'off') == 'single_pass' else ''),
            ('SINGLE_PASS_GLOSSARY_HEADER_PROMPT', self.config.get('single_pass_glossary_header_prompt', '')),
            ('GLOSSARY_TRANSLATION_PROMPT', self.config.get('glossary_translation_prompt', '')),
            ('GLOSSARY_FORMAT_INSTRUCTIONS', self.config.get('glossary_format_instructions', '')),
            ('GLOSSARY_DISABLE_HONORIFICS_FILTER', '1' if self.config.get('glossary_disable_honorifics_filter') else '0'),
            ('GLOSSARY_STRIP_HONORIFICS', '1' if self.config.get('strip_honorifics') else '0'),
            ('GLOSSARY_FUZZY_THRESHOLD', str(self.config.get('glossary_fuzzy_threshold', 0.90))),
            ('GLOSSARY_ENTRY_TYPE_FILTER_MODE', self.config.get('glossary_entry_type_filter_mode', 'Loose')),
            ('GLOSSARY_USE_LEGACY_CSV', '1' if self.config.get('glossary_use_legacy_csv') else '0'),
            ('GLOSSARY_OUTPUT_LEGACY_JSON', '1' if self.config.get('glossary_output_legacy_json') else '0'),
            ('GLOSSARY_INCLUDE_ALL_CHARACTERS', '1' if self.config.get('glossary_include_all_characters') else '0'),
            ('GLOSSARY_SKIP_IDENTICAL_ENTRIES', '1' if self.config.get('glossary_skip_identical_entries', True) else '0'),
            ('GLOSSARY_CJK_SCRIPT_FILTER', '1' if self.config.get('glossary_cjk_script_filter', False) else '0'),
            ('GLOSSARY_SKIP_GENDER_TRACKING', '1' if self.config.get('glossary_skip_gender_tracking', False) else '0'),
            ('GLOSSARY_GENDER_NOISE_THRESHOLD', str(self.config.get('glossary_gender_noise_threshold', 10))),
            ('GLOSSARY_GENDER_TRACKING_BIAS', str(self.config.get('glossary_gender_tracking_bias', 'none'))),
            ('GLOSSARY_USE_SMART_FILTER', '1' if self.config.get('glossary_use_smart_filter', True) else '0'),
            ('GLOSSARY_MAX_SENTENCES', str(self.config.get('glossary_max_sentences', 200))),
            ('COMPRESS_GLOSSARY_PROMPT', '1' if self.config.get('compress_glossary_prompt') else '0'),
            ('COMPRESS_GLOSSARY_CONSIDER_TRANSLATED_COLUMN', '1' if getattr(self, 'compress_glossary_consider_translated_column_var', self.config.get('compress_glossary_consider_translated_column', False)) else '0'),
            ('COMPRESS_GLOSSARY_MULTIPASS_EXCLUDE_MATCHING', '1' if getattr(self, 'compress_glossary_multipass_exclude_matching_var', self.config.get('compress_glossary_multipass_exclude_matching', True)) else '0'),
            ('GLOSSARY_INCLUDE_GENDER_CONTEXT', '1' if self.config.get('include_gender_context') else '0'),
            ('GLOSSARY_ENABLE_GENDER_NUANCE', '1' if self.config.get('enable_gender_nuance', True) else '0'),
            ('GLOSSARY_INCLUDE_DESCRIPTION', '1' if self.config.get('include_description') else '0'),
            # Add missing environment variables that GlossaryManager.py reads
            ('GLOSSARY_MIN_FREQUENCY', str(self.config.get('glossary_min_frequency', 2))),
            ('GLOSSARY_MAX_NAMES', str(self.config.get('glossary_max_names', 50))),
            ('GLOSSARY_MAX_TITLES', str(self.config.get('glossary_max_titles', 30))),
            ('CONTEXT_WINDOW_SIZE', str(self.config.get('context_window_size', 5))),
            ('GLOSSARY_MAX_TEXT_SIZE', str(self.config.get('glossary_max_text_size', 50000))),
            ('GLOSSARY_CHAPTER_SPLIT_THRESHOLD', str(self.config.get('glossary_chapter_split_threshold', 8192))),
            ('GLOSSARY_FILTER_MODE', self.config.get('glossary_filter_mode', 'strict')),
            ('GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL', '1' if self.config.get('never_consider_in_between_files_as_special', True) else '0'),
            ('GLOSSARY_DUPLICATE_ALGORITHM', self.config.get('glossary_duplicate_algorithm', 'auto')),
            ('GLOSSARY_PARTIAL_RATIO_WEIGHT', str(self.config.get('glossary_partial_ratio_weight', 0.45))),
            ('GLOSSARY_PARTIAL_RATIO_GENDER_ONLY', '1' if self.config.get('glossary_partial_ratio_gender_only', False) else '0'),
            ('GLOSSARY_ALIAS_AWARE_NAME_MATCHING', '1' if self.config.get('glossary_alias_aware_name_matching', False) else '0'),
            ('GLOSSARY_ALIAS_AWARE_GENDER_ONLY', '1' if self.config.get('glossary_alias_aware_gender_only', True) else '0'),
            ('GLOSSARY_TARGET_LANGUAGE', self.config.get('glossary_target_language', 'English')),
            # Glossary anti-duplicate parameters
            ('GLOSSARY_ENABLE_ANTI_DUPLICATE', '1' if self.config.get('glossary_enable_anti_duplicate', False) else '0'),
            ('GLOSSARY_TOP_P', str(self.config.get('glossary_top_p', 1.0))),
            ('GLOSSARY_MIN_P', str(self.config.get('glossary_min_p', 0.0))),
            ('GLOSSARY_BYPASS_MIN_P_ALLOWLIST', '1' if self.config.get('glossary_bypass_min_p_allowlist', False) else '0'),
            ('GLOSSARY_TOP_K', str(self.config.get('glossary_top_k', 0))),
            ('GLOSSARY_FREQUENCY_PENALTY', str(self.config.get('glossary_frequency_penalty', 0.0))),
            ('GLOSSARY_PRESENCE_PENALTY', str(self.config.get('glossary_presence_penalty', 0.0))),
            ('GLOSSARY_REPETITION_PENALTY', str(self.config.get('glossary_repetition_penalty', 1.0))),
            ('GLOSSARY_CANDIDATE_COUNT', str(self.config.get('glossary_candidate_count', 1))),
            ('GLOSSARY_CUSTOM_STOP_SEQUENCES', str(self.config.get('glossary_custom_stop_sequences', ''))),
            ('GLOSSARY_LOGIT_BIAS_ENABLED', '1' if self.config.get('glossary_logit_bias_enabled', False) else '0'),
            ('GLOSSARY_LOGIT_BIAS_STRENGTH', str(self.config.get('glossary_logit_bias_strength', -0.5))),
            ('GLOSSARY_BIAS_COMMON_WORDS', '1' if self.config.get('glossary_bias_common_words', False) else '0'),
            ('GLOSSARY_BIAS_REPETITIVE_PHRASES', '1' if self.config.get('glossary_bias_repetitive_phrases', False) else '0'),
            ('GLOSSARY_CUSTOM_ENTRY_TYPES', json.dumps(self.config.get('custom_entry_types', {}))),
            ('GLOSSARY_CUSTOM_FIELDS', json.dumps(self.config.get('custom_glossary_fields', []))),
        ]

    def debug_environment_variables(self, show_all=False):
        """Debug and verify all critical environment variables are set correctly.

        Args:
            show_all (bool): If True, shows all environment variables. If False, only shows critical ones.
        """
        # Check if debug mode is enabled
        debug_mode = self.config.get('show_debug_buttons', False)
        
        if debug_mode:
            self.append_log("🔍 [ENV_DEBUG] Starting comprehensive environment variable check...")
        
        # Critical environment variables that should always be set
        critical_env_vars = {
            # Glossary-related
            'GLOSSARY_SYSTEM_PROMPT': 'Manual glossary extraction prompt',
            'AUTO_GLOSSARY_PROMPT': 'Auto glossary generation prompt',
            'APPEND_GLOSSARY_PROMPT': 'Append glossary prompt',
            'GLOSSARY_CUSTOM_ENTRY_TYPES': 'Custom entry types configuration (JSON)',
            'GLOSSARY_CUSTOM_FIELDS': 'Custom glossary fields (JSON)',
            'GLOSSARY_TRANSLATION_PROMPT': 'Glossary translation prompt',
            'GLOSSARY_FORMAT_INSTRUCTIONS': 'Glossary formatting instructions',
            'GLOSSARY_DISABLE_HONORIFICS_FILTER': 'Honorifics filter disable flag',
            'GLOSSARY_STRIP_HONORIFICS': 'Strip honorifics flag',
            'GLOSSARY_FUZZY_THRESHOLD': 'Fuzzy matching threshold',
            'GLOSSARY_USE_LEGACY_CSV': 'Legacy CSV format flag',
            'GLOSSARY_MAX_SENTENCES': 'Maximum sentences for glossary processing',
            
            # OpenRouter/NVIDIA settings
            'OPENROUTER_USE_HTTP_ONLY': 'OpenRouter/NVIDIA HTTP-only transport',
            'USE_NVIDIA_HTTP': 'NVIDIA HTTP-only transport',
            'OPENROUTER_ACCEPT_IDENTITY': 'OpenRouter identity encoding',
            'OPENROUTER_PREFERRED_PROVIDER': 'OpenRouter preferred provider',
            
            # General application settings
            'EXTRACTION_WORKERS': 'Number of extraction worker threads',
            'PDF_EXTRACTION_WORKERS': 'PDF input extraction workers (or auto)',
            'PDF_PARAGRAPH_ALIGNMENT': 'PDF paragraph alignment override',
            'PDF_HEADER_ALIGNMENT': 'PDF header alignment override',
            'PDF_PARAGRAPH_JUSTIFICATION': 'PDF paragraph justification override',
            'PDF_RTL_PARAGRAPH_LAYOUT': 'PDF right-to-left paragraph layout',
            'ENABLE_GUI_YIELD': 'GUI yield during processing',
            'RETAIN_SOURCE_EXTENSION': 'Retain source file extension',
            'DOWNLOAD_REMOTE_IMAGE_URLS': 'Download remote EPUB image URLs',
            'GLOSSARY_PARALLEL_ENABLED': 'Glossary parallel processing enabled',
            
            # Debug/Logging
            'DEBUG_SAVE_REQUEST_PAYLOADS': 'Save API request payloads',
            'DEBUG_SAVE_REQUEST_PAYLOADS_VERBOSE': 'Verbose payload logging',
            'SHOW_DEBUG_BUTTONS': 'Show debug buttons in UI',
            
            # QA Scanner settings
            'QA_FOREIGN_CHAR_THRESHOLD': 'Foreign character detection threshold',
            'QA_TARGET_LANGUAGE': 'Target language for QA checks',
            'QA_CHECK_ENCODING': 'Check for encoding issues',
            'QA_CHECK_REPETITION': 'Check for repetitive text',
            'QA_CHECK_ARTIFACTS': 'Check for translation artifacts',
            'QA_CHECK_GLOSSARY_LEAKAGE': 'Check for glossary leakage',
            'QA_MIN_FILE_LENGTH': 'Minimum file length for QA',
            'QA_REPORT_FORMAT': 'QA report format',
            'QA_AUTO_SAVE_REPORT': 'Auto-save QA reports',
            'QA_CACHE_ENABLED': 'QA cache enabled',
            'QA_SDLXLIFF_TAG_RETENTION_THRESHOLD': 'SDLXLIFF source tag retention threshold',
            'QA_SDLXLIFF_TAG_SURPLUS_TOLERANCE': 'SDLXLIFF surplus output tag tolerance',
            'QA_SDLXLIFF_MIN_SOURCE_PARAGRAPH_TAGS': 'SDLXLIFF minimum source paragraph tags',
            'AI_HUNTER_MAX_WORKERS': 'AI Hunter maximum workers',
        }
        
        # Optional/Informational environment variables
        optional_env_vars = {
            # Post-translation scanning phase
            'SCAN_PHASE_ENABLED': 'Enable post-translation scanning phase',
            'SCAN_PHASE_MODE': 'Scanning mode (quick-scan/aggressive/ai-hunter/custom)',
            
            # AI Model settings
            'ENABLE_GEMINI_THINKING': 'Enable Gemini thinking mode',
            'THINKING_BUDGET': 'Gemini thinking budget',
            'ENABLE_GPT_THINKING': 'Enable GPT-4o reasoning',
            'GPT_REASONING_TOKENS': 'GPT reasoning effort tokens',
            'GPT_EFFORT': 'GPT reasoning effort level',
            'OPENROUTER_USE_REASONING_TOKENS': 'Use OpenRouter reasoning token budget instead of effort',
            'PASS_THINKING_TO_OPENAI_COMPATIBLE': 'Force reasoning effort on unknown OpenAI-compatible routes',
            'FORCE_SERVICE_TIER_UNKNOWN_ROUTES': 'Force service tier on unknown OpenAI-compatible routes',
            'ENABLE_DEEPSEEK_THINKING': 'Enable DeepSeek thinking mode',
            'DEEPSEEK_USE_RESPONSES_API': 'Use the DeepSeek Responses API format',
            
            # API Endpoints
            'OPENAI_CUSTOM_BASE_URL': 'Custom OpenAI API base URL',
            'USE_CUSTOM_IMAGE_EDIT_ENDPOINT': 'Use custom image/video output and manga image edit endpoint',
            'CUSTOM_IMAGE_EDIT_BASE_URL': 'Custom image/video output and manga image edit base URL',
            'OPENAI_IMAGE_EDIT_BASE_URL': 'Alias for custom image edit base URL',
            'CUSTOM_OPENAI_PREFIX_ROUTES': 'Custom OpenAI-compatible prefix routes',
            'OLLAMA_SETTINGS_JSON': 'Local Ollama settings',
            'GROQ_API_URL': 'Groq API endpoint',
            'FIREWORKS_API_URL': 'Fireworks API endpoint',
            'USE_CUSTOM_OPENAI_ENDPOINT': 'Use custom OpenAI endpoint',
            'USE_GEMINI_OPENAI_ENDPOINT': 'Use Gemini OpenAI-compatible endpoint',
            'GEMINI_OPENAI_ENDPOINT': 'Gemini OpenAI endpoint URL',
            
            # Image Compression
            'ENABLE_IMAGE_COMPRESSION': 'Enable image compression',
            'AUTO_COMPRESS_ENABLED': 'Auto compress images',
            'TARGET_IMAGE_TOKENS': 'Target image token count',
            'IMAGE_COMPRESSION_FORMAT': 'Image compression format',
            'WEBP_QUALITY': 'WebP quality',
            'JPEG_QUALITY': 'JPEG quality',
            'PNG_COMPRESSION': 'PNG compression level',
            'MAX_IMAGE_DIMENSION': 'Max image dimension',
            'MAX_IMAGE_SIZE_MB': 'Max image size MB',
            'PRESERVE_TRANSPARENCY': 'Preserve image transparency',
            'OPTIMIZE_FOR_OCR': 'Optimize images for OCR',
            'PROGRESSIVE_ENCODING': 'Progressive image encoding',
            'SAVE_COMPRESSED_IMAGES': 'Save compressed images',
            'IMAGE_CHUNK_OVERLAP_PERCENT': 'Image chunk overlap percentage',
            'IMAGE_CHUNK_MIN_OVERLAP_PIXELS': 'Image chunk minimum overlap pixels',
            'IMAGE_SMART_CHUNKING': 'Smart line-boundary image chunking',
            'IMAGE_SMART_CHUNK_MAX_FOREGROUND_RATIO': 'Smart chunk max foreground row ratio',
            'IMAGE_SMART_CHUNK_MIN_GAP_ROWS': 'Smart chunk minimum whitespace gap rows',
            'IMAGE_SMART_CHUNK_CUT_PADDING_ROWS': 'Smart chunk cut padding rows',
            'VISION_OCR_FUZZY_CHUNK_DEDUPE': 'Fuzzy OCR chunk dedupe',
            'VISION_OCR_FUZZY_CHUNK_DEDUPE_THRESHOLD': 'Fuzzy OCR chunk dedupe threshold',
            'VISION_OCR_FUZZY_CHUNK_DEDUPE_MIN_LENGTH': 'Fuzzy OCR chunk dedupe minimum length',
            'VISION_OCR_BATCH_TRANSLATION': 'Batch Vision OCR chunk requests',
            'VISION_OCR_BATCH_SIZE': 'Vision API request slot override',
            'VISION_OCR_SOURCE_PREPASS': 'Vision OCR source prepass mode (auto/on/off)',
            
            # Metadata and Headers
            'TRANSLATE_METADATA_FIELDS': 'Metadata fields to translate (JSON)',
            'METADATA_TRANSLATION_MODE': 'Metadata translation mode',
            'BATCH_TRANSLATE_HEADERS': 'Batch translate headers',
            'HEADERS_PER_BATCH': 'Headers per batch',
            'FAILED_TRANSLATION_RETRY_ATTEMPTS': 'Failed TOC/header entry retry attempts',
            'PARTIAL_B2_ENTRIES_PER_REQUEST': 'Partial.b2 entries per JSON request',
            'UPDATE_HTML_HEADERS': 'Update HTML headers',
            'SAVE_HEADER_TRANSLATIONS': 'Save header translations',
            'IGNORE_HEADER': 'Ignore header metadata',
            'USE_TITLE': 'Use title metadata',
            'SKIP_TITLE_TAG_TRANSLATION': 'Skip HTML/XHTML title tag translation',
            
            # Extraction
            'TEXT_EXTRACTION_METHOD': 'Text extraction method',
            'FILE_FILTERING_LEVEL': 'File filtering level',
            'EXTRACTION_MODE': 'Extraction mode',
            'ENHANCED_FILTERING': 'Enhanced filtering level',
            
            # Anti-Duplicate
            'ENABLE_ANTI_DUPLICATE': 'Enable anti-duplicate measures',
            'TOP_P': 'Top-P sampling parameter',
            'MIN_P': 'Min-P sampling parameter',
            'BYPASS_MIN_P_ALLOWLIST': 'Bypass Min-P provider allowlist',
            'TOP_K': 'Top-K sampling parameter',
            'FREQUENCY_PENALTY': 'Frequency penalty',
            'PRESENCE_PENALTY': 'Presence penalty',
            'REPETITION_PENALTY': 'Repetition penalty',
            'CANDIDATE_COUNT': 'Candidate count',
            'CUSTOM_STOP_SEQUENCES': 'Custom stop sequences',
            'LOGIT_BIAS_ENABLED': 'Logit bias enabled',
            'LOGIT_BIAS_STRENGTH': 'Logit bias strength',
            'BIAS_COMMON_WORDS': 'Bias against common words',
            'BIAS_REPETITIVE_PHRASES': 'Bias against repetitive phrases',
            
            # Azure
            'AZURE_API_VERSION': 'Azure API version',
            
            # Fallback Keys
            'USE_FALLBACK_KEYS': 'Use fallback API keys',
            'FALLBACK_KEYS': 'Fallback API keys (JSON)',
            
            # Manga Integration and Manga Settings Dialog variables
            'MANGA_FULL_PAGE_CONTEXT': 'Enable full page context translation',
            'MANGA_VISUAL_CONTEXT_ENABLED': 'Include page image in requests',
            'MANGA_CREATE_SUBFOLDER': "Create 'translated' subfolder for output",
            'MANGA_BG_OPACITY': 'Background opacity (0-255)',
            'MANGA_BG_STYLE': 'Background style (box/circle/wrap)',
            'MANGA_BG_REDUCTION': 'Background reduction factor',
            'MANGA_FONT_SIZE': 'Fixed font size (0=auto)',
            'MANGA_FONT_STYLE': 'Font style name',
            'MANGA_FONT_PATH': 'Selected font path',
            'MANGA_FONT_SIZE_MODE': 'Font size mode (fixed/multiplier)',
            'MANGA_FONT_SIZE_MULTIPLIER': 'Font size multiplier (for multiplier mode)',
            'MANGA_MAX_FONT_SIZE': 'Maximum font size',
            'MANGA_AUTO_MIN_SIZE': 'Automatic minimum readable font size',
            'MANGA_FREE_TEXT_ONLY_BG_OPACITY': 'Apply BG opacity only to free text',
            'MANGA_FORCE_CAPS_LOCK': 'Force caps lock',
            'MANGA_STRICT_TEXT_WRAPPING': 'Strict text wrapping (force fit)',
            'MANGA_CONSTRAIN_TO_BUBBLE': 'Constrain text to bubble bounds',
            'MANGA_TEXT_COLOR': 'Text color RGB (R,G,B)',
            'MANGA_SHADOW_ENABLED': 'Shadow enabled',
            'MANGA_SHADOW_COLOR': 'Shadow color RGB (R,G,B)',
            'MANGA_SHADOW_OFFSET_X': 'Shadow offset X',
            'MANGA_SHADOW_OFFSET_Y': 'Shadow offset Y',
            'MANGA_SHADOW_BLUR': 'Shadow blur radius',
            'MANGA_SKIP_INPAINTING': 'Skip inpainting',
            'MANGA_INPAINT_QUALITY': 'Inpainting quality preset',
            'MANGA_INPAINT_DILATION': 'Inpainting dilation (px)',
            'MANGA_INPAINT_PASSES': 'Inpainting passes',
            'MANGA_INPAINT_METHOD': 'Inpainting method (local/cloud/hybrid/skip)',
            'MANGA_LOCAL_INPAINT_METHOD': 'Local inpainting model type',
            'MANGA_FONT_ALGORITHM': 'Font sizing algorithm preset',
            'MANGA_PREFER_LARGER': 'Prefer larger font sizing',
            'MANGA_BUBBLE_SIZE_FACTOR': 'Use bubble size factor for sizing',
            'MANGA_LINE_SPACING': 'Line spacing multiplier',
            'MANGA_MAX_LINES': 'Maximum lines per bubble',
            'MANGA_QWEN2VL_MODEL_SIZE': 'Qwen2-VL model size selection',
            'MANGA_RAPIDOCR_USE_RECOGNITION': 'RapidOCR: use recognition step',
            'MANGA_RAPIDOCR_LANGUAGE': 'RapidOCR detection language',
            'MANGA_RAPIDOCR_DETECTION_MODE': 'RapidOCR detection mode',
            'MANGA_FULL_PAGE_CONTEXT_PROMPT_LEN': 'Length of full page context prompt',
            'MANGA_OCR_PROMPT_LEN': 'Length of OCR system prompt',
            # Manga Advanced Settings (Memory Management)
            'MANGA_AUTO_CLEANUP_MODELS': 'Auto cleanup models after translation',
            'MANGA_UNLOAD_MODELS_AFTER_TRANSLATION': 'Unload models after translation (reset instance)',
            'MANGA_USE_SINGLETON_MODELS': 'Use singleton model instances',
            'MANGA_PARALLEL_PROCESSING': 'Enable parallel processing',
            'MANGA_MAX_WORKERS': 'Maximum worker threads',
            'MANGA_PARALLEL_PANEL_TRANSLATION': 'Enable parallel panel translation',
            'MANGA_PANEL_MAX_WORKERS': 'Maximum concurrent panels',
            'MANGA_DEBUG_MODE': 'Manga debug mode',
            'MANGA_SAVE_INTERMEDIATE': 'Save intermediate debug images',
            'MANGA_CONCISE_LOGS': 'Concise pipeline logs (suppress verbose steps)',
            'MANGA_SKIP_INPAINTING': 'Skip inpainting step (show detected bubbles only)',
        }
        
        # Check critical variables
        missing_critical = []
        empty_critical = []
        set_critical = []
        
        for var_name, description in critical_env_vars.items():
            value = os.environ.get(var_name)
            
            if value is None:
                missing_critical.append(var_name)
                if debug_mode:
                    self.append_log(f"❌ [ENV_DEBUG] CRITICAL MISSING: {var_name} - {description}")
            elif not value.strip():
                empty_critical.append(var_name)
                if debug_mode:
                    self.append_log(f"⚠️ [ENV_DEBUG] CRITICAL EMPTY: {var_name} - {description}")
            else:
                set_critical.append(var_name)
                if debug_mode:
                    value_preview = str(value)[:100] + ('...' if len(str(value)) > 100 else '')
                    self.append_log(f"✅ [ENV_DEBUG] {var_name}: {value_preview}")
        
        # Treat previous 'optional' as critical as well
        for var_name, description in optional_env_vars.items():
            value = os.environ.get(var_name)
            if value is None:
                missing_critical.append(var_name)
                if debug_mode:
                    self.append_log(f"❌ [ENV_DEBUG] CRITICAL MISSING: {var_name} - {description}")
            elif not str(value).strip():
                empty_critical.append(var_name)
                if debug_mode:
                    self.append_log(f"⚠️ [ENV_DEBUG] CRITICAL EMPTY: {var_name} - {description}")
            else:
                set_critical.append(var_name)
                if debug_mode:
                    value_preview = str(value)[:100] + ('...' if len(str(value)) > 100 else '')
                    self.append_log(f"✅ [ENV_DEBUG] {var_name}: {value_preview}")
        
        # Summary (now includes all former optional variables)
        total_critical = len(critical_env_vars) + len(optional_env_vars)
        if debug_mode:
            self.append_log(f"🔍 [ENV_DEBUG] Summary: {len(set_critical)}/{total_critical} critical vars set")
        
        if missing_critical and debug_mode:
            self.append_log(f"❌ [ENV_DEBUG] {len(missing_critical)} MISSING: {', '.join(missing_critical)}")
            
        if empty_critical and debug_mode:
            self.append_log(f"⚠️ [ENV_DEBUG] {len(empty_critical)} EMPTY: {', '.join(empty_critical)}")
            
        # Check for initialization issues
        if missing_critical or empty_critical:
            if debug_mode:
                self.append_log("❌ [ENV_DEBUG] RECOMMENDATION: Some variables are not initialized!")
                self.append_log("🔧 [ENV_DEBUG] Try calling self.initialize_environment_variables() on startup")
            return False
        else:
            if debug_mode:
                self.append_log("✅ [ENV_DEBUG] All critical environment variables are properly set")
            return True

    def initialize_environment_variables(self):
        """Initialize all environment variables from config on startup.
        Call this method during application initialization to ensure all environment variables are set.
        """
        # Check if debug mode is enabled
        debug_mode = self.config.get('show_debug_buttons', False)
        
        if debug_mode:
            self.append_log("🚀 [INIT] Initializing all environment variables from config...")
        
        # Wire verbose payload saving to GUI debug mode
        try:
            os.environ['DEBUG_SAVE_REQUEST_PAYLOADS_VERBOSE'] = '1' if debug_mode else '0'
            # Also reflect debug mode for the client
            os.environ['SHOW_DEBUG_BUTTONS'] = '1' if debug_mode else '0'
            # Set DEBUG_MODE for general debug logging (used by epub_converter, etc.)
            os.environ['DEBUG_MODE'] = '1' if debug_mode else '0'
            # Ensure capture itself is enabled
            os.environ['DEBUG_SAVE_REQUEST_PAYLOADS'] = '1'
            if debug_mode:
                self.append_log("🔍 [INIT] Verbose payload logging enabled (DEBUG_SAVE_REQUEST_PAYLOADS_VERBOSE=1)")
                self.append_log("🔍 [INIT] Definitive payload capture enabled (DEBUG_SAVE_REQUEST_PAYLOADS=1)")
                self.append_log("🔍 [INIT] Debug mode enabled (DEBUG_MODE=1)")
        except Exception:
            pass
        
        try:
            def _positive_int_config(key, default):
                try:
                    return str(max(1, int(self.config.get(key, default) or default)))
                except (TypeError, ValueError):
                    return str(default)

            def _bool_config(key, default=False):
                value = self.config.get(key, default)
                if isinstance(value, str):
                    return value.strip().lower() in ('1', 'true', 'yes', 'on')
                return bool(value)

            def _int_config(key, default, minimum=0, maximum=None):
                try:
                    numeric_value = int(float(self.config.get(key, default)))
                except (TypeError, ValueError):
                    numeric_value = int(default)
                numeric_value = max(int(minimum), numeric_value)
                if maximum is not None:
                    numeric_value = min(int(maximum), numeric_value)
                return str(numeric_value)

            def _float_config(key, default, minimum=0.0, maximum=None):
                try:
                    numeric_value = float(self.config.get(key, default))
                except (TypeError, ValueError):
                    numeric_value = float(default)
                numeric_value = max(float(minimum), numeric_value)
                if maximum is not None:
                    numeric_value = min(float(maximum), numeric_value)
                return f"{numeric_value:g}"

            def _choice_config(key, default, allowed):
                value = str(self.config.get(key, default) or default).strip().lower()
                return value if value in allowed else default

            authnd_auto_enabled = bool(self.config.get('authnd_token_concurrency_auto', True))
            if authnd_auto_enabled:
                authnd_token_limit, authnd_subprocess_limit, _authnd_cores = _authnd_auto_token_limits()
            else:
                authnd_token_limit = _positive_int_config('authnd_token_concurrency', 1)
                authnd_subprocess_limit = _positive_int_config('authnd_token_subprocess_concurrency', 1)

            # Initialize glossary-related environment variables
            env_mappings = [
                ('GLOSSARY_SYSTEM_PROMPT', self.config.get('manual_glossary_prompt', getattr(self, 'manual_glossary_prompt', ''))),
                ('AUTO_GLOSSARY_PROMPT', self.config.get('unified_auto_glosary_prompt3', getattr(self, 'unified_auto_glosary_prompt3', ''))),
                ('GLOSSARY_REFINEMENT_ENABLED', '1' if self.config.get('glossary_refinement_enabled', False) else '0'),
                ('GLOSSARY_REFINEMENT_SYSTEM_PROMPT', self.config.get('glossary_refinement_system_prompt') or getattr(self, 'glossary_refinement_system_prompt', '')),
                ('GLOSSARY_REFINEMENT_USER_PROMPT', self.config.get('glossary_refinement_user_prompt', getattr(self, 'glossary_refinement_user_prompt', ''))),
                ('GLOSSARY_REFINEMENT_TYPE_MODE', self.config.get('glossary_refinement_type_mode', 'all')),
                ('GLOSSARY_REFINEMENT_SELECTED_TYPES', ','.join(self.config.get('glossary_refinement_selected_types', []))),
                ('GLOSSARY_REFINEMENT_CHUNKING_MODE', self.config.get('glossary_refinement_chunking_mode', 'all')),
                ('GLOSSARY_REFINEMENT_SKIP_DEDUPE', '1' if self.config.get('glossary_refinement_skip_dedupe', False) else '0'),
                ('GLOSSARY_REFINEMENT_WAIT_FOR_COMPLETION', '1' if self.config.get('glossary_refinement_wait_for_completion', False) else '0'),
                ('GLOSSARY_REFINEMENT_REOPEN_ON_SOURCE_CHANGE', '1' if self.config.get('glossary_refinement_reopen_on_source_change', False) else '0'),
                ('GLOSSARY_DISABLE_HONORIFICS_FILTER', '1' if self.config.get('glossary_disable_honorifics_filter', False) else '0'),
                ('GLOSSARY_STRIP_HONORIFICS', '1' if self.config.get('strip_honorifics', False) else '0'),
                ('GLOSSARY_FUZZY_THRESHOLD', str(self.config.get('glossary_fuzzy_threshold', 0.90))),
                ('GLOSSARY_ENTRY_TYPE_FILTER_MODE', self.config.get('glossary_entry_type_filter_mode', 'none')),
                ('GLOSSARY_TRANSLATION_PROMPT', self.config.get('glossary_translation_prompt', '')),
                ('GLOSSARY_FORMAT_INSTRUCTIONS', self.config.get('glossary_format_instructions', '')),
                ('GLOSSARY_USE_LEGACY_CSV', '1' if self.config.get('glossary_use_legacy_csv', False) else '0'),
                ('GLOSSARY_MAX_SENTENCES', str(self.config.get('glossary_max_sentences', 10))),
                
                # OpenRouter settings
                ('OPENROUTER_USE_HTTP_ONLY', '1' if self.config.get('openrouter_use_http_only', False) else '0'),
                ('OPENROUTER_ACCEPT_IDENTITY', '1' if self.config.get('openrouter_accept_identity', False) else '0'),
                ('OPENROUTER_PREFERRED_PROVIDER', (str(self.config.get('openrouter_preferred_provider', 'Auto') or '').strip() or 'Auto')),

                # Thinking toggles
                ('ENABLE_DEEPSEEK_THINKING', '1' if self.config.get('enable_deepseek_thinking', True) else '0'),
                ('DEEPSEEK_USE_RESPONSES_API', '1' if self.config.get('deepseek_use_responses_api', False) else '0'),
                ('ENABLE_STREAMING', '1' if bool(getattr(self, 'enable_streaming_var', self.config.get('enable_streaming', False))) else '0'),
                ('ALLOW_BATCH_STREAM_LOGS', '1' if bool(getattr(self, 'allow_batch_stream_logs_var', self.config.get('allow_batch_stream_logs', False))) else '0'),
                ('ALLOW_AUTHGPT_BATCH_STREAM_LOGS', '1' if bool(getattr(self, 'allow_authgpt_batch_stream_logs_var', self.config.get('allow_authgpt_batch_stream_logs', False))) else '0'),
                ('ENABLE_THOUGHTS', '1' if self.config.get('enable_thoughts', True) else '0'),
                ('GEMINI_SERVICE_TIER', str(self.config.get('gemini_service_tier', 'off') or 'off')),
                ('STREAM_THINKING_LOGS', '1' if bool(getattr(self, 'stream_thinking_logs_var', self.config.get('stream_thinking_logs', False))) else '0'),
                ('AUTHZA_USE_GENERAL_API', '1' if _bool_config('authza_use_general_api', False) else '0'),
                ('HTML2TEXT_ESCAPE_SNOB', '1' if self.config.get('html2text_escape_snob', False) else '0'),
                ('CONVERT_BR_TO_PARAGRAPHS', '1' if self.config.get('convert_br_to_paragraphs', True) else '0'),
                ('PRESERVE_ASTERISK_SEPARATOR_LINES', '1' if self.config.get('preserve_asterisk_separator_lines', True) else '0'),
                
                # General settings
                ('EXTRACTION_WORKERS', str(self.config.get('extraction_workers', 1)) if self.config.get('enable_parallel_extraction', False) else '1'),
                ('PDF_EXTRACTION_WORKERS', str(self.config.get('pdf_extraction_workers', 'auto') or 'auto')),
                ('PDF_PARAGRAPH_ALIGNMENT', str(self.config.get('pdf_paragraph_alignment', 'source') or 'source')),
                ('PDF_HEADER_ALIGNMENT', str(self.config.get('pdf_header_alignment', 'source') or 'source')),
                ('PDF_PARAGRAPH_JUSTIFICATION', str(self.config.get('pdf_paragraph_justification', 'source') or 'source')),
                ('PDF_RTL_PARAGRAPH_LAYOUT', '1' if self.config.get('pdf_rtl_paragraph_layout', False) else '0'),
                ('ENABLE_GUI_YIELD', '1' if self.config.get('enable_gui_yield', True) else '0'),
                ('RETAIN_SOURCE_EXTENSION', '1' if self.config.get('retain_source_extension', False) else '0'),
                ('DOWNLOAD_REMOTE_IMAGE_URLS', '1' if self.config.get('download_remote_image_urls', False) else '0'),
                ('REMOTE_IMAGE_DOWNLOAD_WORKERS', str(_int_config('remote_image_download_workers', 4, 1, 32))),
                ('REMOTE_IMAGE_DOWNLOAD_INTERVAL', _float_config('remote_image_download_interval', 0.5, 0.0, 60.0)),
                ('AUTHND_TOKEN_CONCURRENCY_AUTO', '1' if authnd_auto_enabled else '0'),
                ('AUTHND_TOKEN_CONCURRENCY', str(authnd_token_limit)),
                ('AUTHND_TOKEN_SUBPROCESS_CONCURRENCY', str(authnd_subprocess_limit)),
                ('AUTHND_TOKEN_TIMEOUT', _int_config('authnd_token_timeout', 180, 30, 600)),
                ('ORDERED_BATCH_DISPATCH_TIMEOUT', _int_config('dispatch_order_timeout', 3, 0, 120)),
                ('GEMINI_FREE_ADAPTIVE_SPLIT', '1' if _bool_config('gemini_free_adaptive_split', True) else '0'),
                ('GLOSSARION_TOR_ENABLED', '1' if _bool_config('tor_proxy_enabled', False) else '0'),
                ('GEMINI_FREE_HTML_TEXT_NODE_TRANSPORT', '1' if _bool_config('gemini_free_html_text_node_transport', True) else '0'),
                ('GEMINI_FREE_SUBCHUNK_PROMPT_CHARS', _int_config('gemini_free_subchunk_prompt_chars', 7000, 300, 7000)),
                ('GEMINI_FREE_SUBCHUNK_URL_CHARS', _int_config('gemini_free_subchunk_url_chars', 14500, 1000, 200000)),
                ('GEMINI_FREE_SUBCHUNK_SAFETY_CHARS', _int_config('gemini_free_subchunk_safety_chars', 600, 0, 20000)),
                ('GEMINI_FREE_MIN_SUBCHUNK_BODY_CHARS', _int_config('gemini_free_min_subchunk_body_chars', 80, 1, 50000)),
                ('GEMINI_FREE_SUBCHUNK_CONCURRENCY', _int_config('gemini_free_subchunk_concurrency', 0, 0, 64)),
                ('GEMINI_FREE_SUBCHUNK_START_DELAY', _float_config('gemini_free_subchunk_start_delay', 5.0, 0.0, 60.0)),
                ('GEMINI_FREE_SUBCHUNK_TIMEOUT', _int_config('gemini_free_subchunk_timeout', 0, 0, 7200)),
                ('GEMINI_FREE_SUBCHUNK_PAYLOAD_FORMAT', _choice_config('gemini_free_subchunk_payload_format', 'auto', {'auto', 'html', 'text'})),
                ('GEMINI_FREE_HTML_SPLITTER', _choice_config('gemini_free_html_splitter', 'beautifulsoup4', {'beautifulsoup4', 'regex'})),
                ('GEMINI_FREE_SUBCHUNK_BALANCER', _choice_config('gemini_free_subchunk_balancer', 'balanced', {'balanced', 'greedy'})),
            ]
            
            # Add QA Scanner environment variables
            qa_settings = self.config.get('qa_scanner_settings', {})
            ai_hunter_config = self.config.get('ai_hunter_config', {})
            qa_env_mappings = [
                ('QA_SCANNER_SETTINGS_JSON', self._get_qa_scanner_settings_json()),
                ('QA_FOREIGN_CHAR_THRESHOLD', str(qa_settings.get('foreign_char_threshold', 0))),
                ('QA_TARGET_LANGUAGE', qa_settings.get('target_language', 'english')),
                ('QA_WHITELIST_EMOTICON_PATTERNS', '1' if qa_settings.get('whitelist_emoticon_patterns', False) else '0'),
                ('QA_EXCLUDE_RUBY_TAGS', '1' if qa_settings.get('exclude_ruby_tags', False) else '0'),
                ('QA_EMOTICON_PATTERNS_JSON', json.dumps(qa_settings.get('emoticon_patterns', DEFAULT_EMOTICON_PATTERNS), ensure_ascii=False)),
                ('QA_EMOTICON_PATTERNS_ARE_REGEX', '1' if qa_settings.get('emoticon_patterns_are_regex', False) else '0'),
                ('QA_CHECK_ENCODING', '1' if qa_settings.get('check_encoding_issues', False) else '0'),
                ('QA_CHECK_REPETITION', '1' if qa_settings.get('check_repetition', True) else '0'),
                ('QA_CHECK_ARTIFACTS', '1' if qa_settings.get('check_translation_artifacts', False) else '0'),
                ('QA_CHECK_GLOSSARY_LEAKAGE', '1' if qa_settings.get('check_glossary_leakage', True) else '0'),
                ('QA_MIN_FILE_LENGTH', str(qa_settings.get('min_file_length', 0))),
                ('QA_REPORT_FORMAT', qa_settings.get('report_format', 'detailed')),
                ('QA_AUTO_SAVE_REPORT', '1' if qa_settings.get('auto_save_report', True) else '0'),
                ('QA_CACHE_ENABLED', '1' if qa_settings.get('cache_enabled', True) else '0'),
                ('QA_SDLXLIFF_TAG_RETENTION_THRESHOLD', str(qa_settings.get('sdlxliff_tag_retention_threshold', 0.9))),
                ('QA_SDLXLIFF_TAG_SURPLUS_TOLERANCE', str(qa_settings.get('sdlxliff_tag_surplus_tolerance', 0.05))),
                ('QA_SDLXLIFF_MIN_SOURCE_PARAGRAPH_TAGS', str(qa_settings.get('sdlxliff_min_source_paragraph_tags', 20))),
                ('QA_CHECK_SILENT_TRUNCATION', '1' if qa_settings.get('check_silent_truncation', False) else '0'),
                ('QA_CHECK_POTENTIAL_TRUNCATION', '1' if qa_settings.get('check_potential_truncation', False) else '0'),
                ('QA_CHECK_AI_TRUNCATION_DETECTION', '1' if qa_settings.get('check_ai_truncation_detection', False) else '0'),
                ('QA_CHECK_WORD_COUNT_RATIO', '1' if qa_settings.get('check_word_count_ratio', True) else '0'),
                ('QA_CHECK_MISSING_BEAUTIFULSOUP_TAGS', '1' if qa_settings.get('check_missing_beautifulsoup_tags', False) else '0'),
                ('AI_HUNTER_MAX_WORKERS', str(ai_hunter_config.get('ai_hunter_max_workers', max(1, (os.cpu_count() or 4) // 2)))),
            ]
            
            # Add Manga Integration and Manga Settings Dialog environment variables
            ms = self.config.get('manga_settings', {}) if isinstance(self.config.get('manga_settings', {}), dict) else {}
            inpaint = ms.get('inpainting', {}) if isinstance(ms.get('inpainting', {}), dict) else {}
            rendering = ms.get('rendering', {}) if isinstance(ms.get('rendering', {}), dict) else {}
            font_cfg = ms.get('font_sizing', {}) if isinstance(ms.get('font_sizing', {}), dict) else {}

            # Convenience getters with fallbacks to top-level keys used by MangaIntegration
            def _rgb_list_to_str(lst, default):
                try:
                    if isinstance(lst, (list, tuple)) and len(lst) == 3:
                        return f"{int(lst[0])},{int(lst[1])},{int(lst[2])}"
                except Exception:
                    pass
                return default

            manga_env_mappings = [
                ('MANGA_FULL_PAGE_CONTEXT', '1' if self.config.get('manga_full_page_context', False) else '0'),
                ('MANGA_VISUAL_CONTEXT_ENABLED', '1' if self.config.get('manga_visual_context_enabled', True) else '0'),
                ('MANGA_CREATE_SUBFOLDER', '1' if self.config.get('manga_create_subfolder', True) else '0'),
                ('MANGA_BG_OPACITY', str(self.config.get('manga_bg_opacity', 130))),
                ('MANGA_BG_STYLE', str(self.config.get('manga_bg_style', 'circle'))),
                ('MANGA_BG_REDUCTION', str(self.config.get('manga_bg_reduction', 1.0))),
                ('MANGA_FONT_SIZE', str(self.config.get('manga_font_size', 0))),
                ('MANGA_FONT_STYLE', str(self.config.get('manga_font_style', 'Default'))),
                ('MANGA_FONT_PATH', str(self.config.get('manga_font_path', ''))),
                ('MANGA_FONT_SIZE_MODE', str(self.config.get('manga_font_size_mode', 'fixed'))),
                ('MANGA_FONT_SIZE_MULTIPLIER', str(self.config.get('manga_font_size_multiplier', 1.0))),
                ('MANGA_MAX_FONT_SIZE', str(self.config.get('manga_max_font_size', rendering.get('auto_max_size', font_cfg.get('max_size', 48))))),
                ('MANGA_AUTO_MIN_SIZE', str(rendering.get('auto_min_size', font_cfg.get('min_size', 10)))),
                ('MANGA_FREE_TEXT_ONLY_BG_OPACITY', '1' if self.config.get('manga_free_text_only_bg_opacity', True) else '0'),
                ('MANGA_FORCE_CAPS_LOCK', '1' if self.config.get('manga_force_caps_lock', True) else '0'),
                ('MANGA_STRICT_TEXT_WRAPPING', '1' if self.config.get('manga_strict_text_wrapping', True) else '0'),
                ('MANGA_CONSTRAIN_TO_BUBBLE', '1' if self.config.get('manga_constrain_to_bubble', True) else '0'),
                ('MANGA_TEXT_COLOR', _rgb_list_to_str(self.config.get('manga_text_color', [102,0,0]), '102,0,0')),
                ('MANGA_SHADOW_ENABLED', '1' if self.config.get('manga_shadow_enabled', True) else '0'),
                ('MANGA_SHADOW_COLOR', _rgb_list_to_str(self.config.get('manga_shadow_color', [204,128,128]), '204,128,128')),
                ('MANGA_SHADOW_OFFSET_X', str(self.config.get('manga_shadow_offset_x', 2))),
                ('MANGA_SHADOW_OFFSET_Y', str(self.config.get('manga_shadow_offset_y', 2))),
                ('MANGA_SHADOW_BLUR', str(self.config.get('manga_shadow_blur', 0))),
                ('MANGA_SKIP_INPAINTING', '1' if self.config.get('manga_skip_inpainting', False) else '0'),
                ('MANGA_INPAINT_QUALITY', str(self.config.get('manga_inpaint_quality', 'high'))),
                ('MANGA_INPAINT_DILATION', str(self.config.get('manga_inpaint_dilation', 15))),
                ('MANGA_INPAINT_PASSES', str(self.config.get('manga_inpaint_passes', 2))),
                ('MANGA_INPAINT_METHOD', str(inpaint.get('method', 'local'))),
                ('MANGA_LOCAL_INPAINT_METHOD', str(inpaint.get('local_method', 'anime_onnx'))),
                # New: pass worker process control to environment for downstream components
                ('MANGA_DISABLE_WORKER_PROCESS', '1' if inpaint.get('disable_worker_process', False) else '0'),
                ('MANGA_FONT_ALGORITHM', str(font_cfg.get('algorithm', 'smart'))),
                ('MANGA_PREFER_LARGER', '1' if font_cfg.get('prefer_larger', True) else '0'),
                ('MANGA_BUBBLE_SIZE_FACTOR', '1' if font_cfg.get('bubble_size_factor', True) else '0'),
                ('MANGA_LINE_SPACING', str(font_cfg.get('line_spacing', 1.3))),
                ('MANGA_MAX_LINES', str(font_cfg.get('max_lines', 10))),
                ('MANGA_QWEN2VL_MODEL_SIZE', str(self.config.get('qwen2vl_model_size', '1'))),
                ('MANGA_RAPIDOCR_USE_RECOGNITION', '1' if self.config.get('rapidocr_use_recognition', True) else '0'),
                ('MANGA_RAPIDOCR_LANGUAGE', str(self.config.get('rapidocr_language', 'auto'))),
                ('MANGA_RAPIDOCR_DETECTION_MODE', str(self.config.get('rapidocr_detection_mode', 'document'))),
                # Prompt lengths for quick sanity without leaking content
                ('MANGA_FULL_PAGE_CONTEXT_PROMPT_LEN', str(len(self.config.get('manga_full_page_context_prompt', '') or ''))),
                ('MANGA_OCR_PROMPT_LEN', str(len(self.config.get('manga_ocr_prompt', '') or ''))),
            ]
            
            # Add Manga Advanced Settings (Memory Management)
            manga_adv = ms.get('advanced', {}) if isinstance(ms.get('advanced', {}), dict) else {}
            manga_advanced_env_mappings = [
                ('MANGA_AUTO_CLEANUP_MODELS', '1' if manga_adv.get('auto_cleanup_models', False) else '0'),
                ('MANGA_UNLOAD_MODELS_AFTER_TRANSLATION', '1' if manga_adv.get('unload_models_after_translation', False) else '0'),
                ('MANGA_PARALLEL_PROCESSING', '1' if manga_adv.get('parallel_processing', False) else '0'),
                ('MANGA_MAX_WORKERS', str(manga_adv.get('max_workers', 4))),
                ('MANGA_PARALLEL_PANEL_TRANSLATION', '1' if manga_adv.get('parallel_panel_translation', False) else '0'),
                ('MANGA_PANEL_MAX_WORKERS', str(manga_adv.get('panel_max_workers', 2))),
                ('MANGA_DEBUG_MODE', '1' if manga_adv.get('debug_mode', False) else '0'),
                ('MANGA_SAVE_INTERMEDIATE', '1' if manga_adv.get('save_intermediate', False) else '0'),
                ('MANGA_CONCISE_LOGS', '1' if manga_adv.get('concise_logs', True) else '0'),
                # Note: MANGA_SKIP_INPAINTING is set from manga_skip_inpainting (line 8951) - don't duplicate here
            ]

            # Combine all environment variable mappings
            env_mappings.extend(qa_env_mappings)
            env_mappings.extend(manga_env_mappings)
            env_mappings.extend(manga_advanced_env_mappings)

            # Add additional environment variables converted from legacy Tkinter to PySide6 attributes
            try:
                import json as _json
            except Exception:
                _json = json
            
            # Calculate resolved_max_retry_tokens
            current_max_tokens = getattr(self, 'max_output_tokens', 128000)
            resolved_max_retry_tokens = self._resolve_max_retry_tokens(current_max_tokens) if hasattr(self, '_resolve_max_retry_tokens') else current_max_tokens
            output_mode = self._get_output_mode()
            env_text_extraction_method = getattr(self, 'text_extraction_method_var', 'standard') if hasattr(self, 'text_extraction_method_var') else 'standard'
            env_extraction_mode = getattr(self, 'extraction_mode_var', 'smart')
            env_enhanced_filtering = getattr(self, 'enhanced_filtering_var', 'smart')
            if output_mode == 'vision':
                env_text_extraction_method = 'enhanced'
                env_extraction_mode = 'enhanced'
                env_enhanced_filtering = getattr(self, 'file_filtering_level_var', env_enhanced_filtering)
            env_auto_glossary_mode = self._current_auto_glossary_mode()
            (
                env_glossary_merging_enabled,
                env_glossary_merge_count,
                env_glossary_chapter_split,
            ) = self._current_glossary_request_env(
                force_balanced_request_merging=(output_mode == 'vision' and env_auto_glossary_mode == 'balanced')
            )

            def _bool_value(value, default=False):
                if value is None:
                    return bool(default)
                if hasattr(value, 'isChecked'):
                    return bool(value.isChecked())
                if hasattr(value, 'get'):
                    try:
                        return _bool_value(value.get(), default)
                    except Exception:
                        return bool(default)
                if isinstance(value, str):
                    return value.strip().lower() in ('1', 'true', 'yes', 'on')
                return bool(value)

            def _explicit_blank_prompt_value(attr_name, config_key, default_attr):
                if hasattr(self, attr_name):
                    value = getattr(self, attr_name)
                    return str(value if value is not None else '')
                if config_key in self.config:
                    value = self.config.get(config_key)
                    return str(value if value is not None else '')
                return str(getattr(self, default_attr, '') or '')
            
            extra_env_mappings = [
                # Rolling summary
                ('USE_ROLLING_SUMMARY', '1' if getattr(self, 'rolling_summary_var', False) else '0'),
                ('SUMMARY_ROLE', getattr(self, 'summary_role_var', 'system')),
                ('ROLLING_SUMMARY_EXCHANGES', str(getattr(self, 'rolling_summary_exchanges_var', '5'))),
                ('ROLLING_SUMMARY_MODE', getattr(self, 'rolling_summary_mode_var', 'replace')),
                ('ROLLING_SUMMARY_SYSTEM_PROMPT', getattr(self, 'rolling_summary_system_prompt', getattr(self, 'default_rolling_summary_system_prompt', ''))),
                ('ROLLING_SUMMARY_USER_PROMPT', getattr(self, 'rolling_summary_user_prompt', getattr(self, 'default_rolling_summary_user_prompt', ''))),
                ('ROLLING_SUMMARY_MAX_ENTRIES', str(getattr(self, 'rolling_summary_max_entries_var', '10'))),
                ('ROLLING_SUMMARY_MAX_TOKENS', str(getattr(self, 'rolling_summary_max_tokens_var', '-1'))),

                # Retry/network controls
                ('RETRY_TRUNCATED', '1' if getattr(self, 'retry_truncated_var', False) else '0'),
                ('PRESERVE_ORIGINAL_TEXT_ON_FAILURE', '1' if getattr(self, 'preserve_original_text_var', False) else '0'),
                ('SAVE_PARTIAL_RESULTS', '1' if getattr(self, 'save_partial_results_var', True) else '0'),
                ('SAVE_PROHIBITED_RESULTS', '1' if getattr(self, 'save_prohibited_results_var', False) else '0'),
                ('DISABLE_EMPTY_SAFETY_HEURISTIC', '1' if getattr(self, 'disable_empty_safety_heuristic_var', True) else '0'),
                ('MISSING_FINISH_AS_PROHIBITED', '1' if getattr(self, 'unknown_finish_as_prohibited_var', False) else '0'),
                ('UNKNOWN_FINISH_AS_PROHIBITED', '1' if getattr(self, 'unknown_finish_as_prohibited_var', False) else '0'),
                ('MAX_RETRY_TOKENS', str(resolved_max_retry_tokens)),
                ('TRUNCATION_RETRY_ATTEMPTS', str(getattr(self, 'truncation_retry_attempts_var', '3'))),
                # Char-ratio truncation (silent truncation detector)
                ('CHAR_RATIO_TRUNCATION_ENABLED', '1' if getattr(self, 'char_ratio_truncation_var', False) else '0'),
                ('CHAR_RATIO_TRUNCATION_PERCENT', str(getattr(self, 'char_ratio_truncation_percent_var', '50'))),
                ('CHAR_RATIO_TRUNCATION_ATTEMPTS', str(getattr(self, 'char_ratio_truncation_attempts_var', '1'))),
                ('CHAR_RATIO_MIN_OUTPUT_CHARS', str(getattr(self, 'char_ratio_min_output_chars_var', '100'))),
                ('RETRY_SPLIT_FAILED', '1' if getattr(self, 'retry_split_failed_var', False) else '0'),
                ('SPLIT_FAILED_RETRY_ATTEMPTS', str(getattr(self, 'split_failed_retry_attempts_var', '1'))),
                ('RETRY_DUPLICATE_BODIES', '1' if getattr(self, 'retry_duplicate_var', False) else '0'),
                ('DUPLICATE_LOOKBACK_CHAPTERS', str(getattr(self, 'duplicate_lookback_var', '5'))),
                ('DISABLE_QA_MARKER_CHECKS', '1' if getattr(self, 'disable_qa_marker_checks_var', True) else '0'),
                ('QA_MARKER_LENGTH_LIMIT', str(getattr(self, 'qa_marker_length_limit_var', '500'))),
                ('DISABLE_REFUSAL_CHECKS', '1' if getattr(self, 'disable_refusal_checks_var', False) else '0'),
                ('REFUSAL_PATTERN_LENGTH_LIMIT', str(getattr(self, 'refusal_pattern_length_limit_var', '1000'))),
                ('RETRY_TIMEOUT', '1' if getattr(self, 'retry_timeout_var', self.config.get('retry_timeout', False)) else '0'),
                ('CHUNK_TIMEOUT', str(getattr(self, 'chunk_timeout_var', '1800'))),
                ('ENABLE_HTTP_TUNING', '1' if self.config.get('enable_http_tuning', False) else '0'),
                ('CONNECT_TIMEOUT', str(getattr(self, 'connect_timeout_var', '10'))),
                ('READ_TIMEOUT', str(getattr(self, 'read_timeout_var', '180'))),
                ('HTTP_POOL_CONNECTIONS', str(getattr(self, 'http_pool_connections_var', '20'))),
                ('HTTP_POOL_MAXSIZE', str(getattr(self, 'http_pool_maxsize_var', '50'))),
                ('IGNORE_RETRY_AFTER', '1' if self.config.get('ignore_retry_after', False) else '0'),
                ('MAX_RETRIES', str(self._resolve_max_retries())),

                # QA/meta preferences
                ('QA_AUTO_SEARCH_OUTPUT', '1' if getattr(self, 'qa_auto_search_output_var', True) else '0'),
                ('INDEFINITE_RATE_LIMIT_RETRY', '1' if getattr(self, 'indefinite_rate_limit_retry_var', False) else '0'),

                # Post-translation scanning phase
                ('SCAN_PHASE_ENABLED', '1' if getattr(self, 'scan_phase_enabled_var', False) else '0'),
                ('SCAN_PHASE_MODE', self._get_scan_phase_mode()),
                ('QA_SCANNER_SETTINGS_JSON', self._get_qa_scanner_settings_json()),

                # Book title handling
                ('TRANSLATE_BOOK_TITLE', '1' if getattr(self, 'translate_book_title_var', True) else '0'),
                ('SKIP_TXT_TITLE_TRANSLATION', '1' if getattr(self, 'skip_txt_title_translation_var', True) else '0'),
                ('SKIP_PDF_TITLE_TRANSLATION', '1' if getattr(self, 'skip_pdf_title_translation_var', False) else '0'),
                ('SKIP_IMAGE_TITLE_TRANSLATION', '1' if getattr(self, 'skip_image_title_translation_var', True) else '0'),
                ('SKIP_TITLE_TAG_TRANSLATION', '1' if getattr(self, 'skip_title_tag_translation_var', False) else '0'),
                ('IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT', str(getattr(
                    self,
                    'image_only_title_tag_system_prompt',
                    self.config.get(
                        'image_only_title_tag_system_prompt',
                        DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT,
                    ),
                ) or DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT)),
                ('BOOK_TITLE_PROMPT', getattr(self, 'book_title_prompt', '')),
                ('GLOSSARY_INCLUDE_BOOK_TITLE', '1' if getattr(self, 'include_book_title_glossary_var', True) else '0'),
                ('GLOSSARY_AUTO_INJECT_BOOK_TITLE', '1' if getattr(self, 'auto_inject_book_title_var', self.config.get('auto_inject_book_title', False)) else '0'),
                ('GLOSSARY_REQUEST_MERGING_ENABLED', env_glossary_merging_enabled),
                ('GLOSSARY_REQUEST_MERGE_COUNT', env_glossary_merge_count),
                ('GLOSSARY_ENABLE_CHAPTER_SPLIT', env_glossary_chapter_split),
                ('GLOSSARY_SKIP_TITLE_HEADER_ONLY', self._glossary_skip_title_header_only_env_value()),
                ('GLOSSARY_ADD_MINIMAL_PASS', self._glossary_add_minimal_pass_env_value()),
                ('GLOSSARY_MATCH_ENGINE', self._glossary_match_engine_env_value()),
                *self._strict_matching_env_dict().items(),
                *self._unified_glossary_env_dict().items(),
                ('GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL', '1' if getattr(self, 'never_consider_in_between_files_as_special_var', self.config.get('never_consider_in_between_files_as_special', True)) else '0'),

                # Safety/merge toggles
                ('EMERGENCY_PARAGRAPH_RESTORE', '1' if getattr(self, 'emergency_restore_var', False) else '0'),
                ('DISABLE_CHAPTER_MERGING', '1' if getattr(self, 'disable_chapter_merging_var', False) else '0'),
                # Request merging (combine multiple chapters into single API request)
                ('REQUEST_MERGING_ENABLED', '1' if getattr(self, 'request_merging_enabled_var', False) else '0'),
                ('REQUEST_MERGE_COUNT', str(getattr(self, 'request_merge_count_var', '3'))),
                ('SPLIT_THE_MERGE', '1' if getattr(self, 'split_the_merge_var', False) else '0'),
                ('DISABLE_MERGE_FALLBACK', '1' if getattr(self, 'disable_merge_fallback_var', False) else '0'),
                # Synthetic headers for merged requests (improves Split-the-Merge reliability)
                ('SYNTHETIC_MERGE_HEADERS', '1' if getattr(self, 'synthetic_merge_headers_var', True) else '0'),

                # Image translation controls
                ('ENABLE_IMAGE_TRANSLATION', '1' if getattr(self, 'enable_image_translation_var', False) else '0'),
                ('PROCESS_WEBNOVEL_IMAGES', '1' if getattr(self, 'process_webnovel_images_var', True) else '0'),
                ('WEBNOVEL_MIN_HEIGHT', str(getattr(self, 'webnovel_min_height_var', '1000'))),
                ('MAX_IMAGES_PER_CHAPTER', str(getattr(self, 'max_images_per_chapter_var', '-1'))),
                ('IMAGE_CHUNK_HEIGHT', str(getattr(self, 'image_chunk_height_var', '1500'))),
                ('MAX_OUTPUT_TOKENS', str(getattr(self, 'max_output_tokens', 128000))),
                ('ENABLE_CHUNK_PROGRESS', '1' if _bool_value(getattr(self, 'enable_chunk_progress_var', self.config.get('enable_chunk_progress', True))) else '0'),
                ('HIDE_IMAGE_TRANSLATION_LABEL', '1' if getattr(self, 'hide_image_translation_label_var', True) else '0'),
                ('DISABLE_EPUB_GALLERY', '1' if getattr(self, 'disable_epub_gallery_var', True) else '0'),
                ('SKIP_NON_SPINE_SPECIAL_FILES', '1' if getattr(self, 'skip_non_spine_special_files_var', False) else '0'),
                ('SKIP_UNREFERENCED_EPUB_IMAGES', '1' if getattr(self, 'skip_unreferenced_epub_images_var', False) else '0'),
                ('DISABLE_AUTOMATIC_COVER_CREATION', '1' if getattr(self, 'disable_automatic_cover_creation_var', True) else '0'),
                # EPUB layout mode (auto / epub2 / epub3)
                ('EPUB_LAYOUT_MODE', getattr(self, 'epub_layout_mode_var', self.config.get('epub_layout_mode', 'auto')) or 'auto'),
                ('LEGACY_EPUB_STRUCTURE', '1' if getattr(self, 'epub_layout_mode_var', self.config.get('epub_layout_mode', 'auto')) == 'epub2' else '0'),
                # New: Use/translate source toc.ncx
                ('USE_TOC_NCX', '1' if getattr(self, 'use_toc_ncx_var', self.config.get('use_toc_ncx', True)) else '0'),
                ('TRANSLATE_TOC_NCX', '1' if getattr(self, 'use_toc_ncx_var', self.config.get('use_toc_ncx', True)) else '0'),  # Unified: mirrors USE_TOC_NCX
                ('SKIP_DUPLICATE_TOC_TRANSLATION', '1' if getattr(self, 'skip_duplicate_toc_translation_var', self.config.get('skip_duplicate_toc_translation', False)) else '0'),
                ('USE_P_TAG_TOC_FALLBACK', '1' if getattr(self, 'use_p_tag_toc_fallback_var', self.config.get('use_p_tag_toc_fallback', False)) else '0'),
                # New: Translate special files (cover, nav, toc, message, etc.)
                ('TRANSLATE_SPECIAL_FILES', '1' if getattr(self, 'translate_special_files_var', False) else '0'),
                ('TRANSLATE_ALL_NUMBERED_HTML', '1' if getattr(self, 'translate_all_numbered_html_var', True) else '0'),
                # Custom special file keywords
                ('SPECIAL_FILE_KEYWORDS', getattr(self, 'special_file_keywords_var', '')),
                ('SPECIAL_FILE_EXACT', getattr(self, 'special_file_exact_var', '')),
                ('DISABLE_ZERO_DETECTION', '1' if getattr(self, 'disable_zero_detection_var', True) else '0'),
                ('DUPLICATE_DETECTION_MODE', getattr(self, 'duplicate_detection_mode_var', 'basic')),
                ('ENABLE_DECIMAL_CHAPTERS', '1' if getattr(self, 'enable_decimal_chapters_var', False) else '0'),

                # Watermark/image toggles
                ('ENABLE_WATERMARK_REMOVAL', '1' if getattr(self, 'enable_watermark_removal_var', True) else '0'),
                ('SAVE_CLEANED_IMAGES', '1' if getattr(self, 'save_cleaned_images_var', False) else '0'),
                ('OUTPUT_MODE', output_mode),
                ('VISION_OCR_FIRST', '1' if output_mode == 'vision' else '0'),
                ('VISION_OCR_PROMPT', str(getattr(self, 'vision_ocr_prompt', self.config.get('vision_ocr_prompt', '')))),
                ('VISION_OCR_USER_PROMPT', str(getattr(self, 'vision_ocr_user_prompt', self.config.get('vision_ocr_user_prompt', '')))),
                ('VISION_OCR_BATCH_TRANSLATION', '1' if _bool_value(getattr(self, 'vision_ocr_batch_translation_var', True)) else '0'),
                ('VISION_OCR_BATCH_SIZE', str(getattr(self, 'vision_ocr_batch_size_var', '-1'))),
                ('VISION_OCR_SKIP_TRANSLATION', '1' if _bool_value(getattr(self, 'vision_ocr_skip_translation_var', False)) else '0'),
                ('VISION_OCR_KEEP_IMAGES', '1' if _bool_value(getattr(self, 'vision_ocr_keep_images_var', False)) else '0'),
                ('VISION_OCR_SOURCE_PREPASS', str(getattr(self, 'vision_ocr_source_prepass_var', self.config.get('vision_ocr_source_prepass', 'auto')) or 'auto')),
                ('ENABLE_IMAGE_OUTPUT_MODE', self._get_allowed_image_output_mode()),
                ('ENABLE_VIDEO_OUTPUT_MODE', self._get_allowed_video_output_mode()),
                ('ENABLE_AUDIO_OUTPUT_MODE', '1' if output_mode == 'audio' else '0'),
                ('ENABLE_REFINEMENT_OUTPUT_MODE', '1' if output_mode == 'refinement' else '0'),
                ('BATCH_TRANSLATION', '1' if getattr(self, 'batch_translation_var', self.config.get('batch_translation', True)) else '0'),
                ('BATCH_SIZE', str(getattr(self, 'batch_size_var', self.config.get('batch_size', '5')))),
                ('BATCHING_MODE', self._translation_batching_mode_for_env()),
                ('BATCH_GROUP_SIZE', str(getattr(self, 'batch_group_size_var', '3'))),
                ('CONSERVATIVE_BATCHING', '1' if self._translation_batching_mode_for_env() == 'conservative' else '0'),
                ('MULTIPASS_MODE', '1' if getattr(self, 'multipass_mode_var', self.config.get('multipass_mode', False)) else '0'),
                ('MULTIPASS_REFINEMENT_MODE', self._get_multipass_refinement_mode()),
                ('REFINEMENT_SYSTEM_PROMPT', self.config.get('refinement_system_prompt') or getattr(self, 'refinement_system_prompt', getattr(self, 'default_refinement_system_prompt', ''))),
                ('REFINEMENT_USER_PROMPT', self.config.get('refinement_user_prompt', getattr(self, 'refinement_user_prompt', ''))),
                ('REFINEMENT_FULL_WITH_RAW_SYSTEM_PROMPT', getattr(self, 'refinement_full_with_raw_system_prompt', None) or self.config.get('refinement_full_with_raw_system_prompt') or getattr(self, 'default_refinement_full_with_raw_system_prompt', '')),
                ('REFINEMENT_FULL_WITH_RAW_USER_PROMPT', getattr(self, 'refinement_full_with_raw_user_prompt', self.config.get('refinement_full_with_raw_user_prompt', getattr(self, 'default_refinement_full_with_raw_user_prompt', '')))),
                ('REFINEMENT_FULL_WITH_RAW_RAW_ROLE', str(getattr(self, 'refinement_full_with_raw_raw_role_var', self.config.get('refinement_full_with_raw_raw_role', 'assistant')) or 'assistant').strip().lower()),
                ('REFINEMENT_FULL_WITH_RAW_RAW_HEADER', _explicit_blank_prompt_value('refinement_full_with_raw_raw_header', 'refinement_full_with_raw_raw_header', 'default_refinement_full_with_raw_raw_header')),
                ('REFINEMENT_FULL_WITH_RAW_RAW_FOOTER', _explicit_blank_prompt_value('refinement_full_with_raw_raw_footer', 'refinement_full_with_raw_raw_footer', 'default_refinement_full_with_raw_raw_footer')),
                ('REFINEMENT_FAILED_SYSTEM_PROMPT', getattr(self, 'refinement_failed_system_prompt', None) or self.config.get('refinement_failed_system_prompt') or getattr(self, 'default_refinement_failed_system_prompt', '')),
                ('REFINEMENT_FAILED_USER_PROMPT', getattr(self, 'refinement_failed_user_prompt', self.config.get('refinement_failed_user_prompt', getattr(self, 'default_refinement_failed_user_prompt', '')))),
                ('REFINEMENT_PARTIAL_SYSTEM_PROMPT', getattr(self, 'refinement_partial_system_prompt', None) or self.config.get('refinement_partial_system_prompt') or getattr(self, 'default_refinement_partial_system_prompt', '')),
                ('REFINEMENT_PARTIAL_USER_PROMPT', getattr(self, 'refinement_partial_user_prompt', self.config.get('refinement_partial_user_prompt', getattr(self, 'default_refinement_partial_user_prompt', '')))),
                ('REFINEMENT_PARTIAL_B_SYSTEM_PROMPT', _explicit_blank_prompt_value('refinement_partial_b_system_prompt', 'refinement_partial_b_system_prompt', 'default_refinement_partial_b_system_prompt')),
                ('REFINEMENT_PARTIAL_B_USER_PROMPT', _explicit_blank_prompt_value('refinement_partial_b_user_prompt', 'refinement_partial_b_user_prompt', 'default_refinement_partial_b_user_prompt')),
                ('REFINEMENT_PARTIAL_B2_SYSTEM_PROMPT', _explicit_blank_prompt_value('refinement_partial_b2_system_prompt', 'refinement_partial_b2_system_prompt', 'default_refinement_partial_b2_system_prompt')),
                ('REFINEMENT_PARTIAL_B2_USER_PROMPT', _explicit_blank_prompt_value('refinement_partial_b2_user_prompt', 'refinement_partial_b2_user_prompt', 'default_refinement_partial_b2_user_prompt')),
                ('PARTIAL_B2_ENTRIES_PER_REQUEST', str(getattr(self, 'partial_b2_entries_per_request_var', self.config.get('partial_b2_entries_per_request', '-1')))),
                # Normalize to uppercase so validation in unified_api_client accepts 1K/2K/4K
                ('IMAGE_OUTPUT_RESOLUTION', str(getattr(self, 'image_output_resolution_var', '1K')).upper()),
                ('NANOGPT_VIDEO_DURATION', str(getattr(self, 'nanogpt_video_duration_var', '60')) + 's'),
                ('NANOGPT_VIDEO_RESOLUTION', str(getattr(self, 'nanogpt_video_resolution_var', '720p'))),

                # Prompts
                ('ENABLE_TRANSLATION_CHUNK_PROMPT', '1' if getattr(self, 'enable_translation_chunk_prompt_var', self.config.get('enable_translation_chunk_prompt', False)) else '0'),
                ('INCLUDE_PREVIOUS_CHUNK', '1' if getattr(self, 'include_previous_chunk_var', self.config.get('include_previous_chunk', False)) else '0'),
                ('PREVIOUS_CHUNK_CONTEXT_LIMIT', str(getattr(self, 'previous_chunk_context_limit_var', self.config.get('previous_chunk_context_limit', 3)))),
                ('TRANSLATION_CHUNK_PROMPT_ROLE', str(getattr(self, 'translation_chunk_prompt_role_var', self.config.get('translation_chunk_prompt_role', 'assistant')) or 'assistant').strip().lower()),
                ('TRANSLATION_CHUNK_PROMPT', str(getattr(self, 'translation_chunk_prompt', ''))),
                ('IMAGE_CHUNK_PROMPT', str(getattr(self, 'image_chunk_prompt', ''))),
                ('VISION_OCR_COMBINED_CONTEXT_PROMPT', str(getattr(self, 'vision_ocr_combined_context_prompt', ''))),
                ('VISION_OCR_TRANSLATION_USER_PROMPT', str(getattr(self, 'vision_ocr_translation_user_prompt', ''))),
                ('ASSISTANT_PROMPT', str(getattr(self, 'assistant_prompt', ''))),  # Optional assistant prefill

                # Safety flags
                ('DISABLE_GEMINI_SAFETY', str(self.config.get('disable_gemini_safety', False)).lower()),
                ('GEMINI_SAFETY_THRESHOLD', str(self.config.get('gemini_safety_threshold', 'BLOCK_NONE'))),

                # OpenRouter (duplicates are okay; ensures presence)
                ('OPENROUTER_USE_HTTP_ONLY', '1' if getattr(self, 'openrouter_http_only_var', False) else '0'),
                ('OPENROUTER_ACCEPT_IDENTITY', '1' if getattr(self, 'openrouter_accept_identity_var', False) else '0'),

                # Misc toggles
                ('auto_update_check', str(getattr(self, 'auto_update_check_var', True))),
                ('FORCE_NCX_ONLY', '1' if getattr(self, 'force_ncx_only_var', True) else '0'),
                ('SINGLE_API_IMAGE_CHUNKS', '1' if getattr(self, 'single_api_image_chunks_var', False) else '0'),

                # Thinking features
                ('ENABLE_GEMINI_THINKING', '1' if getattr(self, 'enable_gemini_thinking_var', True) else '0'),
                ('THINKING_BUDGET', str(getattr(self, 'thinking_budget_var', '-1')) if getattr(self, 'enable_gemini_thinking_var', True) else '0'),
                ('GEMINI_THINKING_LEVEL', getattr(self, 'thinking_level_var', 'high')),
                ('ENABLE_GPT_THINKING', '1' if getattr(self, 'enable_gpt_thinking_var', True) else '0'),
                ('GPT_REASONING_TOKENS', str(getattr(self, 'gpt_reasoning_tokens_var', '2000')) if getattr(self, 'enable_gpt_thinking_var', True) else ''),
                ('GPT_EFFORT', getattr(self, 'gpt_effort_var', 'medium')),
                ('OPENROUTER_USE_REASONING_TOKENS', '1' if getattr(self, 'openrouter_use_reasoning_tokens_var', False) else '0'),
                ('PASS_THINKING_TO_OPENAI_COMPATIBLE', '1' if getattr(self, 'pass_thinking_all_openai_var', False) else '0'),
                ('FORCE_SERVICE_TIER_UNKNOWN_ROUTES', '1' if getattr(self, 'force_service_tier_unknown_routes_var', False) else '0'),

                # Custom API endpoints
                ('OPENAI_CUSTOM_BASE_URL', getattr(self, 'openai_base_url_var', '')),
                ('USE_CUSTOM_IMAGE_EDIT_ENDPOINT', '1' if getattr(self, 'use_custom_image_edit_endpoint_var', False) else '0'),
                ('CUSTOM_IMAGE_EDIT_BASE_URL', getattr(self, 'custom_image_edit_endpoint_var', '') if getattr(self, 'use_custom_image_edit_endpoint_var', False) else ''),
                ('OPENAI_IMAGE_EDIT_BASE_URL', getattr(self, 'custom_image_edit_endpoint_var', '') if getattr(self, 'use_custom_image_edit_endpoint_var', False) else ''),
                ('CUSTOM_OPENAI_PREFIX_ROUTES', self._custom_prefix_routes_env_json()),
                ('OLLAMA_SETTINGS_JSON', self._ollama_settings_env_json()),
                ('OPENAI_TTS_ENDPOINT', getattr(self, 'openai_tts_endpoint_var', '') or (getattr(self, 'openai_base_url_var', '') if str(getattr(self, 'openai_base_url_var', '')).rstrip('/').endswith('/audio/speech') else '')),
                ('TTS_VOICE', getattr(self, 'tts_voice_var', '') or ''),
                ('GROQ_API_URL', getattr(self, 'groq_base_url_var', '')),
                ('FIREWORKS_API_URL', getattr(self, 'fireworks_base_url_var', '')),
                ('USE_CUSTOM_OPENAI_ENDPOINT', '1' if getattr(self, 'use_custom_openai_endpoint_var', False) else '0'),
                ('USE_GEMINI_OPENAI_ENDPOINT', '1' if getattr(self, 'use_gemini_openai_endpoint_var', False) else '0'),
                ('GEMINI_OPENAI_ENDPOINT', getattr(self, 'gemini_openai_endpoint_var', 'generativelanguage.googleapis.com')),
                ('OVERRIDE_GEMMA_FOR_CUSTOM_ENDPOINT', '1' if getattr(self, 'override_gemma_for_custom_endpoint_var', True) else '0'),
                ('FORCE_NATIVE_ANTHROPIC', '1' if getattr(self, 'force_native_anthropic_var', False) else '0'),
                ('ANTHROPIC_BASE_URL', getattr(self, 'anthropic_base_url_var', '')),
                ('FUZZY_AUTO_MAPPING', '1' if getattr(self, 'fuzzy_auto_mapping_var', False) else '0'),
                ('FUZZY_AUTO_MAPPING_THRESHOLD', str(getattr(self, 'fuzzy_auto_mapping_threshold_var', 80))),

                # PDF output
                ('ENABLE_PDF_OUTPUT', '1' if getattr(self, 'enable_pdf_output_var', False) else '0'),
                ('PDF_GENERATE_TOC', '1' if self.config.get('pdf_generate_toc', False) else '0'),
                ('PDF_TOC_PAGE_NUMBERS', '1' if self.config.get('pdf_toc_page_numbers', True) else '0'),
                ('PDF_PAGE_NUMBERS', '1' if self.config.get('pdf_page_numbers', True) else '0'),
                ('PDF_PAGE_NUMBER_ALIGNMENT', self.config.get('pdf_page_number_alignment', 'center')),
                ('PDF_RENDER_BATCH_SIZE', str(self.config.get('pdf_render_batch_size', 50))),
                ('PDF_FAST_RENDERING', '1' if self.config.get('pdf_fast_rendering', True) else '0'),
                ('PDF_USE_RAPID_WORKSPACE_COMPILER', '1' if self.config.get('pdf_use_rapid_workspace_compiler', True) else '0'),
                # Image compression quality sub-settings
                ('IMAGE_COMPRESSION_QUALITY', str(self.config.get('image_compression_quality', 80))),
                ('EXCLUDE_COVER_COMPRESSION', '1' if self.config.get('exclude_cover_compression', True) else '0'),
                ('EXCLUDE_GIF_COMPRESSION', '1' if self.config.get('exclude_gif_compression', True) else '0'),


                # Image compression settings
                ('ENABLE_IMAGE_COMPRESSION', '1' if getattr(self, 'enable_image_compression_var', False) else '0'),
                ('AUTO_COMPRESS_ENABLED', '1' if getattr(self, 'auto_compress_enabled_var', True) else '0'),
                ('TARGET_IMAGE_TOKENS', str(getattr(self, 'target_image_tokens_var', '1000'))),
                ('IMAGE_COMPRESSION_FORMAT', getattr(self, 'image_format_var', 'auto')),
                ('WEBP_QUALITY', str(getattr(self, 'webp_quality_var', 85))),
                ('JPEG_QUALITY', str(getattr(self, 'jpeg_quality_var', 85))),
                ('PNG_COMPRESSION', str(getattr(self, 'png_compression_var', 6))),
                ('MAX_IMAGE_DIMENSION', str(getattr(self, 'max_image_dimension_var', '2048'))),
                ('MAX_IMAGE_SIZE_MB', str(getattr(self, 'max_image_size_mb_var', '10'))),
                ('PRESERVE_TRANSPARENCY', '1' if getattr(self, 'preserve_transparency_var', False) else '0'),
                ('OPTIMIZE_FOR_OCR', '1' if getattr(self, 'optimize_for_ocr_var', True) else '0'),
                ('PROGRESSIVE_ENCODING', '1' if getattr(self, 'progressive_encoding_var', True) else '0'),
                ('SAVE_COMPRESSED_IMAGES', '1' if getattr(self, 'save_compressed_images_var', False) else '0'),
                ('USE_FALLBACK_KEYS', '1' if self.config.get('use_fallback_keys', False) else '0'),
                ('USE_MAIN_KEY_FALLBACK', '1' if self.config.get('use_main_key_fallback', True) else '0'),
                ('FALLBACK_KEYS', _json.dumps(self.config.get('fallback_keys', []))),
                ('USE_GLOSSARY_KEYS', '1' if self.config.get('use_glossary_keys', False) else '0'),
                ('GLOSSARY_API_KEYS', _json.dumps(self.config.get('glossary_keys', []))),
                ('USE_GLOSSARY_REFINEMENT_KEYS', '1' if self.config.get('use_glossary_refinement_keys', False) else '0'),
                ('GLOSSARY_REFINEMENT_API_KEYS', _json.dumps(self.config.get('glossary_refinement_keys', []))),
                ('USE_METADATA_KEYS', '1' if self.config.get('use_metadata_keys', False) else '0'),
                ('METADATA_API_KEYS', _json.dumps(self.config.get('metadata_keys', []))),
                ('USE_ROLLING_SUMMARY_KEYS', '1' if self.config.get('use_rolling_summary_keys', False) else '0'),
                ('ROLLING_SUMMARY_API_KEYS', _json.dumps(self.config.get('rolling_summary_keys', []))),
                ('USE_TRUNCATION_RETRY_KEYS', '1' if self.config.get('use_truncation_retry_keys', False) else '0'),
                ('TRUNCATION_RETRY_API_KEYS', _json.dumps(self.config.get('truncation_retry_keys', []))),
                ('IMAGE_CHUNK_OVERLAP_PERCENT', str(getattr(self, 'image_chunk_overlap_var', '3'))),
                ('IMAGE_CHUNK_MIN_OVERLAP_PIXELS', str(getattr(self, 'image_chunk_min_overlap_pixels_var', '80'))),
                ('IMAGE_SMART_CHUNKING', '1' if getattr(self, 'image_smart_chunking_var', True) else '0'),
                ('VISION_OCR_FUZZY_CHUNK_DEDUPE', '1' if getattr(self, 'vision_ocr_fuzzy_chunk_dedupe_var', False) else '0'),

                # Metadata and batch header settings
                ('TRANSLATE_METADATA_FIELDS', _json.dumps(getattr(self, 'translate_metadata_fields', {}))),
                ('METADATA_TRANSLATION_MODE', self.config.get('metadata_translation_mode', 'together')),
                ('BATCH_TRANSLATE_HEADERS', '1' if getattr(self, 'batch_translate_headers_var', True) else '0'),
                ('HEADERS_PER_BATCH', str(getattr(self, 'headers_per_batch_var', '-1'))),
                ('FAILED_TRANSLATION_RETRY_ATTEMPTS', str(getattr(self, 'failed_translation_retry_attempts_var', 3))),
                ('UPDATE_HTML_HEADERS', '1' if getattr(self, 'update_html_headers_var', True) else '0'),
                ('SAVE_HEADER_TRANSLATIONS', '1' if getattr(self, 'save_header_translations_var', True) else '0'),
                ('IGNORE_HEADER', '1' if getattr(self, 'ignore_header_var', False) else '0'),
                ('ALLOW_AI_MARKDOWN_HEADERS', '1' if getattr(self, 'allow_ai_markdown_headers_var', False) else '0'),
                ('USE_TITLE', '0' if getattr(self, 'skip_title_tag_translation_var', False) else '1'),

                # Extraction mode
                ('TEXT_EXTRACTION_METHOD', env_text_extraction_method),
                ('FILE_FILTERING_LEVEL', getattr(self, 'file_filtering_level_var', 'smart') if hasattr(self, 'file_filtering_level_var') else 'smart'),
                ('EXTRACTION_MODE', env_extraction_mode),
                ('ENHANCED_FILTERING', env_enhanced_filtering),
                ('USE_HTML2TEXT', '1' if output_mode == 'vision' or env_text_extraction_method in ('enhanced', 'html2text', 'markdown') or env_extraction_mode == 'enhanced' else '0'),

                # Anti-duplicate
                ('ENABLE_ANTI_DUPLICATE', '1' if getattr(self, 'enable_anti_duplicate_var', False) else '0'),
                ('TOP_P', str(getattr(self, 'top_p_var', '1.0'))),
                ('MIN_P', str(getattr(self, 'min_p_var', '0.0'))),
                ('BYPASS_MIN_P_ALLOWLIST', '1' if getattr(self, 'bypass_min_p_allowlist_var', False) else '0'),
                ('TOP_K', str(getattr(self, 'top_k_var', '0'))),
                ('FREQUENCY_PENALTY', str(getattr(self, 'frequency_penalty_var', '0.0'))),
                ('PRESENCE_PENALTY', str(getattr(self, 'presence_penalty_var', '0.0'))),
                ('REPETITION_PENALTY', str(getattr(self, 'repetition_penalty_var', '1.0'))),
                ('CANDIDATE_COUNT', str(getattr(self, 'candidate_count_var', '1'))),
                ('CUSTOM_STOP_SEQUENCES', getattr(self, 'custom_stop_sequences_var', '')),
                ('LOGIT_BIAS_ENABLED', '1' if getattr(self, 'logit_bias_enabled_var', False) else '0'),
                ('LOGIT_BIAS_STRENGTH', str(getattr(self, 'logit_bias_strength_var', '-0.5'))),
                ('BIAS_COMMON_WORDS', '1' if getattr(self, 'bias_common_words_var', False) else '0'),
                ('BIAS_REPETITIVE_PHRASES', '1' if getattr(self, 'bias_repetitive_phrases_var', False) else '0'),

                # Azure API version
                ('AZURE_API_VERSION', self.config.get('azure_api_version', '2025-01-01-preview')),
            ]

            env_mappings.extend(extra_env_mappings)
            
            initialized_count = 0
            for env_key, env_value in env_mappings:
                try:
                    old_value = os.environ.get(env_key, '<NOT SET>')
                    env_text = str(env_value) if env_value is not None else ''
                    if env_key == 'IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT':
                        import large_env
                        large_env.set_env(env_key, env_text)
                        new_value = large_env.get_env(env_key, '') or ''
                    else:
                        os.environ[env_key] = env_text
                        new_value = os.environ[env_key]
                    
                    if old_value != new_value and debug_mode:
                        self.append_log(f"🔍 [INIT] ENV {env_key}: '{old_value}' → '{new_value[:50]}{'...' if len(str(new_value)) > 50 else ''}'")
                    
                    initialized_count += 1
                except Exception as e:
                    if debug_mode:
                        self.append_log(f"❌ [INIT] Failed to initialize {env_key}: {e}")
            
            # JSON environment variables
            try:
                # Prefer in-memory types, then config, then sensible defaults
                custom_entry_types = getattr(self, 'custom_entry_types', None)
                if not custom_entry_types:
                    custom_entry_types = self.config.get('custom_entry_types')
                if not custom_entry_types:
                    custom_entry_types = {
                        'character': {'enabled': True, 'has_gender': True},
                        'term': {'enabled': True, 'has_gender': False},
                        'surnames': {'enabled': True, 'has_gender': False},
                        'titles': {'enabled': True, 'has_gender': True},
                        'locations': {'enabled': True, 'has_gender': False},
                        'nicknames': {'enabled': True, 'has_gender': True}
                    }
                custom_types_json = json.dumps(custom_entry_types)
                os.environ['GLOSSARY_CUSTOM_ENTRY_TYPES'] = custom_types_json
                if debug_mode:
                    self.append_log(f"🔍 [INIT] ENV GLOSSARY_CUSTOM_ENTRY_TYPES: {len(custom_types_json)} chars")
                initialized_count += 1
            except Exception as e:
                if debug_mode:
                    self.append_log(f"❌ [INIT] Failed to initialize GLOSSARY_CUSTOM_ENTRY_TYPES: {e}")
            
            try:
                # Always write GLOSSARY_CUSTOM_FIELDS so downstream consumers
                # (extract_glossary_from_epub._apply_description_rule_placeholders,
                # GlossaryManager, etc.) see a deterministic value and never fall
                # back to a stale env var from a previous run. An empty list is
                # serialized as "[]" — still a valid JSON array that decodes to
                # []. The prompt placeholder logic treats [] the same as "no
                # description field", which is the correct semantic.
                custom_glossary_fields = self.config.get('custom_glossary_fields', [])
                custom_fields_json = json.dumps(custom_glossary_fields)
                os.environ['GLOSSARY_CUSTOM_FIELDS'] = custom_fields_json
                if debug_mode:
                    self.append_log(f"🔍 [INIT] ENV GLOSSARY_CUSTOM_FIELDS: {len(custom_fields_json)} chars")
                initialized_count += 1
            except Exception as e:
                if debug_mode:
                    self.append_log(f"❌ [INIT] Failed to initialize GLOSSARY_CUSTOM_FIELDS: {e}")
                
            if debug_mode:
                self.append_log(f"✅ [INIT] Successfully initialized {initialized_count} environment variables")
            
            # Verify initialization (optional - don't fail if debug method doesn't exist)
            try:
                return self.debug_environment_variables(show_all=False)
            except AttributeError:
                # Method doesn't exist (e.g., in test mocks), return True since variables were set
                if debug_mode:
                    self.append_log("✅ [INIT] Environment variables initialized successfully (debug verification skipped)")
                return True
            
        except Exception as e:
            if debug_mode:
                self.append_log(f"❌ [INIT] Environment variable initialization failed: {e}")
                import traceback
                self.append_log(f"❌ [INIT] Traceback: {traceback.format_exc()}")
            return False

    # ---- split-outs: the desktop job runners call these at the original position ----
    def _glossary_extraction_paths(self, file_path):
        """Glossary output paths for one source (verbatim from _extract_glossary_from_text_file).

        Creates the shared Glossary folder and migrates legacy glossary files, exactly as the
        desktop does before building the glossary environment.
        """
        # Determine output directory
        epub_base = os.path.splitext(os.path.basename(file_path))[0]
        override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')
        save_glossary_in_output = bool(self.config.get('save_glossary_in_output', False))

        if override_dir:
            shared_glossary_dir = os.path.join(override_dir, "Glossary")
        else:
            shared_glossary_dir = "Glossary"
        # On macOS .app bundles, cwd can be '/' (read-only root).
        # Only on macOS — on Windows this changes the output dir and breaks glossary progress tracking.
        if sys.platform == 'darwin' and not os.path.isabs(shared_glossary_dir):
            shared_glossary_dir = os.path.join(os.path.dirname(os.path.abspath(file_path)), shared_glossary_dir)
        os.makedirs(shared_glossary_dir, exist_ok=True)
        try:
            from glossary_paths import get_book_glossary_path, migrate_all_legacy_glossary_files
            migrate_all_legacy_glossary_files(shared_glossary_dir, logger=self.append_log)
            output_path = get_book_glossary_path(
                shared_glossary_dir, epub_base, f"{epub_base}_glossary.json"
            )
        except Exception:
            glossary_dir = os.path.join(shared_glossary_dir, epub_base)
            os.makedirs(glossary_dir, exist_ok=True)
            output_path = os.path.join(glossary_dir, f"{epub_base}_glossary.json")
        output_side_backup_dir = self._output_side_glossary_backup_dir_for_source(file_path)
        return shared_glossary_dir, output_path, output_side_backup_dir, save_glossary_in_output

    def _build_glossary_extraction_env(self, file_path, api_key, model, *, shared_glossary_dir,
                                       save_glossary_in_output, output_side_backup_dir,
                                       force_balanced_request_merging=False):
        """env_updates for extract_glossary_from_epub (verbatim from _extract_glossary_from_text_file).

        Returns ``(env_updates, resolved_glossary_tokens, glossary_token_cfg, cjk_script_filter_enabled)``;
        the caller applies env_updates to os.environ exactly where the desktop did.
        """
        # Set up environment variables
        # Resolve glossary-specific output token limit; fall back to global when -1
        glossary_token_cfg = self.config.get('glossary_max_output_tokens', -1)
        if str(glossary_token_cfg) == '-1':
            resolved_glossary_tokens = int(self.max_output_tokens)
        else:
            resolved_glossary_tokens = int(glossary_token_cfg)

        glossary_batching_mode = self._glossary_batching_mode_for_env()
        auto_glossary_mode = self._current_auto_glossary_mode()
        (
            glossary_request_merging_enabled,
            glossary_request_merge_count,
            glossary_enable_chapter_split,
        ) = self._current_glossary_request_env(
            force_balanced_request_merging=force_balanced_request_merging
        )
        cjk_script_filter_enabled = self._current_glossary_cjk_script_filter_enabled()
        glossary_target_language = (
            self.config.get('glossary_target_language')
            or self.config.get('output_language')
            or getattr(self, 'lang_var', 'English')
            or 'English'
        )

        env_updates = {
            'GLOSSARY_TEMPERATURE': str(self.config.get('manual_glossary_temperature', 0.1)),
            'DISABLE_TEMPERATURE': '1' if self.disable_temperature_var else '0',
            'GLOSSARY_CONTEXT_LIMIT': str(self.config.get('manual_context_limit', 2)),
            'MODEL': self.model_var,
            'OPENAI_API_KEY': api_key,
            'OPENAI_OR_Gemini_API_KEY': api_key,
            'API_KEY': api_key,
            'MAX_OUTPUT_TOKENS': str(resolved_glossary_tokens),
            'GLOSSARY_MAX_OUTPUT_TOKENS': str(resolved_glossary_tokens),
            'BATCH_TRANSLATION': "1" if self.batch_translation_var else "0",
            'BATCH_SIZE': str(self.batch_size_var),
            'BATCHING_MODE': glossary_batching_mode,
            'BATCH_GROUP_SIZE': str(getattr(self, 'batch_group_size_var', '3')),
            'CONSERVATIVE_BATCHING': "1" if glossary_batching_mode == 'conservative' else "0",
            'GLOSSARY_SYSTEM_PROMPT': (
                self._parallel_epub_system_prompt_for_file(file_path)
                or self.manual_glossary_prompt
            ),
            'CHAPTER_RANGE': self.chapter_range_entry.text().strip(),
            'USE_SPINE_ORDER': '1' if (hasattr(self, 'use_spine_order_checkbox') and self.use_spine_order_checkbox.isChecked()) else '0',
            'GLOSSARY_DISABLE_HONORIFICS_FILTER': '1' if self.config.get('glossary_disable_honorifics_filter', False) else '0',
            'GLOSSARY_HISTORY_ROLLING': "1",
            'DISABLE_GEMINI_SAFETY': str(self.config.get('disable_gemini_safety', False)).lower(),
            'GEMINI_SAFETY_THRESHOLD': str(self.config.get('gemini_safety_threshold', 'BLOCK_NONE')),
            'OPENROUTER_USE_HTTP_ONLY': '1' if self.openrouter_http_only_var else '0',
            'GLOSSARY_DUPLICATE_KEY_MODE': 'skip',  # Always use skip mode for new format
            'SEND_INTERVAL_SECONDS': str(self.delay_entry.text()),
            'API_QUEUE_SIZE': self.api_queue_entry.text().strip() or '4',
            'THREAD_SUBMISSION_DELAY_SECONDS': self.thread_delay_entry.text().strip() or '0.0001',
            'CONTEXTUAL': self._glossary_contextual_env_value(),
            'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),
            'GLOSSARY_TARGET_LANGUAGE': str(glossary_target_language),

            # Glossary anti-duplicate parameters (separate from translation)
            'GLOSSARY_ENABLE_ANTI_DUPLICATE': '1' if self.config.get('glossary_enable_anti_duplicate', False) else '0',
            'GLOSSARY_TOP_P': str(self.config.get('glossary_top_p', 1.0)),
            'GLOSSARY_MIN_P': str(self.config.get('glossary_min_p', 0.0)),
            'GLOSSARY_BYPASS_MIN_P_ALLOWLIST': '1' if self.config.get('glossary_bypass_min_p_allowlist', False) else '0',
            'GLOSSARY_TOP_K': str(self.config.get('glossary_top_k', 0)),
            'GLOSSARY_FREQUENCY_PENALTY': str(self.config.get('glossary_frequency_penalty', 0.0)),
            'GLOSSARY_PRESENCE_PENALTY': str(self.config.get('glossary_presence_penalty', 0.0)),
            'GLOSSARY_REPETITION_PENALTY': str(self.config.get('glossary_repetition_penalty', 1.0)),
            'GLOSSARY_CANDIDATE_COUNT': str(self.config.get('glossary_candidate_count', 1)),
            'GLOSSARY_CUSTOM_STOP_SEQUENCES': str(self.config.get('glossary_custom_stop_sequences', '')),
            'GLOSSARY_LOGIT_BIAS_ENABLED': '1' if self.config.get('glossary_logit_bias_enabled', False) else '0',
            'GLOSSARY_LOGIT_BIAS_STRENGTH': str(self.config.get('glossary_logit_bias_strength', -0.5)),
            'GLOSSARY_BIAS_COMMON_WORDS': '1' if self.config.get('glossary_bias_common_words', False) else '0',
            'GLOSSARY_BIAS_REPETITIVE_PHRASES': '1' if self.config.get('glossary_bias_repetitive_phrases', False) else '0',

            # NEW GLOSSARY ADDITIONS
            'GLOSSARY_MIN_FREQUENCY': str(self.glossary_min_frequency_var),
            'GLOSSARY_MAX_NAMES': str(self.glossary_max_names_var),
            'GLOSSARY_MAX_TITLES': str(self.glossary_max_titles_var),
            'CONTEXT_WINDOW_SIZE': str(self.context_window_size_var),
            'GLOSSARY_SHARED_DIR': shared_glossary_dir,
            'SAVE_GLOSSARY_IN_OUTPUT': '1' if save_glossary_in_output else '0',
            'GLOSSARY_OUTPUT_BACKUP_DIR': output_side_backup_dir if save_glossary_in_output else '',
            'VISION_OCR_SOURCE_PREPASS': str(self.config.get('vision_ocr_source_prepass', 'auto') or 'auto'),
            'ENABLE_AUTO_GLOSSARY': "1" if auto_glossary_mode == 'minimal' else "0",
            'AUTO_GLOSSARY_MODE': auto_glossary_mode,
            'SINGLE_PASS_GLOSSARY_MODE': '1' if auto_glossary_mode == 'single_pass' else '',
            'SINGLE_PASS_GLOSSARY_HEADER_PROMPT': self.config.get('single_pass_glossary_header_prompt', ''),
            'APPEND_GLOSSARY': "0" if auto_glossary_mode == 'no_glossary' else ("1" if self.append_glossary_var else "0"),
            'GLOSSARY_STRIP_HONORIFICS': '1' if hasattr(self, 'strip_honorifics_var') and self.strip_honorifics_var else '1',
            'AUTO_GLOSSARY_PROMPT': getattr(self, 'unified_auto_glosary_prompt3', ''),
            'APPEND_GLOSSARY_PROMPT': getattr(self, 'append_glossary_prompt', '- Follow this reference glossary for consistent translation (Do not output any raw entries):\n'),
            'GLOSSARY_TRANSLATION_PROMPT': getattr(self, 'glossary_translation_prompt', ''),
            'GLOSSARY_CUSTOM_ENTRY_TYPES': json.dumps(getattr(self, 'custom_entry_types', {})),
            'GLOSSARY_CUSTOM_FIELDS': json.dumps(getattr(self, 'custom_glossary_fields', [])),
            'GLOSSARY_FUZZY_THRESHOLD': str(self.config.get('glossary_fuzzy_threshold', 0.90)),
            'GLOSSARY_ENTRY_TYPE_FILTER_MODE': self.config.get('glossary_entry_type_filter_mode', 'Loose'),
            'MANUAL_GLOSSARY': self.manual_glossary_path if hasattr(self, 'manual_glossary_path') and self.manual_glossary_path else '',
            'GLOSSARY_FORMAT_INSTRUCTIONS': self.glossary_format_instructions if hasattr(self, 'glossary_format_instructions') else '',
            'GLOSSARY_MAX_SENTENCES': str(self.config.get('glossary_max_sentences', 200)),
            'GLOSSARY_MAX_TEXT_SIZE': str(self.config.get('glossary_max_text_size', 0)),
            'GLOSSARY_FILTER_MODE': self.config.get('glossary_filter_mode', 'all'),
            'COMPRESSION_FACTOR': str(getattr(self, 'compression_factor_var', self.config.get('compression_factor', 1.0))),
            'GLOSSARY_INCLUDE_ALL_CHARACTERS': '1' if getattr(self, 'glossary_include_all_characters_var', False) else '0',
            'GLOSSARY_SKIP_IDENTICAL_ENTRIES': '1' if getattr(self, 'glossary_skip_identical_entries_var', True) else '0',
            'GLOSSARY_CJK_SCRIPT_FILTER': '1' if cjk_script_filter_enabled else '0',
            'GLOSSARY_SKIP_GENDER_TRACKING': '1' if getattr(self, 'glossary_skip_gender_tracking_var', False) else '0',
            'GLOSSARY_GENDER_NOISE_THRESHOLD': str(getattr(self, 'glossary_gender_noise_threshold_var', self.config.get('glossary_gender_noise_threshold', 10))),
            'GLOSSARY_GENDER_TRACKING_BIAS': str(getattr(self, 'glossary_gender_tracking_bias_var', self.config.get('glossary_gender_tracking_bias', 'none'))),
            'GLOSSARY_PARTIAL_RATIO_GENDER_ONLY': '1' if getattr(self, 'glossary_partial_ratio_gender_only_var', False) else '0',
            'GLOSSARY_ALIAS_AWARE_NAME_MATCHING': '1' if getattr(self, 'glossary_alias_aware_name_matching_var', False) else '0',
            'GLOSSARY_ALIAS_AWARE_GENDER_ONLY': '1' if getattr(self, 'glossary_alias_aware_gender_only_var', True) else '0',
            'USE_MAIN_KEY_FALLBACK': '1' if self.config.get('use_main_key_fallback', True) else '0',
            'USE_FALLBACK_KEYS': '1' if getattr(self, 'use_fallback_keys_var', False) else '0',
            'FALLBACK_KEY_SHUFFLE': '1' if self.config.get('fallback_key_shuffle', False) else '0',
            'DISABLE_EMPTY_SAFETY_HEURISTIC': '1' if getattr(self, 'disable_empty_safety_heuristic_var', True) else '0',
            'MISSING_FINISH_AS_PROHIBITED': '1' if getattr(self, 'unknown_finish_as_prohibited_var', False) else '0',
            'UNKNOWN_FINISH_AS_PROHIBITED': '1' if getattr(self, 'unknown_finish_as_prohibited_var', False) else '0',
            # Ensure fallback key pool is available to UnifiedClient (parity with translation path)
            'FALLBACK_KEYS': json.dumps(self.config.get('fallback_keys', [])),
            'USE_GLOSSARY_KEYS': '1' if getattr(self, 'use_glossary_keys_var', False) else '0',
            'GLOSSARY_API_KEYS': json.dumps(self.config.get('glossary_keys', [])),
            'USE_GLOSSARY_REFINEMENT_KEYS': '1' if getattr(self, 'use_glossary_refinement_keys_var', False) else '0',
            'GLOSSARY_REFINEMENT_API_KEYS': json.dumps(self.config.get('glossary_refinement_keys', [])),
            'USE_METADATA_KEYS': '1' if getattr(self, 'use_metadata_keys_var', False) else '0',
            'METADATA_API_KEYS': json.dumps(self.config.get('metadata_keys', [])),

            # Glossary-specific overrides (with fallback to global settings)
            'GLOSSARY_REQUEST_MERGING_ENABLED': glossary_request_merging_enabled,
            'GLOSSARY_REQUEST_MERGE_COUNT': glossary_request_merge_count,
            'GLOSSARY_COMPRESSION_FACTOR': str(self.config.get('glossary_compression_factor', getattr(self, 'compression_factor_var', 1.0))),
            'GLOSSARY_REFINEMENT_COMPRESSION_FACTOR': str(getattr(self, 'compression_factor_var', self.config.get('compression_factor', 1.0))),
            'GLOSSARY_OUTPUT_LEGACY_JSON': '1' if getattr(self, 'glossary_output_legacy_json_var', False) else '0',
            'GLOSSARY_ENABLE_CHAPTER_SPLIT': glossary_enable_chapter_split,
            'GLOSSARY_SKIP_TITLE_HEADER_ONLY': self._glossary_skip_title_header_only_env_value(),
            'GLOSSARY_ADD_MINIMAL_PASS': self._glossary_add_minimal_pass_env_value(),
            'GLOSSARY_MATCH_ENGINE': self._glossary_match_engine_env_value(),
            **self._strict_matching_env_dict(),
            **self._unified_glossary_env_dict(),
            'GLOSSARY_NEVER_CONSIDER_IN_BETWEEN_FILES_AS_SPECIAL': '1' if getattr(self, 'never_consider_in_between_files_as_special_var', self.config.get('never_consider_in_between_files_as_special', True)) else '0',
            # Optional assistant prefill prompt
            'ASSISTANT_PROMPT': getattr(self, 'assistant_prompt', '') or '',
            # Subprocess PDF extraction to prevent GUI lag
            'USE_ASYNC_CHAPTER_EXTRACTION': '1',
            'PDF_USE_TOC_SECTIONS': '1' if getattr(self, 'pdf_use_toc_sections_var', True) else '0',
            'PDF_EXTRACTION_WORKERS': str(getattr(self, 'pdf_extraction_workers_var', 'auto') or 'auto'),
            'PDF_PARAGRAPH_ALIGNMENT': str(getattr(self, 'pdf_paragraph_alignment_var', 'source') or 'source'),
            'PDF_HEADER_ALIGNMENT': str(getattr(self, 'pdf_header_alignment_var', 'source') or 'source'),
            'PDF_PARAGRAPH_JUSTIFICATION': str(getattr(self, 'pdf_paragraph_justification_var', 'source') or 'source'),
            'PDF_RTL_PARAGRAPH_LAYOUT': '1' if getattr(self, 'pdf_rtl_paragraph_layout_var', False) else '0',
            # Custom API endpoints (must be propagated so UnifiedClient routes
            # Gemini / OpenAI / Anthropic requests through the user's custom endpoint
            # during glossary extraction, especially when using glossary keys pool).
            'USE_GEMINI_OPENAI_ENDPOINT': '1' if getattr(self, 'use_gemini_openai_endpoint_var', False) else '0',
            'GEMINI_OPENAI_ENDPOINT': getattr(self, 'gemini_openai_endpoint_var', '') or 'generativelanguage.googleapis.com',
            'OVERRIDE_GEMMA_FOR_CUSTOM_ENDPOINT': '1' if getattr(self, 'override_gemma_for_custom_endpoint_var', True) else '0',
            'USE_CUSTOM_OPENAI_ENDPOINT': '1' if getattr(self, 'use_custom_openai_endpoint_var', False) else '0',
            'OPENAI_CUSTOM_BASE_URL': getattr(self, 'openai_base_url_var', '') or '',
            'USE_CUSTOM_IMAGE_EDIT_ENDPOINT': '1' if getattr(self, 'use_custom_image_edit_endpoint_var', False) else '0',
            'CUSTOM_IMAGE_EDIT_BASE_URL': (getattr(self, 'custom_image_edit_endpoint_var', '') or '') if getattr(self, 'use_custom_image_edit_endpoint_var', False) else '',
            'OPENAI_IMAGE_EDIT_BASE_URL': (getattr(self, 'custom_image_edit_endpoint_var', '') or '') if getattr(self, 'use_custom_image_edit_endpoint_var', False) else '',
            'CUSTOM_OPENAI_PREFIX_ROUTES': self._custom_prefix_routes_env_json(),
            'OLLAMA_SETTINGS_JSON': self._ollama_settings_env_json(),
            'AUTHZA_USE_GENERAL_API': '1' if bool(getattr(self, 'authza_use_general_api_var', self.config.get('authza_use_general_api', False))) else '0',
            'OPENAI_TTS_ENDPOINT': getattr(self, 'openai_tts_endpoint_var', '') or (getattr(self, 'openai_base_url_var', '') if str(getattr(self, 'openai_base_url_var', '')).rstrip('/').endswith('/audio/speech') else ''),
            'GROQ_API_URL': getattr(self, 'groq_base_url_var', '') or '',
            'FIREWORKS_API_URL': getattr(self, 'fireworks_base_url_var', '') or '',
            'FORCE_NATIVE_ANTHROPIC': '1' if getattr(self, 'force_native_anthropic_var', False) else '0',
            'ANTHROPIC_BASE_URL': getattr(self, 'anthropic_base_url_var', '') or '',
        }

        # Add project ID for Vertex AI
        if '@' in model or model.startswith('vertex/'):
            google_creds = self.config.get('google_cloud_credentials')
            if google_creds and os.path.exists(google_creds):
                try:
                    with open(google_creds, 'r') as f:
                        creds_data = json.load(f)
                        env_updates['GOOGLE_CLOUD_PROJECT'] = creds_data.get('project_id', '')
                        # Use the user's configured location, not hardcoded us-central1
                        env_updates['VERTEX_AI_LOCATION'] = self.vertex_location_var if hasattr(self, 'vertex_location_var') else 'global'
                except:
                    pass

        if self.custom_glossary_fields:
            env_updates['GLOSSARY_CUSTOM_FIELDS'] = json.dumps(self.custom_glossary_fields)
        return env_updates, resolved_glossary_tokens, glossary_token_cfg, cjk_script_filter_enabled

    def _build_epub_compile_env(self, folder):
        """Export the EPUB-compile environment into os.environ (verbatim from run_epub_converter_direct)."""
        # Set environment variables for EPUB converter
        os.environ['DISABLE_EPUB_GALLERY'] = "1" if self.disable_epub_gallery_var else "0"
        os.environ['SKIP_NON_SPINE_SPECIAL_FILES'] = "1" if getattr(self, 'skip_non_spine_special_files_var', False) else "0"
        os.environ['SKIP_UNREFERENCED_EPUB_IMAGES'] = "1" if getattr(self, 'skip_unreferenced_epub_images_var', False) else "0"
        os.environ['DISABLE_AUTOMATIC_COVER_CREATION'] = "1" if getattr(self, 'disable_automatic_cover_creation_var', True) else "0"
        os.environ['TRANSLATE_SPECIAL_FILES'] = "1" if getattr(self, 'translate_special_files_var', False) else "0"
        os.environ['REMOVE_DUPLICATE_H1_P'] = "1" if getattr(
            self,
            'remove_duplicate_h1_p_var',
            self.config.get('remove_duplicate_h1_p', False),
        ) else "0"
        os.environ['FIX_STRAY_P_GT_EPUB'] = "1" if getattr(
            self,
            'fix_stray_p_gt_epub_var',
            self.config.get('fix_stray_p_gt_epub', False),
        ) else "0"

        # If user selected a CSS override file, pass it to EPUB converter
        css_override_path = getattr(self, 'epub_css_override_path_var', self.config.get('epub_css_override_path', ''))
        if css_override_path:
            os.environ['EPUB_CSS_OVERRIDE_PATH'] = css_override_path
        else:
            os.environ.pop('EPUB_CSS_OVERRIDE_PATH', None)

        # Resolve the source EPUB through the library's validating
        # resolver: library_origins.txt → library_raw_inputs.txt →
        # Library/Raw → source_epub.txt (last resort). Candidates are
        # content-validated against translation_progress.json, so a
        # sidecar poisoned by a multi-EPUB run is invalidated instead
        # of ordering chapters from the wrong book.
        source_epub_path = ''
        try:
            from library_core import _find_raw_source_for_folder  # GUI-free (epub_library re-exports it)
            source_epub_path = _find_raw_source_for_folder(folder) or ''
        except Exception as e:
            self.append_log(f"⚠️ Could not resolve source EPUB reference: {e}")
            source_epub_path = ''
        if source_epub_path and os.path.exists(source_epub_path):
            os.environ['EPUB_PATH'] = source_epub_path
            self.append_log(f"✅ Using source EPUB for proper chapter ordering: {os.path.basename(source_epub_path)}")
        else:
            self.append_log("ℹ️ No valid source EPUB reference found - using filename-based ordering")

        # Set API credentials and model
        api_key = self.api_key_entry.text()
        if api_key:
            os.environ['API_KEY'] = api_key
            os.environ['OPENAI_API_KEY'] = api_key
            os.environ['OPENAI_OR_Gemini_API_KEY'] = api_key

        model = self.model_var
        if model:
            os.environ['MODEL'] = model

        # This path creates a fresh API client from environment variables,
        # so refresh the metadata-key pool from the current GUI config.
        os.environ['USE_METADATA_KEYS'] = (
            '1' if self.config.get('use_metadata_keys', False) else '0'
        )
        os.environ['METADATA_API_KEYS'] = json.dumps(
            self.config.get('metadata_keys', []) or []
        )

        # Set translation parameters from GUI
        os.environ['TRANSLATION_TEMPERATURE'] = str(self.trans_temp.text())
        os.environ['DISABLE_TEMPERATURE'] = '1' if self.disable_temperature_var else '0'
        os.environ['MAX_OUTPUT_TOKENS'] = str(self.max_output_tokens)
        os.environ['OUTPUT_LANGUAGE'] = self.config.get('output_language', 'English')

        # Set batch translation settings
        os.environ['BATCH_TRANSLATE_HEADERS'] = "1" if self.batch_translate_headers_var else "0"
        os.environ['HEADERS_PER_BATCH'] = str(self.headers_per_batch_var)
        os.environ['TOC_NCX_PER_BATCH'] = str(self.toc_ncx_per_batch_var)
        os.environ['FAILED_TRANSLATION_RETRY_ATTEMPTS'] = str(
            self.failed_translation_retry_attempts_var
        )
        os.environ['PARTIAL_B2_ENTRIES_PER_REQUEST'] = str(getattr(self, 'partial_b2_entries_per_request_var', self.config.get('partial_b2_entries_per_request', '-1')))
        os.environ['UPDATE_HTML_HEADERS'] = "1" if self.update_html_headers_var else "0"
        os.environ['SAVE_HEADER_TRANSLATIONS'] = "1" if self.save_header_translations_var else "0"
        os.environ['ALLOW_AI_MARKDOWN_HEADERS'] = "1" if getattr(self, 'allow_ai_markdown_headers_var', False) else "0"
        # Set Chapter Headers prompts from config - replace {target_lang} with output language
        output_lang = self.config.get('output_language', 'English')
        batch_header_system_prompt = self.config.get('batch_header_system_prompt',
            "You are a professional translator specializing in novel chapter titles. "
            "You must translate chapter titles to {target_lang}. "
            "Respond with only the translated JSON, nothing else. "
            "Maintain the original tone and style while making titles natural in the target language."
        ).replace('{target_lang}', output_lang)
        batch_header_prompt = self.config.get('batch_header_prompt',
            "Translate these chapter titles to {target_lang}.\n"
            "- For titles with parenthetical text, translate both the main title and the parenthetical content.\n"
            "- Translate the meaning accurately - don't use overly dramatic words unless the original implies them.\n"
            "- Preserve the chapter number format exactly as shown.\n"
            "Return ONLY a JSON object with chapter numbers as keys.\n"
            "Format: {\"1\": \"translated title\", \"2\": \"translated title\"}"
        ).replace('{target_lang}', output_lang)
        os.environ['BATCH_HEADER_SYSTEM_PROMPT'] = batch_header_system_prompt
        os.environ['BATCH_HEADER_PROMPT'] = batch_header_prompt
        os.environ['BATCH_HEADER_PREPEND_NUMBER_PATTERN'] = str(
            self.config.get('batch_header_prepend_number_pattern', '') or '')

        # Header, TOC and metadata requests append the glossary from these.
        for env_key, env_value in self._glossary_env_mappings():
            os.environ[env_key] = str(env_value)
        os.environ['GLOSSARY_MATCH_ENGINE'] = self._glossary_match_engine_env_value()
        os.environ.update(self._strict_matching_env_dict())
        os.environ.update(self._unified_glossary_env_dict())

        # Set metadata translation settings
        os.environ['TRANSLATE_METADATA_FIELDS'] = json.dumps(self.translate_metadata_fields)
        os.environ['METADATA_TRANSLATION_MODE'] = self.config.get('metadata_translation_mode', 'together')
        print(f"[DEBUG] METADATA_FIELD_PROMPTS from env: {os.getenv('METADATA_FIELD_PROMPTS', 'NOT SET')[:100]}...")

        # Debug: Log what we're setting
        self.append_log(f"[DEBUG] Setting TRANSLATE_METADATA_FIELDS: {self.translate_metadata_fields}")
        self.append_log(f"[DEBUG] Enabled fields: {[k for k, v in self.translate_metadata_fields.items() if v]}")

        # Set book title translation settings
        os.environ['TRANSLATE_BOOK_TITLE'] = "1" if self.translate_book_title_var else "0"
        os.environ['SKIP_TXT_TITLE_TRANSLATION'] = "1" if getattr(self, 'skip_txt_title_translation_var', True) else "0"
        os.environ['SKIP_PDF_TITLE_TRANSLATION'] = "1" if getattr(self, 'skip_pdf_title_translation_var', False) else "0"
        skip_title_tag = bool(getattr(self, 'skip_title_tag_translation_var', False))
        os.environ['SKIP_TITLE_TAG_TRANSLATION'] = "1" if skip_title_tag else "0"
        os.environ['USE_TITLE'] = "0" if skip_title_tag else "1"
        # Replace {target_lang} variable in book title prompts with output language
        output_lang = self.config.get('output_language', 'English')
        self.append_log(f"[DEBUG] output_language from config: '{output_lang}'")
        self.append_log(f"[DEBUG] book_title_prompt before: '{self.book_title_prompt}'")
        book_title_prompt_formatted = self.book_title_prompt.replace('{target_lang}', output_lang)
        book_title_system_prompt_formatted = self.config.get('book_title_system_prompt', '').replace('{target_lang}', output_lang)
        self.append_log(f"[DEBUG] book_title_prompt after: '{book_title_prompt_formatted}'")
        self.append_log(f"[DEBUG] book_title_system_prompt after: '{book_title_system_prompt_formatted}'")
        os.environ['BOOK_TITLE_PROMPT'] = book_title_prompt_formatted
        os.environ['BOOK_TITLE_SYSTEM_PROMPT'] = book_title_system_prompt_formatted

        # Set metadata system prompt - replace {target_lang} with output language
        os.environ['METADATA_SYSTEM_PROMPT'] = self.config.get('metadata_system_prompt', '').replace('{target_lang}', output_lang)

        # Set prompts
        import large_env
        large_env.set_env('SYSTEM_PROMPT', self.prompt_text.toPlainText().strip())
        large_env.set_env(
            'IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT',
            str(getattr(
                self,
                'image_only_title_tag_system_prompt',
                self.config.get(
                    'image_only_title_tag_system_prompt',
                    DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT,
                ),
            ) or DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT),
        )

        # PDF output settings
        os.environ['ENABLE_PDF_OUTPUT'] = '1' if getattr(self, 'enable_pdf_output_var', self.config.get('enable_pdf_output', False)) else '0'
        os.environ['PDF_GENERATE_TOC'] = '1' if self.config.get('pdf_generate_toc', False) else '0'
        os.environ['PDF_TOC_PAGE_NUMBERS'] = '1' if self.config.get('pdf_toc_page_numbers', True) else '0'
        os.environ['PDF_PAGE_NUMBERS'] = '1' if self.config.get('pdf_page_numbers', True) else '0'
        os.environ['PDF_PAGE_NUMBER_ALIGNMENT'] = self.config.get('pdf_page_number_alignment', 'center')
        os.environ['PDF_RENDER_BATCH_SIZE'] = str(self.config.get('pdf_render_batch_size', 50))
        os.environ['PDF_FAST_RENDERING'] = '1' if self.config.get('pdf_fast_rendering', True) else '0'
        os.environ['PDF_USE_RAPID_WORKSPACE_COMPILER'] = '1' if self.config.get('pdf_use_rapid_workspace_compiler', True) else '0'


        # EPUB structure settings
        _epub_layout = getattr(self, 'epub_layout_mode_var', self.config.get('epub_layout_mode', 'auto'))
        # Fallback if value is invalid
        if _epub_layout is None or _epub_layout not in ('auto', 'epub2', 'epub3'):
            _epub_layout = 'auto'
        os.environ['EPUB_LAYOUT_MODE'] = _epub_layout
        os.environ['LEGACY_EPUB_STRUCTURE'] = '1' if _epub_layout == 'epub2' else '0'
        os.environ['USE_TOC_NCX'] = '1' if getattr(self, 'use_toc_ncx_var', self.config.get('use_toc_ncx', True)) else '0'
        os.environ['TRANSLATE_TOC_NCX'] = os.environ['USE_TOC_NCX']  # Unified: translate always mirrors use
        os.environ['SKIP_DUPLICATE_TOC_TRANSLATION'] = '1' if getattr(self, 'skip_duplicate_toc_translation_var', self.config.get('skip_duplicate_toc_translation', False)) else '0'
        os.environ['DEDUPLICATE_TOC'] = '1' if getattr(self, 'deduplicate_toc_var', self.config.get('deduplicate_toc', False)) else '0'
        os.environ['FORCE_NCX_ONLY'] = '1' if getattr(self, 'force_ncx_only_var', True) else '0'
        os.environ['ATTACH_CSS_TO_CHAPTERS'] = '1' if getattr(self, 'attach_css_to_chapters_var', self.config.get('attach_css_to_chapters', False)) else '0'
        os.environ['EPUB_USE_HTML_METHOD'] = '1' if getattr(self, 'epub_use_html_method_var', self.config.get('epub_use_html_method', False)) else '0'
        os.environ['ENABLE_IMAGE_COMPRESSION'] = '1' if self.config.get('enable_image_compression', False) else '0'
        os.environ['IMAGE_COMPRESSION_QUALITY'] = str(self.config.get('image_compression_quality', 80))
        os.environ['EXCLUDE_COVER_COMPRESSION'] = '1' if self.config.get('exclude_cover_compression', True) else '0'
        os.environ['EXCLUDE_GIF_COMPRESSION'] = '1' if self.config.get('exclude_gif_compression', True) else '0'

    def _build_pdf_compile_env(self):
        """Export the PDF-compile header/TOC environment into os.environ (verbatim from run_pdf_converter_direct)."""
        # PDF compilation owns the same optional bookmark/header batch
        # phase as EPUB compilation. Export the live settings because the
        # Library can invoke this after a restart, when translation-run
        # environment variables have not been populated yet.
        os.environ['USE_TOC_NCX'] = '1' if getattr(
            self, 'use_toc_ncx_var', self.config.get('use_toc_ncx', True)
        ) else '0'
        os.environ['TRANSLATE_TOC_NCX'] = os.environ['USE_TOC_NCX']
        os.environ['BATCH_TRANSLATE_HEADERS'] = '1' if getattr(
            self,
            'batch_translate_headers_var',
            self.config.get('batch_translate_headers', True),
        ) else '0'
        os.environ['HEADERS_PER_BATCH'] = str(
            getattr(self, 'headers_per_batch_var', '-1')
        )
        os.environ['TOC_NCX_PER_BATCH'] = str(
            getattr(self, 'toc_ncx_per_batch_var', '-1')
        )
        os.environ['UPDATE_HTML_HEADERS'] = '1' if getattr(
            self, 'update_html_headers_var', True
        ) else '0'
        os.environ['SAVE_HEADER_TRANSLATIONS'] = '1' if getattr(
            self, 'save_header_translations_var', True
        ) else '0'
        os.environ['SKIP_DUPLICATE_TOC_TRANSLATION'] = '1' if getattr(
            self,
            'skip_duplicate_toc_translation_var',
            self.config.get('skip_duplicate_toc_translation', False),
        ) else '0'
        output_language = self.config.get('output_language', 'English')
        os.environ['OUTPUT_LANGUAGE'] = output_language
        os.environ['MAX_OUTPUT_TOKENS'] = str(self.max_output_tokens)
        os.environ['TRANSLATION_TEMPERATURE'] = str(self.trans_temp.text())
        os.environ['BATCH_HEADER_SYSTEM_PROMPT'] = str(
            self.config.get('batch_header_system_prompt', '') or ''
        ).replace('{target_lang}', output_language)
        os.environ['BATCH_HEADER_PROMPT'] = str(
            self.config.get('batch_header_prompt', '') or ''
        ).replace('{target_lang}', output_language)



# ---------------------------------------------------------------------------
# Convenience builders (mobile, previews, tests). They call the verbatim owner
# methods above; ``owner`` is a TranslatorGUI or a headless_owner.HeadlessOwner.
# ---------------------------------------------------------------------------

#: build_glossary_env result: the env_updates the desktop applies before calling
#: extract_glossary_from_epub.main, plus the output paths it passes on argv.
GlossaryEnv = namedtuple("GlossaryEnv", "env_updates output_path shared_glossary_dir resolved_tokens")


def build_translation_env(owner, input_path, api_key):
    """Translation env for one input file (``owner._get_environment_variables``).

    Exactly the desktop call: it returns the env dict without writing os.environ,
    but (like the desktop) configures the in-memory key pools and may migrate
    legacy glossary files in the shared Glossary folder.
    """
    return owner._get_environment_variables(input_path, api_key)


def build_glossary_env(owner, input_path, api_key, force_balanced=False):
    """Glossary-extraction env for one source (the desktop _extract_glossary_from_text_file steps).

    Creates the shared Glossary folder like the desktop does. The caller handles
    the Vertex AI credential/API-key resolution that precedes it on desktop.
    """
    (
        shared_glossary_dir,
        output_path,
        output_side_backup_dir,
        save_glossary_in_output,
    ) = owner._glossary_extraction_paths(input_path)
    model = str(getattr(owner, 'model_var', '') or '').strip()
    env_updates, resolved_tokens, _glossary_token_cfg, _cjk_filter = owner._build_glossary_extraction_env(
        input_path,
        api_key,
        model,
        shared_glossary_dir=shared_glossary_dir,
        save_glossary_in_output=save_glossary_in_output,
        output_side_backup_dir=output_side_backup_dir,
        force_balanced_request_merging=bool(force_balanced),
    )
    return GlossaryEnv(env_updates, output_path, shared_glossary_dir, resolved_tokens)


def _large_env_module():
    try:
        import large_env
    except Exception:
        return None
    return large_env if isinstance(getattr(large_env, "_store", None), dict) else None


def capture_env_delta(fn):
    """Run ``fn()`` and return the environment change it made, then restore os.environ.

    Returns ``{'set': {name: value}, 'unset': [name, ...]}``; values exported through
    ``large_env`` (Windows values over 32k characters) are included in ``set``.
    """
    large_env = _large_env_module()
    before_env = dict(os.environ)
    before_store = dict(large_env._store) if large_env is not None else {}
    try:
        fn()
        after_env = dict(os.environ)
        after_store = dict(large_env._store) if large_env is not None else {}
    finally:
        os.environ.clear()
        os.environ.update(before_env)
        if large_env is not None:
            large_env._store.clear()
            large_env._store.update(before_store)
    changed = {k: v for k, v in after_env.items() if before_env.get(k) != v}
    changed.update({k: v for k, v in after_store.items() if before_store.get(k) != v})
    removed = sorted(k for k in before_env if k not in after_env)
    return {"set": changed, "unset": removed}


def build_epub_compile_env(owner, folder):
    """EPUB-compile env delta (``owner._build_epub_compile_env``) without leaving it applied."""
    return capture_env_delta(lambda: owner._build_epub_compile_env(folder))


def build_pdf_compile_env(owner):
    """PDF-compile header/TOC env delta (``owner._build_pdf_compile_env``) without leaving it applied."""
    return capture_env_delta(owner._build_pdf_compile_env)


def build_startup_env(owner):
    """Startup env delta (``owner.initialize_environment_variables``) without leaving it applied."""
    return capture_env_delta(owner.initialize_environment_variables)
