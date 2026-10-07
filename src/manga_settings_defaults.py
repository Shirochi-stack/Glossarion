"""Canonical manga translator settings defaults (GUI-free).

Shared GUI-free core (Glossarion mobile rewrite, milestone U8).

* :func:`default_manga_settings` is ``MangaSettingsDialog.default_settings`` moved verbatim
  (manga_settings_dialog.py, ``__init__``): the nested ``config['manga_settings']`` defaults.
  The dialog now builds its ``default_settings`` from it, the mobile settings pages show it,
  and ``src/mobile/tools/schema_extract.py`` reads the ``return`` literal below as the
  ``manga_settings.*`` anchor.
* :func:`merge_manga_settings` is the dialog's ``_merge_settings`` (defaults deep-merged with
  the saved ``manga_settings``), without the dialog's quirk of sharing nested dicts with its
  ``default_settings`` (the merge always starts from a fresh default tree).
* :data:`MANGA_TOP_LEVEL_DEFAULTS`: the top-level ``manga_*`` / ``rapidocr_*`` /
  ``qwen2vl_*`` keys with the defaults the manga translator tab actually runs with
  (``MangaTranslationTab._load_rendering_settings`` and the OCR provider row). Some differ
  from the startup env defaults in ``settings_schema_data`` (``manga_bg_opacity`` 0 vs 130,
  ``manga_full_page_context`` True vs False, ``manga_free_text_only_bg_opacity`` False vs
  True, ``manga_shadow_color`` white vs [204, 128, 128]): tests/parity/DISCREPANCIES.md, "U8
  Manga run env, ..." item 2. tests/test_manga_env.py replays
  ``_load_rendering_settings`` on an empty config and checks every entry. The long prompt
  defaults are not repeated here; use ``manga_env.default_manga_ocr_prompt()`` /
  ``default_manga_glossary_prompt()`` / ``default_full_page_context_prompt()``.

* :func:`font_preset_updates` (preset, config=None): the config writes of the Rendering
  font-size preset buttons. The values are not repeated here: the call runs the desktop's
  ``_set_font_preset`` through ``manga_env`` (imported lazily, on call).

Rules: Python 3.10 compatible; stdlib only at import time; never import PySide6, translator_gui
or dpi_setup.
"""

import copy

__all__ = [
    "MANGA_SETTINGS_SECTIONS",
    "MANGA_TOP_LEVEL_DEFAULTS",
    "MANGA_PROMPT_KEYS",
    "default_manga_settings",
    "deep_merge",
    "font_preset_updates",
    "merge_manga_settings",
    "top_level_manga_default",
]


def default_manga_settings():
    """A fresh ``manga_settings`` default tree (``MangaSettingsDialog.default_settings``)."""
    return {
        'preprocessing': {
            'enabled': False,
            'auto_detect_quality': True,
            'contrast_threshold': 0.4,
            'sharpness_threshold': 0.3,
            'noise_threshold': 20,
            'enhancement_strength': 1.5,
            'denoise_strength': 10,
            'max_image_dimension': 2000,
            'max_image_pixels': 2000000,
            'chunk_height': 2000,
            'chunk_overlap': 100,
            # Inpainting tiling
            'inpaint_tiling_enabled': False,  # Off by default
            'inpaint_tile_size': 512,  # Default tile size
            'inpaint_tile_overlap': 64  # Overlap to avoid seams
        },
        'compression': {
            'enabled': False,
            'format': 'jpeg',
            'jpeg_quality': 85,
            'png_compress_level': 6,
            'webp_quality': 85
        },
        'ocr': {
            'language_hints': ['ja', 'ko', 'zh'],
            'confidence_threshold': 0.0,  # DEFAULT 0.0 (accept all, like comic-translate) to avoid missing text
            'cloud_ocr_confidence': 0.0,  # Explicit default for cloud OCR (Azure/Google)
            'min_region_size': 50,  # Minimum dimension for cloud OCR regions (0 = disabled)
            'merge_nearby_threshold': 20,
            'azure_merge_multiplier': 3.0,
            'text_detection_mode': 'document',
            'enable_rotation_correction': True,
            'bubble_detection_enabled': True,
            'roi_locality_enabled': False,
            'bubble_model_path': '',
            'bubble_confidence': 0.3,
            'detector_type': 'rtdetr_onnx',
            'rtdetr_onnx_variant': 'detector.onnx',
            'rtdetr_confidence': 0.3,
            'detect_empty_bubbles': True,
            'detect_text_bubbles': True,
            'detect_free_text': True,
            'rtdetr_model_url': '',
            'use_rtdetr_for_ocr_regions': True,  # On by default for best accuracy
            'enable_fallback_ocr': False,  # Disabled by default - fallback OCR for empty RT-DETR blocks
            'ocr_batch_enabled': True,
            'ocr_batch_size': 8,
            'ocr_max_concurrency': 2,
            'ocr_request_delay_ms': 100,  # Existing Google ROI OCR base delay (adds small jitter)
            'ocr_max_retries': 0,
            'manga_ocr_disable_thinking': True,
            # Toggles for RT-DETR behavior customization
            'skip_rtdetr_merging': False,    # Do not merge overlapping RT-DETR regions (manual mode behavior)
            'preserve_empty_blocks': False,  # Keep empty RT-DETR blocks even if OCR found no text
            # Azure settings removed - new API is synchronous, no polling/version settings needed
            'min_text_length': 0,
            'exclude_english_text': False,
            'english_exclude_threshold': 0.7,
            'english_exclude_min_chars': 4,
            'english_exclude_short_tokens': False
        },
        'advanced': {
            'format_detection': True,
            'webtoon_mode': 'auto',
            'debug_mode': False,
            'save_intermediate': False,
            'parallel_processing': True,
            'max_workers': 2,
            'parallel_panel_translation': False,
            'panel_max_workers': 2,
            'auto_cleanup_models': False,
            'unload_models_after_translation': False,
            'auto_convert_to_onnx': False,  # Disabled by default
            'auto_convert_to_onnx_background': True,
            'quantize_models': False,
            'onnx_quantize': False,
            'torch_precision': 'fp16',
            # HD strategy defaults (mirrors comic-translate)
            'hd_strategy': 'resize',                # 'original' | 'resize' | 'crop'
            'hd_strategy_resize_limit': 1536,       # long-edge cap for resize
            'hd_strategy_crop_margin': 16,          # pixels padding around cropped ROIs
            'hd_strategy_crop_trigger_size': 1024,  # only crop if long edge exceeds this
            # RAM cap defaults
            'ram_cap_enabled': False,
            'ram_cap_mb': 4096,
            'ram_cap_mode': 'soft',
            'ram_gate_timeout_sec': 15.0,
            'ram_min_floor_over_baseline_mb': 256
            },
        'inpainting': {
            'batch_size': 10,
            'enable_cache': True,
            'method': 'local',
            'local_method': 'anime'
        },
        'font_sizing': {
        'algorithm': 'smart',  # 'smart', 'conservative', 'aggressive'
        'prefer_larger': True,  # Prefer larger readable text
        'max_lines': 10,  # Maximum lines before forcing smaller
        'line_spacing': 1.3,  # Line height multiplier
        'bubble_size_factor': True  # Scale font based on bubble size
        },
        
        # Mask dilation settings with new iteration controls
        'mask_dilation': 0,
        'dilation_kernel_size': 5,  # Kernel size for dilation operations
        'use_all_iterations': True,  # Master control - use same for all by default
        'all_iterations': 2,  # Value when using same for all
        'text_bubble_dilation_iterations': 2,  # Text-filled speech bubbles
        'empty_bubble_dilation_iterations': 3,  # Empty speech bubbles
        'free_text_dilation_iterations': 0,  # Free text (0 for clean B&W)
        'bubble_dilation_iterations': 2,  # Legacy support
        'dilation_iterations': 2,  # Legacy support
        
        # Cloud inpainting settings
        'cloud_inpaint_model': 'ideogram-v2',
        'cloud_custom_version': '',
        'cloud_inpaint_prompt': 'clean background, smooth surface',
        'cloud_negative_prompt': 'text, writing, letters',
        'cloud_inference_steps': 20,
        'cloud_timeout': 60,
        'manual_edit': {
            'translate_prompt': 'output only the {language} translation of this text:',  # Prompt template with {language} placeholder
            'translate_target_language': 'English',  # Default language
            'manga_output_token_limit': -1,  # -1 or 0 = use main GUI's output token limit
            'translate_this_text_tokens': 2048,  # Token limit for "Translate This Text" context menu action
            'translate_this_text_disable_thinking': True
        }
    }


#: Nested sections of ``manga_settings`` (dict-valued defaults), in dialog order.
MANGA_SETTINGS_SECTIONS = tuple(k for k, v in default_manga_settings().items() if isinstance(v, dict))


def deep_merge(base, update):
    """``MangaSettingsDialog._merge_settings``' inner merge: *update* into *base* (in place)."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def merge_manga_settings(config):
    """The effective ``manga_settings``: defaults deep-merged with ``config['manga_settings']``.

    *config* is a config.json dict (or the ``manga_settings`` dict itself when it has none of
    the top-level config keys). The result is a new tree; *config* is not modified. Values are
    deep-copied, so editing the result never writes through into *config*.
    """
    if isinstance(config, dict) and 'manga_settings' in config:
        existing = config.get('manga_settings') or {}
    elif isinstance(config, dict) and any(key in config for key in MANGA_SETTINGS_SECTIONS):
        existing = config
    else:
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    return deep_merge(default_manga_settings(), copy.deepcopy(existing))


#: Top-level config keys whose defaults are prompts (see the module docstring).
MANGA_PROMPT_KEYS = ('manga_ocr_prompt', 'manga_glossary_prompt', 'manga_full_page_context_prompt')

#: Top-level manga keys -> the default the manga translator tab runs with.
MANGA_TOP_LEVEL_DEFAULTS = {
    # OCR provider row (_build_pyside6_interface: manga_ocr_provider or ocr_provider or 'custom-api')
    'manga_ocr_provider': 'custom-api',
    # _load_rendering_settings
    'manga_bg_opacity': 0,
    'manga_free_text_only_bg_opacity': False,
    'manga_bg_style': 'circle',
    'manga_bg_reduction': 1.0,
    'manga_font_size': 0,
    'manga_font_path': None,
    'manga_skip_inpainting': False,
    'manga_inpaint_quality': 'high',
    'manga_inpaint_dilation': 15,
    'manga_inpaint_passes': 2,
    'manga_disable_inpaint_performance_mode': False,
    'manga_font_size_mode': 'fixed',
    'manga_font_size_multiplier': 1.0,
    'manga_force_caps_lock': True,
    'manga_constrain_to_bubble': True,
    # manga_max_font_size: rendering.auto_max_size, else font_sizing.max_size, else 48
    'manga_max_font_size': 48,
    'manga_strict_text_wrapping': True,
    'manga_safe_area_enabled': False,
    'manga_safe_area_scale': 1.0,
    'manga_text_color': [102, 0, 0],
    'manga_shadow_enabled': True,
    'manga_shadow_color': [255, 255, 255],
    'manga_shadow_offset_x': 2,
    'manga_shadow_offset_y': 2,
    'manga_shadow_blur': 0,
    'manga_font_style': 'Default',
    'manga_full_page_context': True,
    'manga_glossary_enabled': False,
    'manga_custom_glossary_path': '',
    'manga_generated_glossary_path': '',
    'manga_glossary_auto_load_suppressed': False,
    'manga_glossary_auto_load_suppressed_root': '',
    'manga_split_first_level_subfolders': False,
    'manga_glossary_debug_ocr_text': False,
    'manga_visual_context_enabled': True,
    'qwen2vl_model_size': '1',
    'rapidocr_use_recognition': True,
    'rapidocr_language': 'auto',
    'rapidocr_detection_mode': 'document',
    'manga_custom_api_ocr_batch_enabled': True,
    'manga_custom_api_ocr_batch_size': 5,
    'manga_batch_image_requests_enabled': True,
    'manga_batch_image_requests_size': 5,
    'manga_create_cbz_at_end': True,
    'manga_auto_consolidate_images': True,
}


def font_preset_updates(preset, config=None):
    """``manga_env.font_preset_updates``: the config entries a font-size preset button writes
    (``{'manga_strict_text_wrapping': False, 'manga_settings.font_sizing.algorithm': ...}``)."""
    import manga_env

    return manga_env.font_preset_updates(preset, config)


def top_level_manga_default(key, default=None):
    """Default of a top-level manga key (a fresh copy for list values)."""
    if key in MANGA_TOP_LEVEL_DEFAULTS:
        return copy.deepcopy(MANGA_TOP_LEVEL_DEFAULTS[key])
    return default
