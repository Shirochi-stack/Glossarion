"""Key-pool runtime setup shared by desktop and mobile.

``apply_key_pools_to_runtime(config)`` is the verbatim key-pool section of
``TranslatorGUI._get_environment_variables`` (translator_gui.py, "Ensure
multi-key env toggles" + "Configure multi-key list in memory"), with
``self.config`` passed in as ``config``. It exports the pool toggle/list
environment variables and installs the in-memory pools on
``unified_api_client.UnifiedClient`` (avoids the Windows env-var size limit).

GUI-free; must stay importable on Python 3.10 without Qt.
"""

import json
import os


def export_key_pool_env(config):
    """Export the key-pool toggles and pool lists as environment variables."""
    # Ensure multi-key env toggles are set early for the main translation path as well
    try:
        if config.get('use_multi_api_keys', False):
            os.environ['USE_MULTI_KEYS'] = '1'
        else:
            os.environ['USE_MULTI_KEYS'] = '0'
        if config.get('use_fallback_keys', False):
            os.environ['USE_FALLBACK_KEYS'] = '1'
        else:
            os.environ['USE_FALLBACK_KEYS'] = '0'
        os.environ['FALLBACK_KEY_SHUFFLE'] = '1' if config.get('fallback_key_shuffle', False) else '0'
        if config.get('use_glossary_keys', False):
            os.environ['USE_GLOSSARY_KEYS'] = '1'
        else:
            os.environ['USE_GLOSSARY_KEYS'] = '0'
        os.environ['USE_GLOSSARY_REFINEMENT_KEYS'] = '1' if config.get('use_glossary_refinement_keys', False) else '0'
        os.environ['GLOSSARY_REFINEMENT_API_KEYS'] = json.dumps(config.get('glossary_refinement_keys', []))
        os.environ['USE_METADATA_KEYS'] = '1' if config.get('use_metadata_keys', False) else '0'
        os.environ['METADATA_API_KEYS'] = json.dumps(config.get('metadata_keys', []))
        if config.get('use_qa_scan_keys', False):
            os.environ['USE_VISION_KEYS'] = '1'
            os.environ['USE_QA_SCAN_KEYS'] = '1'
        else:
            os.environ['USE_VISION_KEYS'] = '0'
            os.environ['USE_QA_SCAN_KEYS'] = '0'
        os.environ['VISION_API_KEYS'] = json.dumps(config.get('qa_scan_keys', []))
        os.environ['QA_SCAN_API_KEYS'] = os.environ['VISION_API_KEYS']
        os.environ['USE_AI_TRUNCATION_DETECTION_KEYS'] = '1' if config.get('use_ai_truncation_detection_keys', False) else '0'
        os.environ['AI_TRUNCATION_DETECTION_API_KEYS'] = json.dumps(config.get('ai_truncation_detection_keys', []))
        os.environ['USE_ROLLING_SUMMARY_KEYS'] = '1' if config.get('use_rolling_summary_keys', False) else '0'
        os.environ['ROLLING_SUMMARY_API_KEYS'] = json.dumps(config.get('rolling_summary_keys', []))
        os.environ['USE_TRUNCATION_RETRY_KEYS'] = '1' if config.get('use_truncation_retry_keys', False) else '0'
        os.environ['TRUNCATION_RETRY_API_KEYS'] = json.dumps(config.get('truncation_retry_keys', []))
    except Exception:
        pass


def apply_key_pools_to_runtime(config, *, export_env=True):
    """Export the key-pool env (optional) and configure the in-memory pools."""
    if export_env:
        export_key_pool_env(config)

    # Configure multi-key list in memory (avoid Windows env var size limit for MULTI_API_KEYS)
    try:
        from unified_api_client import UnifiedClient
        if config.get('use_multi_api_keys', False) and config.get('multi_api_keys', []):
            UnifiedClient.set_in_memory_multi_keys(
                config.get('multi_api_keys', []),
                force_rotation=config.get('force_key_rotation', True),
                rotation_frequency=config.get('rotation_frequency', 1),
            )
        else:
            UnifiedClient.clear_in_memory_multi_keys()

        # Configure glossary key pool in memory (mirrors multi-key setup)
        if config.get('use_glossary_keys', False) and config.get('glossary_keys', []):
            UnifiedClient.set_in_memory_glossary_keys(
                config.get('glossary_keys', []),
                force_rotation=config.get('force_key_rotation', True),
                rotation_frequency=config.get('rotation_frequency', 1),
            )
        else:
            UnifiedClient.clear_in_memory_glossary_keys()
        if config.get('use_glossary_refinement_keys', False) and config.get('glossary_refinement_keys', []):
            UnifiedClient.set_in_memory_glossary_refinement_keys(
                config.get('glossary_refinement_keys', []),
                force_rotation=config.get('force_key_rotation', True),
                rotation_frequency=config.get('rotation_frequency', 1),
            )
        else:
            UnifiedClient.clear_in_memory_glossary_refinement_keys()

        # Configure Vision key pool in memory (mirrors glossary-key setup)
        if config.get('use_qa_scan_keys', False) and config.get('qa_scan_keys', []):
            UnifiedClient.set_in_memory_vision_keys(
                config.get('qa_scan_keys', []),
                force_rotation=config.get('force_key_rotation', True),
                rotation_frequency=config.get('rotation_frequency', 1),
            )
        else:
            UnifiedClient.clear_in_memory_vision_keys()

        # Configure rolling summary key pool in memory (used by context='summary').
        if config.get('use_rolling_summary_keys', False) and config.get('rolling_summary_keys', []):
            UnifiedClient.set_in_memory_rolling_summary_keys(
                config.get('rolling_summary_keys', []),
                force_rotation=config.get('force_key_rotation', True),
                rotation_frequency=config.get('rotation_frequency', 1),
            )
        else:
            UnifiedClient.clear_in_memory_rolling_summary_keys()

        # Configure truncation retry key pool in memory (used by RETRY_TRUNCATED attempts)
        if config.get('use_truncation_retry_keys', False) and config.get('truncation_retry_keys', []):
            UnifiedClient.set_in_memory_truncation_retry_keys(
                config.get('truncation_retry_keys', []),
                force_rotation=config.get('force_key_rotation', True),
                rotation_frequency=config.get('rotation_frequency', 1),
            )
        else:
            UnifiedClient.clear_in_memory_truncation_retry_keys()

        # Configure Image gen/edit key pool for image output and manga custom-image-edit requests.
        if config.get('use_inpainter_keys', False) and config.get('inpainter_keys', []):
            UnifiedClient.set_in_memory_inpainter_keys(
                config.get('inpainter_keys', []),
                force_rotation=config.get('force_key_rotation', True),
                rotation_frequency=config.get('rotation_frequency', 1),
            )
        else:
            UnifiedClient.clear_in_memory_inpainter_keys()

        # Configure Audio / TTS key pool for audio output mode requests.
        if config.get('use_tts_keys', False) and config.get('tts_keys', []):
            UnifiedClient.set_in_memory_tts_keys(
                config.get('tts_keys', []),
                force_rotation=config.get('force_key_rotation', True),
                rotation_frequency=config.get('rotation_frequency', 1),
            )
        else:
            UnifiedClient.clear_in_memory_tts_keys()
    except Exception:
        pass
