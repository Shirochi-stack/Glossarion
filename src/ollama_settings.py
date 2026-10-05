"""Pure settings helpers for the local ``ollamapull/`` model route.

Moved verbatim from ollama_settings_dialog.py so the run environment and the
mobile app can normalise, serialise and validate Ollama request settings
without importing the Qt dialog. ollama_settings_dialog re-imports every name.

GUI-free; must stay importable on Python 3.10 without Qt.
"""

from __future__ import annotations

import copy
import json
import math


OLLAMAPULL_PREFIX = "ollamapull/"


def is_ollamapull_route(value: str) -> bool:
    """Recognize the route while a user is still typing its model name."""
    value = str(value or "").strip().casefold()
    return value == "ollamapull" or value.startswith(OLLAMAPULL_PREFIX)


def ollamapull_model_name(value: str) -> str:
    """Return the native Ollama model name, or an empty string for other routes."""
    value = str(value or "").strip()
    if not value.casefold().startswith(OLLAMAPULL_PREFIX):
        return ""
    return value[len(OLLAMAPULL_PREFIX):].strip()


def normalize_ollama_settings(value) -> dict:
    """Preserve future settings while supplying the defaults used by the UI."""
    settings = copy.deepcopy(value) if isinstance(value, dict) else {}
    settings.setdefault("auto_update", True)
    if not isinstance(settings.get("models"), dict):
        settings["models"] = {}
    return settings


def ollama_settings_json(config: dict) -> str:
    """Serialize the shared settings for translation workers and key tests."""
    return json.dumps(
        normalize_ollama_settings((config or {}).get("ollama_settings")),
        ensure_ascii=False,
        separators=(",", ":"),
    )


# Ollama's published Modelfile parameters, plus native request options useful
# for runtime tuning. Unlisted future options remain available in Advanced.
OPTION_GROUPS = (
    ("Context and output", (
        ("num_ctx", "Context size", "int"),
        ("num_predict", "Maximum generated tokens", "int"),
        ("draft_num_predict", "Speculative decoding", "int"),
        ("num_keep", "Prompt tokens to retain", "int"),
        ("num_batch", "Prompt batch size", "int"),
    )),
    ("Sampling", (
        ("temperature", "Temperature", "float"),
        ("top_k", "Top K", "int"),
        ("top_p", "Top P", "float"),
        ("min_p", "Min P", "float"),
        ("typical_p", "Typical P", "float"),
        ("tfs_z", "Tail free sampling", "float"),
        ("repeat_last_n", "Repeat lookback", "int"),
        ("repeat_penalty", "Repeat penalty", "float"),
        ("presence_penalty", "Presence penalty", "float"),
        ("frequency_penalty", "Frequency penalty", "float"),
        ("penalize_newline", "Penalize newline", "bool"),
        ("mirostat", "Mirostat mode", "int"),
        ("mirostat_tau", "Mirostat target", "float"),
        ("mirostat_eta", "Mirostat learning rate", "float"),
        ("seed", "Random seed", "int"),
        ("stop", "Stop sequences (JSON array)", "array"),
    )),
    ("Runtime", (
        ("num_gpu", "GPU layers", "int"),
        ("main_gpu", "Primary GPU", "int"),
        ("num_thread", "CPU threads", "int"),
        ("low_vram", "Low VRAM mode", "bool"),
        ("use_mmap", "Memory mapping", "bool"),
        ("use_mlock", "Lock model in memory", "bool"),
        ("numa", "NUMA mode", "bool"),
        ("vocab_only", "Vocabulary only", "bool"),
    )),
)
COMMON_OPTION_KEYS = {
    name for _group, rows in OPTION_GROUPS for name, _label, _kind in rows
}
RESERVED_REQUEST_KEYS = {"model", "messages", "stream", "options", "think", "keep_alive", "format"}

# The slider covers the useful everyday range; the adjacent number field also
# accepts values beyond it, preserving custom settings from older versions.
SLIDER_OPTIONS = {
    "temperature": (0.0, 2.0, 0.01, 0.8),
    "top_p": (0.0, 1.0, 0.01, 0.9),
    "min_p": (0.0, 1.0, 0.01, 0.0),
    "typical_p": (0.0, 1.0, 0.01, 1.0),
    "tfs_z": (0.0, 2.0, 0.01, 1.0),
    "repeat_penalty": (0.0, 2.0, 0.01, 1.1),
    "presence_penalty": (-2.0, 2.0, 0.01, 0.0),
    "frequency_penalty": (-2.0, 2.0, 0.01, 0.0),
    "mirostat_tau": (0.0, 10.0, 0.1, 5.0),
    "mirostat_eta": (0.0, 1.0, 0.01, 0.1),
}


def parse_option(text: str, kind: str):
    """Parse a setting without silently changing what will reach Ollama."""
    value = text.strip()
    if kind == "int":
        return int(value)
    if kind == "float":
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("must be a finite number")
        return number
    if kind == "bool":
        if value.casefold() in ("true", "1", "yes", "on"):
            return True
        if value.casefold() in ("false", "0", "no", "off"):
            return False
        raise ValueError("enter true or false")
    if kind == "array":
        result = json.loads(value)
        if not isinstance(result, list) or not all(isinstance(item, str) for item in result):
            raise ValueError("enter a JSON array of strings")
        return result
    raise ValueError(f"unknown option type: {kind}")


def _json_object(text: str, label: str) -> dict:
    value = json.loads(text.strip() or "{}")
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _reported_parameter_defaults(details: dict) -> dict:
    """Read the defaults Ollama reports in /api/show's parameters block."""
    defaults = {}
    parameters = details.get("parameters", "") if isinstance(details, dict) else ""
    if isinstance(parameters, str):
        for line in parameters.splitlines():
            line = line.strip()
            if line.casefold().startswith("parameter "):
                line = line[10:].strip()
            name, separator, value = line.partition(" ")
            if separator and name:
                defaults[name] = value.strip()
    return defaults
