"""Session-wide model temperature exclusions and HTTP 400 compatibility."""
import re
import threading

_rejected_models = set()
_lock = threading.Lock()


def is_temperature_rejection(status, detail):
    return status == 400 and "temperature" in str(detail or "").lower()


def model_rejects_temperature(model):
    name = str(model or "").strip().lower()
    # Accept provider prefixes, dated aliases, and both Claude naming orders.
    match = re.search(
        r"(?:^|/)claude-(?:(sonnet|opus|fable)-(\d+)(?:[.-](\d{1,2})(?!\d))?"
        r"|(\d+)(?:[.-](\d{1,2})(?!\d))?-(sonnet|opus|fable))(?=$|[-:@])",
        name,
    )
    if match:
        family = match[1] or match[6]
        version = (int(match[2] or match[4]), int(match[3] or match[5] or 0))
        minimum = {"sonnet": (5, 5), "opus": (4, 7), "fable": (5, 0)}[family]
        if version >= minimum:
            return True
    with _lock:
        return name in _rejected_models


def remember_temperature_rejection(model):
    name = str(model or "").strip().lower()
    if name:
        with _lock:
            _rejected_models.add(name)


def strip_temperature(payload):
    """Remove sampling temperature from common provider payload shapes."""
    if not isinstance(payload, dict):
        return False
    removed = "temperature" in payload
    payload.pop("temperature", None)
    for key in ("parameters", "input", "generationConfig", "generation_config", "extra_body"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            removed = strip_temperature(nested) or removed
    return removed
