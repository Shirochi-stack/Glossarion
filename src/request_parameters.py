"""Per-key request-body parameters for API key pools."""

import json


# These fields define the request itself and must stay under Glossarion's control.
RESERVED_REQUEST_PARAMETERS = frozenset({
    'api_key', 'authorization', 'extra_body', 'extra_headers',
    'input', 'messages', 'model',
    'stream', 'stream_options',
})


def normalize_request_parameters(value):
    """Return JSON-safe, non-structural parameters from a saved key entry."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for name, item in value.items():
        if not isinstance(name, str):
            continue
        name = name.strip()
        if not name or name.lower() in RESERVED_REQUEST_PARAMETERS:
            continue
        try:
            json.dumps(item, allow_nan=False)
        except (TypeError, ValueError):
            continue
        result[name] = item
    return result


def parse_parameter_value(text):
    """Accept plain text as a string, or JSON for numbers, booleans and objects."""
    text = str(text).strip()
    if not text:
        return ''
    try:
        return json.loads(text)
    except ValueError:
        return text


def display_parameter_value(value):
    """Show ordinary strings without JSON quotes in the table editor."""
    if isinstance(value, str):
        if value.lower() in {'true', 'false', 'null'} or value[:1] in {'{', '[', '"'}:
            return json.dumps(value, ensure_ascii=False)
        try:
            float(value)
            return json.dumps(value, ensure_ascii=False)
        except ValueError:
            pass
        return value
    return json.dumps(value, ensure_ascii=False)
