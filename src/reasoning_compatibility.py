"""Request-local reasoning compatibility for OpenAI-style endpoints."""
import json
import re


EFFORT_ORDER = ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra')
NO_NONE_SUPPORT = (r'(?:^|/)gpt-?6(?:[.-]|$)',)


class ReasoningEffortRejected(RuntimeError):
    """A permanent reasoning validation failure; repeating it cannot help."""
    status_code = 400


def _effort_fields(payload):
    if not isinstance(payload, dict):
        return
    for container in (payload, payload.get('extra_body')):
        if not isinstance(container, dict):
            continue
        if 'reasoning_effort' in container:
            yield container, 'reasoning_effort', container
        reasoning = container.get('reasoning')
        if isinstance(reasoning, dict) and 'effort' in reasoning:
            yield reasoning, 'effort', container


def normalize_none_effort(payload, log=lambda message: None):
    model = str(payload.get('model') or '').strip().lower()
    if not any(re.search(pattern, model) for pattern in NO_NONE_SUPPORT):
        return
    changed = False
    for target, key, container in _effort_fields(payload):
        if target[key] == 'none':
            target[key] = 'low'
            if container.get('thinking') == {'type': 'disabled'}:
                container.pop('thinking')
            changed = True
    if changed:
        name = 'Astra' if 'astra' in model else 'GPT-6'
        log(f'📝 {name} does not support none, using low instead')


def supported_reasoning_efforts(status, error):
    """None means unrelated; an empty list means no usable advertised fallback."""
    if status != 400:
        return None
    if isinstance(error, str):
        try:
            error = json.loads(error)
        except (ValueError, TypeError):
            pass
    if isinstance(error, dict):
        error = error.get('error', error)
    if isinstance(error, dict):
        param = str(error.get('param', ''))
        message = str(error.get('message', ''))
    else:
        param, message = '', str(error)
    text = (param + ' ' + message).lower()
    if not re.search(r'reasoning[._]effort', text):
        return None
    if not any(marker in text for marker in ('unsupported value', 'not supported', 'does not support', 'invalid value')):
        return None
    match = re.search(r'supported values(?: are)?\s*:\s*([^\n.]+)', message.lower())
    if not match:
        return []
    return [effort for effort in EFFORT_ORDER if re.search(r'\b' + effort + r'\b', match.group(1))]


def repair_reasoning_effort(payload, supported, log):
    for target, key, container in _effort_fields(payload):
        requested = target[key]
        if requested not in EFFORT_ORDER or requested in supported or not supported:
            continue
        index = EFFORT_ORDER.index(requested)
        # Prefer more thinking when supported levels are equally close.
        selected = min(supported, key=lambda value: (abs(EFFORT_ORDER.index(value) - index), -EFFORT_ORDER.index(value)))
        target[key] = selected
        if selected != 'none' and container.get('thinking') == {'type': 'disabled'}:
            container.pop('thinking')
        if selected == 'none' and key == 'effort':
            target.pop('summary', None)
        log(f'📝 Reasoning effort {requested} is not supported; retrying with {selected}.')
        return True
    return False


def call_with_reasoning_retry(call, payload, log, check_cancel):
    """Repair one explicit rejection, retaining all other request parameters."""
    for attempt in range(2):
        try:
            return call(payload)
        except Exception as exc:
            status = getattr(exc, 'status_code', None)
            response = getattr(exc, 'response', None)
            if status is None:
                status = getattr(response, 'status_code', None)
            error = getattr(exc, 'body', None) or str(exc)
            supported = supported_reasoning_efforts(status, error)
            if supported is None:
                raise
            check_cancel()
            if attempt or not repair_reasoning_effort(payload, supported, log):
                raise ReasoningEffortRejected(str(exc)) from exc
