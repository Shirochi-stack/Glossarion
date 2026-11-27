"""Find the real endpoint of a custom prefix route after a 404, and update the route.

A custom prefix route (``custom_prefix_routes``: ``{"prefix", "routing", "endpoint_type"}``) points a
model-name prefix at a base URL. When the request there answers HTTP 404 the base URL is often just
one step off (``https://status.example.net`` instead of ``https://api.example.net/v1``). This module
searches a short fixed list of likely bases on the same registrable domain, tells a wrong URL from a
right one without spending a generation, and rewrites the route.

Rules
* Only a 404 starts a search, at most once per prefix per process (``handle_not_found``); nothing loops.
* Candidates (``candidate_bases``), fixed order, deduplicated, never the current URL, never another
  registrable domain: (1) current base + ``/v1``; (2) ``api.<domain>/v1``, then ``api.<domain>``;
  (3) ``<domain>/v1``; (4) when the path already ends in ``/v1`` or ``/v2``, the other one.
  localhost and IP bases only try the ``/v1`` variants (1) and (4).
* A candidate is probed WITHOUT the API key: ``POST <candidate><endpoint_type>`` with ``{}`` and a
  short timeout. 404 or a connection error = wrong; 400/401/403/405/422 = the endpoint exists.
  The first hit is confirmed once WITH the key: ``GET <candidate>/models`` answering 200 JSON, else
  the 400/401/422 evidence of the probe. No real generation is ever sent.
* On success the route is rewritten in ``os.environ`` (what the request code reads), in every
  registered listener (the desktop owner, the mobile settings store) and, when no listener took it,
  in config.json; the old URL is kept under ``custom_prefix_previous_routing``.

GUI-free (desktop and mobile share it); Python 3.10 compatible.
"""

import copy
import json
import os
import re
import threading
from urllib.parse import urlsplit, urlunsplit

#: Statuses of a key-less ``POST {}`` that show the endpoint is there (it refused the empty request).
EXISTS_STATUSES = frozenset({400, 401, 403, 405, 422})
#: Of those, the ones that count as proof without a ``/models`` answer.
CONFIRM_STATUSES = frozenset({400, 401, 422})
#: config.json key that keeps the URL a route had before discovery changed it: ``{prefix: old_url}``.
PREVIOUS_ROUTING_KEY = 'custom_prefix_previous_routing'

PROBE_TIMEOUT = (4, 8)  # (connect, read) seconds for one probe or confirm request

_VERSION_TAIL = re.compile(r'/v(\d+)$', re.IGNORECASE)
_ENDPOINT_VERSION_HEAD = re.compile(r'^/v\d+/', re.IGNORECASE)
#: Second-level labels that make ``example.co.uk`` the registrable domain, not ``co.uk``.
_SECOND_LEVEL = frozenset({'co', 'com', 'net', 'org', 'gov', 'edu', 'ac', 'or', 'ne', 'go'})

_lock = threading.Lock()
_prefix_locks = {}
_session = {}  # prefix key -> new routing found, or None (searched, nothing)
_listeners = []


# ---------------------------------------------------------------- candidate generation

def is_local_host(host):
    """True for localhost, ``*.localhost``, ``*.local`` names and IP literals (v4 or v6)."""
    host = (host or '').strip('[]').lower()
    if not host:
        return True
    if host == 'localhost' or host.endswith('.localhost') or host.endswith('.local'):
        return True
    if ':' in host:
        return True  # IPv6 literal
    return bool(re.fullmatch(r'\d{1,3}(\.\d{1,3}){3}', host))


def registrable_domain(host):
    """The registrable domain of *host* (``status.5n7.net`` -> ``5n7.net``, ``a.b.co.uk`` -> ``b.co.uk``).

    A heuristic without a public-suffix list: the last two labels, three when the second-level label
    is a common one under a two-letter country code. Single-label names come back unchanged.
    """
    labels = [p for p in (host or '').lower().strip('.').split('.') if p]
    if len(labels) <= 2:
        return '.'.join(labels)
    if len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL:
        return '.'.join(labels[-3:])
    return '.'.join(labels[-2:])


def endpoint_carries_version(endpoint_type):
    """True when the endpoint path starts with a version segment (``/v1/messages``, ``/v1/ocr``)."""
    return bool(_ENDPOINT_VERSION_HEAD.match(str(endpoint_type or '')))


def supports_discovery(endpoint_type):
    """Endpoint types with a placeholder (``/{model_id}``) cannot be probed with a fixed path."""
    endpoint = str(endpoint_type or '').strip()
    return bool(endpoint.startswith('/')) and '{' not in endpoint


def _netloc(parts, host):
    return f"{host}:{parts.port}" if parts.port else host


def _build(parts, host, path):
    return urlunsplit((parts.scheme, _netloc(parts, host), path, '', '')).rstrip('/')


def candidate_bases(base_url, endpoint_type='/chat/completions'):
    """Likely replacement bases for *base_url*, in the order they are tried (see module doc)."""
    raw = str(base_url or '').strip().rstrip('/')
    if not raw.lower().startswith(('http://', 'https://')):
        return []
    parts = urlsplit(raw)
    host = (parts.hostname or '').lower()
    if not host:
        return []
    path = parts.path.rstrip('/')
    current = _build(parts, host, path)
    local = is_local_host(host)
    domain = registrable_domain(host)
    # The endpoint path already holds the version (/v1/messages): a base must not add a second one.
    add_version = not endpoint_carries_version(endpoint_type)
    tail = _VERSION_TAIL.search(path)

    ordered = []
    if add_version and not tail:
        ordered.append(_build(parts, host, path + '/v1'))
    if not local:
        api_host = f'api.{domain}'
        if add_version:
            ordered.append(_build(parts, api_host, '/v1'))
        ordered.append(_build(parts, api_host, ''))
        if add_version:
            ordered.append(_build(parts, domain, '/v1'))
        else:
            ordered.append(_build(parts, domain, ''))
    if tail and tail.group(1) in ('1', '2'):
        other = '2' if tail.group(1) == '1' else '1'
        ordered.append(_build(parts, host, path[:tail.start()] + '/v' + other))

    result = []
    seen = {current.lower()}
    for candidate in ordered:
        key = candidate.lower()
        if key in seen:
            continue
        cand_host = (urlsplit(candidate).hostname or '').lower()
        if not local and not (cand_host == domain or cand_host.endswith('.' + domain)):
            continue
        seen.add(key)
        result.append(candidate)
    return result


# ---------------------------------------------------------------- probing

def _http(method, url, *, headers=None, json_body=None, timeout=PROBE_TIMEOUT):
    """``(status, body_text)``; status None when nothing answered. Redirects are not followed."""
    try:
        import requests
    except ImportError:
        requests = None
    try:
        if requests is not None:
            resp = requests.request(method, url, headers=headers or {}, json=json_body,
                                    timeout=timeout, allow_redirects=False)
            return resp.status_code, (resp.text or '')[:200000]
        import urllib.error
        import urllib.request
        data = json.dumps(json_body).encode('utf-8') if json_body is not None else None
        hdrs = dict(headers or {})
        if data is not None:
            hdrs.setdefault('Content-Type', 'application/json')
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout[1]) as resp:
                return resp.status, resp.read(200000).decode('utf-8', 'replace')
        except urllib.error.HTTPError as err:
            return err.code, ''
    except Exception:
        return None, ''


def _probe(url):
    """Key-less ``POST {}`` to *url*; the status or None."""
    return _http('POST', url, json_body={})[0]


def _models_url(candidate, endpoint_type):
    head = _ENDPOINT_VERSION_HEAD.match(str(endpoint_type or ''))
    version = head.group(0)[:-1] if head else ''
    return f"{candidate}{version}/models"


def _confirm_with_key(candidate, endpoint_type, api_key, get=None):
    """True when ``GET <candidate>/models`` with the key answers 200 JSON."""
    if not api_key:
        return False
    headers = {'Authorization': f'Bearer {api_key}', 'x-api-key': str(api_key),
               'anthropic-version': '2023-06-01', 'Accept': 'application/json'}
    status, body = (get or _http)('GET', _models_url(candidate, endpoint_type), headers=headers)
    if status != 200:
        return False
    try:
        return isinstance(json.loads(body), (dict, list))
    except Exception:
        return False


class DiscoveryResult:
    """What ``discover_endpoint`` found: ``found`` (new base or None), ``tried`` [(url, status)],
    ``current_ok`` (the current URL itself answers, so it is not a wrong URL) and ``evidence``."""

    def __init__(self):
        self.found = None
        self.tried = []
        self.current_ok = False
        self.evidence = ''


def discover_endpoint(base_url, endpoint_type='/chat/completions', api_key='', *, probe=None, get=None):
    """Search the candidates of *base_url*; see the module doc. *probe* / *get* are test seams."""
    probe = probe or _probe
    result = DiscoveryResult()
    endpoint = str(endpoint_type or '/chat/completions').strip()
    current = str(base_url or '').strip().rstrip('/')
    # A 404 can also be a real endpoint answering "model not found": check the current URL first.
    status = probe(f"{current}{endpoint}")
    if status in EXISTS_STATUSES:
        result.current_ok = True
        return result
    for candidate in candidate_bases(current, endpoint):
        status = probe(f"{candidate}{endpoint}")
        result.tried.append((candidate, status))
        if status not in EXISTS_STATUSES:
            continue
        if _confirm_with_key(candidate, endpoint, api_key, get):
            result.found, result.evidence = candidate, 'models'
        elif status in CONFIRM_STATUSES:
            result.found, result.evidence = candidate, str(status)
        return result  # the first hit ends the search, confirmed or not
    return result


# ---------------------------------------------------------------- applying a found route

def _prefix_key(prefix):
    key = str(prefix or '').strip().replace('\\', '/').lstrip('/').lower()
    return key if key.endswith('/') or not key else key + '/'


def apply_route_to_config(config, prefix, old_url, new_url):
    """Set the route's routing to *new_url* in the config dict and keep *old_url*; True when changed."""
    key = _prefix_key(prefix)
    routes = config.get('custom_prefix_routes')
    if not isinstance(routes, list):
        return False
    changed = False
    for route in routes:
        if isinstance(route, dict) and _prefix_key(route.get('prefix')) == key:
            if 'base_url' in route and 'routing' not in route:
                route['base_url'] = new_url
            else:
                route['routing'] = new_url
            changed = True
    if not changed:
        return False
    previous = config.get(PREVIOUS_ROUTING_KEY)
    if not isinstance(previous, dict):
        previous = {}
    previous[str(prefix)] = old_url
    config[PREVIOUS_ROUTING_KEY] = previous
    return True


def update_env_routes(prefix, new_url):
    """Rewrite the route in ``CUSTOM_OPENAI_PREFIX_ROUTES`` (the text the request code reads)."""
    raw = os.environ.get('CUSTOM_OPENAI_PREFIX_ROUTES', '') or ''
    try:
        parsed = json.loads(raw)
    except Exception:
        return False
    key = _prefix_key(prefix)
    if isinstance(parsed, dict):
        parsed = [{'prefix': k, 'routing': v} for k, v in parsed.items()]
    if not isinstance(parsed, list):
        return False
    changed = False
    for route in parsed:
        if isinstance(route, dict) and _prefix_key(route.get('prefix')) == key:
            route['routing'] = new_url
            route.pop('base_url', None)
            changed = True
    if changed:
        os.environ['CUSTOM_OPENAI_PREFIX_ROUTES'] = json.dumps(parsed, ensure_ascii=False)
    return changed


def register_route_listener(callback):
    """``callback(prefix, old_url, new_url)`` runs after a route changed; return True when it saved the
    change to its settings store (the desktop owner, the mobile config store)."""
    with _lock:
        if callback not in _listeners:
            _listeners.append(callback)


def unregister_route_listener(callback):
    with _lock:
        if callback in _listeners:
            _listeners.remove(callback)


def persist_route_to_config_file(prefix, old_url, new_url, path=None):
    """Write the route change into config.json, leaving every other key as it is on disk."""
    try:
        import config_store
        current = config_store.load_config_retrying(path, decrypt=False)
        scratch = {k: copy.deepcopy(current.get(k)) for k in ('custom_prefix_routes', PREVIOUS_ROUTING_KEY)
                   if k in current}
        if not apply_route_to_config(scratch, prefix, old_url, new_url):
            return False
        config_store.update_config_file(scratch, path)
        return True
    except Exception:
        return False


def record_route_update(prefix, old_url, new_url):
    """Make the new URL the route's routing everywhere; True when a settings store kept it."""
    update_env_routes(prefix, new_url)
    with _lock:
        listeners = list(_listeners)
    saved = False
    for callback in listeners:
        try:
            if callback(prefix, old_url, new_url):
                saved = True
        except Exception:
            continue
    if not saved:
        saved = persist_route_to_config_file(prefix, old_url, new_url)
    return saved


# ---------------------------------------------------------------- the entry point of the request code

def reset_session():
    """Forget which prefixes were searched (tests; a fresh process starts empty anyway)."""
    with _lock:
        _session.clear()
        _prefix_locks.clear()


def handle_not_found(prefix, routing, endpoint_type='/chat/completions', api_key='', *, log=print,
                     probe=None, get=None):
    """React to a 404 on a custom prefix route; returns the new base URL to retry on, or None.

    Searches once per prefix per process. A second call (another worker thread whose request failed
    on the same old URL) gets the answer of the first without another search.
    """
    key = _prefix_key(prefix)
    old = str(routing or '').strip().rstrip('/')
    if not key or not old or not supports_discovery(endpoint_type):
        return None
    with _lock:
        prefix_lock = _prefix_locks.setdefault(key, threading.Lock())
    with prefix_lock:
        if key in _session:
            known = _session[key]
            return known if known and known.rstrip('/').lower() != old.lower() else None
        _session[key] = None  # marked before the work: whatever happens, this prefix is searched once
        result = discover_endpoint(old, endpoint_type, api_key, probe=probe, get=get)
        if result.current_ok:
            log(f"🔧 {prefix}: 404 at {old}, but the endpoint answers; no other URL searched")
            return None
        if not result.found:
            tried = ', '.join(url for url, _status in result.tried) or 'nothing to try'
            log(f"🔧 {prefix}: 404 at {old} - tried {tried}; no endpoint found, keeping the original")
            return None
        _session[key] = result.found
        record_route_update(prefix, old, result.found)
        log(f"🔧 {prefix}: 404 at {old} — found {result.found}, endpoint updated (old URL kept)")
        return result.found
