"""key_pool_service (U4 chain step 3): unit tests and tier D parity of the multi_api_key_manager moves.

Moved in milestone U4 out of ``multi_api_key_manager`` into the GUI-free ``key_pool_service``,
with the dialog methods left as thin wrappers:

* ``MultiAPIKeyDialog``: ``_dedicated_pool_specs`` / ``_dedicated_pool_spec``,
  ``_export_pool_order`` / ``_pool_title`` / ``_pool_toggle_key``, ``_sanitize_imported_keys``,
  ``_collect_pools_for_export``, the pure parts of ``_import_keys`` / ``_import_legacy_list`` /
  ``_import_pool_aware`` / ``_export_keys``, the entry construction + validation of ``_add_key`` /
  ``_add_fallback_key`` / ``_add_glossary_key`` / ``_dedicated_add_key``, the client setup / probe /
  response check / timeout cancel of ``_submit_single_test`` / ``_test_single_fallback_key`` /
  ``_test_single_glossary_key`` / ``_dedicated_test_single_key``, the Translation / Fallback /
  Glossary group descriptions;
* ``RefusalPatternsDialog``: the default patterns, the config reads, the length-limit save and the
  "Load Patterns" merge;
* module level ``_model_needs_api_key`` / ``_api_key_test_timeout_seconds``;
* ``unified_api_client.UnifiedClient._get_refusal_patterns`` (default list).

The oracle is the source at ``U4_BASE_SHA`` (``git show``): multi_api_key_manager.py executed as a
separate module, ``_get_refusal_patterns`` extracted with ``ast``. Both sides run on identical
harnesses (fake widgets, message boxes, file dialogs, timers, ``QMetaObject``, a fake
``UnifiedClient``) for ``PARITY_U4_STATES`` (default 500) random states per moved function; every
observable is compared: calls in order (widgets, dialog hooks, client attribute writes, sends,
queued slots, message boxes), stdout, config, key pool contents, written files and exceptions.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_key_pool_service.py
"""

from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import os
import random
import subprocess
import sys
import textwrap
import threading
import time
import types
from datetime import datetime as _real_datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
MOBILE_APP = SRC / "mobile" / "app"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import key_pool_service as kps  # noqa: E402

#: U4 parent commit (the source the moves were taken from).
U4_BASE_SHA = "9f06f8b3efa6195a0486bddeed1c07e4af719b3c"
STATES = max(1, int(os.environ.get("PARITY_U4_STATES", "500")))
SEED = int(os.environ.get("PARITY_U4_SEED", "4343"))


# =============================================================================================
# Unit tests (GUI-free)
# =============================================================================================

def test_imports_without_qt_and_parses_as_python_310():
    code = (
        "import sys; sys.modules['PySide6'] = None; sys.path.insert(0, %r)\n"
        "import key_pool_service\n"
        "bad = [m for m in ('PySide6', 'translator_gui', 'dpi_setup', 'multi_api_key_manager',"
        " 'unified_api_client') if sys.modules.get(m) is not None]\n"
        "assert not bad, bad\n"
        "print('ok')\n" % str(SRC)
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr
    source = (SRC / "key_pool_service.py").read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 10))
    # Line endings follow the checkout (CRLF on Windows with autocrlf, LF on Linux CI); never mixed.
    data = (SRC / "key_pool_service.py").read_bytes()
    assert data.count(b"\r\n") in (0, data.count(b"\n")), "mixed line endings"


def test_pool_specs_cover_all_eleven_pools():
    from key_contexts import POOL_CONTEXTS

    assert kps.POOL_IDS == ('main', 'fallback', 'glossary', 'glossary_refinement', 'qa_scan', 'metadata',
                            'ai_truncation_detection', 'rolling_summary', 'truncation_retry', 'inpainter', 'tts')
    assert set(kps.POOL_IDS) == set(POOL_CONTEXTS)
    for pool, spec in kps.POOL_SPECS.items():
        for field in ('title', 'label', 'config_key', 'toggle_key', 'description', 'use_envs', 'keys_envs'):
            assert field in spec, (pool, field)
        assert kps.pool_config_key(pool) == spec['config_key']
    assert {p: s['config_key'] for p, s in kps.POOL_SPECS.items()}['main'] == 'multi_api_keys'
    # the dedicated part is the dialog's spec table, a fresh copy every call
    specs = kps.dedicated_pool_specs()
    specs['tts']['title'] = 'changed'
    assert kps.dedicated_pool_specs()['tts']['title'] == 'Audio / TTS Keys'
    assert kps.POOL_SPECS['tts']['title'] == 'Audio / TTS Keys'
    with pytest.raises(ValueError):
        kps.dedicated_pool_spec('glossary')
    with pytest.raises(ValueError):
        kps.pool_spec('nope')
    # desktop order: no Glossary pool; unknown names echo back / have no toggle
    assert kps.export_pool_order()[:2] == ['main', 'fallback'] and 'glossary' not in kps.export_pool_order()
    assert kps.pool_title('glossary') == 'glossary' and kps.pool_toggle_key('glossary') is None


def test_in_memory_pool_methods_exist_on_unified_client():
    uac = pytest.importorskip("unified_api_client")
    for pool, spec in kps.POOL_SPECS.items():
        for name in (spec.get('set_method'), spec.get('clear_method')):
            if name and not name.startswith('_unused'):
                assert callable(getattr(uac.UnifiedClient, name, None)), (pool, name)


def test_refusal_defaults_are_one_list(tmp_path, monkeypatch):
    assert len(kps.DEFAULT_REFUSAL_PATTERNS) == 30
    assert all(p == p.lower() for p in kps.DEFAULT_REFUSAL_PATTERNS)
    first, second = kps.default_refusal_patterns(), kps.default_refusal_patterns()
    assert first == list(kps.DEFAULT_REFUSAL_PATTERNS) and first is not second
    uac = pytest.importorskip("unified_api_client")
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "missing.json"))
    assert uac.UnifiedClient._get_refusal_patterns(None) == list(kps.DEFAULT_REFUSAL_PATTERNS)
    # U6: the QA scanner and the translator's QA-failure check import this list (no copies left)
    scan = SRC / "scan_html_folder.py"
    tree = ast.parse(scan.read_text(encoding="utf-8-sig"))
    assert not [node for node in tree.body if isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "DEFAULT_REFUSAL_PATTERNS"]
    assert any(isinstance(node, ast.ImportFrom) and node.module == "key_pool_service"
               and [(a.name, a.asname) for a in node.names] == [("DEFAULT_REFUSAL_PATTERNS", None)]
               for node in tree.body)
    tkr = ast.parse((SRC / "TransateKRtoEN.py").read_text(encoding="utf-8-sig"))
    check = next(node for node in tkr.body if isinstance(node, ast.FunctionDef)
                 and node.name == "is_qa_failed_response")
    imports = [node for node in ast.walk(check) if isinstance(node, ast.ImportFrom)
               and node.module == "key_pool_service"]
    assert [(a.name, a.asname) for node in imports for a in node.names] == [
        ("DEFAULT_REFUSAL_PATTERNS", "refusal_patterns")]
    assert not [node for node in ast.walk(check) if isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "refusal_patterns"]
    scan_html_folder = pytest.importorskip("scan_html_folder")
    assert scan_html_folder.DEFAULT_REFUSAL_PATTERNS is kps.DEFAULT_REFUSAL_PATTERNS


def test_refusal_config_helpers():
    assert kps.load_refusal_patterns({}) == list(kps.DEFAULT_REFUSAL_PATTERNS)
    assert kps.load_refusal_patterns({'refusal_patterns': []}) == []
    assert kps.load_disable_refusal_checks({}) is True
    assert kps.load_disable_refusal_checks({'disable_refusal_checks': 0}) is False
    assert kps.load_refusal_length_limit({'refusal_pattern_length_limit': '250'}) == 250
    assert kps.load_refusal_length_limit({'refusal_pattern_length_limit': 'x'}) == 1000
    assert kps.load_refusal_length_limit({'refusal_pattern_length_limit': None}) == 1000
    assert [kps.parse_refusal_length_limit(t) for t in (' 42 ', '0', '-3', 'abc', '')] == [42, 1000, 1000, 1000, 1000]
    with pytest.raises(ValueError):
        kps.parse_refusal_length_limit('²')  # the dialog's except then saves 1000
    patterns = ['as an ai']
    added, skipped = kps.merge_refusal_pattern_lines(patterns, ['  As An AI\n', '# c\n', '\n', 'New One\n', 'new one'])
    assert (added, skipped, patterns) == (1, 2, ['as an ai', 'new one'])


def test_new_entries_and_validation():
    entry = kps.new_key_entry('k', 'm', google_region='us-east5')
    assert tuple(entry) == kps.NEW_KEY_ENTRY_FIELDS
    assert entry['enabled'] is True and entry['api_call_delay'] == 0.0 and entry['times_used'] == 0
    assert kps.missing_model_error('') == "Please enter a model name" and kps.missing_model_error('m') is None
    assert kps.added_key_extra_info('/a/b/creds.json', 'https://' + 'x' * 40) == \
        f" (Google: creds.json, Azure: https://{'x' * 22}...)"
    assert kps.added_key_extra_info(None, None) == ""

    main = kps.new_main_key_entry(' k ', 'gpt', 15).to_dict()
    assert main['cooldown'] == 15 and main['azure_api_version'] == kps.DEFAULT_AZURE_API_VERSION
    ok, err = kps.validate_entry({'api_key': ' k ', 'model': ' gpt ', 'cooldown': 30}, 'main')
    assert err is None and ok['api_key'] == 'k' and ok['model'] == 'gpt' and ok['cooldown'] == 30
    assert set(ok) >= set(main)
    ok, err = kps.validate_entry({'api_key': 'k', 'model': 'm', 'individual_key_temperature': '-1',
                                  'individual_output_token_limit': '0', 'request_parameters': {'model': 'x', 'top_k': 3},
                                  'disabled_contexts': ['Glossary', 'glossary']}, 'fallback')
    assert err is None
    assert ok['individual_key_temperature'] is None and ok['individual_output_token_limit'] is None
    assert ok['request_parameters'] == {'top_k': 3} and ok['disabled_contexts'] == ['glossary']
    assert 'cooldown' not in ok  # non-main pools keep their dict shape
    assert kps.validate_entry({'api_key': 'k', 'model': '  '}, 'tts') == (None, "Please enter a model name")


def _config_with_pools(rng=None):
    rng = rng or random.Random(0)
    cfg = {}
    for pool, spec in kps.POOL_SPECS.items():
        cfg[spec['config_key']] = [kps.new_key_entry(f"{pool}-{i}", f"model-{i}") for i in range(rng.randint(0, 2))]
        cfg[spec['toggle_key']] = rng.random() < 0.5
    return cfg


def test_export_import_roundtrip_includes_the_glossary_pool(monkeypatch):
    cfg = _config_with_pools(random.Random(3))
    cfg['glossary_keys'] = [kps.new_key_entry('g-key', 'gemini-2.5-flash')]
    cfg['use_glossary_keys'] = True
    payload = kps.export_pools(cfg)
    assert payload['format'] == 'glossarion-key-pools' and payload['version'] == 1
    assert list(payload['pools'])[:3] == ['main', 'fallback', 'glossary']
    assert set(payload['pools']) == set(kps.POOL_IDS)
    assert payload['pools']['glossary'] == {'title': 'Glossary Keys', 'enabled': True, 'keys': cfg['glossary_keys']}
    payload['pools']['glossary']['keys'][0]['api_key'] = 'mutated'
    assert cfg['glossary_keys'][0]['api_key'] == 'g-key'  # no aliasing
    text = json.dumps(kps.export_pools(cfg))

    fresh = {}
    plan = kps.import_pools(text, fresh, apply=True)
    assert plan['error'] is None and plan['kind'] == 'pools' and not plan['legacy']
    assert plan['applied'] == sum(len(cfg[s['config_key']]) for s in kps.POOL_SPECS.values())
    for spec in kps.POOL_SPECS.values():
        assert fresh[spec['config_key']] == cfg[spec['config_key']]
        assert fresh[spec['toggle_key']] == cfg[spec['toggle_key']]

    # the desktop import knows only its export order: Glossary is reported as unknown there
    desktop = kps.import_pools(text, known=kps.export_pool_order())
    assert desktop['unknown'] == ['glossary']

    only = kps.export_pools(cfg, pools=['tts', 'main'])
    assert list(only['pools']) == ['main', 'tts']


def test_import_formats_and_errors():
    legacy = kps.import_pools([{'api_key': 'a', 'model': 'm'}, {'model': 'no key'}, 'junk'])
    assert legacy['legacy'] and legacy['skipped'] == 2 and legacy['items'][0][0] == 'main'
    assert legacy['items'][0][1][0]['cooldown'] == 60  # normalised like the desktop pool
    single = kps.import_pools({'api_key': 'a', 'model': 'm'})
    assert single['legacy'] and len(single['items'][0][1]) == 1
    cfg = {'multi_api_keys': [{'api_key': 'old', 'model': 'x'}]}
    kps.import_pools([{'api_key': 'new', 'model': 'y'}], cfg, apply=True)
    assert [k['api_key'] for k in cfg['multi_api_keys']] == ['old', 'new']
    assert kps.import_pools({'nothing': 1})['error'] == kps.INVALID_IMPORT_MESSAGE
    assert kps.import_pools([])['error'] == kps.NO_VALID_KEYS_MESSAGE
    assert kps.import_pools({'pools': {'bogus': []}})['error'] == kps.NO_POOLS_MESSAGE
    mixed = kps.import_pools({'pools': {'tts': [{'api_key': 'a', 'model': 'm'}, 3], 'x': [], 'fallback': 'bad',
                                        'metadata': {'keys': [], 'enabled': False}}})
    assert mixed['items'] == [('tts', [{'api_key': 'a', 'model': 'm'}], None), ('metadata', [], False)]
    assert mixed['skipped'] == 1 and mixed['unknown'] == ['x']
    dry = {}
    assert 'applied' not in kps.import_pools({'pools': {'tts': []}}, dry, apply=True, dry_run=True) and dry == {}


def test_build_test_request_variants():
    standard = kps.build_test_request({'api_key': 'k', 'model': 'm', 'google_credentials': 'c.json',
                                       'google_region': 'r', 'use_individual_endpoint': False,
                                       'azure_endpoint': 'https://e', 'azure_api_version': 'v'}, 'fallback', timeout=5)
    assert standard['variant'] == 'standard' and standard['timeout'] == 5 and standard['testable']
    assert [a for step in standard['steps'] for a in step['attrs']] == [
        ('current_key_google_creds', 'c.json'), ('google_creds_path', 'c.json'), ('current_key_google_region', 'r')]
    assert standard['client_kwargs'] == {'api_key': 'k', 'model': 'm', 'output_dir': None}
    assert standard['send_kwargs'] == {'temperature': 0.7, 'max_tokens': 1000}
    dedicated = kps.build_test_request({'api_key': 'k', 'model': 'm', 'azure_endpoint': 'https://e',
                                        'azure_api_version': 'v'}, 'metadata', timeout=5)
    # dedicated pools set the endpoint attributes even with the endpoint toggle off (desktop behaviour)
    assert [a for step in dedicated['steps'] for a in step['attrs']] == [
        ('current_key_azure_endpoint', 'https://e'), ('current_key_azure_api_version', 'v')]
    for pool in ('tts', 'inpainter'):
        req = kps.build_test_request({'api_key': 'k', 'model': 'm'}, pool, timeout=5)
        assert not req['testable'] and req['reason']
    assert kps.build_test_request({'api_key': 'k', 'model': 'm'}, 'main', timeout=1, api_key='o', model='p')[
        'client_kwargs'] == {'api_key': 'o', 'model': 'p', 'output_dir': None}


class _ProbeClient:
    """A fake UnifiedClient for run_key_test."""

    behaviour = "pass"
    instances = []

    def __init__(self, api_key=None, model=None, output_dir=None):
        self.kwargs = dict(api_key=api_key, model=model, output_dir=output_dir)
        self.tls = types.SimpleNamespace()
        self.sent = []
        self.closed = False
        self.openai_client = types.SimpleNamespace(close=self._close)
        _ProbeClient.instances.append(self)

    def _close(self):
        self.closed = True

    def _get_thread_local_client(self):
        return self.tls

    def send(self, messages, **kwargs):
        self.sent.append((messages, kwargs))
        if self.behaviour == "pass":
            return ("API test successful", "stop")
        if self.behaviour == "odd":
            return ("hello", "stop")
        if self.behaviour == "429":
            raise RuntimeError("HTTP 429 Too Many Requests")
        if self.behaviour == "slow":
            time.sleep(1.0)
            return ("API test successful", "stop")
        raise RuntimeError("boom")


@pytest.mark.parametrize("behaviour, status, ok", [
    ("pass", "passed", True), ("odd", "failed", False), ("429", "rate_limited", False), ("boom", "error", False),
])
def test_run_key_test_reports_status(behaviour, status, ok, monkeypatch):
    monkeypatch.setattr(_ProbeClient, "behaviour", behaviour)
    _ProbeClient.instances.clear()
    lines = []
    result = kps.run_key_test({'api_key': 'k', 'model': 'm', 'google_region': 'eu'}, 'main', timeout=5,
                              client_cls=_ProbeClient, log=lines.append)
    assert result['status'] == status and result['ok'] is ok and result['last_test_result'] == status
    client = _ProbeClient.instances[0]
    assert client.tls.max_retries_override == 1 and client.current_key_google_region == 'eu'
    assert client.sent[0][0] == kps.test_messages() and client.sent[0][1] == {'temperature': 0.7, 'max_tokens': 1000}
    assert lines[:2] == ["[DEBUG] Set max_retries_override=1 for key test", "[DEBUG] Set Google region for test: eu"]


def test_run_key_test_timeout_and_untestable_pools(monkeypatch):
    monkeypatch.setattr(_ProbeClient, "behaviour", "slow")
    _ProbeClient.instances.clear()
    resets = []
    monkeypatch.setattr(kps, "reset_api_watchdog", lambda: resets.append(1))
    started = time.monotonic()
    result = kps.run_key_test({'api_key': 'k', 'model': 'm'}, 'glossary', timeout=0.1, client_cls=_ProbeClient)
    assert time.monotonic() - started < 0.9  # the hung probe is left on its daemon thread
    assert result == {'ok': False, 'status': 'timeout', 'message': 'Timed out (0.1s)', 'last_test_result': 'timeout'}
    assert _ProbeClient.instances[0]._cancelled is True and _ProbeClient.instances[0].closed and resets == [1]
    for pool in ('tts', 'inpainter'):
        res = kps.run_key_test({'api_key': 'k', 'model': 'm'}, pool, client_cls=_ProbeClient)
        assert res['ok'] is None and res['status'] == 'untestable' and res['message']


def test_mobile_key_backend_binds_the_shared_service(monkeypatch):
    """The mobile Multi-Key Manager (glossarion_mobile.ui.screens.keys.KeyBackend) on the real service."""
    if not (MOBILE_APP / "glossarion_mobile" / "ui" / "screens" / "keys.py").exists():
        pytest.skip("mobile app not present")
    monkeypatch.syspath_prepend(str(MOBILE_APP))
    try:
        from glossarion_mobile.ui.screens.keys import KeyBackend
    except Exception as exc:  # pragma: no cover - mobile package import problems are not ours
        pytest.skip(f"mobile keys screen not importable: {exc}")
    backend = KeyBackend(module=kps)
    specs = backend.pool_specs()
    assert [s.id for s in specs] == list(kps.POOL_IDS)
    assert specs[2].title == 'Glossary Keys' and specs[2].description == kps.POOL_SPECS['glossary']['description']
    entry = backend.new_entry('k', 'm')
    clean, error = backend.validate(dict(entry, model=' gpt '), 'main')
    assert error is None and clean['model'] == 'gpt' and 'cooldown' in clean
    assert backend.validate({'api_key': 'k', 'model': ''}, 'main') == (None, "Please enter a model name")
    cfg = _config_with_pools(random.Random(5))
    payload = backend.export_payload(cfg, ['glossary'])
    assert list(payload['pools']) == ['glossary']
    exported = kps.export_pools(cfg)
    plan = backend.import_plan(exported, cfg)
    assert plan.error is None and [p for p, _k, _e in plan.items] == list(exported['pools'])
    assert sorted(exported['pools']) == sorted(kps.POOL_IDS)
    assert backend.refusal_defaults() == list(kps.DEFAULT_REFUSAL_PATTERNS)
    try:
        import flet  # noqa: F401 - KeyBackend.run_test normalises through the Flet key editor module
    except ImportError:
        return
    monkeypatch.setattr(_ProbeClient, "behaviour", "pass")
    real_send = kps.send_test_request
    monkeypatch.setattr(kps, "send_test_request",
                        lambda request, **kw: real_send(request, **dict(kw, client_cls=_ProbeClient)))
    assert backend.run_test({'api_key': 'k', 'model': 'm'}, 'main', timeout=5)['status'] == 'passed'
    assert backend.run_test({'api_key': 'k', 'model': 'm'}, 'tts', timeout=5)['status'] == 'untestable'


# =============================================================================================
# Tier D parity: frozen legacy (git show U4_BASE_SHA) vs working tree
# =============================================================================================

def _git_text(relpath):
    try:
        data = subprocess.run(["git", "show", f"{U4_BASE_SHA}:{relpath}"], cwd=str(REPO_ROOT),
                              capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"legacy source {relpath}@{U4_BASE_SHA[:8]} unavailable: {exc}")
    return data.decode("utf-8-sig").replace("\r\n", "\n")


@pytest.fixture(scope="module")
def modules():
    """(legacy module @U4_BASE_SHA, live multi_api_key_manager) with the same HAS_GUI."""
    return load_modules()


def load_modules():
    import multi_api_key_manager as live

    source = _git_text("src/multi_api_key_manager.py")
    assert "key_pool_service" not in source
    saved = os.environ.get("GLOSSARION_HEADLESS_KEY_MANAGER")
    if live.HAS_GUI:
        os.environ.pop("GLOSSARION_HEADLESS_KEY_MANAGER", None)
    else:
        os.environ["GLOSSARION_HEADLESS_KEY_MANAGER"] = "1"
    try:
        legacy = types.ModuleType("legacy_multi_api_key_manager")
        legacy.__file__ = str(SRC / "multi_api_key_manager.py")
        exec(compile(source, f"multi_api_key_manager.py@{U4_BASE_SHA[:8]}", "exec"), legacy.__dict__)
    finally:
        if saved is None:
            os.environ.pop("GLOSSARION_HEADLESS_KEY_MANAGER", None)
        else:
            os.environ["GLOSSARION_HEADLESS_KEY_MANAGER"] = saved
    assert legacy.HAS_GUI == live.HAS_GUI
    return legacy, live


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict") and hasattr(value, "api_key"):
        return ("APIKeyEntry", _plain(value.to_dict()))
    if isinstance(value, Harness):
        return "<self>"
    return f"<{type(value).__name__}>"


class W:
    """A fake widget (line edit / combo / check box / spin box / button)."""

    def __init__(self, log, name, text="", checked=False, value=0):
        self._log, self._name = log, name
        self._text, self._checked, self._value, self._style = text, checked, value, "base"

    def _rec(self, *event):
        self._log.append(("widget", self._name) + tuple(_plain(e) for e in event))

    def text(self):
        return self._text

    def currentText(self):
        return self._text

    def setText(self, text):
        self._rec("setText", text)
        self._text = text

    def setCurrentText(self, text):
        self._rec("setCurrentText", text)
        self._text = text

    def clear(self):
        self._rec("clear")
        self._text = ""

    def isChecked(self):
        return self._checked

    def setChecked(self, value):
        self._rec("setChecked", value)
        self._checked = bool(value)

    def blockSignals(self, value):
        self._rec("blockSignals", value)

    def value(self):
        return self._value

    def setValue(self, value):
        self._rec("setValue", value)
        self._value = value

    def styleSheet(self):
        return self._style

    def setStyleSheet(self, style):
        self._rec("setStyleSheet", len(style))
        self._style = style


class Harness:
    """``self`` for the dialog methods: real methods from the class under test, recorded hooks."""

    RECORDED = frozenset()
    RETURNS = {}

    def __init__(self, log):
        object.__setattr__(self, "_log", log)

    def __getattr__(self, name):
        if name in type(self).RECORDED:
            log = object.__getattribute__(self, "_log")

            def recorded(*args, **kwargs):
                log.append(("hook", name, _plain(args), _plain(kwargs)))
                fn = type(self).RETURNS.get(name)
                return fn(*args, **kwargs) if fn else None
            return recorded
        raise AttributeError(name)


def _harness_class(dialog_cls, names, recorded, returns=None):
    namespace = {name: dialog_cls.__dict__[name] for name in names}
    namespace["RECORDED"] = frozenset(recorded)
    namespace["RETURNS"] = dict(returns or {})
    return type(f"H_{dialog_cls.__name__}", (Harness,), namespace)


class FakeTG:
    def __init__(self, log, config, executor=True):
        self._log = log
        self.config = config
        if executor:
            self.executor = SyncExecutor()

    def _ensure_executor(self):
        self._log.append(("tg", "_ensure_executor"))

    def save_config(self, show_message=True):
        self._log.append(("tg", "save_config", show_message, _plain(self.config)))


class FakeTGNoConfig:
    def save_config(self, show_message=True):  # pragma: no cover - never reached without config
        raise AssertionError


class SyncExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)


def _make_qt_fakes(log, answers):
    class FakeMessageBox:
        Yes, No, Question = 16384, 65536, 4

        def __init__(self, parent=None):
            log.append(("msgbox", "instance"))

        @staticmethod
        def critical(parent, title, text, *args):
            log.append(("msgbox", "critical", title, text))

        @staticmethod
        def warning(parent, title, text, *args):
            log.append(("msgbox", "warning", title, text))

        @staticmethod
        def information(parent, title, text, *args):
            log.append(("msgbox", "information", title, text))

        @staticmethod
        def question(parent, title, text, *args):
            log.append(("msgbox", "question", title, text))
            return FakeMessageBox.Yes if answers.get("question", True) else FakeMessageBox.No

        def __getattr__(self, name):  # setWindowTitle / setText / setIcon / ... on an instance
            def call(*args):
                log.append(("msgbox", name, _plain(args)))
            return call

        def findChildren(self, _cls):
            return []

        def exec_(self):
            log.append(("msgbox", "exec_"))
            return FakeMessageBox.Yes if answers.get("question", True) else FakeMessageBox.No

    class FakeFileDialog:
        @staticmethod
        def getOpenFileName(parent, title, directory, filters):
            log.append(("filedialog", "open", title, filters))
            return answers.get("open", ""), ""

        @staticmethod
        def getSaveFileName(parent, title, directory, filters):
            log.append(("filedialog", "save", title, filters))
            return answers.get("save", ""), ""

    class FakeTimer:
        @staticmethod
        def singleShot(ms, fn, *args):
            log.append(("timer", ms))

    class FakeMeta:
        @staticmethod
        def invokeMethod(obj, name, conn, *args):
            log.append(("invoke", name, str(conn), [_plain(a) for a in args]))

    def fake_q_arg(type_, value):
        return ("Q_ARG", getattr(type_, "__name__", str(type_)), value)

    return {"QMessageBox": FakeMessageBox, "QFileDialog": FakeFileDialog, "QTimer": FakeTimer,
            "QMetaObject": FakeMeta, "Q_ARG": fake_q_arg}


class FakeDatetime:
    @staticmethod
    def now():
        return _real_datetime(2026, 10, 5, 12, 34, 56, 789)


@contextlib.contextmanager
def _patched(module, names):
    saved = {name: module.__dict__.get(name, _MISSING) for name in names}
    module.__dict__.update(names)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is _MISSING:
                module.__dict__.pop(name, None)
            else:
                module.__dict__[name] = value


_MISSING = object()


@contextlib.contextmanager
def _fake_module(name, module):
    saved = sys.modules.get(name, _MISSING)
    sys.modules[name] = module
    try:
        yield
    finally:
        if saved is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved


def _outcome(fn, *args):
    try:
        return ("ok", _plain(fn(*args)))
    except Exception as exc:
        return ("raise", type(exc).__name__, str(exc))


def _run_side(module, harness_cls, setup, call, answers, extra_patches=None):
    """Run one side: build the harness, patch Qt fakes, call; return every observable."""
    log = []
    fakes = _make_qt_fakes(log, answers)
    patches = dict(fakes)
    patches["datetime"] = FakeDatetime
    patches.update((extra_patches or {}).get("module", {}))
    winsound = types.ModuleType("winsound")
    winsound.MB_OK = 0
    winsound.MessageBeep = lambda *a: log.append(("winsound",))
    out = io.StringIO()
    with _patched(module, patches), _patched(kps, {"datetime": FakeDatetime}), \
            _fake_module("winsound", winsound), contextlib.redirect_stdout(out):
        owner = harness_cls(log)
        state = setup(owner, log)
        try:
            result = call(owner, state)
            error = None
        except Exception as exc:  # the dialog's own exceptions are part of the behaviour
            result, error = None, (type(exc).__name__, str(exc))
    observed = {"log": log, "stdout": out.getvalue(), "result": _plain(result), "error": error}
    observed.update(state.get("observe", lambda: {})())
    return observed


def _compare(legacy, live, harness_names, recorded, setup, call, answers, *, returns=None,
             legacy_patches=None, live_patches=None, label=""):
    legacy_cls = _harness_class(legacy.MultiAPIKeyDialog if "MultiAPIKeyDialog" in harness_names[0] else
                                legacy.RefusalPatternsDialog, harness_names[1:], recorded, returns)
    live_cls = _harness_class(live.MultiAPIKeyDialog if "MultiAPIKeyDialog" in harness_names[0] else
                              live.RefusalPatternsDialog, harness_names[1:], recorded, returns)
    old = _run_side(legacy, legacy_cls, lambda o, l: setup(o, l, legacy), call, answers, legacy_patches)
    new = _run_side(live, live_cls, lambda o, l: setup(o, l, live), call, answers, live_patches)
    assert old == new, label


# ---- module level --------------------------------------------------------------------------------

def test_parity_module_level_and_pool_specs(modules):
    legacy, live = modules
    import model_options

    rng = random.Random(SEED)
    models = list(model_options.get_model_options())[:400]
    models += ["", "authgpt/x", "authgpt0/y", "ollama/llama3", "gemini-2.5-flash", "deepseek-chat", None]
    while len(models) < STATES:
        models.append("".join(rng.choice("ab/-0123xyz") for _ in range(rng.randint(0, 12))))
    for model in models:
        assert legacy._model_needs_api_key(model) == live._model_needs_api_key(model), model
        assert legacy._api_key_test_timeout_seconds(model) == live._api_key_test_timeout_seconds(model), model
    assert (legacy._DEFAULT_KEY_TEST_TIMEOUT_SECONDS, legacy._OPTIONAL_API_KEY_TEST_TIMEOUT_SECONDS) == \
        (live._DEFAULT_KEY_TEST_TIMEOUT_SECONDS, live._OPTIONAL_API_KEY_TEST_TIMEOUT_SECONDS)

    names = ("MultiAPIKeyDialog", "_dedicated_pool_specs", "_dedicated_pool_spec", "_export_pool_order",
             "_pool_title", "_pool_toggle_key", "_sanitize_imported_keys")
    legacy_h = _harness_class(legacy.MultiAPIKeyDialog, names[1:], ())([])
    live_h = _harness_class(live.MultiAPIKeyDialog, names[1:], ())([])
    assert legacy_h._dedicated_pool_specs() == live_h._dedicated_pool_specs() == kps.dedicated_pool_specs()
    assert legacy_h._export_pool_order() == live_h._export_pool_order()
    pool_names = list(kps.POOL_IDS) + ["", "MAIN", "unknown", None, 3, ("main",)]
    while len(pool_names) < STATES:
        pool_names.append(rng.choice(list(kps.POOL_IDS)) + rng.choice(["", "_x", " "]))
    pool_names.append(["unhashable"])
    for name in pool_names:
        for method in ("_pool_title", "_pool_toggle_key", "_dedicated_pool_spec"):
            assert _outcome(getattr(legacy_h, method), name) == _outcome(getattr(live_h, method), name), (method, name)
    for _ in range(STATES):
        raw = _random_key_list(rng) if rng.random() < 0.9 else rng.choice([None, "x", 3, {"a": 1}])
        assert legacy.MultiAPIKeyDialog._sanitize_imported_keys(copy.deepcopy(raw)) == \
            live.MultiAPIKeyDialog._sanitize_imported_keys(copy.deepcopy(raw))


def test_parity_group_descriptions_come_from_pool_specs():
    tree = ast.parse(_git_text("src/multi_api_key_manager.py"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiAPIKeyDialog")
    wanted = {"_create_translation_key_pool_section": ("translation_pool_desc", "main"),
              "_create_fallback_section": ("desc_label", "fallback"),
              "_create_glossary_section": ("desc_label", "glossary")}
    found = {}
    for fn in cls.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in wanted:
            target, pool = wanted[fn.name]
            for node in ast.walk(fn):
                if (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == target
                        and isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "QLabel"):
                    found[pool] = ast.literal_eval(node.value.args[0])
                    break
    assert set(found) == {"main", "fallback", "glossary"}
    for pool, text in found.items():
        assert kps.POOL_SPECS[pool]["description"] == text, pool


# ---- random data -----------------------------------------------------------------------------------

_MODELS = ["gpt-4o", "gemini-2.5-flash", "authgpt/gpt-6-luna", "deepseek-chat", "claude-sonnet-4-6",
           "ollama/llama3", "eh/x", "", "  spaced  "]


def _random_key(rng, valid=True):
    key = {"api_key": rng.choice(["sk-a", "sk-" + "b" * 20, "", "AIza" + "c" * 30]),
           "model": rng.choice(_MODELS)}
    if not valid:
        key.pop(rng.choice(["api_key", "model"]))
    for name, choices in (("enabled", [True, False]), ("cooldown", [60, 10, 300]),
                          ("google_credentials", [None, "", "C:/x/creds.json"]),
                          ("google_region", [None, "us-east5", "europe-west4"]),
                          ("use_individual_endpoint", [True, False]),
                          ("azure_endpoint", [None, "https://res.openai.azure.com/openai/deployments/x"]),
                          ("azure_api_version", [None, "2024-06-01"]),
                          ("individual_output_token_limit", [None, 4096, "0"]),
                          ("individual_key_temperature", [None, 0.3, -1]),
                          ("api_call_delay", [0.0, 2]),
                          ("disabled_contexts", [None, ["glossary"], ["tts", "other"]]),
                          ("request_parameters", [None, {"top_k": 3}, {"model": "no"}]),
                          ("times_used", [0, 7]), ("last_test_result", [None, "passed", "failed"])):
        if rng.random() < 0.5:
            key[name] = rng.choice(choices)
    return key


def _random_key_list(rng):
    return [_random_key(rng, valid=rng.random() < 0.85) if rng.random() < 0.95 else rng.choice([1, "x", None])
            for _ in range(rng.randint(0, 4))]


def _random_config(rng):
    cfg = {}
    for spec in kps.POOL_SPECS.values():
        if rng.random() < 0.8:
            cfg[spec["config_key"]] = [_random_key(rng) for _ in range(rng.randint(0, 3))]
        if rng.random() < 0.8:
            cfg[spec["toggle_key"]] = rng.choice([True, False, 0, 1, "yes"])
    cfg["force_key_rotation"] = rng.choice([True, False])
    cfg["rotation_frequency"] = rng.choice([1, 3])
    return cfg


# ---- MultiAPIKeyDialog: import / export ----------------------------------------------------------

_IO_NAMES = ("MultiAPIKeyDialog", "_export_pool_order", "_pool_title", "_pool_toggle_key", "_read_pool_keys",
             "_collect_pools_for_export", "_set_checkbox_silent", "_sanitize_imported_keys", "_import_keys",
             "_import_legacy_list", "_apply_imported_pool", "_import_pool_aware", "_export_keys",
             "_dedicated_pool_specs", "_dedicated_pool_spec", "_dedicated_attr", "_dedicated_widget",
             "_dedicated_keys", "_dedicated_set_keys")
_IO_RECORDED = ("_refresh_key_list", "_notify_authgpt_visibility", "_load_fallback_keys",
                "_refresh_visible_key_pool_views", "_broadcast_key_pool_config_changed", "_show_status")


def _io_setup(rng_state, tmp_path):
    def setup(owner, log, module):
        cfg = copy.deepcopy(rng_state["config"])
        owner.translator_gui = FakeTG(log, cfg)
        owner.key_pool = module.APIKeyPool()
        owner.key_pool.load_from_list([k for k in copy.deepcopy(rng_state["live_main"]) if isinstance(k, dict)])
        owner.enabled_checkbox = W(log, "enabled_checkbox")
        owner.use_fallback_checkbox = W(log, "use_fallback_checkbox")
        for pool in kps.dedicated_pool_specs():
            setattr(owner, f"{pool}_keys_checkbox", W(log, f"{pool}_keys_checkbox"))
        if rng_state.get("file_text") is not None:
            Path(rng_state["open"]).write_text(rng_state["file_text"], encoding="utf-8")
        save = rng_state.get("save")
        if save:
            for candidate in (Path(save), Path(save + ".json")):
                if candidate.exists():
                    candidate.unlink()

        def observe():
            written = None
            if save:
                for candidate in (Path(save), Path(save + ".json")):
                    if candidate.exists():
                        written = (candidate.name, candidate.read_bytes())
            return {"config": _plain(cfg), "pool": [k.to_dict() for k in owner.key_pool.get_all_keys()],
                    "written": written}
        return {"observe": observe}
    return setup


def _random_import_payload(rng):
    kind = rng.random()
    if kind < 0.6:
        pools = {}
        for name in rng.sample(list(kps.POOL_IDS) + ["bogus", "Main"], rng.randint(0, 6)):
            shape = rng.random()
            if shape < 0.6:
                value = {"keys": _random_key_list(rng)}
                if rng.random() < 0.7:
                    value["enabled"] = rng.choice([True, False, None, 1])
                if rng.random() < 0.3:
                    value["title"] = "t"
            elif shape < 0.9:
                value = _random_key_list(rng)
            else:
                value = rng.choice(["x", 3, None])
            pools[name] = value
        doc = {"format": "glossarion-key-pools", "version": 1, "pools": pools}
        if rng.random() < 0.1:
            doc["pools"] = list(pools)
        return json.dumps(doc)
    if kind < 0.8:
        return json.dumps(_random_key_list(rng))
    if kind < 0.9:
        return json.dumps(_random_key(rng, valid=rng.random() < 0.7))
    return rng.choice(["{not json", '"just a string"', "42", "{}", "null"])


def test_parity_import_export(modules, tmp_path):
    legacy, live = modules
    rng = random.Random(SEED + 1)
    for i in range(STATES):
        state = {"config": _random_config(rng),
                 "live_main": [_random_key(rng) for _ in range(rng.randint(0, 3))]}
        action = rng.random()
        answers = {"question": rng.random() < 0.7}
        if action < 0.6:
            state["open"] = str(tmp_path / "import.json") if rng.random() < 0.95 else ""
            state["file_text"] = _random_import_payload(rng) if state["open"] else None
            answers["open"] = state["open"]
            call = lambda o, s: o._import_keys()  # noqa: E731
        else:
            state["save"] = str(tmp_path / rng.choice(["export.json", "export", "EXPORT.JSON"])) \
                if rng.random() < 0.9 else ""
            answers["save"] = state["save"]
            if rng.random() < 0.15:
                state["config"] = {}
                state["live_main"] = []
            call = lambda o, s: o._export_keys()  # noqa: E731
        _compare(legacy, live, _IO_NAMES, _IO_RECORDED, _io_setup(state, tmp_path), call, answers,
                 label=f"state {i}: {json.dumps(state)[:400]}")


# ---- MultiAPIKeyDialog: add key ------------------------------------------------------------------

_ADD_NAMES = ("MultiAPIKeyDialog", "_add_key", "_add_fallback_key", "_add_glossary_key", "_dedicated_add_key",
              "_dedicated_pool_specs", "_dedicated_pool_spec", "_dedicated_attr", "_dedicated_widget",
              "_dedicated_keys", "_dedicated_set_keys")
_ADD_RECORDED = ("_sync_parent_google_credentials", "_set_combo_text_silently", "_toggle_individual_endpoint_fields",
                 "_toggle_fallback_individual_endpoint_fields", "_toggle_glossary_individual_endpoint_fields",
                 "_refresh_key_list", "_load_fallback_keys", "_load_glossary_keys", "_dedicated_load_keys",
                 "_show_status", "_show_fallback_status", "_show_glossary_status", "_dedicated_status",
                 "_schedule_key_pool_config_flush", "_model_affects_parent_provider_controls",
                 "_broadcast_key_pool_config_changed")
_ADD_RETURNS = {"_model_affects_parent_provider_controls": lambda model: len(str(model)) % 2 == 0}
_ADD_PREFIX = {"main": "", "fallback": "fallback_", "glossary": "glossary_"}


def _add_setup(state):
    def setup(owner, log, module):
        cfg = copy.deepcopy(state["config"])
        owner.translator_gui = FakeTG(log, cfg)
        owner.key_pool = module.APIKeyPool()
        owner.key_pool.load_from_list(copy.deepcopy(state["live_main"]))
        widgets = state["widgets"]
        pool = state["pool"]
        if pool in _ADD_PREFIX:
            p = _ADD_PREFIX[pool]
            key_attr = "api_key_entry" if pool == "main" else f"{p}key_entry"
            names = {key_attr: ("text", widgets["key"]), f"{p}model_combo": ("text", widgets["model"]),
                     f"{p}google_creds_entry": ("text", widgets["creds"]),
                     f"{p}google_region_entry": ("text", widgets["region"]),
                     f"{p}individual_endpoint_toggle": ("checked", widgets["toggle"]),
                     f"{p}azure_endpoint_entry": ("text", widgets["endpoint"]),
                     f"{p}azure_api_version_combo": ("text", widgets["version"])}
            if pool == "main":
                names["cooldown_spinbox"] = ("value", widgets["cooldown"])
        else:
            names = {f"{pool}_{suffix}": kind_value for suffix, kind_value in (
                ("key_entry", ("text", widgets["key"])), ("model_combo", ("text", widgets["model"])),
                ("google_creds_entry", ("text", widgets["creds"])), ("google_region_entry", ("text", widgets["region"])),
                ("individual_endpoint_toggle", ("checked", widgets["toggle"])),
                ("azure_endpoint_entry", ("text", widgets["endpoint"])),
                ("azure_api_version_combo", ("text", widgets["version"])))}
        for name, (kind, value) in names.items():
            setattr(owner, name, W(log, name, **{kind: value}))

        def observe():
            return {"config": _plain(cfg), "pool": [k.to_dict() for k in owner.key_pool.get_all_keys()]}
        return {"observe": observe}
    return setup


def test_parity_add_key(modules):
    legacy, live = modules
    rng = random.Random(SEED + 2)
    pools = ["main", "fallback", "glossary"] + list(kps.dedicated_pool_specs())
    for i in range(STATES):
        pool = pools[i % len(pools)]
        state = {"pool": pool, "config": _random_config(rng),
                 "live_main": [_random_key(rng) for _ in range(rng.randint(0, 2))],
                 "widgets": {"key": rng.choice(["", "  sk-x  ", "AIzaSyD"]),
                             "model": rng.choice(["", "   ", " gpt-4o ", "gemini-2.5-flash", "authgpt/x"]),
                             "creds": rng.choice(["", "  ", "C:/creds/service.json", " relative.json "]),
                             "region": rng.choice(["", "us-east5", " europe-west4 "]),
                             "toggle": rng.random() < 0.5,
                             "endpoint": rng.choice(["", "https://" + "e" * rng.randint(1, 60) + ".azure.com"]),
                             "version": rng.choice(["2025-01-01-preview", " 2024-06-01 ", ""]),
                             "cooldown": rng.choice([60, 10, 3600])}}
        if rng.random() < 0.1:
            state["config"].pop(kps.POOL_SPECS[pool]["config_key"], None)
        method = {"main": "_add_key", "fallback": "_add_fallback_key", "glossary": "_add_glossary_key"}.get(pool)
        call = (lambda o, s, m=method: getattr(o, m)()) if method else \
            (lambda o, s, p=pool: o._dedicated_add_key(p))
        _compare(legacy, live, _ADD_NAMES, _ADD_RECORDED, _add_setup(state), call, {},
                 returns=_ADD_RETURNS, label=f"state {i}: {json.dumps(state)[:400]}")


# ---- MultiAPIKeyDialog: key tests ------------------------------------------------------------------

_TEST_NAMES = ("MultiAPIKeyDialog", "_submit_single_test", "_test_single_fallback_key", "_test_single_glossary_key",
               "_dedicated_test_single_key", "_dedicated_pool_specs", "_dedicated_pool_spec")
_TEST_RECORDED = ("_handle_test_result", "_update_fallback_test_result", "_update_fallback_timeout_status",
                  "_update_glossary_test_result", "_update_glossary_timeout_status", "_dedicated_update_test_result",
                  "_dedicated_update_timeout")


def _fake_unified_module(log, state, release):
    class FakeTLS:
        def __setattr__(self, name, value):
            log.append(("tls", name, _plain(value)))
            object.__setattr__(self, name, value)

    class FakeOpenAI:
        def close(self):
            log.append(("openai_client", "close"))

    class FakeInner:
        def close(self):
            log.append(("openai_client._client", "close"))

    class FakeUnifiedClient:
        @staticmethod
        def _model_needs_api_key(model):
            return not str(model or "").startswith("authgpt")

        def __init__(self, **kwargs):
            log.append(("client", "init", _plain(kwargs)))
            oc = state["openai"]
            if oc == "close":
                object.__setattr__(self, "openai_client", FakeOpenAI())
            elif oc == "inner":
                object.__setattr__(self, "openai_client", types.SimpleNamespace(_client=FakeInner()))

        def __setattr__(self, name, value):
            log.append(("client", "set", name, _plain(value)))
            object.__setattr__(self, name, value)

        def _get_thread_local_client(self):
            if state["tls"] == "raise":
                raise RuntimeError("no tls")
            return FakeTLS()

        def send(self, messages, **kwargs):
            log.append(("client", "send", _plain(messages), _plain(kwargs)))
            if state["slow"]:
                release.wait(5)
            response = state["response"]
            if isinstance(response, Exception):
                raise response
            return response

    def watchdog_reset():
        log.append(("watchdog_reset",))
        release.set()

    module = types.ModuleType("unified_api_client")
    module.UnifiedClient = FakeUnifiedClient
    module._api_watchdog_reset = watchdog_reset
    module.is_stop_requested = lambda: False
    return module


_RESPONSES = [("API test successful", "stop"), ("api TEST SUCCESSFUL!", None), ("nope", "stop"), None, ("",),
              ("a", "b", "c"), ({"x": 1}, None), "API Test Successful", (), ("x",),
              RuntimeError("HTTP 429 rate limit"), ValueError("boom " + "z" * 80)]


def _random_test_entry(rng):
    entry = {"api_key": rng.choice(["sk-a", "", "AIza" + "q" * 20]), "model": rng.choice(_MODELS + ["authgpt/x"])}
    for name, choices in (("google_credentials", [None, "", "C:/c/creds.json", 123]),
                          ("google_region", [None, "", "us-east5", 7]),
                          ("use_individual_endpoint", [True, False, None, "yes"]),
                          ("azure_endpoint", [None, "", "https://" + "a" * 70, 5]),
                          ("azure_api_version", [None, "", "2024-06-01"])):
        if rng.random() < 0.6:
            entry[name] = rng.choice(choices)
    return entry


def test_parity_key_tests(modules, monkeypatch):
    legacy, live = modules
    rng = random.Random(SEED + 3)
    pools = ["main", "fallback", "glossary"] + list(kps.dedicated_pool_specs())
    for i in range(STATES):
        pool = pools[i % len(pools)]
        state = {"pool": pool, "entry": _random_test_entry(rng), "tls": rng.choice(["ok", "ok", "raise"]),
                 "openai": rng.choice(["close", "inner", None]), "slow": rng.random() < 0.02,
                 "response": rng.choice(_RESPONSES), "index": rng.randint(0, 2)}
        timeout = 0.3 if state["slow"] else 30

        def setup(owner, log, module, state=state):
            release = threading.Event()
            fake = _fake_unified_module(log, state, release)
            monkeypatch.setitem(sys.modules, "unified_api_client", fake)
            owner.translator_gui = FakeTG(log, {})
            owner.key_pool = module.APIKeyPool()
            entries = [dict(state["entry"], model=f"filler-{n}") for n in range(state["index"])] + [state["entry"]]
            owner.key_pool.load_from_list(copy.deepcopy(entries))
            return {}

        def call(owner, _s, state=state):
            pool, entry, index = state["pool"], copy.deepcopy(state["entry"]), state["index"]
            if pool == "main":
                return owner._submit_single_test(index)
            if pool == "fallback":
                return owner._test_single_fallback_key(entry, index)
            if pool == "glossary":
                return owner._test_single_glossary_key(entry, index)
            return owner._dedicated_test_single_key(pool, entry, index)

        patches = {"module": {"_api_key_test_timeout_seconds": lambda model, t=timeout: t}}
        _compare(legacy, live, _TEST_NAMES, _TEST_RECORDED, setup, call, {},
                 legacy_patches=patches, live_patches=patches,
                 label=f"state {i}: {state!r}"[:500])


# ---- RefusalPatternsDialog ------------------------------------------------------------------------

_REFUSAL_NAMES = ("RefusalPatternsDialog", "_load_patterns", "_get_default_patterns", "_load_disable_refusal_checks",
                  "_load_refusal_length_limit", "_reset_to_defaults", "_load_patterns_from_file", "_save_and_close")
_REFUSAL_RECORDED = ("_refresh_tree",)


def test_parity_refusal_patterns_dialog(modules, tmp_path):
    legacy, live = modules
    rng = random.Random(SEED + 4)
    lines_pool = ["i cannot assist\n", "  As An AI  \n", "# comment\n", "\n", "New Pattern\n", "another one",
                  "I CANNOT HELP\n", "   \n", "#x\n", "zz unique\n"]
    for i in range(STATES):
        cfg = {}
        if rng.random() < 0.7:
            cfg["refusal_patterns"] = rng.choice([[], ["x"], ["as an ai", "y"], None, "str"])
        if rng.random() < 0.7:
            cfg["disable_refusal_checks"] = rng.choice([True, False, 0, "", None, "no"])
        if rng.random() < 0.7:
            cfg["refusal_pattern_length_limit"] = rng.choice([1000, "250", "abc", None, 0, -5, 1.7, [1]])
        has_config = rng.random() < 0.9
        action = rng.choice(["load", "load", "defaults", "reset", "file", "save"])
        file_lines = [rng.choice(lines_pool) for _ in range(rng.randint(0, 6))]
        open_name = rng.choice([str(tmp_path / "patterns.txt"), "", str(tmp_path / "missing.txt")])
        tg_vars = rng.random() < 0.5
        buttons = rng.random() < 0.5
        limit_text = rng.choice([" 42 ", "0", "-3", "abc", "", "²", "1000", "७"])
        checked = rng.random() < 0.5
        start_patterns = rng.choice([["as an ai"], [], ["a", "new pattern"]])
        answers = {"question": rng.random() < 0.5, "open": open_name}

        def setup(owner, log, module, cfg=cfg, has_config=has_config, file_lines=file_lines,
                  tg_vars=tg_vars, buttons=buttons, limit_text=limit_text, checked=checked,
                  start_patterns=start_patterns):
            config = copy.deepcopy(cfg)
            tg = FakeTG(log, config) if has_config else FakeTGNoConfig()
            if tg_vars:
                tg.disable_refusal_checks_var = None
                tg.refusal_pattern_length_limit_var = None
            if buttons:
                tg._refusal_patterns_btn_other = W(log, "btn_other")
            owner.translator_gui = tg
            owner.patterns = list(start_patterns)
            owner._load_btn = W(log, "load_btn")
            owner._save_btn = W(log, "save_btn")
            owner.disable_refusal_checks_cb = W(log, "disable_cb", checked=checked)
            owner.refusal_length_limit_entry = W(log, "limit_entry", text=limit_text)
            target = tmp_path / "patterns.txt"
            target.write_text("".join(file_lines), encoding="utf-8")

            def observe():
                return {"config": _plain(config), "patterns": list(owner.patterns),
                        "tg": _plain({k: v for k, v in vars(tg).items() if k.endswith("_var")})}
            return {"observe": observe}

        def call(owner, _s, action=action):
            if action == "load":
                return (owner._load_patterns(), owner._load_disable_refusal_checks(), owner._load_refusal_length_limit())
            if action == "defaults":
                return owner._get_default_patterns()
            if action == "reset":
                return owner._reset_to_defaults()
            if action == "file":
                return owner._load_patterns_from_file()
            return owner._save_and_close()

        _compare(legacy, live, _REFUSAL_NAMES, _REFUSAL_RECORDED, setup, call, answers, label=f"state {i}: {action}")


# ---- unified_api_client._get_refusal_patterns -------------------------------------------------------

def test_parity_unified_client_refusal_patterns(tmp_path, monkeypatch):
    uac = pytest.importorskip("unified_api_client")
    source = _git_text("src/unified_api_client.py")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnifiedClient")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_refusal_patterns")
    lines = source.split("\n")
    namespace = {"os": os, "json": json}
    exec(textwrap.dedent("\n".join(lines[node.lineno - 1:node.end_lineno])), namespace)
    legacy_fn = namespace["_get_refusal_patterns"]
    rng = random.Random(SEED + 5)
    for i in range(STATES):
        path = tmp_path / f"cfg{i % 7}.json"
        choice = rng.random()
        if choice < 0.15:
            if path.exists():
                path.unlink()
        elif choice < 0.3:
            path.write_text("{broken", encoding="utf-8")
        else:
            cfg = {}
            if rng.random() < 0.8:
                cfg["refusal_patterns"] = rng.choice([[], ["x"], ["a", "b"], None, "s", {"a": 1}, 0])
            path.write_text(json.dumps(cfg), encoding="utf-8")
        monkeypatch.setenv("CONFIG_FILE", str(path))
        assert legacy_fn(None) == uac.UnifiedClient._get_refusal_patterns(None), i
