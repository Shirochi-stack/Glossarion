"""Hotfix 3: the native in-app browser's layers agree on every name, read from their sources.

Nothing here compiles Dart, Kotlin or Swift (there is no toolchain on the host); the sources are parsed
instead, so a renamed method, argument, answer key or event fails here and not first in Build Mobile:

    services/native_webview.py   NativeBridge.call("web_x", ...)          (the app's driver)
    flet_glossarion_native/native.py   GlossarionNative.web_x -> _web_call("web_x", {args})
    lib/src/native_service.dart  _kWebMethods, `case 'web_x':` -> _platform(name, args); `case 'web':`
    HiddenWebPages.kt / .swift   call.argument("key") / args["key"], answer keys, events

The extension's own tests (``extensions/flet_glossarion_native/tests/test_web_pages.py``) check the
platform files against ``web_pages.py``; this file checks what the app really sends and reads. It also
pins that the on-device ``native_webview`` check runs in the smoke suite the Android emulator and the
iOS simulator jobs block on, and the driver-aware limitations text in Accounts › Experimental.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import inspect
import re
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
REPO_DIR = MOBILE_DIR.parents[1]
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import browser_driver  # noqa: E402
from glossarion_mobile.diagnostics import selftest, webview_check  # noqa: E402
from glossarion_mobile.services import native as native_mod  # noqa: E402
from glossarion_mobile.services import native_webview as nw  # noqa: E402
from glossarion_mobile.services import webview_bridge as wb  # noqa: E402

EXT_DIR = MOBILE_DIR / "extensions" / "flet_glossarion_native"
EXT_PY = EXT_DIR / "src" / "flet_glossarion_native"
FLUTTER = EXT_DIR / "src" / "flutter" / "flet_glossarion_native"
DART = FLUTTER / "lib" / "src" / "native_service.dart"
KOTLIN_DIR = FLUTTER / "android" / "src" / "main" / "kotlin" / "com" / "glossarion" / "flet_glossarion_native"
KOTLIN_WEB = KOTLIN_DIR / "HiddenWebPages.kt"
KOTLIN_PLUGIN = KOTLIN_DIR / "GlossarionNativePlugin.kt"
SWIFT_DIR = FLUTTER / "ios" / "flet_glossarion_native" / "Sources" / "flet_glossarion_native"
SWIFT_WEB = SWIFT_DIR / "HiddenWebPages.swift"
SWIFT_PLUGIN = SWIFT_DIR / "GlossarionNativePlugin.swift"
DRIVER = APP_DIR / "glossarion_mobile" / "services" / "native_webview.py"
CHECK = APP_DIR / "glossarion_mobile" / "diagnostics" / "webview_check.py"
BUILD_MOBILE = REPO_DIR / ".github" / "workflows" / "build-mobile.yml"
ANDROID_SMOKE = MOBILE_DIR / "ci" / "android_smoke.sh"

#: Answer keys every method shares (``web_pages.normalize_result``); the per-method checks skip them.
COMMON_KEYS = {"ok", "error", "message"}


def _text(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _tree(path: Path) -> ast.Module:
    return ast.parse(_text(path))


def _load_web_pages():
    """The extension's contract module, loaded from its file (it imports nothing of Flet)."""
    spec = importlib.util.spec_from_file_location("_hotfix3_web_pages_contract", EXT_PY / "web_pages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WP = _load_web_pages()
WEB_METHODS = tuple(native_mod.WEB_METHODS)


# ============================================================================ source helpers
def _skip_literal(src: str, i: int, quote: str) -> int:
    """Index after the string literal starting at ``src[i] == quote``."""
    i += 1
    while i < len(src):
        if src[i] == "\\":
            i += 2
            continue
        if src[i] == quote:
            return i + 1
        i += 1
    return i


def _block(src: str, open_index: int) -> str:
    """``src[open_index:]`` up to the brace closing the one at ``open_index`` (strings and comments skipped)."""
    assert src[open_index] == "{", src[open_index:open_index + 40]
    depth, i = 0, open_index
    while i < len(src):
        ch = src[i]
        if ch == '"':
            i = _skip_literal(src, i, '"')
            continue
        if src.startswith("//", i):
            i = src.find("\n", i)
            i = len(src) if i < 0 else i
            continue
        if src.startswith("/*", i):
            end = src.find("*/", i + 2)
            i = len(src) if end < 0 else end + 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[open_index:i + 1]
        i += 1
    raise AssertionError("unbalanced block")


def _function_body(src: str, keyword: str, name: str) -> str:
    """Body of ``fun name(`` (Kotlin) / ``func name(`` (Swift); "" for an expression body or none."""
    match = re.search(rf"\b{keyword} {re.escape(name)}\(", src)
    if match is None:
        return ""
    depth, i = 0, match.end() - 1
    while i < len(src):  # the parameter list (it may hold closures with braces in Swift)
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    rest = src[i + 1:]
    brace, equals = rest.find("{"), rest.find("=")
    if brace < 0 or (0 <= equals < brace and "\n" not in rest[:equals]):
        return ""
    return _block(src, i + 1 + brace)


def _without_comments(src: str) -> str:
    """``src`` without ``//`` and ``/* */`` comments (string literals kept)."""
    out, i = [], 0
    while i < len(src):
        if src[i] == '"':
            end = _skip_literal(src, i, '"')
            out.append(src[i:end])
            i = end
        elif src.startswith("//", i):
            end = src.find("\n", i)
            i = len(src) if end < 0 else end
        elif src.startswith("/*", i):
            end = src.find("*/", i + 2)
            i = len(src) if end < 0 else end + 2
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def _code_lines(text: str) -> str:
    """``text`` without whole-line comments (``//`` / ``///``)."""
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("//"))


def _switch_groups(body: str, case_re: str) -> list[tuple[set, str]]:
    """``[(labels, body)]`` of a ``switch``: consecutive case lines share the code that follows them."""
    groups: list[tuple[set, str]] = []
    labels: set = set()
    code: list[str] = []
    for line in body.splitlines():
        found = re.match(case_re, line)
        if found or re.match(r"^\s*default\s*:", line):
            if code and labels:
                groups.append((labels, "\n".join(code)))
                labels, code = set(), []
            elif code:
                code = []
            if found:
                labels.add(found.group(1))
            continue
        if labels:
            code.append(line)
    if labels and code:
        groups.append((labels, "\n".join(code)))
    return groups


def _dart_slice(dart: str, start: str, end: str) -> str:
    return dart[dart.index(start):dart.index(end, dart.index(start))]


# ============================================================================ the app's calls
def _is_web_call(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "call"
            and bool(node.args) and isinstance(node.args[0], ast.Constant)
            and str(node.args[0].value).startswith("web_"))


def _get_keys(func: ast.AST, variables: set) -> set:
    """String keys of ``<variable>.get("key")`` calls inside ``func``."""
    return {node.args[0].value for node in ast.walk(func)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name) and node.func.value.id in variables
            and node.args and isinstance(node.args[0], ast.Constant)}


def _app_calls() -> list[dict]:
    """Every ``<native>.call("web_x", ...)`` of the driver: positional count, keywords (``default`` is
    NativeBridge's own) and the answer keys the calling function reads."""
    calls, seen = [], set()
    for func in ast.walk(_tree(DRIVER)):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assigned: dict[int, set] = {}  # id(call) -> the variables holding its answer
        for node in ast.walk(func):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value.value if isinstance(node.value, ast.Await) else node.value
                if _is_web_call(value):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    assigned.setdefault(id(value), set()).update(t.id for t in targets if isinstance(t, ast.Name))
        for node in ast.walk(func):
            if _is_web_call(node) and id(node) not in seen:
                seen.add(id(node))
                calls.append({"method": node.args[0].value, "function": func.name, "positional": len(node.args) - 1,
                              "keywords": {k.arg for k in node.keywords if k.arg != "default"},
                              "reads": _get_keys(func, assigned.get(id(node), set()))})
    return calls


APP_CALLS = _app_calls()


def _wrapper_methods() -> dict[str, dict]:
    """``GlossarionNative.web_x`` of the extension: its signature, the ``_web_call`` name, argument keys and
    the answer keys it reads."""
    tree = _tree(EXT_PY / "native.py")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GlossarionNative")
    found = {}
    for func in cls.body:
        if not (isinstance(func, ast.AsyncFunctionDef) and func.name.startswith("web_")):
            continue
        dicts = {t.id: node.value for node in ast.walk(func) if isinstance(node, ast.Assign)
                 for t in node.targets if isinstance(t, ast.Name) and isinstance(node.value, ast.Dict)}
        sent, keys = [], set()
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_web_call":
                sent.append(node.args[0].value)
                payload = node.args[1]
                if isinstance(payload, ast.Name):
                    payload = dicts[payload.id]
                assert isinstance(payload, ast.Dict), (func.name, ast.dump(payload)[:80])
                keys |= {k.value for k in payload.keys}
        reads = {node.args[0].value for node in ast.walk(func)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == "result"
                 and node.args and isinstance(node.args[0], ast.Constant)}
        found[func.name] = {"args": func.args, "sent": sent, "keys": keys, "reads": reads}
    return found


WRAPPER = _wrapper_methods()


# ============================================================================ the platform side
def _kotlin_methods() -> dict[str, dict]:
    kotlin = _text(KOTLIN_WEB)
    handle = _function_body(kotlin, "fun", "handle")
    found = {}
    for method, branch in re.findall(r'^\s*"(web_[a-z_]+)" -> (.*)$', handle, re.M):
        text = branch
        for name in set(re.findall(r"\b([a-zA-Z]+)\(", branch)):
            text += "\n" + _function_body(kotlin, "fun", name)
        found[method] = {
            "reads": set(re.findall(r'call\.argument<.*?>\("([a-z_]+)"\)', text)),
            "answers": set(re.findall(r'"([a-z_]+)" to\b', text)),
        }
    return found


def _swift_methods() -> dict[str, dict]:
    swift = _text(SWIFT_WEB)
    handle = _function_body(swift, "func", "handle")
    found = {}
    for labels, branch in _switch_groups(handle, r'^\s*case "([a-z_]+)":\s*$'):
        (method,) = labels
        text = branch
        for name in set(re.findall(r"\b([a-zA-Z]+)\(", branch)):
            text += "\n" + _function_body(swift, "func", name)
        found[method] = {
            "reads": set(re.findall(r'args\["([a-z_]+)"\]', text)),
            "answers": set(re.findall(r'answer\["([a-z_]+)"\]\s*=', text)) | set(re.findall(r'"([a-z_]+)":', text)),
        }
    return found


KOTLIN = _kotlin_methods()
SWIFT = _swift_methods()


# ============================================================================ methods
def test_the_driver_uses_exactly_the_web_methods_dart_forwards():
    dart = _text(DART)
    declared = set(re.findall(r"'([a-z_]+)'", re.search(r"const Set<String> _kWebMethods = <String>\{(.*?)\};",
                                                         dart, re.S).group(1)))
    assert declared == set(WEB_METHODS) == set(WP.WEB_METHODS)
    used = {call["method"] for call in APP_CALLS}
    assert used == set(WEB_METHODS), "the driver calls a method Dart does not forward (or skips one)"
    invoke = _dart_slice(dart, "Future<dynamic> _invokeMethod(", "dynamic _defaultResult(")
    groups = [(labels, body) for labels, body in _switch_groups(invoke, r"^\s*case '([a-z_]+)':\s*$")
              if labels & set(WEB_METHODS)]
    assert len(groups) == 1, "every web method shares one Dart case group"
    labels, body = groups[0]
    code = _code_lines(body)
    assert labels == set(WEB_METHODS)
    # forwarded under its own name (no literal of another method), off-device the typed default
    assert "_platform(name, args)" in code and "return _defaultResult(name);" in code
    assert not re.search(r"_platform\('", code)
    default = _dart_slice(dart, "dynamic _defaultResult(", "Future<dynamic> _platform(")
    assert "_kWebMethods.contains(name)" in default and "'error': 'unavailable'" in default


def test_the_wrapper_sends_each_method_under_its_own_name():
    assert set(WRAPPER) == set(WEB_METHODS)
    for method, info in WRAPPER.items():
        assert info["sent"] == [method], (method, info["sent"])


@pytest.mark.parametrize("call", APP_CALLS, ids=lambda c: f"{c['function']}-{c['method']}")
def test_every_driver_call_fits_the_wrapper_signature(call):
    """``NativeBridge.call(method, *args, **kwargs)`` passes everything but ``default`` on to the wrapper."""
    args = WRAPPER[call["method"]]["args"]
    positional = [a.arg for a in args.posonlyargs + args.args][1:]  # minus self
    required = positional[:len(positional) - len(args.defaults)]
    kwonly = {a.arg for a in args.kwonlyargs}
    kw_required = {a.arg for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is None}
    given = positional[:call["positional"]]
    assert call["positional"] <= len(positional), call
    assert set(required) <= set(given) | call["keywords"], call
    assert call["keywords"] <= (set(positional) - set(given)) | kwonly, call
    assert kw_required <= call["keywords"], call


def test_native_bridge_call_passes_the_script_timeout_through():
    """``web_eval``'s ``timeout=`` must reach the wrapper (the extension waits the script's budget + 5 s);
    NativeBridge.call keeps only ``default`` for itself, and its outer guard outlasts every route script."""
    params = list(inspect.signature(native_mod.NativeBridge.call).parameters)
    assert params == ["self", "method", "args", "default", "kwargs"]
    (eval_call,) = [c for c in APP_CALLS if c["method"] == "web_eval"]
    assert "timeout" in eval_call["keywords"]
    longest_route_script = 30.0  # authchatgpt / authds wait up to 30 000 ms for one script
    assert longest_route_script + WP.EVAL_SLACK < native_mod.DEFAULT_TIMEOUT
    assert "web_eval" not in native_mod.METHOD_TIMEOUTS


@pytest.mark.parametrize("method", WEB_METHODS)
def test_platform_code_reads_the_arguments_python_sends(method):
    sent = WRAPPER[method]["keys"]
    assert KOTLIN[method]["reads"] == sent, (method, KOTLIN[method]["reads"], sent)
    swift = SWIFT[method]["reads"]
    if method == "web_open":
        # WebKit never hands intent: / market: links to other apps: iOS accepts and ignores prevent_links
        assert sent - swift == {"prevent_links"} and swift <= sent
    else:
        assert swift == sent, (method, swift, sent)


@pytest.mark.parametrize("method", WEB_METHODS)
def test_answer_keys_python_reads_are_sent_by_both_platforms(method):
    reads = set(WRAPPER[method]["reads"])
    for call in APP_CALLS:
        if call["method"] == method:
            reads |= call.get("reads", set())
    if method == "web_status":  # the self-test reads the page list of the status answer
        func = next(n for n in ast.walk(_tree(CHECK)) if isinstance(n, ast.FunctionDef) and n.name == "_native_page_ids")
        reads |= {n.args[0].value for n in ast.walk(func) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args
                  and isinstance(n.args[0], ast.Constant)}
        assert {"pages", "page_id"} <= reads
    reads -= COMMON_KEYS
    assert reads <= KOTLIN[method]["answers"], (method, reads - KOTLIN[method]["answers"])
    assert reads <= SWIFT[method]["answers"], (method, reads - SWIFT[method]["answers"])


def test_answer_shapes_and_error_codes():
    kotlin, swift = _text(KOTLIN_WEB), _text(SWIFT_WEB)
    for source in (kotlin, swift):  # {ok, error, message} on both platforms
        assert re.search(r'"ok" to true, "error" to null, "message" to null', source) or \
            '["ok": true, "error": NSNull(), "message": NSNull()]' in source
        assert re.search(r'"ok" to false, "error" to code, "message" to message', source) or \
            '["ok": false, "error": code, "message": message]' in source
    kotlin_codes = set(re.findall(r'const val ERR_[A-Z_]+ = "([a-z_]+)"', kotlin))
    swift_codes = set(re.findall(r'static let err[A-Za-z]+ = "([a-z_]+)"', swift))
    python_codes = {e.value for e in WP.WebPageError}
    assert kotlin_codes == swift_codes == python_codes - {"timeout"}
    driver_codes = {nw._GONE, nw._NO_PAGE, nw._EXISTS, nw._NATIVE_ERROR}
    assert driver_codes <= kotlin_codes and nw._TIMEOUT == WP.WebPageError.TIMEOUT.value
    assert "result.error(" not in _without_comments(kotlin) and "FlutterError(" not in _without_comments(swift)


def test_capability_handshake_keys():
    kotlin, swift = _text(KOTLIN_WEB), _text(SWIFT_WEB)
    produced_kotlin = set(re.findall(r'"([a-z_]+)" to', _function_body(kotlin, "fun", "computeCapability")))
    produced_swift = set(re.findall(r'"([a-z_]+)":', _function_body(swift, "func", "capability")))
    assert produced_kotlin == produced_swift == {"version", "max_pages", "provider"}
    assert 'info["web_pages"] = it' in _text(KOTLIN_PLUGIN) and 'info["web_pages"] = webPages.capability()' in _text(SWIFT_PLUGIN)
    platform_info = _dart_slice(_text(DART), "Future<Map<String, dynamic>> _platformInfo()", "Future<void> _clearShared(")
    assert "info.addAll(_stringMap(native));" in platform_info  # the platform map reaches Python whole
    reads = set()
    for func in ast.walk(_tree(EXT_PY / "web_pages.py")):
        if isinstance(func, ast.FunctionDef) and func.name == "capability":
            reads = {n.args[0].value for n in ast.walk(func) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args
                     and isinstance(n.args[0], ast.Constant)}
    assert {"native", "web_pages", "version", "max_pages"} <= reads
    install = next(n for n in ast.walk(_tree(DRIVER)) if isinstance(n, ast.AsyncFunctionDef) and n.name == "install")
    driver_reads = {n.args[0].value for n in ast.walk(install) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args
                    and isinstance(n.args[0], ast.Constant)}
    assert driver_reads <= produced_kotlin


# ============================================================================ the event
def _web_page_event_fields() -> set:
    tree = _tree(EXT_PY / "types.py")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WebPageEvent")
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}


def test_every_platform_event_reaches_a_python_handler():
    """Kotlin / Swift ``invokeMethod("x")`` -> Dart ``case 'x':`` -> ``triggerEvent('x')`` -> the
    ``on_x`` field Flet checks -> ``NativeBridge`` event ``x``: for every event, ``web`` included."""
    dart = _text(DART)
    platform_call = _dart_slice(dart, "Future<dynamic> _onPlatformCall(", "void _onTaskData(")
    dart_cases = set(re.findall(r"case '([a-z_]+)':", platform_call))
    sent = set()
    for path in (KOTLIN_PLUGIN, KOTLIN_WEB, SWIFT_PLUGIN, SWIFT_WEB):
        sent |= set(re.findall(r'invokeMethod\("([a-z_]+)"', _text(path)))
    assert "web" in sent and sent <= dart_cases, sent - dart_cases
    triggered = set(re.findall(r"triggerEvent\('([a-z_]+)'", dart))
    tree = _tree(EXT_PY / "native.py")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GlossarionNative")
    handlers = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and n.target.id.startswith("on_")}
    assert {f"on_{name}" for name in triggered} == handlers
    assert triggered == set(native_mod.EVENTS)
    web_group = [body for labels, body in _switch_groups(platform_call, r"^\s*case '([a-z_]+)':\s*$") if labels == {"web"}]
    assert len(web_group) == 1 and "control.triggerEvent('web', _stringMap(args));" in web_group[0]
    on_web = next(n for n in cls.body if isinstance(n, ast.AnnAssign) and n.target.id == "on_web")
    assert "WebPageEvent" in ast.unparse(on_web.annotation)
    driver = _text(DRIVER)
    assert 'add_listener("web", self.on_web_event)' in driver and 'remove_listener("web", self.on_web_event)' in driver


def test_the_driver_reads_only_event_keys_the_dataclass_carries():
    fields = _web_page_event_fields()
    assert fields == set(WP.WEB_EVENT_KEYS)
    tree = _tree(DRIVER)
    reads, kinds = set(), set()
    for func in ast.walk(tree):
        if isinstance(func, ast.FunctionDef) and func.name in ("on_native_event", "on_web_event"):
            for node in ast.walk(func):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "payload"):
                    reads.add(node.args[0].value)
                if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "kind"
                        and isinstance(node.comparators[0], ast.Constant)):
                    kinds.add(node.comparators[0].value)
    assert reads <= fields, reads - fields
    kotlin_types = set(re.findall(r'const val EVENT_[A-Z_]+ = "([a-z_]+)"', _text(KOTLIN_WEB)))
    swift_types = set(re.findall(r'static let event[A-Za-z]+ = "([a-z_]+)"', _text(SWIFT_WEB)))
    assert kinds == kotlin_types == swift_types == set(WP.WEB_EVENT_TYPES)


def test_an_event_through_the_real_bridge_reaches_its_page():
    """The keys travel as a flet event dataclass; ``NativeBridge`` hands them to the driver as a dict."""
    import dataclasses

    fields = sorted(_web_page_event_fields())
    event_cls = dataclasses.make_dataclass("WebPageEvent", [("name", str, "web"), ("data", object, None),
                                                            ("control", object, None)]
                                           + [(f, object, None) for f in fields])
    seen = []
    bridge = native_mod.NativeBridge(native=native_mod.NativeStub("host"))
    bridge.add_listener("web", seen.append)
    asyncio.run(bridge.native.on_web(event_cls(type="finished", page_id="p1", url="https://example.com/",
                                               ok=False, error="net::ERR", code=-2, main_frame=True, crashed=None)))
    (payload,) = seen
    assert payload["page_id"] == "p1" and payload["type"] == "finished" and payload["ok"] is False
    assert set(fields) <= set(payload)


# ============================================================================ CI runs the check
def test_the_native_webview_check_is_in_the_smoke_suite():
    names = [name for name, _ in selftest.SUITES["smoke"]]
    assert "native_webview" in names and names.index("native_webview") == names.index("thread_stack") + 1
    assert dict(selftest.SUITES["smoke"])["native_webview"] is selftest.check_native_webview


def test_a_failing_native_webview_check_fails_the_device_smoke_suite(monkeypatch, tmp_path):
    """On a device (strict) a missing native driver turns the smoke suite's marker line into FAIL, which
    both CI device jobs treat as a failed run."""
    monkeypatch.setattr(browser_driver, "get_driver", lambda: None)  # whatever another test left registered
    paths = types.SimpleNamespace(platform="android", is_device=True, logs=tmp_path, home=tmp_path, data=tmp_path)
    monkeypatch.setattr(selftest.rb, "get_state", lambda: types.SimpleNamespace(paths=paths))
    monkeypatch.setattr(webview_check, "DRIVER_WAIT_SECONDS", 0.0)
    monkeypatch.delenv("GLOSSARION_SIMULATE_PLATFORM", raising=False)
    monkeypatch.delenv(nw.ENV_SWITCH, raising=False)
    result = selftest.run_selftest("smoke", only={"native_webview"}, emit=False, write_report=False)
    assert result["strict"] is True and result["ok"] is False and result["failed"] == 1
    assert "native in-app browser is not installed" in result["checks"][0]["error"]
    line = selftest.summary_line(result)
    assert line.startswith("GLOSSARION_SELFTEST FAIL ") and '"name":"native_webview"' in line


def test_android_emulator_smoke_blocks_on_the_smoke_suite():
    script = _text(ANDROID_SMOKE)
    assert 'SELFTEST_URL="${SELFTEST_URL:-glossarion://app/__selftest__?suite=smoke}"' in script
    assert re.search(r'^run_suite smoke "\$SELFTEST_URL"', script, re.M)
    assert "find_marker 'GLOSSARION_SELFTEST FAIL'" in script
    if not BUILD_MOBILE.exists():
        pytest.skip("no .github in this tree")
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(_text(BUILD_MOBILE))
    job = workflow["jobs"]["android-smoke"]
    assert "continue-on-error" not in job
    (step,) = [s for s in job["steps"] if "android_smoke.sh" in str(s.get("with", {}).get("script", ""))
               and "RUN_LLAMA=1" not in str(s["with"]["script"])]
    assert "continue-on-error" not in step and "RUN_E2E=0" not in step["with"]["script"]


def test_ios_simulator_smoke_is_required_and_runs_the_smoke_suite():
    if not BUILD_MOBILE.exists():
        pytest.skip("no .github in this tree")
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(_text(BUILD_MOBILE))
    assert workflow["env"]["IOS_SIMULATOR_SMOKE_POLICY"] == "required"
    job = workflow["jobs"]["ios-simulator-smoke"]
    assert job["env"]["SELFTEST_SUITE"] == "smoke"
    run = "\n".join(str(step.get("run", "")) for step in job["steps"])
    assert 'SIMCTL_CHILD_GLOSSARION_CI_SELFTEST="$SELFTEST_SUITE"' in run
    assert "has_marker 'GLOSSARION_SELFTEST FAIL'" in run and "wait_for 'GLOSSARION_SELFTEST PASS'" in run
    assert selftest.rb.CI_SELFTEST_ENV == "GLOSSARION_CI_SELFTEST"


# ============================================================================ the churn on the real shell
def _load_test_module(file_name: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TB = _load_test_module("test_bootstrap.py", "_hotfix3_contract_tb_helpers")
storage = _TB.storage
app_env = _TB.app_env
needs_flet_app = pytest.mark.skipif(importlib.util.find_spec("flet") is None or importlib.util.find_spec("msgpack") is None,
                                    reason="flet/msgpack not installed")


@needs_flet_app
@pytest.mark.parametrize("where", ["home", "welcome", "logs"])
def test_screen_churn_pushes_one_view_over_any_screen_of_the_real_app(app_env, where):
    """The self-test's screen change on the real App / AppShell: wherever it starts (the chat home; the
    first-run Welcome a fresh CI install shows; Settings › Logs & diagnostics, where "Run self-test" lives)
    it pushes exactly one View, then the same screens are back on top."""
    from glossarion_mobile.ui.router import build_route

    tf = _load_test_module("test_ui_foundations.py", "_hotfix3_contract_tf_helpers")

    async def scenario():
        _m, conn, session, page, app = await tf._start("android", first_run=(where == "welcome"))
        try:
            if where == "welcome":
                assert await tf._wait(lambda: tf._routes(page)[-1] == "/welcome")
            elif where == "logs":
                await app.navigate(build_route("settings.logs"))
                assert tf._routes(page)[-1] == "/settings/logs"
            routes, entries = tf._routes(page), list(app.shell.stack)
            report = await webview_check.screen_churn(app, settle=0)
            detail = webview_check._check_churn(report)
            assert detail["proved_view_change"] is True and not report["restored"], report
            assert report["views_pushed"] == report["views_before"] + 1 and report["after"] == report["before"]
            assert await tf._wait(lambda: tf._routes(page) == routes), tf._routes(page)
            assert list(app.shell.stack) == entries  # the very same screens (Logs keeps its self-test run)
        finally:
            jobs = getattr(app, "jobs", None)
            if jobs is not None:
                jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())


# ============================================================================ Accounts › Experimental
def test_accounts_experimental_rows_show_the_driver_s_limitations(monkeypatch):
    pytest.importorskip("flet")
    from glossarion_mobile.ui.components.reason_chip import tile_reason_chip
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens import accounts

    screen = accounts.AccountsScreen(parse_route("/settings/accounts"), oauth=None)
    monkeypatch.setattr(wb, "availability", lambda: (True, ""))
    for name, text, other in ((nw.NATIVE_NAME, wb.NATIVE_LIMITATIONS, wb.LIMITATIONS),
                              ("flet-webview", wb.LIMITATIONS, wb.NATIVE_LIMITATIONS)):
        monkeypatch.setattr(wb, "current", lambda name=name: types.SimpleNamespace(name=name, registered=True))
        tile = screen._experimental_tile()
        assert {tile_reason_chip(row).detail for row in tile.controls} == {text}
        sheet = screen.show_experimental("AuthND (NVIDIA Build)", accounts.EXPERIMENTAL_ACCOUNTS[0][2])
        assert sheet.body.endswith(text) and other not in sheet.body
    assert "new in-app pages open there too" in wb.NATIVE_LIMITATIONS
    assert "waits until you open the app again" in wb.LIMITATIONS


@pytest.mark.parametrize("path", [Path(__file__), APP_DIR / "glossarion_mobile" / "ui" / "screens" / "accounts.py"],
                         ids=lambda p: p.name)
def test_uniform_line_endings_and_python_310_syntax(path):
    data = path.read_bytes()
    assert data.count(b"\r\n") in (0, data.count(b"\n")), path
    ast.parse(data.decode("utf-8"), feature_version=(3, 10))
