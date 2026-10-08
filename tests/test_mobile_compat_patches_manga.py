"""Mobile-compat patches P17, P27, P28 and P29 (bubble_detector.py, local_inpainter.py).

Desktop (no GLOSSARION_* env) must behave exactly as before:
  * RT-DETR ONNX stays C++-only (P27 is mobile / RTDETR_ONNX_ALLOW_PYTHON=1 only),
  * a missing huggingface_hub keeps its "not installed" error (P28 is mobile only),
  * the LocalInpainter worker process is unchanged (P17),
  * torch-less loads are still refused and ONNX_AVAILABLE still needs 'onnx' (P29).

AST checks pin the gates. Runtime checks load fresh copies of the two modules
under a simulated mobile or desktop environment (torch, huggingface_hub, the
C++ backend and the 'onnx' package blocked through sys.modules) and run tiny
synthetic ONNX models through the real onnxruntime code paths. They skip when
onnx/onnxruntime/cv2 are not installed. Downloads go to a local HTTP server
through HF_ENDPOINT; nothing touches the network or src/config.json.
"""

import ast
import http.server
import importlib.util
import os
import sys
import threading
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
BUBBLE = "bubble_detector.py"
INPAINT = "local_inpainter.py"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _tree(name):
    return ast.parse((SRC / name).read_text(encoding="utf-8"), filename=name)


def _parents(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _func(tree, name, cls=None):
    scope = tree
    if cls is not None:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    for node in ast.walk(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _call_name(call):
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and _call_name(n) == name]


def _calls_attr(node, owner, attr):
    found = []
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr
                and isinstance(n.func.value, ast.Name) and n.func.value.id == owner):
            found.append(n)
    return found


def _guarded_by(node, parents, predicate):
    """True when an enclosing ``if`` has a test for which predicate(test) holds."""
    child, cur = node, parents.get(node)
    while cur is not None:
        if isinstance(cur, ast.If) and child is not cur.test and predicate(cur.test):
            return True
        child, cur = cur, parents.get(cur)
    return False


def _strings(node):
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _same_expr(node, source):
    """Structural comparison (ast.unparse spacing/parentheses differ across Python versions)."""
    return ast.dump(node) == ast.dump(ast.parse(source, mode="eval").body)


def _imports_module(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name == name for a in node.names):
            return True
    return False


# ---------------------------------------------------------------------------
# AST: gates and desktop defaults
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [BUBBLE, INPAINT])
def test_patched_modules_parse_on_python_310(name):
    ast.parse((SRC / name).read_text(encoding="utf-8"), feature_version=(3, 10))


@pytest.mark.parametrize("name", [BUBBLE, INPAINT])
def test_patched_modules_import_mobile_runtime(name):
    assert _imports_module(_tree(name), "mobile_runtime")


def test_rtdetr_python_fallback_gate_is_mobile_or_explicit_opt_in():
    fn = _func(_tree(BUBBLE), "_rtdetr_python_onnx_allowed")
    assert _calls_attr(fn, "mobile_runtime", "is_mobile")
    assert "RTDETR_ONNX_ALLOW_PYTHON" in _strings(fn)
    compares = [n for n in ast.walk(fn) if isinstance(n, ast.Compare)]
    assert any(isinstance(c.comparators[0], ast.Constant) and c.comparators[0].value == "1" for c in compares)


def test_rtdetr_python_session_is_only_created_behind_the_gate():
    tree = _tree(BUBBLE)
    parents = _parents(tree)
    loader = _func(tree, "load_rtdetr_onnx_model", cls="BubbleDetector")
    is_gate = lambda test: bool(_calls(test, "_rtdetr_python_onnx_allowed"))

    attach_calls = _calls(loader, "_attach_rtdetr_onnx_python_session")
    # import failure, constructor failure, load_model() == False
    assert len(attach_calls) == 3
    for call in attach_calls:
        assert _guarded_by(call, parents, is_gate), ast.dump(call)

    # the shared-session reuse shortcut is gated as well
    reuse = [n for n in ast.walk(loader) if isinstance(n, ast.If) and "_rtdetr_onnx_shared_session" in ast.dump(n.test)]
    assert reuse and all(is_gate(n.test) for n in reuse)

    # the loader itself never builds a Python session; only the helper does, on CPU
    assert not _calls(loader, "InferenceSession")
    helper = _func(tree, "_attach_rtdetr_onnx_python_session", cls="BubbleDetector")
    assert len(_calls(helper, "InferenceSession")) == 1
    assert "CPUExecutionProvider" in _strings(helper)

    # desktop error branches are still there, verbatim
    messages = _strings(loader)
    assert "C++ backend load failed - RT-DETR unavailable (Python fallback disabled)" in messages
    joined = "\n".join(
        ast.unparse(n) for n in ast.walk(loader) if isinstance(n, ast.JoinedStr)
    )
    assert "C++ backend not importable - RT-DETR unavailable: " in joined
    assert "Failed to load RT-DETR ONNX: " in joined


def test_hf_download_fallback_is_mobile_only():
    tree = _tree(BUBBLE)
    parents = _parents(tree)
    fn = _func(tree, "_hf_download_fn")
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    urllib_returns = [r for r in returns if isinstance(r.value, ast.Name) and r.value.id == "hf_urllib_download"]
    assert len(urllib_returns) == 1
    assert _guarded_by(urllib_returns[0], parents, lambda t: bool(_calls_attr(t, "mobile_runtime", "is_mobile")))
    # first statement prefers the real hf_hub_download
    first = fn.body[1] if isinstance(fn.body[0], ast.Expr) else fn.body[0]
    assert isinstance(first, ast.If) and "hf_hub_download" in ast.unparse(first.test)

    for method in ("load_model", "load_rtdetr_onnx_model"):
        body = _func(tree, method, cls="BubbleDetector")
        assert not _calls(body, "hf_hub_download"), method
        assert _calls(body, "_hf_download_fn"), method


def test_hf_urllib_download_uses_part_file_and_size_check():
    fn = _func(_tree(BUBBLE), "hf_urllib_download")
    assert ".part" in _strings(fn)
    assert _calls_attr(fn, "os", "replace")
    assert "Content-Length" in _strings(fn)


def test_inpainter_worker_process_gate_is_processes_available():
    init = _func(_tree(INPAINT), "__init__", cls="LocalInpainter")
    assigns = [
        n for n in ast.walk(init)
        if isinstance(n, ast.Assign) and any(ast.unparse(t) == "self._mp_enabled" for t in n.targets)
    ]
    assert len(assigns) == 1
    assert _same_expr(assigns[0].value, "bool(enable_worker_process) and mobile_runtime.processes_available()")


def test_inpainter_onnx_available_keeps_desktop_requirement():
    tree = _tree(INPAINT)
    tries = [n for n in tree.body if isinstance(n, ast.Try)]
    both = [
        t for t in tries
        if {"onnx", "onnxruntime"} <= {a.name for s in t.body if isinstance(s, ast.Import) for a in s.names}
    ]
    assert len(both) == 1, "desktop ONNX_AVAILABLE must still import onnx and onnxruntime together"
    mobile_ifs = [n for n in tree.body if isinstance(n, ast.If) and "ONNX_AVAILABLE" in ast.dump(n.test)]
    assert len(mobile_ifs) == 1
    assert _same_expr(mobile_ifs[0].test, "not ONNX_AVAILABLE and mobile_runtime.is_mobile()")


def test_inpainter_torchless_onnx_is_mobile_only():
    tree = _tree(INPAINT)
    helper = _func(tree, "_onnx_inpaint_without_torch", cls="LocalInpainter")
    first = helper.body[1] if isinstance(helper.body[0], ast.Expr) else helper.body[0]
    assert isinstance(first, ast.If)
    assert _same_expr(first.test, "not (ONNX_AVAILABLE and mobile_runtime.is_mobile())")
    assert isinstance(first.body[0], ast.Return) and _same_expr(first.body[0].value, "False")

    loader = _func(tree, "load_model", cls="LocalInpainter")
    gate = [
        n for n in ast.walk(loader)
        if isinstance(n, ast.Assign) and any(ast.unparse(t) == "onnx_without_torch" for t in n.targets)
    ]
    assert len(gate) == 1 and _calls(gate[0].value, "_onnx_inpaint_without_torch")
    messages = _strings(loader)
    assert "PyTorch not available in this build" in messages
    assert "PyTorch modules not properly loaded" in messages
    def first_message(if_node):
        first = if_node.body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Call) and first.value.args
                and isinstance(first.value.args[0], ast.Constant)):
            return first.value.args[0].value
        return None

    refusals = {
        first_message(n): n.test for n in ast.walk(loader)
        if isinstance(n, ast.If)
        and first_message(n) in ("PyTorch not available in this build", "PyTorch modules not properly loaded")
    }
    assert _same_expr(refusals["PyTorch not available in this build"],
                      "not TORCH_AVAILABLE and not onnx_without_torch")
    assert _same_expr(refusals["PyTorch modules not properly loaded"],
                      "(torch is None or nn is None) and not onnx_without_torch")
    # the post-ONNX refusal for non-ONNX files when torch is missing
    late = [n for n in ast.walk(loader) if isinstance(n, ast.If) and _same_expr(n.test, "onnx_without_torch")]
    assert len(late) == 1 and isinstance(late[0].body[-1], ast.Return)


def test_inpainter_download_fallback_is_mobile_only():
    tree = _tree(INPAINT)
    parents = _parents(tree)
    fn = _func(tree, "download_model")
    imports = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.ImportFrom) and n.module == "bubble_detector"
    ]
    assert len(imports) == 1
    assert _guarded_by(
        imports[0], parents,
        lambda t: bool(_calls_attr(t, "mobile_runtime", "is_mobile")) and "ImportError" in ast.unparse(t),
    )


# ---------------------------------------------------------------------------
# Runtime: fresh module copies under a simulated environment
# ---------------------------------------------------------------------------

# The mobile dependency set: onnxruntime without the 'onnx' package. The synthetic
# models are embedded below so these tests also run in that environment (3.13).
_HAVE_RUNTIME = all(importlib.util.find_spec(m) is not None for m in ("onnxruntime", "cv2", "numpy", "PIL"))
_HAVE_ONNX_PACKAGE = importlib.util.find_spec("onnx") is not None
runtime = pytest.mark.skipif(not _HAVE_RUNTIME, reason="onnxruntime, cv2, numpy and PIL are needed")
# bubble_detector imports numpy and cv2 at module level; a test that builds no ONNX session (the
# manga_models hand-off of hf_urllib_download) needs only those, so it also runs where onnxruntime
# is not installed (CI's python-app job).
_HAVE_BACKEND_IMPORTS = all(importlib.util.find_spec(m) is not None for m in ("cv2", "numpy"))
backend = pytest.mark.skipif(not _HAVE_BACKEND_IMPORTS, reason="numpy and cv2 are needed to import bubble_detector")

_ENV_KEYS = (
    "GLOSSARION_MOBILE", "GLOSSARION_NO_PROCESSES", "FLET_PLATFORM", "RTDETR_ONNX_ALLOW_PYTHON",
    "RTDETR_ONNX_FILENAME", "HF_ENDPOINT", "CUDA_LAUNCH_BLOCKING", "TORCH_USE_CUDA_DSA",
    "ORT_DISABLE_MEMORY_ARENA", "GRACEFUL_STOP", "ENABLE_ONNX_CPP_BACKEND", "ONNX_QUANTIZE",
    "MODEL_QUANTIZE", "HD_STRATEGY", "PARALLEL_PANEL_TRANSLATION_ENABLED", "DML_MAX_CONCURRENT",
    "BUBBLE_CACHE_DIR", "ONNX_CACHE_DIR", "MODEL_CACHE_DIR", "GLOSSARION_CUDA_DEBUG",
)
# Absent on the phone; blocked so the fresh copies take their fallback branches quickly.
_MOBILE_ABSENT = ("torch", "ultralytics", "transformers", "huggingface_hub", "onnx_cpp_backend",
                  "onnxruntime_extensions", "onnx")


def _rtdetr_model_bytes():
    from onnx import TensorProto, helper

    def const(name, dtype, dims, vals):
        return helper.make_node("Constant", [], [name], value=helper.make_tensor(name + "_v", dtype, dims, vals))

    graph = helper.make_graph(
        [
            const("labels", TensorProto.INT64, [1, 3], [0, 1, 2]),
            const("boxes", TensorProto.FLOAT, [1, 3, 4],
                  [10, 20, 110, 220, 300, 40, 400, 90, 50, 500, 150, 560]),
            const("scores", TensorProto.FLOAT, [1, 3], [0.9, 0.8, 0.1]),
        ],
        "fake_rtdetr",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640]),
         helper.make_tensor_value_info("orig_target_sizes", TensorProto.INT64, [1, 2])],
        [helper.make_tensor_value_info("labels", TensorProto.INT64, [1, 3]),
         helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 3, 4]),
         helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


def _inpaint_model_bytes():
    """output = image + mask: masked pixels (zeroed by the anime preprocessing) become white."""
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Add", ["image", "mask"], ["output"])],
        "fake_inpaint",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "H", "W"]),
         helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, "H", "W"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "H", "W"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


# base64 of _rtdetr_model_bytes() / _inpaint_model_bytes() (checked by the test below)
_RTDETR_MODEL_B64 = (
    "CAg6ggMKNRIGbGFiZWxzIghDb25zdGFudCohCgV2YWx1ZSoVCAEIAxAHOgMAAQJCCGxhYmVsc192oAEECmISBWJveGVz"
    "IghDb25zdGFudCpPCgV2YWx1ZSpDCAEIAwgEEAEiMAAAIEEAAKBBAADcQgAAXEMAAJZDAAAgQgAAyEMAALRCAABIQgAA"
    "+kMAABZDAAAMREIHYm94ZXNfdqABBAo+EgZzY29yZXMiCENvbnN0YW50KioKBXZhbHVlKh4IAQgDEAEiDGZmZj/NzEw/"
    "zczMPUIIc2NvcmVzX3agAQQSC2Zha2VfcnRkZXRyWiIKBmltYWdlcxIYChYIARISCgIIAQoCCAMKAwiABQoDCIAFWiMK"
    "EW9yaWdfdGFyZ2V0X3NpemVzEg4KDAgHEggKAggBCgIIAmIYCgZsYWJlbHMSDgoMCAcSCAoCCAEKAggDYhsKBWJveGVz"
    "EhIKEAgBEgwKAggBCgIIAwoCCARiGAoGc2NvcmVzEg4KDAgBEggKAggBCgIIA0IECgAQDQ=="
)
_INPAINT_MODEL_B64 = (
    "CAg6kwEKGgoFaW1hZ2UKBG1hc2sSBm91dHB1dCIDQWRkEgxmYWtlX2lucGFpbnRaIQoFaW1hZ2USGAoWCAESEgoCCAEK"
    "AggDCgMSAUgKAxIBV1ogCgRtYXNrEhgKFggBEhIKAggBCgIIAQoDEgFICgMSAVdiIgoGb3V0cHV0EhgKFggBEhIKAggB"
    "CgIIAwoDEgFICgMSAVdCBAoAEA0="
)


@pytest.fixture(scope="module")
def models():
    import base64
    return {"rtdetr": base64.b64decode(_RTDETR_MODEL_B64), "inpaint": base64.b64decode(_INPAINT_MODEL_B64)}


@pytest.mark.skipif(not _HAVE_ONNX_PACKAGE, reason="the onnx package builds the models")
def test_embedded_models_match_their_builders(models):
    assert models["rtdetr"] == _rtdetr_model_bytes()
    assert models["inpaint"] == _inpaint_model_bytes()


def test_host_smoke_plants_the_same_synthetic_rtdetr_export():
    """src/mobile/tools/host_smoke.py's manga_pipeline check embeds this export (no onnx on the phone)."""
    tree = ast.parse((SRC / "mobile" / "tools" / "host_smoke.py").read_text(encoding="utf-8"))
    values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and any(getattr(t, "id", None) == "SYNTHETIC_RTDETR_ONNX_B64" for t in node.targets)]
    assert values == [_RTDETR_MODEL_B64]


_MISSING = object()


class _Sandbox:
    def __init__(self, monkeypatch, tmp_path):
        self.mp = monkeypatch
        self.tmp = tmp_path
        self._saved = {}  # sys.modules entries replaced by block(); restored by close()
        self._before = set(sys.modules)

    def _set_module(self, name, value):
        self._saved.setdefault(name, sys.modules.get(name, _MISSING))
        if value is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value

    def close(self):
        for name, original in self._saved.items():
            if original is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        # A blocked package that was first imported for real inside the test (after
        # being unblocked) must not leave orphaned submodules behind.
        for key in list(sys.modules):
            if key not in self._before and key.split(".")[0] in _MOBILE_ABSENT:
                sys.modules.pop(key, None)

    def env(self, *, mobile):
        if mobile:
            self.mp.setenv("GLOSSARION_MOBILE", "1")
            self.mp.setenv("GLOSSARION_NO_PROCESSES", "1")
        else:
            self.mp.delenv("GLOSSARION_MOBILE", raising=False)
            self.mp.delenv("GLOSSARION_NO_PROCESSES", raising=False)

    def load(self, filename, *, mobile, absent=_MOBILE_ABSENT):
        self.env(mobile=mobile)
        for name in _MOBILE_ABSENT:
            if name in absent:
                self._set_module(name, None)  # import raises ImportError, as on the phone
            elif name in self._saved:
                self._set_module(name, self._saved[name])  # unblock what an earlier load blocked
        if filename == INPAINT:
            # LocalInpainter imports bubble_detector by name (detector, download fallback)
            self.load(BUBBLE, mobile=mobile, absent=absent)
        stem = filename[:-3]
        spec = importlib.util.spec_from_file_location(f"_under_test_{stem}_{id(self)}", SRC / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if stem == "bubble_detector":
            # local_inpainter imports bubble_detector by name; never let it import (and
            # cache) the real module under this simulated environment.
            self.mp.setitem(sys.modules, "bubble_detector", module)
        return module

    def write(self, name, data):
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    for key in _ENV_KEYS:
        # setenv records the original value (or its absence) for teardown
        monkeypatch.setenv(key, "x")
        monkeypatch.delenv(key)
    monkeypatch.setenv("BUBBLE_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("ONNX_CACHE_DIR", str(tmp_path / "onnx_cache"))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "inpaint_cache"))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.chdir(tmp_path)
    # _check_stop() imports manga_translator; keep the real (heavy) module out of this test
    stub = types.ModuleType("manga_translator")
    stub.MangaTranslator = type("MangaTranslator", (), {"is_globally_cancelled": staticmethod(lambda: False)})
    monkeypatch.setitem(sys.modules, "manga_translator", stub)
    box = _Sandbox(monkeypatch, tmp_path)
    try:
        yield box
    finally:
        box.close()


class _FileHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (http.server API)
        self.server.hits.append(self.path)
        body = self.server.files.get(self.path)
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(self.server.declared.get(self.path, len(body))))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def hf_server(sandbox):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FileHandler)
    server.files, server.declared, server.hits = {}, {}, []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sandbox.mp.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_address[1]}")
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _fake_hf(bd, sandbox, models, calls):
    def fake(repo_id, filename, cache_dir=None, local_dir=None, **_):
        calls.append((repo_id, filename))
        if filename == "config.json":
            return sandbox.write("models/config.json", b"{}")
        return sandbox.write(f"models/{filename}", models["rtdetr"])
    bd.hf_hub_download = fake
    return fake


def _cpp_stub(sandbox, *, constructor_raises=False, load_ok=False):
    module = types.ModuleType("onnx_cpp_backend")

    class ONNXCppBackend:
        def __init__(self):
            if constructor_raises:
                raise RuntimeError("Could not find ONNX C++ library")

        def load_model(self, path, use_gpu=False):
            return load_ok

    module.ONNXCppBackend = ONNXCppBackend
    sandbox.mp.setitem(sys.modules, "onnx_cpp_backend", module)


EXPECTED_DETECTIONS = {
    "bubbles": [(10, 20, 100, 200)],
    "text_bubbles": [(300, 40, 100, 50)],
    "text_free": [],
}


# ----- P27 / P28: bubble_detector -------------------------------------------

@runtime
def test_rtdetr_desktop_stays_cpp_only_without_backend(sandbox, models, caplog):
    bd = sandbox.load(BUBBLE, mobile=False)
    calls = []
    _fake_hf(bd, sandbox, models, calls)
    assert bd._rtdetr_python_onnx_allowed() is False

    det = bd.BubbleDetector(config_path=str(sandbox.tmp / "cfg.json"))
    with caplog.at_level("ERROR"):
        assert det.load_rtdetr_onnx_model() is False
    assert det.rtdetr_onnx_session is None and det.rtdetr_onnx_loaded is False
    assert bd.BubbleDetector._rtdetr_onnx_shared_session is None
    assert bd.BubbleDetector._rtdetr_onnx_loaded is False
    assert "C++ backend not importable - RT-DETR unavailable" in caplog.text
    # desktop still downloads through hf_hub_download
    assert ("ogkalu/comic-text-and-bubble-detector", "detector.onnx") in calls


@runtime
@pytest.mark.parametrize("constructor_raises", [True, False], ids=["dll-missing", "load-false"])
def test_rtdetr_desktop_cpp_failures_still_fail(sandbox, models, caplog, constructor_raises):
    bd = sandbox.load(BUBBLE, mobile=False)
    _fake_hf(bd, sandbox, models, [])
    _cpp_stub(sandbox, constructor_raises=constructor_raises, load_ok=False)
    det = bd.BubbleDetector(config_path=str(sandbox.tmp / "cfg.json"))
    with caplog.at_level("ERROR"):
        assert det.load_rtdetr_onnx_model() is False
    assert det.rtdetr_onnx_session is None
    assert bd.BubbleDetector._rtdetr_onnx_shared_session is None
    expected = ("Failed to load RT-DETR ONNX: Could not find ONNX C++ library" if constructor_raises
                else "C++ backend load failed - RT-DETR unavailable (Python fallback disabled)")
    assert expected in caplog.text


@runtime
def test_rtdetr_mobile_uses_python_onnxruntime_and_detects(sandbox, models):
    bd = sandbox.load(BUBBLE, mobile=True)
    calls = []
    _fake_hf(bd, sandbox, models, calls)
    assert bd._rtdetr_python_onnx_allowed() is True

    det = bd.BubbleDetector(config_path=str(sandbox.tmp / "cfg.json"))
    assert det.load_rtdetr_onnx_model() is True
    assert det.rtdetr_onnx_loaded is True and det.rtdetr_onnx_session is not None
    cls = bd.BubbleDetector
    assert cls._rtdetr_onnx_shared_session is det.rtdetr_onnx_session
    assert cls._rtdetr_onnx_providers == ["CPUExecutionProvider"]
    assert cls._rtdetr_onnx_use_cpp is False and cls._rtdetr_onnx_loaded is True
    assert cls._rtdetr_onnx_model_key == ("ogkalu/comic-text-and-bubble-detector", "detector.onnx")

    import numpy as np
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    assert det.detect_with_rtdetr_onnx(image=image, confidence=0.3) == EXPECTED_DETECTIONS

    import cv2
    image_path = str(sandbox.tmp / "page.png")
    cv2.imwrite(image_path, image)
    assert det.detect_bubbles(image_path, confidence=0.3, use_rtdetr=True) == [(10, 20, 100, 200), (300, 40, 100, 50)]

    # A second detector attaches to the shared session without downloading again.
    downloads = len(calls)
    other = bd.BubbleDetector(config_path=str(sandbox.tmp / "cfg.json"))
    assert other.load_rtdetr_onnx_model() is True
    assert other.rtdetr_onnx_session is det.rtdetr_onnx_session
    assert len(calls) == downloads

    # force_reload builds a new session
    assert other.load_rtdetr_onnx_model(force_reload=True) is True
    assert other.rtdetr_onnx_session is not det.rtdetr_onnx_session


@runtime
@pytest.mark.parametrize("constructor_raises", [True, False], ids=["dll-missing", "load-false"])
def test_rtdetr_mobile_falls_back_when_cpp_backend_cannot_load(sandbox, models, constructor_raises):
    bd = sandbox.load(BUBBLE, mobile=True)
    _fake_hf(bd, sandbox, models, [])
    _cpp_stub(sandbox, constructor_raises=constructor_raises, load_ok=False)
    det = bd.BubbleDetector(config_path=str(sandbox.tmp / "cfg.json"))
    assert det.load_rtdetr_onnx_model() is True
    assert det.rtdetr_onnx_session is not None
    assert bd.BubbleDetector._rtdetr_onnx_cpp_backend is None


@runtime
def test_rtdetr_python_fallback_opt_in_on_desktop(sandbox, models):
    bd = sandbox.load(BUBBLE, mobile=False)
    sandbox.mp.setenv("RTDETR_ONNX_ALLOW_PYTHON", "1")
    _fake_hf(bd, sandbox, models, [])
    det = bd.BubbleDetector(config_path=str(sandbox.tmp / "cfg.json"))
    assert det.load_rtdetr_onnx_model() is True
    import numpy as np
    assert det.detect_with_rtdetr_onnx(image=np.zeros((50, 50, 3), np.uint8)) == EXPECTED_DETECTIONS


@runtime
def test_hf_download_fn_falls_back_only_on_mobile(sandbox):
    bd = sandbox.load(BUBBLE, mobile=False)
    assert bd.hf_hub_download is None  # huggingface_hub blocked
    assert bd._hf_download_fn() is None  # desktop: existing "install huggingface_hub" error path
    sandbox.env(mobile=True)
    assert bd._hf_download_fn() is bd.hf_urllib_download

    sentinel = object()
    bd.hf_hub_download = sentinel
    assert bd._hf_download_fn() is sentinel
    sandbox.env(mobile=False)
    assert bd._hf_download_fn() is sentinel


@runtime
def test_hf_urllib_download_writes_atomically_and_reuses_cache(sandbox, hf_server):
    bd = sandbox.load(BUBBLE, mobile=True)
    payload = os.urandom(3 * 1024 * 1024 + 17)
    hf_server.files["/owner/repo/resolve/main/sub/model.onnx"] = payload
    progress = []

    local_dir = str(sandbox.tmp / "cache")
    path = bd.hf_urllib_download(repo_id="owner/repo", filename="sub/model.onnx", cache_dir=local_dir,
                                 local_dir=local_dir, local_dir_use_symlinks=False,
                                 progress_callback=lambda *a: progress.append(a))
    assert Path(path) == Path(local_dir) / "sub" / "model.onnx"
    assert Path(path).read_bytes() == payload
    assert not Path(path + ".part").exists()
    assert progress and progress[-1][0] == 100

    hits = len(hf_server.hits)
    assert bd.hf_urllib_download("owner/repo", "sub/model.onnx", local_dir=local_dir) == path
    assert len(hf_server.hits) == hits  # cached copy, no request


@runtime
@pytest.mark.parametrize("case", ["short-body", "missing"])
def test_hf_urllib_download_leaves_nothing_behind_on_failure(sandbox, hf_server, case):
    bd = sandbox.load(BUBBLE, mobile=True)
    if case == "short-body":
        hf_server.files["/owner/repo/resolve/main/model.onnx"] = b"x" * 10
        hf_server.declared["/owner/repo/resolve/main/model.onnx"] = 100
    local_dir = sandbox.tmp / "cache"
    with pytest.raises(Exception):
        bd.hf_urllib_download("owner/repo", "model.onnx", local_dir=str(local_dir))
    assert not (local_dir / "model.onnx").exists()
    assert not (local_dir / "model.onnx.part").exists()


def _pin_payload(sandbox, repo_id, filename, payload, *, revision="main"):
    """Point the manga_models registry entry for (repo, file) at a synthetic payload served by
    hf_server under ``revision`` (the real pins are the Hugging Face LFS objects)."""
    import dataclasses
    import hashlib
    import manga_models

    real = manga_models.find_spec(repo_id, filename)
    assert real is not None, (repo_id, filename)
    pinned = dataclasses.replace(real, revision=revision, size=len(payload),
                                 sha256=hashlib.sha256(payload).hexdigest())
    original = manga_models.find_spec
    sandbox.mp.setattr(manga_models, "find_spec",
                       lambda repo, name: pinned if (repo, name) == (repo_id, filename) else original(repo, name))
    return pinned


@backend
def test_hf_urllib_download_hands_registered_models_to_manga_models(sandbox, hf_server, models):
    """A pinned model (manga_models) is fetched by its download manager: pinned revision,
    size + sha256 verification (a wrong body is rejected and nothing lands), ``.partial``."""
    bd = sandbox.load(BUBBLE, mobile=True)
    repo, name = "ogkalu/comic-text-and-bubble-detector", "detector.onnx"
    spec = _pin_payload(sandbox, repo, name, models["rtdetr"], revision="0123abcd")
    url = f"/{repo}/resolve/0123abcd/{name}"
    local_dir = sandbox.tmp / "pinned"
    hf_server.files[url] = bytes(reversed(models["rtdetr"]))  # same size, wrong bytes
    import manga_models
    with pytest.raises(manga_models.ChecksumMismatch):
        bd.hf_urllib_download(repo, name, local_dir=str(local_dir))
    assert not (local_dir / name).exists() and not (local_dir / (name + ".partial")).exists()
    hf_server.files[url] = models["rtdetr"]
    progress = []
    path = bd.hf_urllib_download(repo_id=repo, filename=name, cache_dir=str(local_dir), local_dir=str(local_dir),
                                 local_dir_use_symlinks=False, progress_callback=lambda *a: progress.append(a))
    assert Path(path) == local_dir / name and Path(path).read_bytes() == models["rtdetr"]
    assert hf_server.hits[-1] == url and progress and progress[-1][0] == 100
    assert manga_models.is_installed(spec, str(local_dir), verify_hash=True)
    hits = len(hf_server.hits)
    assert bd.hf_urllib_download(repo, name, local_dir=str(local_dir)) == path
    assert len(hf_server.hits) == hits  # installed: no request
    # the run's immediate Stop (MangaTranslator's global cancellation) interrupts a download
    sandbox.mp.setattr(sys.modules["manga_translator"].MangaTranslator, "is_globally_cancelled",
                       staticmethod(lambda: True))
    with pytest.raises(manga_models.DownloadCancelled):
        bd.hf_urllib_download(repo, name, local_dir=str(sandbox.tmp / "stopped"))
    assert len(hf_server.hits) == hits
    # an explicit other revision is not the pinned file: the plain urllib path serves it
    hf_server.files[f"/{repo}/resolve/other/{name}"] = b"plain"
    assert Path(bd.hf_urllib_download(repo, name, local_dir=str(sandbox.tmp / "o"), revision="other")
                ).read_bytes() == b"plain"


@runtime
def test_rtdetr_mobile_end_to_end_download_without_huggingface_hub(sandbox, hf_server, models):
    bd = sandbox.load(BUBBLE, mobile=True)
    repo = "/ogkalu/comic-text-and-bubble-detector/resolve/main/"
    hf_server.files[repo + "config.json"] = b"{}"
    hf_server.files[repo + "detector.onnx"] = models["rtdetr"]
    _pin_payload(sandbox, "ogkalu/comic-text-and-bubble-detector", "detector.onnx", models["rtdetr"])
    det = bd.BubbleDetector(config_path=str(sandbox.tmp / "cfg.json"))
    assert det.load_rtdetr_onnx_model() is True
    assert (sandbox.tmp / "models" / "detector.onnx").read_bytes() == models["rtdetr"]
    assert not (sandbox.tmp / "models" / "detector.onnx.partial").exists()
    assert (sandbox.tmp / "models" / "config.json").exists()
    import numpy as np
    assert det.detect_with_rtdetr_onnx(image=np.zeros((64, 64, 3), np.uint8)) == EXPECTED_DETECTIONS


@runtime
def test_rtdetr_desktop_without_huggingface_hub_keeps_error(sandbox, hf_server, models, caplog):
    bd = sandbox.load(BUBBLE, mobile=False)
    hf_server.files["/ogkalu/comic-text-and-bubble-detector/resolve/main/detector.onnx"] = models["rtdetr"]
    det = bd.BubbleDetector(config_path=str(sandbox.tmp / "cfg.json"))
    with caplog.at_level("ERROR"):
        assert det.load_rtdetr_onnx_model() is False
    assert "huggingface_hub required to fetch RT-DETR ONNX" in caplog.text
    assert hf_server.hits == []


# ----- P17 / P29 (+ P28 download): local_inpainter ---------------------------

def _inpainter(module, sandbox, **kwargs):
    module.BUBBLE_DETECTOR_AVAILABLE = False
    return module.LocalInpainter(config_path=str(sandbox.tmp / "inpaint_config.json"), **kwargs)


@runtime
def test_inpainter_onnx_available_desktop_unchanged(sandbox):
    without_onnx = sandbox.load(INPAINT, mobile=False)  # 'onnx' blocked, like the Mac NoCuda build
    assert without_onnx.ONNX_AVAILABLE is False
    with_onnx = sandbox.load(INPAINT, mobile=False, absent=("torch", "huggingface_hub", "onnx_cpp_backend"))
    assert with_onnx.ONNX_AVAILABLE is _HAVE_ONNX_PACKAGE  # desktop needs onnx + onnxruntime


@runtime
def test_inpainter_onnx_available_on_mobile_without_onnx_package(sandbox):
    module = sandbox.load(INPAINT, mobile=True)
    assert module.ONNX_AVAILABLE is True
    assert module.TORCH_AVAILABLE is False


@runtime
@pytest.mark.parametrize("mobile", [False, True], ids=["desktop", "mobile"])
def test_inpainter_worker_process_gate(sandbox, mobile):
    module = sandbox.load(INPAINT, mobile=mobile)
    started = []
    sandbox.mp.setattr(module.LocalInpainter, "_start_worker", lambda self: started.append(True))
    inp = _inpainter(module, sandbox, enable_worker_process=True)
    assert inp._mp_enabled is (not mobile)
    assert started == ([] if mobile else [True])
    off = _inpainter(module, sandbox, enable_worker_process=False)
    assert off._mp_enabled is False


@runtime
def test_inpainter_mobile_loads_onnx_without_torch_and_inpaints(sandbox, models):
    module = sandbox.load(INPAINT, mobile=True)
    model_path = sandbox.write("fake_anime.onnx", models["inpaint"])
    inp = _inpainter(module, sandbox)
    assert inp._mp_enabled is False
    assert inp.load_model("anime_onnx", model_path) is True
    assert inp.use_onnx is True and inp.onnx_session is not None and inp.model_loaded is True

    import numpy as np
    image = np.full((64, 64, 3), 60, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    result = inp.inpaint(image, mask, refinement="fast")
    assert result.shape == image.shape
    assert int(result[20:44, 20:44].min()) == 255  # model output inside the mask
    assert int(result[:8, :8].max()) == 60 and int(result[:8, :8].min()) == 60  # untouched outside


@runtime
def test_inpainter_desktop_without_torch_still_refuses_onnx(sandbox, models, caplog):
    module = sandbox.load(INPAINT, mobile=False, absent=("torch", "huggingface_hub", "onnx_cpp_backend"))
    assert module.ONNX_AVAILABLE is _HAVE_ONNX_PACKAGE and module.TORCH_AVAILABLE is False
    model_path = sandbox.write("fake_anime.onnx", models["inpaint"])
    inp = _inpainter(module, sandbox, enable_worker_process=False)
    with caplog.at_level("WARNING"):
        assert inp.load_model("anime_onnx", model_path) is False
    assert "PyTorch not available in this build" in caplog.text
    assert inp.onnx_session is None and inp.model_loaded is False


@runtime
def test_inpainter_mobile_keeps_refusing_torch_only_models(sandbox, hf_server, caplog):
    module = sandbox.load(INPAINT, mobile=True)
    inp = _inpainter(module, sandbox)
    torch_file = sandbox.write("aot_traced.pt", b"PK\x03\x04 not an onnx file")
    with caplog.at_level("WARNING"):
        assert inp.load_model("aot", torch_file) is False
    assert "PyTorch not available in this build" in caplog.text
    assert inp.model_loaded is False

    # *_onnx method pointed at a torch file: the ONNX download is tried (404 here),
    # then the torch-only loaders are refused instead of crashing on torch=None.
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert inp.load_model("anime_onnx", torch_file) is False
    assert "model is not ONNX" in caplog.text
    assert any("lama-manga-dynamic.onnx" in hit for hit in hf_server.hits)
    assert inp.model_loaded is False


@runtime
def test_inpainter_mobile_downloads_onnx_model_without_huggingface_hub(sandbox, hf_server, models):
    module = sandbox.load(INPAINT, mobile=True)
    hf_server.files["/ogkalu/lama-manga-onnx-dynamic/resolve/main/lama-manga-dynamic.onnx"] = models["inpaint"]
    _pin_payload(sandbox, "ogkalu/lama-manga-onnx-dynamic", "lama-manga-dynamic.onnx", models["inpaint"])
    inp = _inpainter(module, sandbox)
    assert inp.load_model("anime_onnx", "") is True
    cached = sandbox.tmp / "inpaint_cache" / "lama-manga-dynamic.onnx"
    assert not Path(str(cached) + ".partial").exists()
    assert cached.read_bytes() == models["inpaint"]
    assert not Path(str(cached) + ".part").exists()
    assert inp.use_onnx is True and inp.onnx_session is not None


@runtime
def test_inpainter_desktop_download_without_huggingface_hub_unchanged(sandbox, hf_server):
    module = sandbox.load(INPAINT, mobile=False)
    hf_server.files["/ogkalu/lama-manga-onnx-dynamic/resolve/main/lama-manga-dynamic.onnx"] = b"\x08data"
    with pytest.raises(ImportError):
        module.download_model(repo_id="ogkalu/lama-manga-onnx-dynamic", filename="lama-manga-dynamic.onnx")
    assert hf_server.hits == []
