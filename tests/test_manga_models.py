"""U8 manga models: on-demand ONNX model manager, model cache paths, phone defaults, mobile availability.

Covers
  * src/manga_models.py: registry (pins agree with bubble_detector / local_inpainter), cache
    paths (desktop defaults unchanged, mobile <data>/models/<kind>), downloads against a local
    HTTP server serving a fake model (fresh, resume after a dropped connection, Range ignored,
    checksum and size failures, cancel + resume, cancel() from another thread, HTTP errors,
    free-space check), status / delete / disk usage, phone defaults and required_models;
  * bubble_detector.py / local_inpainter.py: the model-cache default goes through manga_models
    on mobile only (desktop expressions unchanged);
  * runtime_bootstrap.py: the env contract exports the same cache layout;
  * azure_document_intelligence_rest.py + ocr_manager.py: the Azure Document Intelligence REST
    client (local server) stands in for the SDK on mobile only;
  * settings_schema.py: torch-only manga combo values / settings are unavailable on mobile;
  * src/mobile/pyproject.toml: the device-only RapidOCR pins.

Nothing touches the network, the real model caches, HOME or src/config.json: every cache
variable points at tmp_path and downloads go to 127.0.0.1.
"""

import ast
import hashlib
import http.server
import importlib.util
import json
import os
import re
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import manga_models as mm  # noqa: E402
import settings_schema as ss  # noqa: E402

_ENV_KEYS = (
    "GLOSSARION_MOBILE", "GLOSSARION_NO_PROCESSES", "FLET_PLATFORM", "GLOSSARION_DATA_DIR",
    "BUBBLE_CACHE_DIR", "MODEL_CACHE_DIR", "ONNX_CACHE_DIR", "HF_ENDPOINT", "CONFIG_FILE",
    # set by bubble_detector / local_inpainter at import or construction: restored after each test
    "ORT_DISABLE_MEMORY_ARENA", "CUDA_LAUNCH_BLOCKING", "TORCH_USE_CUDA_DSA", "TORCHDYNAMO_DISABLE",
)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Desktop-like environment; every model cache under tmp_path."""
    for key in _ENV_KEYS:
        monkeypatch.setenv(key, "x")  # records the original value (or its absence) for teardown
        monkeypatch.delenv(key)
    monkeypatch.setenv("BUBBLE_CACHE_DIR", str(tmp_path / "bubble"))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "inpaint"))
    monkeypatch.setenv("ONNX_CACHE_DIR", str(tmp_path / "onnx"))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _mobile(monkeypatch, on=True):
    if on:
        monkeypatch.setenv("GLOSSARION_MOBILE", "1")
        monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    else:
        monkeypatch.delenv("GLOSSARION_MOBILE", raising=False)
        monkeypatch.delenv("GLOSSARION_NO_PROCESSES", raising=False)


# =========================================================================== registry
def _literal_assign(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


def _tree(name):
    return ast.parse((SRC / name).read_text(encoding="utf-8"), filename=name)


def test_registry_shape():
    assert [s.key for s in mm.specs()] == list(mm.REGISTRY)
    assert {s.kind for s in mm.specs()} == {mm.KIND_DETECTOR, mm.KIND_INPAINTER}
    for spec in mm.specs():
        assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256), spec.key
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision), spec.key  # pinned commit, never 'main'
        assert spec.size > 1024 * 1024 and spec.title and spec.description
        assert spec.cache_env == ("BUBBLE_CACHE_DIR" if spec.kind == mm.KIND_DETECTOR else "MODEL_CACHE_DIR")
    # exactly one phone default per kind, and it is the one the overrides select
    for kind, key in ((mm.KIND_DETECTOR, mm.MOBILE_DETECTOR_KEY), (mm.KIND_INPAINTER, mm.MOBILE_INPAINTER_KEY)):
        assert [s.key for s in mm.specs(kind) if s.phone_default] == [key]
    assert mm.get_spec(mm.MOBILE_DETECTOR_KEY).filename == "detector-v4-s_int8.onnx"
    assert mm.get_spec(mm.MOBILE_DETECTOR_KEY).size < 12 * 1024 * 1024  # ~11 MB INT8 export
    assert mm.get_spec(mm.MOBILE_INPAINTER_KEY).selector == "aot_onnx"
    with pytest.raises(KeyError):
        mm.get_spec("nope")


def test_registry_agrees_with_the_backend_modules():
    bubble = _tree("bubble_detector.py")
    filenames = None
    for node in ast.walk(bubble):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RTDETR_ONNX_FILENAMES"
                                                for t in node.targets):
            filenames = ast.literal_eval(node.value)
    assert filenames == {s.filename for s in mm.specs(mm.KIND_DETECTOR)}
    assert {s.selector for s in mm.specs(mm.KIND_DETECTOR)} == filenames
    source = (SRC / "bubble_detector.py").read_text(encoding="utf-8")
    assert "self.rtdetr_onnx_repo = 'ogkalu/comic-text-and-bubble-detector'" in source
    assert {s.repo_id for s in mm.specs(mm.KIND_DETECTOR)} == {"ogkalu/comic-text-and-bubble-detector"}

    jit = _literal_assign(_tree("local_inpainter.py"), "LAMA_JIT_MODELS")
    for spec in mm.specs(mm.KIND_INPAINTER):
        info = jit[spec.key]
        assert info.get("is_onnx") is True
        assert (info["repo_id"], info["filename"]) == (spec.repo_id, spec.filename)
        if info.get("md5") and len(info["md5"]) == 64:  # those 'md5' fields hold the HF sha256
            assert info["md5"] == spec.sha256


def test_lookups_and_urls(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    spec = mm.get_spec("aot_onnx")
    assert spec.url() == f"https://huggingface.co/ogkalu/aot-inpainting/resolve/{spec.revision}/aot.onnx"
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:1/")
    assert spec.url().startswith("http://127.0.0.1:1/ogkalu/aot-inpainting/resolve/")
    assert mm.find_spec("ogkalu/aot-inpainting", "aot.onnx") is spec
    assert mm.find_spec("ogkalu/aot-inpainting", "aot_traced.pt") is None
    assert mm.spec_for_detector_variant("detector.onnx").key == "rtdetr"
    assert mm.spec_for_detector_variant("/x/detector_int8.onnx").key == "rtdetr_int8"
    assert mm.spec_for_inpaint_method("ANIME_ONNX").key == "anime_onnx"
    for method in ("anime", "aot", "lama", "mat", "custom-image-edit", ""):
        assert mm.spec_for_inpaint_method(method) is None
    assert mm.format_size(11120765) == "11 MB" and mm.format_size(5 * 1024 ** 3 // 4) == "1.2 GB"


# =========================================================================== paths
def test_cache_dirs_desktop_defaults_unchanged(env, monkeypatch):
    for key in ("BUBBLE_CACHE_DIR", "MODEL_CACHE_DIR", "ONNX_CACHE_DIR"):
        monkeypatch.delenv(key)
    monkeypatch.setenv("GLOSSARION_DATA_DIR", str(env / "data"))  # ignored on desktop
    assert mm.default_cache_dir("BUBBLE_CACHE_DIR") == "models"
    assert mm.default_cache_dir("ONNX_CACHE_DIR") == "models"
    assert mm.default_cache_dir("MODEL_CACHE_DIR") == os.path.expanduser("~/.cache/inpainting")
    assert mm.default_cache_dir("BUBBLE_CACHE_DIR", "elsewhere") == "elsewhere"
    assert mm.model_path("rtdetr") == os.path.join("models", "detector.onnx")
    with pytest.raises(KeyError):
        mm.default_cache_dir("TIKTOKEN_CACHE_DIR")


def test_cache_dirs_on_mobile(env, monkeypatch):
    for key in ("BUBBLE_CACHE_DIR", "MODEL_CACHE_DIR", "ONNX_CACHE_DIR"):
        monkeypatch.delenv(key)
    _mobile(monkeypatch)
    assert mm.default_cache_dir("MODEL_CACHE_DIR", "x") == "x"  # no data dir: the module default
    data = env / "data"
    monkeypatch.setenv("GLOSSARION_DATA_DIR", str(data))
    root = data / "models"
    assert mm.mobile_models_root() == str(root)
    assert mm.cache_env(root) == {
        "BUBBLE_CACHE_DIR": str(root / "detector"),
        "MODEL_CACHE_DIR": str(root / "inpainting"),
        "ONNX_CACHE_DIR": str(root / "onnx"),
    }
    for name, path in mm.cache_env(root).items():
        assert mm.default_cache_dir(name, "ignored") == path
        assert mm.cache_dir(name) == path
    assert mm.model_path("aot_onnx") == str(root / "inpainting" / "aot.onnx")
    monkeypatch.setenv("MODEL_CACHE_DIR", str(env / "explicit"))  # the variable always wins
    assert mm.model_path("aot_onnx") == str(env / "explicit" / "aot.onnx")


def _helper(module_file):
    """Compile ``_model_cache_default`` out of a backend module (no heavy imports)."""
    tree = _tree(module_file)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_model_cache_default")
    import mobile_runtime
    namespace = {"mobile_runtime": mobile_runtime}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), module_file, "exec"), namespace)
    return namespace["_model_cache_default"]


@pytest.mark.parametrize("module_file", ["bubble_detector.py", "local_inpainter.py"])
def test_backend_cache_default_goes_through_manga_models_on_mobile_only(env, monkeypatch, module_file):
    helper = _helper(module_file)
    monkeypatch.setenv("GLOSSARION_DATA_DIR", str(env / "data"))
    assert helper("BUBBLE_CACHE_DIR", "models") == "models"  # desktop: unchanged
    assert helper("MODEL_CACHE_DIR", "/home/x/.cache/inpainting") == "/home/x/.cache/inpainting"
    _mobile(monkeypatch)
    assert helper("BUBBLE_CACHE_DIR", "models") == str(env / "data" / "models" / "detector")
    assert helper("ONNX_CACHE_DIR", "models") == str(env / "data" / "models" / "onnx")
    monkeypatch.setitem(sys.modules, "manga_models", None)  # module missing: the desktop default
    assert helper("BUBBLE_CACHE_DIR", "models") == "models"


def test_backend_cache_expressions_keep_desktop_defaults():
    bubble = (SRC / "bubble_detector.py").read_text(encoding="utf-8")
    assert ("self.cache_dir = os.environ.get('BUBBLE_CACHE_DIR', "
            "_model_cache_default('BUBBLE_CACHE_DIR', 'models'))") in bubble
    inpaint = (SRC / "local_inpainter.py").read_text(encoding="utf-8")
    assert "ONNX_CACHE_DIR = os.environ.get('ONNX_CACHE_DIR', _model_cache_default('ONNX_CACHE_DIR', 'models'))" in inpaint
    assert ("CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', _model_cache_default('MODEL_CACHE_DIR', "
            "os.path.expanduser('~/.cache/inpainting')))") in inpaint
    for name in ("bubble_detector.py", "local_inpainter.py"):
        fn = next(n for n in _tree(name).body if isinstance(n, ast.FunctionDef) and n.name == "_model_cache_default")
        first = fn.body[1]  # after the docstring: the desktop early return
        assert isinstance(first, ast.If) and "is_mobile" in ast.unparse(first.test)
        assert ast.unparse(first.body[0]) == "return desktop_default"


_HAVE_MANGA_STACK = all(importlib.util.find_spec(m) is not None for m in ("numpy", "cv2"))
_BLOCKED = ("torch", "ultralytics", "transformers", "huggingface_hub", "onnx_cpp_backend", "onnxruntime_extensions")


def _fresh(module_file, monkeypatch):
    for name in _BLOCKED:
        monkeypatch.setitem(sys.modules, name, None)
    spec = importlib.util.spec_from_file_location(f"_u8_{module_file[:-3]}_{time.monotonic_ns()}", SRC / module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not _HAVE_MANGA_STACK, reason="numpy and cv2 are needed to import the backend modules")
@pytest.mark.parametrize("mobile", [False, True], ids=["desktop", "mobile"])
def test_backend_modules_resolve_cache_dirs(env, monkeypatch, mobile):
    for key in ("BUBBLE_CACHE_DIR", "MODEL_CACHE_DIR", "ONNX_CACHE_DIR"):
        monkeypatch.delenv(key)
    monkeypatch.setenv("GLOSSARION_DATA_DIR", str(env / "data"))
    _mobile(monkeypatch, mobile)
    inpainter = _fresh("local_inpainter.py", monkeypatch)
    bubble = _fresh("bubble_detector.py", monkeypatch)
    detector = bubble.BubbleDetector(config_path=str(env / "missing_config.json"))
    if mobile:
        root = env / "data" / "models"
        assert inpainter.CACHE_DIR == str(root / "inpainting") == mm.cache_dir("MODEL_CACHE_DIR")
        assert inpainter.ONNX_CACHE_DIR == str(root / "onnx")
        assert detector.cache_dir == str(root / "detector") and os.path.isdir(detector.cache_dir)
        assert not (env / "models").exists()  # nothing in the cwd
    else:
        assert inpainter.CACHE_DIR == os.path.expanduser("~/.cache/inpainting")
        assert inpainter.ONNX_CACHE_DIR == "models" and detector.cache_dir == "models"
    # an explicit variable wins on both platforms
    monkeypatch.setenv("BUBBLE_CACHE_DIR", str(env / "explicit"))
    assert bubble.BubbleDetector(config_path=str(env / "missing_config.json")).cache_dir == str(env / "explicit")


def _runtime_bootstrap():
    """runtime_bootstrap loaded by path under a private name (stdlib only; no app package import)."""
    name = "_u8_runtime_bootstrap"
    if name in sys.modules:
        return sys.modules[name]
    path = SRC / "mobile" / "app" / "glossarion_mobile" / "runtime_bootstrap.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve string annotations through sys.modules
    try:
        spec.loader.exec_module(module)
    except SyntaxError as exc:  # pragma: no cover - the app targets 3.13
        sys.modules.pop(name, None)
        pytest.skip(f"runtime_bootstrap needs a newer Python: {exc}")
    return module


def test_bootstrap_exports_the_manga_models_layout(env, monkeypatch):
    rb = _runtime_bootstrap()
    assert rb.MODEL_CACHE_SUBDIRS == mm.CACHE_ENV_SUBDIRS
    for name in ("data", "cache", "temp"):
        monkeypatch.setenv(f"FLET_APP_STORAGE_{name.upper()}", str(env / "storage" / name))
    paths = rb.resolve_paths(SRC / "mobile" / "app", platform="android")
    assert paths.models == paths.data / "models"
    assert paths.writable_dirs()["models"] == paths.models
    contract = paths.env_contract()
    expected = mm.cache_env(paths.models)
    assert {k: contract[k] for k in expected} == expected
    # and manga_models computes the same dirs from the contract's data dir alone
    for key in expected:
        monkeypatch.delenv(key, raising=False)
    _mobile(monkeypatch)
    monkeypatch.setenv("GLOSSARION_DATA_DIR", contract["GLOSSARION_DATA_DIR"])
    assert {k: mm.default_cache_dir(k) for k in expected} == expected


# =========================================================================== downloads
class _ModelHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802 (http.server API)
        srv = self.server
        rng = self.headers.get("Range")
        srv.requests.append((self.path, rng))
        if srv.status_queue:
            code = srv.status_queue.pop(0)
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = srv.files.get(self.path)
        if body is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start = 0
        if rng and not srv.ignore_range:
            match = re.fullmatch(r"bytes=(\d+)-", rng)
            start = int(match.group(1))
            if start >= len(body):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(body)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            reported = srv.range_start_override if srv.range_start_override is not None else start
            self.send_header("Content-Range", f"bytes {reported}-{len(body) - 1}/{len(body)}")
        else:
            self.send_response(200)
        payload = body[start:]
        self.send_header("Content-Length", str(srv.declared if srv.declared is not None else len(payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        limit = len(payload)
        if srv.cut_after:
            limit = min(limit, srv.cut_after.pop(0))
        sent = 0
        try:
            while sent < limit:
                chunk = payload[sent:min(limit, sent + srv.chunk)]
                self.wfile.write(chunk)
                self.wfile.flush()
                sent += len(chunk)
                srv.sent += len(chunk)
                if srv.delay:
                    time.sleep(srv.delay)
        except (ConnectionError, OSError):
            pass
        if limit < len(payload):
            self.close_connection = True

    def log_message(self, *args):
        pass


@pytest.fixture
def server(env):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ModelHandler)
    srv.daemon_threads = True
    srv.files, srv.requests, srv.status_queue, srv.cut_after = {}, [], [], []
    srv.ignore_range, srv.declared, srv.range_start_override = False, None, None
    srv.chunk, srv.delay, srv.sent = 64 * 1024, 0.0, 0
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    srv.base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


def _fake(server, monkeypatch, size=700 * 1024 + 13, *, key="fake_model", corrupt=False):
    payload = os.urandom(size)
    spec = mm.ModelSpec(
        key=key, kind=mm.KIND_INPAINTER, title="Fake model", repo_id="owner/fake-repo", filename="fake.onnx",
        revision="0123456789abcdef0123456789abcdef01234567", sha256=hashlib.sha256(payload).hexdigest(),
        size=size, cache_env="MODEL_CACHE_DIR", selector="fake_onnx", description="test",
    )
    monkeypatch.setenv("HF_ENDPOINT", server.base)
    served = bytearray(payload)
    if corrupt:
        served[size // 2] ^= 0xFF
    server.files[f"/owner/fake-repo/resolve/{spec.revision}/fake.onnx"] = bytes(served)
    return spec, payload


def test_download_fresh_verifies_and_renames(env, server, monkeypatch):
    spec, payload = _fake(server, monkeypatch)
    reports = []
    path = mm.download(spec, progress=reports.append, throttle=0.0, chunk_size=32 * 1024)
    assert path == str(env / "inpaint" / "fake.onnx") == mm.model_path(spec)
    assert Path(path).read_bytes() == payload
    assert not Path(mm.partial_path(spec)).exists()
    assert [r.phase for r in reports][-2:] == ["verify", "done"]
    downloads = [r.downloaded for r in reports if r.phase == "download"]
    assert downloads == sorted(downloads) and downloads[-1] == spec.size and reports[-1].percent == 100
    assert server.requests == [(f"/owner/fake-repo/resolve/{spec.revision}/fake.onnx", None)]
    assert mm.active_downloads() == {}
    status = mm.status(spec)
    assert status.state == "installed" and status.bytes_on_disk == spec.size and status.partial_bytes == 0
    assert mm.is_installed(spec, verify_hash=True) and mm.verify(spec)
    # installed: no network at all
    assert mm.download(spec) == path and len(server.requests) == 1


def test_resume_after_dropped_connection(env, server, monkeypatch):
    spec, payload = _fake(server, monkeypatch)
    server.cut_after = [300 * 1024]
    with pytest.raises(mm.ModelDownloadError):
        mm.download(spec, retries=0, chunk_size=16 * 1024)
    partial = Path(mm.partial_path(spec))
    have = partial.stat().st_size
    assert 0 < have <= 300 * 1024 and not Path(mm.model_path(spec)).exists()
    assert mm.status(spec).state == "partial" and mm.status(spec).partial_bytes == have
    sent_before = server.sent
    assert Path(mm.download(spec, retries=0)).read_bytes() == payload
    assert server.requests[-1][1] == f"bytes={have}-"
    assert server.sent - sent_before == spec.size - have  # only the missing tail was transferred
    assert not partial.exists()


def test_transient_failures_resume_within_one_call(env, server, monkeypatch):
    spec, payload = _fake(server, monkeypatch)
    server.status_queue = [503]
    server.cut_after = [200 * 1024]
    path = mm.download(spec, retries=3, backoff=0.0, chunk_size=16 * 1024)
    assert Path(path).read_bytes() == payload
    ranges = [r for _p, r in server.requests]
    assert ranges[0] is None and ranges[1] is None and ranges[2] and ranges[2].startswith("bytes=")


def test_flaky_connection_that_keeps_making_progress_finishes(env, server, monkeypatch):
    """Only consecutive attempts without new bytes count against ``retries``: a phone connection
    that drops every 100 KB still finishes, a connection that delivers nothing does not."""
    spec, payload = _fake(server, monkeypatch)
    server.cut_after = [100 * 1024] * 7  # 8 attempts for 700 KB + 13 bytes
    assert Path(mm.download(spec, retries=1, backoff=0.0, chunk_size=16 * 1024)).read_bytes() == payload
    assert len(server.requests) == 8 and all(r for _p, r in server.requests[1:])

    stalled, _payload = _fake(server, monkeypatch, key="stalled_model")
    mm.delete(stalled)
    before = len(server.requests)
    server.cut_after = [0, 0, 0, 0]
    with pytest.raises(mm.ModelDownloadError, match="download failed"):
        mm.download(stalled, retries=1, backoff=0.0)
    assert len(server.requests) - before == 2


def test_server_ignoring_range_restarts_cleanly(env, server, monkeypatch):
    spec, payload = _fake(server, monkeypatch)
    Path(mm.partial_path(spec)).parent.mkdir(parents=True)
    Path(mm.partial_path(spec)).write_bytes(b"\0" * 1000)  # stale bytes the server will not resume
    server.ignore_range = True
    assert Path(mm.download(spec)).read_bytes() == payload
    assert server.requests[0][1] == "bytes=1000-"


def test_range_answered_with_another_range_restarts_from_zero(env, server, monkeypatch):
    spec, payload = _fake(server, monkeypatch)
    Path(mm.partial_path(spec)).parent.mkdir(parents=True)
    Path(mm.partial_path(spec)).write_bytes(payload[:5000])
    server.range_start_override = 4096  # wrong Content-Range start: the partial is dropped
    assert Path(mm.download(spec, retries=0)).read_bytes() == payload
    assert [r for _p, r in server.requests] == ["bytes=5000-", None]  # one restart from zero


def test_checksum_failure_keeps_nothing(env, server, monkeypatch):
    spec, _payload = _fake(server, monkeypatch, corrupt=True)
    with pytest.raises(mm.ChecksumMismatch):
        mm.download(spec)
    assert not Path(mm.model_path(spec)).exists() and not Path(mm.partial_path(spec)).exists()
    assert mm.status(spec).state == "missing"


def test_size_mismatch_fails_before_downloading(env, server, monkeypatch):
    spec, _payload = _fake(server, monkeypatch)
    server.declared = spec.size + 10
    with pytest.raises(mm.ChecksumMismatch, match="expected"):
        mm.download(spec)
    assert not Path(mm.partial_path(spec)).exists() and not Path(mm.model_path(spec)).exists()


def test_cancel_keeps_partial_and_resumes(env, server, monkeypatch):
    spec, payload = _fake(server, monkeypatch, size=2 * 1024 * 1024 + 5)
    server.chunk, server.delay = 32 * 1024, 0.002
    cancel = threading.Event()

    def progress(report):
        if report.phase == "download" and report.downloaded >= 256 * 1024:
            cancel.set()

    with pytest.raises(mm.DownloadCancelled):
        mm.download(spec, progress=progress, cancel=cancel, throttle=0.0, chunk_size=32 * 1024)
    have = Path(mm.partial_path(spec)).stat().st_size
    assert 256 * 1024 <= have < spec.size
    assert mm.status(spec).state == "partial" and mm.active_downloads() == {}
    server.delay = 0.0
    assert Path(mm.download(spec)).read_bytes() == payload
    assert server.requests[-1][1] == f"bytes={have}-"


def test_cancel_from_another_thread(env, server, monkeypatch):
    spec, _payload = _fake(server, monkeypatch, size=3 * 1024 * 1024)
    server.chunk, server.delay = 16 * 1024, 0.005
    outcome = {}
    started = threading.Event()

    def run():
        try:
            mm.download(spec, progress=lambda r: started.set(), throttle=0.0, chunk_size=16 * 1024)
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            outcome["error"] = exc

    worker = threading.Thread(target=run)
    worker.start()
    assert started.wait(10)
    assert mm.status(spec).state == "downloading"
    with pytest.raises(mm.ModelDownloadError, match="downloading"):
        mm.delete(spec)  # refused while running
    assert mm.cancel(spec) is True
    worker.join(10)
    assert isinstance(outcome.get("error"), mm.DownloadCancelled)
    assert mm.cancel(spec) is False  # nothing running any more
    assert Path(mm.partial_path(spec)).exists()


def test_cancel_also_stops_a_second_caller_waiting_for_the_same_model(env, server, monkeypatch):
    spec, payload = _fake(server, monkeypatch, size=3 * 1024 * 1024)
    server.chunk, server.delay = 16 * 1024, 0.005
    outcome = {}
    started = threading.Event()

    def run(name):
        try:
            outcome[name] = mm.download(spec, progress=lambda r: started.set(), throttle=0.0, chunk_size=16 * 1024)
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            outcome[name] = exc

    first = threading.Thread(target=run, args=("first",))
    first.start()
    assert started.wait(10)
    second = threading.Thread(target=run, args=("second",))  # waits on the model's path lock
    second.start()
    deadline = time.monotonic() + 10
    while len(mm._CANCELS.get(spec.key) or ()) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(mm._CANCELS[spec.key]) == 2
    assert mm.cancel(spec) is True
    first.join(10)
    second.join(10)
    assert isinstance(outcome["first"], mm.DownloadCancelled)
    assert isinstance(outcome["second"], mm.DownloadCancelled)  # did not start a download of its own
    assert spec.key not in mm._CANCELS and mm.active_downloads() == {}
    server.delay = 0.0
    assert Path(mm.download(spec)).read_bytes() == payload  # the kept partial resumes


def test_http_errors(env, server, monkeypatch):
    spec, _payload = _fake(server, monkeypatch)
    server.status_queue = [404]
    with pytest.raises(mm.ModelDownloadError, match="HTTP 404"):
        mm.download(spec, retries=3, backoff=0.0)
    assert len(server.requests) == 1  # a 404 is not retried
    server.status_queue = [500, 500, 500]
    with pytest.raises(mm.ModelDownloadError, match="HTTP 500"):
        mm.download(spec, retries=2, backoff=0.0)
    assert len(server.requests) == 4


def test_insufficient_space(env, server, monkeypatch):
    spec, _payload = _fake(server, monkeypatch)
    usage = types.SimpleNamespace(total=10 ** 9, used=10 ** 9, free=1024)
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda path: usage)
    with pytest.raises(mm.InsufficientSpace):
        mm.download(spec)
    assert server.requests == []


def test_delete_and_disk_usage(env, server, monkeypatch):
    spec, _payload = _fake(server, monkeypatch)
    mm.download(spec)
    Path(mm.partial_path(spec)).write_bytes(b"x" * 10)
    (env / "bubble").mkdir(exist_ok=True)
    (env / "bubble" / "config.json").write_bytes(b"{}")
    usage = mm.disk_usage()
    assert usage["dirs"]["MODEL_CACHE_DIR"] == {"path": str(env / "inpaint"), "bytes": spec.size + 10}
    assert usage["dirs"]["BUBBLE_CACHE_DIR"]["bytes"] == 2
    assert usage["total_bytes"] == spec.size + 12 == int(usage)
    assert set(usage["models"]) == set(mm.REGISTRY)  # registered models only
    assert mm.delete(spec) == spec.size + 10
    assert not Path(mm.model_path(spec)).exists() and not Path(mm.partial_path(spec)).exists()
    assert mm.delete(spec) == 0


def test_shared_cache_dir_is_counted_once(env, monkeypatch):
    monkeypatch.setenv("ONNX_CACHE_DIR", str(env / "bubble"))
    (env / "bubble").mkdir()
    (env / "bubble" / "a.onnx").write_bytes(b"1234")
    usage = mm.disk_usage()
    assert usage["dirs"]["ONNX_CACHE_DIR"]["bytes"] == usage["dirs"]["BUBBLE_CACHE_DIR"]["bytes"] == 4
    assert usage["total_bytes"] == 4


def test_registered_model_end_to_end_over_hf_endpoint(env, server, monkeypatch):
    """A registry entry downloads from {HF_ENDPOINT}/{repo}/resolve/{pinned revision}/{file} and is
    verified against its pinned sha256 (a fake body fails, so nothing is installed)."""
    spec = mm.get_spec(mm.MOBILE_DETECTOR_KEY)
    monkeypatch.setenv("HF_ENDPOINT", server.base)
    server.files[f"/{spec.repo_id}/resolve/{spec.revision}/{spec.filename}"] = b"\0" * spec.size
    server.chunk = 1024 * 1024
    with pytest.raises(mm.ChecksumMismatch, match="sha256"):
        mm.download(spec)
    assert server.requests[0][0] == f"/{spec.repo_id}/resolve/{spec.revision}/{spec.filename}"
    assert not Path(mm.model_path(spec)).exists()
    assert mm.model_path(spec) == str(env / "bubble" / spec.filename)


def test_legacy_progress_adapter():
    seen = []
    adapter = mm.legacy_progress(lambda *a: seen.append(a))
    adapter(mm.DownloadProgress("k", 512 * 1024, 1024 * 1024, 2 * 1024 * 1024))
    assert seen == [(50, 0.5, 1.0, 2.0)]
    assert mm.legacy_progress(None) is None


# =========================================================================== phone defaults
def _dialog_defaults():
    """The canonical desktop manga_settings defaults (manga_settings_defaults when present, else
    the literal MangaSettingsDialog.default_settings)."""
    if (SRC / "manga_settings_defaults.py").exists():
        import manga_settings_defaults
        return manga_settings_defaults.default_manga_settings()
    tree = _tree("manga_settings_dialog.py")
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "default_settings"):
            return ast.literal_eval(node.value)
    raise AssertionError("default_settings not found")


def test_phone_defaults_override_real_desktop_defaults(monkeypatch):
    defaults = _dialog_defaults()
    flat = mm.mobile_default_overrides()
    assert flat["manga_settings.ocr.rtdetr_onnx_variant"] == "detector-v4-s_int8.onnx"
    assert flat["manga_settings.advanced.hd_strategy_resize_limit"] == 1024
    assert flat["manga_local_inpaint_model"] == flat["manga_settings.inpainting.local_method"] == "aot_onnx"
    for dotted, value in flat.items():
        if not dotted.startswith("manga_settings."):
            continue
        node = defaults
        *parents, leaf = dotted.split(".")[1:]
        for part in parents:
            node = node[part]
        assert leaf in node, dotted
        assert node[leaf] != value, f"{dotted} equals the desktop default"
    # manga_translator uses ocr.rtdetr_max_concurrency as the number of parallel cloud OCR calls
    # per page (Google Vision region OCR); the phone's default OCR is cloud, so it stays as is
    assert "manga_settings.ocr.rtdetr_max_concurrency" not in flat
    assert "manga_settings.ocr.ocr_max_concurrency" not in flat


def test_apply_mobile_defaults_only_on_mobile(monkeypatch):
    defaults = _dialog_defaults()
    snapshot = json.dumps(defaults, sort_keys=True)
    _mobile(monkeypatch, False)
    same = mm.apply_mobile_defaults(defaults)
    assert same == defaults and same is not defaults
    assert mm.apply_mobile_top_level_defaults({"manga_local_inpaint_model": "anime_onnx"}) == {
        "manga_local_inpaint_model": "anime_onnx"}
    _mobile(monkeypatch)
    phone = mm.apply_mobile_defaults(defaults)
    assert phone["ocr"]["rtdetr_onnx_variant"] == "detector-v4-s_int8.onnx"
    assert phone["ocr"]["detector_type"] == "rtdetr_onnx" and phone["ocr"]["ocr_batch_size"] == 8  # untouched keys
    assert phone["advanced"]["hd_strategy_resize_limit"] == 1024 and phone["advanced"]["max_workers"] == 1
    assert phone["advanced"]["panel_max_workers"] == 1
    assert phone["advanced"]["parallel_panel_translation"] == defaults["advanced"]["parallel_panel_translation"]
    assert phone["inpainting"]["local_method"] == "aot_onnx" and phone["inpainting"]["method"] == "local"
    assert mm.apply_mobile_top_level_defaults({"x": 1}) == {"x": 1, "manga_local_inpaint_model": "aot_onnx"}
    assert json.dumps(defaults, sort_keys=True) == snapshot  # input never mutated
    _mobile(monkeypatch, False)
    assert mm.apply_mobile_defaults({}, force=True)["advanced"]["max_workers"] == 1


def test_apply_mobile_run_defaults_puts_the_phone_defaults_into_a_runs_config(env, monkeypatch):
    """The run reads the config with the desktop fallbacks (detector.onnx, anime_onnx, 1536 px,
    desktop worker counts); on mobile the phone defaults Settings shows must be in it."""
    _mobile(monkeypatch, False)
    desktop = {"manga_settings": {"ocr": {}}}
    assert mm.apply_mobile_run_defaults(desktop) is desktop and desktop == {"manga_settings": {"ocr": {}}}
    _mobile(monkeypatch)
    fresh: dict = {}
    assert mm.apply_mobile_run_defaults(fresh) is fresh
    assert fresh["manga_settings"]["ocr"]["rtdetr_onnx_variant"] == "detector-v4-s_int8.onnx"
    assert fresh["manga_settings"]["inpainting"]["local_method"] == "aot_onnx"
    assert fresh["manga_local_inpaint_model"] == "aot_onnx"
    assert fresh["manga_settings"]["advanced"] == {"hd_strategy_resize_limit": 1024, "max_workers": 1,
                                                   "panel_max_workers": 1, "unload_models_after_translation": True}
    assert [s.key for s in mm.required_models(fresh)] == [s.key for s in mm.required_models({})]  # the UI's models
    # a stored value always wins (blank / None count as not stored); other keys stay untouched
    stored = {"manga_settings": {"ocr": {"rtdetr_onnx_variant": "detector_int8.onnx", "bubble_detection_enabled": False},
                                 "advanced": {"max_workers": 3, "hd_strategy_resize_limit": None, "x": 1}},
              "manga_local_inpaint_model": "lama_onnx", "other": "kept"}
    mm.apply_mobile_run_defaults(stored)
    assert stored["manga_settings"]["ocr"] == {"rtdetr_onnx_variant": "detector_int8.onnx",
                                               "bubble_detection_enabled": False}
    assert stored["manga_settings"]["advanced"] == {"max_workers": 3, "hd_strategy_resize_limit": 1024, "x": 1,
                                                    "panel_max_workers": 1, "unload_models_after_translation": True}
    assert stored["other"] == "kept"
    # the local inpainter is stored twice: the run follows the top-level value Settings shows
    assert stored["manga_settings"]["inpainting"]["local_method"] == "lama_onnx"
    nested_only = {"manga_settings": {"inpainting": {"local_method": "anime_onnx"}}}
    mm.apply_mobile_run_defaults(nested_only)
    assert nested_only["manga_local_inpaint_model"] == "anime_onnx"
    both = {"manga_local_inpaint_model": "aot_onnx", "manga_settings": {"inpainting": {"local_method": "anime_onnx"}}}
    mm.apply_mobile_run_defaults(both)
    assert both["manga_settings"]["inpainting"]["local_method"] == "aot_onnx"
    assert [s.key for s in mm.required_models(both)][-1] == "aot_onnx"
    # a non-dict stored where the defaults expect a section is the user's: left alone
    odd = {"manga_settings": {"advanced": "custom"}}
    mm.apply_mobile_run_defaults(odd)
    assert odd["manga_settings"]["advanced"] == "custom"
    _mobile(monkeypatch, False)
    assert mm.apply_mobile_run_defaults({}, force=True)["manga_local_inpaint_model"] == "aot_onnx"


def test_required_models(env, monkeypatch):
    keys = lambda cfg: [s.key for s in mm.required_models(cfg)]  # noqa: E731
    assert keys({}) == ["rtdetr", "anime_onnx"]  # desktop GUI defaults
    assert keys(None) == ["rtdetr", "anime_onnx"]
    _mobile(monkeypatch)
    assert keys({}) == ["rtdetr_v4_s_int8", "aot_onnx"]
    cfg = {"manga_settings": {"ocr": {"rtdetr_onnx_variant": "detector_int8.onnx"}},
           "manga_local_inpaint_model": "lama_onnx"}
    assert keys(cfg) == ["rtdetr_int8", "lama_onnx"]
    assert keys({"manga_skip_inpainting": True}) == ["rtdetr_v4_s_int8"]
    assert keys({"manga_inpaint_method": "cloud"}) == ["rtdetr_v4_s_int8"]
    assert keys({"manga_settings": {"ocr": {"bubble_detection_enabled": False}}}) == ["aot_onnx"]
    assert keys({"manga_settings": {"ocr": {"detector_type": "yolo"}}, "manga_local_inpaint_model": "anime"}) == []
    own = env / "my_aot.onnx"
    own.write_bytes(b"\x08x")
    assert keys({"manga_aot_onnx_model_path": str(own)}) == ["rtdetr_v4_s_int8"]  # the user's own file
    missing = mm.missing_models({})
    assert [s.key for s in missing] == ["rtdetr_v4_s_int8", "aot_onnx"]


# =========================================================================== load / unload
def test_load_warms_a_detector_through_bubble_detector(env, monkeypatch):
    calls = []

    class FakeDetector:
        def __init__(self, config_path="config.json"):
            calls.append(("init", config_path))

        def load_rtdetr_onnx_model(self, model_id=None, force_reload=False, onnx_filename=None):
            calls.append(("load", model_id, onnx_filename))
            return True

        def unload(self, release_shared=False):
            calls.append(("unload", release_shared))

    monkeypatch.setitem(sys.modules, "bubble_detector", types.SimpleNamespace(BubbleDetector=FakeDetector))
    spec = mm.get_spec(mm.MOBILE_DETECTOR_KEY)
    with pytest.raises(mm.ModelDownloadError, match="not downloaded"):
        mm.load(spec)
    path = Path(mm.model_path(spec))
    path.parent.mkdir(parents=True)
    with open(path, "wb") as fh:
        fh.truncate(spec.size)  # the right size counts as installed (no hashing on status)
    monkeypatch.setenv("CONFIG_FILE", str(env / "config.json"))
    assert mm.load(spec) is True and mm.loaded() == [spec.key]
    assert calls[:2] == [("init", str(env / "config.json")), ("load", spec.repo_id, spec.filename)]
    assert mm.unload(spec) is True and calls[-1] == ("unload", True) and mm.loaded() == []
    assert mm.unload(spec) is False
    with pytest.raises(mm.ModelDownloadError, match="inpainter"):
        mm.load("aot_onnx")


# =========================================================================== Azure Document Intelligence REST
class _AzureHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload=None, headers=None):
        body = json.dumps(payload).encode() if payload is not None else b""
        self.send_response(code)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        srv = self.server
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        srv.posts.append({"path": self.path, "key": self.headers.get("Ocp-Apim-Subscription-Key"),
                          "type": self.headers.get("Content-Type"), "body": body})
        if srv.post_codes:
            code = srv.post_codes.pop(0)
            return self._send(code, {"error": {"code": "Busy", "message": "try later"}}, {"Retry-After": "0"})
        if "/documentintelligence/" in self.path and srv.legacy_only:
            return self._send(404, {"error": {"code": "404", "message": "Resource not found"}})
        self._send(202, None, {"Operation-Location": f"{srv.base}/ops/1?api-version=x", "Retry-After": "0"})

    def do_GET(self):  # noqa: N802
        srv = self.server
        srv.gets.append({"path": self.path, "key": self.headers.get("Ocp-Apim-Subscription-Key")})
        state = srv.states.pop(0) if srv.states else "succeeded"
        if state == "succeeded":
            return self._send(200, {"status": "succeeded", "analyzeResult": srv.result})
        if state == "failed":
            return self._send(200, {"status": "failed", "error": {"code": "InvalidImage", "message": "bad"}})
        self._send(200, {"status": "running"}, {"Retry-After": "0"})

    def log_message(self, *args):
        pass


_ANALYZE_RESULT = {
    "apiVersion": "2024-11-30",
    "modelId": "prebuilt-read",
    "content": "こんにちは\nWORLD",
    "pages": [{
        "pageNumber": 1, "width": 200, "height": 100, "unit": "pixel", "angle": 0,
        "lines": [
            {"content": "こんにちは", "polygon": [10, 20, 110, 20, 110, 40, 10, 40]},
            {"content": "WORLD", "polygon": [5.4, 50.6, 80, 50, 80, 70, 5, 70]},
        ],
        "words": [{"content": "WORLD", "polygon": [5, 50, 80, 50, 80, 70, 5, 70], "confidence": 0.97}],
    }],
}


@pytest.fixture
def azure(env):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _AzureHandler)
    srv.daemon_threads = True
    srv.posts, srv.gets, srv.states, srv.post_codes = [], [], [], []
    srv.legacy_only, srv.result = False, _ANALYZE_RESULT
    srv.base = f"http://127.0.0.1:{srv.server_address[1]}"
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


_HAVE_REQUESTS = importlib.util.find_spec("requests") is not None
needs_requests = pytest.mark.skipif(not _HAVE_REQUESTS, reason="requests is needed")


@needs_requests
def test_azure_rest_client_analyze_flow(azure):
    from azure_document_intelligence_rest import DocumentAnalysisRestClient, Point

    azure.states = ["running", "running"]
    client = DocumentAnalysisRestClient(azure.base + "/", "secret-key", poll_interval=0.0)
    poller = client.begin_analyze_document("prebuilt-read", document=b"\xff\xd8jpeg", locale="ja")
    result = poller.result()
    assert poller.done() and poller.status() == "succeeded"
    post = azure.posts[0]
    assert post["path"] == "/documentintelligence/documentModels/prebuilt-read:analyze?api-version=2024-11-30&locale=ja"
    assert post["key"] == "secret-key" and post["type"] == "application/octet-stream" and post["body"] == b"\xff\xd8jpeg"
    assert len(azure.gets) == 3 and all(g["key"] == "secret-key" for g in azure.gets)
    page = result.pages[0]
    assert result.model_id == "prebuilt-read" and page.page_number == 1 and page.unit == "pixel"
    assert [line.content for line in page.lines] == ["こんにちは", "WORLD"]
    assert page.lines[0].polygon == [Point(10, 20), Point(110, 20), Point(110, 40), Point(10, 40)]
    assert not hasattr(page.lines[0], "confidence")  # like the SDK's DocumentLine
    assert page.words[0].confidence == 0.97
    assert poller.result() is result  # cached


@needs_requests
def test_azure_rest_client_falls_back_to_the_form_recognizer_route(azure):
    from azure_document_intelligence_rest import DocumentAnalysisRestClient

    azure.legacy_only = True
    result = DocumentAnalysisRestClient(azure.base, "k", poll_interval=0.0).begin_analyze_document(
        "prebuilt-read", document=b"img").result()
    assert [p["path"].split("?")[0] for p in azure.posts] == [
        "/documentintelligence/documentModels/prebuilt-read:analyze",
        "/formrecognizer/documentModels/prebuilt-read:analyze",
    ]
    assert azure.posts[1]["path"].endswith("api-version=2023-07-31")
    assert result.pages[0].lines[1].content == "WORLD"


@needs_requests
def test_azure_rest_client_errors_retries_and_stop(azure):
    from azure_document_intelligence_rest import DocumentAnalysisRestClient, DocumentIntelligenceError

    azure.post_codes = [429]
    client = DocumentAnalysisRestClient(azure.base, "k", poll_interval=0.0, backoff=0.0)
    assert client.begin_analyze_document("prebuilt-read", document=b"x").result().pages
    assert len(azure.posts) == 2  # 429 retried

    azure.post_codes = [401]
    with pytest.raises(DocumentIntelligenceError) as info:
        client.begin_analyze_document("prebuilt-read", document=b"x")
    assert info.value.status_code == 401 and "Busy" in str(info.value)

    azure.states = ["failed"]
    with pytest.raises(DocumentIntelligenceError, match="InvalidImage"):
        client.begin_analyze_document("prebuilt-read", document=b"x").result()

    stop = threading.Event()
    stopper = DocumentAnalysisRestClient(azure.base, "k", poll_interval=5.0, stop_check=stop.is_set)
    poller = stopper.begin_analyze_document("prebuilt-read", document=b"x")
    stop.set()
    with pytest.raises(DocumentIntelligenceError, match="cancelled"):
        poller.result()
    with pytest.raises(ValueError):
        DocumentAnalysisRestClient("", "k")


_HAVE_OCR_STACK = _HAVE_REQUESTS and all(importlib.util.find_spec(m) is not None for m in ("numpy", "cv2", "PIL"))


@pytest.fixture
def ocr_provider(env, monkeypatch):
    if not _HAVE_OCR_STACK:
        pytest.skip("numpy, cv2, PIL and requests are needed to import ocr_manager")
    stub = types.ModuleType("manga_translator")
    stub.MangaTranslator = type("MangaTranslator", (), {"is_globally_cancelled": staticmethod(lambda: False)})
    monkeypatch.setitem(sys.modules, "manga_translator", stub)
    monkeypatch.setitem(sys.modules, "azure.ai.formrecognizer", None)  # the SDK is not in the build
    import ocr_manager
    logs = []
    provider = ocr_manager.AzureDocumentIntelligenceProvider(log_callback=lambda m, level="info": logs.append(m))
    provider.logs = logs
    return ocr_manager, provider


def test_azure_provider_uses_rest_on_mobile(azure, ocr_provider, monkeypatch):
    ocr_manager, provider = ocr_provider
    import numpy as np
    from azure_document_intelligence_rest import DocumentAnalysisRestClient

    _mobile(monkeypatch)
    assert provider.check_installation() is True
    assert provider.load_model(azure_endpoint=azure.base, azure_key="mobile-key") is True
    assert isinstance(provider.client, DocumentAnalysisRestClient) and provider.is_loaded
    provider.client.poll_interval = 0.0
    results = provider.detect_text(np.full((100, 200, 3), 255, np.uint8), language_hint="zh")
    assert [r.text for r in results] == ["こんにちは", "WORLD"]
    assert results[0].bbox == (10, 20, 100, 20) and results[0].confidence == 0.9
    assert results[1].vertices[0] == (5, 50)
    assert azure.posts[0]["path"].endswith("locale=zh-Hans") and azure.posts[0]["key"] == "mobile-key"
    # manga_translator's empty-block fallback calls the client directly with the SDK keywords
    crop = provider.client.begin_analyze_document("prebuilt-read", document=b"crop", locale="ja").result()
    assert [line.content for page in crop.pages for line in page.lines] == ["こんにちは", "WORLD"]


def test_azure_provider_desktop_without_sdk_unchanged(azure, ocr_provider, monkeypatch):
    _ocr_manager, provider = ocr_provider
    _mobile(monkeypatch, False)
    assert provider.check_installation() is False
    assert provider.load_model(azure_endpoint=azure.base, azure_key="k") is False
    assert provider.client is None and azure.posts == []
    assert any("SDK not installed" in m for m in provider.logs)


@pytest.mark.skipif(importlib.util.find_spec("rapidocr_onnxruntime") is None, reason="rapidocr_onnxruntime not installed")
def test_rapidocr_provider_reads_text(ocr_provider):
    """The provider the device build ships (rapidocr-onnxruntime 1.2.3 there; whatever is
    installed here): load + detect on a synthetic page through OCRManager."""
    ocr_manager, _provider = ocr_provider
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (640, 160), "white")
    try:
        font = ImageFont.truetype("arial.ttf", 56)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 56)
        except Exception:
            pytest.skip("no TrueType font for the synthetic page")
    ImageDraw.Draw(image).text((30, 40), "HELLO 123", fill="black", font=font)
    manager = ocr_manager.OCRManager()
    assert manager.check_provider_status("rapidocr")["installed"] is True
    assert manager.load_provider("rapidocr") is True
    results = manager.detect_text(np.array(image)[:, :, ::-1].copy(), "rapidocr")
    text = " ".join(r.text for r in results).upper()
    assert "HELLO" in text.replace(" ", "") or "123" in text, text
    assert all(len(r.bbox) == 4 and 0.0 <= r.confidence <= 1.0 for r in results)


# =========================================================================== settings availability
_MOBILE_OFF_VALUES = {
    "manga_ocr_provider": ("manga-ocr", "Qwen2-VL", "easyocr", "doctr", "paddleocr"),
    "manga_settings.ocr.detector_type": ("rtdetr", "yolo", "custom"),
    "manga_local_inpaint_model": ("aot", "lama", "anime", "lama_official", "mat", "ollama", "sd_local"),
    "manga_settings.inpainting.local_method": ("anime", "ollama"),
    "manga_inpaint_method": ("hybrid",),
    "manga_settings.inpainting.method": ("hybrid",),
}
_MOBILE_ON_VALUES = {
    "manga_ocr_provider": ("custom-api", "google", "azure", "azure-document-intelligence", "rapidocr"),
    "manga_settings.ocr.detector_type": ("rtdetr_onnx",),
    "manga_local_inpaint_model": ("aot_onnx", "anime_onnx", "lama_onnx", "custom-image-edit"),
    "manga_inpaint_method": ("local", "cloud"),
}


def test_torch_only_manga_values_are_unavailable_on_mobile():
    for key, values in _MOBILE_OFF_VALUES.items():
        for value in values:
            available, reason = ss.is_value_available(key, value, "mobile")
            assert available is False and reason, (key, value)
            assert ss.is_value_available(key, value, "desktop") == (True, "")
    for key, values in _MOBILE_ON_VALUES.items():
        for value in values:
            assert ss.is_value_available(key, value, "mobile") == (True, ""), (key, value)
    assert "PyTorch" in ss.is_value_available("manga_ocr_provider", "manga-ocr")[1]
    assert "desktop either" in ss.is_value_available("manga_local_inpaint_model", "sd_local")[1]
    assert set(ss.unavailable_values("manga_settings.ocr.detector_type")) == {"rtdetr", "yolo", "custom"}
    assert ss.unavailable_values("manga_settings.ocr.detector_type", "desktop") == {}
    item = ss.spec("manga_settings.ocr.detector_type")
    assert {(v, p) for v, p, _r in item.unavailable_values} == {("rtdetr", "mobile"), ("yolo", "mobile"),
                                                                ("custom", "mobile")}


@pytest.mark.parametrize("key", [
    "manga_settings.advanced.auto_convert_to_onnx", "manga_settings.advanced.auto_convert_to_onnx_background",
    "manga_settings.advanced.quantize_models", "manga_settings.advanced.onnx_quantize",
    "manga_settings.advanced.torch_precision", "qwen2vl_model_size", "manga_settings.ocr.bubble_model_path",
    "manga_settings.ocr.bubble_max_detections_yolo",
])
def test_torch_only_manga_settings_are_unavailable_on_mobile(key):
    available, reason = ss.is_available(key, "mobile")
    assert available is False and reason
    assert ss.is_available(key, "desktop") == (True, "")
    assert ss.spec(key).section  # still listed


def test_every_desktop_combo_option_is_covered():
    """Each option of the desktop manga combos is runnable on mobile (ONNX registry, cloud, endpoint)
    or carries a mobile reason: nothing silently missing."""
    text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(SRC.glob("manga*.py")))
    local = re.search(r"local_model_combo\.addItems\(\[([^\]]+)\]\)", text)
    if local is None or "ocr_providers = [" not in text:
        pytest.skip("the desktop combo lists moved; update this cross-check")
    options = re.findall(r"'([^']+)'", local.group(1))
    assert {"aot_onnx", "anime_onnx", "lama_onnx", "ollama", "sd_local"} <= set(options)
    runnable = {s.selector for s in mm.specs(mm.KIND_INPAINTER)} | {"custom-image-edit"}
    for option in options:
        available, _reason = ss.is_value_available("manga_local_inpaint_model", option)
        assert available is (option in runnable), option
    block = re.search(r"ocr_providers = \[(.+?)\n\s*\]", text, re.S).group(1)
    providers = re.findall(r"^\s*#?\s*\('([^']+)',", block, re.M)
    assert {"custom-api", "rapidocr", "manga-ocr", "paddleocr"} <= set(providers)
    cloud_or_onnx = {"custom-api", "google", "azure", "azure-document-intelligence", "rapidocr"}
    for provider in providers:
        assert ss.is_value_available("manga_ocr_provider", provider)[0] is (provider in cloud_or_onnx), provider


# =========================================================================== packaging
def test_rapidocr_is_a_device_only_dependency():
    try:
        import tomllib
    except ImportError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    data = tomllib.loads((SRC / "mobile" / "pyproject.toml").read_text(encoding="utf-8"))
    project = " ".join(data["project"]["dependencies"])
    assert "rapidocr" not in project and "opencv-python-headless" in project  # host uv sync stays headless
    flet = data["tool"]["flet"]
    for platform in ("android", "ios"):
        deps = flet[platform]["dependencies"]
        for pin in ("rapidocr-onnxruntime==1.2.3", "pyclipper==1.4.0", "shapely==2.1.2"):
            assert pin in deps, (platform, pin)
    # rapidocr reads its config/models through __file__: Android ships it extracted
    assert "rapidocr_onnxruntime" in flet["android"]["extract_packages"]
