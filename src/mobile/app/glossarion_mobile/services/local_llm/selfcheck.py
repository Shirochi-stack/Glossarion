"""Self-test bodies of the on-device engine (``diagnostics.selftest`` / ``diagnostics.e2e``).

* ``engine_check()`` (smoke ``llama_engine``): ``llama_cpp`` imports, its version is
  ``ENGINE_VERSION``, ``llama_print_system_info()`` loads the native libraries and libmtmd (vision)
  loads; ``gpu_backend_check()``: the build ships the GPU backend the app offers (Metal on iOS, OpenCL
  on Android arm64, none on x86_64) and nothing registered it while the GPU is off.
* ``fake_route_check(root)`` (e2e ``local_route_fake_engine``): ``LocalLLMServer`` + ``FakeEngine`` +
  ``UnifiedClient`` with ``local/fake-echo``, offline, plain and streaming.
* ``llama_suite()`` (suite ``llama``, ``glossarion://app/__selftest__?suite=llama``): downloads the
  preset named by ``GLOSSARION_LLAMA_SUITE_PRESET`` (default Garbage) into the models folder (a verified
  file is reused), loads it with ``n_ctx`` 2048, generates at most 16 tokens through ``UnifiedClient``,
  unloads, and prints ``GLOSSARION_LLAMA {...}``;
* ``llama_vision_suite()`` (suite ``llama``, second check ``llama_vision_describe``): the same preset with
  its vision projector (mmproj), a tiny test picture described through ``UnifiedClient.send_image`` (the
  chat's Vision mode and image jobs use it) and libmtmd; prints ``GLOSSARION_LLAMA_VISION {...}``.

Each one restores ``GLOSSARION_LOCAL_LLM_URL`` and the default service, so the app's own engine keeps
working after a self-test.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import time
from typing import Any, Iterator, Optional

from glossarion_mobile.services.local_llm import downloads as dl
from glossarion_mobile.services.local_llm import engine as eng
from glossarion_mobile.services.local_llm import hf, presets
from glossarion_mobile.services.local_llm import service as svc

__all__ = ["MARKER_LLAMA", "MARKER_LLAMA_VISION", "PRESET_ENV", "engine_check", "fake_route_check", "llama_suite",
           "llama_vision_suite", "sample_image_png"]

MARKER_LLAMA = "GLOSSARION_LLAMA"
MARKER_LLAMA_VISION = "GLOSSARION_LLAMA_VISION"
PRESET_ENV = "GLOSSARION_LLAMA_SUITE_PRESET"
SUITE_CTX = 2048
SUITE_TOKENS = 16


class _DictPrefs:
    def __init__(self, data: Optional[dict] = None) -> None:
        self.data = dict(data or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


@contextlib.contextmanager
def _isolated_route() -> Iterator[None]:
    """Keep the app's endpoint URL and default service across a check's own service."""
    saved_url = os.environ.get(svc.URL_ENV)
    saved_default = svc.default_service()
    try:
        yield
    finally:
        if saved_url is None:
            os.environ.pop(svc.URL_ENV, None)
        else:
            os.environ[svc.URL_ENV] = saved_url
        svc.set_default_service(saved_default)


def engine_check() -> dict:
    llama_cpp = importlib.import_module("llama_cpp")
    version = str(getattr(llama_cpp, "__version__", ""))
    if version != eng.ENGINE_VERSION:
        raise AssertionError(f"llama_cpp {version} != ENGINE_VERSION {eng.ENGINE_VERSION}")
    info = eng.system_info()
    if not info or info.startswith("unavailable"):
        raise AssertionError(f"llama_print_system_info failed: {info}")
    if "CPU" not in info:  # Android arm64 loads its CPU backend at run time (a variant per feature set)
        raise AssertionError(f"no CPU backend registered (CPU variant {eng.cpu_variant()}): {info!r}")
    vision, reason = eng.vision_status()  # libmtmd (our own build): the vision models' image encoder
    if not vision:
        raise AssertionError(reason or eng.NO_MTMD)
    return {"version": version, "system_info": info[:300], "cpu_variant": eng.cpu_variant(), "vision": vision,
            "gpu": gpu_backend_check()}


def gpu_backend_check() -> dict:
    """The optional GPU backend is shipped where it should be and inert until the app asks for it:
    iOS loaded libggml-metal (linked into libggml), the Android arm64 build names OpenCL, and no GPU
    backend is registered with ggml unless ``register_gpu_backend`` ran (the "Use GPU" setting)."""
    ext = importlib.import_module("llama_cpp._ctypes_extensions")
    name_of = getattr(ext, "gpu_backend_name", None)
    if name_of is None:
        raise AssertionError("this llama-cpp-python build has no GPU loader (mobile.patch 8.)")
    backend = name_of()
    offered = eng.gpu_offer()
    if offered is not None and offered != backend:  # (iOS 13 / 14: Metal shipped, not offered)
        raise AssertionError(f"the app offers GPU backend {offered!r}, the engine build ships {backend!r}")
    requested = bool(getattr(ext, "GPU_STATE", {}))
    registered = []
    libggml = getattr(ext, "_LIBGGML", None)
    if libggml is not None:
        import ctypes

        by_name = libggml.ggml_backend_reg_by_name
        by_name.restype = ctypes.c_void_p
        by_name.argtypes = [ctypes.c_char_p]
        registered = [n for n in ("Metal", "OpenCL") if by_name(n.encode("ascii"))]
    if registered and not requested:
        raise AssertionError(f"GPU backend {registered} registered although the GPU was never switched on")
    return {"backend": backend, "registered": registered, "requested": requested,
            "state": dict(getattr(ext, "GPU_STATE", {}) or {})}


def _client_send(model: str, messages: list, *, stream: bool, max_tokens: int) -> tuple:
    import unified_api_client as uac

    previous = os.environ.get("ENABLE_STREAMING")
    os.environ["ENABLE_STREAMING"] = "1" if stream else "0"
    try:
        client = uac.UnifiedClient(api_key="", model=model)
        result = client.send(messages, temperature=0.2, max_tokens=max_tokens)
    finally:
        if previous is None:
            os.environ.pop("ENABLE_STREAMING", None)
        else:
            os.environ["ENABLE_STREAMING"] = previous
    content = result[0] if isinstance(result, tuple) else getattr(result, "content", result)
    finish = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    return str(content or ""), finish


def fake_route_check(root: Any) -> dict:
    """The shared ``local/`` route end to end on this device, without a model download."""
    registry = dl.Registry(os.path.join(str(root), "models"))
    os.makedirs(registry.root, exist_ok=True)
    with open(registry.file_path("fake-echo"), "wb") as handle:
        handle.write(b"GGUF-fake")
    registry.add("fake-echo", {"repo": "glossarion/selftest", "file": "fake-echo.gguf", "revision": "local",
                               "sha256": "0" * 64, "size": 9, "downloaded_at": int(time.time())})
    engine = eng.FakeEngine(think=True)
    with _isolated_route():
        service = svc.LocalEngineService(prefs=_DictPrefs(), registry=registry, engine_factory=lambda: engine,
                                         watch=False, memory=lambda: (0, None))
        service.start()
        try:
            plain, _finish = _client_send("local/fake-echo", [{"role": "system", "content": "Translate."},
                                                              {"role": "user", "content": "안녕 world"}],
                                          stream=False, max_tokens=64)
            streamed, _finish = _client_send("local/fake-echo", [{"role": "user", "content": "stream check"}],
                                             stream=True, max_tokens=64)
            models = service.list_models()
        finally:
            service.stop()
    if plain.strip() != "Echo: 안녕 world":
        raise AssertionError(f"plain answer was {plain!r}")
    if streamed.strip() != "Echo: stream check":
        raise AssertionError(f"streamed answer was {streamed!r}")
    if "<think>" in plain + streamed:
        raise AssertionError("a think block leaked")
    return {"plain": plain, "streamed": streamed, "models": [m["id"] for m in models],
            "enable_thinking": engine.calls[-1]["params"].get("enable_thinking")}


def _preset(name: str) -> dict:
    for row in presets.DEFAULT_PRESETS:
        if row["name"].casefold() == name.casefold():
            return dict(row)
    raise AssertionError(f"no default preset called {name!r} (known: {[p['name'] for p in presets.DEFAULT_PRESETS]})")


def _progress(say: Any, label: str) -> Any:
    last = [0.0]

    def progress(update: dict) -> None:
        now = time.monotonic()
        if now - last[0] >= 10:
            last[0] = now
            say(f"llama suite: {label} {update.get('phase')} {update.get('bytes_done', 0)}/{update.get('bytes_total', 0)}")

    return progress


def _ensure_model(row: dict, registry: dl.Registry, say: Any, *, projector: bool = False) -> tuple:
    """``(stem, sha256_ok, downloaded)``: the preset's GGUF in ``registry`` (a verified file is reused), and
    with ``projector`` its vision projector too (the preset's ``mmproj``, sha256 checked)."""
    stem = registry.find(row["repo"], row["file"])
    sha_ok = False
    downloaded = False
    listing = None
    if stem:
        sha_ok = str(registry.load()[stem].get("sha256") or "") == row["sha256"]
    if not sha_ok:
        listing = hf.list_repo(hf.HfRef(row["repo"], row["file"]))
        file = hf.pick_file(listing.files, file=row["file"])
        if file.sha256 != row["sha256"]:
            raise AssertionError(f"Hugging Face lists sha256 {file.sha256} for {row['file']}, expected {row['sha256']}")
        stem = registry.stem_for(row["repo"], file.path)
        started = time.monotonic()
        entry = dl.download_blocking(dl.DownloadSpec(row["repo"], file.path, listing.commit, file.size, file.sha256,
                                                     stem), registry, progress=_progress(say, stem),
                                     should_stop=lambda: None)
        downloaded = True
        sha_ok = entry["sha256"] == row["sha256"]
        say(f"llama suite: downloaded {stem} in {time.monotonic() - started:.0f} s")
    if projector:
        have = (registry.load().get(stem) or {}).get("mmproj") or {}
        if str(have.get("sha256") or "") != row.get("mmproj_sha256"):
            listing = listing or hf.list_repo(hf.HfRef(row["repo"], row["file"]))
            file = hf.pick_projector(listing.projectors, row.get("mmproj"))
            if file is None or file.sha256 != row.get("mmproj_sha256"):
                raise AssertionError(f"Hugging Face lists {file and file.sha256} for {row.get('mmproj')}, "
                                     f"expected {row.get('mmproj_sha256')}")
            started = time.monotonic()
            entry = dl.download_blocking(dl.DownloadSpec(row["repo"], file.path, listing.commit, file.size, file.sha256,
                                                         registry.projector_key(stem), "mmproj", stem), registry,
                                         progress=_progress(say, f"{stem} projector"), should_stop=lambda: None)
            downloaded = True
            sha_ok = sha_ok and entry["sha256"] == row["mmproj_sha256"]
            say(f"llama suite: downloaded {stem}'s vision projector in {time.monotonic() - started:.0f} s")
    return stem, sha_ok, downloaded


def llama_suite(*, emit: Optional[Any] = None, log: Optional[Any] = None) -> dict:
    """Download (or reuse), load, generate, unload; returns the GLOSSARION_LLAMA payload."""
    say = log or (lambda text: None)
    row = _preset(os.environ.get(PRESET_ENV, "").strip() or "Garbage")
    registry = dl.Registry()
    stem, sha_ok, downloaded = _ensure_model(row, registry, say)
    prefs = _DictPrefs({svc.SETTINGS_KEY: {"n_ctx": SUITE_CTX}, presets.PREF_KEY: [row]})
    payload: dict = {"model": f"local/{stem}", "sha256_ok": sha_ok, "downloaded": downloaded}
    with _isolated_route():
        service = svc.LocalEngineService(prefs=prefs, registry=registry, watch=False)
        service.start()
        try:
            ok, error = service.load_blocking(stem, force=True)
            if not ok:
                raise AssertionError(f"load failed: {error}")
            payload["load_s"] = service.log[-1][1] if service.log else None
            started = time.monotonic()
            text, finish = _client_send(f"local/{stem}", [
                {"role": "system", "content": "You translate Korean to English. Reply with the translation only."},
                {"role": "user", "content": "안녕하세요, 만나서 반갑습니다."}], stream=False, max_tokens=SUITE_TOKENS)
            seconds = time.monotonic() - started
            usage = getattr(service.engine, "last_usage", {}) or {}
            tokens = int(usage.get("completion_tokens") or 0)
            payload.update({"tokens": tokens, "tok_s": round(tokens / seconds, 2) if seconds > 0 else None,
                            "text": text[:120], "finish": finish, "system_info": (eng.system_info() or "")[:200]})
        finally:
            service.stop()
    payload["ok"] = bool(payload.get("text", "").strip()) and sha_ok
    if emit is not None:
        emit(MARKER_LLAMA, payload)
    if not payload["ok"]:
        raise AssertionError(f"llama suite failed: {payload}")
    return payload


def sample_image_png(size: int = 96) -> bytes:
    """A tiny test picture: a red disc on white (what the vision check asks the model to describe)."""
    import io

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (size, size), (255, 255, 255))
    margin = size // 6
    ImageDraw.Draw(image).ellipse((margin, margin, size - margin, size - margin), fill=(220, 20, 20))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _client_send_image(model: str, png: bytes, *, max_tokens: int) -> tuple:
    """``UnifiedClient.send_image``: the shared path of the chat's Vision mode and the image jobs."""
    import base64

    import unified_api_client as uac

    previous = os.environ.get("ENABLE_STREAMING")
    os.environ["ENABLE_STREAMING"] = "0"
    try:
        client = uac.UnifiedClient(api_key="", model=model)
        result = client.send_image([{"role": "system", "content": "You describe pictures briefly."},
                                    {"role": "user", "content": "What shape and colour is in this picture? "
                                                                "Answer in a few words."}],
                                   base64.b64encode(png).decode("ascii"), temperature=0.2, max_tokens=max_tokens)
    finally:
        if previous is None:
            os.environ.pop("ENABLE_STREAMING", None)
        else:
            os.environ["ENABLE_STREAMING"] = previous
    content = result[0] if isinstance(result, tuple) else getattr(result, "content", result)
    finish = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    return str(content or ""), finish


def llama_vision_suite(*, emit: Optional[Any] = None, log: Optional[Any] = None) -> dict:
    """The smallest vision preset (``GLOSSARION_LLAMA_SUITE_PRESET``, default Garbage: Qwen3.5 0.8B and its
    mmproj-F16): download (or reuse) the model and its projector, load it, describe a tiny test picture
    through ``UnifiedClient.send_image`` -> ``local/`` -> libmtmd, unload; the GLOSSARION_LLAMA_VISION
    payload. Passes on a non-empty description with verified files (``red`` says whether it saw red)."""
    say = log or (lambda text: None)
    row = _preset(os.environ.get(PRESET_ENV, "").strip() or "Garbage")
    if not row.get("mmproj"):
        raise AssertionError(f"preset {row['name']} has no vision projector")
    available, reason = eng.vision_status()
    if not available:
        raise AssertionError(reason or eng.NO_MTMD)
    registry = dl.Registry()
    stem, sha_ok, downloaded = _ensure_model(row, registry, say, projector=True)
    prefs = _DictPrefs({svc.SETTINGS_KEY: {"n_ctx": SUITE_CTX}, presets.PREF_KEY: [row]})
    payload: dict = {"model": f"local/{stem}", "sha256_ok": sha_ok, "downloaded": downloaded,
                     "projector": row.get("mmproj")}
    png = sample_image_png()
    with _isolated_route():
        service = svc.LocalEngineService(prefs=prefs, registry=registry, watch=False)
        service.start()
        try:
            if not service.is_vision(stem):
                raise AssertionError(f"local/{stem} has no projector after the download")
            ok, error = service.load_blocking(stem, force=True)
            if not ok:
                raise AssertionError(f"load failed: {error}")
            started = time.monotonic()
            text, finish = _client_send_image(f"local/{stem}", png, max_tokens=SUITE_TOKENS)
            seconds = time.monotonic() - started
            usage = getattr(service.engine, "last_usage", {}) or {}
            payload.update({"image_px": 96, "seconds": round(seconds, 1), "text": text[:160], "finish": finish,
                            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                            "tokens": int(usage.get("completion_tokens") or 0),
                            "projector_loaded": bool(getattr(service.engine, "vision_loaded", False)),
                            "red": "red" in text.lower()})
        finally:
            service.stop()
    payload["ok"] = bool(payload.get("text", "").strip()) and sha_ok and bool(payload.get("projector_loaded"))
    if emit is not None:
        emit(MARKER_LLAMA_VISION, payload)
    if not payload["ok"]:
        raise AssertionError(f"llama vision check failed: {payload}")
    return payload
