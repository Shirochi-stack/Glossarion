"""Self-test bodies of the on-device engine (``diagnostics.selftest`` / ``diagnostics.e2e``).

* ``engine_check()`` (smoke ``llama_engine``): ``llama_cpp`` imports, its version is
  ``ENGINE_VERSION`` and ``llama_print_system_info()`` loads all four native libraries.
* ``fake_route_check(root)`` (e2e ``local_route_fake_engine``): ``LocalLLMServer`` + ``FakeEngine`` +
  ``UnifiedClient`` with ``local/fake-echo``, offline, plain and streaming.
* ``llama_suite()`` (suite ``llama``, ``glossarion://app/__selftest__?suite=llama``): downloads the
  preset named by ``GLOSSARION_LLAMA_SUITE_PRESET`` (default Garbage) into the models folder (a verified
  file is reused), loads it with ``n_ctx`` 2048, generates at most 16 tokens through ``UnifiedClient``,
  unloads, and prints ``GLOSSARION_LLAMA {...}``.

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

__all__ = ["MARKER_LLAMA", "PRESET_ENV", "engine_check", "fake_route_check", "llama_suite"]

MARKER_LLAMA = "GLOSSARION_LLAMA"
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
    return {"version": version, "system_info": info[:300]}


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


def llama_suite(*, emit: Optional[Any] = None, log: Optional[Any] = None) -> dict:
    """Download (or reuse), load, generate, unload; returns the GLOSSARION_LLAMA payload."""
    say = log or (lambda text: None)
    row = _preset(os.environ.get(PRESET_ENV, "").strip() or "Garbage")
    registry = dl.Registry()
    stem = registry.find(row["repo"], row["file"])
    sha_ok = False
    downloaded = False
    if stem:
        entry = registry.load()[stem]
        sha_ok = str(entry.get("sha256") or "") == row["sha256"]
    if not sha_ok:
        commit, files = hf.list_files(hf.HfRef(row["repo"], row["file"]))
        file = hf.pick_file(files, file=row["file"])
        if file.sha256 != row["sha256"]:
            raise AssertionError(f"Hugging Face lists sha256 {file.sha256} for {row['file']}, expected {row['sha256']}")
        stem = registry.stem_for(row["repo"], file.path)
        spec = dl.DownloadSpec(row["repo"], file.path, commit, file.size, file.sha256, stem)
        last = [0.0]

        def progress(update: dict) -> None:
            now = time.monotonic()
            if now - last[0] >= 10:
                last[0] = now
                say(f"llama suite: {update.get('phase')} {update.get('bytes_done', 0)}/{update.get('bytes_total', 0)}")

        started = time.monotonic()
        entry = dl.download_blocking(spec, registry, progress=progress, should_stop=lambda: None)
        downloaded = True
        sha_ok = entry["sha256"] == row["sha256"]
        say(f"llama suite: downloaded {stem} in {time.monotonic() - started:.0f} s")
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
