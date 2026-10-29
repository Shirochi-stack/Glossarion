"""Vision for on-device models (``services/local_llm``): the projector (``mmproj-*.gguf``) is listed and
picked on Hugging Face, downloaded in the same tap with the model's resume / sha256 rules and recorded
on it; the endpoint takes OpenAI ``image_url`` parts (``data:`` URLs) for a vision model, scales them
down with ``safe_image`` before the engine sees them, checks RAM with the projector and keeps the 400
for text-only models; ``UnifiedClient.send_image`` (the chat's Vision mode and image jobs) reaches it
through the ``local/`` route. FakeEngine and a fake Hugging Face; no network, no model.

Run from src/mobile:
    .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore tests_host/test_local_llm_vision.py
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os

import pytest

import local_llm_fakes as fakes
from glossarion_mobile.services.local_llm import downloads as dl
from glossarion_mobile.services.local_llm import engine as eng
from glossarion_mobile.services.local_llm import hf, presets
from glossarion_mobile.services.local_llm import service as svc
from glossarion_mobile.services.local_llm import vision as vis
from test_local_llm_server import _raw, make_service

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

REPO = "tester/Tiny-GGUF"
MODEL = "Tiny-Q4_K_M.gguf"
PAYLOAD = os.urandom(400_000)
PROJECTOR = os.urandom(150_000)


def png_data_url(width=1600, height=1200, fmt="PNG", mode="RGB") -> str:
    image = Image.new(mode, (width, height), (200, 30, 30) if mode == "RGB" else (200, 30, 30, 128))
    out = io.BytesIO()
    image.save(out, format=fmt)
    mime = {"PNG": "image/png", "WEBP": "image/webp", "JPEG": "image/jpeg"}[fmt]
    return f"data:{mime};base64," + base64.b64encode(out.getvalue()).decode()


def image_message(url, text="What is this?"):
    return {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}]}


def add_projector(service, stem, size=2_000):
    """Give a registered stem a projector file + its registry entry."""
    path = service.registry.file_path(service.registry.projector_key(stem))
    with open(path, "wb") as handle:
        handle.write(b"P" * size)
    assert service.registry.attach_projector(stem, {"repo": f"t/{stem}", "file": "mmproj-F16.gguf", "revision": "r",
                                                    "sha256": "1" * 64, "size": size, "downloaded_at": 1})


# ---- Hugging Face: projectors are listed apart and picked F16 first -------------------------------------


def test_list_repo_lists_projectors_apart_and_pick_projector_prefers_f16(monkeypatch):
    server = fakes.FakeHF()
    monkeypatch.setattr(hf, "HF_ENDPOINT", server.url)
    try:
        server.trees["unsloth/Qwen3.5-0.8B-GGUF"] = fakes.QWEN_08B_TREE + [("mmproj-F32.gguf", 402381664, "3" * 64)]
        listing = hf.list_repo(hf.HfRef("unsloth/Qwen3.5-0.8B-GGUF"))
    finally:
        server.close()
    assert listing.commit == fakes.FakeHF.COMMIT
    assert [p.path for p in listing.projectors] == ["mmproj-BF16.gguf", "mmproj-F16.gguf", "mmproj-F32.gguf"]
    assert not any(f.name.startswith("mmproj") for f in listing.files)
    f16 = hf.pick_projector(listing.projectors)
    assert f16.path == "mmproj-F16.gguf" and f16.size == 204987232
    assert f16.sha256 == presets.DEFAULT_PRESETS[0]["mmproj_sha256"]
    assert hf.pick_projector(listing.projectors, "mmproj-BF16.gguf").path == "mmproj-BF16.gguf"
    assert hf.pick_projector(listing.projectors, "mmproj-Q4.gguf") is None
    assert hf.pick_projector([]) is None
    q8 = [hf.HfFile("mmproj-model-Q8_0.gguf", 5), hf.HfFile("mmproj-model-f32.gguf", 9)]
    assert hf.pick_projector(q8).path == "mmproj-model-Q8_0.gguf"
    assert hf.pick_projector([hf.HfFile("mmproj-x.gguf", 9), hf.HfFile("mmproj-y.gguf", 3)]).path == "mmproj-y.gguf"


def test_default_presets_carry_their_vision_projector():
    for row in presets.DEFAULT_PRESETS:
        assert row["mmproj"] == "mmproj-F16.gguf" and row["mmproj_size"] > 100_000_000
        assert len(row["mmproj_sha256"]) == 64
    light = next(r for r in presets.DEFAULT_PRESETS if r["name"] == "Light")
    assert presets.download_size(light) == 1339752704  # the projector downloads on the first image
    prefs = fakes.FakePrefs()
    rows = presets.restore_defaults(prefs)
    assert rows[1]["mmproj_sha256"] == light["mmproj_sha256"]
    # renaming keeps the projector; another repository drops it
    clean, error = presets.validate_preset({"name": "Fast", "repo": light["repo"], "file": light["file"]}, rows, 1)
    assert error is None and clean["mmproj"] == "mmproj-F16.gguf" and clean["mmproj_size"] == 668227264
    clean, error = presets.validate_preset({"name": "Other", "repo": "someone/Else-GGUF"}, rows, 1)
    assert error is None and "mmproj" not in clean
    # a bad projector name is dropped, not trusted
    assert "mmproj" not in presets.save_presets(prefs, [dict(light, mmproj="../x.bin")])[0]


# ---- downloads: the projector rides the model's download machinery ------------------------------------------


def test_projector_download_is_recorded_on_its_model_and_deleted_with_it(tmp_path, monkeypatch):
    server = fakes.FakeHF()
    server.add(REPO, {MODEL: PAYLOAD, "mmproj-F16.gguf": PROJECTOR})
    monkeypatch.setattr(hf, "HF_ENDPOINT", server.url)
    try:
        registry = dl.Registry(str(tmp_path / "models"))
        listing = hf.list_repo(hf.HfRef(REPO))
        model = hf.pick_file(listing.files)
        stem = registry.stem_for(REPO, model.path)
        dl.download_blocking(dl.DownloadSpec(REPO, model.path, listing.commit, model.size, model.sha256, stem),
                             registry, progress=lambda u: None, should_stop=lambda: None)
        projector = hf.pick_projector(listing.projectors)
        key = registry.projector_key(stem)
        # paused: the sidecar says what it is, and Resume rebuilds the projector spec from it
        spec = dl.DownloadSpec(REPO, projector.path, listing.commit, projector.size, projector.sha256, key, "mmproj", stem)
        with pytest.raises(dl.DownloadCancelled):
            dl.download_blocking(spec, registry, progress=lambda u: None, should_stop=lambda: "pause")
        manager = dl.DownloadManager(registry=registry)
        assert [s.stem for s in manager.scan()] == [key]
        resumed = manager.pending_spec(key)
        assert (resumed.kind, resumed.owner, resumed.file) == ("mmproj", stem, "mmproj-F16.gguf")
        entry = dl.download_blocking(resumed, registry, progress=lambda u: None, should_stop=lambda: None)
    finally:
        server.close()
    assert entry["sha256"] == hashlib.sha256(PROJECTOR).hexdigest()
    loaded = registry.load()
    assert list(loaded) == [stem]  # the projector is no model of its own
    assert loaded[stem]["mmproj"]["file"] == "mmproj-F16.gguf" and loaded[stem]["mmproj"]["size"] == len(PROJECTOR)
    assert registry.projector_path(stem) == registry.file_path(f"{stem}.mmproj")
    with open(registry.projector_path(stem), "rb") as handle:
        assert handle.read() == PROJECTOR
    # a projector file that went away makes the model text-only again
    os.rename(registry.projector_path(stem), str(tmp_path / "moved"))
    assert "mmproj" not in registry.load()[stem] and registry.projector_path(stem) is None
    os.rename(str(tmp_path / "moved"), registry.file_path(f"{stem}.mmproj"))
    assert registry.remove(stem)
    assert not os.path.exists(registry.file_path(f"{stem}.mmproj")) and registry.load() == {}
    # a projector for a model that is gone is not kept
    assert not registry.attach_projector("gone", {"file": "mmproj-F16.gguf"})


# ---- the endpoint and the shared route ---------------------------------------------------------------------


@pytest.fixture()
def vision_service(tmp_path, monkeypatch):
    monkeypatch.delenv(svc.URL_ENV, raising=False)
    service, engine = make_service(tmp_path, stems=("seer", "blind"))
    add_projector(service, "seer")
    service.start()
    yield service, engine
    service.stop()
    svc.set_default_service(None)


def test_deleting_a_model_cancels_its_paused_image_support(vision_service):
    service, _engine = vision_service
    key = service.registry.projector_key("blind")
    part = service.registry.file_path(key) + dl.PART_SUFFIX
    with open(part, "wb") as handle:
        handle.write(b"P" * 100)
    with open(part + ".json", "w", encoding="utf-8") as handle:
        json.dump({"url": "x", "repo": "t/blind", "file": "mmproj-F16.gguf", "revision": "r", "size": 1000,
                   "sha256": None, "stem": key, "kind": "mmproj", "owner": "blind"}, handle)
    assert service.downloads.pending_spec(key).owner == "blind"
    assert service.delete("blind")
    assert not os.path.exists(part) and not os.path.exists(part + ".json")
    assert service.downloads.pending_spec(key) is None and key not in service.downloads.states


def test_models_say_which_ones_read_images(vision_service):
    service, _engine = vision_service
    data = json.loads(_raw(service, "GET", f"/{service.server.token}/v1/models")[1])["data"]
    assert {m["id"]: m["meta"]["vision"] for m in data} == {"blind": False, "seer": True}


def test_image_parts_reach_a_vision_model_scaled_down_as_jpeg(vision_service):
    import openai

    service, engine = vision_service
    client = openai.OpenAI(base_url=service.server.base_url, api_key="x")
    answer = client.chat.completions.create(model="local/seer", max_tokens=50, messages=[
        image_message(png_data_url(2400, 900, "WEBP"), "Describe")])
    assert answer.choices[0].message.content == "Echo: Describe [1 image]"
    # the engine loaded the model with its projector, and the projector only for the image request
    assert engine.mmproj.endswith("seer.mmproj.gguf") and engine.projector_loads == 1
    part = engine.calls[-1]["messages"][0]["content"][1]
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    with Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))) as seen:
        assert seen.format == "JPEG" and max(seen.size) == vis.MAX_SIDE and seen.size == (896, 336)
    # usage counts the image estimate; a text request afterwards needs nothing new
    assert answer.usage.prompt_tokens >= vis.IMAGE_TOKENS
    text = client.chat.completions.create(model="local/seer", messages=[{"role": "user", "content": "plain"}])
    assert text.choices[0].message.content == "Echo: plain" and len(engine.loads) == 1
    snap = service.snapshot()
    assert snap.vision and snap.model == "seer"
    # streaming works the same way, small images are not upscaled, transparency becomes white
    chunks = client.chat.completions.create(model="local/seer", stream=True, messages=[
        image_message(png_data_url(40, 30, "PNG", "RGBA"), "tiny")])
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "Echo: tiny [1 image]"
    url = engine.calls[-1]["messages"][0]["content"][1]["image_url"]["url"]
    with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as seen:
        assert seen.size == (40, 30) and seen.mode == "RGB"


def test_a_phone_photo_is_decoded_small_and_turned_upright(monkeypatch):
    # a portrait camera JPEG: landscape pixels + EXIF orientation 6 (rotate 90 degrees clockwise)
    photo = Image.new("RGB", (4000, 3000), (30, 30, 200))
    photo.paste((220, 20, 20), (0, 0, 4000, 300))  # the photo's top edge (right once turned upright)
    exif = Image.Exif()
    exif[0x0112] = 6
    out = io.BytesIO()
    photo.save(out, format="JPEG", exif=exif.tobytes())
    url = "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()
    from PIL import ImageOps

    decoded: list = []
    real_transpose = ImageOps.exif_transpose

    def transpose(image, *args, **kwargs):
        decoded.append(image.size)
        return real_transpose(image, *args, **kwargs)

    monkeypatch.setattr(ImageOps, "exif_transpose", transpose)
    prepared = vis.prepare_image_url(url)
    # libjpeg decoded at half scale (2000 x 1500), never below the 896 px the model gets
    assert decoded == [(2000, 1500)]
    with Image.open(io.BytesIO(base64.b64decode(prepared.split(",", 1)[1]))) as seen:
        assert seen.size == (672, 896)  # portrait, as the camera held it
        red, _green, blue = seen.convert("RGB").getpixel((640, 448))
        assert red > 150 and blue < 100  # the top band is on the right after a clockwise turn
        assert not seen.getexif().get(0x0112)


def test_image_requests_that_cannot_run_get_clear_400s(vision_service):
    service, engine = vision_service
    token = service.server.token

    def post(model, message):
        status, data = _raw(service, "POST", f"/{token}/v1/chat/completions", body={"model": model, "messages": [message]})
        return status, json.loads(data)["error"]["message"] if status != 200 else ""

    assert post("blind", image_message(png_data_url(20, 20))) == (400, "This on-device model has no vision support")
    assert post("seer", image_message("https://example.com/cat.png")) == (
        400, "On-device models take images as data: URLs only")
    assert post("seer", {"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}) == (
        400, "This on-device model cannot read input_audio parts")
    status, message = post("seer", image_message("data:image/png;base64,bm90IGFuIGltYWdl"))
    assert status == 400 and message.startswith("The image could not be read")
    # only the formats safe_image accepts are decoded (an EPS header never reaches Pillow's EPS plugin)
    eps = "data:application/postscript;base64," + base64.b64encode(b"%!PS-Adobe-3.0 EPSF-3.0\n").decode()
    status, message = post("seer", image_message(eps))
    assert status == 400 and message.startswith("The image could not be read")
    assert not [c for c in engine.calls if vis.image_count(c["messages"])]  # nothing reached the engine


def test_the_first_image_checks_ram_with_the_projector(tmp_path, monkeypatch):
    monkeypatch.delenv(svc.URL_ENV, raising=False)
    registry = dl.Registry(str(tmp_path / "m"))
    os.makedirs(registry.root, exist_ok=True)
    with open(registry.file_path("seer"), "wb") as handle:
        handle.write(b"GGUF")
    registry.add("seer", {"repo": "t/seer", "file": "seer.gguf", "revision": "r", "sha256": "0" * 64,
                          "size": 500 * 1024 ** 2, "downloaded_at": 1})
    engine = eng.FakeEngine()
    memory = {"value": (8 * 1024 ** 3, int(1.6 * 1024 ** 3))}
    service = svc.LocalEngineService(prefs=fakes.FakePrefs({svc.SETTINGS_KEY: {"n_ctx": 2048}}), registry=registry,
                                     engine_factory=lambda: engine, watch=False, platform="android",
                                     memory=lambda: memory["value"])
    add_projector(service, "seer", size=10)
    entries = registry.load()
    entries["seer"]["mmproj"]["size"] = 900 * 1024 ** 2  # a 0.9 GB projector
    registry.save(entries)
    assert service.ram_check("seer")[0] == "ok"
    verdict, message = service.ram_check("seer", vision=True)
    assert verdict == "confirm" and "seer with its image support needs about" in message
    # text loads; the first image loads the projector too: tight on free RAM is a warning only (owner 2026-10-27)
    run = service.chat("seer", {"messages": [{"role": "user", "content": "hi"}]}, lambda: False)
    with run:
        assert "".join(run.deltas) == "Echo: hi"
    assert engine.projector_loads == 0
    run = service.chat("seer", {"messages": [image_message(png_data_url(30, 30), "ok now")]}, lambda: False)
    with run:
        assert "".join(run.deltas) == "Echo: ok now [1 image]"
    assert engine.projector_loads == 1
    # more than the phone has is still refused
    service.unload()
    memory["value"] = (int(1.2 * 1024 ** 3), int(1.0 * 1024 ** 3))
    with pytest.raises(svc.LocalLLMError) as info:
        service.chat("seer", {"messages": [image_message(png_data_url(30, 30))]}, lambda: False)
    assert info.value.status == 507 and info.value.message.startswith("This phone has")


def test_a_context_overflow_from_the_images_is_a_400_not_a_broken_stream(vision_service, monkeypatch):
    service, engine = vision_service

    def overflow(messages, params, should_stop):
        raise eng.ContextOverflow("Prompt exceeds n_ctx: 9000 > 8192")
        yield ""  # pragma: no cover

    monkeypatch.setattr(engine, "chat", overflow)
    for stream in (False, True):
        status, data = _raw(service, "POST", f"/{service.server.token}/v1/chat/completions",
                            body={"model": "seer", "stream": stream, "messages": [image_message(png_data_url(30, 30))]})
        error = json.loads(data)["error"]
        assert status == 400 and error["code"] == "context_length_exceeded"
        assert error["message"].startswith("This model's maximum context length is")
    assert not service.gate.busy  # the engine was given back


@pytest.mark.parametrize("stream", [False, True])
def test_send_image_reaches_a_vision_model_through_the_local_route(vision_service, monkeypatch, stream):
    """``UnifiedClient.send_image``: what the chat's Vision mode and the image jobs call."""
    import unified_api_client as uac

    service, engine = vision_service
    monkeypatch.setenv("ENABLE_STREAMING", "1" if stream else "0")
    png = io.BytesIO()
    Image.new("RGB", (1200, 1800), (10, 200, 10)).save(png, format="PNG")
    client = uac.UnifiedClient(api_key="", model="local/seer")
    result = client.send_image([{"role": "system", "content": "Describe images."},
                                {"role": "user", "content": "What is in it?"}],
                               base64.b64encode(png.getvalue()).decode(), temperature=0.2, max_tokens=64)
    content = result[0] if isinstance(result, tuple) else result.content
    assert "[1 image]" in content
    sent = engine.calls[-1]["messages"]
    images = [p for m in sent if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image_url"]
    assert len(images) == 1
    with Image.open(io.BytesIO(base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1]))) as seen:
        assert seen.size == (597, 896)


@pytest.mark.parametrize("stream", [False, True])
def test_send_image_to_a_text_only_model_fails_at_once_with_the_clear_message(vision_service, monkeypatch, stream):
    """The 400 "no vision support" cannot change on a retry: one request, the endpoint's own message."""
    import time

    import unified_api_client as uac

    service, _engine = vision_service
    monkeypatch.setenv("ENABLE_STREAMING", "1" if stream else "0")
    calls = []
    chat = service.chat
    monkeypatch.setattr(service, "chat", lambda *a, **k: calls.append(a[0]) or chat(*a, **k))
    png = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 200, 10)).save(png, format="PNG")
    client = uac.UnifiedClient(api_key="", model="local/blind")
    started = time.monotonic()
    with pytest.raises(uac.UnifiedClientError) as caught:
        client.send_image([{"role": "user", "content": "What is in it?"}], base64.b64encode(png.getvalue()).decode(),
                          temperature=0.2, max_tokens=64)
    assert "This on-device model has no vision support" in str(caught.value)
    assert len(calls) == 1 and time.monotonic() - started < 10


def test_a_projector_downloaded_while_loaded_text_only_reloads_with_it(tmp_path, monkeypatch):
    server = fakes.FakeHF()
    server.add(REPO, {MODEL: PAYLOAD, "mmproj-F16.gguf": PROJECTOR})
    monkeypatch.setattr(hf, "HF_ENDPOINT", server.url)
    monkeypatch.delenv(svc.URL_ENV, raising=False)
    try:
        service, engine = make_service(tmp_path, stems=())

        async def scenario():
            listing = await service.resolve(hf.HfRef(REPO))
            model = hf.pick_file(listing.files)
            assert (await service.download(service.spec_for(REPO, listing.commit, model))).done
            stem = service.registry.find(REPO, model.path)
            assert service.load_blocking(stem) == (True, None) and not engine.vision
            spec = service.projector_spec_for(stem, REPO, listing.commit, hf.pick_projector(listing.projectors))
            assert (await service.download(spec)).done
            return stem

        stem = asyncio.run(scenario())
    finally:
        server.close()
    assert service.is_vision(stem) and service.loaded_stem is None  # unloaded: the next request loads both
    run = service.chat(stem, {"messages": [image_message(png_data_url(30, 30), "now")]}, lambda: False)
    with run:
        assert "".join(run.deltas) == "Echo: now [1 image]"
    assert engine.mmproj.endswith(f"{stem}.mmproj.gguf")


def test_vision_status_without_libmtmd(monkeypatch, tmp_path):
    """An engine without libmtmd (pypi.flet.dev's build 2) says so instead of crashing on an image."""
    monkeypatch.setattr(eng, "_VISION", [])

    def fail(name):
        raise FileNotFoundError("Shared library with base name 'mtmd' not found")

    monkeypatch.setattr(eng.importlib, "import_module", fail)
    available, reason = eng.vision_status()
    assert not available and reason.startswith(eng.NO_MTMD) and "mtmd" in reason
    monkeypatch.setattr(eng, "_VISION", [])


def test_the_llama_suite_describes_an_image_and_the_smoke_check_needs_libmtmd(monkeypatch):
    from glossarion_mobile.diagnostics import selftest
    from glossarion_mobile.services.local_llm import selfcheck

    assert [name for name, _fn in selftest.SUITES["llama"]] == ["llama_download_generate", "llama_vision_describe"]
    png = selfcheck.sample_image_png()
    with Image.open(io.BytesIO(png)) as image:
        assert image.size == (96, 96) and image.getpixel((48, 48))[0] > 200 and image.getpixel((1, 1)) == (255, 255, 255)
    assert selfcheck.MARKER_LLAMA_VISION == "GLOSSARION_LLAMA_VISION"
    monkeypatch.setattr(eng, "vision_status", lambda: (False, "no libmtmd"))
    monkeypatch.setattr(selfcheck.importlib, "import_module",
                        lambda name: type("M", (), {"__version__": eng.ENGINE_VERSION, "GPU_STATE": {}, "_LIBGGML": None,
                                                    "gpu_backend_name": staticmethod(lambda: None)})())
    monkeypatch.setattr(eng, "system_info", lambda: "CPU : SSE3 = 1")
    with pytest.raises(AssertionError, match="no libmtmd"):
        selfcheck.engine_check()
    monkeypatch.setattr(eng, "vision_status", lambda: (True, None))
    assert selfcheck.engine_check()["vision"] is True
    # a GGML_BACKEND_DL build that registered no CPU backend fails the smoke check
    monkeypatch.setattr(eng, "system_info", lambda: "")
    with pytest.raises(AssertionError, match="llama_print_system_info failed"):
        selfcheck.engine_check()
    monkeypatch.setattr(eng, "system_info", lambda: "RPC : 1 |")
    with pytest.raises(AssertionError, match="no CPU backend registered"):
        selfcheck.engine_check()


def test_system_info_names_the_runtime_selected_cpu_variant(monkeypatch):
    """Android arm64 registers one of four CPU backends at import; the engine card says which."""
    modules = {"llama_cpp": type("L", (), {"llama_print_system_info": staticmethod(
                   lambda: b"CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | DOTPROD = 1 | REPACK = 1 | ")})(),
               "llama_cpp._ctypes_extensions": type("E", (), {"CPU_BACKEND_VARIANT": "android_armv8.6_1"})()}

    def fake_import(name):
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr(eng.importlib, "import_module", fake_import)
    assert eng.cpu_variant() == "android_armv8.6_1"
    assert eng.system_info() == ("CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | DOTPROD = 1 | "
                                 "REPACK = 1 | (CPU variant android_armv8.6_1)")
    modules["llama_cpp._ctypes_extensions"].CPU_BACKEND_VARIANT = "builtin"   # x86_64 / iOS: nothing to add
    assert eng.system_info().endswith("REPACK = 1 |")
    del modules["llama_cpp._ctypes_extensions"]                                # pypi.flet.dev's loader
    assert eng.cpu_variant() is None and eng.system_info().endswith("REPACK = 1 |")
    del modules["llama_cpp"]
    assert eng.system_info() is None
