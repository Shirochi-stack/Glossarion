"""Mobile-compat patches in the API / auth layer (Glossarion Mobile, milestone U1).

Covered patches (build-ci-compat P18-P21, P26 and the key setters):
  * P18  unified_api_client optional SDK/route imports: on mobile they survive
         non-ImportError load failures (e.g. OSError from a native library) with
         the same fallbacks; desktop still catches ImportError only
  * P19  http_logger: GLOSSARION_HTTP_LOG=0 off switch, GLOSSARION_HTTP_LOG_DIR
  * P20  unified_api_client Payloads dir via mobile_runtime.data_dir()
  * P21  multi-key config fallback honours CONFIG_FILE on mobile only (refusal
         patterns already honoured it on desktop)
  * P26  duplicate_detection_config: RapidFuzz Jaro-Winkler when jellyfish is
         missing, on mobile only
  * api_key_encryption.set_key_material / token_encryption.set_symmetric_key
  * authgpt_auth begin_oauth() / complete_from_redirect() split, paste fallback,
    GLOSSARION_OAUTH_RETURN_URL success-page seam

Every gate is also checked for its desktop default (mobile env vars unset).
No network: the OAuth token endpoint is faked and the "browser" is http.client
talking to the loopback server on a free port.
"""
import ast
import base64
import builtins
import hashlib
import http.client
import importlib
import json
import os
import random
import socket
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MOBILE_ENV = (
    "GLOSSARION_MOBILE",
    "GLOSSARION_NO_PROCESSES",
    "GLOSSARION_DATA_DIR",
    "CONFIG_FILE",
    "GLOSSARION_HTTP_LOG",
    "GLOSSARION_HTTP_LOG_DIR",
    "GLOSSARION_API_KEY_FERNET",
    "GLOSSARION_TOKEN_KEY_B64",
    "GLOSSARION_OAUTH_RETURN_URL",
    "FLET_PLATFORM",
)


@pytest.fixture(autouse=True)
def _desktop_env(monkeypatch):
    """Every test starts from the desktop environment (no mobile env vars)."""
    for name in MOBILE_ENV:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# P19 http_logger
# ---------------------------------------------------------------------------

def _run_http_logger_probe(tmp_path, env, call_args=""):
    """enable_detailed_http_logging() in a subprocess (it patches requests globally).

    ``http_logger.__file__`` is pointed into tmp_path so the desktop default
    folder (next to the module) is created there, not in src/.
    """
    code = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(SRC)!r})
        import http_logger
        http_logger.__file__ = {str(tmp_path / "http_logger.py")!r}
        import requests
        original = requests.Session.request
        result = http_logger.enable_detailed_http_logging({call_args})
        print(json.dumps({{
            "result": None if result is None else str(result),
            "patched": http_logger._patched,
            "requests_patched": requests.Session.request is not original,
        }}))
        """
    )
    child_env = {k: v for k, v in os.environ.items() if k not in MOBILE_ENV}
    child_env.update(env)
    child_env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=child_env, cwd=str(tmp_path), capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_http_logger_desktop_default_logs_next_to_module(tmp_path):
    info = _run_http_logger_probe(tmp_path, {})
    assert info == {
        "result": str(tmp_path / "http_requests"),
        "patched": True,
        "requests_patched": True,
    }
    assert (tmp_path / "http_requests").is_dir()


def test_http_logger_off_switch_patches_nothing(tmp_path):
    info = _run_http_logger_probe(tmp_path, {"GLOSSARION_HTTP_LOG": "0"})
    assert info == {"result": None, "patched": False, "requests_patched": False}
    assert not (tmp_path / "http_requests").exists()


def test_http_logger_dir_override(tmp_path):
    target = tmp_path / "data" / "logs" / "http_requests"
    info = _run_http_logger_probe(
        tmp_path, {"GLOSSARION_HTTP_LOG": "1", "GLOSSARION_HTTP_LOG_DIR": str(target)}
    )
    assert info["result"] == str(target) and info["patched"] is True
    assert target.is_dir()
    assert not (tmp_path / "http_requests").exists()


def test_http_logger_absolute_argument_beats_dir_override(tmp_path):
    explicit = tmp_path / "explicit"
    info = _run_http_logger_probe(
        tmp_path,
        {"GLOSSARION_HTTP_LOG_DIR": str(tmp_path / "ignored")},
        call_args=repr(str(explicit)),
    )
    assert info["result"] == str(explicit)
    assert not (tmp_path / "ignored").exists()


def test_http_logger_off_switch_also_stops_an_enabled_logger(tmp_path, monkeypatch):
    import http_logger

    monkeypatch.setattr(http_logger, "_log_folder", tmp_path)

    def save():
        http_logger._save_http_log("GET", "https://example.test/v1", {"Authorization": "x"}, None, 200, {}, "{}")

    save()
    assert len(list(tmp_path.glob("http_*.json"))) == 1
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    save()
    assert len(list(tmp_path.glob("http_*.json"))) == 1


# ---------------------------------------------------------------------------
# unified_api_client: P18, P20, P21
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def uac():
    # Import with the HTTP logger off so this process's requests stay unpatched.
    saved = os.environ.get("GLOSSARION_HTTP_LOG")
    os.environ["GLOSSARION_HTTP_LOG"] = "0"
    try:
        module = importlib.import_module("unified_api_client")
    finally:
        if saved is None:
            os.environ.pop("GLOSSARION_HTTP_LOG", None)
        else:
            os.environ["GLOSSARION_HTTP_LOG"] = saved
    return module


_OPTIONAL_IMPORT_ROOTS = {
    "openai", "httpx", "google", "grpc_gemini_client", "anthropic", "cohere", "mistralai",
    "deepl", "authgpt_auth", "authgrok_auth", "authgem_auth", "ocagy_cli", "antigravity_proxy",
    "glm_proxy", "authnd_auth", "gemini_free", "opera_aria", "authcd_auth",
}


def _optional_import_blocks():
    source = (SRC / "unified_api_client.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    start = lines.index("# OpenAI SDK") + 1
    end = next(i for i, line in enumerate(lines) if line.startswith("from functools import lru_cache")) + 1
    blocks = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Try) or not (start <= node.lineno <= end):
            continue
        roots = set()
        for stmt in node.body:
            if isinstance(stmt, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in stmt.names)
            elif isinstance(stmt, ast.ImportFrom) and stmt.module:
                roots.add(stmt.module.split(".")[0])
        if roots & _OPTIONAL_IMPORT_ROOTS:
            blocks.append((node, ast.get_source_segment(source, node)))
    return blocks


def _all_handlers(try_node):
    for handler in try_node.handlers:
        yield handler
        for stmt in handler.body:
            if isinstance(stmt, ast.Try):
                yield from _all_handlers(stmt)


def _failing_import(name, globals=None, locals=None, fromlist=(), level=0):
    raise OSError(f"simulated native library load failure: {name}")


def _missing_import(name, globals=None, locals=None, fromlist=(), level=0):
    raise ModuleNotFoundError(f"No module named {name!r}")


def _exec_optional_import(segment, failing_import, mobile_failure):
    namespace = {
        "__builtins__": dict(vars(builtins), __import__=failing_import),
        "__name__": "optional_import_probe",
        "_MOBILE_IMPORT_FAILURE": mobile_failure,
    }
    exec(compile(segment, "<unified_api_client optional import>", "exec"), namespace)
    return namespace


def test_optional_imports_use_the_gated_handler_and_keep_fallbacks():
    blocks = _optional_import_blocks()
    # openai, httpx, genai, grpc, anthropic, cohere, mistral, deepl, google translate
    # + 11 route modules (authgpt ... authcd)
    assert len(blocks) == 20
    for node, segment in blocks:
        for handler in _all_handlers(node):
            # (ImportError, ...) also keeps the collector's "guarded import" detection.
            assert ast.unparse(handler.type) == "(ImportError, _MOBILE_IMPORT_FAILURE)", segment
        # Desktop (_MOBILE_IMPORT_FAILURE is ImportError): a missing module takes the
        # fallback, any other import-time failure propagates exactly as before.
        desktop = _exec_optional_import(segment, _missing_import, ImportError)
        with pytest.raises(OSError):
            _exec_optional_import(segment, _failing_import, ImportError)
        # Mobile (_MOBILE_IMPORT_FAILURE is Exception): the same fallbacks for any Exception.
        mobile = _exec_optional_import(segment, _failing_import, Exception)
        for namespace in (desktop, mobile):
            for handler in _all_handlers(node):
                for stmt in handler.body:
                    if (
                        isinstance(stmt, ast.Assign)
                        and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                        and isinstance(stmt.value, ast.Constant)
                    ):
                        name = stmt.targets[0].id
                        assert namespace[name] == stmt.value.value, (name, segment)
                    elif isinstance(stmt, ast.ClassDef):
                        assert issubclass(namespace[stmt.name], Exception)


def test_optional_imports_desktop_values(uac):
    assert uac._MOBILE_IMPORT_FAILURE is ImportError
    assert uac.AUTHGPT_AVAILABLE is True
    assert uac._authgpt_send is not None
    assert uac.MistralSDKStyle in (None, "legacy", "modern")


_BROKEN_IMPORT_PROBE = textwrap.dedent(
    """
    import importlib.abc, importlib.machinery, json, sys
    sys.path.insert(0, {src!r})

    class _Broken(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        # Import of these modules raises AttributeError (e.g. a protobuf/pydantic mismatch).
        names = {names!r}

        def find_spec(self, fullname, path=None, target=None):
            if fullname in self.names:
                return importlib.machinery.ModuleSpec(fullname, self)
            return None

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            raise AttributeError("simulated broken module " + module.__name__)

    sys.meta_path.insert(0, _Broken())
    try:
        import unified_api_client as uac
    except AttributeError as exc:
        print(json.dumps({{"imported": False, "error": str(exc)}}))
    else:
        print(json.dumps({{"imported": True, "deepl": uac.DEEPL_AVAILABLE, "authgpt": uac.AUTHGPT_AVAILABLE}}))
    """
)


@pytest.mark.parametrize("mobile", [False, True], ids=["desktop", "mobile"])
def test_broken_optional_module_fails_loudly_only_on_desktop(mobile, tmp_path):
    code = _BROKEN_IMPORT_PROBE.format(src=str(SRC), names=["deepl", "authgpt_auth"])
    env = {k: v for k, v in os.environ.items() if k not in MOBILE_ENV}
    env.update(PYTHONIOENCODING="utf-8", GLOSSARION_HTTP_LOG="0")
    if mobile:
        env["GLOSSARION_MOBILE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=str(tmp_path), capture_output=True, text=True,
        encoding="utf-8", timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    if mobile:
        assert result == {"imported": True, "deepl": False, "authgpt": False}
    else:
        assert result["imported"] is False and "simulated broken module" in result["error"]


def test_payloads_dir_desktop_default_is_next_to_module(uac, tmp_path, monkeypatch):
    assert not getattr(sys, "frozen", False)
    monkeypatch.setattr(uac, "_payloads_resolved_dir", None)
    monkeypatch.setattr(uac, "_PAYLOADS_DISABLED", False)
    monkeypatch.setattr(uac, "__file__", str(tmp_path / "unified_api_client.py"))
    assert uac._payloads_dir() == str(tmp_path / "Payloads")


def test_payloads_dir_uses_glossarion_data_dir(uac, tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("GLOSSARION_DATA_DIR", str(data))
    monkeypatch.setattr(uac, "_payloads_resolved_dir", None)
    monkeypatch.setattr(uac, "_PAYLOADS_DISABLED", False)
    monkeypatch.setattr(uac, "__file__", str(tmp_path / "bundle" / "unified_api_client.py"))
    assert uac._payloads_dir() == str(data / "Payloads")
    assert not (tmp_path / "bundle").exists()


def _write_config(path, keys, patterns):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"use_multi_api_keys": True, "multi_api_keys": keys, "refusal_patterns": patterns}),
        encoding="utf-8",
    )


def test_multi_key_fallback_desktop_default_reads_config_next_to_module(uac, tmp_path, monkeypatch):
    _write_config(tmp_path / "config.json", [{"api_key": "desktop", "model": "m"}], ["desktop refusal"])
    monkeypatch.setattr(uac, "__file__", str(tmp_path / "unified_api_client.py"))
    assert uac.UnifiedClient._load_multi_keys_from_config_file() == [{"api_key": "desktop", "model": "m"}]
    assert uac.UnifiedClient._get_refusal_patterns(None) == ["desktop refusal"]


def test_multi_key_fallback_honours_config_file_only_on_mobile(uac, tmp_path, monkeypatch):
    _write_config(tmp_path / "bundle" / "config.json", [{"api_key": "bundle"}], ["bundle refusal"])
    mobile_config = tmp_path / "data" / "config.json"
    _write_config(mobile_config, [{"api_key": "mobile"}], ["mobile refusal"])
    monkeypatch.setattr(uac, "__file__", str(tmp_path / "bundle" / "unified_api_client.py"))
    monkeypatch.setenv("CONFIG_FILE", str(mobile_config))
    # Desktop: an unrelated CONFIG_FILE is ignored by the multi-key fallback, as on HEAD;
    # _get_refusal_patterns honoured CONFIG_FILE on desktop already (unchanged).
    assert uac.UnifiedClient._load_multi_keys_from_config_file() == [{"api_key": "bundle"}]
    assert uac.UnifiedClient._get_refusal_patterns(None) == ["mobile refusal"]
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    assert uac.UnifiedClient._load_multi_keys_from_config_file() == [{"api_key": "mobile"}]
    assert uac.UnifiedClient._get_refusal_patterns(None) == ["mobile refusal"]
    # Mobile without CONFIG_FILE: <GLOSSARION_DATA_DIR>/config.json
    monkeypatch.delenv("CONFIG_FILE")
    monkeypatch.setenv("GLOSSARION_DATA_DIR", str(tmp_path / "data"))
    assert uac.UnifiedClient._load_multi_keys_from_config_file() == [{"api_key": "mobile"}]


# ---------------------------------------------------------------------------
# P26 duplicate_detection_config
# ---------------------------------------------------------------------------

_BASE_NAMES = [
    "Kim Sang-hyun", "Kim Sanghyun", "Park Ji-sung", "Ji-sung Park", "Lee Min-ho", "Choi Yu-na",
    "Catherine", "Katherine", "Kathryn", "Jonathan", "Johnathan", "Alexander", "Aleksandr",
    "Arthur Pendragon", "Gwendolyn", "Seraphina", "Lucien", "Lucian", "Elizabeth", "Elisabeth",
    "Al", "Bo", "X", "Ian", "Ivy",
    "김상현", "김상현님", "박지성", "이민호", "최유나", "한서준", "윤하늘",
    "田中太郎", "佐藤花子", "鈴木一郎", "王小明", "李华", "张伟", "陈静",
    "ルナ", "アリス", "Ålesund", "Zoë", "Renée",
]


def _name_pairs(count=200):
    rng = random.Random(1455)
    pairs = []
    while len(pairs) < count:
        a = rng.choice(_BASE_NAMES)
        mode = rng.randrange(6)
        if mode == 0:
            b = a
        elif mode == 1:
            b = rng.choice(_BASE_NAMES)
        elif mode == 2 and len(a) > 1:
            i = rng.randrange(len(a))
            b = a[:i] + a[i + 1:]
        elif mode == 3 and len(a) > 1:
            i = rng.randrange(len(a) - 1)
            b = a[:i] + a[i + 1] + a[i] + a[i + 2:]
        elif mode == 4:
            b = a + rng.choice(["님", "-san", " Jr.", "씨", "様", "a"])
        else:
            b = a.replace("-", "") if "-" in a else a.lower()
        if a and b:
            pairs.append((a, b))
    return pairs


@pytest.fixture
def ddc(monkeypatch):
    monkeypatch.setenv("GLOSSARY_DUPLICATE_ALGORITHM", "auto")
    monkeypatch.setenv("GLOSSARY_FUZZY_THRESHOLD", "0.90")
    monkeypatch.delenv("GLOSSARY_PARTIAL_RATIO_WEIGHT", raising=False)
    return importlib.import_module("duplicate_detection_config")


@pytest.fixture
def reload_ddc(ddc, monkeypatch):
    """Re-import duplicate_detection_config under a given env (mobile flag, jellyfish blocked).

    The module picks its Jaro-Winkler source at import time; it is reloaded
    under the desktop environment again afterwards.
    """
    def load(*, mobile, block_jellyfish):
        if mobile:
            monkeypatch.setenv("GLOSSARION_MOBILE", "1")
        else:
            monkeypatch.delenv("GLOSSARION_MOBILE", raising=False)
        if block_jellyfish:
            monkeypatch.setitem(sys.modules, "jellyfish", None)
        return importlib.reload(ddc)

    yield load
    monkeypatch.undo()  # env and sys.modules["jellyfish"] back first, then re-import
    importlib.reload(ddc)


def test_rapidfuzz_jaro_winkler_matches_jellyfish_on_200_names():
    jellyfish = pytest.importorskip("jellyfish")
    from rapidfuzz.distance import JaroWinkler

    pairs = _name_pairs()
    assert len(pairs) == 200
    for a, b in pairs:
        assert abs(jellyfish.jaro_winkler_similarity(a, b) - JaroWinkler.similarity(a, b)) <= 1e-9, (a, b)


def test_similarity_fallback_path_equals_jellyfish_path(reload_ddc, monkeypatch):
    pytest.importorskip("jellyfish")
    ddc = reload_ddc(mobile=True, block_jellyfish=False)
    assert ddc._HAS_JELLYFISH and ddc._rf_jaro_winkler is not None
    pairs = _name_pairs()
    jw_only = {"algorithms": ["jaro_winkler"], "partial_ratio_weight": 0.45}
    auto = ddc.get_duplicate_detection_config()
    with_jellyfish = [
        (ddc.calculate_similarity_with_config(a, b, jw_only), ddc.calculate_similarity_with_config(a, b, auto))
        for a, b in pairs
    ]
    monkeypatch.setattr(ddc, "_HAS_JELLYFISH", False)
    monkeypatch.setattr(ddc, "_jellyfish", None)
    assert ddc.get_duplicate_detection_config()["algorithms"] == auto["algorithms"]
    with_rapidfuzz = [
        (ddc.calculate_similarity_with_config(a, b, jw_only), ddc.calculate_similarity_with_config(a, b, auto))
        for a, b in pairs
    ]
    for (pair, jf, rf) in zip(pairs, with_jellyfish, with_rapidfuzz):
        assert abs(jf[0] - rf[0]) <= 1e-9 and abs(jf[1] - rf[1]) <= 1e-9, pair


def test_rapidfuzz_path_works_without_jellyfish_on_mobile(reload_ddc):
    ddc = reload_ddc(mobile=True, block_jellyfish=True)
    assert not ddc._HAS_JELLYFISH and ddc._rf_jaro_winkler is not None
    config = ddc.get_duplicate_detection_config()
    assert "jaro_winkler" in config["algorithms"]
    score = ddc.calculate_similarity_with_config("Catherine", "Katherine", {"algorithms": ["jaro_winkler"]})
    assert abs(score - 25 / 27) <= 1e-9
    assert ddc.get_algorithm_display_info()[-1] == "Jaro-Winkler (RapidFuzz)"


def test_desktop_without_jellyfish_drops_jaro_winkler_as_before(reload_ddc, monkeypatch):
    ddc = reload_ddc(mobile=False, block_jellyfish=True)
    assert not ddc._HAS_JELLYFISH and ddc._rf_jaro_winkler is None
    for preset in ("auto", "aggressive", "no-such-preset"):
        monkeypatch.setenv("GLOSSARY_DUPLICATE_ALGORITHM", preset)
        assert "jaro_winkler" not in ddc.get_duplicate_detection_config()["algorithms"], preset
    assert "Jaro-Winkler" not in " ".join(ddc.get_algorithm_display_info())
    monkeypatch.setenv("GLOSSARY_DUPLICATE_ALGORITHM", "auto")
    config = ddc.get_duplicate_detection_config()
    # The reviewer's probe: these stay below the 0.90 threshold on desktop without jellyfish.
    assert ddc.calculate_similarity_with_config("Kim Minjun", "Kim Minjoon", config) < 0.9
    assert ddc.calculate_similarity_with_config("Catherine", "Katherine", {"algorithms": ["jaro_winkler"]}) == 0.0


def test_duplicate_detection_desktop_defaults(ddc, monkeypatch):
    assert ddc._rf_jaro_winkler is None  # RapidFuzz Jaro-Winkler is a mobile-only source
    if ddc._HAS_JELLYFISH:
        assert ddc.get_duplicate_detection_config()["algorithms"] == ["basic", "token_sort", "partial", "jaro_winkler"]
        assert ddc.get_algorithm_display_info() == ["RapidFuzz", "Jaro-Winkler"]
    # Neither library: jaro_winkler is dropped exactly as before the fallback existed.
    monkeypatch.setattr(ddc, "_HAS_JELLYFISH", False)
    monkeypatch.setattr(ddc, "_jellyfish", None)
    monkeypatch.setattr(ddc, "_rf_jaro_winkler", None)
    assert "jaro_winkler" not in ddc.get_duplicate_detection_config()["algorithms"]
    assert "Jaro-Winkler" not in " ".join(ddc.get_algorithm_display_info())
    assert ddc.calculate_similarity_with_config("Catherine", "Katherine", {"algorithms": ["jaro_winkler"]}) == 0.0


# ---------------------------------------------------------------------------
# api_key_encryption.set_key_material
# ---------------------------------------------------------------------------

@pytest.fixture
def ake(monkeypatch):
    pytest.importorskip("cryptography")
    module = importlib.import_module("api_key_encryption")
    monkeypatch.setattr(module, "_key_material", None)
    monkeypatch.setattr(module, "_handler", None)
    return module


def _fernet_key():
    from cryptography.fernet import Fernet
    return Fernet.generate_key()


def _forbid_key_file(module, monkeypatch):
    def no_key_file(self):
        raise AssertionError("the key file must not be used")
    monkeypatch.setattr(module.APIKeyEncryption, "_get_or_create_cipher", no_key_file)


def test_api_key_desktop_default_uses_key_file_next_to_module(ake, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(ake.APIKeyEncryption, "_get_or_create_cipher", lambda self: sentinel)
    handler = ake.APIKeyEncryption()
    assert handler.cipher is sentinel
    name = "glossarion_key.txt" if sys.platform == "darwin" else ".glossarion_key"
    assert handler.key_file == Path(ake.__file__).parent / name


def test_set_key_material_replaces_key_file(ake, monkeypatch):
    _forbid_key_file(ake, monkeypatch)
    key = _fernet_key()
    ake.set_key_material(key)
    assert key.decode() not in "".join(os.environ.values())
    handler = ake.APIKeyEncryption()
    assert handler.key_file is None
    token = handler.encrypt_value("sk-mobile-123")
    assert token.startswith("ENC:") and handler.decrypt_value(token) == "sk-mobile-123"

    ake.set_key_material(_fernet_key())
    other = ake.APIKeyEncryption()
    assert other.decrypt_value(token) == token  # a different key cannot decrypt it


def test_set_key_material_accepts_raw_bytes_and_rejects_garbage(ake, monkeypatch):
    _forbid_key_file(ake, monkeypatch)
    raw = os.urandom(32)
    ake.set_key_material(raw)
    token = ake.APIKeyEncryption().encrypt_value("secret")
    ake.set_key_material(base64.urlsafe_b64encode(raw).decode())  # str form of the same key
    assert ake.APIKeyEncryption().decrypt_value(token) == "secret"
    for bad in (b"short", "not a fernet key", 12345, base64.urlsafe_b64encode(os.urandom(16))):
        with pytest.raises(ValueError):
            ake.set_key_material(bad)


def test_set_key_material_rebuilds_shared_handler(ake, monkeypatch):
    _forbid_key_file(ake, monkeypatch)
    ake.set_key_material(_fernet_key())
    first = ake.get_handler()
    encrypted = ake.encrypt_config({"api_key": "sk-1", "multi_api_keys": [{"api_key": "sk-2"}]})
    assert encrypted["api_key"].startswith("ENC:")
    assert ake.decrypt_config(encrypted) == {"api_key": "sk-1", "multi_api_keys": [{"api_key": "sk-2"}]}
    ake.set_key_material(_fernet_key())
    assert ake.get_handler() is not first
    assert ake.decrypt_config(encrypted)["api_key"] == encrypted["api_key"]
    ake.set_key_material(None)
    assert ake._key_material is None and ake._handler is None


def test_api_key_env_fallback_for_host_tests(ake, monkeypatch):
    _forbid_key_file(ake, monkeypatch)
    env_key = _fernet_key()
    monkeypatch.setenv("GLOSSARION_API_KEY_FERNET", env_key.decode())
    token = ake.APIKeyEncryption().encrypt_value("from-env")
    assert ake.APIKeyEncryption().key_file is None
    ake.set_key_material(_fernet_key())  # the setter wins over the env fallback
    assert ake.APIKeyEncryption().decrypt_value(token) == token
    ake.set_key_material(None)
    assert ake.APIKeyEncryption().decrypt_value(token) == "from-env"


def test_api_key_invalid_env_falls_back_to_desktop_path(ake, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(ake.APIKeyEncryption, "_get_or_create_cipher", lambda self: sentinel)
    monkeypatch.setenv("GLOSSARION_API_KEY_FERNET", "definitely-not-a-key")
    handler = ake.APIKeyEncryption()
    assert handler.cipher is sentinel and handler.key_file is not None


# ---------------------------------------------------------------------------
# token_encryption.set_symmetric_key and the iOS/Android key path
# ---------------------------------------------------------------------------

class _PlatformSys:
    """Stands in for token_encryption's ``sys`` with a different sys.platform."""

    def __init__(self, platform):
        self.platform = platform

    def __getattr__(self, name):
        return getattr(sys, name)


@pytest.fixture
def te(monkeypatch, tmp_path):
    module = importlib.import_module("token_encryption")
    monkeypatch.setattr(module, "_symmetric_key_override", None)
    key_dir = tmp_path / "home" / ".glossarion"
    monkeypatch.setattr(module, "_KEY_DIR", str(key_dir))
    monkeypatch.setattr(module, "_KEY_FILE", str(key_dir / ".token_key"))
    shell_calls = []

    def no_shell(*args, **kwargs):
        shell_calls.append(args)
        raise AssertionError(f"token_encryption shelled out: {args!r}")

    monkeypatch.setattr(module, "subprocess", types.SimpleNamespace(run=no_shell))
    monkeypatch.setattr(module, "_test_shell_calls", shell_calls, raising=False)
    monkeypatch.setattr(module, "_test_key_file", key_dir / ".token_key", raising=False)
    return module


def _use_platform(te, monkeypatch, platform):
    monkeypatch.setattr(te, "sys", _PlatformSys(platform))


def test_token_key_desktop_linux_default_uses_key_file(te, monkeypatch):
    _use_platform(te, monkeypatch, "linux")
    key = te._get_symmetric_key()
    assert len(key) == 32 and te._test_key_file.is_file()
    assert te._get_symmetric_key() == key
    assert te._test_shell_calls == []


def test_token_key_desktop_macos_default_uses_keychain(te, monkeypatch):
    _use_platform(te, monkeypatch, "darwin")
    stored = []
    monkeypatch.setattr(te, "_keychain_load_key", lambda: stored[0] if stored else None)
    monkeypatch.setattr(te, "_keychain_store_key", lambda key: stored.append(key))
    deleted = []
    monkeypatch.setattr(te, "_keychain_delete_key", lambda: deleted.append(True))
    key = te._get_symmetric_key()
    assert stored == [key] and te._get_symmetric_key() == key
    assert not te._test_key_file.exists()
    te.clear_encryption_keys()
    assert deleted == [True]


@pytest.mark.parametrize("platform", ["ios", "android"])
def test_token_key_on_mobile_never_shells_out(te, monkeypatch, platform):
    _use_platform(te, monkeypatch, platform)
    key = te._get_symmetric_key()
    assert len(key) == 32 and te._test_key_file.is_file()
    assert te._get_symmetric_key() == key
    te.clear_encryption_keys()
    assert not te._test_key_file.exists()
    assert te._test_shell_calls == []


def test_token_key_macos_under_glossarion_mobile_uses_file(te, monkeypatch):
    _use_platform(te, monkeypatch, "darwin")
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    assert len(te._get_symmetric_key()) == 32 and te._test_key_file.is_file()
    te.clear_encryption_keys()
    assert te._test_shell_calls == []


@pytest.mark.parametrize("platform", ["ios", "android", "darwin", "linux"])
def test_set_symmetric_key_wins_over_keychain_and_file(te, monkeypatch, platform):
    _use_platform(te, monkeypatch, platform)
    monkeypatch.setattr(te, "_keychain_load_key", lambda: (_ for _ in ()).throw(AssertionError("keychain")))
    key = os.urandom(32)
    te.set_symmetric_key(key)
    assert te._get_symmetric_key() == key
    assert not te._test_key_file.exists()
    te.set_symmetric_key(base64.b64encode(key).decode())
    assert te._get_symmetric_key() == key
    te.set_symmetric_key(base64.urlsafe_b64encode(key))
    assert te._get_symmetric_key() == key
    for bad in (b"short", "a" * 31, 7, base64.b64encode(os.urandom(16))):
        with pytest.raises(ValueError):
            te.set_symmetric_key(bad)
    te.set_symmetric_key(None)
    assert te._symmetric_key_override is None
    assert te._test_shell_calls == []


def test_token_round_trip_on_android_with_setter_key(te, monkeypatch, tmp_path):
    _use_platform(te, monkeypatch, "android")
    te.set_symmetric_key(os.urandom(32))
    path = tmp_path / "authgpt_tokens.json"
    te.save_encrypted_tokens({"access_token": "a", "refresh_token": "r"}, str(path))
    assert path.read_bytes().startswith(b"GLSE1:")
    assert te.load_encrypted_tokens(str(path)) == {"access_token": "a", "refresh_token": "r"}
    te.set_symmetric_key(os.urandom(32))
    with pytest.raises(Exception):
        te.decrypt_tokens(path.read_bytes())
    assert not te._test_key_file.exists()


def test_token_env_fallback_for_host_tests(te, monkeypatch):
    _use_platform(te, monkeypatch, "ios")
    env_key = os.urandom(32)
    monkeypatch.setenv("GLOSSARION_TOKEN_KEY_B64", base64.b64encode(env_key).decode())
    assert te._get_symmetric_key() == env_key
    setter_key = os.urandom(32)
    te.set_symmetric_key(setter_key)
    assert te._get_symmetric_key() == setter_key
    te.set_symmetric_key(None)
    monkeypatch.setenv("GLOSSARION_TOKEN_KEY_B64", "a" * 32)  # 24 bytes, not a key: ignored
    key = te._get_symmetric_key()
    assert key != env_key and te._test_key_file.is_file()


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI is Windows-only")
def test_windows_keeps_dpapi_even_with_setter_key(te, monkeypatch):
    te.set_symmetric_key(os.urandom(32))
    monkeypatch.setattr(te, "_get_symmetric_key", lambda: (_ for _ in ()).throw(AssertionError("not DPAPI")))
    data = te.encrypt_tokens({"k": "v"})
    assert te.decrypt_tokens(data) == {"k": "v"}


# ---------------------------------------------------------------------------
# authgpt_auth: begin_oauth / complete_from_redirect
# ---------------------------------------------------------------------------

LEGACY_SUCCESS_HTML = (
    "<html><body style='font-family:sans-serif;text-align:center;padding-top:60px'>"
    "<h1>&#10004; Authenticated!</h1>"
    "<p>You can close this tab and return to Glossarion.</p>"
    "</body></html>"
)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_is_free(port):
    """True when nothing listens on 127.0.0.1:*port*, i.e. a new callback server could bind it.

    Binds the way HTTPServer does (``allow_reuse_address``): on Linux/macOS a connection
    the server accepted and closed leaves a TIME_WAIT entry that fails a plain bind for
    ~60 s, while SO_REUSEADDR still fails against a socket that is listening. Windows
    keeps the plain bind: SO_REUSEADDR there would bind over a live listener.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if os.name != "nt":
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def _http_get(host, port, path):
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read().decode("utf-8")
    finally:
        conn.close()


def test_port_probe_sees_a_live_listener_but_not_time_wait():
    """``_port_is_free`` fails while an HTTPServer listens, also after it served a request,
    and passes once it is closed although the served connection left TIME_WAIT behind."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _UntilClose(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")  # no Content-Length: the client reads to EOF, so the server closes first

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _UntilClose)
    port = server.server_address[1]
    try:
        assert not _port_is_free(port)
        handler = threading.Thread(target=server.handle_request, daemon=True)
        handler.start()
        status, _, body = _http_get("127.0.0.1", port, "/")
        assert status == 200 and body == "ok"
        handler.join(10)
        assert not handler.is_alive()
        assert not _port_is_free(port)
    finally:
        server.server_close()
    assert _port_is_free(port)


class _FakeTokenEndpoint:
    """Stands in for requests.post(OPENAI_TOKEN_URL, ...)."""

    def __init__(self, email="mobile@example.test"):
        self.calls = []
        claims = {
            "email": email,
            "https://api.openai.com/auth": {"chatgpt_plan_type": "plus", "chatgpt_account_id": "acct-1"},
        }
        payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
        self.id_token = f"e30.{payload}.sig"

    def __call__(self, url, data=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "data": dict(data or {}), "timeout": timeout})
        body = {
            "access_token": f"access-{len(self.calls)}",
            "refresh_token": f"refresh-{len(self.calls)}",
            "expires_in": 3600,
            "id_token": self.id_token,
        }

        class _Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return dict(body)

        return _Response()


@pytest.fixture
def authgpt(monkeypatch):
    module = importlib.import_module("authgpt_auth")
    monkeypatch.setattr(module, "CALLBACK_PORT", _free_port())
    endpoint = _FakeTokenEndpoint()
    monkeypatch.setattr(module.requests, "post", endpoint)
    opened = []
    monkeypatch.setattr(module.webbrowser, "open", lambda url, *a, **k: opened.append(url) or True)
    monkeypatch.setattr(module, "_test_endpoint", endpoint, raising=False)
    monkeypatch.setattr(module, "_test_opened", opened, raising=False)
    return module


def _expected_exchange(module, code, session):
    return {
        "url": module.OPENAI_TOKEN_URL,
        "data": {
            "grant_type": "authorization_code",
            "client_id": module.OPENAI_CLIENT_ID,
            "code": code,
            "redirect_uri": session.redirect_uri,
            "code_verifier": session.code_verifier,
        },
        "timeout": 30,
    }


def test_begin_oauth_loopback_callback_then_complete_saves_tokens(authgpt, tmp_path):
    port = authgpt.CALLBACK_PORT
    token_file = tmp_path / "authgpt_tokens_3.json"
    store = authgpt.AuthGPTTokenStore(token_file=str(token_file), account_id=3)
    session = authgpt.begin_oauth(store, timeout=30)
    try:
        assert authgpt._test_opened == []  # the caller opens the browser itself
        query = parse_qs(urlparse(session.auth_url).query)
        assert query["client_id"] == [authgpt.OPENAI_CLIENT_ID]
        assert query["redirect_uri"] == [f"http://localhost:{port}/auth/callback"] == [session.redirect_uri]
        assert query["state"] == [session.state]
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(session.code_verifier.encode("ascii")).digest()
        ).rstrip(b"=").decode("ascii")
        assert query["code_challenge"] == [challenge]
        assert query["code_challenge_method"] == ["S256"]
        assert session.account_id == 3 and session.store is store
        assert session.server_running and not session.callback_received

        pending_file = Path(store.pending_oauth_file)
        assert pending_file.name == "authgpt_tokens_3.oauth_pending"
        raw = pending_file.read_bytes()
        assert raw.startswith(b"GLSE1:")
        assert session.code_verifier.encode() not in raw and session.state.encode() not in raw
        pending = store.load_pending_oauth()
        assert pending["code_verifier"] == session.code_verifier
        assert pending["state"] == session.state
        assert pending["redirect_uri"] == session.redirect_uri
        assert pending["auth_url"] == session.auth_url and pending["account_id"] == 3

        status, headers, _ = _http_get("127.0.0.1", port, f"/auth/callback?code=loopback-code&state={session.state}")
        assert status == 302 and headers["Location"] == f"http://localhost:{port}/success"
        assert session.wait_for_callback(5)
        tokens = authgpt.complete_from_redirect(session)
        assert session.server_running  # still serving the success page
        status, _, body = _http_get("127.0.0.1", port, "/success")
        assert status == 200 and body == LEGACY_SUCCESS_HTML
        assert session.wait(5)
        assert not session.server_running
    finally:
        session.close()

    assert authgpt._test_endpoint.calls == [_expected_exchange(authgpt, "loopback-code", session)]
    assert tokens["access_token"] == "access-1"
    assert store.load_tokens()["access_token"] == "access-1"
    reloaded = authgpt.AuthGPTTokenStore(token_file=str(token_file), account_id=3)
    assert reloaded.load_tokens()["refresh_token"] == "refresh-1"
    assert not pending_file.exists()
    assert _port_is_free(port)


def test_begin_oauth_also_listens_on_ipv6_loopback(authgpt, tmp_path):
    if not socket.has_ipv6:
        pytest.skip("no IPv6")
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", authgpt.CALLBACK_PORT))
    except OSError:
        pytest.skip("::1 unavailable")
    store = authgpt.AuthGPTTokenStore(token_file=str(tmp_path / "t.json"))
    session = authgpt.begin_oauth(store, timeout=30)
    try:
        assert len(session._servers) == 2
        status, _, _ = _http_get("::1", authgpt.CALLBACK_PORT, f"/auth/callback?code=v6-code&state={session.state}")
        assert status == 302
        assert session.wait_for_callback(5)
        assert authgpt.complete_from_redirect(session)["access_token"] == "access-1"
    finally:
        session.close()
    assert authgpt._test_endpoint.calls[0]["data"]["code"] == "v6-code"


def test_paste_full_redirect_url_after_loopback_died(authgpt, tmp_path):
    token_file = tmp_path / "authgpt_tokens.json"
    store = authgpt.AuthGPTTokenStore(token_file=str(token_file))
    session = authgpt.begin_oauth(store, timeout=30)
    session.close()  # loopback gone (app killed while the browser was open)
    assert not session.server_running
    # A fresh process: only the encrypted pending file survives.
    fresh_store = authgpt.AuthGPTTokenStore(token_file=str(token_file))
    redirect = (
        f"http://localhost:{authgpt.CALLBACK_PORT}/auth/callback"
        f"?code=pasted-code&scope=openid+profile&state={session.state}"
    )
    tokens = authgpt.complete_from_redirect(fresh_store, f"  {redirect}\n")
    assert authgpt._test_endpoint.calls == [_expected_exchange(authgpt, "pasted-code", session)]
    assert tokens["access_token"] == "access-1"
    assert fresh_store.load_tokens()["access_token"] == "access-1"
    assert fresh_store.load_pending_oauth() is None
    assert not Path(fresh_store.pending_oauth_file).exists()


def test_paste_with_live_session_or_saved_dict(authgpt, tmp_path):
    store = authgpt.AuthGPTTokenStore(token_file=str(tmp_path / "a.json"))
    session = authgpt.begin_oauth(store, timeout=30)
    authgpt.complete_from_redirect(session, f"code-from-paste#{session.state}")
    assert not session.server_running  # a pasted redirect frees the port
    assert _port_is_free(authgpt.CALLBACK_PORT)
    assert authgpt._test_endpoint.calls[-1]["data"]["code"] == "code-from-paste"

    other = authgpt.AuthGPTTokenStore(token_file=str(tmp_path / "b.json"))
    session2 = authgpt.begin_oauth(other, timeout=30)
    session2.close()
    saved = other.load_pending_oauth()
    authgpt.complete_from_redirect(saved, f"?code=dict-code&state={session2.state}", store=other)
    assert other.load_tokens()["access_token"] == "access-2"
    assert authgpt._test_endpoint.calls[-1]["data"]["code_verifier"] == session2.code_verifier


@pytest.mark.parametrize(
    "pasted, expected",
    [
        ("http://localhost:1455/auth/callback?code=c1&state=s1", ("c1", "s1", None)),
        ("localhost:1455/auth/callback?state=s2&code=c2", ("c2", "s2", None)),
        ("http://localhost:1455/auth/callback#code=c3&state=s3", ("c3", "s3", None)),
        ("?code=c4&state=s4", ("c4", "s4", None)),
        ("code=c5&state=s5", ("c5", "s5", None)),
        ("c6#s6", ("c6", "s6", None)),
        ("  c7  ", ("c7", None, None)),
        ("http://localhost:1455/auth/callback?error=access_denied&state=s8", (None, "s8", "access_denied")),
        ("", (None, None, None)),
    ],
)
def test_parse_pasted_redirect(authgpt, pasted, expected):
    assert authgpt._parse_oauth_redirect(pasted) == expected


def test_state_mismatch_is_rejected(authgpt, tmp_path):
    port = authgpt.CALLBACK_PORT
    store = authgpt.AuthGPTTokenStore(token_file=str(tmp_path / "authgpt_tokens.json"))
    session = authgpt.begin_oauth(store, timeout=30)
    try:
        _http_get("127.0.0.1", port, "/auth/callback?code=evil-code&state=forged")
        assert session.wait_for_callback(5)
        with pytest.raises(RuntimeError, match="state mismatch"):
            authgpt.complete_from_redirect(session)
        with pytest.raises(RuntimeError, match="state mismatch"):
            authgpt.complete_from_redirect(store, f"http://localhost:{port}/auth/callback?code=evil&state=forged")
        with pytest.raises(RuntimeError, match="OAuth error: access_denied"):
            authgpt.complete_from_redirect(store, f"http://localhost:{port}/auth/callback?error=access_denied")
        with pytest.raises(RuntimeError, match="authorization code"):
            authgpt.complete_from_redirect(store, f"http://localhost:{port}/auth/callback?state={session.state}")
    finally:
        session.close()
    assert authgpt._test_endpoint.calls == []
    assert store.load_tokens() is None
    # The genuine redirect can still be pasted afterwards.
    assert store.load_pending_oauth()["state"] == session.state


def test_nothing_pending_or_expired(authgpt, tmp_path):
    store = authgpt.AuthGPTTokenStore(token_file=str(tmp_path / "t.json"))
    with pytest.raises(RuntimeError, match="No pending ChatGPT sign-in"):
        authgpt.complete_from_redirect(store, "?code=x&state=y")
    with pytest.raises(RuntimeError, match="No pending ChatGPT sign-in"):
        authgpt.complete_from_redirect(None, "?code=x&state=y")
    store.save_pending_oauth(
        {"code_verifier": "v", "state": "s", "redirect_uri": "http://localhost:1/auth/callback",
         "created_at": time.time() - authgpt.PENDING_OAUTH_MAX_AGE_SECONDS - 5}
    )
    assert Path(store.pending_oauth_file).exists()
    assert store.load_pending_oauth() is None
    assert not Path(store.pending_oauth_file).exists()
    assert authgpt._test_endpoint.calls == []


def test_begin_oauth_without_store_or_persist_writes_nothing(authgpt, tmp_path):
    session = authgpt.begin_oauth(timeout=30)
    try:
        assert session.store is None
    finally:
        session.close()
    store = authgpt.AuthGPTTokenStore(token_file=str(tmp_path / "t.json"))
    session = authgpt.begin_oauth(store, persist=False, timeout=30)
    session.close()
    assert list(tmp_path.iterdir()) == []


def test_begin_oauth_rejects_busy_port(authgpt):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", authgpt.CALLBACK_PORT))
        busy.listen(1)
        with pytest.raises(RuntimeError, match="already in use"):
            authgpt.begin_oauth(timeout=30)


def test_run_oauth_flow_keeps_desktop_behaviour(authgpt, monkeypatch):
    port = authgpt.CALLBACK_PORT
    begin_calls = []
    real_begin = authgpt.begin_oauth

    def spy_begin(*args, **kwargs):
        begin_calls.append((args, kwargs))
        session = real_begin(*args, **kwargs)
        begin_calls.append(session)
        return session

    monkeypatch.setattr(authgpt, "begin_oauth", spy_begin)
    pages = []

    def browser(url, *args, **kwargs):
        state = parse_qs(urlparse(url).query)["state"][0]

        def visit():
            status, headers, _ = _http_get("127.0.0.1", port, f"/auth/callback?code=desktop-code&state={state}")
            pages.append((status, headers.get("Location")))
            pages.append(_http_get("127.0.0.1", port, "/success"))

        threading.Thread(target=visit, daemon=True).start()
        return True

    monkeypatch.setattr(authgpt.webbrowser, "open", browser)
    tokens = authgpt.run_oauth_flow(timeout=20)

    assert begin_calls[0] == ((), {"timeout": 20, "persist": False, "dual_stack": False, "auto_close": False})
    session = begin_calls[1]
    assert len(session._servers) == 1 and session.store is None
    assert pages[0] == (302, f"http://localhost:{port}/success")
    assert pages[1][0] == 200 and pages[1][2] == LEGACY_SUCCESS_HTML
    assert tokens["access_token"] == "access-1" and tokens["refresh_token"] == "refresh-1"
    assert authgpt._test_endpoint.calls == [_expected_exchange(authgpt, "desktop-code", session)]
    assert not session.server_running and _port_is_free(port)


def test_run_oauth_flow_timeout_and_error_messages(authgpt, monkeypatch):
    port = authgpt.CALLBACK_PORT
    with pytest.raises(RuntimeError, match="timed out"):
        authgpt.run_oauth_flow(timeout=0.3)
    assert _port_is_free(port)

    def denied(url, *args, **kwargs):
        threading.Thread(
            target=lambda: (
                _http_get("127.0.0.1", port, "/auth/callback?error=access_denied"),
                _http_get("127.0.0.1", port, "/success"),
            ),
            daemon=True,
        ).start()
        return True

    monkeypatch.setattr(authgpt.webbrowser, "open", denied)
    with pytest.raises(RuntimeError, match="OAuth error: access_denied"):
        authgpt.run_oauth_flow(timeout=20)
    assert authgpt._test_endpoint.calls == []


def test_success_page_desktop_default_is_unchanged(authgpt):
    assert authgpt._oauth_success_html() == LEGACY_SUCCESS_HTML


def test_success_page_returns_to_the_app_when_return_url_set(authgpt, monkeypatch, tmp_path):
    monkeypatch.setenv("GLOSSARION_OAUTH_RETURN_URL", "glossarion://app/oauth/return")
    page = authgpt._oauth_success_html()
    assert "href='glossarion://app/oauth/return?p=authgpt'" in page
    assert "Return to Glossarion" in page and "location.href" in page
    monkeypatch.setenv("GLOSSARION_OAUTH_RETURN_URL", "glossarion://app/oauth/return?x=1")
    assert "href='glossarion://app/oauth/return?x=1&amp;p=authgpt'" in authgpt._oauth_success_html()

    # The live loopback page carries it too. With the return URL set the callback itself is
    # answered with it (no /success hop the app would also have to serve), and the callback
    # ends the sign-in like the /success page does (run_oauth_flow's session.wait()).
    monkeypatch.setenv("GLOSSARION_OAUTH_RETURN_URL", "glossarion://app/oauth/return")
    store = authgpt.AuthGPTTokenStore(token_file=str(tmp_path / "t.json"))
    session = authgpt.begin_oauth(store, timeout=30)
    try:
        status, _, body = _http_get("127.0.0.1", authgpt.CALLBACK_PORT, f"/auth/callback?code=c&state={session.state}")
        assert status == 200 and "glossarion://app/oauth/return?p=authgpt" in body
        assert session.wait_for_callback(5) and session.wait(5) and not session.server_running
    finally:
        session.close()


def test_mobile_port_probe_tolerates_time_wait_only_on_mobile(authgpt, monkeypatch):
    seen = []
    real_socket = socket.socket

    class _RecordingSocket(real_socket):
        def setsockopt(self, *args):
            seen.append(args)
            return super().setsockopt(*args)

    monkeypatch.setattr(socket, "socket", _RecordingSocket)
    assert authgpt._find_available_port() == authgpt.CALLBACK_PORT
    assert seen == []  # desktop: unchanged plain bind probe
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    assert authgpt._find_available_port() == authgpt.CALLBACK_PORT
    if os.name == "nt":
        assert seen == []  # SO_REUSEADDR on Windows would allow port hijacking
    else:
        assert seen == [(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)]
