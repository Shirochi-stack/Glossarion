"""Device/host self-test.

Run on device by the deep link ``glossarion://app/__selftest__?suite=smoke`` (CI
emulator smoke) or the "Run self-test" button; on the host by
``tests_host/test_bootstrap.py`` or ``python -m glossarion_mobile.diagnostics.selftest``.

``run_selftest()`` is blocking (call it from a worker thread, never from the
Flet loop), returns a JSON-serialisable dict, writes ``<logs>/selftest-last.json``
and prints exactly one marker line::

    GLOSSARION_SELFTEST PASS {"suite":"smoke","passed":11,...}
    GLOSSARION_SELFTEST FAIL {"suite":"smoke","failed":1,"failed_checks":[...]}

In strict mode (default on Android/iOS) a missing package is a failure; on the
host (non-strict) checks whose packages are missing are reported as skipped.
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

from glossarion_mobile import runtime_bootstrap as rb
from glossarion_mobile.diagnostics import fixtures

__all__ = ["SUITES", "CheckSkipped", "run_selftest", "summary_line", "main"]

_RUN_LOCK = threading.Lock()


class CheckSkipped(Exception):
    """Raised by a check that cannot run here (non-strict mode only)."""


class Context:
    def __init__(self, strict: bool) -> None:
        self.strict = strict
        self.state = rb.get_state()
        self.paths = self.state.paths if self.state is not None else None

    def need(self, module: str):
        """Import ``module``; missing -> failure (strict) or skip (host)."""
        try:
            return importlib.import_module(module)
        except ImportError as exc:
            if self.strict:
                raise
            raise CheckSkipped(f"{module} not installed ({exc})") from exc

    def require_bootstrap(self):
        if self.paths is None:
            raise AssertionError("runtime_bootstrap.bootstrap() has not run in this process")
        return self.paths


# --------------------------------------------------------------------------
# Checks (each returns a JSON-serialisable detail dict or raises)
# --------------------------------------------------------------------------


def check_env_contract(ctx: Context) -> dict[str, Any]:
    paths = ctx.require_bootstrap()
    expected = paths.env_contract()
    mismatched = {
        key: {"expected": value, "actual": os.environ.get(key)}
        for key, value in expected.items()
        if os.environ.get(key) != value
    }
    ca_file = os.environ.get("SSL_CERT_FILE")
    problems: list[str] = []
    if not ca_file or not os.path.isfile(ca_file):
        problems.append(f"SSL_CERT_FILE missing or not a file: {ca_file!r}")
    if os.environ.get("REQUESTS_CA_BUNDLE") != ca_file:
        problems.append("REQUESTS_CA_BUNDLE != SSL_CERT_FILE")
    if os.path.normcase(os.getcwd()) != os.path.normcase(str(paths.data)):
        problems.append(f"cwd is {os.getcwd()!r}, expected the data dir")
    stack = rb.current_thread_stack_size()
    if stack < rb.THREAD_STACK_SIZE:
        problems.append(f"thread stack size {stack} < 16 MiB")
    if paths.backend_dir is None:
        problems.append(f"backend dir unresolved ({paths.backend_source})")
    elif str(paths.backend_dir) not in sys.path:
        problems.append("backend dir not on sys.path")
    if mismatched or problems:
        raise AssertionError(json.dumps({"mismatched": mismatched, "problems": problems}, ensure_ascii=True))
    return {"checked": len(expected) + 2, "backend": paths.backend_source, "platform": paths.platform}


def check_writable_dirs(ctx: Context) -> dict[str, Any]:
    paths = ctx.require_bootstrap()
    failures: dict[str, str] = {}
    for name, directory in paths.writable_dirs().items():
        probe = Path(directory) / f".glossarion-probe-{os.getpid()}-{threading.get_ident()}"
        try:
            probe.write_text("ok", encoding="utf-8")
            if probe.read_text(encoding="utf-8") != "ok":
                failures[name] = "read-back mismatch"
        except OSError as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                probe.unlink()
            except OSError:
                pass
    if failures:
        raise AssertionError(json.dumps(failures, ensure_ascii=True))
    return {"dirs": sorted(paths.writable_dirs())}


def check_backend_imports(ctx: Context) -> dict[str, Any]:
    paths = ctx.require_bootstrap()
    if paths.backend_dir is None:
        raise AssertionError(f"backend dir unresolved ({paths.backend_source})")
    root = os.path.normcase(str(paths.backend_dir))
    timings: dict[str, float] = {}
    missing_third_party: dict[str, str] = {}
    errors: dict[str, str] = {}
    outside: list[str] = []
    for name in rb.WARM_IMPORT_MODULES:
        t0 = time.monotonic()
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] not in rb.WARM_IMPORT_MODULES and exc.name != name:
                missing_third_party[name] = str(exc)
            else:
                errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        except BaseException as exc:  # a backend import calling sys.exit() must not end the test
            if isinstance(exc, KeyboardInterrupt):
                raise
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        timings[name] = round(time.monotonic() - t0, 3)
        filename = os.path.normcase(os.path.abspath(getattr(module, "__file__", "") or ""))
        if not filename.startswith(root + os.sep):
            outside.append(name)
    if errors or outside or (missing_third_party and ctx.strict):
        raise AssertionError(
            json.dumps(
                {"errors": errors, "missing": missing_third_party, "not_from_backend": outside},
                ensure_ascii=True,
            )
        )
    if missing_third_party:
        raise CheckSkipped(f"third-party packages missing on host: {missing_third_party}")
    return {"secs": timings, "backend": paths.backend_source}


def check_tiktoken_offline(ctx: Context) -> dict[str, Any]:
    tiktoken = ctx.need("tiktoken")
    tk_load = importlib.import_module("tiktoken.load")
    cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
    if not cache_dir:
        raise AssertionError("TIKTOKEN_CACHE_DIR is not set")
    missing = [
        name
        for name, (url, _digest) in rb.TIKTOKEN_ENCODINGS.items()
        if not (Path(cache_dir) / rb.tiktoken_cache_key(url)).is_file()
    ]
    if missing:
        message = f"tiktoken cache not seeded for {missing} in {cache_dir} (run tools/prepare_assets.py)"
        if ctx.strict:
            raise AssertionError(message)
        raise CheckSkipped(message)

    def no_network(blobpath: str) -> bytes:
        raise AssertionError(f"tiktoken tried to download {blobpath} (cache miss)")

    original = getattr(tk_load, "read_file", None)
    detail: dict[str, Any] = {"cache_dir": cache_dir}
    try:
        if original is not None:
            tk_load.read_file = no_network
        for name in rb.TIKTOKEN_ENCODINGS:
            encoding = tiktoken.get_encoding(name)
            tokens = encoding.encode(fixtures.KOREAN_SAMPLE)
            if encoding.decode(tokens) != fixtures.KOREAN_SAMPLE:
                raise AssertionError(f"{name}: decode(encode(x)) != x")
            detail[name] = len(tokens)
    finally:
        if original is not None:
            tk_load.read_file = original
    detail["version"] = getattr(tiktoken, "__version__", None)
    return detail


def check_epub_lxml(ctx: Context) -> dict[str, Any]:
    ebooklib = ctx.need("ebooklib")
    epub = importlib.import_module("ebooklib.epub")
    lxml_html = ctx.need("lxml.html")
    paths = ctx.paths
    asset = fixtures.find_selftest_epub(paths.assets_dir if paths is not None else None)
    source = "asset"
    if asset is None:
        if ctx.strict:
            raise AssertionError("selftest EPUB missing from assets/selftest (tools/prepare_assets.py)")
        base = paths.temp if paths is not None else None
        asset = fixtures.build_tiny_epub(fixtures.scratch_dir(base) / "selftest-tiny.epub")
        source = "generated"
    expected: dict[str, Any] = {}
    if source == "asset":
        manifest = asset.parent / "MANIFEST.toml"
        if manifest.is_file():
            data = rb._load_toml(manifest) or {}
            expected = data.get("epub", {}) if isinstance(data, dict) else {}
        digest = expected.get("sha256")
        if digest and rb._sha256_file(asset) != digest:
            raise AssertionError(f"{asset.name}: sha256 differs from assets/selftest/MANIFEST.toml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        book = epub.read_epub(str(asset))
    documents = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    paragraphs = 0
    hangul = False
    for item in documents:
        content = item.get_content()
        if not content:
            continue
        tree = lxml_html.fromstring(content)
        paragraphs += len(tree.xpath("//p"))
        text = tree.text_content()
        hangul = hangul or any("가" <= ch <= "힣" for ch in text)
    if not documents or paragraphs == 0:
        raise AssertionError(f"{asset.name}: {len(documents)} documents, {paragraphs} paragraphs")
    if not hangul:
        raise AssertionError(f"{asset.name}: no Hangul text decoded")
    chapters = expected.get("chapters")
    if isinstance(chapters, int) and len(documents) < chapters:
        raise AssertionError(f"{asset.name}: {len(documents)} documents < {chapters} chapters in MANIFEST.toml")
    return {"epub": asset.name, "source": source, "documents": len(documents), "paragraphs": paragraphs}


def check_fernet(ctx: Context) -> dict[str, Any]:
    fernet_mod = ctx.need("cryptography.fernet")
    key = fernet_mod.Fernet.generate_key()
    cipher = fernet_mod.Fernet(key)
    secret = "sk-selftest-🔑-값".encode("utf-8")
    token = cipher.encrypt(secret)
    if cipher.decrypt(token) != secret:
        raise AssertionError("Fernet round-trip mismatch")
    cryptography = importlib.import_module("cryptography")
    return {"version": getattr(cryptography, "__version__", None), "token_len": len(token)}


_KEY_FILE_NAMES = (".glossarion_key", "glossarion_key.txt")  # api_key_encryption's desktop key files


def check_encryption_keys(ctx: Context) -> dict[str, Any]:
    """The backend encrypts with the SecureStorage keys (``services.secure_keys``), not a key file."""
    from glossarion_mobile.services import secure_keys

    paths = ctx.require_bootstrap()
    status = secure_keys.current_status()
    if status is None:
        message = "encryption keys were never installed (SecureStorage -> set_key_material/set_symmetric_key)"
        if ctx.strict:
            raise AssertionError(message)
        raise CheckSkipped(message)
    if not status.installed:
        raise AssertionError(f"key setters failed: {status.errors}")
    if status.degraded and ctx.strict:
        raise AssertionError(f"SecureStorage unusable, keys from {status.source}: {status.notes}")
    ctx.need("cryptography.fernet")
    aek = importlib.import_module("api_key_encryption")
    handler = aek.get_handler()
    key_file = getattr(handler, "key_file", "<missing attribute>")
    if key_file is not None:
        raise AssertionError(f"api_key_encryption uses the key file {key_file!r}, not the SecureStorage key")
    secret = "sk-selftest-🔑-값"
    token = handler.encrypt_value(secret)
    if not isinstance(token, str) or not token.startswith("ENC:"):
        raise AssertionError(f"encrypt_value did not encrypt (got {str(token)[:16]!r})")
    if handler.decrypt_value(token) != secret:
        raise AssertionError("decrypt_value(encrypt_value(x)) != x")
    try:
        injected = secure_keys.decrypt_with_installed_api_key(base64.b64decode(token[4:])).decode("utf-8")
    except Exception as exc:
        raise AssertionError(f"the API-key handler does not use the installed key: {type(exc).__name__}: {exc}")
    if injected != secret:
        raise AssertionError("the installed key decrypted a different value")
    tok = importlib.import_module("token_encryption")
    active_token_key = getattr(tok, "_get_symmetric_key")()
    if secure_keys.fingerprint(active_token_key) != status.token_key_fingerprint:
        raise AssertionError("token_encryption does not use the installed key")
    # The desktop branch writes its key file next to the backend (or under HOME).
    # On the host the backend dir may be the repo src/, where a desktop key file is normal.
    must_be_clean = [paths.home, paths.data] + ([Path(aek.__file__).resolve().parent] if ctx.strict else [])
    stray = [str(d / name) for d in must_be_clean for name in _KEY_FILE_NAMES if (d / name).exists()]
    if stray:
        raise AssertionError(f"key file(s) exist although the SecureStorage keys are in use: {stray}")
    return {
        "source": status.source,
        "degraded": status.degraded,
        "api_key": status.api_key_fingerprint,
        "token_key": status.token_key_fingerprint,
    }


def check_openai_pydantic(ctx: Context) -> dict[str, Any]:
    openai = ctx.need("openai")
    pydantic = ctx.need("pydantic")
    jiter = ctx.need("jiter")
    parsed = jiter.from_json(b'{"a":[1,2,3],"b":"\xea\xb0\x80"}')
    if parsed != {"a": [1, 2, 3], "b": "가"}:
        raise AssertionError(f"jiter parsed {parsed!r}")

    class Probe(pydantic.BaseModel):
        a: list[int]
        b: str

    model = Probe.model_validate_json('{"a":[4],"b":"x"}')
    if model.a != [4]:
        raise AssertionError("pydantic validation mismatch")
    client = openai.OpenAI(api_key="selftest", base_url="http://127.0.0.1:9/v1", max_retries=0)
    try:
        base_url = str(client.base_url)
    finally:
        client.close()
    return {
        "openai": getattr(openai, "__version__", None),
        "pydantic": getattr(pydantic, "VERSION", None),
        "jiter": getattr(jiter, "__version__", None),
        "client_base_url": base_url,
    }


def check_pymupdf(ctx: Context) -> dict[str, Any]:
    fitz = ctx.need("fitz")
    data = fixtures.build_tiny_pdf_bytes()
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        page = doc[0]
        pixmap = page.get_pixmap(dpi=72)
        png = pixmap.tobytes("png")
        text = page.get_text()
        if "Glossarion self-test" not in text:
            raise AssertionError(f"text extraction returned {text[:80]!r}")
        return {
            "version": getattr(fitz, "VersionBind", None),
            "pages": doc.page_count,
            "png_bytes": len(png),
            "size": [pixmap.width, pixmap.height],
        }
    finally:
        doc.close()


def check_cv2(ctx: Context) -> dict[str, Any]:
    cv2 = ctx.need("cv2")
    numpy = ctx.need("numpy")
    image = numpy.zeros((16, 16, 3), dtype=numpy.uint8)
    image[4:12, 4:12] = (152, 143, 225)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ok, buffer = cv2.imencode(".png", image)
    if not ok or gray.shape != (16, 16):
        raise AssertionError("cv2 cvtColor/imencode failed")
    return {"version": getattr(cv2, "__version__", None), "png_bytes": int(len(buffer)), "numpy": numpy.__version__}


def check_onnxruntime(ctx: Context) -> dict[str, Any]:
    ort = ctx.need("onnxruntime")
    providers = list(ort.get_available_providers())
    if "CPUExecutionProvider" not in providers:
        raise AssertionError(f"no CPUExecutionProvider in {providers}")
    return {"version": getattr(ort, "__version__", None), "providers": providers}


def check_thread_stack(ctx: Context) -> dict[str, Any]:
    """Deep C recursion (json/repr of nested lists) on a fresh worker thread."""
    outcome: dict[str, Any] = {}

    def work() -> None:
        # Each level is a C-level recursion (list repr / json encoder), which is
        # what overflows small native thread stacks. Hitting the interpreter's
        # own C-recursion guard (RecursionError) is fine: the point is no SIGSEGV.
        reached = 0
        for depth in (1000, 2500):
            nested: list[Any] = []
            for _ in range(depth):
                nested = [nested]
            try:
                repr(nested)
                json.dumps(nested)
                reached = depth
            except RecursionError as exc:
                outcome["recursion_guard"] = f"depth {depth}: {exc}"
                break
        outcome["depth_ok"] = reached

    thread = threading.Thread(target=work, name="gl-selftest-stack")
    thread.start()
    thread.join(60)
    if thread.is_alive():
        raise AssertionError("stack probe thread did not finish")
    outcome["stack_size"] = rb.current_thread_stack_size()
    if outcome["stack_size"] < rb.THREAD_STACK_SIZE:
        raise AssertionError(f"thread stack size {outcome['stack_size']} < 16 MiB")
    return outcome


SUITES: dict[str, tuple[tuple[str, Callable[[Context], dict[str, Any]]], ...]] = {
    "smoke": (
        ("env_contract", check_env_contract),
        ("writable_dirs", check_writable_dirs),
        ("backend_imports", check_backend_imports),
        ("tiktoken_offline", check_tiktoken_offline),
        ("epub_lxml", check_epub_lxml),
        ("fernet", check_fernet),
        ("encryption_keys", check_encryption_keys),
        ("openai_pydantic_jiter", check_openai_pydantic),
        ("pymupdf", check_pymupdf),
        ("cv2", check_cv2),
        ("onnxruntime", check_onnxruntime),
        ("thread_stack", check_thread_stack),
    ),
}


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def summary_line(result: dict[str, Any]) -> str:
    """The marker line for ``result`` (kept under ~900 chars for os_log/logcat)."""
    status = "PASS" if result.get("ok") else "FAIL"
    summary: dict[str, Any] = {
        "suite": result.get("suite"),
        "passed": result.get("passed"),
        "failed": result.get("failed"),
        "skipped": result.get("skipped"),
        "secs": result.get("secs"),
        "platform": result.get("platform"),
        "strict": result.get("strict"),
    }
    failed = [c for c in result.get("checks", []) if c.get("status") == "fail"]
    if result.get("error"):
        summary["error"] = _short(result["error"], 160)
    if failed:
        summary["failed_checks"] = [{"name": c["name"], "error": _short(c.get("error", ""), 120)} for c in failed]
    line = json.dumps(summary, separators=(",", ":"), ensure_ascii=True)
    while len(line) > rb.MARKER_MAX_CHARS and summary.get("failed_checks"):
        summary["failed_checks"] = [{"name": c["name"]} for c in failed][: max(1, len(summary["failed_checks"]) - 1)]
        summary["truncated"] = True
        line = json.dumps(summary, separators=(",", ":"), ensure_ascii=True)
    return f"{rb.MARKER_SELFTEST} {status} {line}"


def run_selftest(
    suite: str = "smoke",
    *,
    strict: Optional[bool] = None,
    emit: bool = True,
    write_report: bool = True,
    only: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Run a suite (blocking). Never raises for check failures."""
    with _RUN_LOCK:
        state = rb.get_state()
        if strict is None:
            strict = bool(state is not None and state.paths.is_device)
        ctx = Context(strict=strict)
        started = time.time()
        t0 = time.monotonic()
        checks: list[dict[str, Any]] = []
        error = None
        definitions = SUITES.get(suite)
        if definitions is None:
            error = f"unknown suite {suite!r} (known: {sorted(SUITES)})"
            definitions = ()
        for name, func in definitions:
            if only is not None and name not in only:
                continue
            c0 = time.monotonic()
            entry: dict[str, Any] = {"name": name}
            try:
                entry["detail"] = func(ctx)
                entry["status"] = "pass"
            except CheckSkipped as exc:
                entry["status"] = "skip"
                entry["reason"] = str(exc)
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                entry["status"] = "fail"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["traceback"] = traceback.format_exc()[-2000:]
            entry["secs"] = round(time.monotonic() - c0, 3)
            checks.append(entry)

        passed = sum(1 for c in checks if c["status"] == "pass")
        failed = sum(1 for c in checks if c["status"] == "fail")
        skipped = sum(1 for c in checks if c["status"] == "skip")
        result: dict[str, Any] = {
            "suite": suite,
            "ok": error is None and failed == 0 and passed > 0,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "secs": round(time.monotonic() - t0, 2),
            "strict": strict,
            "platform": state.paths.platform if state is not None else rb.detect_platform(),
            "python": sys.version.split()[0],
            "started_at": started,
            "checks": checks,
        }
        if error:
            result["error"] = error
        if write_report and state is not None:
            try:
                report = state.paths.logs / "selftest-last.json"
                report.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
                result["report"] = str(report)
            except OSError:
                pass
        if emit:
            line = summary_line(result)
            rb.emit_marker(line.split(" ", 1)[0], line.split(" ", 1)[1])
        return result


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Glossarion mobile self-test")
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--strict", action="store_true", help="missing packages fail instead of skip")
    parser.add_argument("--json", action="store_true", help="print the full result JSON to stdout")
    args = parser.parse_args(argv)
    rb.bootstrap()
    result = run_selftest(args.suite, strict=True if args.strict else None)
    if args.json:
        print(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    return 0 if result["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
