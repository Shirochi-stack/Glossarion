"""Host tests for the U10 document-destination API of flet-glossarion-native.

* The Python service over a fake Flet channel: safe defaults, argument marshalling, timeouts,
  result normalisation, events.
* ``documents.py`` helpers (ids, modes, results, log redaction).
* ``documents_fake.FakeDocumentsNative``: the Android semantics (write-mode chain, "missing only
  when the root answers", create retries, grants) for every scenario the U10 critic raised.
* Static checks of the Dart / Kotlin / Swift sources: method names, error codes, event and ref
  keys, constants, manifest, iOS 13 API use, and brace / paren / bracket balance (no Flutter SDK
  here, so these stand in for a compile).

Run: python -m pytest -p no:cacheprovider -W ignore src/mobile/extensions/flet_glossarion_native/tests
"""

import ast
import asyncio
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ft = pytest.importorskip("flet")

from flet.utils.from_dict import from_dict  # noqa: E402

import flet_glossarion_native as gn  # noqa: E402
from flet_glossarion_native import documents as docs  # noqa: E402
from flet_glossarion_native.documents_fake import FakeCloudProvider, FakeDocumentsNative  # noqa: E402

FLUTTER_PKG = ROOT / "src" / "flutter" / "flet_glossarion_native"
DART_SERVICE = FLUTTER_PKG / "lib" / "src" / "native_service.dart"
KOTLIN_DIR = FLUTTER_PKG / "android" / "src" / "main" / "kotlin" / "com" / "glossarion" / "flet_glossarion_native"
KOTLIN_PLUGIN = KOTLIN_DIR / "GlossarionNativePlugin.kt"
KOTLIN_DOCS = KOTLIN_DIR / "DocumentDestinations.kt"
SWIFT_DIR = FLUTTER_PKG / "ios" / "flet_glossarion_native" / "Sources" / "flet_glossarion_native"
SWIFT_PLUGIN = SWIFT_DIR / "GlossarionNativePlugin.swift"
SWIFT_DOCS = SWIFT_DIR / "DocumentDestinations.swift"
MANIFEST = FLUTTER_PKG / "android" / "src" / "main" / "AndroidManifest.xml"
ANDROID_NS = "{http://schemas.android.com/apk/res/android}"

DOC_METHODS = {
    "pick_folder", "pick_save_location", "pick_document", "list_children", "create_file",
    "create_folder", "write_file", "rename_document", "stat", "delete", "query_root",
}
PLATFORM_METHODS = DOC_METHODS | {"release", "list_grants", "cancel_document_op"}
PYTHON_METHODS = PLATFORM_METHODS | {"take_document_results"}

FOLDER = {"platform": "android", "kind": "folder", "id": "d1",
          "uri": "content://com.x/tree/root", "document": "content://com.x/tree/root/document/root"}
FILE = {"platform": "android", "kind": "file", "id": "d2", "uri": None,
        "document": "content://com.x/document/9"}


def run(coro):
    return asyncio.run(coro)


class FakePage:
    def __init__(self, platform, web=False):
        self.platform = platform
        self.web = web


def attach(native, platform, web=False):
    page = FakePage(platform, web=web)
    native._page_or_none = lambda: page
    return page


def recording_client(native, results=None, errors=None):
    calls = []

    async def fake_invoke(method_name, arguments=None, timeout=None):
        calls.append((method_name, arguments, timeout))
        error = (errors or {}).get(method_name)
        if error is not None:
            raise error
        value = (results or {}).get(method_name)
        return value(arguments) if callable(value) else value

    native._invoke_method = fake_invoke
    return calls


def forbid_client(native):
    async def fail(*args, **kwargs):
        raise AssertionError("native client must not be called")

    native._invoke_method = fail


def call_all(native, tmp_path):
    src = tmp_path / "book.epub"
    src.write_bytes(b"x" * 10)
    return {
        "pick_folder": run(native.pick_folder()),
        "pick_save_location": run(native.pick_save_location("a.epub", "application/epub+zip", str(src))),
        "pick_document": run(native.pick_document(["application/epub+zip"])),
        "list_children": run(native.list_children(FOLDER)),
        "create_file": run(native.create_file(FOLDER, "a.epub", "application/epub+zip")),
        "create_folder": run(native.create_folder(FOLDER, "Book")),
        "write_file": run(native.write_file(FILE, str(src))),
        "rename_document": run(native.rename_document(FILE, "b.epub")),
        "stat": run(native.stat(FILE)),
        "delete": run(native.delete(FILE)),
        "query_root": run(native.query_root(FOLDER)),
        "release": run(native.release(FOLDER)),
        "list_grants": run(native.list_grants()),
        "cancel_document_op": run(native.cancel_document_op("op")),
        "take_document_results": run(native.take_document_results()),
    }


def assert_safe_defaults(answers):
    for name in DOC_METHODS:
        answer = answers[name]
        assert answer["ok"] is False and answer["error"] == "unavailable", name
        assert answer["retryable"] is False
    assert answers["release"] is False and answers["cancel_document_op"] is False
    assert answers["list_grants"] == [] and answers["take_document_results"] == []


# =============================================================== Python service, fake channel


def test_python_exposes_every_document_method():
    for name in PYTHON_METHODS:
        assert asyncio.iscoroutinefunction(getattr(gn.GlossarionNative, name)), name


def test_defaults_when_not_attached(tmp_path):
    native = gn.GlossarionNative()
    forbid_client(native)
    assert_safe_defaults(call_all(native, tmp_path))


@pytest.mark.parametrize("platform", [ft.PagePlatform.WINDOWS, ft.PagePlatform.MACOS, ft.PagePlatform.LINUX])
def test_defaults_on_desktop(platform, tmp_path):
    native = gn.GlossarionNative()
    attach(native, platform)
    forbid_client(native)
    assert_safe_defaults(call_all(native, tmp_path))
    assert run(native.get_platform_info())["documents"] is False


def test_defaults_on_web(tmp_path):
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID, web=True)
    forbid_client(native)
    assert_safe_defaults(call_all(native, tmp_path))


def test_android_marshals_pickers_with_long_timeouts():
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    target = dict(FOLDER, name="Glossarion", own_folder=False)
    calls = recording_client(native, results={
        "pick_folder": {"ok": True, "target": target, "persisted": True},
        "pick_save_location": {"ok": True, "document": FILE,
                               "write": {"ok": False, "error": "odd", "message": "x"}},
        "pick_document": {"ok": True, "document": FILE},
    })
    answer = run(native.pick_folder(initial=FOLDER))
    assert answer["ok"] is True and answer["error"] is None and answer["target"] == target
    saved = run(native.pick_save_location("Book.epub", "application/epub+zip", "/cache/snap.epub",
                                          mode_chain=["rwt", "bogus", "wt", "rwt"], op_id="op1"))
    assert saved["ok"] is True and saved["write"]["error"] == "provider_error" and saved["write"]["retryable"]
    run(native.pick_document(["application/pdf", ""]))
    by_name = {name: (args, timeout) for name, args, timeout in calls}
    args, timeout = by_name["pick_folder"]
    assert timeout == docs.PICKER_TIMEOUT
    assert args["initial"] == FOLDER and args["initial"] is not FOLDER  # copied plain map
    assert isinstance(args["op_id"], str) and len(args["op_id"]) >= 16
    args, timeout = by_name["pick_save_location"]
    assert timeout == docs.PICKER_TIMEOUT
    assert args == {
        "op_id": "op1", "name": "Book.epub", "mime_type": "application/epub+zip",
        "source_path": "/cache/snap.epub", "initial": None, "mode_chain": ["rwt", "wt"],
    }
    assert by_name["pick_document"][0]["mime_types"] == ["application/pdf"]


def test_write_file_marshals_and_sizes_its_timeout(tmp_path):
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    src = tmp_path / "snap.epub"
    src.write_bytes(b"\0" * (3 * 1024 * 1024))
    calls = recording_client(native, results={
        "write_file": lambda a: {"ok": True, "document": FILE, "mode": "wt", "written": 3 * 1024 * 1024,
                                 "verified_size": 3 * 1024 * 1024, "op_id": a["op_id"]},
    })
    answer = run(native.write_file(FOLDER, str(src), name="Book.epub", mime_type="application/epub+zip",
                                   on_exists="adopt", op_id="w1"))
    assert answer["ok"] is True and answer["mode"] == "wt" and answer["op_id"] == "w1"
    name, args, timeout = calls[0]
    assert name == "write_file"
    assert args == {
        "op_id": "w1", "ref": FOLDER, "source_path": str(src), "name": "Book.epub",
        "mime_type": "application/epub+zip", "on_exists": "adopt", "mode_chain": ["wt", "rwt", "w"],
        "verify": True,
    }
    assert timeout == pytest.approx(123.0)
    run(native.write_file(FILE, str(src), timeout=5.0, verify=False))
    assert calls[1][2] == 5.0 and calls[1][1]["verify"] is False and calls[1][1]["op_id"]


def test_string_refs_are_accepted():
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    calls = recording_client(native, results={"list_children": {"ok": True, "children": []}, "stat": {"ok": True}})
    run(native.list_children("content://com.x/tree/abc"))
    run(native.stat("content://com.x/tree/abc/document/abc%2Fbook.epub"))
    assert calls[0][1]["folder"] == {"kind": "folder", "uri": "content://com.x/tree/abc"}
    assert calls[1][1]["document"] == {"kind": "file", "document": "content://com.x/tree/abc/document/abc%2Fbook.epub"}


def test_invalid_mode_chain_is_refused_locally(tmp_path):
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    forbid_client(native)
    answer = run(native.write_file(FILE, str(tmp_path / "x"), mode_chain=["a", "rwx"]))
    assert answer["ok"] is False and answer["error"] == "bad_args" and answer["op_id"]


def test_results_are_normalised():
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.IOS)
    recording_client(native, results={
        "stat": {"ok": False, "error": "weird_code", "message": None},
        "delete": "nonsense",
        "query_root": {"ok": False, "error": "provider_error", "scope": "nope"},
        "list_children": {"ok": True, "children": [FILE], "complete": False},
    })
    stat = run(native.stat(FILE))
    assert stat["error"] == "provider_error" and stat["retryable"] is True and "weird_code" in stat["message"]
    assert run(native.delete(FILE))["error"] == "provider_error"
    root = run(native.query_root(FOLDER))
    assert root["scope"] is None and root["retryable"] is True
    listing = run(native.list_children(FOLDER))
    assert listing["ok"] is True and listing["children"] == [FILE] and listing["complete"] is False


def test_write_timeout_cancels_the_native_copy(tmp_path):
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    src = tmp_path / "s.txt"
    src.write_text("abc")
    calls = recording_client(native, results={"cancel_document_op": True},
                             errors={"write_file": TimeoutError("Timeout waiting for invokeMethod write_file")})
    answer = run(native.write_file(FILE, str(src), op_id="slow"))
    assert answer["ok"] is False and answer["error"] == "timeout" and answer["retryable"] is True
    assert [(c[0], c[1]) for c in calls] == [
        ("write_file", calls[0][1]), ("cancel_document_op", {"op_id": "slow"}),
    ]


def test_missing_dart_service_marks_unavailable():
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    calls = recording_client(native, errors={
        "pick_folder": RuntimeError("Timeout waiting for invoke method listener for GlossarionNative(3).pick_folder"),
    })
    assert run(native.pick_folder())["error"] == "unavailable"
    assert run(native.list_children(FOLDER))["error"] == "unavailable"
    assert len(calls) == 1 and native.native_available is False


def test_platform_errors_map_without_logging_uris(caplog):
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    recording_client(native, errors={
        "stat": RuntimeError("GlossarionNative.stat failed: native_error: content://secret/tree/x"),
        "delete": RuntimeError("GlossarionNative.delete failed: bad_args: document is required"),
    })
    with caplog.at_level(logging.DEBUG, logger="flet_glossarion_native"):
        assert run(native.stat(FILE))["error"] == "provider_error"
        assert run(native.delete(FILE))["error"] == "bad_args"
    assert "content://" not in caplog.text and "secret" not in caplog.text


def test_small_methods_coerce_results():
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    calls = recording_client(native, results={
        "release": True,
        "list_grants": [{"uri": "content://x/tree/1", "read": True, "write": True}, "junk"],
        "cancel_document_op": 1,
        "take_document_results": [{"type": "pick_result", "op_id": "p", "status": "cancelled"}, None],
    })
    assert run(native.release("content://x/tree/1")) is True
    assert run(native.list_grants()) == [{"uri": "content://x/tree/1", "read": True, "write": True}]
    assert run(native.cancel_document_op("w")) is True
    assert run(native.cancel_document_op("")) is False
    assert run(native.take_document_results()) == [{"type": "pick_result", "op_id": "p", "status": "cancelled"}]
    assert calls[0][1] == {"target": "content://x/tree/1"}


def test_save_to_downloads_passes_replace_uri():
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    calls = recording_client(native, results={"save_to_downloads": "content://media/external/downloads/7"})
    uri = "content://media/external/downloads/7"
    assert run(native.save_to_downloads("/x/a.epub", "a.epub", "application/epub+zip", replace_uri=uri)) == uri
    assert run(native.save_to_downloads("/x/b.epub", "b.epub", "application/epub+zip")) == uri
    assert calls[0][1]["replace_uri"] == uri and calls[1][1]["replace_uri"] is None
    assert calls[0][2] >= 600


def test_document_event_decodes_with_flet_from_dict():
    native = gn.GlossarionNative()
    progress = from_dict(gn.DocumentEvent, {
        "control": native, "name": "document", "type": "progress", "op_id": "w", "written": 5, "total": 9,
    })
    assert progress.type == gn.DocumentEventType.PROGRESS and (progress.written, progress.total) == (5, 9)
    late = from_dict(gn.DocumentEvent, {
        "control": native, "name": "document", "type": "pick_result", "op_id": "p", "kind": "folder",
        "status": "ok", "result": {"ok": True, "target": {"id": "d1", "size": None}}, "unknown": 1,
    })
    assert late.type == gn.DocumentEventType.PICK_RESULT and late.result["target"]["id"] == "d1"
    from flet.controls.control_event import get_event_field_type
    assert get_event_field_type(native, "on_document") is gn.DocumentEvent


# ============================================================================ documents.py


def test_stable_id_is_fnv1a64():
    # Reference vectors of FNV-1a/64.
    assert docs.stable_id("") == "dcbf29ce484222325"
    assert docs.stable_id("a") == "daf63dc4c8601ec8c"
    assert docs.stable_id("foobar") == "d85944171f73967e8"
    assert docs.stable_id("android:content://x/tree/1") == docs.stable_id("android:content://x/tree/1")


def test_mode_and_timeout_helpers():
    assert docs.normalize_modes(None) == ("wt", "rwt", "w")
    assert docs.normalize_modes("RWT") == ("rwt",)
    assert docs.normalize_modes(["w", "x", "w", "rw"]) == ("w", "rw")
    assert docs.normalize_modes([]) == ()
    assert docs.write_timeout(0) == 120.0
    assert docs.write_timeout(None) == 120.0
    assert docs.write_timeout(10 * 1024 * 1024) == 130.0
    assert docs.write_timeout(10 ** 13) == docs.MAX_WRITE_TIMEOUT
    assert docs.write_timeout("junk") == 120.0


def test_result_helpers():
    failure = docs.error_result(docs.DocumentError.MISSING, "gone", scope="document", proven=True)
    assert failure == {"ok": False, "error": "missing", "message": "gone", "scope": "document",
                       "retryable": False, "proven": True}
    assert docs.error_result("nonsense")["error"] == "provider_error"
    assert docs.normalize_result({"ok": True, "x": 1}) == {"ok": True, "x": 1, "error": None,
                                                         "message": None, "retryable": False}
    assert docs.normalize_result(None)["error"] == "provider_error"
    assert docs.is_retryable("provider_error") and docs.is_retryable(docs.DocumentError.TIMEOUT)
    assert not docs.is_retryable("no_space") and not docs.is_retryable("missing")


def test_ref_helpers_and_redaction():
    assert docs.ref_id(FOLDER) == "d1"
    legacy = {"platform": "android", "uri": "content://com.x/tree/root"}
    assert docs.ref_id(legacy) == docs.stable_id("android:content://com.x/tree/root")
    assert docs.ref_id("x") is None
    assert docs.same_destination(FOLDER, dict(FOLDER, name="other"))
    assert not docs.same_destination(FOLDER, FILE) and not docs.same_destination({}, {})
    label = docs.redact("content://com.google.android.apps.docs.storage/tree/acc%3D1%3Bdoc%3Dsecret")
    assert label.startswith("com.google.android.apps.docs.storage#") and "secret" not in label
    assert "Drive" in docs.redact(dict(FOLDER, provider_label="Drive")) and "tree" not in docs.redact(FOLDER)
    assert docs.redact("Ym9va21hcms=").startswith("bookmark#")
    assert docs.names_of([{"name": "a"}, {"x": 1}, "junk"]) == ["a"]


# ======================================================== Android semantics (fake provider)


def make(**options):
    provider = FakeCloudProvider(label="Drive", **options)
    native = FakeDocumentsNative(provider, chunk_bytes=4)
    folder = provider.add_folder("Glossarion")
    provider.next_pick("folder", folder)
    target = run(native.pick_folder())["target"]
    return provider, native, folder, target


def src_file(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def test_pick_folder_persists_and_ids_the_tree():
    provider, native, folder, target = make()
    tree = provider.tree_uri(folder)
    assert target["id"] == docs.stable_id("android:" + tree) and target["kind"] == "folder"
    assert target["uri"] == tree and target["persisted"] is True and target["own_folder"] is False
    assert set(target) <= set(docs.REF_KEYS)
    assert run(native.list_grants())[0]["uri"] == tree
    provider.next_pick("folder", None)
    assert run(native.pick_folder())["error"] == "cancelled"


def test_create_then_overwrite_in_place(tmp_path):
    provider, native, folder, target = make()
    first = run(native.write_file(target, src_file(tmp_path, "a", b"version one"), name="Book.epub"))
    assert first["ok"] and first["created"] and first["mode"] == "wt" and first["verified_size"] == 11
    doc = first["document"]
    second = run(native.write_file(doc, src_file(tmp_path, "b", b"v2")))
    assert second["ok"] and not second["created"] and second["document"]["id"] == doc["id"]
    files = provider.all_files()
    assert len(files) == 1 and bytes(files[0].data) == b"v2" and files[0].versions == 2


def test_wt_refused_falls_back_to_rwt(tmp_path):
    provider, native, folder, target = make(accepted_modes=("rwt", "w", "r"))
    answer = run(native.write_file(target, src_file(tmp_path, "a", b"data"), name="x.txt"))
    assert answer["ok"] and answer["mode"] == "rwt"
    assert [a["error"] for a in answer["attempts"]] == ["unsupported_mode", "ok"]


def test_non_truncating_w_never_leaves_a_stale_tail(tmp_path):
    provider, native, folder, target = make(accepted_modes=("w", "r"), truncates_w=False, seekable=False)
    node = provider.add_file("Book.epub", b"0123456789", parent=folder)
    doc = run(native.list_children(target, names=["Book.epub"]))["children"][0]
    shorter = run(native.write_file(doc, src_file(tmp_path, "s", b"abc")))
    assert shorter["error"] == "unsupported_mode" and shorter["needs_replace"] is True
    assert bytes(node.data) == b"0123456789"  # untouched: no write was attempted
    longer = run(native.write_file(doc, src_file(tmp_path, "l", b"abcdefghijkl")))
    assert longer["ok"] and longer["mode"] == "w" and bytes(node.data) == b"abcdefghijkl"
    # The caller's create-new + delete-old fallback for the shorter book:
    fresh = run(native.write_file(target, src_file(tmp_path, "s2", b"abc"), name="Book.epub"))
    assert fresh["ok"] and fresh["document"]["id"] != doc["id"]
    assert run(native.delete(doc))["ok"] is True
    assert [bytes(n.data) for n in provider.all_files()] == [b"abc"]


def test_seekable_w_is_cut_to_length(tmp_path):
    provider, native, folder, target = make(accepted_modes=("w", "r"), truncates_w=False, seekable=True)
    node = provider.add_file("t.txt", b"0123456789", parent=folder)
    doc = run(native.list_children(target, names=["t.txt"]))["children"][0]
    # Equal or longer is allowed; the descriptor is a real file so the tail is cut anyway.
    answer = run(native.write_file(doc, src_file(tmp_path, "x", b"ABCDEFGHIJ")))
    assert answer["ok"] and answer["verified_size"] == 10 and bytes(node.data) == b"ABCDEFGHIJ"


def test_offline_provider_is_not_deleted(tmp_path):
    provider, native, folder, target = make()
    doc = run(native.write_file(target, src_file(tmp_path, "a", b"one"), name="b.epub"))["document"]
    provider.offline = True
    answer = run(native.write_file(doc, src_file(tmp_path, "b", b"two")))
    assert answer["error"] == "provider_error" and answer["retryable"] is True
    assert answer["scope"] == "document" and "proven" not in answer
    assert len(provider.all_files()) == 1  # nothing re-created


def test_deleted_document_is_missing_only_when_the_root_answers(tmp_path):
    provider, native, folder, target = make()
    doc = run(native.write_file(target, src_file(tmp_path, "a", b"one"), name="b.epub"))["document"]
    provider.delete_document(provider.files_named("b.epub", folder)[0])
    answer = run(native.write_file(doc, src_file(tmp_path, "b", b"two")))
    assert answer["error"] == "missing" and answer["scope"] == "document" and answer["proven"] is True
    again = run(native.write_file(target, src_file(tmp_path, "c", b"two"), name="b.epub"))
    assert again["ok"] and again["created"] and len(provider.all_files()) == 1


def test_document_moved_out_affects_only_that_document(tmp_path):
    provider, native, folder, target = make()
    doc = run(native.write_file(target, src_file(tmp_path, "a", b"one"), name="b.epub"))["document"]
    provider.move_out(provider.files_named("b.epub", folder)[0])
    answer = run(native.write_file(doc, src_file(tmp_path, "b", b"two")))
    assert (answer["error"], answer["scope"]) == ("missing", "document")
    assert run(native.query_root(target))["ok"] is True


def test_revoked_grant_needs_a_relink(tmp_path):
    provider, native, folder, target = make()
    doc = run(native.write_file(target, src_file(tmp_path, "a", b"one"), name="b.epub"))["document"]
    provider.revoke()
    answer = run(native.write_file(doc, src_file(tmp_path, "b", b"two")))
    assert (answer["error"], answer["scope"]) == ("permission_lost", "target")
    assert run(native.query_root(target))["error"] == "permission_lost"
    provider.uninstall()
    assert run(native.query_root(target))["provider_missing"] is True


def test_per_file_grant(tmp_path):
    provider = FakeCloudProvider(label="Drive", supports_tree=False)
    native = FakeDocumentsNative(provider)
    provider.next_pick("save_location", name="Book.epub")
    picked = run(native.pick_save_location("Book.epub", "application/epub+zip", src_file(tmp_path, "a", b"v1")))
    doc = picked["document"]
    assert picked["ok"] and picked["write"]["ok"] and doc["uri"] is None and doc["persisted"] is True
    assert run(native.write_file(doc, src_file(tmp_path, "b", b"v2")))["ok"]
    assert [bytes(n.data) for n in provider.all_files()] == [b"v2"]
    provider.delete_document(provider.all_files()[0])
    gone = run(native.write_file(doc, src_file(tmp_path, "c", b"v3")))
    assert (gone["error"], gone["scope"], gone["proven"]) == ("missing", "document", False)
    provider.revoke()
    assert run(native.stat(doc))["error"] == "permission_lost"
    provider.next_pick("folder", provider.add_folder("x"))
    assert run(native.pick_folder())["error"] == "cancelled"  # not offered as a folder


def test_failed_create_that_appears_late_is_adopted(tmp_path):
    provider, native, folder, target = make()
    provider.fail_next_creates = 1
    provider.create_appears_late = True
    answer = run(native.write_file(target, src_file(tmp_path, "a", b"data"), name="Book.epub"))
    assert answer["ok"] and answer["created"] and len(provider.files_named("Book.epub", folder)) == 1


def test_failed_create_is_retried_once(tmp_path):
    provider, native, folder, target = make()
    provider.fail_next_creates = 1
    assert run(native.write_file(target, src_file(tmp_path, "a", b"data"), name="Book.epub"))["ok"]
    assert len(provider.files_named("Book.epub", folder)) == 1
    provider.fail_next_creates = 2
    answer = run(native.create_file(target, "Other.epub", "application/epub+zip"))
    assert answer["error"] == "provider_error" and answer["retryable"] is True
    assert provider.files_named("Other.epub", folder) == []


def test_on_exists_rules(tmp_path):
    provider, native, folder, target = make()
    existing = provider.add_file("Book.epub", b"old", parent=folder)
    adopted = run(native.create_file(target, "Book.epub", "application/epub+zip", on_exists="adopt"))
    assert adopted["adopted"] and adopted["document"]["name"] == "Book.epub"
    refused = run(native.create_file(target, "Book.epub", "application/epub+zip", on_exists="fail"))
    assert refused["error"] == "exists" and refused["document"]["name"] == "Book.epub"
    assert run(native.create_file(target, "Book.epub", "application/epub+zip"))["created"]
    sub = run(native.create_folder(target, "Series A"))
    again = run(native.create_folder(target, "Series A"))
    assert sub["created"] and again["adopted"] and sub["folder"]["id"] == again["folder"]["id"]
    assert sub["folder"]["kind"] == "folder"
    inside = run(native.write_file(sub["folder"], src_file(tmp_path, "a", b"x"), name="Book.epub"))
    assert inside["ok"] and bytes(existing.data) == b"old"


def test_no_space_cancel_and_source_errors(tmp_path):
    provider, native, folder, target = make()
    provider.no_space = True
    full = run(native.write_file(target, src_file(tmp_path, "a", b"data"), name="a.txt"))
    assert full["error"] == "no_space" and full["retryable"] is False
    provider.no_space = False
    doc = run(native.write_file(target, src_file(tmp_path, "b", b"data"), name="b.txt"))["document"]
    events = []
    native.on_progress = events.append

    def cancel_after_first_chunk(event):
        events.append(event)
        native.cancelled.add("op-c")

    native.on_progress = cancel_after_first_chunk
    stopped = run(native.write_file(doc, src_file(tmp_path, "c", b"0123456789abcdef"), op_id="op-c"))
    assert stopped["error"] == "cancelled" and stopped["remote_damaged"] is True
    assert events and events[0]["op_id"] == "op-c" and events[0]["written"] == 4
    missing = run(native.write_file(doc, str(tmp_path / "nope")))
    assert (missing["error"], missing["scope"]) == ("source_missing", "source")
    provider.fail_after_bytes = 2
    native.on_progress = None
    broken = run(native.write_file(doc, src_file(tmp_path, "d", b"abcdef")))
    assert broken["error"] == "provider_error" and broken["remote_damaged"] is True


def test_rename_keeps_the_document_and_refuses_taken_names(tmp_path):
    provider, native, folder, target = make()
    doc = run(native.write_file(target, src_file(tmp_path, "a", b"x"), name="Book (1).epub"))["document"]
    run(native.write_file(target, src_file(tmp_path, "b", b"y"), name="Other.epub"))
    renamed = run(native.rename_document(doc, "Book.epub"))
    assert renamed["ok"] and renamed["document"]["name"] == "Book.epub" and renamed["document"]["id"] == doc["id"]
    assert sorted(n.name for n in provider.all_files()) == ["Book.epub", "Other.epub"]
    taken = run(native.rename_document(renamed["document"], "Other.epub"))
    assert (taken["error"], taken["scope"]) == ("exists", "document")
    root = run(native.rename_document(target, "Elsewhere"))
    assert root["error"] == "bad_args"
    provider.supports_rename = False
    assert run(native.rename_document(renamed["document"], "New.epub"))["error"] == "unavailable"
    provider.delete_document(provider.files_named("Book.epub", folder)[0])
    gone = run(native.rename_document(renamed["document"], "New.epub"))
    assert (gone["error"], gone.get("proven")) == ("missing", True)


def test_release_only_drops_the_named_grant(tmp_path):
    provider, native, folder, target = make()
    doc = run(native.write_file(target, src_file(tmp_path, "a", b"x"), name="a.txt"))["document"]
    assert run(native.release(doc)) is False  # a child has no grant of its own
    assert len(run(native.list_grants())) == 1
    assert run(native.release(target)) is True and run(native.list_grants()) == []


# ======================================================== static cross-language checks


def _source(path):
    return path.read_text(encoding="utf-8")


def _code_only(text, lang):
    """Text without comments and string literals (interpolations dropped too)."""
    out = []
    i, n = 0, len(text)
    quotes = ('"', "'") if lang in ("kotlin", "dart") else ('"',)

    def skip_string(i, quote):
        triple = text.startswith(quote * 3, i)
        if triple:
            end = text.find(quote * 3, i + 3)
            return n if end < 0 else end + 3
        i += 1
        while i < n:
            ch = text[i]
            if lang == "swift" and text.startswith("\\(", i):
                i = skip_group(i + 2, "(", ")")
                continue
            if lang in ("kotlin", "dart") and text.startswith("${", i):
                i = skip_group(i + 2, "{", "}")
                continue
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                return i + 1
            if ch == "\n":
                raise AssertionError(f"unterminated string near: {text[max(0, i - 60):i]!r}")
            i += 1
        return i

    def skip_group(i, open_ch, close_ch):
        depth = 1
        while i < n and depth:
            ch = text[i]
            if ch in quotes:
                i = skip_string(i, ch)
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
            i += 1
        return i

    while i < n:
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        ch = text[i]
        if ch in quotes:
            i = skip_string(i, ch)
            continue
        out.append(ch)
        i += 1
    return "".join(out)


@pytest.mark.parametrize("path,lang", [
    (KOTLIN_DOCS, "kotlin"), (KOTLIN_PLUGIN, "kotlin"), (SWIFT_DOCS, "swift"), (SWIFT_PLUGIN, "swift"),
    (DART_SERVICE, "dart"),
])
def test_sources_are_balanced(path, lang):
    code = _code_only(_source(path), lang)
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, ch in enumerate(code):
        if ch in "([{":
            stack.append((ch, index))
        elif ch in ")]}":
            assert stack and stack[-1][0] == pairs[ch], f"{path.name}: unbalanced {ch!r} near {code[max(0, index - 80):index]!r}"
            stack.pop()
    assert not stack, f"{path.name}: unclosed {stack[-1][0]!r} near {code[stack[-1][1]:stack[-1][1] + 80]!r}"


def test_method_names_match_across_languages():
    dart = _source(DART_SERVICE)
    block = dart[dart.index("Future<dynamic> _invokeMethod"):dart.index("dynamic _defaultResult")]
    dart_cases = set(re.findall(r"case '([a-z_]+)':", block))
    assert PYTHON_METHODS <= dart_cases
    dart_set = set(re.findall(r"'([a-z_]+)'", dart[dart.index("_kDocumentMethods"):dart.index("};")]))
    assert dart_set == DOC_METHODS

    kotlin = _source(KOTLIN_DOCS)
    kotlin_methods = set(re.findall(r'"([a-z_]+)"', re.search(r"val METHODS = setOf\((.*?)\)", kotlin, re.S).group(1)))
    assert kotlin_methods == PLATFORM_METHODS
    handled = set(re.findall(r'"([a-z_]+)" ->', kotlin))
    assert PLATFORM_METHODS <= handled

    swift = _source(SWIFT_DOCS)
    swift_methods = set(re.findall(r'"([a-z_]+)"', re.search(r"static let methods: Set<String> = \[(.*?)\]", swift, re.S).group(1)))
    assert swift_methods == PLATFORM_METHODS
    swift_cases = set(re.findall(r'case "([a-z_]+)":', swift))
    assert PLATFORM_METHODS <= swift_cases

    python = {name for name in dir(gn.GlossarionNative) if not name.startswith("_")}
    assert PYTHON_METHODS <= python
    # Both plugins route these methods to their DocumentDestinations before anything else.
    assert "docs.handles(call.method)" in _source(KOTLIN_PLUGIN)
    assert "DocumentDestinations.methods.contains(call.method)" in _source(SWIFT_PLUGIN)


def test_error_codes_match_python():
    codes = {e.value for e in docs.DocumentError}
    kotlin = set(re.findall(r'const val ERR_[A-Z_]+ = "([a-z_]+)"', _source(KOTLIN_DOCS)))
    swift = set(re.findall(r'static let err[A-Za-z]+ = "([a-z_]+)"', _source(SWIFT_DOCS)))
    assert kotlin <= codes and swift <= codes
    assert codes == kotlin | swift | {"timeout"}  # timeout is Python's own
    assert "unsupported_mode" in kotlin  # only Android has write modes
    kotlin_retry = re.search(r"RETRYABLE = setOf\((.*?)\)", _source(KOTLIN_DOCS)).group(1)
    retry_names = re.findall(r"ERR_[A-Z_]+", kotlin_retry)
    kotlin_values = dict(re.findall(r'const val (ERR_[A-Z_]+) = "([a-z_]+)"', _source(KOTLIN_DOCS)))
    assert {kotlin_values[name] for name in retry_names} | {"timeout"} == set(docs.RETRYABLE_ERRORS)
    swift_retry = re.search(r"static let retryable: Set<String> = \[(.*?)\]", _source(SWIFT_DOCS)).group(1)
    swift_values = dict(re.findall(r'static let (err[A-Za-z]+) = "([a-z_]+)"', _source(SWIFT_DOCS)))
    assert {swift_values[name.strip()] for name in swift_retry.split(",")} | {"timeout"} == set(docs.RETRYABLE_ERRORS)
    scopes = {s.value for s in docs.DocumentScope}
    assert set(re.findall(r'const val SCOPE_[A-Z]+ = "([a-z]+)"', _source(KOTLIN_DOCS))) == scopes
    assert set(re.findall(r'static let scope[A-Za-z]+ = "([a-z]+)"', _source(SWIFT_DOCS))) == scopes


def test_document_event_keys_match_dataclass():
    allowed = {f.name for f in fields(gn.DocumentEvent)}
    kotlin = _source(KOTLIN_DOCS)
    keys = set()
    for name in ("pickEvent", "progress"):
        body = kotlin[kotlin.index(f"private fun {name}("):]
        body = body[:body.index("\n    }\n")]
        keys |= set(re.findall(r'"([a-z_]+)" to', body))
    swift = _source(SWIFT_DOCS)
    swift_keys = set(re.findall(r'"([a-z_]+)":', swift[swift.index("func onDartAttach"):swift.index("// MARK: - Resolving")]))
    swift_keys |= set(re.findall(r'"([a-z_]+)":', swift[swift.index("private func emitProgress"):swift.index("private func isCancelled")]))
    for found in (keys, swift_keys):
        assert {"type", "op_id"} <= found
        assert found <= allowed, found - allowed
    types = {t.value for t in gn.DocumentEventType}
    assert set(re.findall(r'const val EVENT_[A-Z_]+ = "([a-z_]+)"', kotlin)) == types
    assert set(re.findall(r'"type": "([a-z_]+)"', swift)) == types
    dart = _source(DART_SERVICE)
    assert "case 'document':" in dart and "control.triggerEvent('document', event)" in dart
    assert "event['type'] == 'pick_result'" in dart and "result['document_results']" in dart


def test_ref_keys_match_the_contract():
    allowed = set(docs.REF_KEYS)
    kotlin = _source(KOTLIN_DOCS)
    ref_body = kotlin[kotlin.index("private fun refMap("):kotlin.index("private fun decorate(")]
    kotlin_keys = set(re.findall(r'"([a-z_]+)" to', ref_body))
    decorate_body = kotlin[kotlin.index("private fun decorate("):kotlin.index("private fun providerInfo(")]
    kotlin_keys |= set(re.findall(r'target\["([a-z_]+)"\]', decorate_body))
    swift = _source(SWIFT_DOCS)
    describe = swift[swift.index("static func describe("):swift.index("static func fileStamp(")]
    swift_keys = set(re.findall(r'"([a-z_]+)":', describe)) | set(re.findall(r'ref\["([a-z_]+)"\]', describe))
    assert kotlin_keys <= allowed, kotlin_keys - allowed
    assert swift_keys <= allowed, swift_keys - allowed
    assert allowed == kotlin_keys | swift_keys | {"persisted"}
    for side in (kotlin_keys, swift_keys):
        assert {"platform", "kind", "id", "name", "provider", "can_write", "own_folder"} <= side


def test_constants_match_python():
    kotlin = _source(KOTLIN_DOCS)
    valid = re.search(r"val VALID_MODES = listOf\((.*?)\)", kotlin).group(1)
    default = re.search(r"val DEFAULT_MODES = listOf\((.*?)\)", kotlin).group(1)
    assert tuple(re.findall(r'"([a-z]+)"', valid)) == docs.VALID_MODES
    assert tuple(re.findall(r'"([a-z]+)"', default)) == docs.DEFAULT_MODE_CHAIN
    assert set(re.findall(r'"([a-z]+)"', re.search(r"TRUNCATING_MODES = setOf\((.*?)\)", kotlin).group(1))) == set(docs.TRUNCATING_MODES)
    swift = _source(SWIFT_DOCS)
    for text, prime in ((kotlin, "0x100000001b3uL"), (swift, "0x100000001b3")):
        assert "0xcbf29ce484222325" in text.lower() and prime in text
    assert 'stableId("android:$uri")' in kotlin and 'stableId("ios:" + canonicalPath(url))' in swift
    assert "CHUNK_BYTES = 1 shl 20" in kotlin and "chunkBytes = 1 << 20" in swift
    codes = [int(v, 16) for v in re.findall(r"REQUEST_[A-Z]+ = (0x[0-9A-F]+)", kotlin)]
    assert len(codes) == 3 and len(set(codes)) == 3 and all(0 < c < 0x10000 for c in codes)


def test_manifest_declares_provider_visibility_and_no_storage_permission():
    root = ET.parse(MANIFEST).getroot()
    actions = [a.get(ANDROID_NS + "name") for q in root.findall("queries") for i in q.findall("intent")
               for a in i.findall("action")]
    assert actions == ["android.content.action.DOCUMENTS_PROVIDER"]
    permissions = {p.get(ANDROID_NS + "name") for p in root.findall("uses-permission")}
    for forbidden in ("READ_EXTERNAL_STORAGE", "WRITE_EXTERNAL_STORAGE", "MANAGE_EXTERNAL_STORAGE",
                      "MANAGE_DOCUMENTS", "READ_MEDIA_IMAGES"):
        assert "android.permission." + forbidden not in permissions


def test_kotlin_follows_the_critic_rules():
    kotlin = _source(KOTLIN_DOCS)
    pick = kotlin[kotlin.index("private fun completePick("):]
    # Persist only what the picker granted.
    assert re.search(r"val flags = grantFlags and\s*\(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION\)", pick)
    assert "takePersistableUriPermission(uri, flags)" in pick
    chain = kotlin[kotlin.index("private fun writeChain("):kotlin.index("private fun streamInto(")]
    # Non-truncating modes only when the new file is not shorter than the cloud copy.
    assert "mode in NON_TRUNCATING_MODES" in chain and "total < old" in chain
    assert 'NON_TRUNCATING_MODES = setOf("w", "rw")' in kotlin
    # FileNotFoundException goes through diagnose(), which needs the root to answer.
    assert "classify(e, notFound = ATTEMPT_NOT_FOUND)" in chain and "return diagnosed(ref" in chain
    diagnose = kotlin[kotlin.index("private fun diagnose("):kotlin.index("private fun diagnosed(")]
    assert diagnose.index("queryRows(root)") < diagnose.index("ERR_MISSING to SCOPE_DOCUMENT")
    stream = kotlin[kotlin.index("private fun streamInto("):kotlin.index("private fun readBackSize(")]
    assert "Os.ftruncate(pfd.fileDescriptor, written)" in stream and "closeWithError" in stream
    assert "readBackSize(ref.document)" in stream and "stale_tail" in stream
    assert "statSize" in kotlin[kotlin.index("private fun readBackSize("):kotlin.index("private fun remoteSize(")]
    create = kotlin[kotlin.index("private fun createChild("):kotlin.index("private fun writeFile(")]
    assert "CREATE_RETRY_DELAY_MS" in create and "beforeIds" in create and "childrenNamed(folder, name)" in create
    # Activity results survive config changes: the listener follows every attach / detach.
    plugin = _source(KOTLIN_PLUGIN)
    assert plugin.count("addActivityResultListener(this)") == 2
    assert plugin.count("removeActivityResultListener(this)") == 2
    assert "PluginRegistry.ActivityResultListener" in plugin
    assert '"document_results" to (documents?.onDartAttach()' in plugin
    assert ".commit()" in kotlin[kotlin.index("private fun savePendingRecord("):]
    assert "Process.myPid()" in kotlin[kotlin.index("fun onDartAttach("):]


def test_downloads_replace_keeps_the_file_name():
    plugin = _source(KOTLIN_PLUGIN)
    replace = plugin[plugin.index("private fun ownsDownloadsEntry("):plugin.index("private fun hasLegacyWritePermission(")]
    assert "DISPLAY_NAME" not in replace and "resolver.update" not in replace
    assert "OWNER_PACKAGE_NAME" in replace and "Os.ftruncate" in replace
    assert 'call.argument<String>("replace_uri")' in plugin
    save = plugin[plugin.index("private fun saveToDownloads("):plugin.index("private fun ownsDownloadsEntry(")]
    assert save.index("overwriteDownloadsEntry(") < save.index("resolver.insert(")


def test_swift_targets_ios13_without_uniformtypeidentifiers():
    swift = _source(SWIFT_DOCS)
    code = _code_only(swift, "swift")
    assert "import UniformTypeIdentifiers" not in swift and "UTType" not in code
    assert "forOpeningContentTypes" not in code
    exporting = code.index("forExporting:")
    assert code.rfind("#available(iOS 14.0, *)", 0, exporting) > code.rfind("}", 0, code.rfind("#available(iOS 14.0, *)", 0, exporting))
    assert ".minimalBookmark" in code and "withSecurityScope" not in code
    assert code.count("startAccessingSecurityScopedResource()") <= code.count("stopAccessingSecurityScopedResource()")
    assert ".forReplacing" in code and ".forDeleting" in code and "replaceItemAt(" in code
    assert ".itemReplacementDirectory" in code and "isTrashed(" in code and "isOwnFolder(" in code
    assert "NSHomeDirectory()" in code
    plugin = _source(SWIFT_PLUGIN)
    assert '"document_results": documents.onDartAttach()' in plugin
    assert "static func safeFileName(" in plugin and "private static func safeFileName(" not in plugin
    podspec = (FLUTTER_PKG / "ios" / "flet_glossarion_native.podspec").read_text(encoding="utf-8")
    assert "Sources/flet_glossarion_native/**/*.swift" in podspec


def test_python_modules_parse_as_python_310():
    for name in ("documents.py", "documents_fake.py", "native.py", "types.py", "__init__.py"):
        source = (ROOT / "src" / "flet_glossarion_native" / name).read_text(encoding="utf-8")
        ast.parse(source, filename=name, feature_version=(3, 10))
    fake = (ROOT / "src" / "flet_glossarion_native" / "documents_fake.py").read_text(encoding="utf-8")
    assert "import flet" not in fake and "import flet" not in (ROOT / "src" / "flet_glossarion_native" / "documents.py").read_text(encoding="utf-8")


def test_new_sources_have_uniform_line_endings():
    for path in (KOTLIN_DOCS, SWIFT_DOCS, KOTLIN_PLUGIN, SWIFT_PLUGIN, DART_SERVICE, MANIFEST,
                 ROOT / "src" / "flet_glossarion_native" / "documents.py",
                 ROOT / "src" / "flet_glossarion_native" / "documents_fake.py",
                 ROOT / "src" / "flet_glossarion_native" / "native.py"):
        data = path.read_bytes()
        assert data.count(b"\r\n") in (0, data.count(b"\n")), path.name
        assert not data.startswith(b"\xef\xbb\xbf"), path.name


# =============================================================== U10 fixes (review round)


def test_save_to_downloads_entry_reports_the_name():
    """The transfer.it handoff needs the name MediaStore really gave the entry ("name (1).ext" when taken); the
    same channel method answers {"uri", "name"} when asked (``report_name``), a plain URI from an older build."""
    native = gn.GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    uri = "content://media/external/downloads/9"
    calls = recording_client(native, results={"save_to_downloads": lambda args: (
        {"uri": uri, "name": "Novel (1).epub"} if args.get("report_name") else uri)})
    entry = run(native.save_to_downloads_entry("/x/Novel.epub", "Novel.epub", "application/epub+zip",
                                               replace_uri="content://media/external/downloads/3"))
    assert entry == {"uri": uri, "name": "Novel (1).epub"}
    assert calls[-1][0] == "save_to_downloads" and calls[-1][1]["report_name"] is True
    assert calls[-1][1]["replace_uri"] == "content://media/external/downloads/3" and calls[-1][2] >= 600
    assert run(native.save_to_downloads("/x/Novel.epub", "Novel.epub", "application/epub+zip")) == uri
    assert "report_name" not in calls[-1][1]  # the plain call keeps answering a URI string
    recording_client(native, results={"save_to_downloads": uri})  # an older platform side: the URI only
    assert run(native.save_to_downloads_entry("/x/a.epub", "a.epub", "application/epub+zip")) == {"uri": uri,
                                                                                                    "name": None}
    recording_client(native, results={"save_to_downloads": None})
    assert run(native.save_to_downloads_entry("/x/a.epub", "a.epub", "application/epub+zip")) is None


def test_a_provider_that_refuses_sub_folders_answers_read_only(tmp_path):
    """AOSP's DocumentsProvider.createDocument default throws UnsupportedOperationException("Create not
    supported"): the fake (like DocumentDestinations.createChild) answers read_only + create_unsupported, which
    the cloud sync takes as "lay the books out flat", instead of a provider_error retried for ever."""
    from flet_glossarion_native.documents_fake import ProviderFault

    provider, native, folder, target = make()
    plain = provider.create

    def files_only(parent_uri, name, is_dir):
        if is_dir:
            raise ProviderFault("unsupported", "Create not supported")
        return plain(parent_uri, name, is_dir)

    provider.create = files_only
    answer = run(native.create_folder(target, "Book"))
    assert answer["ok"] is False and answer["error"] == "read_only" and answer.get("create_unsupported") is True
    assert answer["retryable"] is False
    assert run(native.create_file(target, "Book.epub", "application/epub+zip"))["ok"]


def test_kotlin_and_swift_follow_the_review_fixes():
    kotlin = _source(KOTLIN_DOCS)
    create = kotlin[kotlin.index("private fun createChild("):kotlin.index("private fun writeFile(")]
    # "Create not supported" (and a refused directory MIME type) is read_only, not a provider_error retried for ever
    assert "error is UnsupportedOperationException" in create and '"create_unsupported" to true' in create
    assert create.index("ERR_READ_ONLY") < create.index("classify(error, notFound = ERR_PROVIDER)")
    # a save location inside Glossarion's own folder is refused before anything is written there
    pick = kotlin[kotlin.index("private fun completePick("):kotlin.index("private fun savePendingRecord(")]
    assert 'document["own_folder"] == true' in pick and "DocumentsContract.deleteDocument(resolver, uri)" in pick
    assert pick.index('document["own_folder"] == true') < pick.index("writeChain(")
    plugin = _source(KOTLIN_PLUGIN)
    downloads = plugin[plugin.index("private fun saveToDownloadsAsync("):plugin.index("private fun saveToDownloads(")]
    assert 'call.argument<Boolean>("report_name")' in downloads and "downloadsEntryName(context, saved)" in downloads
    name = plugin[plugin.index("private fun downloadsEntryName("):plugin.index("private fun ownsDownloadsEntry(")]
    assert "MediaStore.MediaColumns.DISPLAY_NAME" in name
    swift = _source(SWIFT_DOCS)
    child = swift[swift.index("private func createChild("):swift.index("private func writeFile(")]
    assert "NSFeatureUnsupportedError" in child and '"create_unsupported": true' in child
