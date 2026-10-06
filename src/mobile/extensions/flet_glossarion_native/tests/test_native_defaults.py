"""Host tests for the Python side of flet-glossarion-native.

They cover the behaviour that matters outside a built app (safe defaults on
desktop/web/unattached), argument marshalling on Android/iOS with a fake
client, event payload decoding through Flet's own from_dict, and the contract
between the Python dataclasses and the Dart/Kotlin/Swift sources (event keys,
channel name, manifest entries, plugin classes).

Run: python -m pytest -p no:cacheprovider -W ignore src/mobile/extensions/flet_glossarion_native/tests
"""

import asyncio
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ft = pytest.importorskip("flet")

from flet.controls.control_event import get_event_field_type  # noqa: E402
from flet.utils.from_dict import from_dict  # noqa: E402

import flet_glossarion_native as gn  # noqa: E402
from flet_glossarion_native import (  # noqa: E402
    BackgroundTaskEvent,
    ForegroundEvent,
    GlossarionNative,
    NotificationAction,
    NotificationButton,
    NotificationEvent,
    ShareEvent,
    SharedItem,
)

FLUTTER_PKG = ROOT / "src" / "flutter" / "flet_glossarion_native"
DART_SERVICE = FLUTTER_PKG / "lib" / "src" / "native_service.dart"
DART_TASK = FLUTTER_PKG / "lib" / "src" / "task_handler.dart"
KOTLIN_DIR = FLUTTER_PKG / "android" / "src" / "main" / "kotlin" / "com" / "glossarion" / "flet_glossarion_native"
KOTLIN_PLUGIN = KOTLIN_DIR / "GlossarionNativePlugin.kt"
KOTLIN_TRAMPOLINE = KOTLIN_DIR / "ShareReceiverActivity.kt"
SWIFT_PLUGIN = (
    FLUTTER_PKG / "ios" / "flet_glossarion_native" / "Sources" / "flet_glossarion_native"
    / "GlossarionNativePlugin.swift"
)
MANIFEST = FLUTTER_PKG / "android" / "src" / "main" / "AndroidManifest.xml"
ANDROID_NS = "{http://schemas.android.com/apk/res/android}"


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


def recording_client(native, results=None, error=None):
    calls = []

    async def fake_invoke(method_name, arguments=None, timeout=None):
        calls.append((method_name, arguments, timeout))
        if error is not None:
            raise error
        return (results or {}).get(method_name)

    native._invoke_method = fake_invoke
    return calls


def forbid_client(native):
    async def fail(*args, **kwargs):
        raise AssertionError("native client must not be called")

    native._invoke_method = fail


ALL_DEFAULTS = [
    ("get_initial_shared", (), []),
    ("clear_shared", (), None),
    ("init_notifications", (), False),
    ("show_notification", (1, "t", "b"), False),
    ("cancel_notification", (1,), None),
    ("get_launch_notification", (), None),
    ("start_job_service", ("t", "x"), False),
    ("update_job_service", ("t", "x"), None),
    ("stop_job_service", (), None),
    ("is_job_service_running", (), False),
    ("begin_background_task", ("job",), -1),
    ("end_background_task", (3,), None),
    ("background_time_remaining", (), None),
    ("start_continued_processing", (None, "t", "s"), False),
    ("update_continued_processing", (1, 2), None),
    ("finish_continued_processing", (True,), None),
    ("save_to_downloads", ("/tmp/a.epub", "a.epub", "application/epub+zip"), None),
]


def call(native, name, args):
    method = getattr(native, name)
    if name == "show_notification":
        return run(method(*args, channel_id="jobs.done"))
    return run(method(*args))


# ---------------------------------------------------------------- control


def test_control_type_and_event_types():
    native = GlossarionNative()
    assert native._c == "GlossarionNative"
    assert get_event_field_type(native, "on_share") is ShareEvent
    assert get_event_field_type(native, "on_foreground") is ForegroundEvent
    assert get_event_field_type(native, "on_background_task") is BackgroundTaskEvent
    assert get_event_field_type(native, "on_notification") is NotificationEvent


def test_contract_method_names_exist():
    for name in [
        "get_platform_info", "get_initial_shared", "clear_shared", "init_notifications",
        "show_notification", "cancel_notification", "get_launch_notification",
        "start_job_service", "update_job_service", "stop_job_service",
        "is_job_service_running", "begin_background_task", "end_background_task",
        "background_time_remaining", "start_continued_processing",
        "update_continued_processing", "finish_continued_processing", "save_to_downloads",
    ]:
        assert asyncio.iscoroutinefunction(getattr(GlossarionNative, name)), name


# ---------------------------------------------------------- safe defaults


@pytest.mark.parametrize("name,args,expected", ALL_DEFAULTS)
def test_defaults_when_not_attached(name, args, expected):
    native = GlossarionNative()
    forbid_client(native)
    assert call(native, name, args) == expected


@pytest.mark.parametrize("platform", [ft.PagePlatform.WINDOWS, ft.PagePlatform.MACOS, ft.PagePlatform.LINUX])
@pytest.mark.parametrize("name,args,expected", ALL_DEFAULTS)
def test_defaults_on_desktop(platform, name, args, expected):
    native = GlossarionNative()
    attach(native, platform)
    forbid_client(native)
    assert call(native, name, args) == expected
    assert native.native_available is False


def test_defaults_on_web_even_for_android_user_agent():
    native = GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID, web=True)
    forbid_client(native)
    assert run(native.get_initial_shared()) == []
    info = run(native.get_platform_info())
    assert info["native"] is False
    assert info["unavailable_reason"] == "web client"


def test_disable_env_forces_defaults(monkeypatch):
    monkeypatch.setenv(gn.native.DISABLE_ENV, "1")
    native = GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    forbid_client(native)
    assert run(native.start_job_service("t", "x")) is False
    assert "disabled" in run(native.get_platform_info())["unavailable_reason"]


def test_platform_info_default_shape():
    native = GlossarionNative()
    attach(native, ft.PagePlatform.WINDOWS)
    forbid_client(native)
    info = run(native.get_platform_info())
    assert info["platform"] == "windows"
    assert info["native"] is False
    assert info["extension_version"] == gn.EXTENSION_VERSION
    assert info["continued_processing"] is False


# ------------------------------------------------- forwarding on mobile


def test_android_forwards_and_converts_results():
    native = GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    calls = recording_client(
        native,
        results={
            "get_platform_info": {"platform": "android", "sdk_int": 35, "notifications_enabled": True},
            "get_initial_shared": [
                {"id": "a", "kind": "file", "path": "/data/x/cache/shared/1/a.epub", "name": "a.epub",
                 "mime_type": "application/epub+zip", "size": "42", "uri": "content://x/1",
                 "source": "view", "future_key": 1},
                {"id": "b", "kind": "text", "text": "hello", "source": "send"},
            ],
            "get_launch_notification": {"notification_id": 7, "action_id": "open", "payload": "/jobs/1"},
            "start_job_service": True,
            "is_job_service_running": True,
            "save_to_downloads": "content://media/external/downloads/12",
            "init_notifications": True,
            "show_notification": True,
        },
    )

    info = run(native.get_platform_info())
    assert info["native"] is True and info["sdk_int"] == 35 and info["platform"] == "android"

    items = run(native.get_initial_shared())
    assert [type(i) for i in items] == [SharedItem, SharedItem]
    assert items[0].is_file and items[0].size == 42 and items[0].source == "view"
    assert items[1].kind == "text" and items[1].text == "hello" and not items[1].is_file

    launch = run(native.get_launch_notification())
    assert isinstance(launch, NotificationEvent)
    assert (launch.notification_id, launch.action_id, launch.payload, launch.launched_app) == (7, "open", "/jobs/1", True)
    assert launch.control is native

    assert run(native.start_job_service(
        "Translating", "Ch 1/80",
        buttons=[NotificationButton("stop", "Stop"), {"id": "open", "text": "Open"}],
    )) is True
    assert run(native.is_job_service_running()) is True
    assert run(native.save_to_downloads("/x/a.epub", "a.epub", "application/epub+zip")) == (
        "content://media/external/downloads/12"
    )
    assert run(native.init_notifications()) is True
    assert run(native.show_notification(
        5, "Done", "Book.epub", channel_id="jobs.done", payload="/library/book/1",
        progress=(3, 10), actions=[NotificationAction("share", "Share")],
    )) is True

    by_name = {name: (args, timeout) for name, args, timeout in calls}
    job_args = by_name["start_job_service"][0]
    assert job_args["buttons"] == [{"id": "stop", "text": "Stop"}, {"id": "open", "text": "Open"}]
    assert job_args["service_id"] == gn.JOB_SERVICE_NOTIFICATION_ID
    assert job_args["channel_id"] == gn.CHANNEL_JOBS_PROGRESS
    channels = by_name["init_notifications"][0]["channels"]
    assert [c["id"] for c in channels] == ["jobs.progress", "jobs.done", "jobs.action"]
    assert channels[0]["importance"] == "low"
    note = by_name["show_notification"][0]
    assert note["progress"] == [3, 10]
    assert note["actions"] == [{"id": "share", "title": "Share"}]
    assert by_name["save_to_downloads"][1] >= 600  # large exports get a long timeout


def test_reserved_job_service_id_is_rejected():
    native = GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    forbid_client(native)
    assert run(native.show_notification(gn.JOB_SERVICE_NOTIFICATION_ID, "t", "b", channel_id="jobs.done")) is False


def test_ios_background_calls():
    native = GlossarionNative()
    attach(native, ft.PagePlatform.IOS)
    calls = recording_client(
        native,
        results={
            "begin_background_task": 12,
            "background_time_remaining": 27.5,
            "start_continued_processing": True,
        },
    )
    assert run(native.begin_background_task("job", expiration_title="Paused", expiration_body="Tap to resume")) == 12
    assert run(native.background_time_remaining()) == 27.5
    assert run(native.start_continued_processing(None, "Translating", "Book.epub")) is True
    run(native.update_continued_processing(3, 10, "Chapter 3/10"))
    run(native.end_background_task(12))
    run(native.end_background_task(-1))  # ignored locally
    names = [c[0] for c in calls]
    assert names == [
        "begin_background_task",
        "background_time_remaining",
        "start_continued_processing",
        "update_continued_processing",
        "end_background_task",
    ]
    assert calls[0][1]["expiration_title"] == "Paused"
    assert calls[2][1]["identifier"] is None and calls[2][1]["strategy"] == "fail"


def test_missing_dart_service_disables_further_calls():
    native = GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    calls = recording_client(
        native,
        error=RuntimeError("Timeout waiting for invoke method listener for GlossarionNative(9).get_platform_info"),
    )
    info = run(native.get_platform_info())
    assert info["native"] is False and "no Dart service" in info["unavailable_reason"]
    assert run(native.start_job_service("t", "x")) is False
    assert len(calls) == 1
    assert native.native_available is False


def test_other_errors_return_default_but_keep_trying():
    native = GlossarionNative()
    attach(native, ft.PagePlatform.ANDROID)
    calls = recording_client(native, error=RuntimeError("GlossarionNative.save_to_downloads failed: save_failed"))
    assert run(native.save_to_downloads("/x", "x", "text/plain")) is None
    assert run(native.save_to_downloads("/x", "x", "text/plain")) is None
    assert len(calls) == 2
    assert native.native_available is True


def test_bad_result_types_fall_back():
    native = GlossarionNative()
    attach(native, ft.PagePlatform.IOS)
    recording_client(native, results={"begin_background_task": "nope", "get_initial_shared": {"x": 1}})
    assert run(native.begin_background_task("job")) == -1
    assert run(native.get_initial_shared()) == []


# --------------------------------------------------------- event payloads


def test_event_payloads_decode_with_flet_from_dict():
    native = GlossarionNative()
    share = from_dict(ShareEvent, {
        "control": native, "name": "share",
        "items": [{"id": "1", "kind": "file", "path": "/p/a.pdf", "size": 3, "unknown": True}],
    })
    assert isinstance(share.items[0], SharedItem) and share.items[0].path == "/p/a.pdf"

    fg = from_dict(ForegroundEvent, {"control": native, "name": "foreground", "type": "button", "button_id": "stop"})
    assert fg.type == gn.ForegroundEventType.BUTTON and fg.button_id == "stop" and fg.is_timeout is False

    timeout = from_dict(ForegroundEvent, {"control": native, "name": "foreground", "type": "timeout", "is_timeout": True})
    assert timeout.type == gn.ForegroundEventType.TIMEOUT and timeout.is_timeout

    bg = from_dict(BackgroundTaskEvent, {
        "control": native, "name": "background_task", "type": "expiring", "task_id": 4, "task_name": "job",
    })
    assert bg.type == gn.BackgroundTaskEventType.EXPIRING and bg.task_id == 4 and bg.name == "background_task"

    note = from_dict(NotificationEvent, {
        "control": native, "name": "notification", "notification_id": 2, "payload": "/jobs/2", "launched_app": False,
    })
    assert note.notification_id == 2 and note.action_id is None


# -------------------------------------------- cross-language contract checks


def _field_names(cls):
    return {f.name for f in fields(cls)}


def _map_keys(snippet):
    return set(re.findall(r"""['"]([a-z_]+)['"]\s*(?::|to)\s""", snippet))


def test_foreground_payload_keys_match_dataclass():
    text = DART_TASK.read_text(encoding="utf-8")
    keys = set()
    for body in re.findall(r"_send\(\{(.*?)\}\)", text, flags=re.S):
        # Keys start an entry (after the brace or a comma); this skips the
        # `cond ? 'a' : 'b'` value expressions.
        keys |= set(re.findall(r"(?:^|,)\s*'([a-z_]+)'\s*:", body))
    assert keys, "no _send payloads found"
    allowed = _field_names(ForegroundEvent) | {"starter"}
    assert keys <= allowed, keys - allowed
    types = set(re.findall(r"'type':\s*(?:isTimeout \? )?'([a-z_]+)'", text)) | {"destroyed"}
    assert types <= {t.value for t in gn.ForegroundEventType}


def test_notification_payload_keys_match_dataclass():
    kotlin = KOTLIN_PLUGIN.read_text(encoding="utf-8")
    body = re.search(r"val event = hashMapOf<String, Any\?>\((.*?)\)\n", kotlin, flags=re.S).group(1)
    kotlin_keys = _map_keys(body)
    swift = SWIFT_PLUGIN.read_text(encoding="utf-8")
    swift_block = swift[swift.index("didReceive response"):swift.index("private func deliverNotification")]
    swift_keys = set(re.findall(r'(?:"|event\[")([a-z_]+)"(?:\]|:)', swift_block))
    allowed = _field_names(NotificationEvent)
    assert {"notification_id", "action_id", "payload", "launched_app"} <= kotlin_keys
    assert kotlin_keys <= allowed, kotlin_keys - allowed
    assert {"notification_id", "launched_app", "payload", "action_id"} <= swift_keys
    assert swift_keys <= allowed, swift_keys - allowed


def test_background_task_payload_keys_match_dataclass():
    swift = SWIFT_PLUGIN.read_text(encoding="utf-8")
    keys = set()
    for block in re.findall(r'"background_task", arguments: \[(.*?)\]\)', swift, flags=re.S):
        keys |= set(re.findall(r'"([a-z_]+)":', block))
    for block in re.findall(r"let event: \[String: Any\] = \[(.*?)\]", swift, flags=re.S):
        keys |= set(re.findall(r'"([a-z_]+)":', block))
    assert {"type", "identifier", "reason", "task_id", "task_name"} <= keys
    assert keys <= _field_names(BackgroundTaskEvent), keys - _field_names(BackgroundTaskEvent)
    types = set(re.findall(r'"type": "([a-z_]+)"', swift))
    assert types == {t.value for t in gn.BackgroundTaskEventType}


def test_shared_item_keys_match_dataclass():
    allowed = _field_names(SharedItem)
    kotlin = KOTLIN_PLUGIN.read_text(encoding="utf-8")
    kotlin_keys = set(re.findall(r'item\["([a-z_]+)"\]', kotlin)) | _map_keys(
        kotlin[kotlin.index('"kind" to "text"') - 200:kotlin.index('"kind" to "text"') + 300]
    )
    swift = SWIFT_PLUGIN.read_text(encoding="utf-8")
    swift_keys = set(re.findall(r'item\["([a-z_]+)"\]', swift))
    for block in re.findall(r"(?:var|let) item: \[String: Any\] = \[(.*?)\]", swift, flags=re.S):
        swift_keys |= set(re.findall(r'"([a-z_]+)":', block))
    dart = DART_SERVICE.read_text(encoding="utf-8")
    dart_block = dart[dart.index("Map<String, dynamic>? _fromSharingIntent"):dart.index("// -------------------------------------------------------- Python -> Dart")]
    dart_keys = set(re.findall(r"'([a-z_]+)':", dart_block))
    for keys in (kotlin_keys, swift_keys, dart_keys):
        assert {"id", "kind"} <= keys
        assert keys <= allowed, keys - allowed


def test_channel_name_and_plugin_classes_consistent():
    channel = "glossarion_native/platform"
    assert f"MethodChannel('{channel}')" in DART_SERVICE.read_text(encoding="utf-8")
    assert f'CHANNEL = "{channel}"' in KOTLIN_PLUGIN.read_text(encoding="utf-8")
    assert f'channelName = "{channel}"' in SWIFT_PLUGIN.read_text(encoding="utf-8")

    pubspec = (FLUTTER_PKG / "pubspec.yaml").read_text(encoding="utf-8")
    assert re.search(r"^  flet: 1\.0\.3$", pubspec, flags=re.M)
    assert "flutter_foreground_task: ^11.0.3" in pubspec
    assert "receive_sharing_intent: ^1.9.0" in pubspec
    assert not re.search(r"^\s+flutter_local_notifications:", pubspec, flags=re.M)
    assert "package: com.glossarion.flet_glossarion_native" in pubspec
    assert pubspec.count("pluginClass: GlossarionNativePlugin") == 2
    assert "package com.glossarion.flet_glossarion_native" in KOTLIN_PLUGIN.read_text(encoding="utf-8")
    assert "class GlossarionNativePlugin" in KOTLIN_PLUGIN.read_text(encoding="utf-8")
    assert "public class GlossarionNativePlugin" in SWIFT_PLUGIN.read_text(encoding="utf-8")
    assert 'case "GlossarionNative":' in (FLUTTER_PKG / "lib" / "src" / "extension.dart").read_text(encoding="utf-8")
    assert (FLUTTER_PKG / "ios" / "flet_glossarion_native.podspec").is_file()
    assert (FLUTTER_PKG / "ios" / "flet_glossarion_native" / "Package.swift").is_file()


def test_every_dart_invoke_method_is_exposed_in_python():
    dart = DART_SERVICE.read_text(encoding="utf-8")
    block = dart[dart.index("Future<dynamic> _invokeMethod"):dart.index("dynamic _defaultResult")]
    dart_methods = set(re.findall(r"case '([a-z_]+)':", block))
    python_methods = {name for name in dir(GlossarionNative) if not name.startswith("_")}
    assert dart_methods <= python_methods, dart_methods - python_methods


def test_manifest_declares_service_trampoline_and_icon():
    root = ET.parse(MANIFEST).getroot()
    app = root.find("application")
    service = app.find("service")
    assert service.get(ANDROID_NS + "name") == "com.pravera.flutter_foreground_task.service.ForegroundService"
    assert service.get(ANDROID_NS + "foregroundServiceType") == "dataSync"
    assert service.get(ANDROID_NS + "exported") == "false"

    activity = app.find("activity")
    assert activity.get(ANDROID_NS + "name") == "com.glossarion.flet_glossarion_native.ShareReceiverActivity"
    assert activity.get(ANDROID_NS + "exported") == "true"
    actions = {}
    for intent_filter in activity.findall("intent-filter"):
        action = intent_filter.find("action").get(ANDROID_NS + "name")
        mimes = {d.get(ANDROID_NS + "mimeType") for d in intent_filter.findall("data") if d.get(ANDROID_NS + "mimeType")}
        actions[action] = mimes
    assert set(actions) == {
        "android.intent.action.VIEW", "android.intent.action.SEND", "android.intent.action.SEND_MULTIPLE",
    }
    required = {
        "application/epub+zip", "application/pdf", "text/plain", "application/zip", "application/x-cbz",
        "application/json", "text/csv", "application/x-subrip", "application/octet-stream",
    }
    for action, mimes in actions.items():
        assert required <= mimes, (action, required - mimes)

    meta = {m.get(ANDROID_NS + "name"): m.get(ANDROID_NS + "resource") for m in app.findall("meta-data")}
    icon_name = re.search(r"_kNotificationIconMetaData =\s*'([^']+)'", DART_SERVICE.read_text(encoding="utf-8")).group(1)
    assert meta[icon_name] == "@drawable/glossarion_native_ic_notification"
    assert (FLUTTER_PKG / "android" / "src" / "main" / "res" / "drawable" / "glossarion_native_ic_notification.xml").is_file()

    permissions = {p.get(ANDROID_NS + "name") for p in root.findall("uses-permission")}
    assert "android.permission.FOREGROUND_SERVICE_DATA_SYNC" in permissions


def test_trampoline_never_forwards_intent_data():
    kotlin = KOTLIN_TRAMPOLINE.read_text(encoding="utf-8")
    forward = kotlin[kotlin.index("val forwarded = Intent(ACTION_SHARED)"):]
    assert "setData(" not in forward and ".data =" not in forward and "setDataAndType(" not in forward
    assert "FLAG_GRANT_READ_URI_PERMISSION" in forward and "FLAG_ACTIVITY_SINGLE_TOP" in forward
    assert "EXTRA_GLOSSARION_URI" in forward


def test_gradle_follows_builtin_kotlin_plugin_template():
    gradle = (FLUTTER_PKG / "android" / "build.gradle.kts").read_text(encoding="utf-8")
    plugins_block = re.search(r"^plugins \{(.*?)^\}", gradle, flags=re.S | re.M).group(1)
    assert 'id("com.android.library")' in plugins_block
    # Flutter 3.44 applies kotlin-android itself; declaring it is deprecated.
    assert "kotlin" not in plugins_block
    assert 'namespace = "com.glossarion.flet_glossarion_native"' in gradle
    assert "JavaVersion.VERSION_17" in gradle and "JvmTarget.JVM_17" in gradle
    assert not (FLUTTER_PKG / "android" / "build.gradle").exists()


def test_job_service_id_matches_dart():
    dart = DART_SERVICE.read_text(encoding="utf-8")
    assert f"_kJobServiceId = {gn.JOB_SERVICE_NOTIFICATION_ID};" in dart


def test_network_security_config_allows_cleartext_to_the_reader_server_only():
    """U5 Reader: the WebView loads http://127.0.0.1:<port>/<token>/ (services/reader_server.py)."""
    import tomllib

    config = FLUTTER_PKG / "android" / "src" / "main" / "res" / "xml" / "glossarion_network_security_config.xml"
    root = ET.parse(config).getroot()
    assert root.tag == "network-security-config"
    assert root.find("base-config") is None  # every other host keeps the platform default
    permitted = set()
    for domain_config in root.findall("domain-config"):
        assert domain_config.get("cleartextTrafficPermitted") == "true"
        for domain in domain_config.findall("domain"):
            assert domain.get("includeSubdomains") == "false"
            permitted.add(domain.text.strip())
    assert permitted == {"127.0.0.1", "localhost"}

    pyproject = tomllib.loads((ROOT.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    application = pyproject["tool"]["flet"]["android"]["manifest_application"]
    assert application == {"networkSecurityConfig": "@xml/" + config.stem}

    server = (ROOT.parents[1] / "app" / "glossarion_mobile" / "services" / "reader_server.py").read_text(encoding="utf-8")
    assert 'host: str = "127.0.0.1"' in server


def test_verify_apk_checks_the_manifest_application_attributes():
    sys.path.insert(0, str(ROOT.parents[1] / "ci"))
    try:
        import verify_apk
    finally:
        sys.path.pop(0)
    names = verify_apk.manifest_application_expectations(ROOT.parents[1] / "pyproject.toml")
    assert names == ["networkSecurityConfig"]
    merged = (
        "N: android=http://schemas.android.com/apk/res/android\n"
        "  E: manifest (line=2)\n"
        "    E: application (line=12)\n"
        "      A: http://schemas.android.com/apk/res/android:label(0x01010001)=\"Glossarion\" (Raw: \"Glossarion\")\n"
        "      A: http://schemas.android.com/apk/res/android:networkSecurityConfig(0x01010527)=@0x7f150001\n"
    )
    assert verify_apk.missing_application_attributes(verify_apk.parse_xmltree(merged), names) == []
    bare = merged.replace("networkSecurityConfig", "icon")
    assert verify_apk.missing_application_attributes(verify_apk.parse_xmltree(bare), names) == ["networkSecurityConfig"]
