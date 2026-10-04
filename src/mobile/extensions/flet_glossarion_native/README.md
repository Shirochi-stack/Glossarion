# flet-glossarion-native

In-repo Flet 1.0.3 user extension that gives the Glossarion mobile app the
native pieces Flet does not ship:

- **Android**: a `dataSync` foreground service that keeps the Python job
  thread alive with the screen off (wraps `flutter_foreground_task` 11.0.3);
  "Open with" and "Share" intake through an exported trampoline activity;
  local notifications; "Save to Downloads" through MediaStore.
- **iOS**: "Open in"/"Copy to Glossarion" file intake, local notifications,
  `beginBackgroundTask` and the iOS 26 `BGContinuedProcessingTask`.

It is installed by `flet build` through `[tool.flet].dev_packages` in
`src/mobile/pyproject.toml` (`extensions/flet_glossarion_native`, resolved
relative to `src/mobile`). Under `flet run` on a desktop, on the web, or in
the Flet companion app there is no Dart service, and every method returns a
safe default.

## Layout

The layout follows Flet 1.0.3's extension template and its first-party
extensions (`flet-secure-storage`, `flet-permission-handler`):

```
pyproject.toml                       wheel = Python package + flutter/<pkg> package data
src/flet_glossarion_native/          GlossarionNative(ft.Service) + dataclass types
src/flutter/flet_glossarion_native/  Dart package / Flutter plugin
  pubspec.yaml                       flet 1.0.3 (lockstep), flutter_foreground_task ^11.0.3,
                                     receive_sharing_intent ^1.9.0
  lib/flet_glossarion_native.dart    exports Extension (Flet's generated main imports it)
  lib/src/extension.dart             createService("GlossarionNative"), FGS port init
  lib/src/native_service.dart        every invoke method + events
  lib/src/task_handler.dart          @pragma('vm:entry-point') glossarionStartCallback + TaskHandler
  android/build.gradle.kts           Flutter 3.44 plugin template: AGP library, built-in Kotlin, JDK 17
  android/src/main/AndroidManifest.xml  FGS <service>, ShareReceiverActivity, icon meta-data
  android/src/main/kotlin/com/glossarion/flet_glossarion_native/
    GlossarionNativePlugin.kt        MethodChannel glossarion_native/platform
    ShareReceiverActivity.kt         VIEW/SEND/SEND_MULTIPLE trampoline
  ios/flet_glossarion_native/Package.swift   SwiftPM (Flutter 3.44 default)
  ios/flet_glossarion_native.podspec         CocoaPods fallback, same sources
  ios/flet_glossarion_native/Sources/flet_glossarion_native/GlossarionNativePlugin.swift
tests/test_native_defaults.py        host tests (defaults, marshalling, cross-language contract)
```

## Python API

```python
from flet_glossarion_native import GlossarionNative, NotificationButton

native = GlossarionNative(
    on_share=on_share,                  # ShareEvent(items: list[SharedItem])
    on_foreground=on_foreground,        # ForegroundEvent(type, button_id, is_timeout)
    on_background_task=on_background,   # BackgroundTaskEvent(type, task_id, task_name, identifier, reason)
    on_notification=on_notification,   # NotificationEvent(notification_id, action_id, payload, launched_app)
)
```

Create the service once inside `main(page)` with the handlers passed to the
constructor, so they are registered before the Dart service starts.

| Method | Android | iOS | Default elsewhere |
|---|---|---|---|
| `get_platform_info()` | sdk_int, notifications/battery state, `fgs_data_sync_time_limited`, paths | system_version, notification status, `continued_processing` | `{"native": False, "unavailable_reason": ...}` |
| `get_initial_shared()` / `clear_shared(delete_files=False)` | yes | yes | `[]` / no-op |
| `init_notifications(channels=None, request_permission=False)` | creates channels (default jobs.progress / jobs.done / jobs.action) | optional permission prompt | `False` |
| `show_notification(id, title, body, *, channel_id, payload, ongoing, progress, indeterminate, actions, auto_cancel, silent)` | NotificationCompat | UNUserNotificationCenter (progress shown as subtitle) | `False` |
| `cancel_notification(id)`, `get_launch_notification()` | yes | yes | no-op / `None` |
| `start_job_service(title, text, *, buttons, wake_lock, wifi_lock, ...)`, `update_job_service`, `stop_job_service`, `is_job_service_running` | flutter_foreground_task | `False` (use the iOS calls below) | `False` / no-op |
| `begin_background_task(name, *, expiration_title, expiration_body, expiration_payload)`, `end_background_task`, `background_time_remaining` | `-1` / no-op / `None` | UIApplication background task | same |
| `start_continued_processing(identifier=None, title, subtitle, *, strategy="fail", expiration_*)`, `update_continued_processing`, `finish_continued_processing` | `False` | iOS 26 BGContinuedProcessingTask | `False` |
| `save_to_downloads(path, display_name, mime_type, subdir="Glossarion")` | MediaStore Downloads | `None` | `None` |

All methods are coroutines and must run on Flet's event loop. From a worker
thread (the job thread), use
`asyncio.run_coroutine_threadsafe(native.update_job_service(text=...), loop)`.

Methods never raise. A failure is logged on the `flet_glossarion_native`
logger and the default is returned. If the client has no Dart service (Flet
1.0.3 then answers "Timeout waiting for invoke method listener ..."), the
service marks itself unavailable for the session. Set
`GLOSSARION_NATIVE_DISABLE=1` to force the defaults in a built app.

Reserved notification ids: **41100** is the Android job-service notification
(`JOB_SERVICE_NOTIFICATION_ID`); **41101** is the iOS "background time
expired" notice posted by the native side.

### Shared items

Files are copied to app-private storage before Python sees them: Android
`cacheDir/shared/<batch>/`, iOS `tmp/shared/<batch>/`. Items that arrived
before the Dart service attached (cold start) come from `get_initial_shared()`;
later ones fire `on_share`. Items stay queued until `clear_shared()`, and each
has a unique `id`. `SharedItem.source` is `view`, `send`, `send_multiple`,
`open_url`, `launch` or `share`. An item whose copy failed has `error` set and
`path=None`.

## Android

### Foreground service

`AndroidManifest.xml` declares flutter_foreground_task's service, which the
library itself does not declare:

```xml
<service android:name="com.pravera.flutter_foreground_task.service.ForegroundService"
    android:exported="false" android:foregroundServiceType="dataSync" android:stopWithTask="true" />
```

- **Buttons.** Notification buttons (e.g. `stop`, `open`) reach Python as
  `on_foreground(type="button", button_id=...)`. The task isolate
  (`GlossarionTaskHandler`) does no work. It only forwards events to the main
  isolate.
- **Android 15+ timeout.** Android 15 ends `dataSync` services after 6 h in
  any 24 h (`onTimeout`). That arrives as `type="timeout"`, `is_timeout=True`.
  Stop gracefully and post a "tap to resume" notification.
- **Stop with task.** `stopWithTask="true"` is the plan's default (an open
  question): swiping the app away stops the service, and jobs stay resumable.
  If that changes, flip the manifest flag and the
  `ForegroundTaskOptions.stopWithTask` value in `native_service.dart` together.
- **No auto-restart.** `allowAutoRestart` is false, because a restarted
  service would have no Python job behind it.
- **Icon.** The service and `show_notification` use
  `@drawable/glossarion_native_ic_notification` (meta-data
  `com.glossarion.native.notification_icon`).

### Open with / Share

`ShareReceiverActivity` is the exported entry point for:

- `ACTION_VIEW`: content/file URIs.
- `ACTION_SEND` and `ACTION_SEND_MULTIPLE`.

It accepts these types: epub, pdf, txt, md, html/xhtml, zip, cbz, json, csv,
srt, vtt, xliff and octet-stream. SEND and SEND_MULTIPLE also accept `image/*`
for manga pages.

Flutter pushes `intent.getData().toString()` as a route whenever MainActivity
receives an intent with data (`FlutterActivityAndFragmentDelegate`). The
trampoline therefore never forwards data. Instead it:

- moves every URI into `EXTRA_GLOSSARION_URI` / `EXTRA_GLOSSARION_URIS`;
- copies shared text and subject into their own extras;
- keeps the URIs in `ClipData`, so `FLAG_GRANT_READ_URI_PERMISSION` carries
  the read grant;
- starts the launch activity with `FLAG_ACTIVITY_SINGLE_TOP` and the custom
  action `com.glossarion.flet_glossarion_native.action.SHARED`.

MainActivity is `singleTop`, so a running app gets `onNewIntent`.
`GlossarionNativePlugin` copies the streams on a background executor and
emits `share`.

The custom action also keeps `receive_sharing_intent` out of these intents,
for three reasons:

- With a null data URI it would emit an item with a null path, which breaks
  its Dart decoder.
- It copies SEND streams on the UI thread, which risks an ANR on large PDFs.
- It would emit duplicates.

`receive_sharing_intent` stays wired in Dart for SEND intents that reach
MainActivity directly and for an optional iOS Share Extension (U9). Its
`url` items for `glossarion:` deep links are dropped, because Flet routes
deep links itself.

### Save to Downloads

- **API 29+.** Writes through `MediaStore.Downloads` with `RELATIVE_PATH`
  `Download/<subdir>`; no permission is needed.
- **API 26-28.** Writes directly to the file and runs the media scanner, but
  only when `WRITE_EXTERNAL_STORAGE` is granted.
  `src/mobile/pyproject.toml` currently removes that permission, so on
  26-28 the method returns `None` and the UI should fall back to Share /
  `FilePicker.save_file`. To enable the legacy path, declare it as
  `"android.permission.WRITE_EXTERNAL_STORAGE" = { maxSdkVersion = "28" }` and
  request it at runtime.

## Notifications and desugaring

**Decision: notifications are native (NotificationCompat on Android,
UNUserNotificationCenter on iOS) behind the planned Python/Dart API.
`flutter_local_notifications` is not a dependency.**

`flutter_local_notifications` 22.x sets `coreLibraryDesugaringEnabled true`
in its own module. AGP's `CheckAarMetadataTask` then fails the app build
unless `:app` enables core-library desugaring too. The Flet 1.0.3 app
template (`templates/build/.../android/app/build.gradle.kts`) does not, and
offers no pyproject hook for Gradle DSL. A plugin cannot fix this from its
own Gradle script:

- **`:app` is fully configured before any plugin script runs.** The
  template's root `build.gradle.kts` runs
  `subprojects { project.evaluationDependsOn(":app") }`. AGP has already
  created `:app`'s variants and registered (or skipped) the
  `L8DexDesugarLibTask` (`TaskManager`: `if (dexing.shouldPackageDesugarLibDex)`).
- **Flipping the flag late gives an inconsistent build.**
  `compileOptions.isCoreLibraryDesugaringEnabled` is a plain var that is not
  locked. Setting it from a plugin's `afterEvaluate`/`projectsEvaluated` hook
  would satisfy the lazily configured `CheckAarMetadataTask`. But
  dexing would then rewrite `java.time` calls to `j$.*` classes, and no L8
  task packages those classes. That is a runtime crash risk.
- **The remaining options sit outside the extension:**
  - a `[tool.flet.template]` fork adding
    `isCoreLibraryDesugaringEnabled = true` plus
    `coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.1.5")` to
    `app/build.gradle.kts`;
  - a Gradle init script in CI that does the same before projects evaluate.

The native implementation needs neither, and it keeps the same Python API:
`init_notifications` / `show_notification` / `cancel_notification` /
`on_notification` / `get_launch_notification`. If desugaring becomes
available (template fork, init script, or a future Flet hook), switching back
to `flutter_local_notifications` is a Dart-only change in
`native_service.dart`.

On iOS the Flet template's AppDelegate never sets
`UNUserNotificationCenter.delegate`, which `flutter_local_notifications` and
`flutter_foreground_task` both ask the app to do. Without it, taps and
foreground presentation are lost. The plugin sets the delegate to the
`FlutterAppDelegate` when it is unset, and FlutterAppDelegate then forwards
to every plugin.

## iOS

- **Open in.** File URLs are handled in `application(_:open:options:)` and in
  the UIScene callbacks `scene(_:willConnectTo:options:)` and
  `scene(_:openURLContexts:)`. Flutter 3.44 auto-migrates the unmodified Flet
  AppDelegate to UIScene, so both paths are implemented. Each file is copied
  with security-scoped access and `NSFileCoordinator` into
  `tmp/shared/<batch>/`. The handler returns `true`, so Flutter never routes
  a file URL.
- **Cold-start deep links.** `receive_sharing_intent` 1.9.0 returns `YES`
  from `scene(_:willConnectTo:options:)` for every connection. Flutter then
  skips its own deep-link handling for cold-start URLs, so
  `glossarion://app/...` opened while the app is not running would be lost.
  This plugin registers first (alphabetical order) and records such a link as
  a `SharedItem(kind="url", source="launch")`. The Python router should
  accept it after whitelisting, and de-duplicate it against `page.route`.
  Warm deep links are unaffected.
- **`begin_background_task`.** On expiry the plugin:
  1. posts the optional local notice;
  2. fires `on_background_task(type="expiring")`;
  3. ends the task itself.

  iOS terminates apps that do not end the task.
- **`start_continued_processing`.** This must run in direct response to a
  user tap. It works as follows:
  1. It registers the identifier on demand and submits
     `BGContinuedProcessingTaskRequest` (strategy `fail` by default).
  2. Identifiers must start with `<bundle id>.job.` and be unique per job,
     because iOS kills an app that registers an identifier twice. With
     `identifier=None` the plugin generates `com.glossarion.app.job.<12 hex>`.
     `pyproject.toml` already lists `com.glossarion.app.job.*` in
     `BGTaskSchedulerPermittedIdentifiers`.
  3. Progress from `update_continued_processing` feeds the Live Activity.
  4. When the user cancels or the system expires the task, the plugin fires
     `type="continued_expired"`; the task has already been completed
     unsuccessfully.

  The iOS 26 code is compiled only with Swift 6.2+ (Xcode 26):
  `#if compiler(>=6.2)` + `#available(iOS 26.0, *)`. Older toolchains build
  and report `continued_processing: False`.
- **Packaging.** SwiftPM (`ios/flet_glossarion_native/Package.swift`, which
  depends on `../FlutterFramework` like receive_sharing_intent 1.9.0 and
  flutter_foreground_task 11.0.3). `flet_glossarion_native.podspec` is the
  CocoaPods fallback.

## Verification status

**Checked locally (2026-10-05):**

- **Kotlin.** `GlossarionNativePlugin.kt` and `ShareReceiverActivity.kt`
  compile with kotlinc 2.2.20 against `android-36` `android.jar`,
  `androidx.core:core:1.15.0` and the Flutter 3.44.8 embedding
  (`flutter_embedding_release-1.0.0-0cd610717b...`), using a stub `R` class.
- **Dart.** `dart analyze` (Flutter 3.44.8, Dart 3.12.2) reports no errors.
  Resolved versions: flet 1.0.3, flutter_foreground_task 11.0.3,
  receive_sharing_intent 1.9.0.
- **App-level resolution.** `flutter pub get` succeeds for an app pubspec
  equal to the released Flet 1.0.3 build template plus this package,
  flet_secure_storage and flet_permission_handler. Flutter registers the
  plugin for Android and iOS (`GlossarionNativePlugin`).
- **Swift.** The file parses cleanly (tree-sitter). It has **not** been
  type-checked, which needs Xcode.
- **Wheel.** `pip wheel` contains every Dart/Kotlin/Swift/Gradle/podspec file
  under `flutter/flet_glossarion_native/`.
- **Python.** `tests/test_native_defaults.py` passes (89 tests).

**Only CI or a device can prove:**

- The Gradle build inside the generated Flet app (manifest merge, built-in
  Kotlin, compileSdk 37 of receive_sharing_intent).
- The iOS Swift compile.
- The U0 device spike:
  - FGS keeps the Python thread alive 20+ min with the screen off.
  - The notification Stop button reaches Python.
  - Open-with never changes `page.route`.
  - The Android 15 timeout event fires.
  - BGContinuedProcessingTask submits with a `com.glossarion.app.job.*` id
    on iOS 26.
