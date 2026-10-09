# flet-glossarion-native

In-repo Flet 1.0.3 user extension that gives the Glossarion mobile app the
native pieces Flet does not ship:

- **Android**: a `dataSync` foreground service that keeps the Python job
  thread alive with the screen off (wraps `flutter_foreground_task` 11.0.3);
  "Open with" and "Share" intake through an exported trampoline activity;
  local notifications; "Save to Downloads" through MediaStore.
- **iOS**: "Open in"/"Copy to Glossarion" file intake, local notifications,
  `beginBackgroundTask` and the iOS 26 `BGContinuedProcessingTask`.
- **Both (U10)**: document destinations: a folder or file the user picks once
  in the system picker (Android Storage Access Framework, iOS Files), written to
  later without asking again. This is how finished books reach the user's own
  cloud app; Glossarion makes no network request for it.

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
    DocumentDestinations.kt          SAF pickers, persisted grants, mode-chain writes (U10)
  ios/flet_glossarion_native/Package.swift   SwiftPM (Flutter 3.44 default)
  ios/flet_glossarion_native.podspec         CocoaPods fallback, same sources
  ios/flet_glossarion_native/Sources/flet_glossarion_native/GlossarionNativePlugin.swift
  ios/flet_glossarion_native/Sources/flet_glossarion_native/DocumentDestinations.swift
src/flet_glossarion_native/documents.py       document-destination contract (refs, results, error codes)
src/flet_glossarion_native/documents_fake.py  in-memory provider answering like DocumentDestinations.kt (tests)
tests/test_native_defaults.py        host tests (defaults, marshalling, cross-language contract)
tests/test_document_destinations.py  U10 API, fake-provider semantics, Dart/Kotlin/Swift static checks
```

## Python API

```python
from flet_glossarion_native import GlossarionNative, NotificationButton

native = GlossarionNative(
    on_share=on_share,                  # ShareEvent(items: list[SharedItem])
    on_foreground=on_foreground,        # ForegroundEvent(type, button_id, is_timeout)
    on_background_task=on_background,   # BackgroundTaskEvent(type, task_id, task_name, identifier, reason)
    on_notification=on_notification,   # NotificationEvent(notification_id, action_id, payload, launched_app)
    on_document=on_document,           # DocumentEvent(type, op_id, written, total, kind, status, result)
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
| `save_to_downloads(path, display_name, mime_type, subdir="Glossarion", replace_uri=None)` | MediaStore Downloads; `replace_uri` overwrites our own entry in place | `None` | `None` |
| `save_to_downloads_entry(...)` (same arguments) | the same call with `report_name`: `{"uri", "name"}`, the name MediaStore gave the entry (`name (1).ext` when taken) | `None` | `None` |
| `pick_folder(*, initial, op_id)` | `ACTION_OPEN_DOCUMENT_TREE` + persisted grant | Files folder picker + minimal bookmark | `{"ok": False, "error": "unavailable"}` |
| `pick_save_location(name, mime_type, source_path=None, *, initial, mode_chain, op_id)` | `ACTION_CREATE_DOCUMENT` + grant (+ first write) | export picker moves a copy, bookmark | same |
| `pick_document(mime_types=None, *, initial, op_id)` | `ACTION_OPEN_DOCUMENT` + grant | open picker + bookmark | same |
| `list_children(folder, *, names=None)`, `create_file(folder, name, mime_type, *, on_exists="rename")`, `create_folder(parent, name, *, on_exists="adopt")` | `DocumentsContract` under the tree grant | coordinated `FileManager` calls | same |
| `write_file(target_or_doc, source_path, *, name, mime_type, on_exists, mode_chain=("wt","rwt","w"), verify=True, op_id, timeout)` | mode chain + read-back length | staged copy swapped in (`replaceItemAt`) | same |
| `rename_document(document, name)` | `DocumentsContract.renameDocument` (keep the returned ref: the URI may change; `unavailable` without rename support, `exists` when the name is taken) | coordinated move inside the picked folder (`unavailable` for a single exported file) | same |
| `stat(document)`, `delete(document)`, `query_root(target)` | query / `deleteDocument` / root query + grant check | resource values / `.forDeleting` / reachability | same |
| `release(target)`, `list_grants()` | `releasePersistableUriPermission`, `persistedUriPermissions` | `True` / `[]` (bookmarks are app data) | `False` / `[]` |
| `cancel_document_op(op_id)`, `take_document_results()` | stop a write; late picker answers | same | `False` / `[]` |

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
- **Stop with task.** The manifest's `android:stopWithTask="true"` is the
  plan's default: swiping the app away stops the service, and jobs stay
  resumable. `ForegroundTaskOptions.stopWithTask` in `native_service.dart`
  must stay unset (null). It is not the same switch: flutter_foreground_task
  11.x then stops the service whenever no activity of the app is resumed
  (`TrackVisibilityUtils`), i.e. on Home or as soon as the sign-in Custom Tab
  opens, which leaves jobs and the OAuth loopback listener unprotected (the app
  is cached and frozen, and the browser hangs on `http://localhost/...`).
  With the option unset the library falls back to the manifest flag
  (`onTaskRemoved` -> `stopSelf`), and the next start clears a value saved by
  an older build.
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

## Document destinations (U10)

The user picks a place once in the system picker; Glossarion keeps write
access and later writes finished books there. The user's cloud app (Drive,
Nextcloud, iCloud Drive, or anything else with a document provider, including
RSAF for rclone remotes) uploads them with the user's own account. There is no
developer account, OAuth client or network code. `documents.py` holds the
contract shared by Python, Kotlin, Swift and the fake:

- **References** are plain JSON dicts the app stores and passes back:
  `platform`, `kind` (`folder` / `file`), `id` (FNV-1a/64 of the tree URI or
  folder path: key cloud records by the destination's `id`), Android `uri`
  (tree) + `document`, iOS `bookmark` + `root` + `path`, `name`, `size`,
  `mtime`, `provider`, `provider_label`, `can_write`, `can_create`,
  `persisted`, `own_folder` (Glossarion's own storage: the app refuses it).
- **Results** are `{"ok", "error", "message", "scope", "retryable", ...}`.
  Error codes: `cancelled`, `permission_lost`, `missing`, `unsupported_mode`,
  `provider_error`, `no_space`, `source_missing`, `source_changed`,
  `read_only`, `size_mismatch`, `exists`, `busy`, `unavailable`, `timeout`,
  `bad_args`. `scope` is `target` (re-link the destination), `document` (only
  this file) or `source`.

Android (`DocumentDestinations.kt`):

- Pickers persist only the flags the picker granted (`data.flags &
  (READ|WRITE)`; asking for more throws). A pick survives activity recreation:
  the pending call is kept in the engine-scoped plugin and the
  `ActivityResultListener` follows every attach/detach. When the whole process
  was recreated, the answer is still persisted and arrives as a
  `document` event `pick_result` (and through `take_document_results()`); a
  picker left open across a process death is reported as `cancelled` on the
  next attach.
- `write_file` streams the caller's private snapshot in 1 MiB chunks through
  `openFileDescriptor` with the mode chain `wt` -> `rwt` -> `w`. A
  FileNotFoundException is not "deleted": Drive throws it for an unsupported
  mode and Nextcloud while offline. A document is `missing` only when the tree
  root answers and the document does not (`proven: True`); otherwise the
  answer is `provider_error` (retry). The non-truncating modes `w` / `rw` are
  used only when the new file is not shorter than the cloud copy, the tail is
  cut with `ftruncate` when the descriptor is a real file, and the written
  length is read back with `'r'` + `statSize` (never `COLUMN_SIZE`). A stale
  tail answers `size_mismatch` with `needs_replace`; when no mode fits the
  answer is `unsupported_mode` with `needs_replace`, and the caller creates a
  new file and deletes the old one (then `rename_document` gives the new copy
  the old name back where the provider can rename). On failure the descriptor is closed with
  `closeWithError` (only providers with an `OnCloseListener` drop the partial
  file) and `remote_damaged` says whether the cloud copy may now be cut short.
- `create_file` looks for same-name items first (`on_exists`: `rename`,
  `adopt`, `fail`). A create that throws is retried once after 1.5 s, and a
  file that appeared anyway (Drive is eventually consistent) is adopted. A
  provider's `UnsupportedOperationException` ("Create not supported", AOSP's
  default) - and for `create_folder` an `IllegalArgumentException` refusing the
  directory type - answers `read_only` with `create_unsupported: true` (not
  retried): the cloud sync then lays the books out flat in the picked folder.
- A save location (`pick_save_location`) inside Glossarion's own folder
  (`own_folder`) is not written: the empty document the Save dialog made is
  deleted and the answer carries `own_folder: true`, so the app refuses it with
  nothing left behind.
- `save_to_downloads(..., replace_uri=)` overwrites Glossarion's own
  MediaStore entry in place and never changes `DISPLAY_NAME`, so a backup app
  watching Download/Glossarion sees an update instead of a new file.
- The manifest declares `<queries>` for `android.content.action.DOCUMENTS_PROVIDER`
  (package visibility, not a permission). No storage permission is added.
  `get_platform_info()` reports `persisted_grant_limit` (128 before Android
  11, 512 from 11; the oldest grants are trimmed silently).

iOS (`DocumentDestinations.swift`):

- Folder picks use the iOS 13 initializers (`documentTypes: ["public.folder"]`),
  so no iOS 14 framework is linked into the 13.0 target; `forExporting:` sits
  behind `#available(iOS 14.0, *)` with the `.moveToService` fallback.
- References are minimal bookmarks. Items inside a picked folder carry the
  folder's bookmark (`root`) and their relative `path`. A child bookmark that
  now resolves outside the folder or into a `.Trash` folder (iCloud "Recently
  Deleted") counts as `missing`. A stale bookmark is refreshed and returned in
  the new reference.
- Writes copy the snapshot into the destination volume's replacement
  directory (progress, cancel), then swap it in under `NSFileCoordinator`
  (`.forReplacing` + `replaceItemAt` / `moveItem`). The cloud file never holds
  half-written bytes, and the old version does not have to be downloaded
  first. Without a replacement directory the write is in place (`mode:
  "in_place"`, warning `non_atomic`).
- Glossarion only writes while it runs: the app wraps a drain in
  `begin_background_task` / continued processing. The File Provider extension
  then uploads on its own schedule.

The app cannot see whether the cloud app finished uploading (quota full,
signed out, Wi-Fi only): it can only report that the file was handed over.

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

**U10 document destinations (2026-10-09):**

- **Kotlin.** `DocumentDestinations.kt` + the updated plugin compile with
  kotlinc 2.2.20 against `android-36` `android.jar`, `androidx.core:core:1.15.0`
  and the Flutter 3.44.8 embedding (stub `R`), without errors or warnings
  (the U0 toolchain in the session scratchpad).
- **Dart.** `dart analyze lib` (Flutter 3.44.8): no new issue (the
  `dart:io` unnecessary-import info is older).
- **Swift.** `DocumentDestinations.swift` is **not** type-checked (no Xcode);
  `tests/test_document_destinations.py` checks it statically (balance,
  method names, error codes, event and reference keys, no iOS 14-only API
  outside `#available`). The iOS CI build is its first compile.
- **Python.** The same test file runs the Android semantics through
  `documents_fake.py`. Provider behaviour needs the device checklist in the
  U10 report.

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
