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
- **Both (hotfix 3)**: hidden in-app browser pages (Android `WebView`, iOS
  `WKWebView`) the extension creates and owns outside the Flutter widget tree,
  for the routes that drive a web page (authchatgpt/, authnd/, authds/,
  search/gemini). Screen changes never rebuild them.

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
  pubspec.yaml                       flet 1.0.3 (lockstep), flutter_foreground_task ^11.0.3
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
    HiddenWebPages.kt                hidden in-app browser pages (hotfix 3)
  ios/flet_glossarion_native/Package.swift   SwiftPM (Flutter 3.44 default)
  ios/flet_glossarion_native.podspec         CocoaPods fallback, same sources
  ios/flet_glossarion_native/Sources/flet_glossarion_native/GlossarionNativePlugin.swift
  ios/flet_glossarion_native/Sources/flet_glossarion_native/DocumentDestinations.swift
  ios/flet_glossarion_native/Sources/flet_glossarion_native/HiddenWebPages.swift
src/flet_glossarion_native/documents.py       document-destination contract (refs, results, error codes)
src/flet_glossarion_native/documents_fake.py  in-memory provider answering like DocumentDestinations.kt (tests)
src/flet_glossarion_native/web_pages.py       in-app browser page contract (methods, events, error codes)
src/flet_glossarion_native/web_pages_fake.py  pages answering like HiddenWebPages.kt (host tests)
tests/test_native_defaults.py        host tests (defaults, marshalling, cross-language contract)
tests/test_document_destinations.py  U10 API, fake-provider semantics, Dart/Kotlin/Swift static checks
tests/test_web_pages.py              hotfix 3 page API, fake semantics, Dart/Kotlin/Swift static checks
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
    on_web=on_web,                     # WebPageEvent(type, page_id, url, ok, error, code, main_frame, crashed)
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
| `move_task_to_back()` (hotfix 4) | `Activity.moveTaskToBack(true)`: the task goes to the background like Home (the app calls it for back on the chat while a job runs, so the Activity, the engine and the Python session stay); `True` when it moved | `False` (no system back) | `False` |
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
| `web_open(page_id, *, width, height, prevent_links=())`, `web_load(page_id, url)`, `web_eval(page_id, script, *, timeout=30)`, `web_title(page_id)`, `web_close(page_id)`, `web_reset()`, `web_status()` | hidden `android.webkit.WebView`s (`HiddenWebPages.kt`) | hidden `WKWebView`s (`HiddenWebPages.swift`) | `{"ok": False, "error": "unavailable"}` |

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

### In-app browser routes in the background (hotfix 1)

This section is about the flet-webview fallback (`services/webview_bridge.py`,
used when the native pages below are missing or switched off with
`GLOSSARION_NATIVE_WEBVIEW=0`). authchatgpt/, authnd/, authds/ and
search/gemini then run in hidden flet-webview pages. Under the foreground
service they need no native code:

- Flutter builds widgets only in frames, and draws none while the app is
  hidden or paused (`SchedulerBinding.scheduleFrame` returns at once). A
  WebView mounted then never runs flet-webview's `initState` (native WebView,
  first load, `load_request` / `run_javascript` listener). So the bridge keeps
  the WebViews built while the app was on screen for the next request while
  jobs run, and a request never mounts a new one in the background.
- A built WebView keeps working in the background. Method calls and console
  and page events travel over the Flet connection, not through frames.
  webview_flutter_android never calls `onPause` or `pauseTimers`. AwContents'
  default renderer policy is `RENDERER_PRIORITY_IMPORTANT`, not waived when
  not visible, so the renderer process keeps the app's FGS importance. Adding
  `onResume`, `resumeTimers` or `setRendererPriorityPolicy(IMPORTANT, false)`
  here would change nothing.
- Chromium still treats a page as hidden while its window is: it aligns timers
  to once a second and stops `requestAnimationFrame`, and Blink freezes the
  hidden page a minute or five later (seen on the emulator in hotfix 4: a page
  load then fails after 90 s). The native pages below report a visible window
  instead; on this fallback the routes' reply polls are bounded
  (`browser_driver.ReplyWatch`), so a frozen page fails its request with the
  reason.

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

A SEND / SEND_MULTIPLE / VIEW intent that reaches MainActivity directly (an
explicit component such as `am start -n com.glossarion.app/.MainActivity`,
never the chooser) is imported by `GlossarionNativePlugin` the same way:
`ShareReceiverActivity.readShare` collects its URIs and text, and the copy runs
on the executor. A URI the app cannot read becomes an item with `error`
("Could not open the shared file: ..." in the app), never a crash.

`receive_sharing_intent` is no longer a dependency. Its Android plugin
handled every SEND intent that reached MainActivity on the UI thread
(`FileDirectory.getAbsolutePath` -> `getDataColumn` -> `openInputStream`) and
let a `SecurityException` for a URI without a read grant kill the app. It
offered nothing else: the iOS Share Extension is not shipped (see
`src/mobile/README.md`), and iOS "Open in" is handled by this plugin.

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

## In-app browser pages (hotfix 3)

The routes that drive a web page (authchatgpt/ chatgpt.com web chat, authnd/,
authds/, search/gemini) used hidden flet-webview controls in `page.overlay`.
Flet 1.0.3 draws the overlay only in the top View, so every screen push / pop
moved the WebView into another widget tree, and Flutter built a new native
WebView that reloaded without the page's hook: an authchatgpt reply in
progress was lost and sent again. The extension now creates and owns the
pages itself, outside the Flutter widget tree, and Python drives them through
the existing service channel. `web_pages.py` holds the contract:

- **Handshake.** `get_platform_info()["web_pages"]` is
  `{"version": 1, "max_pages": 8, "provider": ...}`;
  `web_pages.capability(info)` reads it. Android leaves it out when API 26+
  reports no WebView provider. The app prefers these pages when the key is
  there; otherwise it falls back to flet-webview unchanged. Desktop uses
  neither.
- **Methods.** `web_open` (Python makes the `page_id`; viewport in dp / pt;
  `prevent_links`), `web_load` (http / https only), `web_eval` (one string
  back: Python wraps every expression in an IIFE that returns
  `JSON.stringify({ok, value})`, so there is no console channel and no
  chunking), `web_title`, `web_close`, `web_reset` (every page of this plugin
  instance; the app sends it at start-up and at shutdown), `web_status`.
  Every answer is `{"ok", "error", "message", ...}` with the codes
  `unavailable`, `bad_args`, `exists`, `too_many`, `no_page`, `gone`,
  `closed`, `native_error` and Python's own `timeout`. Messages never carry
  page content; exceptions are reported by type name (Flet's `TimeoutError`
  text embeds the call arguments).
- **Events.** `on_web` gets `started`, `finished` (`ok`, `error`), `url`
  (redirects, commits, pushState), `error` (`code`, `main_frame`; Android
  forwards at most 8 sub-resource errors per document) and `gone` (the
  renderer stopped; `crashed`: Android's `didCrash()`, null on iOS, which gives
  no reason). Flet only sends them when `on_web` has a
  handler at construction. Console output, scripts, results and page content
  are never sent or logged.
- **Threading.** Kotlin and Swift run every page call, WebView callback and
  `channel.invokeMethod` on the main thread. Each channel result is answered
  exactly once, and a pending `web_eval` answers `closed` / `gone` when its page
  is destroyed. The page registry belongs to the plugin instance:
  flutter_foreground_task's headless engine gets its own instance, which never
  opens a page, so a job ending never touches the app's pages.

Android (`HiddenWebPages.kt`):

- One 1 x 1 px `FrameLayout` (application context, `clipChildren`, not
  focusable, `FOCUS_BLOCK_DESCENDANTS`, hidden from accessibility and
  autofill) sits at index 0 of the Activity's `android.R.id.content`, behind the
  opaque FlutterView, which takes every touch. Each WebView is laid out at its
  viewport (dp x density) at (0,0). The 1 px clip keeps it drawn, alpha stays
  1, and it is never `GONE` / `INVISIBLE`, so Chromium treats the page as
  visible whenever the app is.
- FlutterActivity attaches plugins before `setContentView`, and a view added to
  the content view then would be dropped. So the ActivityAware callbacks only
  store the Activity; the attach is posted, finds the content view with
  `peekDecorView()` and runs only while a page exists. ViewRootImpl lays
  nothing out while the Activity is stopped, so every WebView is measured and
  laid out by hand after it is added or re-parented: a page opened in the
  background gets its viewport too.
- Settings follow webview_flutter_android's defaults plus JavaScript (DOM
  storage, overview mode, wide viewport, `setSupportMultipleWindows(true)` with
  no `onCreateWindow`, so `window.open` opens nothing). File and content access
  are off. The system user agent, mixed content, Safe Browsing and third-party
  cookies keep their defaults; the process-wide `CookieManager` is shared with
  flet-webview, so logins carry over. `setRendererPriorityPolicy(IMPORTANT,
  false)` on API 26+. Never called: `onPause`, `pauseTimers`,
  `setWebContentsDebuggingEnabled`, `addJavascriptInterface`.
- Console messages are swallowed, JS dialogs cancelled (`beforeunload`
  confirmed), permission and geolocation requests denied, and `intent:` /
  `market:` / `file:` / `content:` navigations refused (`prevent_links`).
- `onRenderProcessGone` returns `true` (false would kill the app) and loses the
  page at once: event `gone`, pending evals answer `gone`, the WebView is
  removed and destroyed and never called again. The id answers `gone` until
  `web_close`; authchatgpt sends its message again on a new page. Cookies are
  flushed once when the last page closes (`web_reset` too: once for all its
  pages, since `CookieManager.flush()` blocks the main thread).
- **Background (hotfix 4).** In the background the Activity window is not
  visible. AwContents shows a page only while it is attached to a visible
  window (`BrowserViewRenderer::IsClientVisible`), so hotfix 3's pages turned
  hidden, and Blink freezes a hidden page a minute (2025+ WebView) to five
  minutes after it was hidden: its timers and fetch streams stop while
  `evaluateJavascript` still answers. A reply in progress then waited until the
  app was opened again (an owner job stood still for hours; the hidden
  flet-webview pages of hotfix 2 froze the same way, during page loads). Every
  page is therefore a `PageWebView`, which reports its window as visible
  (`onWindowVisibilityChanged` passes `VISIBLE` on, and the attach reports it
  too, because `View.dispatchAttachedToWindow` sends nothing for a window that
  is not shown). The container still sits behind the opaque FlutterView, so the
  page is never seen, and nothing is drawn while the Activity is stopped. Pages
  open and run while the app is hidden (no parking or warm-up). Detached from
  its window (no Activity) Chromium hides a page again until the next attach.
  The CI emulator proves it: the self-test suite `background`
  (`ci/android_smoke.sh` presses HOME) watches a loopback page for 75 s in the
  background (`document.visibilityState` stays `visible`, its timers and
  fetches keep running). Python bounds every reply too
  (`browser_driver.ReplyWatch`): a page that stops answering or running fails
  its request after 90 s with the reason instead of waiting out the request
  timeout.
- No manifest, Gradle or dependency change (no androidx.webkit).

iOS (`HiddenWebPages.swift`):

- A 1 x 1 pt clipped `UIView` at index 0 of the key window (behind the
  FlutterView; no user interaction, hidden from accessibility, alpha 1, not
  hidden). Every page call attaches it first (`DocumentDestinations.keyWindow()`,
  shared with the pickers), because WebKit throttles a `WKWebView` outside a
  window.
- `WKWebsiteDataStore.default()` and the default process pool, like
  webview_flutter_wkwebview: cookies and localStorage are shared with the
  flet-webview pages. No `uiDelegate` (alerts complete silently, `window.open`
  opens nothing), no user scripts, no `decidePolicyFor` (`prevent_links` is
  accepted and ignored: WebKit never hands such links to other apps).
- Failures with `NSURLErrorCancelled` and WebKit's 102 (a navigation replaced
  by the next one) are ignored; other failures send `error` and `finished`
  (`ok: false`). `webViewWebContentProcessDidTerminate` sends `gone` with
  `crashed` null: WebKit does not say whether the process crashed or iOS
  reclaimed it.
- **Background.** WebKit still suspends the web content process soon after the
  app leaves the screen, so requests pause until Glossarion is opened again.
  Python counts none of that time against the page (hotfix 4: the driver's
  `pages_suspended`, iOS with the app hidden; `browser_driver.ReplyWatch`), so
  a suspended reply waits for the app instead of failing as a stopped page.
  Screen changes no longer lose the page. If iOS terminates the web content
  process in the background, the page reports `gone` when the app returns and
  authchatgpt sends its message again on a new page.
- The podspec links `WebKit`; SwiftPM autolinks `import WebKit`. Deployment
  target, Swift version and `Package.swift` are unchanged. No `@MainActor`, no
  `async`, no iOS 14+ API.

Debug builds only: Flet's `triggerControlEvent` and `Control.invokeMethod`
`debugPrint` event payloads (URLs) and call arguments (scripts); Flet's
`main.dart` silences `debugPrint` in release builds. The on-device smoke check
`native_webview` proves the pages on the CI emulator and simulator.
`GLOSSARION_NATIVE_WEBVIEW=0` (a CI / developer switch; phones cannot set it)
falls back to flet-webview for comparison.

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
- **Cold-start deep links.** `receive_sharing_intent` 1.9.0 (a dependency
  until the Android share crash fix) returned `YES`
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
- **In-app browser routes.** WebKit suspends a backgrounded app's web content
  process. The hidden pages of authchatgpt/, authnd/, authds/ and search/gemini
  (native `WKWebView`s since hotfix 3, see above) therefore pause in the
  background even while a background task or continued processing keeps Python
  running. Public API offers no way around this; the requests continue when the
  app is opened again (Accounts › Experimental says so). The reply bound of
  hotfix 4 does not count that time on iOS (it would otherwise fail every
  attempt of a chapter while the user is away).
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

**Hotfix 3 in-app browser pages (2026-10-26):**

- **Kotlin.** `HiddenWebPages.kt` and the wired plugin pass a Kotlin frontend
  (K1) analysis with no error or new warning against the API 35 `android.jar`,
  JDK 17 and stubs of the Flutter embedding / androidx.core signatures they
  call (the session scratchpad's harness on Android lint's bundled Kotlin
  2.1 compiler). No codegen ran; the Build Mobile CI build is the first
  compile with kotlinc 2.2.20.
- **Swift, Dart.** Not compiled here (no Xcode, no Flutter SDK);
  `tests/test_web_pages.py` checks them statically (balance, method / code /
  event names, exact WKNavigationDelegate signatures, no `@MainActor` / async /
  heterogeneous literals passed to `result`, the platform-info ordering).
- **Python.** `tests/test_web_pages.py` covers the API (defaults, marshalling,
  privacy of errors) and `web_pages_fake.py`. The on-device proof is the
  `native_webview` smoke check.

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
- Hotfix 1: an authchatgpt/ book job keeps translating with the app on Home
  and the screen off (Android), one request after another on the reused
  hidden page.
- Hotfix 3: an authchatgpt/ reply survives opening Settings / Library / Reader
  while it answers; logins (ChatGPT token, DeepSeek sign-in, Google consent)
  carry over into the native pages.
