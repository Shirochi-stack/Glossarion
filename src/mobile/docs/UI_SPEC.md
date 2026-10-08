# Glossarion Mobile: chat-first UI specification (Flet 1.0.3)

**Status.** This is the definitive UI specification for `src/mobile/`, committed in milestone U0. It is the planning draft with every plan §5 correction applied: the always-visible output-mode row, nothing hidden, opaque routes, Library key parity, reader positions in Prefs, the dependency rule, and ChatGPT sign-in for the default model. It also closes the gaps found by the completeness check and applies the Flet 1.0.3 API corrections. It replaces §7–§11 of the earlier mobile-app design, which was form-first with a bottom NavigationBar.

**Authority.**
- Plan `hi-so-glossarion-has-goofy-bear.md` §5 overrides this document, and this document overrides the earlier designs.
- `FEATURE_MAP.md` maps every inventory feature to a surface defined here.
- `tests_host/test_feature_map.py` checks that the map stays complete against `feature_inventory.toml`.

**Kept from the mobile-app design.**
- Services: JobService, FileBridge, IntentRouter, OAuthBridge, MobileConfigStore, Prefs, SecureKeys.
- Threading rules: UiDispatcher, io_pool, and one exclusive job thread.
- The native extension, bootstrap and lifecycle.
- The per-screen internals of these screens:
  - Model Manager, Multi-Key Manager, Accounts
  - Glossary settings tabs, QA, Converter, Manga
  - Async, Review, SDLXLIFF

**Rules that apply throughout**
- Package root is `src/mobile/app/glossarion_mobile/`, with the UI under `ui/` (Appendix A).
- Deep-link scheme is `glossarion`, host `app`. The bundle id is `com.glossarion.app`.
- A HeadlessOwner is built only on the job thread.
- Config is snapshotted at job start, so Settings shows the banner "Changes apply to the next run".
- Schema defaults are display-only and never written to config. Mobile writes keys sparsely.
- The default model stays `authgpt/gpt-6-luna`, the desktop default. "Sign in with ChatGPT" is Welcome step 1 and the fix action of the Send button's `blocked` state.
- Shared module names follow plan §2:
  - `direct_text_store`, `library_core`, `library_covers`, `reader_doc`, `live_stream`
  - `progress_core`, `progress_actions`, `glossary_progress_core`, `glossary_files`, `glossary_document`, `parallel_epub_core`
  - `model_catalog_core` (plus the existing `model_options`), `key_pool_service`, `prompt_profiles`
  - `settings_schema` / `settings_rules`, `run_env`, `job_runner`
- **Milestone tags.** Tags such as "(U3)" or "(U9, optional)" say when a surface ships (plan §8). A surface tagged "U9, optional" is a mobile-only addition, so leaving it out never hides a desktop feature.
- Every Flet API named in this spec was checked against the flet 1.0.3 sources. The resulting constraints are collected in §5.0 and Appendix C.

**Sources**
- Plan: `C:\Users\ADMIN\.claude\plans\hi-so-glossarion-has-goofy-bear.md` (§5 UI and Decisions)
- Direct Text inventory: translator_gui.py `_InputOutputDialog` 1938–13124
- Library and Progress blueprints: epub_library.py, Retranslation_GUI.py
- Mobile chat and translator app surveys

---

## 0. Principles

1. **One home.** The home screen is the Direct Text chat. Every other desktop feature can be reached in at most two taps through one of these:
   - the **drawer**: destinations, with Settings and Help in its footer;
   - the composer: the **＋ sheet**, the **output-mode control** (chip or toggles in the action row), **slash commands** or **quick-action chips**;
   - an **action on a card**: job card, book card or chapter row.
2. **Data parity.**
   - These formats are unchanged:
     - `direct_text_chats.json` v2 and the `Direct Text/<chat>/` folder tree;
     - `translation_progress.json` v2.1 and `*_glossary_progress.json` v2.2;
     - `config.json` keys and the Library registries.
   - Data that exists only on mobile goes in sidecar files (Appendix B).
3. **Shared logic only.**
   - The UI binds to these shared modules:
     - `direct_text_store`, `library_core`, `reader_doc`;
     - `progress_core`, `glossary_progress_core`, `glossary_*`;
     - `settings_schema` / `settings_rules`, `model_options` / `model_catalog_core`, `key_pool_service`, and `run_env` / `job_runner`.
   - The UI never derives a status, a count or a path itself.
4. **Compact Material 3.**
   - `VisualDensity.COMPACT`, tonal surfaces, no dividers and no shadows.
   - **Hit targets are 48 × 48 dp everywhere.** Visuals may be smaller: 40 dp icon buttons, 32 dp chips, 28 dp composer chips. Smaller visuals get transparent padding or `size_constraints` to reach 48 dp.
5. **Thumb first.**
   - Primary actions sit in the bottom 40% of the screen: composer, extended FABs, bottom action bars and sheets.
   - The top app bar holds navigation and secondary actions.
6. **Nothing silently missing, nothing hidden.**
   - Excluded features appear where users would look for them. Each is **disabled, with a reason chip** (`ReasonChip`, §5.2).
     - The exclusions come from the plan: antigravity/, ocagy*/, ocz/, authza*/, autharena*/, search/opera, Tor, ollamapull/, Claude Code / Grok CLI credential import, DPI scaling and self-installing updates.
     - Their chip reads "Not available on mobile · <reason>".
   - Dependency-blocked features appear the same way, with the chip "Needs <package> · not in this build".
   - There is no "show desktop-only settings" switch.
   - Their config values round-trip untouched.
7. **Dependency rule** (plan Decisions).
   - A non-npm feature ships when its packages resolve for Android and iOS on pypi.flet.dev or PyPI.
     - `check_mobile_wheels.py` verifies this in the milestone that enables the feature.
     - Examples: grpcio 1.81 for Gemini gRPC; the Google Cloud Translate, TTS and Vision SDKs; azure-ai-documentintelligence; RapidOCR when pyclipper and shapely wheels exist.
   - If a package does not resolve, the feature uses a REST equivalent where one exists:
     - Vertex REST + google-auth;
     - Google Vision REST;
     - Translation v2 REST;
     - Text-to-Speech REST;
     - Document Intelligence REST.
   - Only native-impossible items stay disabled with a reason:
     - sentence-transformers silent-truncation embeddings;
     - torch-only manga models: manga-ocr, Qwen2-VL, EasyOCR, DocTR, PaddleOCR, torch inpainters, RT-DETR PyTorch and YOLO;
     - Argos offline MT (ctranslate2);
     - process priority and affinity.
8. **Opaque routes.** Routes and deep links carry only ids and enums, never file paths or user text (§1.4).

---

## 1. Shell and navigation

### 1.1 Size classes (`ui/responsive.py`, from `page.width` in `on_resize`; the shell is rebuilt only when the class changes)

| Class | Width (dp) | Shell | Chat column | Secondary content |
|---|---|---|---|---|
| Phone | < 600 | Modal `NavigationDrawer` (width min(0.8·w, 360); opened by edge swipe or ☰) | Full width, 12 dp gutters | Bottom sheets (≤ 90% height) or pushed Views |
| Large phone | 600–899 | Modal drawer (max 360) | Full width, 16 dp gutters; max 760 | Sheets; Book page Overview uses two columns in landscape |
| Tablet | ≥ 900 | Persistent sidebar of 300 dp (same content as the drawer) | Centered, max 860 | Right **SidePanel** of 380 dp, toggled per surface: chat settings, job detail, compare, term sheet |
| Wide | ≥ 1200 | Sidebar 320 | max 860 | SidePanel can be pinned open, giving three panes. Settings, Glossary editor, Book page and Manga use master–detail |

### 1.2 Shell composition

**Phone and large phone**
- `page.views` is a stack:
  - `View("/")` is the chat home: `appbar=ChatHeader`, `drawer=ChatDrawer`, and body `SafeArea(Column[Transcript(expand), JobStrip?, StatusCaption?, Composer])`.
- Every destination is pushed as its own `View(route)`. Back pops it.
- A screen opened from another screen (Files or the Reader from the Book page, a job's detail from the Overview) is pushed on top of it, so back returns there. A drawer destination and a top-level destination (Library, Jobs, Glossaries, Tools, Settings) start from the route's static parents instead, and so does a deep link or notification, unless it opens a full-screen surface or its static parents are already open. Re-opening a screen that is already on the stack returns to that depth.
- Views that are not routes (the keyword delete confirmation, the device checks) carry a `View.route` that no other View uses: the client keys Views by route, and back resolves to the first View with the popped route.

**Tablet**
- One root `View` with `Row[Sidebar, MainArea(expand), SidePanel?]`.
- The router swaps `MainArea.content` and keeps an internal back stack per destination. The sidebar never unmounts.
- The system / gesture back pops that stack (§1.6; the root View cannot pop while the main area shows a screen). With the chat in the main area it leaves the app.
- The Reader and full-screen editors are always pushed as top-level Views, so they take the whole screen even on tablets.

**Router (`ui/router.py`)**
- `page.on_route_change` parses the route against a **whitelist** (§1.4).
- Phone: rebuilds `page.views = [root, *stack]`. Tablet: updates MainArea.
- `page.on_view_pop` pops.
- Unknown routes are ignored. So are `content:`, `file:` and foreign hosts. Open-with never pushes a route (§1.5).

### 1.3 Drawer (`ui/shell/drawer.py`; the same component is the tablet sidebar)

On phone the drawer is a `NavigationDrawer(controls=…)` (see Appendix C item 3). On tablet the same content sits in a persistent `Container`.

From top to bottom:

1. **Header row (56 dp).**
   - Halgakos avatar (28 dp, `CircleAvatar` with `assets/icon.png`) and "Glossarion".
   - Trailing: `IconButton(edit_square)` "New chat" and `IconButton(chat_bubble_outline dashed)` "New scratch chat".
2. **Unified search.** `SearchBar`, placeholder "Search chats, books, glossaries".
   - Opening it switches the drawer body to results grouped as **Chats · Books · Glossaries · Files**, with filter chips for the same groups. **Series** joins them (U9: matched on the series name).
   - Chats match on title and message text; the index is built in io_pool from the v2 JSON plus response files, lazily.
   - Books use `library_core.query.book_matches_query`. Glossaries match on file names and terms.
3. **Destinations row.** A horizontally scrolling row of `Chip`s with leading icons:
   - **Library** · **Jobs** (with a `Badge` count of running and queued jobs; a pulsing dot while running) · **Glossaries** · **Tools**.
   - One tap navigates and closes the drawer.
   - Settings is deliberately not a destination chip. It lives in the footer (item 7).
4. **Pinned.** Chat rows. Hidden when empty.
5. **Series (U9, optional; shipped).** Shown once at least one series exists (§2.15). Series chats leave Recents. One `ExpansionTile` per series:
   - leading colour dot and series name; trailing count;
   - children are its chats plus "＋ New chat in series" and "Series page ›".
6. **Recents.**
   - Group headers: **Today / Yesterday / Previous 7 days / <Month YYYY>** (labelSmall, primary colour).
   - The list is lazy (`ListView(build_controls_on_demand=True)`).
7. **Footer** (sticky, `surfaceContainerLow`, 56 dp):
   - **Status chip.** Examples: "Gemini 3.5 Flash · 5 keys ✓", "GPT-6 Luna · ChatGPT #2 ✓", "GPT-6 Luna · Sign in with ChatGPT", "No API key: set up".
     - The last two use the warning colour.
     - The chip opens ModelSheet. In the sign-in case it opens the LoginSheet directly.
   - `IconButton(settings)` opens **Settings**.
   - `IconButton(help_outline)` opens **Help** (bundled user guide).

**Chat row (44 dp visual, 48 dp target).**
- Title (bodyMedium, one line, ellipsis).
- Trailing state icons:
  - `ProgressRing(14)` while that chat owns a running job;
  - 📎 count (attachment workspaces);
  - push-pin when pinned.
- The selected row has a `secondaryContainer` background.

**Row interactions.**
- **Tap:** switch chat. Allowed during a run on mobile; the running job keeps going and shows in the JobStrip. This departs from desktop, which blocks switching during a run.
- **Long-press** (`ListTile.on_long_press`): an `ActionSheet` (custom bottom sheet, §5.2) with:
  - Rename
  - Pin / Unpin
  - Move to Series… (U9)
  - Attachments (N)
  - Export chat
  - Duplicate as scratch
  - Delete
- **Delete:**
  - If the chat has no attachment workspaces: remove immediately and show an undo `SnackBar` (6 s). The actual `rmtree` is deferred; it is persisted in Prefs `pending_deletes` so the delete still completes if the app dies during the undo window.
  - If the chat has workspaces: a `ConfirmDialog` with the desktop text "This permanently removes the conversation, its messages, and every saved output file… cannot be undone", listing the folder name, followed by the same undo snackbar.
  - Folder validation uses the shared `_validated_chat_output_folder` rules: the folder must be a direct child of `Direct Text/`.

### 1.4 Routes (whitelist; ids are opaque, never file paths or user text)

| Route | Surface | Phone presentation |
|---|---|---|
| `/` | Chat home: current chat | root View |
| `/chat/<cid>` | Specific chat (`cid` = v2 int id; `s<uuid>` = scratch) | root View |
| `/chat/<cid>/settings` | Chat settings | BottomSheet (90%); SidePanel on tablet |
| `/chat/<cid>/attachments` | Attachments manager | View |
| `/chat/<cid>/compose` | Full-screen composer | full-screen View |
| `/chat/<cid>/m/<mid>` | Jump to message (jump procedure, §2.8) | root View |
| `/chat/<cid>/m/<mid>/edit` | Output editor | full-screen View |
| `/series/<sid>` | Series page (U9, optional) | View |
| `/library?shelf=progress\|completed` | Library home | View |
| `/library/scan-raw` | Scan for Raw | View |
| `/library/book/<bid>?tab=overview\|chapters\|glossary\|output&filter=<group>` | Book page | View |
| `/library/book/<bid>/metadata` | Metadata editor | full-screen View |
| `/reader/<bid>?ch=<index>&mode=translated\|original\|bilingual` | Reader (full screen on every size class) | full-screen View |
| `/jobs`, `/jobs/<jid>` (alias `/job/<jid>`) | Jobs page, job detail | View |
| `/glossary`, `/glossary/<gid>?tab=editor\|general\|balanced\|minimal\|refinement`, `/glossary/<gid>/entry/<n>`, `/glossary/unified`, `/glossary/parallel-pair` | Glossaries | View; an entry opens as a sheet |
| `/tools`, and `/tools/{progress,progress/glossary,qa,convert,headers,async,review,sdlxliff,manga,rpgmaker}` with `?out=<oid>` / `?tab=<enum>` | Tools | View |
| `/tools/qa/report/<rid>` | QA report viewer | View |
| `/tools/files/<root>`, `/tools/files/<root>/<fid>` | File browser at a root (`output`, `library`, `inbox`, `chats` or a workspace `oid`), optionally opened at a folder | View |
| `/tools/text/<fid>?hit=<n>` | Text editor on a file. `hit` is the index of a QA issue or search hit, computed by the opener | full-screen View |
| `/settings`, `/settings/s/<section>#<key>`, `/settings/{models,keys,keys/<pool>,accounts,profiles,profiles/<pid>,prefill,endpoints,appearance,notifications,storage,backup,import,logs,updates,about,danger}` | Settings | View |
| `/welcome` | First-run welcome | full-screen View |
| `/oauth/return?p=<provider>` | OAuth return | handled, no View |
| `/__selftest__?suite=smoke` | CI and Diagnostics self-test; prints `GLOSSARION_SELFTEST PASS <json>` or `FAIL <json>` | handled, no View |

**Ids.** All ids resolve through registries owned by the shared cores or by Prefs. An unknown id shows "This item is no longer available" and pops.
- `bid` / `oid`: the first 12 hex characters of `sha1(normalized workspace or library path)`, resolved through the `library_core` id registry.
- `gid`: the same scheme over the glossary path.
- `mid`: the first 12 hex characters of `sha1(fp)`, where `fp` is the message fingerprint (Appendix B). Fingerprints contain file names, so they never appear in a route themselves.
- `fid`: a 12-hex id from the FileRef registry (Prefs `file_refs`, a bounded LRU), for files and folders under the safe roots.
- `rid`: a QA report id (12 hex over the report path).
- `jid` = job id, `pid` = prompt-profile slug, `sid` = series id. `n` and `index` are integers.
- `<pool>` is an enum: `translation`, `fallback`, `glossary`, `glossary_refinement`, `qa_vision`, `metadata`, `ai_truncation`, `rolling_summary`, `truncation_retry`, `inpainter`, `tts`.
- `<section>` and `#<key>` are schema ids and config key names (enums), never values.

**Never in a route:** file paths, file names, search terms, find/replace strings, prompts or model queries. They are passed in-process to the opened surface (the router's `args` object). A deep link can therefore open a surface but never inject text.

### 1.5 Deep links

The scheme is `glossarion://app/<route>`, configured in `[tool.flet.deep_linking] scheme="glossarion" host="app"`.

**Examples**
- `glossarion://app/job/3f2a…`: notification tap. Opens the job's **origin** (the chat message or the Book page Chapters tab) and falls back to `/jobs/<jid>`.
- `glossarion://app/library/book/ab12cd34ef56?tab=chapters&filter=failed`
- `glossarion://app/reader/ab12cd34ef56?ch=12&mode=bilingual`
- `glossarion://app/settings/s/response_handling#retry_timeout`
- `glossarion://app/settings/keys/glossary`
- `glossarion://app/oauth/return?p=authgpt`: brings the app forward after OAuth. On iOS it also closes the in-app browser view.
- `glossarion://app/__selftest__?suite=smoke`: the CI emulator / simulator self-test.

**Rules**
- The router accepts both a bare path and a full URI.
- Query values are only ids or enums (§1.4). Text and file paths are never placed in a URL.
- Open-with and Share never reach the router. The native trampoline moves the file URI into an intent extra, and the IntentRouter imports the file (§7.6).
- `content:`, `file:` and foreign hosts are ignored.

### 1.6 Android back and gesture behaviour (first matching rule wins)

1. An open sheet or dialog closes.
2. Selection mode exits.
3. An open drawer closes.
4. Inside the Reader, visible chrome hides (a second back leaves the Reader).
5. A pushed View pops.
6. On the chat root, the system default applies (leave the app).

iOS uses the swipe-from-left-edge back gesture on pushed Views. On the chat root, the same edge swipe opens the drawer.

### 1.7 Global JobStrip (`ui/shell/job_strip.py`)

**Anatomy.** A 44 dp mini-player; `surfaceContainerHighest`, radius 12, margin 8/4.
- **Leading:** a determinate `ProgressRing(28)` with the job-kind icon in its centre (indeterminate until a total is known).
- **Two text lines:**
  - Title (labelLarge): "Translating · Book.epub", "Extracting glossary · Book", "QA scan · 3 books", "Compiling EPUB · Book".
  - Subtitle (labelSmall): "Ch 12/80 · 3 in flight · 12:41"; "Glossary · 24/80 chapters"; "Waiting for your glossary decision" (warning colour).
- **Trailing:**
  - a `Badge` "+2" when jobs are queued;
  - a Stop `IconButton` that uses the same state machine as Send (§2.4).

**Where it shows**
- Every View except the Reader (the Reader has its own live panel).
- On the chat home, only when the running job belongs to **another** chat. In the owning chat, the JobCard is the progress UI, and a "↓ Running job" chip appears above the composer when that card is scrolled off screen.
- On tablets it sits docked at the bottom of the sidebar.

**Interactions**
- **Tap:** opens job detail (sheet on phone, SidePanel on tablet).
- **Swipe down** (`Dismissible`): hides the strip until the next state change.
- **After a job ends:** "Done · Open" or "Failed · View" for 10 s, then the strip hides.

### 1.8 Jobs page (`/jobs`) and job detail (`/jobs/<jid>`)

**Sections** (one `ListView`, with section headers):
- **Running:** at most one; exclusive per D3.
- **Queued:** a `ReorderableListView` with drag handles.
- **Interrupted:** jobs recovered from `jobs/active.state`. Actions: **Resume** and **Discard**.
- **Finished:** the last 50 jobs from `jobs/history.state`.

**Row anatomy.**
- Kind icon; title; origin line ("Chat · My novel" or "Library · Book title").
- `ProgressBar` (Running only); state chip: Queued · Starting · Running · Stopping · Force stopping · Done · Failed · Cancelled · Interrupted.
- Trailing actions: Stop (running), ✕ Cancel (queued, confirm "Cancel job / Keep job"), Retry (failed), Open (done).

**App bar ⋯ menu.** Pause queue / Resume queue (stops starting the next job); Cancel all queued; Clear finished.

**Extended FAB.** "Pause queue" / "Resume queue" (LNReader pattern). It is shown only when ≥ 1 job is queued.

**Launch banner** (`LaunchBanner`; desktop crash/shutdown recovery).
- **When.** On launch, if `jobs/active.state` holds interrupted jobs. By then `shutdown_utils.restore_in_progress_rows_for_shutdown` has already restored their in-progress rows.
- **What.** The chat home shows a `Banner` once: "N interrupted jobs", with:
  - **Resume**: resubmits the most recent job's JobSpec;
  - **Review**: → `/jobs`, Interrupted section;
  - ✕.
- The same jobs stay listed under Interrupted until they are resumed or discarded.

**Job detail**
- **Header:** title, origin link, state, progress, elapsed, ETA.
- **Request cards:** live per-API-request cards from the classified log stream (shared `direct_text_store` classifier), sorted by spine order.
- **LogConsole:** filter chips All / Errors / Thinking / API; search; follow toggle; Copy; Share log. Lines are selectable text in the mono family at the log size (§6.2, 8/11).
- **Outputs** as `FileChip`s.
- **Buttons:** Open origin, Stop, Resume.

**Empty state.** 🗂 "No jobs yet" / "Translations, glossary extractions, scans and compiles you start appear here."

### 1.9 Notifications (native extension; channels)

| Channel | Importance | Content | Actions / tap |
|---|---|---|---|
| `jobs.progress` | low, ongoing (Android FGS) | "Translating *Book*: 12/80 chapters · 3 in flight"; "…: waiting for your glossary decision" (the glossary gates only) / "…: waiting for your answer" (any other question) while the job asks (at most once per second; a change the throttle holds back is sent on the next tick) | **Stop** (graceful, then force on a second tap) · **Open** → `glossarion://app/job/<jid>` |
| `jobs.done` | default | "Done: *Book*" / "Stopped: *Book* (12/80)" / "Failed: *Book*" | tap → origin; actions **Open** · **Share** (when there is one compiled output) |
| `jobs.action` | high | "Glossary ready: review needed" (**Accept** · **Review**) / "Answer needed: *question*" (Tools › Async batch) / "Sign-in required for ChatGPT #2" / "Paused (system limit): tap to resume" | tap / Review → the approval card (owning chat, or a Library book's page) / Tools › Async batch / LoginSheet / Jobs; **Accept** answers the glossary gate Yes |

**When a job asks** (glossary approval, the Library review gate, an Async batch question), the `jobs.action` notification is posted whenever the screen that answers it is not showing: the owning chat, the Book page, or Tools › Async batch. That covers the app hidden and the app open on another screen; in the second case a snackbar also appears ("Glossary ready: review needed" · **Review**, or "Answer needed: …" · **Answer**). Nothing is posted while that screen is on top. The notification is withdrawn as soon as the question is answered or given up (job event `question_resolved` {id, kind}, delivered while the app is hidden too). With "Always accept generated glossaries" on (§2.11, §2.14), the gate is answered on the job thread and nothing is posted.

**Accept** answers the running job's pending glossary question Yes. Android action buttons always bring the app forward, so there is no background accept. A chat run is answered through the chat's run controller (the card's ✓ Yes); any other job through `JobService.answer`. Then the snackbar "Glossary accepted · translating" shows and the route opens. A stale Accept (the job ended, another job's notification, no glossary question pending) only opens the route. A tap that cold-started the app (the launch notification) is routed once the app is ready (`JobsFeature.route_launch_notification`, called by `app._after_ready`), with the same rules.

**While the app is hidden**, the UI dispatcher parks the jobs view, so `BackgroundExecution` runs a 1 s ticker that keeps the ongoing notification current: on Android while the foreground service is up, on iOS while continued processing is up. It stops when the app is visible again or the queue drains.

**Swiped away.** Android 14+ lets users dismiss a foreground-service notification. While a job runs, the job notification is posted again (at most once per second), like a VPN's; nothing comes back after the service stops.

**Permission.** Android 13+ asks for `POST_NOTIFICATIONS` on the first Run, at Welcome step 4 and from Settings › Notifications & background. It is one request (`BackgroundExecution.request_notification_permission`), and "asked" is remembered only for a definite answer, so a failed or dismissed request is asked again by the next Run. The first long job started while notifications are off shows once: "Notifications are off: you won't see progress, finished jobs or glossary reviews" · **Turn on** → `/settings/notifications`. A post the system refuses is logged.

On iOS the same text goes to local notifications (action category Accept / Review), plus the `BGContinuedProcessingTask` progress on iOS 26+.

---

## 2. Chat home (Direct Text equivalent)

### 2.1 Header (`ChatHeader`: `AppBar`, height 56; transparent at rest, tinted once the transcript has scrolled)

**Tint.** `AppBar` in Flet 1.0.3 has no `surface_tint` or `scrolled_under_elevation`, so the tint is done by hand:
- The header starts with `bgcolor=surface` and `elevation_on_scroll=0`.
- `Transcript.on_scroll` swaps `bgcolor` to `surfaceContainer` once the offset passes 4 px, and back at 0.

**Leading.** ☰ `IconButton(menu)`, 40 dp visual and 48 dp target. It carries a `Badge` dot when a job is running in another chat.

**Title** (`Column`, tight spacing):
- **Line 1:** the chat title (titleMedium, one line, ellipsis). Long-press (a `GestureDetector(on_long_press_start)` around the title) opens `TextPromptDialog` "Rename chat" / "Chat name:" (whitespace collapsed, ≤ 120 characters).
- **Line 2:** labelSmall at 60% opacity, built from `TextSpan`s, each with its own `on_click`. If span taps prove unreliable inside the AppBar title, the fallback is a `Row` of three compact `TextButton`s (Appendix C item 4):
  - **model span** ("Gemini 3.5 Flash") → ModelSheet, Model tab;
  - " · ";
  - **profile span** ("Korean web novel") → ModelSheet, Profile tab;
  - " · ";
  - **target span** ("→ English") → ModelSheet, Language tab;
  - a trailing "▾".
  - When per-chat or series overrides are active, a 16 dp tonal chip reading "custom" is appended; it opens Chat settings.
  - At ≥ 160% text scale, line 2 shows only the model span plus "▾".

**Actions**
- **Scratch toggle** (`chat_bubble_outline` drawn dashed via a custom icon asset):
  - shown only while the chat is empty;
  - in a scratch chat it becomes a "Scratch" chip plus a `TextButton("Save")`.
- **New chat** (`edit_square`). The current chat is reused if it is empty (desktop `_new_chat` rule).
- **⋯** (`PopupMenuButton`):
  - Chat settings · Attachments (N) · Jump to… · Search in chat · Text size · Move to Series… (U9) · Export chat · Delete chat.

### 2.2 ModelSheet (`ui/settings/model_sheet.py`)

**Presentation**
- **Phone:** a draggable `BottomSheet` at 90% height; `fullscreen=True` when the window is under 640 dp tall.
- **Tablet:** a custom 420 dp overlay panel anchored under the header subtitle. It is a `Container` in `page.overlay`, dismissed by tapping the scrim. Flet 1.0.3 has no anchored popover control.

**Structure**
- **Title row:** "Model", or "Use once" in one-shot mode. Its ⋯ (`PopupMenuButton`) offers:
  - **Refresh online models**: re-polls the selected model's provider, ignoring the 24 h TTL (desktop "🌐 Refresh Online Models");
  - **Manage models** → `/settings/models`;
  - a **Hide unpolled models** switch (`hide_unpolled`).
- **Tabs** (`TabBar`): **Model · Profile · Language**.

**Model tab**
- **Search.** A `SearchBar` using the shared ranking (`model_options` / `model_catalog_core`): exact > prefix > path-segment > contains.
- **Provider chips** (scrolling `Chip` row): All · OpenAI · Google · Anthropic · DeepSeek · xAI · Mistral · OpenRouter · NVIDIA · Local · Custom · …
- **Sections:**
  - ★ Favorites (Prefs)
  - Recent (last 8)
  - Account aliases ("authgpt2/…", from the numbered account prefix completion)
  - Provider groups. Each group is sorted with polled models first, marked "✓ polled".
    - Each group header has a 🌐 `IconButton`, "Refresh online models for <provider>". It runs the same action as the ⋯ item, scoped to that provider, and the header shows the poll state.
  - Custom routes
- **Row (56 dp).** The list interleaves section headers with rows, so it does not set `first_item_prototype`; only the flat search-results list (rows only) may. Each row shows the model name, a provider badge and a status dot:
  - **green:** key or login OK;
  - **amber:** needs a key or sign-in. An inline `TextButton` reads "Sign in" or "Add key". For `authgpt/…`, including the default `authgpt/gpt-6-luna`, it reads "Sign in with ChatGPT";
  - **grey, with a `ReasonChip` "Not available on mobile":** the excluded routes antigravity/, ocagy*/, ocz/, authza*/, autharena*/, search/opera and ollamapull/. These rows stay visible, and a value set on desktop is preserved.
- **Row long-press** (`ListTile.on_long_press`) opens an `ActionSheet`: ★ Favorite · Copy id · Provider info · Refresh this provider.
- **Shimmer.** A `Shimmer` border runs on a provider group while it auto-polls (24 h TTL, `due_provider_catalog_for_model`) or refreshes.
- **Thinking & effort.** Pinned below the list in an `ExpansionTile`.
  - It renders the **schema fields for the selected model's family**, using `route_controls(model)`:
    - GPT / OpenRouter / NIM / OpenCode: effort, plus the OpenRouter token budget;
    - Gemini: enable / level / budget;
    - Anthropic: enable / budget / adaptive / effort;
    - DeepSeek: enable / effort / Responses format.
  - These are the same keys as Settings › Thinking & reasoning.
  - Switch **"Apply to this chat only"**: when on, values go to the chat overrides instead of config.
- **Route row.** Built from `route_controls`; shown under the search when relevant.
  - KeyField: "Add API key", or the masked key + Test.
  - LoginChip(s), with a slot dropdown "#1 ▾ / + Add account".
  - Gemini 📊 status.
  - GCP project dropdown for authgem-vertex/.
  - Vertex credentials PathTile + location dropdown. Vertex ships under the dependency rule: the SDK when its wheels resolve, otherwise Vertex REST + google-auth.
  - **Poe setup** for poe/ routes: opens `PoeSetupSheet` (deprecation warning, p-b cookie SecretTile, link to the guide, Test).
  - The excluded-route reason (`ReasonChip`) when an excluded model is selected.
- **Footer:** "Manage models" → `/settings/models` · "Keys" → `/settings/keys` · "Accounts" → `/settings/accounts` · ⓘ Provider information (bundled `docs/model_providers.md`).
- **Scope.** Selecting a model writes the global `model` key, unless "Apply to this chat only" is on.
- **One-shot mode.** When the sheet is opened from long-press Send → "Translate once with another model…", the title reads "Use once" and the selection is only a JobSpec override.
- **Field mode (`ModelPicker`).** The same component works as a form field in KeyEditor, Manga, Glossary and the QA AI-Hunter settings.
  - The field is read-only with a ▾. It opens this sheet without the Profile and Language tabs.
  - The ⋯ menu and the per-provider refresh buttons are the same as above.

**Profile tab**
- Radio list of prompt profiles (built-in badge; "modified" dot), each with a three-line preview.
- Below the list:
  - an **Assistant prefill** dropdown (Asst. Prompt profiles);
  - a **System ⇄ User** role toggle (`SegmentedButton`).
- Actions: Edit (→ `/settings/profiles/<pid>`), New, Manage.

**Language tab**
- Target language list from `language_options.TARGET_LANGUAGES`, with search, a Recent section and editable free text.
- Scope switch: "Global default" / "This chat". As built (U9) a series' model, profile and language are set from the Series page › Defaults card (Chat settings in series mode); the ModelSheet has no "This series" scope.

### 2.3 Composer (`ui/chat/composer.py`)

**Container.**
- Radius 24, `bgcolor=surfaceContainerHigh`, padding 8 × 12, no border, horizontal margin 8. The bottom is protected by `SafeArea`.
- There is no drop target. Flet's `DragTarget` only accepts in-app `Draggable`s, so files from other apps arrive through Share / Open-with (§7.6).

**Rows, top to bottom.** The order follows desktop Direct Text: attachments, input, output mode, actions. The output mode has no line of its own: it sits in the action row, right after ＋ (item 3).

> **Change (owner request, 2026-10-07).** The output mode used to be a separate row with all six toggles always visible ("Output: Text" + 📝 👁️ 🖼️ 🎬 🔊 ✨). On phones that row took a line from the text field. Now the **active mode is always visible** in the action row, and the six toggles are shown inline from a 600 dp chat column; on narrower columns one chip opens a menu of all six. The ＋ sheet (§2.5) still shows the six toggles on a line of their own.

1. **Chips row** (`Row(scroll=AUTO)`; shown only when non-empty):
   - **Attachment chip (`FileChip`, §5.3):**
     - icon by type: `menu_book` for epub, `picture_as_pdf`, `photo` for images, `collections` for cbz, `subtitles`, `description` for txt/md/json/csv/sdlxliff;
     - the file name (one line, middle ellipsis);
     - meta line: "EPUB · 1.2 MB · 48 ch" (io_pool fills in the chapter count later);
     - × to remove; ⋯ (`PopupMenuButton`): Preview · Add to Library · Remove.
     - Tap opens a preview: the Reader for epub, the image viewer, or the TextEditor read-only.
   - **Pasted-text chip (`PastedTextChip`):**
     - "Pasted text · 12,345 chars · ≈3.1k tokens".
     - Its ⋯ menu offers: Show in text field · Save as .txt attachment · Remove.
   - **One file attachment per turn**, matching desktop. Picking several files produces a **Batch** (§2.12.5).
2. **TextField:**
   - `multiline=True, min_lines=1, max_lines=6, shift_enter=True, border=NONE, dense=True, content_padding=4`.
   - Hint (desktop strings):
     - "Message to translate…" when there is no attachment;
     - "Add optional instructions for the attached file…" when there is one.
   - An **expand icon** (`open_in_full`, 20 dp visual, 48 dp target) appears at the top-right from 3 lines up. It opens `/chat/<cid>/compose`: a full-screen editor with a token count and "Done".
   - **Draft autosave** per chat, debounced 450 ms, written to the v2 `draft` field.
   - **Paste-to-chip.** When one `on_change` adds more than 10,000 characters:
     - that paste moves into a Pasted-text chip;
     - the field keeps any text typed before it;
     - a light haptic fires.
     - On send, the chip's text becomes the v2 `["user", text]` content.
3. **Action row** (height 48; one line that never wraps or overflows):
   - **＋** `IconButton(add)`. It rotates 45° into × while the ＋ sheet is open (`AnimatedRotation` via `rotate`). Long-press (`IconButton.on_long_press`) opens the Photos picker.
   - **Output mode** (`OutputModeRow(inline=True)`, §5.4). This is the desktop `_InputOutputDialog` "Output:" row: the modes **📝 👁️ 🖼️ 🎬 🔊 ✨** (Text, Vision, Image, Video, Audio, Refine). The active mode is always visible.
     - Width rules (`responsive.output_row_style`). They use the **chat-column width**, which is the composer's own width. On tablets that is the window minus the 300/320 dp sidebar, capped at 760/860 dp:

       | Chat column | Shown |
       |---|---|
       | < 600 dp, or ≥ 160% text scale | **Mode chip:** the active mode's emoji and a small ▾, as a 32 dp tonal pill inside a 48 dp target |
       | ≥ 600 dp | the six icon toggles inline (no "Output:" label); the selected toggle has a tonal fill |
       | once they fit next to ＋, the token hint, Send and one option pill (about 810 dp at 100% text: tablets from about 1110 dp, wide screens) | toggles that also show their text label ("📝 Text") |

     - **Mode chip:** `tooltip` "Output mode: Text" and `Semantics(label="Output mode: Text", button=True)`.
       - Tapping it opens a `PopupMenuButton` menu with the six modes (emoji + label, a check on the active one), a divider, and **"Options for <Mode>…"**, which opens that mode's `ModeOptionsSheet` (§2.6).
       - Picking a mode switches to it. Picking the checked mode does nothing.
     - **Toggles:** each has a 32 dp visual inside a 48 dp hit target, a `tooltip` ("Output mode: Vision") and a `Semantics` label that includes "selected".
       - **Tap an inactive toggle:** switches the mode (`selection_click` haptic).
       - **Tap the active toggle:** opens that mode's options sheet (`ModeOptionsSheet`, §2.6).
     - The choice is persisted as described in §2.6. The chip, the toggles and the ＋ sheet's row (§2.5) share one `OutputModeState` Signal, so they always agree.
     - After an automatic switch to Vision, the chip (or the active toggle) carries a small dot. Its tooltip reads "Output mode: Vision · auto" and it is spoken as "Output mode: Vision, automatic".
     - A mode that cannot fully run in this build stays visible. For example, video playback without flet-video: the options sheet shows the limitation with a `ReasonChip`. Nothing is hidden.
   - **Option pills.** Shown only when they differ from the default. Tapping a `Chip` opens the related sheet; its trailing × (`on_delete`) resets it.
     - "Glossary: Manual" / "Glossary: Off" / "Glossary: Main";
     - "Thinking off";
     - "Multipass on" (shown when "Force Multipass off" is *disabled* for this chat);
     - "→ Japanese" when the chat target differs from the global default;
     - "Model: once" while a one-shot model is armed.
     - At ≥ 160% text scale, the pills collapse into a single "Options (3)" chip that opens Chat settings.
     - The pills take the space left between the output mode and the token hint (`expand`), and scroll horizontally inside it.
   - **Token hint.** labelSmall "≈1.2k tok", one line, shown when the text exceeds 200 characters. It hides at ≥ 160% text scale (§7.5), so the row still fits on a 320 dp phone.
     - Counted with tiktoken in io_pool, debounced 450 ms.
     - Encoding fallback order (the desktop's): `encoding_for_model(model)` → o200k_base → cl100k_base.
   - **Send/Stop** button (§2.4).

**Status caption.** One line in labelSmall at 60% opacity, directly above the composer. It is shown only when the state is not Ready.

Exact desktop footer strings:
- "Translating…"
- "Attached X" / "Attached X · Vision enabled"
- "Manual glossary required"
- "Glossary ready — choose Edit, Yes, or No"
- "Starting translation…"
- "Finishing current request… Tap again to force stop" ("Click" becomes "Tap")
- "Force stopping…" · "Stopping…" · "Stopped"
- "No translated output was produced"
- "Run ended; showing streamed output"
- "Could not start"

Mobile-only string:
- "Sign in with ChatGPT to use GPT-6 Luna", with the fix buttons of §2.4.

**Hardware keyboard (tablets).** These go through `page.on_keyboard_event`:
- Enter sends; Shift+Enter inserts a newline; Ctrl+Enter sends.
- Ctrl+= / Ctrl+− / Ctrl+0 control text size.
- Esc closes sheets.

On soft keyboards, Enter inserts a newline; you send with the button.

**Supported attachment types** (desktop 5229–5238, plus the main-window list):
- `.txt .epub .pdf .md .markdown .html .htm .xhtml .xml .json .csv .tsv .srt .ass .lrc .vtt .log .sdlxliff .zip .cbz`
- images: `.png .jpg .jpeg .gif .bmp .webp .tif .tiff .svg .ico .heic .heif .avif .jxl`
- `.mp4`

Unsupported files get the snackbar "Unsupported attachment".

**Auto-switch** (desktop 5336–5383).
- Attaching an image or CBZ switches the mode to Vision. The composer's chip (or active toggle) then shows the "· auto" dot and tooltip, and the ＋ sheet's label reads "Output: Vision · auto".
- Removing the attachment, or attaching a non-visual file, restores the previous mode.

### 2.4 Send/Stop state machine (`SendStopButton`: `AnimatedSwitcher`, scale+fade 200 ms, 40 dp circle in a 48 dp target)

| State | When | Visual | Tap | Long-press menu |
|---|---|---|---|---|
| `idle_empty` | no text, no chip, no attachment | `arrow_upward` at 38% | — | — |
| `idle_ready` | content present | `FilledIconButton(arrow_upward)`, primary | send (light haptic) | **Translate once with another model…** · **Add without translating** (records the user turn only; the message later shows a "Translate" chip) · **Send as scratch** (spawns a scratch chat with this content) |
| `queue` | another chat's or a book's job is running (the desktop "blocked" case) | `schedule_send`, tonal | queues a job and shows the snackbar "Queued · runs after <title>" with **Undo** | Queue · **Stop current & send** (confirm) |
| `blocked` | any of: engine not ready ("Preparing engine…"); a glossary approval pending in this chat; manual glossary not provided; attachment file missing; excluded route selected; no key; **no ChatGPT sign-in for an `authgpt/` model, including the default `authgpt/gpt-6-luna`** | `arrow_upward` muted (38%), with a `tooltip` giving the reason; the status caption shows the reason and a fix button | snackbar with the reason and the fix action: **Sign in with ChatGPT** (opens LoginSheet) · Add key · Choose model · Provide glossary | — |
| `running` | this chat's job is running | `FilledIconButton(stop)`, `error` colour | calls `request_stop()`. If graceful stop is active → `finishing`, otherwise → `stopping` | **Force stop now** (heavy haptic) |
| `finishing` | graceful stop requested | `hourglass_bottom` inside a small ring, amber (`warning`) | within 2 s: force → `stopping`. After 2 s, tapping still forces (calls stop again, as desktop does) | Force stop now |
| `stopping` | force stop in progress | `ProgressRing(18)`, grey, disabled | — | — |

**Implementation notes**
- **`blocked` taps.** The `blocked` button is rendered muted but is not `disabled=True`. A disabled control receives no events, and on touch a tooltip only shows on hover or long-press, so a tap must still arrive to explain the reason.
- **Long-press menu.** It is a `ContextMenu(primary_trigger=ContextMenuTrigger.LONG_PRESS)` wrapping the button. Its items are rebuilt on each state change. A `PopupMenuButton` cannot be opened from code.
- **ChatGPT case.**
  - The status caption reads "Sign in with ChatGPT to use GPT-6 Luna".
  - It carries a `FilledTonalButton` "Sign in with ChatGPT" and a `TextButton` "Choose another model".
  - After a successful sign-in the state re-evaluates to `idle_ready`, so nothing needs retyping.

Transitions follow `JobSnapshot.state`:
- RUNNING → `running`
- STOPPING → `finishing`
- FORCE_STOPPING → `stopping`
- terminal states → `idle_*`

The button never reads `os.environ`. It only observes JobService signals.

### 2.5 ＋ sheet (`PlusSheet`: `BottomSheet(show_drag_handle=True, draggable=True, scrollable=True)`, ≤ 85% height)

Every tap triggers `HapticFeedback.light_impact`.

1. **Attach tiles.** A `Row` of 72 dp tiles, radius 14, tonal. Each tile is a `Container` with `on_click` and `on_long_press`. Order: Files · From Library · Photos · Camera · Clipboard (From Library second, so it needs no scrolling on a phone; device fix 2026-10-08).
   - **Files:** `FilePicker.pick_files(allow_multiple=True, allowed_extensions=…)`.
     - Long-press offers "Pick folder…" (`FilePicker.get_directory_path`).
     - On Android, a SAF failure falls back to "Pick a .zip instead".
   - **Photos:** images. CBZ comes in through Files.
   - **Camera:** `flet-camera`, when the package resolves for both platforms (Appendix C).
     - A multi-capture "Scan pages" mode produces a set of images. When there is more than one, they can be zipped into a CBZ.
     - If the package is not in the build, the tile is disabled with a `ReasonChip`, and Photos remains available.
   - **From Library:** opens the in-chat Library picker (the SourcePicker's Library mode: one list, no segments; the Library's "Filter title or tag…" search over titles, raw titles and tags, filtering on every keystroke; newest first, the Library's Date sort; each row is the Library's own list row with its 48 × 72 cover, type badge, size, progress pill and warning chips, no ⋯; "Open Library" in its header closes the picker and opens the Library). A Completed book with neither a workspace nor a raw file on this device is listed but disabled with the ReasonChip "No raw file on this device". One tap attaches the book's raw file to this chat (snackbar "Attached <name> from the Library" · Remove) and remembers it (sidecar meta `library_attachment` {bid, path}); Send then defaults to "Save to: Library" (§2.12.1). The empty-chat chip "From Library" (§2.13) and `/library [title]` (§2.7) open the same picker: attaching a Library book never leaves the chat. A long-press on a row starts a multi-selection (as on the Library shelves): further taps add or remove books and "Use N" puts them in one batch plan (§2.12; Start runs one translate job, each book in its own Library workspace). Rows are built a page at a time (the Library's page size; the next page when the list nears its end).
   - **Clipboard:** pastes clipboard text (`Clipboard.get()`) as a chip or into the field.

   Files are imported through FileBridge. It copies them into `Inbox/`, or into `Library/Raw/` when "Add to Library" is chosen in the chip menu.
2. **Output mode.** An `OutputModeRow` on a line of its own, sharing the composer's Signal (§2.3). It shows the label "Output: Text" (labelMedium; "Output: Vision · auto" after an automatic switch) and the six toggles. Where the composer shows the mode chip, the row shows the six icon toggles only. When the composer shows labelled toggles, the row shows the label plus labelled toggles. Below it, inline, the active mode's options (the `ModeOptionsSheet` content, §2.6), animated with `AnimatedSwitcher`.
3. **Tools** (`ListTile`s; also available as slash commands):

   | Tool | Action |
   |---|---|
   | Extract glossary | Glossary job card |
   | QA scan | Quick Scan of the chat's latest book workspace (its `Attachments/<stem>` folder, or the Library folder it moved to) as a `qa_scan` job (`qa_model.chat_qa_job`) with the mobile duplicate-check sample size (the saved Tools › QA value; mobile default 0 = off). The chat gets a "QA scan" card; progress, Stop and Done · Open are in the chat's JobStrip. A chat without a book workspace opens Tools › QA Scanner |
   | Compile EPUB / PDF | Uses the last attachment workspace |
   | Translate headers / metadata | |
   | Manga translator | Opens Tools › Manga with the attached images or CBZ |
   | Generate review | |
   | Async batch (50% off) | Plan card in async variant |
   | Progress manager | For the chat's attachment workspace |
   | Glossary progress | |
   | Retranslate chapters | Opens Chapters with selection mode |

   A tool that needs an input uses the current attachment or the last attachment workspace. Otherwise it opens the SourcePicker.
4. **This chat:** Glossary policy… · Chat settings… · Move to Series… (U9).

### 2.6 Output modes and the mode options sheet (`ModeOptionsSheet`)

**Where it opens.** Tapping the **active** toggle of the `OutputModeRow` (§2.3), or "Options for <Mode>…" in the composer's mode-chip menu, opens the sheet. Its content is also shown inline in the ＋ sheet.
- Title: "Output: <mode>".
- A switch **"This chat only"** at the top sends changes to the chat override instead of the global key.

**Persistence.** The mode is saved to `direct_text_output_mode`, or to the chat override when "This chat only" is on.
- It never touches the global `output_mode`. That key is the default for Library and book jobs, and it lives in Settings › Translation defaults (desktop parity).
- The env mapping comes from shared `run_env`:
  - `OUTPUT_MODE`
  - `VISION_OCR_FIRST`
  - `ENABLE_IMAGE_TRANSLATION`
  - `ENABLE_{IMAGE,VIDEO,AUDIO,REFINEMENT}_OUTPUT_MODE`

| Mode | Toggle (semantics icon) | Options (bound schema keys) | Assistant card renders |
|---|---|---|---|
| 📝 Text | `notes` | hint "Translate text, documents, subtitles and books" | Markdown translation |
| 👁️ Vision | `visibility` | Vision OCR prompt (PromptTile: `vision_ocr_prompt` / `vision_ocr_user_prompt`) · Skip translation (OCR only) `vision_ocr_skip_translation` · Batch Vision API requests + slots (`vision_ocr_batch_translation`, `vision_ocr_batch_size`, −1 = inherit) · Keep OCR image `vision_ocr_keep_images` · Process long images · Hide labels and remove OCR images · Vision keys (KeyPoolTile) | a collapsible "OCR" section, followed by the translation; the source thumbnail when the image was kept |
| 🖼️ Image | `image` | Output resolution `SegmentedButton` 1K / 2K / 4K (`image_output_resolution`) · Batch requests + slots · Image keys · custom image-edit endpoint status chip → Settings › Endpoints | generated image gallery (grid; tap opens the MediaViewer) |
| 🎬 Video | `movie` | Duration chips 5 / 10 / 15 / 20 / 30 / 60 s (`nanogpt_video_duration`) · Resolution 360p / 480p / 720p / 1080p (`nanogpt_video_resolution`) | VideoCard (§2.9) |
| 🔊 Audio | `volume_up` | TTS voice/file (`tts_voice`, with suggestions; Google Cloud TTS voices ship under the dependency rule, through the SDK or REST) · TTS keys (KeyPoolTile) | AudioCard + file chip |
| ✨ Refine | `auto_fix_high` | Refinement mode dropdown Full / Full + raw / Failed / Partial / Partial.b / Partial.b2 (`multipass_refinement_mode`) · Raw prompt role (`refinement_full_with_raw_raw_role`, Full + raw only) · Refine prompt (PromptTile → PromptEditor) | refined text; "Compare with original" (diff, stacked) when `unrefined_backup_file` exists |

**Generate from prompt (no input).** Image, Video and Audio only.
- A `FilledTonalButton` in the sheet submits a GENERATE_MEDIA job, the desktop generative-only run.
- **The prompt is the composer text** (decided). The job sets the owner attribute `_generative_prompt_override` (`image_job.GENERATIVE_PROMPT_ATTR`), which the moved runner's `_generative_prompt_source()` hook reads instead of the main-window prompt editor, then runs the desktop worker on `[image_job.GENERATIVE_MODE_SENTINEL]` (U7: the hook lives with the runner in `image_job`, not in `run_env`; the sentinel is needed for Audio, which the desktop's empty-selection rule does not treat as generative).
- The button is enabled only when the composer has text and no attachment. Otherwise it stays visible, disabled, with a reason chip:
  - "Type a prompt in the composer first";
  - "Remove the attachment to generate from a prompt".
- On success, the user turn is recorded as `["user", prompt]` and the media card follows.

**As built (U7).**
- The run_env hook is the owner attribute `image_job.GENERATIVE_PROMPT_ATTR` (`_generative_prompt_override`), read first by the shared `_run_generative_prompt_mode`; the `generate_media` job kind sets it and runs the desktop composition (`DirectTextRunOptions` with `selected_files = [GENERATIVE_MODE_SENTINEL]`, the Direct Text run environment, `_prepare_translation_run` + `_translation_worker`). The generated file lands in the run root and `ChatStore.finish_run` copies it into the chat folder as `Direct Text N.<ext>` and promotes the response's marker, like any Direct Text media response.
- Key pools: Vision keys = the `qa_scan` pool, Image keys = the `inpainter` pool (desktop "Image Keys"), TTS keys = the `tts` pool; each opens Settings › API keys. Batch for Image is the Vision batch pair (`vision_ocr_batch_translation` / `vision_ocr_batch_size`, desktop "Parallel Vision API requests for OCR chunks and EPUB/PDF image output").
- The tiles are evaluated with the sheet's mode selected (the desktop shows a mode's sub-settings only while it is selected); the two video keys show the desktop labels "Video Duration" / "Video Resolution" (the generated schema label is the neighbouring "Output Resolution:").
- Audio: without `google-cloud-texttospeech` in the build, a ReasonChip says Google Cloud TTS voices need it (dependency rule; the REST fallback is a later tier-B item).
- Vision: an image or CBZ is an attachment turn, so its responses sit in a JobCard; the card's collapsible "OCR (N)" section lists the OCR text the run cached in its workspace (`OCR/single`, `OCR/chunks`; `chat/media_model.ocr_entries`), and an image attachment's card shows the source image (the kept original).

### 2.7 Slash commands and quick-action chips

**Slash popover.** Typing "/" at the start of the field opens a `Container` overlay above the composer (inside the root `Stack`), containing a `ListView` of matching commands (max 6 visible). Tapping a command inserts it or runs it.
- U9 (`ui/chat/slash.py`): the popover sits in the chat column directly above the composer card. A command with an argument goes into the field on tap ("/model "); Send / Enter on a complete command runs it instead of sending; a command that sends nothing to the model (`/library`, `/model`, `/qa`, `/mode`, `/settings`, ...: every command except `/glossary`, `/headers`, `/metadata`, `/manga`, `/review`, `/async`, `/retranslate`) also runs while Send is blocked (not signed in, no API key; not while the engine loads). Each command runs the handler its button uses (＋ sheet tools, ModelSheet tabs, Chat settings, drawer). `/lang` and `/profile` with an argument apply it directly; `/policy manual` opens Chat settings (it needs a glossary file); `/settings <query>` opens the shared settings search prefilled. A command whose argument is optional (`/library`) runs on a tap.

| Command | Effect |
|---|---|
| `/glossary` | extract glossary |
| `/qa` | QA scan of this chat's book (Quick Scan, as ＋ › QA scan) |
| `/compile epub` / `/compile pdf` | compile |
| `/headers` / `/metadata` | Translate headers / metadata |
| `/manga` | Manga translator |
| `/review` | Review generator |
| `/async` | Async batch |
| `/progress` | Progress manager |
| `/retranslate <range>` | Retranslate chapters |
| `/mode text\|vision\|image\|video\|audio\|refine` | switch output mode |
| `/model <query>` | ModelSheet prefiltered |
| `/profile <name>` | |
| `/lang <language>` | |
| `/policy none\|attachments\|off\|manual` | glossary policy |
| `/scratch` | scratch chat |
| `/export` | export chat |
| `/library [title]` | the in-chat Library picker (§2.5); with a title, one attachable match is attached at once (an exact title wins), several open the picker with the query, none says "No Library book matches “title”" · Open Library |
| `/jobs` | Jobs |
| `/settings <query>` | settings search |

**Quick-action chips** (a scrolling `Row` of `Chip`s directly above the composer; at most 5; dismissible):
- **With an attachment in the composer:** "Translate" · "Extract glossary first" · "Translate as manga" (images/CBZ) · "Open in Reader" (EPUB / TXT; a TXT with no workspace yet opens in the Reader's text mode).
- **After a Result card:** "Read" · "Compile EPUB" · "QA scan" (that turn's workspace, §2.12.4) · "Retry failed".
- **On an empty chat:** the suggestion chips listed in §2.13.

### 2.8 Transcript (`Transcript`: `ListView(build_controls_on_demand=False, spacing=12, padding=12)` over a Python-side window)

**Window.** On open, the last `direct_text_rendered_card_limit` messages are rendered (default 20, at least 3), within a 120,000-character budget (desktop rule).
- The window *is* the virtualisation. Only windowed cards exist as controls, so every rendered card is built and can be a scroll target. That is why the list uses `build_controls_on_demand=False`.
- Scrolling near the top (`on_scroll` within 600 px of the start):
  - prepends `limit/3` older messages;
  - drops the same number from the far end when the budget is exceeded;
  - restores the offset with `scroll_to(offset=…)`.
- Loader rows (desktop strings): "↑ Scroll for earlier messages (N hidden)" and "Scroll for newer messages (N hidden) ↓".

**Lazy bodies.**
- Assistant content and thinking are read in io_pool from `Chat Messages/NNNNNN-response.*` and `-thinking.md`, with an LRU of 128 entries (desktop 4283–4309).
- A `Shimmer` placeholder is shown until the body loads.

**Streaming**
- Live cards always render at the tail.
- Text is pushed through a `UiDispatcher` channel. Repaints are coalesced every 280–900 ms, depending on the streamed size, and at least every 450 ms when auto-scroll is off (desktop cadence).
- **Auto-scroll** follows the tail with `scroll_to(offset=-1)`. `auto_scroll` itself stays `False`, because `scroll_to(scroll_key=…)` requires that.
  - It stops when `direct_text_disable_auto_scroll` is set, or when the user has scrolled up by more than one screen.
  - In that case a `FloatingActionButton(mini=True)` "↓" appears, with a `Badge` counting new cards.

**Keys and the jump procedure.**
- Each card has `key=ft.ScrollKey(mid)`, where `mid` is the 12-hex hash of the message fingerprint (Appendix B).
  - As built (U7 review): the key sits on the card's `CardSlot` (`chat/transcript.py`), a `Container` that stays the same object across re-renders with the card as its `content`, and a card whose inputs did not change is passed again as the same object (`ChatView._card`). Flet 1.0.3 diffs a *new* control matched to an old one by key as an immutable copy and freezes it, so a card re-created under its key could never be updated in place (Copy ✓, the AudioCard, a Vision run's OCR section, Compare with original, the jump highlight, live job progress). The highlight pulses the slot.
- A plain string key would become a `ValueKey`, which is not a scroll target.
- `scroll_to(scroll_key=…)` only reaches items that are already built.

So Jump-to, search hits, deep links (`/chat/<cid>/m/<mid>`) and the Input/Output steppers all use one procedure:
1. If the target is outside the rendered window, rebuild the window centred on it (the desktop approach), keeping the loader rows correct.
2. Let one frame pass: `update()`, then yield to the loop.
3. Call `await transcript.scroll_to(scroll_key=ft.ScrollKey(mid), duration=250)` and highlight the card for 1.5 s.
4. If the card is still not built (very tall neighbours), fall back to `scroll_to(offset=…)`, estimated from measured card heights, then retry step 3 once.

### 2.9 Message anatomy

**User text** (`["user", text]`)
- Right-aligned bubble; max width 75% on phone / 640 on tablet.
- `surfaceContainerHighest`, radius 18 (the bottom-right corner is 6), bodyLarge.
- The bubble is not wrapped in a `SelectionArea`, because a touch long-press would start a text selection instead of opening the menu. "Select text" opens a `SelectableTextSheet`.
- More than 12 lines collapse, with "Show more".
- **Long-press** (`GestureDetector` → `ActionSheet`): Copy · Edit & resend (refills the composer, then sends as a new turn; the earlier result becomes a version, §2.10) · Translate again · Select text · Delete.

**User file** (`["user_file", name, path, size, prompt, role]`)
- Right-aligned document card, `surfaceContainerHighest`, radius 16:
  - type icon (40 dp), **file name** (titleSmall), "EXT · 1.2 MB";
  - if a prompt exists: a thin spacer, then a labelSmall role label ("User instruction" / "System instruction" / "Assistant instruction") and the prompt text.
- **Tap:** preview (Reader, image viewer, or TextEditor read-only).
- **Long-press** (`Container.on_long_press` → `ActionSheet`): Copy instruction · Run again (new Plan) · Open workspace (Progress) · Delete.

**Assistant** (`["assistant", content, thinking, processing_label, output_folder, request_label, storage]`). Full width, no bubble.

1. **Header row** (labelSmall, 60%):
   - Halgakos avatar 20 dp; "GLOSSARION"; the request label; the timestamp.
   - Request label examples: "Request 3", "Chapter 12 (chunk 1/3) · ch012.xhtml · Request 15", "Header batch 1/2", "Metadata translation", "Header / TOC translation". Plain-text turns hide the internal labels (desktop rule).
   - Timestamp: "HH:MM" today, "Mon DD · HH:MM" this year, otherwise "YYYY Mon DD · HH:MM".
2. **Thinking disclosure** (`ThinkingDisclosure`):
   - **While streaming:** a row "▸ Thinking (1,234 tokens) …" with a `Shimmer` sheen. Other labels:
     - "Processing"
     - "Generating Text (567 tokens)"
     - "NVIDIA queue / prefill · Headers in 1.2s · Waiting for first token"
   - **On completion** it auto-collapses to "Token summary · Thinking N · Text M ›".
   - **Expanded:** a monospace `Container` (bodySmall mono, `surfaceContainerLow`, radius 8) holding `Markdown` of the **last 50,000 characters**, prefixed with "… earlier thinking output omitted …" when cut.
   - Fallback texts: "Waiting for the thinking stream…" (live) and "No thinking stream was emitted for this response." (saved).
   - The expanded state is persisted in the v2 `expanded` indices.
3. **Content:**
   - `Markdown(selectable=True, extension_set=GITHUB_WEB)`.
   - HTML/XHTML responses go through shared `direct_text_store.markup_to_display()`: sanitize as desktop 10528–10661, then html2text to Markdown for display. "View as HTML" renders the sanitized HTML in an `HtmlView` sheet (§5.5):
     - `flet_webview.WebView` on Android and iOS;
     - on Windows/Linux dev, where the WebView raises, the html2text Markdown rendering plus "Open in browser".
   - **Long outputs** (> 6,000 characters or > 60 lines): show the first ~40 lines, then "Show full translation (N chars)". This opens a full-screen paged view: the paragraphs rendered in a `ListView`, never as one giant `Markdown`.
   - **Pending placeholder:** italic "Working on your translation …" with 3 shimmer lines.
4. **Media** (from `[GENERATED_IMAGE|VIDEO|AUDIO:<path>]` markers and `storage.media_*`):
   - **Image:** `Image(src=path, fit=CONTAIN)` at max 76% of the viewport width (280–980) and max height 760 (a single image's height comes from its header size, `media_model.image_size`), radius 12. Tap opens the `MediaViewer` (full-screen `InteractiveViewer`) with Save / Share. If the file is missing: "Generated image unavailable." plus the file name.
   - **Video** (`VideoCard`): `flet_video.Video(playlist=[VideoMedia(path)], aspect_ratio=16/9)` with the package's default controls.
     - 1.0.3 has no `show_controls` or aspect parameter; `aspect_ratio` comes from `LayoutControl`.
     - ⋯: Save as… / Share / Open externally.
   - **Audio** (`AudioCard`). It uses the `flet_audio.Audio` service. Header "🔊 Generated audio".
     - Controls: play/pause `IconButton`, seek `Slider`, "m:ss / m:ss".
     - A volume `IconButton` that reveals a volume `Slider` (default 75%, as in the desktop media player).
     - ⋯ Save / Share / **Open externally**.
     - If playback fails: "Native audio playback is unavailable", with Share and Open externally.
5. **Action row** (§2.10).

**Special assistant cards.** These are persisted v2 messages identified by `request_label`:
- `"Extraction report"` → `ExtractionReportSection` inside the JobCard Result.
- `"Attachment actions"` → `AttachmentActionsRow` inside the JobCard Result.
- `"Library job"` (created by mobile) → `LibraryLinkCard`. Desktop renders this one as a normal Markdown card.

### 2.10 Per-message actions, refinement chips, version switcher

**Action row** (`MessageActionsRow`)
- 18 dp icons in 48 dp hit boxes (`IconButton` with 48 × 48 `size_constraints`), spacing 0, `onSurfaceVariant`.
- Each button has a `tooltip`.
- It is always visible under the **latest** assistant reply of each turn. Older replies reveal it on tap and fold it after 4 s.
- Buttons:
  - **Copy** (becomes ✓ for 1.6 s, desktop "✓ Copied");
  - **Retranslate** (`refresh`);
  - **Show source** (`compare_arrows`; text turns only). It toggles an inline tinted source block (`secondaryContainer` at 40%).
  - **Share;**
  - **⋯**.

**⋯ sheet** (`MessageMoreSheet`):
- **Edit translation** → `/chat/<cid>/m/<fp>/edit`:
  - a full-screen editor (`FullScreenEditor`):
    - `flet_code_editor.CodeEditor`, with `CodeLanguage.MARKDOWN` for .md/.txt and `CodeLanguage.XML` for .html/.xhtml (the package has no HTML language);
    - a plain `TextField` as the fallback;
  - "Save" / "Cancel", plus Ctrl+S / Esc on hardware keyboards.
  - Saving goes through shared `_save_response_output_edit`. It writes the .md/.txt/.html/.xhtml copies atomically and, for attachment chapter cards, also writes back to the real chapter file via `translation_progress.json`.
  - Disabled if the backing file is missing.
- **Add term to glossary.** Opens an EntrySheet (raw, translated, type, gender) and a target glossary picker: chat or series manual glossary, book glossary, or a new one.
- **Glossary terms used.** A sheet rendering `glossary_usage.build_chapter_footnote` for this output.
- Copy as Markdown / HTML / plain text.
- Save media as… (image/video/audio).
- Share file.
- Open files (FileBrowser at the output folder).
- Open in Reader (attachment chapter cards).
- View as HTML.
- Branch into new chat (new v2 session with messages up to here; folders are not copied).
- Delete message (confirm). This is mobile-only; it re-indexes `expanded` and the sidecar. Deleting a user turn (its long-press sheet) removes the turn together with its responses (U7). The shared store names a response's `Chat Messages/NNNNNN-*` files after its message index, so the deleted responses' own body files are removed and every later response's files are renamed to its new index (otherwise the next response or an Edit translation at a reused index would overwrite a surviving one); attachment outputs stay on disk. Job cards stay on their turns: the turns of the chat's jobs are recorded by fingerprint in the sidecar `job_turns` (keyed by the run root, which Resume / Retry keep). The delete runs on the io pool and saves the history at once.

**Refinement chips.** These sit under the latest *text-turn* reply in a scrolling row of `Chip`s:
- "More natural" · "More literal" · "Keep honorifics" · "Fix names (glossary)" · "Retranslate with…"

Each one re-runs the same source text with a fixed instruction passed through the existing attached-text prompt mechanism (`DIRECT_TEXT_ATTACHMENT_PROMPT`, role from settings). The text turn is written to a temp `.txt` as desktop does, so no new backend is needed. "Retranslate with…" opens the ModelSheet in one-shot mode.

**Version switcher.**
- Every retranslation, refinement chip or edit-and-resend appends a new assistant group.
- The original turn shows "‹ 2/3 ›" (`Row[IconButton, Text labelMedium, IconButton]`) at the left of its action row, and only the selected version renders.
  - As built (U7, `VersionSwitcher` in `chat/messages.py`): a version is a whole turn (the user message and its responses, recorded by Edit & resend / Retranslate / Translate again / Run again), so the switcher sits right under the visible member's user turn rather than in an action row; an attachment turn's responses live in a JobCard, which has no action row. The selected member renders at its own position; the others are hidden (`chat/chat_ops.version_view`).
- Version groups live in the sidecar. Desktop, which does not know about versions, shows all of them as consecutive cards, which degrades gracefully.

### 2.11 Glossary approval card (`GlossaryApprovalCard`)

**When.** A chat run's automatic glossary generation finishes, and the worker blocks on the shared approval `threading.Event` (desktop 45687–45726). The card renders at the tail.

**Notification.** Whenever this chat is not on screen (app hidden, or open on another screen), a `jobs.action` notification "Glossary ready: review needed" with **Accept** / **Review** is sent. With the app open elsewhere, a snackbar with **Review** also appears (§1.9). Nothing is sent while this chat is on screen, and the notification is withdrawn once the card is answered.

**Always accept generated glossaries** (mobile only; the desktop always asks). Chat settings › Glossary › "Always accept generated glossaries" (default off; All chats = Prefs `chat_auto_accept_glossary`, This chat = sidecar `auto_accept_glossary`). The effective value is captured at Send into the job (`params.auto_accept_glossary`; Resume / Retry keep it). The job answers the glossary gate Yes on its own thread (`JobService._job_ask`) and logs "✅ Direct Text: generated glossary accepted (Always accept is on)"; no card or notification is shown, and it works with the screen off. The request cards are not frozen at the gate (mobile divergence, Appendix C). A Stop before the gate declines it. The Library "Review glossary before translating" gate is never auto-accepted, and Async batch questions always ask.

**Anatomy** (`Card`, `tertiaryContainer` tint, radius 16):
- Header "GLOSSARION · ACTION REQUIRED" (labelSmall, tertiary).
- Title "Glossary generation complete" (titleMedium).
- Question "Accept this glossary and start translation?"
- A FileChip with the glossary file name and "N entries".
- A preview of the first 5 entries (raw → translated, bodySmall) and "View all".

**Buttons**
- **✏️ Edit** (`FilledTonalButton`) opens "Edit Generated Glossary — <file>":
  - a full-screen editor offering **Table** (the Glossary editor in single-file mode) or **Raw** (CodeEditor, monospace, no-wrap).
  - "Save" is atomic and keeps the BOM.
- **✓ Yes** (`FilledButton`, success colour) continues the run.
- **■ No** (`OutlinedButton`, error colour) rejects and calls `stop_translation()`.
- **Always accept** (`TextButton`, chat cards only): turns on the All-chats "Always accept generated glossaries" (Prefs `chat_auto_accept_glossary`; this chat's own "ask" override in its sidecar is cleared) and answers this question Yes like ✓ Yes; snackbar "Generated glossaries are accepted automatically from now on" · Chat settings. The Library review gate's sheet has no such button.

**No file.** Edit is disabled and the body reads "No editable glossary file was found. You can continue or stop this run."

**While waiting.** The Send button is `blocked` and the status caption reads "Glossary ready — choose Edit, Yes, or No".

**Persistence.** The card is not persisted (desktop parity). If the app is killed while waiting, the job becomes Interrupted (§1.8).

### 2.12 Job card lifecycle (`JobCard`)

The card is rendered in the assistant position after a `user_file` turn (or a Batch). It groups every following assistant message up to the next user turn.

**Data**
- Live state comes from `JobSnapshot` plus the classified request stream.
- Result state is rebuilt from persisted v2 messages: request cards, "Extraction report" and "Attachment actions".

#### 2.12.1 Plan (state `PLAN`, transient; stored in the sidecar `pending_plan`)

A Plan card is created when you send an attachment that is an EPUB, PDF, CBZ, ZIP, SDLXLIFF, subtitle bundle or folder, or a TXT/MD over 20,000 characters, unless "Skip plan for attachments" is on in Chat settings. Smaller files start immediately (desktop behaviour).

**Header.** Cover thumbnail (EPUB/PDF/CBZ via `library_covers`, otherwise a type icon), file name (titleMedium), and the detected type.

**Facts line.** "48 chapters · ≈310k tokens · 12.4 MB".
- Tokens are estimated from the extracted text size in io_pool, with a `Shimmer` while it runs ("estimating…").
- Conversions are announced, for example "ZIP → EPUB on start".

**Chips row** (`Row(wrap=True)`; each chip opens its editor):
- model · profile · "→ English";
- glossary:
  - "Glossary: Balanced (auto)" — the *effective* mode (critic gap);
  - or "glossary.csv · auto-mapped";
  - or "Policy: Manual";
  - a tap opens the `PlanGlossarySheet` (below);
- "📝 Text" output mode (opens the `ModeOptionsSheet`, scoped to this run);
- range: "All chapters" / "Ch 1–50";
- **destination:** "Save to: This chat" (default for files from Files / Photos / Clipboard; Direct Text semantics) or "Save to: Library" (normal pipeline: workspace in the output root, registered in the Library, post-QA scan per settings, a Library-job card in the chat).
  - A Library book attached from the chat (§2.5 From Library) defaults to "Save to: Library" and continues its own workspace. Its Plan card shows "📚 Library · <the book card's pill>" and "Open book".
  - Start first asks the desktop "Output Folder Mismatch" question when the book's workspace lies under another output root (`translate_sheet.confirm_output_root`); Cancel keeps the plan.
  - With "Skip plan for attachments" (or a short TXT) such a send starts the Library job at once.
  - The job's origin and the Library-job card carry the book id, so its "Library" button opens the Book page. The card's data is also kept in the sidecar meta `tool_jobs` (Appendix B), because the desktop history drops unknown storage keys.

**Run options** (`ExpansionTile`, collapsed; summary "Batch 10 · Temp 0.3 · Rolling summary"). These are schema tiles bound to:
- chapter range (N or N-M) + spine order + 🔍 preview list;
- input and output token limits;
- chunk size;
- temperature (+ disable);
- batch translation + size;
- context mode (+ history limit, summarize, retain);
- multipass + mode;
- post-translation QA scan;
- remove AI artifacts.

Switch **"Only for this run"** (default on): values go into JobSpec overrides. When off, they write config.

**Buttons:** "Choose chapters" · "Review glossary" (when a glossary exists or is auto-mapped) · `FilledButton` **"Start"** · `TextButton` "Cancel" (removes the plan and the unsent user_file turn; the sidecar forgets that turn too, so a cancelled Run again / Retranslate leaves no version-group member behind).

**Async variant.** A split ▾ on Start offers "Run as async batch (50% off)".

**PlanGlossarySheet.** Opened from the glossary chip on the Plan card, the TranslateSheet (§3.10) and the BatchPlanCard (§2.12.5).
- **Header:** "Glossary for this run", with the effective mode: "Balanced (auto)", "Manual: glossary.csv" or "Off".
- **Actions:**
  - **Load file…**: FilePicker for csv / json / txt / md. The file becomes the run's manual glossary (desktop 📄 Load Glossary).
  - **Use book glossary**: when the Library book has one.
  - **Clear ✕**: desktop loaded-glossary ✕.
  - **Map glossaries**: Batch only; desktop "Map Glossaries to EPUBs".
  - **Review glossary**: opens the glossary editor in single-file mode.
- A link "Change glossary mode" goes to Settings › Glossary › General.

#### 2.12.2 Queued
"Queued · starts after <title>" with **Cancel**.

#### 2.12.3 Running

**Header.** A `ProgressRing` and a state label:
- "Extracting chapters…" / "Generating glossary…" / "Translating" / "Waiting for your glossary decision" / "Translating headers…" / "Compiling EPUB…" / "Stopping after current request…" / "Force stopping…"

**Progress.**
- A `ProgressBar`: determinate from `ProgressWatcher` (`translation_progress.json` summary through `progress_core.compute_stats`), indeterminate before the output folder is known.
- Line: "Chapter 12/48 · chunk 2/3 · 3 in flight · ETA 14 min · 12:41".
- ETA comes from a UI-only moving average of chapter completion times. The in-flight count comes from `get_api_watchdog_state()`.

**Current item.** The last request label (bodySmall).

**Requests (N)** (`ExpansionTile`)
- Live request cards sorted by spine order. Each row: label, phase chip (Processing / Thinking / Generating), token counts, and a 3-line streaming preview.
- Tapping a row opens a `RequestSheet` with the full streaming content and thinking.
- These cards are persisted as v2 assistant messages exactly as desktop does.

**Log** (`ExpansionTile`). The last 200 lines in a `LogConsole` with a filter.

**Issue chips.** "Rate limited · retrying in 30 s", "Key cooling (2)", "Waiting for network…". They are derived from log classification.

**Buttons:**
- **Stop** (state machine);
- **Open reader** (Reader in overlay mode on the workspace, so you can read while it translates);
- **Progress** (`/tools/progress?out=` or the Book page Chapters tab);
- **Log** (job detail).

#### 2.12.4 Result

**Status line**, one of:
- "Done · 48/48 chapters"
- "Stopped · 12/48 chapters" → **Resume**, which resubmits the same JobSpec; the backend resumes from progress and chunks
- "Finished with issues · 3 failed"

When failures exist, a pinned error chip **"3 failed – Retry"** sits at the top.

**ExtractionReportSection** (`ExpansionTile`, collapsed). Summary: "Extraction: Ready · 48/48 payloads · 2 issues". Body rows:
- Source
- "Extraction: Ready/Incomplete · <mode> mode · <lang>"
- "Chapter payloads: x/y ready"
- "Content: N text · N image-only · N mixed · N empty/minimal"
- Resources extracted
- API requests
- "Tokens: Thinking · Text"
- Elapsed
- Potential issues (at most 4, shown as warning chips)

**Output chips.** Compiled EPUB / PDF, `*_translated.txt`, glossary.csv, SRT/ASS/LRC, SDLXLIFF.
- Tap: epub opens the Reader, pdf opens Share/open, text opens TextEditor, an image opens the viewer.

**AttachmentActionsRow** (wraps; the first 3 are visible, the rest go under "More"):
- **Read**
- **Export / Share** (ExportSheet)
- **Compile** (EPUB / PDF)
- **QA scan**
- **Progress**
- **Retry failed**
- **Glossary** (Book page Glossary tab equivalent for the workspace)
- **Open in Library**
- **Files**

**Library hand-off (automatic; device report #4).** There is no Migrate action. When a chat run of a book attachment (`library_core.RAW_IMPORT_EXTENSIONS`) finishes Done, `ChatFeature.auto_migrate` runs the desktop Migrate (`direct_text_store.migrate_attachment`): the workspace moves to the output root, only the newest top-level EPUB/PDF is kept, stored paths are rewritten, and the raw file is recorded in the Library registry. The raw stays where it is (no Library/Raw copy), so the raw-stem → workspace rule keeps working.
- The move waits while a job is running or writing the workspace: it needs the jobs' `JOB_LOCK`, taken without waiting, and no active job; the next idle transition retries it.
- It leaves alone workspaces whose job card still offers Resume or Retry failed, scratch chats, workspaces with no `translation_progress.json`, and non-book attachments.
- A same-named Library folder is merged into silently only when it is the same book: its `source_epub.txt` points at the same raw file or identical content, and any translation it already holds is of the same chapters (pipeline `content_hash`; a freed Inbox name can hold another book). Otherwise the workspace stays and the snackbar "A Library book named X already exists" · **Merge…** opens "Attachment folder already exists" → **Merge and replace** / **Cancel** (also in the Attachments manager's ⋯, §2.17).
- Success: snackbar "Added to the Library" · **Open book**; several in one sweep: "Added N books to the Library" · **Library**. Sweeps also run once after launch (books finished by earlier versions, or before the app was killed) and whenever the job queue goes idle.

**Open in Library** opens the Book page of the Library book the workspace became. Until the move the action is disabled with the ReasonChip "Added to the Library when the translation finishes" (non-book attachments: "Only books (EPUB, TXT, PDF, HTML) are added to the Library"). Resume / Retry failed of a workspace that moved into the Library open the Library's translate sheet (`ChatEnv.library_translate`).

**QA scan** runs a Quick Scan of the turn's own workspace (never the run's deleted temp root; a workspace that moved into the Library is followed). It shows "Nothing to scan in this turn's workspace" when there is no HTML (or TXT for a TXT/PDF source). The scan gets its own "QA scan" card with the scan state, its settings line ("Quick Scan · duplicate check off (sample size 0)") and Job · Report (QA report viewer; "The scan has not finished yet" / "This scan wrote no report") · Chapters (Progress manager). A Direct Text workspace goes through the shared scan path's `allow_direct_text` opt-in (`qa_model.chat_qa_job`; owner-approved). When a chat QA job ends, Result cards re-read their workspace's failed + QA-failed chapters for the "N QA failed" chip.

#### 2.12.5 Batch (several files)
- One `BatchPlanCard` lists the files as FileChips, each with a per-file glossary chip ("Map glossaries" sheet; desktop "Map Glossaries to EPUBs").
- It has an **Include subfolders** switch for folders.
- One Start submits a TRANSLATE job with sub-items.
- In history, consecutive `user_file` turns are recorded and each file's results follow its turn.

### 2.13 Error and empty states (all inside the transcript)

| Situation | UI (exact copy) | Actions |
|---|---|---|
| Empty chat | Halgakos (64 dp), "What would you like to translate?", "Paste text into the composer below, or attach a supported file. Translations stream into this conversation as they are generated." | suggestion chips: Paste text · Attach a book · From Library (opens the in-chat Library picker, §2.5) · Translate a manga page |
| Start failure | ErrorCard: "**Translation could not be started.**" + `ExcType: msg` (mono) | Retry · Copy error · View log |
| Empty result | "*No translated output was produced for this message.*" (or per request "*No translated output was emitted for this request.*") | Retranslate · View log |
| File-only result | "**Translation completed.** The translated EPUB file is available from the output chips below." | output chips |
| Missing body | "*The saved response file is missing or unreadable.*" / "*The saved thinking log file is missing or unreadable.*" | Delete message |
| Persist failure | ⚠️ line in Thinking + a banner "Saved to a temporary folder" | Retry save |
| No key / login | ErrorCard "No API key for <provider>" / "Sign-in required for ChatGPT #2" | Add key · Sign in · Choose model |
| Default model, not signed in | ErrorCard "Sign in with ChatGPT to use GPT-6 Luna" (`authgpt/gpt-6-luna` is the default model) | **Sign in with ChatGPT** · Choose another model · Add API key |
| Interrupted jobs at launch | LaunchBanner "N interrupted jobs" (§1.8) | Resume · Review |
| Excluded route | "<route> isn't available on mobile", with a ReasonChip | Choose model |
| Safety block | "Blocked by the provider's safety filter" (Gemini prohibited-use detection) | Retranslate with… · Settings › Provider options & safety |
| Offline | Banner on the running card: "Waiting for network…" | — |
| Attachment missing | snackbar "Attachment missing" | Remove chip |
| Manual glossary missing | "Manual glossary required" caption + ManualGlossarySheet | Provide glossary |

### 2.14 Chat settings sheet (`ChatSettingsSheet`: `BottomSheet` 90%, `SidePanel` on tablet)

**Scope.** A top `SegmentedButton`: **This chat · All chats**. "All chats" edits the global `direct_text_*` config keys, which desktop shares.

In "This chat", every control shows "Inherited from: All chats" (or "Series <name>" once Series ships, U9) with a ↺ reset. A "custom" badge appears in the header subtitle while any override is set.

**Layering.** Global config → Series defaults (U9, optional) → per-chat overrides. The effective values are applied as JobSpec overrides at send time and never written to config.

Sections are `ExpansionTile`s. The labels are exact desktop labels.

1. **Model & prompt.** Model override · Prompt profile override · Target language · Default output mode (`direct_text_output_mode`).
2. **Glossary.** `RadioGroup` bound to `direct_text_glossary_override_mode`:
   - "No Override" (`none`)
   - "No Override (Attachments Only)" (`attachments_only`, default)
   - "Force No Glossary" (`no_glossary`)
   - "Force Manual Glossary" (`manual`)
   
   The legacy booleans `direct_text_force_no_glossary` / `direct_text_manual_glossary` are mirrored by the shared code.
   
   **Manual** shows "Provide Manual Glossary" (`ManualGlossarySheet`):
   - a paste box (monospace);
   - "Browse…" (FilePicker csv/json/txt/md);
   - "Use glossary".
   
   It is asked **before every send** (desktop parity) and prefilled with the last glossary.

   - "Always accept generated glossaries" (mobile; All chats → Prefs `chat_auto_accept_glossary`, This chat → sidecar `auto_accept_glossary`; never config.json; default off). Help: "Skip the Edit / Yes / No card; translation starts with the generated glossary. Desktop always asks." Not shown on the Series defaults sheet (§2.11).
3. **Run behaviour:**
   - "Force Multipass off" (`direct_text_force_multipass_off`, default from legacy `direct_text_force_simple_mode`)
   - "Disable all thinking" (`direct_text_disable_thinking`)
   - "Skip prompt profile" (`direct_text_skip_prompt_profile`)
   - "Attached-text prompt role" `SegmentedButton` User / System / Assistant (`direct_text_attachment_prompt_role`)
   - "Skip plan for attachments" (mobile, sidecar)
4. **Conversation:**
   - "Disable conversation auto-scroll" (`direct_text_disable_auto_scroll`)
   - "Rendered conversation cards" `NumberTile` 4–200, step 2 (`direct_text_rendered_card_limit`). On mobile this sets the initial window and page size.
   - Text size (85–150%, per chat, sidecar).
5. **Series** (U9, optional). Current series and "Move to Series…".

**Footer.** "Reset chat overrides". Every change auto-saves; global keys use a 600 ms debounce.

### 2.15 Series (projects): optional, mobile-only, U9

Series is a mobile-only grouping stored in `mobile_series.json`. It is not a desktop feature, and it is optional; it shipped in U9.

**As built (U9).** `mobile_series.json` sits beside the chat sidecar (`{version: 1, series: [{id, name, color, cover_bid, book_ids, defaults, created_at}]}`, 12-hex route-safe ids, 8 swatches). A chat's series is `series_id` in `direct_text_chats.mobile.json` (scratch chats cannot join). Layering runs through `ChatStoreAdapter.override_layers` (Global → Series defaults → the chat's own overrides; `own_overrides()` is the chat alone), so runs, the header subtitle, the "custom" chip and the option pills see series defaults with no backend change. The Series page edits defaults with the Chat settings sheet in `subject="series"` mode. Not built: the ModelSheet "This series" scope, Glossaries "Use as manual glossary" for a series and a series target in "Add term to glossary" (a series' manual glossary pre-fills the chat's manual-glossary sheet instead).

**Fields.** Name, colour (8 tonal swatches), icon or cover (from a linked Library book), linked book ids, and these defaults:
- model
- prompt profile
- target language
- glossary policy, plus a manual glossary path or "use book glossary"
- default output mode
- direct_text overrides

**Series page (`/series/<sid>`).**
- Header with the colour and name, and an Edit button.
- Cards:
  - **Defaults:** opens ChatSettingsSheet in series scope.
  - **Glossary:** opens it in the Glossary editor; shows the term count.
  - **Books:** linked books as BookCards with progress.
  - **Chats:** list, with "＋ New chat in series".

**Where it appears.** The drawer's Series section, "Move to Series…" (in the chat ⋯ menu, the long-press sheet and Chat settings), and series-scoped search.

### 2.16 Scratch chat

- It is not written to `direct_text_chats.json`. Its output folder lives under `cache/Direct Text Scratch/<uuid>/`.
  - As built (U7): each scratch chat (id `s<uuid hex>`) is a `direct_text_store.ChatStore` of its own whose history file and output root are `cache/Direct Text Scratch/<uuid>/`, so its chat folder is `…/<uuid>/Direct Text/<title> - <ts>_<uuid8>/` and every shared folder rule (managed folder, attachments, Migrate) applies unchanged. The sidecar `scratch` list records the folders; a relaunch removes them (a scratch chat is never saved implicitly).
- A banner at the top of the transcript reads "Scratch chat — not saved", with a **Save** action.
- **Save:** assigns a v2 id, moves the folder into `Direct Text/<safe title> - <ts>_<uuid8>/` and rewrites paths through shared code (`_ensure_conversation_output_folder_for_session` names the folder, `_relocate_session_attachment_paths` rewrites the stored paths).
- **Leaving unsaved:** the confirm sheet "Discard scratch chat?" (Discard / Save). It is not shown when the chat is empty (an empty scratch chat is discarded silently).
- Entry points: drawer / header "New scratch chat", long-press Send › "Send as scratch", the drawer row's "Duplicate as scratch" (the messages are copied, off the UI loop; the copy gets its own copies of the original's `Chat Messages` body files, because Delete message or an edit in the original renames or rewrites those; attachment outputs and generated media stay shared by absolute path).

### 2.17 Attachments manager (`/chat/<cid>/attachments`)

- **Title:** "Attachments — {title}".
- **Intro:** "Workspaces waiting to be added to the Library (running, resumable, or a name already in the Library)". Finished books move into the Library by themselves (§2.12.4); there is no Migrate button.
- **Cards** (one per `Attachments/<stem>`):
  - icon, name, size, and a mini progress line ("48/48 · EPUB ready"). As built (U7, `chat/chat_ops.workspace_summary`): a read-only count of the workspace's `translation_progress.json` chapter statuses plus the compiled EPUB / PDF Migrate keeps (`_preferred_attachment_compiled_documents`); `progress_core.compute_stats` needs the full Progress Manager view build, which may seed the workspace, so a list card does not run it (⋯ › Progress opens it);
  - ⋯ (`ActionSheet`): Open in Reader · Progress · Share output · Merge into Library… (only when a different Library book already has the name; the desktop merge dialog) · Delete workspace (confirm).
- **Empty state:** "No attachment workspaces remain in this conversation."
- Delete workspace and Merge into Library… are disabled (ReasonChip) on a workspace while this chat's run is live, or while an active or queued job's folder is that workspace or inside it (`attachments.workspace_busy`, shared with the auto-migrate and `ChatView._workspace_busy`). They are checked again when the question is answered. Every move runs only while no job owns the process environment (`migrate_when_idle`); otherwise "Wait for the running job to finish, then try again".

### 2.18 Jump-to, search in chat, export, text size

- **Jump to… sheet** (`JumpToSheet`; the desktop Input/Output navigators):
  - **Step header:** "Input 3/7 · Output 5/12" counters, each with ▲/▼ step `IconButton`s.
    - Stepping moves to the previous or next input (user) or output (assistant) card, using the jump procedure of §2.8, and highlights it for 1.5 s.
    - The counters follow the card nearest the viewport centre.
  - Tabs **Inputs (N) / Outputs (N)**. Rows read "1. preview…" or "📎 name — prompt". The row preview replaces the desktop hover popup.
  - Tapping a row runs the jump procedure (§2.8) and highlights the card for 1.5 s.
  - The sheet is also reachable by long-pressing the "↓" FAB (`GestureDetector` wrapper).
- **Search in chat.** The header becomes a `TextField`.
  - Matches across bodies are loaded lazily, and the count shows as "3/17".
  - ▲▼ step through matches (jump procedure); Esc closes.
  - While search is open, a compact step row under the field also shows "▲ ▼ Output 5/12", so the navigators stay available.
- **Export chat** (ExportSheet): a Markdown transcript (share), or a ZIP of the chat folder plus the v2 JSON subset for this session.
- **Text size.** A `Slider` (85–150%) scales the transcript and composer for that chat. The transcript has no pinch-zoom, to avoid conflicts with scrolling.

### 2.19 Persistence and compatibility contract (`direct_text_store.ChatStore`, extracted from translator_gui 3424–4567)

**File**
- `direct_text_chats.json` beside `CONFIG_FILE` (`<data>/direct_text_chats.json`). The `GLOSSARION_DIRECT_TEXT_HISTORY` env var overrides the location.
- Format: `{version: 2, current_chat_id, sessions:[…]}`, written atomically, saves debounced 450 ms.

**Desktop normalizer limits.** Desktop's normalizer keeps only:
- session keys `id, title, messages, draft, attachment, output_folder, output_folder_name, next_output_index, expanded`;
- message roles `user`, `user_file`, `assistant`;
- storage keys `content_path, content_text_path, content_html_path, content_xhtml_path, thinking_path, image_path, media_path, media_kind, content_chars, thinking_chars, created_at`.

Mobile writes nothing else into this file. Every mobile extra goes into `direct_text_chats.mobile.json` (Appendix B), keyed by message fingerprints.

**Output tree.** Created by shared code, under `OUTPUT_DIRECTORY` = `docs/Output`:

```
<output root>/Direct Text/<safe title> - YYYYmmdd_HHMMSS_<uuid8>/
  Direct Text 1.txt …
  Attachments/<source stem>/…
  Chat Messages/NNNNNN-response.{md,txt,html,xhtml}
  Chat Messages/NNNNNN-thinking.md
```

Paths in the JSON are relative to the history file's folder, with an absolute fallback.

**Mobile-only mutations that stay v2-valid**
- **Delete message:** removes the tuple, renames the surviving responses' managed `Chat Messages/NNNNNN-*` files to their new indices (the removed responses' own files are deleted), re-indexes `expanded`, remaps sidecar fingerprints and `job_turns`.
- **Versions:** appended assistant messages.
- **Rename:** ≤ 120 characters.
- **Auto-title:** the first 42 characters + "…" on the first send while the title is still "New chat".

**Import from desktop.** Settings › Data › Import from desktop › "Direct Text chats". Pick a ZIP containing `direct_text_chats.json` + `Direct Text/`. Paths are rebased to the mobile output root; unresolved files are flagged as "missing" cards.

---

## 3. Library (first-class; `library_core` + `progress_core`)

### 3.1 Library home (`/library`)

**App bar**
- back (phone), title "Library", search `IconButton` (expands into a `SearchBar` with placeholder "Filter title or tag…"), `tune` (Filter sheet), grid/list toggle.
- ⋯ menu: Scan for raw (N) · Refresh · Library settings. There is no Organize / Undo (see §3.4).

**Shelf bar**
- `SegmentedButton` "In progress (N) · Completed (N)" (persisted as `epub_library_tab`).
- A trailing teal assist `Chip` "Scan for raw (N)", shown only when N > 0.

**Body.**
- **Grid:** `GridView(max_extent=card_w, spacing 8)`, with `child_aspect_ratio` taken from the preset (`card_w` : `cover_h` + text rows).
  - **Density** maps every desktop `epub_library_card_size` preset (`epub_library._SIZE_PRESETS`) to its `card_w` in dp:

    | Preset | Key | card_w | Note |
    |---|---|---|---|
    | 2XS | `2xs` | 78 | |
    | XS | `xs` | 92 | |
    | S | `compact` | 110 | desktop default |
    | M | `normal` | 140 | |
    | L | `large` | 180 | |
    | XL | `xl` | 230 | |
    | 2XL | `2xl` | 290 | |
    | 3XL | `3xl` | 360 | |
    | 4XL | `4xl` | 440 | |
    | 5XL | `5xl` | 530 | |
    | 6XL | `6xl` | 630 | |

  - Presets wider than the viewport render as one column. That clamp is display-only: the stored value is rewritten only when the user picks a preset.
- **List:** 72 dp rows with a 48 × 72 cover.
- Paging appends `epub_library_page_size` cards when the grid nears its end. Values: 20 / 50 / 100 / 250 / 500 / All (desktop default 20). There are no pager buttons; "All" renders through windowing.
- Pull-to-refresh (`PullToRefresh`, §5.1) runs a full rescan. ⋯ Refresh does the same.

**Extended FAB** (bottom-right, collapses while scrolling):
- In progress: "Import EPUB" (raw `.epub .txt .pdf .html .htm`).
- Completed: "Add translation" (`.epub`).

Import **copies** into `Library/Raw/` (or `Translated/`), records origins, and creates the workspace (`source_epub.txt` + an empty v2.1 progress file) through `library_core.actions.import_paths`. The new book appears as a "🆕 Not started" card.

**Empty states** (exact, adapted to mobile):
- In progress: "No translations in progress.\nUse “Import EPUB” to start one."
- Completed: "Your Library is empty.\n\nAdd finished .epub files with “Add translation” to see them here."

**Filter sheet** (`TabBar`: Filter · Sort · Display)
- **Filter:**
  - Format chips All / EPUB / TXT / PDF / HTML / IMG (`epub_library_format_filter`);
  - state chips (tri-state): In progress · Ready to compile · Not started · Outdated · Has QA failures · Missing raw;
  - Series (U9).
- **Sort:** Date / A-Z / Size (+ Reverse) (`epub_library_sort`).
- **Display:** Grid / List · Density (all 11 desktop presets; see Body) · "Raw titles" (`epub_library_show_raw_titles`) · Show language badge · Show progress bar on cover · Page size 20 / 50 / 100 / 250 / 500 / All (`epub_library_page_size`).

### 3.2 BookCard anatomy (exact strings from epub_library 5522–5963)

1. **Cover.** `Stack[Image(cached thumb), Ribbon, ProgressBar, ContinueButton]`, radius 8, 2:3.
   - **Ribbon** (top-left diagonal corner, labelSmall bold caps):

     | Ribbon | Colour |
     |---|---|
     | **NOT STARTED** | rgba(138,180,208,.92) |
     | **IN PROGRESS** | rgba(108,99,255,.92) |
     | **READY TO COMPILE** | rgba(60,170,110,.95) |
     | **OUTDATED PROGRESS** | rgba(255,179,71,.92) |
     | **⚙ COMPILING…** | rgba(255,209,102,.95) with #1e1616 text |

     There is no ribbon when completed.
   - **Progress bar.** A 3 dp `ProgressBar` along the bottom edge for in-progress books.
   - **Continue button.** A 28 dp ▶ `FilledIconButton` (48 dp hit area) at bottom-right when a reading position exists. It opens the Reader at that position.
   - **Cover fallback:** Halgakos placeholder.
2. **Title.** labelLarge, 2-line clamp, ellipsis. Raw titles use `card_raw_title`.
3. **Info row.** "x.x MB" / "N KB" in grey + a type badge (text, coloured dot):
   - EPUB #6c63ff · PDF #e74c3c · TXT #2ecc71 · HTML #3498db · IMG #f39c12 · FOLDER #ffd166
   - optional language chip ("KO"), from `metadata.language`.
4. **Warning chips:**
   - "⚠ missing raw" (#ff9e6d on 15% tint)
   - "⚠ +N" (#ffb347; tap lists the conflicting files)
   - **new:** "QA ⚠ N" (error tint) when `failed_chapters` > 0
5. **Pill** (in progress and not completed; pinned to the bottom of the card):

   | Pill | Colour |
   |---|---|
   | "⚠ Outdated Progress file" | #ffb347 on rgba(255,179,71,.18) |
   | "🆕 Not started" | #8ab4d0 on 15% |
   | "✨ Ready to compile (done/total)" | #6ee8a0 on 16% |
   | "⏳ done/total" (or "⏳ In progress") + " NN%" | #ffd166 on rgba(108,99,255,.18), border #6c63ff; the % part in #8ab4d0 |

   The percentage is `floor(done*100/total)`.

Counts come from `progress_core.compute_stats` (via `library_core`), so the card always agrees with the Book page.

**Light theme.** The same hues at M3 tone 40 on tone 95 tints. They are app-side constants in `ui/theme/colors.py`, because Flet themes have no custom colour roles.

### 3.3 Selection and bulk actions

- **Tap** always opens the Book page, on both shelves and for every type. A finished TXT book, a PDF without a workspace and a Library/Translated EPUB open it the same way an In-progress book does. ⋯ › ↗ Share sends the file to another app (`LibraryContext.share_books`). TXT books read in the Reader (📖 Open in Reader).
- **Long-press** (`Container.on_long_press`) enters selection mode with a medium haptic. A **contextual top bar** shows "N selected", Select all (current shelf), and Close. Selection is kept per shelf, keyed by path.
- **Bottom action bar** (`BottomAppBar` with icon+label buttons):
  - **Translate** ("Load N for translation"; opens a TranslateSheet / BatchPlan);
  - **Metadata** ("Translate Metadata for N EPUBs");
  - **Compile** (EPUB / PDF);
  - **Delete N**;
  - **More**: Delete glossary files (N) · Restore glossary backup · Clear saved raw link (for N items) · Share · Add to Series (U9).
- **Single-card ⋯** (from list rows, or long-press when only one card is selected) is an `ActionSheet` (custom bottom sheet, §5.2) with the desktop labels and visibility rules:
  - 📑 Open Book Details
  - 📖 Open in Reader (EPUB and TXT) / Open Translated EPUB / Open in EPUB reader (a PDF with a workspace)
  - 🔁 Load for translation
  - 🌐 Translate Metadata
  - 📘 Compile EPUB / 📄 Compile PDF
  - 📋 Copy Path (developer setting)
  - ✂️ Clear saved raw link
  - 🗑️ Delete glossary files / ↩️ Restore glossary backup
  - 🗑️ Delete
  - "Reveal …", "Open Output Folder" and "Open Library Folder" are replaced by "Files" (FileBrowser) and "Share".
  - A Library/Translated EPUB uses the workspace the scan resolved (`LibraryService.workspace_for`, desktop `_resolve_book_output_folder`, kept only when it holds `translation_progress.json`) for 📘 Compile EPUB / 📄 Compile PDF and Files.

### 3.4 File-changing actions (all through `library_core.actions` plan/execute; nothing deletes outside the Library or output roots)

- **Organize / Undo**: automatic, with no control (device fix 2026-10-08). Finished chat books move into the Library by themselves (chat auto-migrate, §2.12.4), and imports are copied into Library/Raw. On mobile the desktop Organize (n) / Undo (n) would only move imported copies around, because Undo counts every file in Library/Raw and Library/Translated. `LibraryService` keeps `plan_organize_blocking` / `execute_organize_blocking` / `plan_undo_blocking` as API. The collision policy dialog (Replace / Keep Both / Skip / Cancel; `organize.collision_text`) stays available to other flows.
- **Delete**
  - If every target is Not started: a simple `ConfirmDialog`.
  - Otherwise, a **full-screen DeleteConfirm View**:
    - one row per target with a checkbox;
    - a contents summary: "· N translated chapter HTML files", images, glossary files, total size;
    - a `TextField` that must contain "halgakos" or "delete" before the red `FilledButton("Delete")` enables.
  - The delete runs in io_pool with a progress row; there is no undo, since it deletes files.
- **Scan for Raw (`/library/scan-raw`)**
  - Folder picker (SAF) or "Library Raw folder" / "Inbox".
  - Match: Exact / Fuzzy, with a `Slider` 40–95 (default 70).
  - Extensions: Auto, or epub / txt / pdf / html chips. Settings `epub_library_scan_raw_*`.
  - **Scan** produces a result list: ☐ · workspace folder · matched file or "— no match" · ratio %.
  - Status line (exact strings):
    - "ⓘ No missing-raw workspaces to pair."
    - "⚠ No candidate files found in this folder (…)."
    - "⚠ 0 of N workspaces matched (…). Try …"
    - "✔ H of N workspaces matched (…)."
  - **Apply** writes `source_epub.txt`, records the raw file, and rescans.

### 3.5 Book page (`/library/book/<bid>`)

**Layout.** An `AppBar` (back; the title fades in after the hero scrolls out; ⋯), then a **sticky summary strip**, then a `Tabs(length=4)` with `TabBar` + `TabBarView`: **Overview · Chapters · Glossary · Output**.

**Summary strip** (48 dp, `surfaceContainerLow`):
- title (one line);
- a linear `ProgressBar` (completed / total, excluding skipped);
- an output-mode badge ("Mode: Text");
- while a job for this book runs: "⏳ Translating · 12/48" with Stop.

**⋯ menu.** Translate… · Compile EPUB / PDF · Translate Metadata · QA scan · Edit metadata.json (CodeEditor, `JSON`) · Files · ↗ Share (`LibraryContext.share_books`, as the card ⋯) · ⟳ Refresh · Clear saved raw link · Add to Series (U9) · Delete.

**Workspace.** Every workspace action (Compile EPUB/PDF, Edit metadata.json, Files, the Output tab, the running-job match, the ⋯ QA scan target) uses the book's resolved workspace, `LibraryService.workspace_for`, like desktop BookDetailsDialog resolving `_resolve_book_output_folder` at each use. A Completed book filed in Library/Translated keeps the workspace actions of the folder it came from, and its row and id stay the card's. A later Library scan that resolves the workspace rebuilds the ⋯ menu. Opening the page never creates a folder.

### 3.6 Overview tab

**Hero.**
- Cover 120 × 180 (phone) / 240 × 340 (tablet); title (titleLarge, raw toggle); author.
- Chips: 🌐 language · 📅 year · type.

**Progress strip** (in-progress books only): "⏳  Translation in progress — {done}/{total} chapters ({pct}%)" in the #ffd166 / #6c63ff style. It is hidden when completed.

**Actions.**
- Primary `FilledButton`:
  - "📖  Start reading" when completed;
  - "📖  Read translated" when in progress;
  - "📖  Read raw source" when there is no translation.
- Tonal "📜  Read raw source" (shown when translated chapters exist).
- **"Continue · Ch 12 · 43%"** when a reading position exists.
- Icon row: 🌐 **Translate…** (TranslateSheet) · 📘 **Compile** (menu) · 🏷 **Translate Metadata** (shown only when a raw EPUB resolves) · ↗ **Share** · 📁 **Files**.
  - Each icon is dimmed to 35% with an explanatory tooltip when its target can't be resolved.

**SYNOPSIS.** Expandable to 4 lines, with "No synopsis available." as fallback.

**METADATA list.**
- 📘 Title · ✍️ Author · 🏛️ Publisher · 🌐 Language · 📅 Year; missing values show "—".
- "✏️ Edit" (enabled only when a workspace exists) opens `/library/book/<bid>/metadata`:
  - a full-screen form: Title, Author, Publisher, Language, Date, Tags (comma/newline), Synopsis;
  - note: "Changes are saved to the output workspace's metadata.json. The original EPUB is not modified.";
  - saved via `merge_manual_metadata_edits` + `save_metadata_json_atomic`.

**TAGS.** Wrapping chips.

**At a glance.**
- The stats chip row (same as Chapters; tap jumps to the filtered Chapters tab).
- A glossary progress summary ("Glossary 72/80 · 1,234 entries"; tap opens the Glossary tab).
- The last job (link).
- No workspace: "📖 N chapters" (the EPUB's chapters; tap opens Chapters) instead of the error.

### 3.7 Chapters tab (Progress Manager parity; `progress_core.build_book_progress`)

**Header.**
- An output-folder chip "📁 <folder name>" (tap → FileBrowser at `/tools/files/<oid>`) and the output-mode badge.
- When the Progress Manager logic creates a missing output folder, a snackbar shows "📁 Created: <folder>".

**Stats row.** A scrolling row of `StatusChip`s with leading emoji and counts. Each chip is hidden when its count is 0, except Completed and the missing/failed groups.

| Chip | Note |
|---|---|
| "✅ Completed n" | |
| "🔗 Merged n" | |
| "🔄 In Progress n" | |
| "❓ Pending n" | |
| "⬜ Not Translated n" | refinement mode: "✨ Not Refined"; audio mode: "🔊 No TTS" |
| "❌ Failed n" | includes qa_failed + refine_failed; refinement mode: "💀 Refine Failed" |
| "⏭️ Skipped n" | |

- **Tap** filters by `STATUS_GROUPS` and toggles.
- **Long-press** jumps to the next matching row and wraps around (desktop click behaviour).
  - Each chip is wrapped in `GestureDetector(on_long_press_start=…)`, because `Chip` has no long-press event.
  - The jump uses the procedure of §2.8 against the chapter list.
- A labelSmall total above: "Total: N" or "Total: N (N-k chapters + k chunks)".

**Toolbar row**
- "🔍 Search chapters…" (matches raw/translated title, filename, chunk text and QA issue strings).
- Filter `IconButton` menu:
  - "Show special files" (persisted `epub_details_show_special_files`, Book Details parity)
  - "Show model info" (persisted `retranslation_show_model_info`)
  - "Show raw titles" (persisted `epub_details_show_raw_titles`)
  - "Rows per page" 20 / 50 / 100 / 250 / 500 / All (persisted `epub_details_chapter_page_size`, desktop default 20). This is the append increment of the list.
  - "QA failures only"
  - "Chunked chapters only"
- ⋯ menu:
  - "Manual editing" (`retranslation_manual_editing`; while sidecars are built the item reads "Creating sidecars… i/N")
  - "🔍 Edit Translation" (SDLXLIFF reviewer)
  - "📊 Glossary Progress" (switches tab)
  - "⟳ Refresh" (full reconcile)
  - "Files"

**List.** `ListView(build_controls_on_demand=True)`, without `first_item_prototype` or `item_extent`.
- Rows have variable height (the expandable QA line and chunk children). A prototype would force every row to the first row's height.
- Beyond 1,500 rows the pane switches to `WindowedList`.
- Rows have stable keys. They are `ft.ScrollKey`s, so stats-chip jumps and deep links can target them: `opf:<pos>:<file>`, `chunk:<key>:<i>`, `meta:<k>`, `artifact:<k>`.

**Row (`ProgressRow`, min 64 dp, two lines; grows for line 3 and chunk children)**
- **Leading `StatusAvatar`:** 32 dp, a Material icon on a 16% tint of the status colour; the emoji is the semantics label.
- **Line 1** (bodyMedium):
  - "Ch.012 · chapter0012.xhtml";
  - completed chapters with a translated title show the translated title (raw title on long-press).
  - Special rows:
    - PDF: "Section 007 · Pages 41-58"
    - "Metadata: <label>"
    - "Table of Contents" / "Chapter Headers"
    - "PDF OCR" summary: "PDF OCR · d/T pages, N cached, N no-text, N failed"
    - "🎨 Image Generation: d/T images…"
    - "Subtitle 003 · src → out · Batches c/t"
- **Line 2** (bodySmall): the output file, or the model when Show model info is on ("(model unknown)" when there is none; blank for pending / not translated). For chunked parents: " · Chunks 1✓ 2⚠ 3✗ 4… 5○ +N".
- **Trailing badges:** ⭐ refined · 💀 refine failed · 📸 image-only ("COPIED" as the model) · "OCR d/T" · "→ Ch.X" (merged) · "(N entries)".
- **Line 3** (QA only; expandable): the first 2 issues with previews (160 characters, 420 for ai_truncation) and "(+N more)".
- **Chunk parents** have a chevron. Expanding shows child rows "↳ Chunk i/T · <status> · <model>".
  - Selecting the parent selects the whole chapter.
  - Selecting only children triggers the "remove only these segments" flow.

**Status vocabulary** (exactly `progress_core.present`):
✅ Completed · 🔗 Merged · ❌ Failed · ❌ QA Failed · 💀 Refine Failed · 🔄 In Progress · ❓ Pending · ⬜ Not Translated · ✨ Not Refined · 🔊 No TTS · ⏭️ Skipped · ❓ Unknown.

The Book page uses this PM vocabulary everywhere. The Library-only badges ("✔ Translated", "⏳ Working") are not shown on mobile. This is a recorded divergence (Appendix C).

**Interactions**
- **Tap a chapter row:** opens the Reader at that chapter (mode Translated if completed, otherwise Original).
- **Tap a special row:** opens its row sheet.
- **Long-press:** selection mode.
  - **Top bar:** "N selected" · Select all (visible) · "Select ▾" (Completed / QA Failed / Failed group) · Clear.
  - **Bottom action bar:** **Retranslate** ("Reset TTS" in audio mode) · **Remove QA mark** · **More ▾**.
  - **More ▾** holds: Remove pending mark · Remove refinement status · Restore in-progress · Resolve QA issue · Insert missing image · Do not skip · Delete audio · 🔍 Edit Translation (SDLXLIFF compact reviewer).
  - Actions that `row_actions()` rejects for the current selection are shown disabled, with the reason.
- **Confirmations:**
  - The text comes verbatim from `plan_retranslation()`, ending "Continue?".
  - For RECYCLED TOC/header pairs, a 3-button dialog: "Delete Both Linked Files" / "Keep <counterpart>" / "Cancel".
  - The result is a summary snackbar with the exact `ActionResult` phrases ("Deleted N files", "Total N chapters ready for translation.", …).
- **Row ⋯** (trailing `IconButton` → `ActionSheet`):
  - 📖 Open in reader
  - 📂 Open file (TextEditor read-only / Share)
  - ✏️ Edit file (find QA issue) → TextEditor at the hit
  - 🔍 Edit Translation
  - 📋 Copy QA issue
  - 🔊 Play audio / 🗑️ Delete audio
  - ⚠️ Resolve QA issue (LLM-token repair, then a Before/After sheet of ≤ 20; or a RESOLVE_QA job for raw foreign text)
  - 🖼️ Insert Missing Image
  - 🧹 Remove QA Failed Mark
  - 🧽 Remove Pending Mark
  - ⭐ Remove refinement status
  - Restore In Progress Status
  - 🌐 Translate / Retranslate this chapter (SINGLE_CHAPTER job)
  - Skipped special rows show only "⏭️ Do not skip (remove keyword '<kw>')".

**Engine-bound actions** (Resolve QA (Partial.b), Insert image, refinement) go through JobService and are capability-gated: they need a model and a key.

**Every write** goes through `mutate_progress` (lock → re-read → three-way merge → atomic replace).

**Image-folder variant.** A thumbnail grid (`GridView`). Rows: "📄 Image n | base | ✅ Completed" and "🖼️ Cover | ⏭️ Skipped (cover)". Selection actions: **Select translated** · **Mark as Skipped** · **Delete Selected** through `progress_core.build_image_folder_progress` / `mark_image_folder_items_skipped` / `delete_image_folder_items` (the desktop view's shared code). Not ported to the v2.1 structure: the view still reads the pre-2.1 flat / `images` progress layout, so against a v2.1 `chapters` file no progress entry is removed (desktop bug kept verbatim; tests/parity/DISCREPANCIES.md U5 desktop bug 3 and U7); its progress writes go through `mutate_progress`.

**Empty state.** "No chapters found yet" / "Start a translation to see chapter progress." and a **Translate…** button.

**No workspace** (an "Add translation" EPUB, or an organized book whose workspace is gone): the tab lists the EPUB's own chapters from `load_book_details` (`BookDetailsModel.row_specs`, the desktop Book Details list). Header "📖 Chapters in this EPUB · N"; rows "Ch.NNN · title" / file name / desktop badge; tap opens the Reader at that chapter. Search, Show special files, Show raw titles and Rows per page apply; selection and the bulk bar are off. With no chapters either: "This book has no output workspace yet". The Progress Manager is never built for such a book, so no folder is created.

### 3.8 Glossary tab (Glossary Progress parity; `glossary_progress_core`)

**Header**
- "📖 {book_title}".
- A file chip "📁 <base>_glossary_progress.json" (tap opens Files).
- If the file disappears, a `Banner`: "⚠️ Progress file was deleted. Waiting for a new glossary progress file…".

**Glossary file card**
- File name, entry count, a per-type breakdown (characters, terms, …), and last modified.
- Buttons: **Open in editor** · **Extract** / **Continue extraction** (job) · ⋯ (Delete glossary files · Restore backup · Load as manual glossary).

**Stats chips.** Total · ✅ Completed · ⏭️ Skipped · 🔄 In Progress · ❌ Failed · 🔗 Merged · ⬜ Not Translated · ✨ Not Refined · 💀 Refine Failed. The last three and Merged are hidden at 0. Tap filters; long-press jumps (`GestureDetector` wrapper, as in §3.7).

**Pinned rows**
- **Minimal Pass:** "Minimal Pass · {icon} {label} · {model} · N entries". Labels include "Skipped - No Entries", "Skipped - Stopped", "Not Translated".
- **Refinement:** an "All Entry Types" aggregate, then one row per active type: "Refinement · {type} → {model} · 1,234 entries · chunks c/t · refined a → b".

**Chapter rows.** "[001] Ch.001 · {icon} {label} · filename → model", with ⭐ / 💀 and QA suffixes. Icons and colours are the GP vocabulary:
- ✅ Completed #27ae60
- ⏭️ Skipped, 📄 Empty (Skipped), 📸 Image Only (Skipped), 🏷️ Title/Header Only (Skipped) — #94a3b8
- ❌ Failed / Qa Failed #e74c3c
- 🔗 Merged #17a2b8
- 🔄 In Progress #f59e0b
- ⬜ Not Completed #5a9fd4

**Row ⋯** (trailing `IconButton` → `ActionSheet`; the desktop per-row context menu):
- 📝 Show footnote
- ✅ Mark as completed
- 🗑️ Remove from progress ("Remove <Ch.X>")
- ✨ Refine this

**Selection bottom bar:** ✅ Mark as Completed · 🗑️ Remove from progress · ✨ Refine (selected types) · More (📝 Show glossary footnote(s) → Markdown sheet with Copy · 📄 Generate completed summary → writes `glossary_footnotes/<book>_completed_glossary_footnotes.md` and shares it · ✅/❌ Skip unmatched entries (toggles `glossary_progress_skip_unmatched_entries`)).

**Path-row actions:** Select All · ✨ Refinement (preview sheet: types, chunk count from `plan_refinement`; Start → job) · Files · ✏️ Open Glossary.

**Writes** go through `glossary_progress_core`, with the extractor's lock + atomic replace. The desktop's plain dumps are a recorded bug; desktop changes only in a separate, user-approved commit.

**Empty state.**
- 📊 "No glossary extraction progress found for: <book>"
- "Run glossary extraction to see chapter progress. Refinement entry types are listed below."
- Refinement rows built from the glossary file, and an **Extract glossary** button.

### 3.9 Output tab

- **Compiled outputs** (EPUB / PDF / TXT / HTML): each has Open · Share · Save to… · Save to Downloads (Android) / Show in Files (iOS) · Delete. Conflicts appear as "⚠ +N".
- **Compile panel:** **Compile EPUB** · **Compile PDF** (PyMuPDF) · links to Settings › EPUB output and Settings › PDF.
- **Workspace files** (collapsible groups):
  - Glossary files
  - metadata.json · TOC.txt · translated_headers.txt · extraction_report.txt
  - SDLXLIFF/ sidecars
  - text_to_speech/ (inline audio players)
  - images
  - QA reports (open in the QA report viewer)
  - Review (📝; open in Review)
- **Raw source row:** file name, Share, "Re-link…" (Scan for raw).
- **Storage line:** "Workspace 84 MB", with a "Files" link.
- The workspace is the resolved one (§3.5): an organized book lists its workspace's compiled outputs plus the Library EPUB itself, and Compile writes into that workspace.

### 3.10 TranslateSheet (Library-origin Plan)

The TranslateSheet is the same `PlanCard` component, presented as a `BottomSheet` (90%). Its glossary chip opens the `PlanGlossarySheet` (§2.12.1). It adds:
- **"Review glossary before translating"** switch. It defaults to off, because desktop has no gate for book-origin jobs. When it is on, the run pauses after glossary generation exactly like a chat run:
  - a `jobs.action` notification "Glossary ready: review needed" (Accept / Review) unless the Book page is on screen (§1.9);
  - an approval sheet over the Book page, built from the `GlossaryApprovalCard` content: ✏️ Edit / ✓ Yes / ■ No.
- "Open in chat instead": creates a chat with the attachment and a plan.
- **Start**: when a book's workspace lies under a different output root than the current one, the desktop question "Output Folder Mismatch" comes first (Cancel / Yes; the shared `LibraryShelf.output_root_mismatch` prompt). Yes switches the process output root (`OUTPUT_DIRECTORY`; config.json `output_directory` is not written, so it lasts for the session). The check runs only while no job holds the process environment. The chat's Library attach asks the same question before Start (`translate_sheet.confirm_output_root`).

The job has origin = book. Progress shows on the Book page summary strip, the JobStrip and Jobs.

### 3.11 Reader (`/reader/<bid>`; full-screen View on every size class; `flet_webview.WebView` + native fallback)

**Document**
- `reader_doc.wrap_reader_html(mobile=True)` (`ReaderDocument.wrap(..., mobile=True)`), served from the in-app localhost HTTP server (`services/reader_server.py`: `http://127.0.0.1:<port>/<token>/reader.html?v=N`, a random 192-bit path token, Host-header guard, document CSP; Android allows cleartext for the loopback host only through the extension's `glossarion_network_security_config.xml`).
- Book content never runs code in the page: chapter HTML loses `<script>`, frames, objects, `<base>`, meta refresh, `on*` handlers and `javascript:` URLs (`document.sanitize_book_html`), the book's CSS cannot close its `<style>` (`inert_css`), and the page's own scripts carry a per-page nonce that the document CSP (`script-src 'nonce-…'`) allows exclusively. `/img/<id>` serves only bytes that are an image by content, events are accepted only as JSON `POST`s, and a book's external link opens after an "Open link?" confirmation (http/https/mailto only).
- Mobile additions: a viewport meta tag, `-webkit-column-break-*`, 16–20 dp padding, safe-area insets, `100dvh`.
- Pagination uses CSS columns. Tap zones (left / right thirds) and swipes are handled in page JS.
- Chapter changes navigate the existing WebView with `load_request` (`reader_view.navigate_webview`; flet-webview reads `url` only when the control is built). The first page builds the WebView with its URL, and a WebView whose navigation fails is rebuilt with the page URL (each build gets its own key, `reader-webview-N`). Device fix 2026-10-08: before it, no chapter after the first one loaded on Android/iOS.
- Events reach Python as console messages `GLRDR:{json}` (`reader_doc.MOBILE_EVENT_PREFIX`, read through `on_console_message`) and, as a twin, `fetch` POSTs to `/<token>/__ev` (`MOBILE_EVENT_PATH`). Every event carries `{type, seq, chapter, …}`; `seq` de-duplicates the two transports and `GLRDR.setTransport('console'|'fetch'|'both')` narrows them. The Reader's own extras (`ui/reader/bridge.py`) add find / anchor / live restyle, a cleared-selection event and the selection rectangle.

**Native fallback.** `reader_doc.html_to_blocks()` feeds a `ListView` of `Text` / `Image` controls. It scrolls; edge taps scroll a screen and at the end/start of the list open the next/previous chapter (previous: at its end; `fallback_view.page_by`: a list known to be able to move never turns on late scroll events; a chapter shorter than the screen turns on the first tap); text is not OS-selectable (long-press a paragraph for its actions); another chapter opens at its top, the same chapter drawn again (Aa) keeps its offset. It is used:
- automatically on Windows/Linux dev, because flet-webview raises outside Android, iOS and macOS;
- for the "Lightweight reader" setting;
- on WebView failures.

**Chrome.** Tapping the centre toggles it, with a 150 ms fade.
- **Top bar** (translucent `surface` at 92%):
  - back, then the chapter title over the book title;
  - **`SegmentedButton` Original · Translated · Bilingual** ("Orig · Trans · Both" under 400 dp);
  - search; ⋯ (Bookmarks, Open Book page, Lightweight reader, Share chapter).
- **Bottom bar:**
  - chapter `Slider` with haptic ticks, plus "Ch 12/48 · 43%" and "Page 3/9" in paged mode;
  - icons ◀ prev chapter · ☰ **Chapters** · **Aa** · 🌐 **Translate** · ▶ next chapter.

**Modes** (from `library_core.plan_open_reader`, the desktop Book Details decision): plain · overlay (in-progress raw + translated response files; refreshed every 3 s only while a job for this book runs) · dual-path (compiled + raw) · PDF workspace · text (`.txt`, mobile `ui/reader/text_book.py`: the sections of a compiled `_translated.txt`, split on the translation's separator, or ~20k-character paragraph-bounded sections; decoded by BOM, else strict UTF-8, else the CJK code page chardet / charset_normalizer pick among the ones that read it: CP949, GB18030 / GBK, Shift-JIS / CP932, Big5, EUC-JP) · TXT workspace (the translation split's `word_count` sections + their translated response files paired by content hash; Original / Translated / Bilingual; an untranslated section shows "This section has not been translated yet." above its raw text). A saved position or bookmark from the other TXT section layout restores at the same book percent (`ReaderSession.position_from_pref`). The desktop opens TXT in an external editor (DISCREPANCIES "Device fixes 2026-10-08").

**Toggle availability.**
- **Original** and **Translated** are available when an overlay, dual path or raw workspace content exists.
- **Bilingual** is new: `reader_doc.build_bilingual_chapter(raw_html, translated_html)` interleaves blocks paragraph by paragraph. When the block counts diverge by more than 15%, it falls back to whole-section order (original, then translated). It is enabled only when both versions of the chapter exist.
- When switching, the position is kept with the proportional page hint.

**Chapters drawer** (the View's `end_drawer` `NavigationDrawer`; a 320 dp side panel on tablet):
- chapter list with progress status icons;
- "Native TOC" switch (TOC.txt → sidecar toc.ncx → EPUB toc.ncx);
- "Show special files".

**Aa sheet** (`BottomSheet`, ≤ 75% height, `barrier_color` transparent so the page previews live):
- `TabBar` **Text · Theme · Layout**.
- **Text:** font family (Embedded CSS / Serif / Sans / Mono / imported fonts) · size `Slider` with A− / A+ and a value pill (pt) · line spacing 1.0–3.0 · margins.
- **Pinch** on the page changes the Aa font size.
  - Page JS posts two-finger scale events; the native fallback uses `GestureDetector(on_scale_update)`.
  - Each step gives a `selection_click` haptic tick, and the value pill flashes.
- **Theme:** 6 swatch cards — Dark, Light, Sepia, Midnight, Forest, Rose (exact `READER_THEMES` colours) — and "Follow app theme".
- **Layout:** Single page / Scroll / Scroll all; Double page appears only on tablets in landscape. Also tap-zone paging on/off, keep screen on, show progress %.
- **Scope switch** at the top: **This book · All books**. "All books" writes `epub_reader_font_size`, `_line_spacing`, `_theme`, `_font_family`, `_layout`; "This book" overrides go to Prefs.
- Changes are applied with `run_javascript` (CSS variables), with no reload.

**Search.** A sheet with the field "Search across book…".
- Results stream in batches of 120 (`search_chapters` in io_pool, 650 ms debounce) as rows with chapter title + excerpt (±34 characters).
- Tap jumps and highlights.

**Selection.** Page JS posts the selected text, and Flet shows a floating `Row` of chips above the selection:
- **Copy**
- **Google Translate → {output_language}** (in Original mode) / **Define on web** (otherwise); opened through `UrlLauncher` in the in-app browser view
- **Add to glossary** (new; EntrySheet prefilled with the raw term)
- **Ask in chat** (new; opens a chat with the quoted text)

**Live "Translate this chapter"** (🌐):
1. If the chapter is completed, confirm "“{title}” is already translated…".
2. `mark_chapter_pending_for_retranslation`, then a SINGLE_CHAPTER job.
3. A native **LivePanel** opens: a half-height draggable sheet, *not* inside the WebView.
   - Status line "🛰️ Translating “f” — waiting for stream…".
   - Content is a `Markdown` / `Text` column fed by `LiveLineClassifier` (90 ms drain).
   - Buttons "🧠 Thinking (n)" (expands the thinking log, at the log size §6.2), "⏹ Stop", "✕ Hide" (the job continues; the 🌐 icon becomes "🛰️ Live view").
4. The outcome uses the exact strings:
   - "✅ Translation finished — loading the translated chapter…"
   - "⏹ Translation stopped — incomplete output cleared."
   - "⚠️ Translation did not complete — incomplete output cleared."
   
   Then the overlay is re-merged and the chapter reloads.

**Reading position**
- The Prefs key `reader_positions` in `mobile_state.json` (not a config.json key) = `{bid: {href, fraction, page, mode, updated}}`. It is saved with a 1 s debounce on each page turn and on pause.
- On open, if the saved position differs from the start, a snackbar offers "Resume at Ch 12 · 43%" · **Resume**.
- **Bookmarks:** ⋯ › Bookmarks lists them, with "Add bookmark here". They are stored in Prefs `reader_bookmarks` (`mobile_state.json`).

### 3.12 Refresh rules (Library, Book page)

- **Library:** a full scan with skeleton cards on first open. While the screen is visible and the app is in the foreground, a quiet `library_core.scan` runs every 2 s in io_pool and is diffed with `card_signature`:
  - only changed cards are replaced;
  - the grid rebuilds only on a structural change;
  - it skips while a scan or delete is running.
  
  `library_dirty` (set after jobs and imports) forces an immediate scan. Pull-to-refresh runs a full rescan.
- **Book page:** `progress_core.snapshot_signature` every 2 s on a worker thread, only while the page is visible. Rows are rebuilt off-thread and mutated in place by row key; batches of more than 96 changes are streamed. Pull-to-refresh does a full reconcile (`read_only=False`).
- **Paused** when the app is backgrounded. On resume, one immediate refresh.

---

## 4. Other surfaces

### 4.1 Glossaries (`/glossary`; drawer destination)

**Home.** A list of glossary files:
- book glossaries (`Glossary/<book>/…`), manual glossaries, the unified glossary;
- each row shows entries count, the book link and modified date.

Search, filter chips (Book · Manual · Unified), and an extended FAB "Import glossary".

File row ⋯ (`ActionSheet`): Open · Use as manual glossary · Share · Delete glossary files · Restore backup.
 App bar ⋯: **Extract glossary** · **Parallel EPUB pair** · **Unified glossary** · **Glossary progress**.

**Glossary view (`/glossary/<gid>`).** A scrollable `TabBar`: **Editor · General · Balanced/Full · Minimal · Refinement**. The settings tabs are global; they are reachable from any file and also at Settings › Glossary.

**Editor**
- **Top bar:** file name ▾ (switch file), ◀ ▶ (prev/next), and an "auto-reload on change" dot.
- **Toolbar:** Search · Filter (type, gender, has description, custom fields) · Undo · Redo · Save (with a dirty dot; Ctrl+S on keyboards).
- **List.** WindowedList rows (min 56 dp; they wrap at large text): **raw** (titleSmall) → translated (bodyMedium), a type badge, a gender chip, and ⚠ conflict.
  - Tap opens the **EntrySheet**: every column plus custom fields, and "Resolve gender…".
  - Swipe left deletes (`Dismissible` + undo).
  - Long-press selects; the bulk bar offers Delete / Export selection / Change type.
- **FAB:** "＋ Entry".
- **⋯ menu:**
  - Save As · Export selection · Backups (list → restore) · Edit raw (CodeEditor: `JSON` for .json, `PLAINTEXT` for .csv/.txt/.md) · Use as manual glossary (chat / next run)
  - switches: Update output files on save · Hide unused entries
  - **Advanced ›** Reload · Clean empty fields · Remove duplicates · Backup settings · Trim entries · Filter entries · Convert format · About format
  - Text size
- **Find / Replace sheet:** the desktop editor's Find / Replace dialog: find, replace with, a live preview ("n rows · n matches"), Find Next · Replace · Replace all. Matching is the desktop's (the shared `row_has_match` / `replace_in_row`): case-insensitive, in every column of the rows the view holds (all rows, or the used ones while Hide unused is on). There is no scope, Match case or Whole word option because the desktop dialog has none (amended in the U6 review; DISCREPANCIES "U6 review"). When nothing in the glossary matches, the dialog "No glossary match — apply to output HTML files?" offers Apply (undoable) or Cancel.
- **Tablet:** a fixed-column grid with horizontal scroll and the entry editor in the SidePanel.

**Settings tabs** (schema-generated; locks show as a purple 🔒 chip "Locked by mode: Minimal")
- **General:** 8 glossary modes, auto-mapping/fuzzy + slider, append, require 100%, skip retries, additional glossary, unified toggles, compression, precise matching + Configure, multipass exclude, log match differences + allowlist, backup in output, gender tracker, append format prompt.
- **Balanced/Full:** entry types (ListEditor), filtering, custom fields, duplicate detection + name-matching sub-options, output format, target language, prompt + profiles, extraction settings, anti-dup subpage, Single Pass header prompt, "Add minimal glossary pass".
- **Minimal**
- **Refinement**

Changes auto-save; ⋯ offers "Discard changes since opening".

**Parallel EPUB pair** (`/glossary/parallel-pair`; a multi-step full-screen View):
1. Pick the raw and translated EPUBs (SourcePicker or FilePicker).
2. Chapters are auto-mapped. Reopening the pair restores the saved mapping (`_parallel_epub_mapping.json`).
3. **Mapping list:**
   - toolbar: **Auto-offset** switch, **Re-map** (auto-map again), an offset −/+ stepper and validation counts;
   - tap a row to pick its translated file in a sheet (the desktop per-row re-map dropdown);
   - a selection offers **Set unmapped**.
4. **Pair wrapper prompt** (PromptTile):
   - placeholders `{raw_text}`, `{translated_text}`, `{raw_filename}`, `{translated_filename}`;
   - a profile Dropdown with New / Save / Reset / Delete;
   - keys `parallel_epub_glossary_profiles`, `parallel_epub_glossary_active_profile`, `parallel_epub_glossary_wrapper_prompt`.
5. **Accept** starts a glossary-only job.

**Unified glossary** (`/glossary/unified`): enable switch, "Exclude gendered active entries", settings, and **Rebuild now** (job).

### 4.2 Tools hub (`/tools`)

A `GridView` of 88 dp tool tiles (icon, name, last source), in groups:

| Group | Tools |
|---|---|
| Translate | Async batch · Review generator · Headers & metadata · RPG Maker |
| Check | QA Scanner · Progress manager · Glossary progress · SDLXLIFF reviewer |
| Build | Converter / Compile · Validate EPUB |
| Images | Manga translator |
| Files | File browser · Text editor |

Each tool picks its input through the **SourcePicker** sheet, with segments: Recent outputs · Library books · Chat workspaces · Browse.

### 4.3 Progress manager (standalone; `/tools/progress?out=<oid>`)

- The same `ChaptersPane` component as Book › Chapters.
- It is used for chat attachment workspaces, non-Library outputs, subtitle ZIPs, image folders and the parallel-pair context.
- Header:
  - an output `Dropdown` when there are several outputs;
  - the output-folder chip "📁 <folder>" (tap → Files);
  - a "📁 Created: <folder>" snackbar when the folder is created (as in §3.7).
- The Glossary Progress pane is at `/tools/progress/glossary`.

### 4.4 QA Scanner (`/tools/qa`)

- **Mode cards** (`RadioGroup` of 4 tonal cards with descriptions): **Quick Scan** (Recommended badge) · **Aggressive** · **AI Hunter** · **Custom** (opens the Custom Mode Settings sheet).
- **Fields:** "Quick Scan duplicate sample size" (-1 = all text, 0 = duplicate check off; on mobile 0 when nothing is saved, and a saved desktop 1000 is migrated to 0 once: `qa_model.migrate_quick_sample_size` at app start and when the screen opens, flag in Prefs `qa_quick_sample_size_mobile_default`; Settings › QA shows the same default, `schema_access.MOBILE_DISPLAY_DEFAULTS`); switch "Auto-search output".
- **Source:** the SourcePicker (multi-select gives a bulk scan). A name mismatch triggers a warning dialog. Direct Text workspaces are listed but not picked here; a chat scans its own workspace (§2.12.4).
- **Run:** a QA_SCAN job (JobCard in Jobs; also posted into the chat when started from chat: Quick Scan, same sample size).
- **Reports** list. It opens the **QA report viewer** (`/tools/qa/report/<rid>`, an `HtmlView`):
  - on Android and iOS, the HTML report in `flet_webview.WebView`, with per-issue "Open in Chapters" (deep link with filter);
  - on Windows/Linux dev, the html2text Markdown rendering plus "Open in browser".
- **Silent truncation.** The heuristic is always available. The embeddings method (sentence-transformers) is shown disabled with a ReasonChip, because it is native-impossible.
- "Settings" opens Settings › QA Scanner. The language multiplier grid is a 2-column NumberTile list. The phrase editors (emoticon whitelist, AI artifacts, thinking preambles) use ListEditor.

### 4.5 Converter / Compile / Headers & metadata

- **Converter (`/tools/convert`):**
  - Source picker and an options summary (links to Settings › EPUB output / PDF).
  - Buttons: **Compile EPUB** · **Compile PDF** · Validate EPUB structure · Rename files (retain extension) · Apply `<br>` → `<p>` to outputs · Generate MD · Generate TXT.
  - Result card: Share / Save / Open in Reader (EPUB and TXT) / Add to Library.
- **Headers & metadata (`/tools/headers`):** Translate headers now / Stop · Delete header files · Delete TOC files · Translate metadata (one or many EPUBs; mode from Settings).

### 4.6 Manga (`/tools/manga?tab=files|settings|editor`)

Entry points: Tools, ＋ › Manga, and the "Translate as manga" quick chip when images or a CBZ are attached.

**Files**
- Thumbnail list with these sources:
  - **Add files**;
  - **Add ZIP/CBZ**;
  - **Add folder**, on both platforms (`FilePicker.get_directory_path`). When Android SAF access fails, it offers "Pick a .zip instead".
- Sort (Name / Number / Date / Reverse) and drag reorder (`ReorderableListView`).
- Image range.
- **Process grouping**: splits first-level subfolders into groups.
- A per-file "Process this image" switch. The selection persists.
- Selection bar: Remove selected · Clear all.
- Start / Stop (state machine); progress; LogConsole.
- Create CBZ at end; Auto consolidate.
- Output actions: **Create CBZ** · **Download images** (saved to the device through the ExportSheet).

**Settings** (schema `manga_*`)
- **OCR provider.** Each provider row carries a status chip: Ready · Needs key · Model not downloaded · Downloading NN% · Not in this build.
  - Available:
    - Google Vision: google-cloud-vision when its wheels resolve, otherwise REST through `google_vision_rest`; credentials PathTile.
    - Azure Computer Vision.
    - Azure Document Intelligence: azure-ai-documentintelligence when its wheels resolve, otherwise its REST API.
    - custom-api LLM vision: Edit prompt, **Disable all thinking**.
    - RapidOCR: ONNX download. It ships when pyclipper and shapely wheels resolve; otherwise it is disabled with a ReasonChip.
  - Shown disabled with the ReasonChip "Needs PyTorch · not available on mobile": manga-ocr, Qwen2-VL, EasyOCR, DocTR, PaddleOCR. They are listed, never hidden.
- **Detection** (bubble detector).
  - RT-DETR ONNX `ModelDownloadRow`:
    - status: Not downloaded / Downloading NN% / Downloaded · size / Loaded;
    - actions: **Download**, **Load** / Unload, Delete;
    - Hugging Face download, with the urllib fallback.
  - RT-DETR PyTorch and YOLOv8 / custom detectors are shown disabled with a ReasonChip.
- **Context:**
  - full-page context + prompt, visual context, image quality, token limit;
  - a "Current translation settings" summary that replaces "Refresh from Main GUI".
- **Glossary workflow:** load / clear / auto-load / compress / debug subfolder.
- **Inpainting.** The method row has a status chip: Preloaded · Loading · Not downloaded · Needs key.
  - Methods:
    - Skip;
    - Replicate (key);
    - custom image edit: endpoint, **custom image edit prompt**, **batch image-edit requests**, Test;
    - local ONNX (aot, anime, lama), with a model download manager.
  - Shown disabled with a ReasonChip, not hidden:
    - Torch JIT and Hybrid: "Needs PyTorch";
    - ollama and sd_local: "Not functional on desktop either".
- **Rendering:**
  - background, font sizing, constraints, wrap, caps;
  - font style + import, colour / shadow;
  - presets Manga / Manhwa / Large Text, reset (the desktop values, measured off the UI loop; U9).
- **Inpainting (U9):** mask presets B&W Manga / Colored / Uniform; Local / API model ⓘ (Model Information) and "Import model file…"; Image keys next to the custom image edit Test.
- **Advanced:**
  - preprocessing, HD strategy, mask, OCR params, merging, batching/ROI;
  - debug, parallel panel, memory cap, experimental brush/eraser;
  - "ONNX conversion / quantization", shown disabled with a ReasonChip (it needs torch).
- **Manual edit.**

**Editor**
- A page strip, and a Source / Translated segmented control (side by side on tablet).
- **Pan mode:** `InteractiveViewer`.
- **Edit mode:** `Stack(Image, canvas.Canvas, GestureDetector(drag_interval=24))`, with tools Select/Move · Box · Circle · Lasso · Brush · Eraser.
  - Brush and Eraser are **Excluded** (disabled with a ReasonChip): the desktop's experimental mask painting stays desktop only.
  - Clear boxes (toolbar, confirm): the desktop Clear Boxes (`manga_editor_core._delete_translated_outputs` / `_clear_saved_page_state`, U9).
- **Workflow buttons:** Detect · Clean · Recognize · Translate · Translate all (MANGA_STEP jobs).
- **Long-press a box → BoxSheet**, with tabs "📝 OCR Recognition Result" and "🌍 Translation Result":
  - edit, Save, **Save & Update Overlay**, re-run OCR / translate, Delete.
- Import / Export OCR JSON; Open auto-saved OCR files; Files.
- Re-rendered images use versioned file names to avoid the `Image` LRU cache.

### 4.7 Async batch (`/tools/async`)

- Provider and model summary; note "50% off · results in up to 24 h".
- "Create batch from…" (SourcePicker + chapter selection).
- Job list (from `data/async_jobs`): status chip, Refresh / Poll, Download results, Apply results, Cancel.

### 4.8 Review generator (`/tools/review`)

- Book / output picker (SourcePicker).
- Mode chips: 50/50 split · Full review (chunked) · Wrap chunks · Volume mode. Volume mode opens a ReorderableListView for file order.
- **📝 Review system prompt:** PromptTile → PromptEditor, bound to `review_system_prompt`, with a `{target_lang}` placeholder chip and ↺ Reset to default.
- **📝 Final prompt:** PromptTile → PromptEditor.
- Start · Review all files · Stop (REVIEW jobs).
- **Output pane:** Markdown; Save / Delete / Restore.
  - Its ⋯ › **Display** sheet is bound to:
    - `review_font_family` (FontTile);
    - `review_font_size`, `review_line_height`;
    - `review_font_color` (ColorTile);
    - `review_header_spacing`, `review_spacing`, `review_list_gap` (NumberTiles);
    - plus **Reset**.
- A 📝 indicator appears on Book › Output when a review exists.

### 4.9 SDLXLIFF reviewer (`/tools/sdlxliff?out=`)

- Book / piece selector (`Dropdown` on phone, side list on tablet).
- Legend filter chips (status colours).
- **Row cards:** source (bodySmall, muted), then an editable output `TextField`, with a status colour bar. Saves follow the shared manual-editing semantics. A row saves on blur and only its own card (and the legend counts) is rebuilt, so the field being typed in keeps its focus and text; full re-renders (piece / book / filter / layout changes, an action's result) save typed text first.
- **Machine translation sheet:**
  - provider Auto / Google / DeepL / Bing / Yandex;
  - Argos is shown disabled with a ReasonChip ("Offline engine (ctranslate2) not available on mobile"), and the google-translate-free fallback chain skips it;
  - "Configure … API key" (Bing region, Yandex folder id);
  - MT preview; **Inject MT**.
- **Flag inaccurate** with "Set Score Threshold…" / "Reset Threshold".
- Piece ⋯: Mark as Completed / Undo; Edit Output.
- Refresh + sidecar regeneration; polling every 2 s while visible: on top of the stack (not under its own Edit Output editor) and the app in the foreground (no reload while a field has focus or unsaved text, or the Notepad document is edited).
- Edits are never dropped (the dialog's `closeEvent` captures the Notepad HTML and flushes its queued edits): leaving the reviewer saves an edited Notepad document and typed row text, and a re-render (piece / book / filter / layout / refresh / an action) saves them first.
- **Notepad layout:** CodeEditor (`CodeLanguage.XML`), tablets only; the document is built on the io pool. On phones the layout toggle is shown disabled with a ReasonChip "Tablet layout only".

### 4.10 RPG Maker, File browser, Text editor

- **RPG Maker (`/tools/rpgmaker`):**
  - Pick a game folder (`FilePicker.get_directory_path`) or a ZIP, instead of the desktop `.exe` Browse filter.
    The shared entry `owner._register_rpgmaker_game_input(source, work_dir)` (`rpgmaker_job.prepare_rpgmaker_game`) makes the game writable (a ZIP is extracted once into `<work>/zips/<name>-<hash of its contents>`, a read-only folder copied once into `<work>/folders/<name>-<hash of its path>/`; each records its source in `.glossarion_rpg_source.json` and is reused only for that source, so another game with the same file name never resumes into it; resume progress stays in `<game>/GTool_Translation`) and registers it, then the job runs the desktop worker on `[game_dir]`, which dispatches it like a game `.exe`. Scan and Translate wait for a running scan (the shared entry also serialises them per folder).
  - The GTool scan prompt (PromptTile).
  - Run.
- **FileBrowser (`/tools/files/<root>[/<fid>]`):**
  - Breadcrumbs. Roots: Output, Library, Inbox, Chat workspaces, or a workspace `oid`.
  - Open with: Reader / image viewer / TextEditor / MediaViewer. As built: Text editor (text files) · Reader (books) · Media viewer (images, video, audio: the chat's full-screen MediaViewer with Share / Save to… / Open externally) · Another app… (Share).
  - Share · Save to… · Rename · Delete (safe-root guard).
  - It replaces Open Output Folder, Reveal and Open Library Folder.
- **TextEditor (`/tools/text/<fid>?hit=<n>`):** `flet_code_editor.CodeEditor`, with the language chosen by extension. The package has no HTML or CSV language, so:

  | Extension | Language |
  |---|---|
  | html, xhtml, xml, sdlxliff | `XML` |
  | css | `CSS` |
  | json | `JSON` |
  | md | `MARKDOWN` |
  | csv, txt, srt, ass | `PLAINTEXT` |

  - It adds find, jump-to-term and a read-only mode. The term arrives in-process, never in the route.
  - Unsaved edits ask "Unsaved changes" (Save / Discard / Cancel) before the screen is left: Back (Android back, the iOS swipe, the app-bar arrow and the tablet main-area back; the View cannot pop while the text is dirty) and every other navigation (the shell's leave guard) share the one question.
  - A plain `TextField` is the fallback.
  - It replaces Notepad / Notepad++.

### 4.11 Model picker and Model Manager

- **Picker:** ModelSheet (§2.2). The same component, in field mode (`ModelPicker`), is used in KeyEditor, Manga, Glossary and QA AI-Hunter settings.
- **Model Manager (`/settings/models`):** `TabBar` **Models · Custom prefixes**.
  - **Models:**
    - search; chips Polled only (`hide_unpolled`), Custom, Removed;
    - "🌐 Poll providers" with a per-provider status chip row;
    - a `ReorderableListView` windowed by provider group. Rows: name, provider badge, ✓.
    - Swipe removes (tombstone). In Removed, swipe restores. FAB "Add model".
  - **Custom prefixes:** route list; an add/edit sheet with prefix, endpoint type, base URL, key and model override, validated by the shared normalizer.

### 4.12 Multi-Key Manager (`/settings/keys`, `/settings/keys/<pool>`)

- **Rotation card:** force rotation, frequency.
- **Pool chips** (scrolling `Chip` row on phone, left list on tablet). There are 11 pools:
  - Translation (main) · Fallback (prohibited content; use main key fallback, shuffle) · Glossary · Glossary refinement · QA/Vision · Metadata · AI truncation detection · Rolling summary · Truncation retry · Inpainter · TTS.
  - Each pool shows an enable switch, a description and a count.
  - The route `/settings/keys/<pool>` opens a pool directly.
- **Pool ⋯** (`PopupMenuButton`):
  - **Clear all keys** (ConfirmDialog "Remove all N keys from <pool>?");
  - Import into this pool;
  - Export this pool.
- **App bar ⋯:**
  - Import / Export all pools;
  - Refusal patterns;
  - two rows shown disabled with a ReasonChip, values preserved:
    - "Lock mouse wheel": a desktop scroll guard;
    - "Key list zoom": mobile follows the app text scale.
- **Key cards.** They sit in a `ReorderableListView` with `ReorderableDragHandle`s; the order is the rotation order (desktop drag reorder).
  - Each card shows:
    - the masked key `sk-…a1B2` and the model;
    - a status chip: Active / Cooling (mm:ss) / Disabled / Passed / Failed;
    - success/error counts and context badges.
  - Tap opens **KeyEditor** (full screen):
    - key and ModelPicker;
    - cooldown 10–3600, limit 0–2M, temperature −1…1, output token limit, delay 0–3600, enabled;
    - individual endpoint (URL, Azure version); Google creds + region;
    - 🧩 request parameters (key/value);
    - request contexts (tri-state chips).
  - Long-press enters multi-select. Bulk actions: enable / disable / test / remove / set contexts / move / copy to pool.
- **Footer:** ＋ Add key · Copy current key · Test selected · Test all (live per-key results) · Import / Export (glossarion-key-pools v1) · Refusal patterns.
- The logic lives in `key_pool_service`. `apply_key_pools_to_runtime` runs at job start.
- **Entry points:** KeyPoolTiles inside settings sections open `/settings/keys/<pool>` directly. They replace the desktop one-pool preview windows.

### 4.13 Accounts / OAuth (`/settings/accounts`)

- **Provider cards: ChatGPT · Grok · Claude · Gemini.**
  - ChatGPT comes first and ships in U3, as a minimal sheet (sign in / out, status), because the default model is `authgpt/gpt-6-luna`.
  - Slots, rotation and the other providers follow in U4.
  - Each has slot rows: "#N · email · ✓ status · refreshed 2 h ago", with ⋯ offering Re-login / Log out / 📊 Status (Gemini).
  - "＋ Add account" uses the next free slot. A rotation note explains that `authgpt0/` / `authgrok0/` / `authgem-vertex0/` use all slots.
  - Gemini also has a GCP project picker and a verification-URL action.
- **LoginSheet** (OAuthBridge): steps Opening browser → Waiting for sign-in → Exchanging token → Done, with **Reopen browser** · **Paste redirect URL / code** · **Cancel**. Claude also exposes the manual paste-code flow.
- **Experimental (U9, best effort):** AuthND (NVIDIA Build) and Gemini Free (search/gemini), through the off-screen WebViewBridge. On platforms without flet-webview (Windows/Linux dev) the rows are disabled with a ReasonChip.
- **"Unavailable on mobile" section.** It is always listed, as an `ExpansionTile` with its count, expanded on the first visit. Each entry is a disabled row with its ReasonChip: Antigravity, OCAGY, OpenCode Zen (ocz/), Z.AI login, Arena, Opera Aria, Tor, Managed Ollama (ollamapull/), Claude Code CLI import, Grok CLI import.

### 4.14 Profiles and prompts

- **`/settings/profiles`:** a list of prompt profiles (built-in badge, modified dot), a FAB "New profile", and ⋯ Import / Export (FilePicker / Share).
- **`/settings/profiles/<pid>`:** a `PromptEditor` (full-screen mono, token count, placeholder chips such as `{split_marker_instruction}` and `{glossary_prompt}`), a System ⇄ User role toggle, and an extraction-override section when the profile defines one. Actions: Save · Save as · Reset to default · Delete.
- **`/settings/prefill`:** Assistant prefill (Asst. Prompt) profiles: list, editor, enable.
- **"All prompts" index:** a searchable list of every PromptTile in the schema, each linking into its section. It covers:
  - Refine prompt; Full + raw prompts / header / footer
  - Configure All translation prompts; title prompt; metadata prompts
  - chunk prompt; image chunk prompt; GTool scan prompt; Vision OCR prompt
  - memory prompts; glossary prompts; QA AI-Hunter prompts; manga OCR / full-page prompts
  - Review system prompt; Review final prompt; custom image edit prompt
  - Parallel EPUB pair wrapper prompt (and its profiles)

### 4.15 Settings (`/settings`; schema-driven, searchable; opened from the drawer footer)

**Home**
- **Search.** A `SearchBar` "Search settings" at the top.
  - It searches label, help, key, env names, `search_terms` and the section path.
  - Results are grouped (max 50), with breadcrumbs.
  - Tapping a result opens `/settings/s/<id>#<key>`. After the SectionPage's first frame it scrolls to the tile with `scroll_to(scroll_key=ft.ScrollKey(key))` and highlights it for 1.5 s.
- **Filter chips:** Modified · Locked · Advanced · Unavailable on mobile.
- **Quick-action chips:** the desktop toolbar items that touch settings.
  - **Save now**;
  - **Backup**: Data › Backup & restore › Create;
  - **Import / Export profiles**.

**Groups and sections.** Each section is a `SectionPage` built from `SettingSpec`.

| Group | Sections (id) |
|---|---|
| General | Appearance (`appearance`: theme mode, accent Halgakos Rose / Desktop Blue / Library Violet, AMOLED dark, text scale 85–130%, reduce motion, haptics; "Auto DPI / GUI scale" shown disabled with a ReasonChip) · Notifications & background (`notifications`) |
| Translation | Defaults (`translation_defaults`: target language, **Output mode** = the global `output_mode` default for book jobs, temperature, token limits, chunk size, pacing = threading delay / API delay / preflight, batch translation, multipass + refine prompt, remove AI artifacts) · Profiles & prompts (custom) · Chat defaults (`direct_text`) · Context & memory (`context_memory`) · Response handling & retries (`response_handling`; Tor proxy, Tor rotation and "GUI Responsiveness Yield" shown disabled with ReasonChips) · Processing & extraction (`processing`) · Metadata, TOC & headers (`metadata_toc_headers`) · EPUB output (`epub_output`) · PDF (`pdf`) · Image & vision (`image_vision`: output-mode sub-settings, with a link "Default output mode → Translation defaults") · Anti-duplicate (`anti_duplicate`) |
| Models & keys | Model Manager · Multi-Key Manager · Accounts · Endpoints (`endpoints`: includes the Ollama / LM Studio LAN host and Test connection; Gemini gRPC transport and Vertex follow the dependency rule; "AuthZA / GLM access mode" and "Load Ollama" shown disabled with ReasonChips) · Thinking & reasoning (`thinking`) · Provider options & safety (`provider_options`: API safety filters, OpenRouter options, service tier) |
| Glossary | General · Balanced/Full · Minimal · Refinement · Unified (`glossary_*`; the same pages as the Glossaries tabs) |
| QA | QA Scanner (`qa_scanner`) |
| Manga | `manga_*` |
| Reader & Library | `reader` · `library` |
| Data | Storage · Backup & restore · Import from desktop · Logs & diagnostics |
| About | Updates · About · User guide · Welcome guide · Danger zone |

*Implementation (U9).* The curated sections are a mobile layer over the desktop schema sections
(`ui/settings/model.CURATED_SECTIONS`, applied by `SchemaAccess.sections()`): `translation_defaults`,
`context_memory`, `epub_output`, `pdf`, `thinking` and `provider_options` take their listed keys out
of the desktop sections, with their own sub-headings (Pacing, Multipass, Rolling summary, Input /
Output, …); `other.response` keeps its id and is regrouped under Streaming / Retries / Truncation /
Duplicates / Failure saving / HTTP / …. The remaining desktop sections keep their ids with the
mobile titles (`other.meta_data` "Metadata, TOC & headers", `other.processing` "Processing &
extraction", `other.processing.extraction` "Chapter extraction", `other.image` "Image & vision").
A desktop id the move emptied (`main.run`, `other.output`) resolves to the section that took its
keys, and `open_setting(section, key)` follows the key. KeyPoolTiles (§4.12) sit at the end of
Context & memory (rolling summary), Response handling (translation, truncation retry), Metadata
(metadata), Provider options (fallback), QA Scanner (AI truncation) and the Glossary Refinement tab.

**Notifications & background** (`/settings/notifications`):
- **Notifications:** **Allow notifications** (the same request as the first Run). The status line shows the real state, read each time the page opens: "Notifications: On" / "Off: Allow notifications asks again" / "Blocked in system settings: open them to turn notifications on". On Android it comes from `post_notifications_granted` / `notifications_enabled` plus the permission status; on iOS from the authorisation status. **Open system settings** appears while notifications are off or blocked. **Send a test notification** posts one on `jobs.done` and reports "Test notification sent." or "The system did not show it: notifications are off or blocked for Glossarion." Then the three channel rows.
- **Glossary review:** switch "Always accept generated glossaries". It is the global value in Prefs `chat_auto_accept_glossary`, default off, and a chat can override it in its settings sheet (§2.14). Help text: "Skip the Edit / Yes / No card: translation starts with the generated glossary. Applies to new sends; a chat can change it in its settings. The desktop always asks."
- **Background:** Disable battery optimisation (Android), Keep the screen on while a job runs, and the iOS background note.

**SectionPage.** It is a `ListView(build_controls_on_demand=False)`. A section has at most about 80 tiles, so every tile is built and is a valid `ScrollKey` target.

**Tiles** (§5.7):
- Each tile has a lock badge with its reason, a "modified" dot, and long-press "Reset to default".
- Some fields are **always shown, disabled, with a ReasonChip**:
  - fields whose `platforms` exclude mobile;
  - fields whose dependency is not in this build.
  - There is no "show desktop-only" switch. Their values round-trip untouched, and the "Unavailable on mobile" filter chip lists them.
- Editing auto-saves with a 600 ms debounce. App bar ⋯: "Save now" · "Discard changes since opening".
- While a job runs, a `Banner` says "Changes apply to the next run".

### 4.16 Data, backup, logs, diagnostics, updates, about

- **Storage:**
  - paths: Output, Library, Inbox, cache;
  - per-folder usage bars; clear caches (reader / covers / temp); ONNX models manager;
  - output folder choice (app storage or iOS Files-visible Documents; arbitrary SAF folders are not supported);
  - Android "Mirror outputs to Downloads/Glossarion" (every finished output is copied there through MediaStore).
- **Backup & restore:**
  - automatic backups list (72 h retention, shared); Create backup / Restore (confirm) / Delete;
  - Export config (keys excluded, or encrypted with a passphrase) / Import config;
  - profiles import/export; key pools import/export.
- **Import from desktop:** config.json (with optional `.glossarion_key` to decrypt `ENC:` values; otherwise "re-enter keys"), Direct Text chats ZIP, Library ZIP.
- **Logs & diagnostics:**
  - live log (LogConsole) and log files list;
  - Debug mode; Check environment (redacted); HTTP logging; Save payloads; memory stats;
  - **Run self-test** (`/__selftest__?suite=smoke`), with a PASS/FAIL result card;
  - crash and freeze logs; "Share logs bundle" (secrets redacted); a previous-crash banner.
- **Updates:** Check now · Check on startup · Skip version · release notes · "Download APK" (external browser) / "Open AltStore source". There is no self-install.
- **About:** version, build, Python / Flet versions, licenses, links, the in-app User guide (bundled docs Markdown), and the mascot.
- **Danger zone:** "Reset settings to defaults". It confirms and creates an automatic backup first.

### 4.17 First-run welcome (`/welcome`; `PageView`, 5 steps; re-runnable from About)

1. **Welcome · Sign in with ChatGPT (U3).**
   - Halgakos and "Translate novels, manga and documents with your own AI accounts and keys".
   - The default model is `authgpt/gpt-6-luna` (the desktop default), so the primary `FilledButton` is **Sign in with ChatGPT**. It opens the LoginSheet through OAuthBridge:
     - loopback on localhost:1455, opened in Custom Tabs / SFSafariViewController;
     - return link `glossarion://app/oauth/return`;
     - paste-redirect fallback.
   - Secondary actions: "Use an API key or another provider" (→ step 2) and "Skip for now".
2. **Other providers** (optional; the extra logins arrive in U4):
   - paste an API key (KeyField + Test), or sign in with Claude / Gemini / Grok;
   - or Local: an Ollama / LM Studio host URL;
   - "Skip".
3. **Target language and glossary mode.** The glossary-mode cards reuse the desktop welcome copy.
4. **Permissions.** Notifications; battery optimization (Android); the iOS background limits note.
5. **Done.** Try-chips: "Paste text to translate" · "Import a book" · "Open Library".

If sign-in is skipped, Send stays in the `blocked` state, with the "Sign in with ChatGPT" fix action (§2.4), until an account or another model is set.

---

## 5. Component catalogue

This section defines every component named in §1–§4 and in `FEATURE_MAP.md`. For each one it gives the anatomy (what it contains), the states it renders, and the Flet 1.0.3 controls it is built from.

Modules live under `ui/` (Appendix A). Components used by more than one surface live in `ui/components/`.

### 5.0 Build conventions (Flet 1.0.3)

- **Custom controls.** Components are `@ft.control` dataclass subclasses of a Flet control, usually `ft.Container`, `ft.Column` or `ft.Row`. Constructor fields are the props.
  - Stateful parts subscribe to `state/` Signals.
  - All mutation runs on the UI loop through `UiDispatcher`. Never call `update()` from a worker thread.
- **48 dp targets.** Give `IconButton` `size_constraints` of 48 × 48 (`BoxConstraints`), or wrap a smaller visual in a transparent `Container(padding=…)`.
- **Long-press.** Use the control's own event where one exists: `ListTile.on_long_press`, `Container.on_long_press`, `IconButton.on_long_press`. Otherwise wrap the control in `GestureDetector(on_long_press_start=…)`.
  - `Chip` has no long-press event. It has only `on_click`, `on_select` and `on_delete`, and `on_click` and `on_select` are mutually exclusive.
- **Menus.**
  - A menu behind a visible ⋯ button is a `PopupMenuButton`.
  - A menu that must open from code or from a long-press is a `ContextMenu`, with `primary_trigger=ContextMenuTrigger.LONG_PRESS` or an explicit `await menu.open()`. `PopupMenuButton` has no `open()`.
  - Longer action lists use `ActionSheet` (§5.2). Flet 1.0.3 has no Material action sheet (only `CupertinoActionSheet`) and no anchored popover.
- **Scroll targets.**
  - A jump target carries `key=ft.ScrollKey(<id>)`. A plain string key becomes a `ValueKey`, which is not a scroll target.
  - `scroll_to(scroll_key=…)` only reaches items that are already built, and it requires `auto_scroll=False`. Every jump therefore first makes sure the target is inside the rendered window (§2.8 procedure).
- **Variable-height lists** never set `first_item_prototype` or `item_extent`. Only a flat list whose items all have the same height by construction (no section headers) may set them, such as the ModelSheet search-results list.
- **Accessibility.**
  - `IconButton` has `tooltip` but no `semantics_label`. When the spoken label must differ, wrap it in `Semantics(label=…, button=True)`.
  - Status is always icon + text + colour.
- **Colours.**
  - Theme roles come from `ft.Colors.*`.
  - Semantic and status colours are app constants in `ui/theme/colors.py`, keyed by brightness, because Flet themes have no custom roles.
- **Dialogs and sheets** (`BottomSheet`, `AlertDialog`, `SnackBar`, `Banner`) open with `page.show_dialog(…)` and close with `components.dialogs.close_dialog(page, dialog)`.
  - Never close a known dialog with `page.pop_dialog()`. It closes the most recently opened dialog, which after a handler has shown a snackbar is the snackbar, not the sheet.
  - `BottomSheet(scrollable=True)` only lifts the 9/16 height cap; it never scrolls. A sheet whose content can outgrow a phone builds its body with `components.sheet` (a scrolling column, an optional pinned action row, the bottom system inset).
- **Extensions.**
  - `flet_webview.WebView` raises outside Android, iOS and macOS. Every WebView surface therefore has a native fallback for Windows/Linux dev.
  - `flet_video.Video` uses its default controls (there is no `show_controls`) and `aspect_ratio`.
  - `flet_code_editor.CodeEditor` languages:
    - `MARKDOWN`;
    - `XML` for html, xhtml and sdlxliff;
    - `JSON`, `CSS`;
    - `PLAINTEXT` for csv and txt.
  - `flet_audio.Audio` is a `Service`.
  - `TextStyle` has no `font_features`.

### 5.1 Shell and navigation

| Component (module) | Anatomy | States | Built from |
|---|---|---|---|
| `AppShell` (`shell/app_shell.py`) | Phone: a `page.views` stack with the chat root View. Tablet: one View with `Row[Sidebar, MainArea, SidePanel?]` | size class phone / large phone / tablet / wide; rebuilt only when the class changes | `View`, `Row`, `Container`, `SafeArea`, `page.on_resize` |
| `Router` (`shell/router.py`) | Whitelist parser for paths and `glossarion://app/` URIs; a per-destination back stack on tablet | known route · ignored route · (backend not ready: the screen opens at once; actions that need the engine wait on "Preparing engine…") | `page.on_route_change`, `page.on_view_pop`, `page.push_route` |
| ~~`BootView`~~ (adopted adaptation, U9) | Not built: the native splash (Halgakos) shows until Flet draws, then the shell mounts immediately. Until the warm import ends Send is `blocked` "Preparing engine…" and the drawer status chip says so; keys that could not be decrypted show the Settings home notice (→ re-enter in Keys) | — | native splash, `SendButton` blocked state, drawer status chip |
| `ChatDrawer` / `Sidebar` (`shell/drawer.py`, `sidebar.py`) | Header (avatar, New chat, New scratch chat) · SearchBar · destination chips · Pinned · Series (U9) · Recents · footer (status chip, Settings, Help) | closed · open · searching (results replace the body) · sidebar (tablet, persistent) | `NavigationDrawer(controls=…)` on phone (Appendix C item 3); a `Container` column on tablet; `SearchBar`, `Chip`, `Control.badge`, `ExpansionTile`, `ListView(build_controls_on_demand=True)` for Recents (group headers are interleaved, so no prototype), `ListTile(on_long_press=…)` |
| `SidePanel` (`shell/side_panel.py`) | 380 dp right panel: title row (title, pin, ✕) and swappable content | hidden · open · pinned (wide) | `Container(width=380)`, `AnimatedSwitcher` |
| `JobStrip` (`shell/job_strip.py`) | 44 dp strip: `ProgressRing` with the kind icon · title + subtitle · queued badge · Stop | running · finishing · stopping · done (10 s) · failed (10 s) · hidden · dismissed until the next state change | `Container`, `ProgressRing`, `Text`, `IconButton`, `Control.badge`, `Dismissible`, `Semantics(live_region=True)` |
| `LaunchBanner` (`shell/launch_banner.py`) | "N interrupted jobs" · **Resume** (most recent) · **Review** (→ `/jobs`) · ✕ | shown once per launch when `jobs/active.state` holds interrupted jobs | `Banner` via `page.show_dialog` |
| `ChatHeader` (`chat/header.py`) | ☰ · title over subtitle spans (model · profile · → target ▾ · "custom") · scratch toggle · New chat · ⋯ | rest · scrolled (manual tint) · scratch · search mode (title becomes a TextField) · text scale ≥ 160% (model span only) | `AppBar(bgcolor=…, elevation_on_scroll=0)`; the tint is a `bgcolor` swap driven by `Transcript.on_scroll` (there is no `surface_tint`); `Text(spans=[TextSpan(on_click=…)])`; `PopupMenuButton` for ⋯; `GestureDetector` for title long-press |
| `PullToRefresh` (`components/pull_to_refresh.py`) | A thin `ProgressBar` above a list; triggers when the user over-scrolls at the top | idle · armed (haptic) · refreshing | `ListView.on_scroll` overscroll events (to verify, Appendix C item 1); fallback: ⋯ Refresh only |
| `MasterDetail` (`components/master_detail.py`) | List pane + detail pane at ≥ 1200 dp | single pane (narrow) · two panes | `Row`, `Container`, `AnimatedSwitcher` |
| `WelcomeFlow` (`settings/welcome.py`) | 5 pages (§4.17), page dots, Back / Next | per step · signing in · skipped | `View`, `PageView`, `FilledButton`, `KeyField`, `flet_permission_handler.PermissionHandler` |

### 5.2 Decisions, feedback and empty/loading states

| Component (module) | Anatomy | States | Built from |
|---|---|---|---|
| `ActionSheet` (`components/action_sheet.py`) | Optional title + subtitle; rows of icon + label (destructive rows in the error colour); a Cancel row | open · item pressed (the sheet closes, then the action runs) · items disabled with a ReasonChip | custom: `BottomSheet(show_drag_handle=True, scrollable=True)` + `ListTile`s. On tablet the same content sits in a centered `AlertDialog` (≤ 560 dp) |
| `ConfirmDialog` (`components/dialogs.py`) | Title, body (the verbatim desktop text when one exists), optional item list, 2–3 buttons (destructive one in the error colour) | idle · running (buttons disabled, ProgressRing) | `AlertDialog`, `TextButton`, `FilledButton` |
| `TextPromptDialog` (`components/dialogs.py`) | Title, label, TextField, validation line, Cancel / OK | valid · invalid (OK disabled, message shown) | `AlertDialog`, `TextField(autofocus=True)` |
| `UndoSnackBar` (`components/dialogs.py`) | Message + **Undo**, 6 s | visible · undone · committed | `SnackBar(action=…, duration=6000)`; the deferred commit is persisted in Prefs `pending_deletes` |
| `ErrorCard` (`components/error_card.py`) | Error icon · bold title · mono `ExcType: msg` (selectable; 6 lines, then expand) · actions (Retry / Copy error / View log / fix action) | error · retrying | `Card`, `Icon`, `Text(selectable=True)`, `FilledTonalButton` |
| `EmptyState` (`components/empty_state.py`) | Icon or Halgakos (64 dp) · title · body · primary and optional secondary action | — | `Column`, `Image`, `Text`, `FilledTonalButton` |
| `Skeleton` (`components/skeleton.py`) | Shimmer blocks shaped like the content | loading · reduce-motion (static tint) | `Shimmer` around `Container`s |
| `ReasonChip` (`components/reason_chip.py`) | Small outlined chip: icon + short reason ("Not on mobile", "Needs grpcio", "Tablet only"). Tap opens an `InfoSheet` with the full reason and the preserved config value | — | `Chip(leading=Icon(info_outline), on_click=…)` |
| `InfoSheet` (`components/info_sheet.py`) | Title + Markdown body (schema help, bundled docs, reasons) | — | `BottomSheet`, `Markdown` |
| Page banners | Page-level notices: "Changes apply to the next run", "Saved to a temporary folder", "Progress file was deleted…", previous-crash notice | shown · dismissed | `Banner` via `page.show_dialog` |

### 5.3 Chips, status and identity

| Component (module) | Anatomy | States | Built from |
|---|---|---|---|
| `FileChip` (`components/file_chip.py`) | Type icon · name (middle ellipsis) · meta ("EPUB · 1.2 MB · 48 ch") · trailing × and/or ⋯ | loading meta (shimmer) · ready · missing (error tint, "Attachment missing") · disabled | `Container` (radius 8; 28 dp visual in the composer, 32 dp elsewhere; 48 dp target) + `Row[Icon, Column[Text, Text]]` + `IconButton`. Tap is `Container.on_click`; the menu is a ⋯ `PopupMenuButton` |
| `PastedTextChip` (`components/file_chip.py`) | "Pasted text · 12,345 chars · ≈3.1k tokens" + ⋯ (Show in text field · Save as .txt attachment · Remove) | counting · ready | same as FileChip |
| `StatusAvatar` (`components/status.py`) | 32 dp circle with a Material icon on a 16% tint of the status colour; the emoji is the semantics label | one per `progress_core.present` status | `Container(shape=CIRCLE)` or `CircleAvatar`, `Icon`, `Semantics(label="Chapter 12, QA Failed")` |
| `StatusChip` (`components/status.py`) | Icon + label + count in the status colour | selected (filter on) · unselected · zero (hidden unless it is a pinned group) | `Chip(leading=…, selected=…, on_select=…)` |
| `StatusChipRow` (`components/status.py`) | StatusChips + "Total: N" | scrolling row · wraps at ≥ 160% | `Row(scroll=AUTO)` or `Row(wrap=True)`; each chip wrapped in `GestureDetector(on_long_press_start=jump_next)` |
| `CountBadge` | Numeric badge on icons and chips | hidden at 0 | `Control.badge` (`Badge(label=…)`) |
| `ProgressStrip` (`components/status.py`) | Linear progress + "d/t · NN%" | indeterminate · determinate · done | `ProgressBar`, `Text` |
| `ModeBadge` (`components/status.py`) | "Mode: Text" on the Book page and in Chapters headers | one per output mode | `Container` + `Text` |
| `LoginChip` (`settings/accounts.py`) | Provider icon · "ChatGPT #2 ✓" or "Sign in with ChatGPT" · slot ▾ | signed in · expired (warning) · signed out · signing in (ProgressRing) · unavailable (ReasonChip) | `Chip(on_click=…)` opens the LoginSheet; slot `PopupMenuButton` ("#1 ▾ / + Add account") |

### 5.4 Chat

| Component (module) | Anatomy | States | Built from |
|---|---|---|---|
| `Composer` (`chat/composer.py`) | Radius-24 tonal card: chips row · TextField · action row (＋ · `OutputModeRow` · option pills · token hint · SendStopButton); the output mode has no line of its own | empty · typing · attachment · pasted chip · blocked caption · running · ≥ 160% text ("Options (n)", no token hint, max_lines 4) | `Container(border_radius=24)`, `Column`, `Row(scroll=AUTO)`, `TextField(multiline=True, min_lines=1, max_lines=6, shift_enter=True)`, `Chip(on_click=…, on_delete=…)` |
| `OutputModeRow` (`chat/output_mode_row.py`) | Composer (`inline=True`, in the action row): the active mode as a chip (emoji + ▾) with a menu of the six modes and "Options for <Mode>…", or six inline toggles 📝 👁️ 🖼️ 🎬 🔊 ✨, without the "Output:" label. ＋ sheet: "Output: Text" label + six toggles, on its own line | selected mode · "· auto" (label in the sheet, a dot + tooltip inline) · chip < 600 dp chat column or ≥ 160% text · icon toggles from 600 dp · text labels once they fit (§2.3) | `Row(tight=True)` inline. The chip is a `PopupMenuButton` (opens on tap) whose 48 dp content holds the 32 dp pill; menu items are `PopupMenuItem(checked=…)`; `Semantics(label="Output mode: Text", button=True)`. Each toggle is `IconButton(icon=Text(emoji), selected=…, tooltip=…, size_constraints=48×48)`; with text labels a 48 dp `Container(on_click=…)` holds the emoji + label pill; `Semantics(selected=…)`. The "· auto" dot is a `Badge` |
| `ModeOptionsSheet` (`chat/mode_options_sheet.py`) | "Output: <mode>" title · "This chat only" switch · the mode's schema tiles · (Image/Video/Audio) "Generate from prompt (no input)" | per mode · Generate disabled with a reason (empty composer / attachment present) | `BottomSheet(scrollable=True)`, setting tiles (§5.7), `FilledTonalButton` |
| `SendStopButton` (`chat/send_button.py`) | 40 dp circle (48 dp target), icon by state, long-press menu | `idle_empty` · `idle_ready` · `queue` · `blocked` · `running` · `finishing` · `stopping` (§2.4) | `AnimatedSwitcher` over `FilledIconButton` / `IconButton` / `ProgressRing`; `ContextMenu(primary_trigger=ContextMenuTrigger.LONG_PRESS, primary_items=[PopupMenuItem…])`; `HapticFeedback` |
| `PlusSheet` (`sheets/plus_sheet.py`; `chat/plus_sheet.py` re-exports it) | Attach tiles · OutputModeRow + active options · Tools list · This chat | — | `BottomSheet(show_drag_handle=True, draggable=True, scrollable=True)`; tiles are `Container(on_click, on_long_press)`; `ListTile`; `FilePicker` |
| `SlashCommandPopover` (`chat/slash.py`) | ≤ 6 matching commands above the composer | hidden · open · no match | `Container` in the root `Stack`, `ListView`, `ListTile` |
| `QuickActionChips` (`chat/quick_chips.py`) | ≤ 5 contextual chips | per context · dismissed | `Row(scroll=AUTO)`, `Chip(on_click=…)` |
| `Transcript` (`chat/transcript.py`) | Rendered window of message cards + loader rows + "↓ new" FAB | loading · window · streaming tail · scrolled up (FAB + badge) | `ListView(build_controls_on_demand=False, on_scroll=…)` over a Python window; items keyed `ft.ScrollKey(mid)`; `FloatingActionButton(mini=True)`; `Shimmer` |
| `UserBubble` (`chat/messages.py`) | Right-aligned bubble; collapses past 12 lines | collapsed · expanded | `Container`, `Text`; long-press `GestureDetector` → ActionSheet. "Select text" opens a `SelectableTextSheet` (no `SelectionArea` on the bubble, because touch long-press would start a selection) |
| `UserFileCard` (`chat/messages.py`) | Type icon, name, "EXT · size", role label + prompt | ready · missing file | `Container(on_click=…, on_long_press=…)` |
| `AssistantMessage` (`chat/messages.py`) | Header row · ThinkingDisclosure · content · media · MessageActionsRow | pending · streaming · done · long (truncated + "Show full translation") · missing body · error | `Column`, `Row`, `CircleAvatar`, `Markdown(selectable=True, extension_set=GITHUB_WEB)` |
| `ThinkingDisclosure` (`chat/thinking.py`) | "▸ Thinking (N tokens)" row → mono body (the last 50,000 chars) | live (shimmer) · collapsed summary · expanded · no stream | `GestureDetector` + `AnimatedSwitcher`, `Shimmer`, `Markdown` in a mono `Container` |
| `MessageActionsRow` (`chat/messages.py`, the card's action row) | Copy · Retranslate · Show source · Share · ⋯ (18 dp icons, 48 dp targets) + VersionSwitcher | visible · auto-folded (older replies) · copied (✓ for 1.6 s) | `Row`, `IconButton(tooltip=…)`, `Clipboard`, `Share` |
| `MessageMoreSheet` (`chat/chat_view.py` `_message_more`) | ActionSheet with the §2.10 items | items disabled with a reason when their file is missing | `ActionSheet` |
| `RefinementChips` (`chat/actions.py`) | More natural · More literal · Keep honorifics · Fix names (glossary) · Retranslate with… | idle · running (disabled) | `Row(scroll=AUTO)`, `Chip` |
| `VersionSwitcher` (`chat/messages.py`) | ‹ 2/3 › | first · middle · last | `Row[IconButton, Text, IconButton]` |
| `GlossaryApprovalCard` (`chat/cards.py`) | Header, title, question, FileChip, 5-entry preview, ✏️ Edit / ✓ Yes / ■ No / Always accept (chat cards) | waiting · no file (Edit disabled) · answered | `Card`, `Column`, `FilledTonalButton`, `FilledButton`, `OutlinedButton` |
| `PlanCard` (`chat/plan_card.py`) | Cover · facts line · chips (model · profile · → target · glossary · mode · range · Save to) · Run options · buttons | estimating · ready · invalid (Start disabled with a reason) · async variant | `Card`, `Image`, `Row(wrap=True)` of `Chip`s, `ExpansionTile`, `FilledButton` + a split `PopupMenuButton` |
| `BatchPlanCard` (`chat/batch_plan.py`) | FileChip list with per-file glossary chips · Include subfolders · Start | same as PlanCard | `Card`, `Column`, `Switch` |
| `JobCard` (`chat/cards.py`) | Plan → Queued → Running (ring, progress, line, current item, Requests, Log, issue chips, buttons) → Result (status, ExtractionReportSection, output chips, AttachmentActionsRow) | `PLAN` · `QUEUED` · `RUNNING` · `STOPPING` · `FORCE_STOPPING` · `DONE` · `STOPPED` · `FAILED` · `INTERRUPTED` | `Card`, `ProgressRing`, `ProgressBar`, `ExpansionTile`, `Chip`, `Row(wrap=True)` |
| `RequestSheet` (`chat/cards.py`) | Full streaming content + thinking for one request | streaming · done | `BottomSheet(scrollable=True)`, `Markdown` |
| `PlanGlossarySheet` (`chat/plan_glossary_sheet.py`) | Effective mode line · Load file… · Use book glossary · Clear ✕ · Map glossaries (batch) · Review glossary | none loaded · file loaded · auto-mapped · policy manual | `BottomSheet`, `ListTile`s, `FilePicker` |
| `ManualGlossarySheet` (`chat/manual_glossary.py`) | Mono paste box · Browse… · Use glossary | empty (Use disabled) · filled | `BottomSheet`, `TextField(multiline=True)`, `FilePicker`, `FilledButton` |
| `ChatSettingsSheet` (`chat/chat_settings.py`) | Scope SegmentedButton · sections (§2.14) · Reset chat overrides | This chat / All chats; each row inherited or overridden | `BottomSheet` or `SidePanel`; `SegmentedButton`, `ExpansionTile`, `RadioGroup`, `Switch`, `Slider` |
| `JumpToSheet` (`chat/jump_to.py`) | Step header (Input i/n ▲▼ · Output j/m ▲▼) · tabs Inputs / Outputs · rows | — | `BottomSheet`, `Tabs` / `TabBar` / `TabBarView`, `ListView`, `IconButton` |
| `ChatSearchBar` (`chat/header.py`) | Header TextField · "3/17" · ▲ ▼ · ✕ + compact Input/Output step row | searching · no matches | `TextField`, `IconButton` |
| `AttachmentsManagerView` (`chat/attachments.py`, `AttachmentsScreen`) | Intro + workspace cards (⋯; no Migrate button) | empty · list · blocked (a job is writing the workspace) | `View`, `ListView`, `Card`, `AlertDialog` |
| `LibraryLinkCard` (`chat/job_card.py`) | Book cover + title + progress + Open | — | `Card`, `Image`, `ProgressBar` |
| `ModelSheet` / `ModelPicker` (`settings/model_sheet.py`) | Tabs Model · Profile · Language; search; provider chips; sections; provider group headers with 🌐 refresh; title ⋯; Thinking & effort; route row; footer | loading catalog · polling a group (shimmer) · results · one-shot ("Use once") · field mode | phone: `BottomSheet(draggable=True, fullscreen=…)`; tablet: a custom overlay panel (`page.overlay` `Container`, 420 dp); `Tabs`, `SearchBar`, `Chip`, `ListView` (`first_item_prototype=True` only for the flat search-results list), `ExpansionTile`, `Switch`, `ListTile(on_long_press=…)` |
| `PoeSetupSheet` (`settings/poe_setup.py`) | Warning (route deprecated) · p-b cookie SecretTile · link to the guide · Test | empty · saved · test passed · test failed | `BottomSheet`, `TextField(password=True, can_reveal_password=True)`, `UrlLauncher` |

### 5.5 Media, files and editors

| Component (module) | Anatomy | States | Built from |
|---|---|---|---|
| `ImageCard` / `ImageGallery` (`chat/media_cards.py`) | Image(s) at ≤ 76% width, radius 12; tap → MediaViewer | loading · ready · missing ("Generated image unavailable.") | `Image(fit=CONTAIN, error_content=…)`; several images: a wrap of square thumbnails (a `GridView` needs a fixed height inside the transcript `ListView`) |
| `VideoCard` (`chat/media_cards.py`) | 16:9 player + ⋯ Save as… / Share / Open externally | loading · playing · error (Open externally) | `flet_video.Video(playlist=[VideoMedia(path)], aspect_ratio=16/9)` with its default controls |
| `AudioCard` + `AudioHub` (`chat/media_cards.py`) | "🔊 Generated audio" · play/pause · seek · "m:ss / m:ss" · volume (default 75%) · ⋯ Save / Share / Open externally | stopped · playing · paused · error ("Native audio playback is unavailable") | one `flet_audio.Audio` service per chat (`AudioHub`, created on the first Play). Playback belongs to the playing file: every AudioCard built for it (a transcript re-render, the MediaViewer) adopts the state / position / duration and gets `on_position_change` / `on_duration_change` / `on_state_change`; `IconButton`, `Slider` |
| `CompareSheet` (`chat/media_cards.py`) | Refine "Compare with original": original paragraph above the refined one (`chat/media_model.compare_blocks`) | — | `BottomSheet`, tinted `Container`s |
| `MediaViewer` (`components/media_viewer.py`) | Full-screen image viewer (pinch / pan, swipe between a response's images) or video/audio player · Save / Share / Open externally | — | `View`, `InteractiveViewer`, `Image`, `PageView`, `flet_video.Video`; actions are the chat's callbacks (`ChatFeature.export_file` / `share_files` / `open_external`) |
| `HtmlView` (not built yet) | Sanitized HTML ("View as HTML", QA report) | WebView (Android/iOS) · fallback (Windows/Linux dev: html2text → `Markdown` + "Open in browser") | `flet_webview.WebView`, or `Markdown` + `UrlLauncher` |
| Output editor (`screens/output_editor.py`, `OutputEditorScreen`; the `FullScreenEditor` role) | App bar (title, dirty dot, Save, Cancel) + editor + footer (token count / position) | clean · dirty · saving · read-only · file missing | `View`; `flet_code_editor.CodeEditor` (`MARKDOWN` / `XML` / `JSON` / `CSS` / `PLAINTEXT`), with a plain `TextField(multiline=True)` fallback |
| `TextEditor` (`tools/text_editor.py`, `TextEditorScreen`) | FullScreenEditor + find bar (hit i/n) | read-only · editing | as FullScreenEditor |
| `FileBrowser` (`screens/files.py`, `FileBrowserScreen`) | Breadcrumbs · root chips (Output, Library, Inbox, Chat workspaces) · rows (icon, name, size, date) · ⋯ (Open with · Share · Save to… · Rename · Delete) | loading · list · empty folder · outside the safe roots (blocked) | `View`, a `Row` of breadcrumbs, `ListView`, `ActionSheet`, `TextPromptDialog` |
| `ExportSheet` (`chat/integration.py` `ChatFeature.export_file`: an `ActionSheet` over `services/files.FileBridge.export_options`) | Share · Save to… · Save to Downloads (Android) / Show in Files (iOS) · Open externally | per platform; unsupported items disabled with a reason | `ActionSheet`; `Share.share_files`, `FilePicker.save_file`, native `save_to_downloads`, `UrlLauncher` |
| `SourcePicker` (`tools/source_picker.py`) | Segments Recent outputs · Library books · Chat workspaces · Browse; list with checkboxes (multi). Opt-in options (the chat's Library mode, device fix 2026-10-08): `segments` (one segment hides the SegmentedButton), `searchable` + `query`, `book_rows` (Library list rows via `models.card_model_for` + `BookListRow(show_more=False)`, `CoverQueue` covers, rows built once and kept across renders), `header_action` (label, callback), `include_unresolved` (`targets.library_targets`); eligibility is computed once per row on the io pool; Tools pickers keep the defaults | single · multi · empty segment | `BottomSheet`, `SegmentedButton`, `ListView`, `Checkbox`, `FilePicker` |
| `SelectableTextSheet` (`components/dialogs.py`) | The full text, selectable, + Copy all | — | `BottomSheet`, `SelectionArea` around `Text` |

### 5.6 Lists, selection and Library

| Component (module) | Anatomy | States | Built from |
|---|---|---|---|
| `WindowedList` (`components/windowed_list.py`) | Python-side window (150 rows) over a provider; page selector beyond 1,500 rows | loading · window · appending · filtered · jump (re-centre, then `ScrollKey`) | `ListView(build_controls_on_demand=True, on_scroll=…)` without `first_item_prototype` (rows vary) |
| `LogConsole` (`components/log_console.py`) | Filter chips All / Errors / Thinking / API · search · follow toggle · Copy · Share; 40-line blocks, ≤ 100 mounted | following · paused · filtered · empty | `ListView`, `Text(selectable=True)` in the mono family at the log size (§6.2, `theme.log_text`), `Chip`, `TextField`, `IconButton`; follow = `scroll_to(offset=-1)` |
| `SelectionTopBar` + `BulkActionBar` (`library/selection.py`, reused) | "N selected", Select all, Select ▾, Close; a bottom bar with ≤ 4 icon+label actions + More | inactive · active; actions disabled with a reason | an `AppBar(actions=…)` swap, `BottomAppBar`, `IconButton`, `TextButton`, `PopupMenuButton` (More) |
| `BookCard` (`library/book_card.py`) | Cover stack (image, ribbon, 3 dp progress, ▶ continue) · title · info row · warning chips · pill | not started · in progress · ready to compile · outdated · compiling · completed; selected | `Container(on_click=…, on_long_press=…)`, `Stack`, `Image`, `ProgressBar`, `FilledIconButton`, `Text`; grid = `GridView(max_extent=card_w)` |
| `ProgressRow` / `ChunkChildRow` (`library/chapters_pane.py`) | StatusAvatar · line 1 · line 2 · badges · QA line · chunk chevron · ⋯ | per status · selected · expanded | `ListTile(on_click=…, on_long_press=…)` or `Container`; children in an inline `Column`; `key=ft.ScrollKey(row_key)` |
| `ChaptersPane` (`library/chapters_pane.py`) | Header chips · StatusChipRow · toolbar · ProgressRow list · selection bars | loading (phase 1/2) · list · empty · "file could not be read" banner | `Column`, `ListView(build_controls_on_demand=True)` (never `first_item_prototype`) or `WindowedList` |
| `GlossaryProgressPane` (`library/glossary_pane.py`) | Header + file chip · file card · stats chips · pinned Minimal/Refinement rows · chapter rows with ⋯ | as ChaptersPane, plus a deleted-file banner | same as ChaptersPane, plus `Banner` |
| `DeleteConfirmView` (`library/delete_confirm.py`) | Per-target checkbox rows · contents summary · keyword field · red Delete | Delete disabled until the keyword is typed · deleting (progress) | `View`, `Checkbox`, `ListView`, `TextField`, `FilledButton` in the error colour |
| `EntrySheet` (`glossary/entry_sheet.py`) | Raw · translated · type · gender · description · custom fields · Resolve gender… · Save / Delete | new · edit · conflict warning | `BottomSheet(scrollable=True)` (SidePanel on tablet), `TextField`, `Dropdown`, `FilledButton` |
| `FindReplaceSheet` (`glossary/find_replace.py`) | Find · replace with · Find Next · Replace · Replace all (case-insensitive, every column: the desktop dialog) | n rows · n matches · none → the "apply to output HTML files?" dialog | `BottomSheet`, `TextField`, `ConfirmDialog` |

### 5.7 Settings tiles and editors (schema-bound; `settings/tiles.py`)

**Common anatomy**
- Label (bodyMedium) and one help line (bodySmall).
- ⓘ opens an `InfoSheet` with the full help.
- The control sits in the trailing slot or below the title.
- A lock badge, "🔒 Locked by mode: Minimal", in the locked purple.
- A "modified" dot.
- A `ReasonChip` when the field is unavailable.

**Common states**
- default
- modified
- locked: disabled, with the reason
- unavailable: disabled, with a ReasonChip
- saving: 600 ms debounce
- error: a validation line

Long-press resets to default. Every tile is built on `ListTile(on_long_press=…)` with `key=ft.ScrollKey(<config key>)`.

| Tile (schema kind) | Anatomy / specifics | Built from |
|---|---|---|
| `SwitchTile` (bool) | label + switch | `Switch` |
| `SegmentedTile` (choice, ≤ 4) | label + segments | `SegmentedButton(show_selected_icon=False)` |
| `DropdownTile` (choice / combo) | editable and filterable when the schema allows it | `Dropdown(editable=…, enable_filter=…)` |
| `NumberTile` (int / float) | value with unit, min/max, −/+ stepper | numeric `TextField` + `IconButton`s |
| `SliderTile` (bounded number) | slider + value text | `Slider(divisions=…)`, `Text` |
| `TextTile` (str / URL) | field + quick-paste chips under URL fields | `TextField`, `Chip` |
| `SecretTile` (secret) | a `KeyField` for cookies, Replicate keys and similar | `KeyField` |
| `KeyField` | masked `sk-…a1B2` · eye · paste · Test → status chip. States: empty · masked · revealed · testing · passed · failed (message) | `TextField(password=True, can_reveal_password=True)`, `IconButton`, `Chip` |
| `PromptTile` (prompt) | 2-line preview + placeholder chips + ↺ Reset; tap → PromptEditor. States: default · modified · profile-bound | `ListTile`, `Text(max_lines=2)`, `Chip` |
| `PromptEditor` | full-screen mono editor · token count · placeholder chips that insert at the cursor (`{split_marker_instruction}`, `{glossary_prompt}`, `{target_lang}`, `{raw_text}`, …) · role toggle (System ⇄ User, when relevant) · Save · Save as profile · Reset to default. States: clean · dirty · over-budget warning | `View`, `TextField(multiline=True)` in the mono family (or `CodeEditor(PLAINTEXT)`), `Chip`, `SegmentedButton` |
| `ListEditor` (list / json) | rows with add / remove / reorder, or JSON through a CodeEditor. States: empty · list · invalid JSON | `ReorderableListView` + `ReorderableDragHandle`, `TextField`, `IconButton`, `CodeEditor(JSON)` |
| `KeyPoolTile` (keypool) | "3 keys · ON" + pool switch; tap → `/settings/keys/<pool>`. States: on · off · empty (warning) | `ListTile`, `Switch` |
| `PathTile` (path) | file name + Import… / Clear. Imports CSS, fonts and credentials JSON into app data. States: none · imported · missing | `ListTile`, `FilePicker.pick_files` |
| `ColorTile` (color) | 8 swatches + hex field | a `Row` of swatch `Container`s, `TextField` |
| `FontTile` (font) | font family dropdown (system + imported) + Import font… | `Dropdown`, `FilePicker` |
| `ActionTile` (action) | a button that runs a registered action (ActionRegistry id) and shows its job state. States: idle · running (ProgressRing, Stop) · done · failed (View log) | `ListTile`, `FilledTonalButton` |
| `InfoTile` / `NavTile` (info / subpage) | read-only text, or a link to a subpage | `ListTile` |
| `SectionPage` / `SectionCard` | section app bar (search, ⋯ Save now / Discard changes since opening) + group cards | `View`, `ListView(build_controls_on_demand=False)`, `ExpansionTile` |
| `SettingsSearch` | SearchBar + grouped results with breadcrumbs (≤ 50) + filter chips. States: empty · results · no match | `SearchBar`, `ListView`, `Chip` |

### 5.8 Reader

| Component (module) | Anatomy | States | Built from |
|---|---|---|---|
| `ReaderView` (`reader/reader_view.py`) | WebView page (or the native fallback) + chrome | loading · ready · chrome visible · live panel open · fallback | `View`, `Stack`, `flet_webview.WebView(on_console_message=…)` + `run_javascript`. Fallback: a `ListView` of `Text` / `Image` from `reader_doc.html_to_blocks()`, with `GestureDetector(on_scale_update=…)` for pinch |
| `ReaderChrome` (`reader/chrome.py`) | Translucent top bar (back, titles, Original · Translated · Bilingual, search, ⋯) + bottom bar (chapter slider, Ch / %, ◀ ☰ Aa 🌐 ▶) | visible · hidden (150 ms fade) | `Container` (surface at 92%), `SegmentedButton`, `Slider`, `IconButton` |
| `ChaptersDrawer` (`reader/toc_drawer.py`) | Chapter list with status icons · Native TOC switch · Show special files | — | the View's `end_drawer` (`NavigationDrawer`) on phone; a 320 dp panel on tablet |
| `AaSheet` (`reader/aa_sheet.py`) | Scope switch · tabs Text / Theme / Layout | live preview (no scrim) | `BottomSheet(barrier_color=TRANSPARENT)`, `Tabs`, `Slider`, swatch `Container`s, `SegmentedButton`, `Dropdown` |
| `SelectionChipRow` (`reader/chrome.py`) | Copy · Google Translate / Define · Add to glossary · Ask in chat | shown while a selection exists | a `Row` of `Chip`s in the `Stack`; `Clipboard`, `UrlLauncher(mode=IN_APP_BROWSER_VIEW)` |
| `LivePanel` (`reader/live_panel.py`) | Status line · streamed text · 🧠 Thinking (n) · ⏹ Stop · ✕ Hide | waiting · streaming · finished · stopped · failed | `BottomSheet(draggable=True)`, `Markdown`, `ExpansionTile`, `FilledTonalButton` |
| `ReaderSearchSheet` (`reader/search_sheet.py`) | Field + streamed results (chapter title + excerpt) | searching · results · none | `BottomSheet`, `TextField`, `ListView` |

### 5.9 Jobs, keys and accounts

| Component (module) | Anatomy | States | Built from |
|---|---|---|---|
| `JobsView` / `JobDetail` (`jobs/`) | Sections Running / Queued / Interrupted / Finished; the detail shows request cards + LogConsole | per job state (§1.8) | `View`, `ListView`, `ReorderableListView`, `ProgressBar`, `Chip`, `FloatingActionButton(icon=…, content=…)` (extended) |
| `KeyCard` (`settings/keys.py`) | Masked key · model · status chip (Active / Cooling mm:ss / Disabled / Passed / Failed) · counts · context badges · drag handle | per status · selected | `Card`, `ReorderableDragHandle`, `Chip` |
| `KeyEditor` (`settings/key_editor.py`) | Full-screen form (§4.12) | new · edit · testing | `View`, setting tiles, `ModelPicker` |
| `LoginSheet` (`settings/login_sheet.py`) | Steps Opening browser → Waiting for sign-in → Exchanging token → Done · Reopen browser · Paste redirect URL / code · Cancel | per step · error (retry) · done | `BottomSheet`, a `Column` of step rows (there is no Stepper control in 1.0.3), `TextField`, `UrlLauncher(mode=IN_APP_BROWSER_VIEW)` |
| `AccountCard` (`settings/accounts.py`) | Provider header · slot rows (#N · email · status · refreshed) · ＋ Add account · rotation note | signed in · expired · none · unavailable (ReasonChip) | `Card`, `ListTile`, `PopupMenuButton` |

### 5.10 Manga

| Component (module) | Anatomy | States | Built from |
|---|---|---|---|
| `MangaEditor` (`tools/manga/editor.py`) | Page strip · Source / Translated switch · canvas with boxes · tool bar · workflow buttons | pan · edit (per tool) · running step | `InteractiveViewer`, `Stack(Image, canvas.Canvas, GestureDetector(drag_interval=24))`, `SegmentedButton` |
| `BoxSheet` (`tools/manga/box_sheet.py`) | Tabs "📝 OCR Recognition Result" / "🌍 Translation Result" · edit · Save · Save & Update Overlay · re-run · Delete | — | `BottomSheet`, `Tabs`, `TextField` |
| `ModelDownloadRow` (`tools/manga/models.py`) | Model name · size · status chip (Not downloaded / Downloading NN% / Ready / Loaded) · Download / Load / Delete | per status | `ListTile`, `ProgressBar`, `Chip`, `IconButton` |

---

## 6. Design system

### 6.1 Colour

**Seed: "Halgakos Rose" `#E18F98`.** This is the measured average of the mascot's hair in `assets/Halgakos.png`, computed with PIL.

Brand accents also measured from the asset:
- **Horn Plum `#5B3D57`**: an override for `tertiary` in light theme (tertiaryContainer is generated); also used as the "brand ink" for the splash, empty states and the GLOSSARION header label.
- **Skirt Navy `#404765`**: a hint for `secondary`.

**Theme.**
- `ft.Theme(color_scheme_seed="#E18F98", use_material3=True, visual_density=COMPACT)` plus `dark_theme` with the same seed.
- `theme_mode` follows Appearance (System by default).
- Optional accents: "Desktop Blue" `#5A9FD4` (Direct Text accent) and "Library Violet" `#6C63FF`.
- AMOLED: surfaces forced to #000000 / #0A0A0A in dark mode.

**Semantic colours** (light / dark). They are app-side constants in `ui/theme/colors.py`, because Flet's `Theme` / `ColorScheme` has no custom roles or extensions:

| Role | Light | Dark |
|---|---|---|
| success | #2E7D4F | #6FD49A |
| warning | #B26A00 | #FFB74D |
| info | #0E7C8C | #5FD0DF |
| locked (desktop purple) | #7C3AED | #B388FF |

**Status palette** (same constants module). One map is used by Progress, Library, Glossary progress, Keys and Jobs:

| Status | Colour |
|---|---|
| completed | success (#27AE60 dark parity) |
| merged | info (#17A2B8) |
| in_progress | #F59E0B |
| pending | outline |
| not_translated | #2B6CB0 / #7FB2F0 |
| not_refined, no_tts | #8A63D2 / #B79CFF |
| refine_failed | #7F5F00 / #D8B24A |
| failed, qa_failed | error |
| skipped | #9AA0A6 |
| cooling | warning |
| disabled | outlineVariant |

The Library ribbon and pill hex values (§3.2) are used verbatim in dark mode; in light mode they use the same hue at tone 40.

**Rule.** Status is always shown as icon + text + colour, never colour alone.

### 6.2 Typography (sp; theme `text_theme`; multiplied by Appearance text scale, on top of the OS scale)

| Style | Size / line | Weight | Use |
|---|---|---|---|
| headlineSmall | 22/28 | 600 | tablet page titles, welcome |
| titleLarge | 18/24 | 600 | sheet titles, book title |
| titleMedium | 16/22 | 600 | chat title, card titles |
| titleSmall | 14/20 | 600 | section headers (primary colour) |
| bodyLarge | 15/22 | 400 | message text, editors |
| bodyMedium | 14/20 | 400 | list primary lines |
| bodySmall | 12/16 | 400 | secondary lines, previews |
| labelLarge | 14/20 | 600 | buttons |
| labelMedium | 12/16 | 600 | chips, pills, tabs |
| labelSmall | 11/14 | 500 | header meta, timestamps, captions |
| ribbon | 10/12 | 700, caps, letter-spacing 0.6 | card ribbons |
| mono | 13/18 | 400 | code, editors, error text (Android `monospace`, iOS `Menlo`) |
| log | 8/11 | 400 | log lines: LogConsole (job detail Log, Diagnostics Live log, Manga Files log), the log-file viewer, Check environment lines, the Reader live Thinking/log pane; mono family; fixed sp: follows the OS font scale, not Appearance text size (`theme.log_text`, `tokens.LOG_STYLE`; device fix 2026-10-08) |

Reader typography is independent (§3.11). CJK uses system fallback fonts.

### 6.3 Spacing, radii, sizes, motion

- **Spacing tokens:** 0 · 2 · 4 · 8 · 12 · 16 · 20 · 24 · 32. Page gutter 12 (phone) / 16 (large phone) / 24 (tablet). Card padding 12; sheet padding 16; list item padding 8 vertical × 12 horizontal.
- **Radii:** 6 (badges) · 8 (chips, fields, covers) · 12 (cards, tiles, JobStrip) · 16 (Plan/Job cards, user file cards) · 18 (user bubble) · 20 (sheet top) · 24 (composer) · full (avatars, send).
- **Heights:**
  - app bar 56 · composer rows 40 (the output-mode row included) · chips 32 (28 inside the composer)
  - list rows: one-line 48, two-line 60, chapter row min 64, chat drawer row 44 visual / 48 target
  - JobStrip 44 · bottom action bar 64
  - minimum touch target 48 × 48
- **Elevation:** 0. Surfaces are tonal:
  - `surface`: page
  - `surfaceContainerLow`: cards
  - `surfaceContainerHigh`: composer, sheets
  - `surfaceContainerHighest`: user bubbles, JobStrip
- **Motion:** 150 ms state changes, 200 ms sheets and send morph, 250 ms route transitions (fade-through on Android, Cupertino on iOS). "Reduce motion" turns shimmer and rotations into cross-fades.

### 6.4 Density rules

1. `VisualDensity.COMPACT` app-wide. Never shrink a *hit area* below 48 dp; shrink the visual instead (40 dp icon buttons, 32 dp chips with transparent padding).
2. Setting tiles: one help line at most; ⓘ opens a sheet with the full schema help.
3. Cards show at most 3 visible actions; the rest go under ⋯ / More.
4. No dividers. Separate with 8–12 dp space or a tonal surface change.
5. Numbers in stats use `labelMedium`. Flet 1.0.3 `TextStyle` has no `font_features`, so tabular figures are not available. Counters that tick live (progress, timers, cooldowns) use the mono family to avoid width jitter.
6. Emoji are kept only in desktop-parity vocabularies: status labels, ribbons/pills, output mode labels, report lines. Everywhere else, Material Symbols.

### 6.5 Light and dark

- System by default; manual Light / Dark / AMOLED.
- Reader themes are independent. "Follow app theme" maps Light → Light and Dark → Dark/Midnight.
- Library card hex accents are tuned per mode as in §6.1.
- Images and covers keep their colours. In AMOLED, covers get a 1 dp outlineVariant border.

### 6.6 Haptics (`HapticFeedback` service; global switch in Appearance)

| Event | Haptic |
|---|---|
| Send, ＋ sheet item, chip toggle, Copy | `light_impact` |
| SegmentedButton change, slider tick (Reader chapter slider, Aa sliders), drawer reaching its open threshold, pull-to-refresh threshold | `selection_click` |
| Long-press entering selection mode, starting a job | `medium_impact` |
| Force stop, destructive confirm | `heavy_impact` |
| Job finished while the app is visible | `vibrate` (short), only if the user enabled "Vibrate on completion" |

---

## 7. Phone interaction details

### 7.1 One-handed reach
- **Bottom 40% holds the primary actions:** composer + Send, extended FABs (Import, New entry, Pause queue), bottom selection/action bars, sheet buttons pinned at the bottom of sheets, and the Reader bottom bar.
- **Top bar:** navigation, titles and rarely used ⋯ items only.
- The drawer opens with an edge swipe (no reach needed). Its most used items (Recents) sit in the middle; New chat sits at top-right because it is also in the header.
- Sheets open at 60% height with their primary action visible without scrolling; dragging expands to 90%.

### 7.2 Bottom sheets vs full screens (rule table)

| Use a bottom sheet when… | Use a full-screen View when… |
|---|---|
| A single decision or short form (≤ 6 controls): mode, policy, model, Aa, action menus, confirmations with choices, filters | Long text editing (prompts, outputs, glossary raw, metadata form, composer > 6 lines) |
| A quick look that returns to context: footnotes, request stream, job detail on phone | Lists over 50 rows that need search or selection: Chapters, glossary editor, keys, model manager |
| A live preview of the page underneath: Aa sheet without a scrim | Multi-step flows: Welcome, Delete-with-keyword, Scan for raw, Parallel pair |

On tablets, sheets become SidePanel content (persistent tasks) or centered dialogs with a 560 dp max width (confirmations).

### 7.3 Virtualization and long lists
- **Lazy building.** Every list over 30 items uses `ListView` / `GridView(build_controls_on_demand=True)`. Two exceptions render a bounded window with `build_controls_on_demand=False`, so that every item is a valid `ScrollKey` target:
  - the Transcript (§2.8);
  - SectionPage (§4.15).
- **Row heights.**
  - `first_item_prototype=True` is allowed only for a flat list whose items all have the same height by construction (no section headers), such as the ModelSheet search-results list.
  - Variable-height rows never set `first_item_prototype` or `item_extent`, so they grow with content and text scale. This covers chapter rows, glossary-progress rows, transcript cards, log blocks and glossary editor rows that wrap.
- **Keys.** Stable keys are used everywhere, so updates mutate controls in place and never rebuild the whole list. Jump targets use `ft.ScrollKey`.
- **WindowedList** (Python-side) beyond 1,500 rows, e.g. glossaries of 10k entries or huge spines:
  - it renders a window of 150 and appends on `on_scroll` near the end;
  - a page selector is shown beyond 1,500;
  - jumps re-centre the window before `scroll_to(scroll_key=…)` (§2.8 procedure).
- **Transcript:** windowed by message count and a character budget. Long Markdown is split per paragraph. Bodies load lazily from files.
- **Covers:** decoded in io_pool (PIL) to 240 px thumbnails in `cache/covers/`. `Image.src` is set when ready; a Halgakos placeholder shows until then.
- **Logs:** coalesced into 40-line `Text` blocks, with at most 100 blocks mounted.
- **UI updates:** batched by the UiDispatcher pump (120 ms). Never call `update()` from worker threads.

### 7.4 Loading, empty and error states

| Surface | Loading | Empty | Error |
|---|---|---|---|
| App boot | Boot View: logo + Shimmer, "Preparing…" | — | "API keys could not be decrypted, re-enter" banner → Keys |
| Chat | Shimmer bubbles while bodies load | §2.13 | ErrorCards §2.13 |
| Drawer search | small `ProgressBar` under the field | "No matches" | — |
| Library | skeleton cards, "Scanning library…" / "Loading books…" | §3.1 strings | Banner "Couldn't read <folder>" + Retry |
| Book page | hero skeleton + row shimmer; phase 1 (metadata) shows before phase 2 (chapters) | "No chapters found yet" | "Progress file could not be read — showing last snapshot" banner (outdated) + Full refresh |
| Reader | "Loading book…" (EPUB or TXT) with ProgressRing over the theme background | "This chapter is empty" | WebView failure → automatic switch to the native renderer + snackbar |
| Glossaries | skeleton rows | "No glossaries yet" + Extract / Import | Parse error card with "Edit raw" |
| Jobs | — | §1.8 | Failed row with "View log" |
| Settings search | — | "No settings match “…”" (unavailable settings appear in results with their ReasonChip) | — |
| Tools | per-tool skeleton | per-tool "Pick a source to start" | ErrorCard with log link |

### 7.5 Accessibility at 200% text scale
- Nothing has a fixed height except images. Variable-height lists never use `first_item_prototype` or `item_extent`, so rows grow.
- **Header:** the subtitle reduces to the model plus "▾"; the "custom" badge moves into Chat settings.
- **Composer:** pills collapse into "Options (n)"; the output mode is the compact mode chip at every width; the token hint hides; the TextField's max_lines drops to 4; Send stays at 48 dp.
- **Bottom action bars:** labels hide (icons with `tooltip` + Semantics labels), and anything beyond 4 goes into "More".
- **Tabs:** `TabBar(scrollable=True)`. Chips wrap instead of scrolling where they are the only way to filter (stats rows wrap at ≥ 160%).
- **Book cards:** the grid drops one density step automatically (M → L) and titles allow 3 lines.
- **Reader:** chrome text scales; page text follows its own Aa size (not the OS scale), with a hint "Text size in Aa".
- **Semantics:**
  - every `IconButton` has a `tooltip`. Where the spoken label must differ, it is wrapped in `Semantics(label=…, button=True)`, because `IconButton` has no `semantics_label` in 1.0.3;
  - status avatars announce "Chapter 12, QA Failed";
  - the JobStrip is a `Semantics(live_region=True)` announcing at most once per 10 s;
  - focus order runs header → transcript → composer.
- **Contrast:** status tints are at least AA for text on their tint. Ribbon text uses dark ink on light ribbons.
- **Motion:** "Reduce motion" is honoured (§6.3).

### 7.6 Keyboard, background and system integration
- The composer stays above the IME (`SafeArea`; the scaffold resizes). Opening a sheet dismisses the keyboard first.
- **Long jobs:**
  - Android: foreground service + ongoing notification.
  - iOS: banner "iOS may pause translation about 30 s after you leave the app (iOS 26+ continues in the background)". The setting "Keep screen on during jobs" (`Wakelock`) defaults to ON on iOS.
- **Share / Open-with** is the drag-and-drop equivalent. Flet's `DragTarget` accepts only in-app `Draggable`s, so files cannot be dropped in from other apps. Shared files are imported to the Inbox, then the IntentRouter action sheet offers:
  - Translate in new chat · Add to Library · Open in Reader (EPUB / TXT) · Extract glossary
  - Manga for images · Load as glossary for csv/json · Import keys · Restore config
- **Offline:** requests retry per settings. The running card shows "Waiting for network…"; no global blocking.

---

## Appendix A: UI package layout (inside `src/mobile/app/glossarion_mobile/ui/`)

```
theme/        theme.py tokens.py colors.py (semantic + status constants, §6.1)
shell/        app_shell.py router.py drawer.py sidebar.py side_panel.py job_strip.py launch_banner.py
              (no boot_view.py: native splash + immediate shell, §5.1)
chat/         chat_view.py header.py (+ ChatSearchBar) transcript.py composer.py output_mode_row.py
              mode_options_sheet.py send_button.py messages.py (UserBubble, UserFileCard, AssistantMessage,
              VersionSwitcher) cards.py (GlossaryApprovalCard, JobCard, RequestSheet) media_cards.py media_model.py
              attachments.py jump_to.py chat_ops.py integration.py context.py; the pure models output_modes.py
              send_state.py direct_text_rules.py run_request.py run_controller.py stream_bridge.py job_binding.py
              transcript_model.py; series_page.py (U9, optional)
sheets/       plus_sheet.py (chat/plus_sheet.py re-exports it) chat_settings.py manual_glossary.py model_sheet.py
screens/      output_editor.py (Edit output) files.py (FileBrowser) and the settings / jobs / accounts screens
library/      library_view.py book_card.py filter_sheet.py selection.py delete_confirm.py scan_raw.py
              book_page.py overview_tab.py chapters_pane.py glossary_pane.py output_tab.py metadata_editor.py translate_sheet.py
reader/       reader_view.py webview_reader.py native_reader.py chrome.py aa_sheet.py toc_drawer.py search_sheet.py live_panel.py
jobs/         jobs_view.py job_detail.py
glossary/     glossaries_home.py glossary_view.py editor.py entry_sheet.py find_replace.py settings_tabs.py parallel_pair.py unified.py
tools/        tools_home.py source_picker.py progress_view.py qa.py qa_report.py converter.py headers.py async_batch.py
              review.py sdlxliff.py rpgmaker.py file_browser.py text_editor.py manga/ (files.py settings.py editor.py
              box_sheet.py models.py)
settings/     settings_home.py search.py section_page.py tiles.py model_sheet.py poe_setup.py model_manager.py keys.py
              key_editor.py accounts.py login_sheet.py profiles.py prompt_editor.py endpoints.py data_*.py about.py welcome.py
components/   shared components of §5: action_sheet.py dialogs.py reason_chip.py info_sheet.py empty_state.py
              status.py log_console.py section_card.py media_viewer.py (U7); error_card.py skeleton.py
              windowed_list.py pull_to_refresh.py master_detail.py surface.py (U9: where a sheet opens per size
              class); planned: file_chip.py html_view.py (the ExportSheet is ChatFeature.export_file, the
              full-screen editor is screens/output_editor.py)
(top level)   motion.py (reduce motion, §6.3) text_scale.py (system font scale probe, §7.5) foreground.py (no
              polling in the background, §7.3) - U9
```

## Appendix B: Mobile-only data files (under `FLET_APP_STORAGE_DATA`, written atomically; never read by desktop)

**Message fingerprints (`fp`)**
- assistant: `a:<storage.created_at>:<basename(content_path)>`
- user: `u:<sha1(text)[:12]>:<k-th occurrence>`
- user_file: `f:<sha1(path)[:12]>:<k>`

A fingerprint contains a file name, so routes use `mid = sha1(fp)[:12]` instead (§1.4).

**`direct_text_chats.mobile.json`**
```
{version:1, chats:{"<id>":{
  pinned, pinned_at, series_id (U9), text_scale, skip_plan, auto_accept_glossary,
  library_attachment:{bid, path},
  tool_jobs:{"<created_at>":{library_job|qa_job, source, bid, folder, summary}},
  overrides:{model, profile, target_language, output_mode, attachment_prompt_role, glossary_override_mode,
             manual_glossary_path, force_multipass_off, disable_thinking, skip_prompt_profile, disable_auto_scroll, thinking:{…}},
  versions:{"<anchor fp>":{members:[fp…], selected:int}},
  pending_plan:{spec…}, add_only:[fp…],
  job_turns:{"run:<run root>": fp | null}}},
 scratch:[…]}
```

- `auto_accept_glossary`: the chat's own "Always accept generated glossaries" (§2.11, §2.14).
- `library_attachment`: the chat's attached Library book (§2.5 From Library); cleared when the attachment is removed or another file is attached.
- `tool_jobs`: a Library-job or QA-scan card's data, keyed by the message's `created_at` (the desktop history keeps only its own message-storage keys).

**`mobile_series.json`** (U9, optional)
```
{version:1, series:[{id, name, color, cover_bid, book_ids, defaults:{…}, created_at}]}
```

**`mobile_state.json`** (Prefs) holds:
- last route;
- favorites and recent models; recent languages;
- **`reader_positions`**: `{bid: {href, fraction, page, mode, updated}}`;
- **`reader_bookmarks`**: `{bid: [{href, fraction, label, created}]}`;
- reader per-book typography overrides;
- `file_refs`: the FileRef registry (`fid` → path, bounded LRU of 2,000);
- dismissed tips; `pending_deletes`;
- haptics and appearance mirrors;
- `chat_auto_accept_glossary`: the All-chats "Always accept generated glossaries" (bool, default off);
- job / background flags: `jobs_notification_permission_asked` (set only after a definite permission answer), `jobs_notifications_off_hint_shown` (the one-time "Notifications are off" snackbar), `jobs_battery_prompt_done`, `keep_screen_on_during_jobs`;
- `qa_quick_sample_size_mobile_default` (bool): the one-time QA sample-size migration (a saved 1000 → 0) has run.

**`config.json`** (shared) gets no new keys. Mobile writes only existing desktop keys, sparsely. Examples:
- `epub_library_card_size`, `epub_library_page_size`;
- `epub_details_show_special_files`, `epub_details_show_raw_titles`, `epub_details_chapter_page_size`;
- `retranslation_show_model_info`;
- `epub_reader_*`, `direct_text_*`.

**Orphans.** Sidecar entries whose fingerprint no longer resolves are dropped on load.

## Appendix C: Decisions, divergences and verification items

**Decided by plan §5 (applied in this spec)**

| Topic | Decision | Where |
|---|---|---|
| Output modes | An always-visible inline row of six toggles with an "Output: Text" label (icons only under 400 dp, text labels on tablet); tapping the active mode opens its options sheet | §2.3, §2.6 |
| Unavailable features | Disabled with a ReasonChip, never hidden; there is no "show desktop-only" switch | §0 item 6, §4.15, §5.2 |
| Dependency rule | Ship when the wheels resolve; use REST equivalents otherwise; only native-impossible items stay disabled | §0 item 7 |
| Hit targets | 48 dp | §0 item 4, §6.3 |
| Reader | A full-screen View on every size class | §1.2, §3.11 |
| Settings entry | Drawer footer (not a destination chip) | §1.3 |
| Series | Optional, U9 | §2.15 |
| Routes | Opaque ids only: `/tools/text/<fid>`, `/tools/files/<root>`, `/settings/keys/<pool>`, `/chat/<cid>/m/<mid>` | §1.4 |
| Library keys | `epub_details_*` persisted; all 11 card-size presets mapped | §3.1, §3.7 |
| Generative-only prompt | The composer text (owner attribute `_generative_prompt_override`, read by `image_job`'s prompt-source hook; U7 moved the hook from the planned `run_env` to the runner's module) | §2.6 |
| Reader positions and bookmarks | `mobile_state.json` (Prefs), not config.json | §3.11, Appendix B |
| Default model | `authgpt/gpt-6-luna`; Sign in with ChatGPT in Welcome step 1 and in the Send `blocked` state | §2.4, §4.17 |
| Paste-link tile | Dropped (no desktop feature fetches web pages); the Clipboard tile replaces it | §2.5 |
| Brand | Halgakos Rose seed, Horn Plum tertiary; Desktop Blue and Library Violet optional | §6.1 |
| Bilingual reader | A new shared `reader_doc.build_bilingual_chapter` | §3.11 |
| Mobile-only chat data | Sidecar `direct_text_chats.mobile.json`; the desktop file stays v2-valid | §2.19, Appendix B |
| Book-origin glossary gate | Optional "Review glossary before translating" on the TranslateSheet | §3.10 |
| Plan card thresholds | EPUB / PDF / CBZ / ZIP / SDLXLIFF / subtitle bundles / folders, and TXT over 20k characters; per-chat "Skip plan" | §2.12.1 |
| Always accept generated glossaries | Mobile-only (Prefs + sidecar, default off); the desktop always asks | §2.11, §2.14 |
| Library hand-off | Automatic auto-migrate of finished chat books; no Migrate / Organize / Undo controls | §2.12.4, §2.17, §3.4 |
| QA sample size | Mobile default 0 (duplicate check off) for chat and Tools scans; desktop keeps 1000 | §4.4 |

**Recorded mobile divergences (accepted in the plan)**
- Chat switching during a run, queued sends (the `queue` state), delete message, and edit-and-resend versions.
- Reader double page only on tablets in landscape. SDLXLIFF Notepad layout only on tablets.
- **Output root.** It is limited to app storage or the iOS Files-visible Documents folder, plus the Android "Mirror outputs to Downloads/Glossarion" option. SAF is used for picking only. (U5: the mirror uses the native `save_to_downloads`, i.e. the MediaStore Downloads collection, which accepts only `Download/...` relative paths and also works below API 29; a Documents/Glossarion target would need the MediaStore Files collection in the native extension.)
- Config is snapshotted at job start; desktop reads live widgets per file.
- The Book page uses the Progress Manager status vocabulary. The Library-only "✔ Translated" / "⏳ Working" badges are not shown.
- **Paging.** Chapter and library lists append pages instead of showing pager buttons. The page-size keys still round-trip and set the append increment.
- Desktop default disagreements found during extraction are preserved, and mobile shows the owner-effective value. Desktop bugs found while extracting (for example the Glossary Progress plain dumps) are recorded and fixed only in separate, user-approved commits.
- **U7 (recorded at the U7 integration).**
  - Retranslate (Book page › Chapters) and Resolve QA's Partial.b branch run as jobs (`retranslate`, `resolve_qa`), so they queue behind a running job; the desktop applies Retranslate Selected at once. The confirmation, the RECYCLED three-button choice and the result texts are the desktop's.
  - Resolve QA follows the desktop row menu: an LLM-token issue is repaired in place, otherwise the raw foreign-text issue runs the single-entry Partial.b job (U5 had tried the job first).
  - "🔊 Open Audio File" hands the file to another app (Share) instead of an inline player. (U9: it is "🔊 Play audio" in the in-app MediaViewer now, with Share as a secondary action.)
  - RPG Maker: a picked game folder is copied into the Inbox and a ZIP game is extracted into `<output root>/RPG Maker` (the shared entry `rpgmaker_job.prepare_rpgmaker_game`; the run patches the game's `data` folder, so it needs a writable copy); "Share game as ZIP" afterwards is mobile-only.
  - Google Cloud TTS voices: `google-cloud-texttospeech` has no Android / iOS build (grpcio >= 1.84 has no cp313 mobile wheel, and its requests floor conflicts with the pinned requests), so the Audio options show a ReasonChip; the REST fallback is a U9 tier-B item (dependency rule). (U9: google-cloud-texttospeech 2.38.0 resolves with the grpcio 1.81 / grpcio-status 1.81 pins and ships; no REST fallback was needed.)
  - The text editor and the SDLXLIFF Notepad layout use a monospace text field while `flet-code-editor` is not in the build.
  - SDLXLIFF reviewer: Mark as Completed updates the reviewer session's copy of the progress; the Book page reads the change from disk on its next 2 s poll.
  - Not built yet at U7 (both built in U9 on shared helpers): source-only SDLXLIFF sidecars for Not Translated rows in Manual editing (`progress_core.untranslated_manual_entries`, moved from Retranslation_GUI's `_progress_manager_untranslated_entries` closure); the "✏️ Edit file" jump to the QA issue (`progress_actions.qa_issue_search_target`).

- **Device fixes 2026-10-08 (owner device report on the U8 APK; tests/parity/DISCREPANCIES.md "Device fixes 2026-10-08").**
  - Any Library card opens the Book page (the desktop opens TXT in an editor and a PDF without a workspace in the system viewer); TXT books and TXT translations read in the Reader.
  - Finished chat books move into the Library automatically; there is no Migrate, Organize or Undo control.
  - The ongoing job notification is re-posted when swiped away (Android 14+; at most once a second, a quicker swipe is re-posted when the second is over) and kept current while the app is hidden; the glossary notification has Accept. The page re-reads the permission when the app comes back from the system settings.
  - An auto-accepted glossary gate does not freeze the chat's request cards (the glossary-phase and translation-phase cards stay one live group until the run finishes).
  - Chat QA scans Direct Text workspaces through the shared opt-in; mobile scans default to sample size 0.
  - Log text is 8 sp.

**Flet 1.0.3 constraints applied.** These came from source verification; details are in §5.0. In summary:
- `scroll_to` needs an `ft.ScrollKey` and only reaches built items.
- A control re-created under the key of the control it replaces in a list is frozen after the diff (setting a property raises "Frozen controls cannot be updated"); identity matches and a swapped `Container.content` are not. Cards that are updated in place stay the same objects (the chat's `CardSlot`s, the SDLXLIFF reviewer's per-build row keys).
- `Chip` has no long-press event.
- `PopupMenuButton` cannot be opened from code, so long-press menus use `ContextMenu`.
- There is no Material action sheet and no anchored popover.
- The `AppBar` scroll tint is a manual `bgcolor` swap.
- `flet_video` uses its default controls plus `aspect_ratio`.
- `TextStyle` has no `font_features`.
- `IconButton` has `tooltip` but no `semantics_label`.
- The code editor uses `XML` for html and `PLAINTEXT` for csv.
- Themes have no custom colour roles.
- `flet-webview` runs only on Android, iOS and macOS, and reads `url` only when the control is built: a built WebView navigates with `load_request`.
- A property assignment of a control that compares equal to the current one is skipped (`Prop.__set__`: `old == value`), so a rebuilt control that should replace an identical one needs its own key.
- `DragTarget` accepts only in-app `Draggable`s.

**Still to verify in the 1.0.3 runtime or on a device.** Each item already has a fallback in this spec.
1. `on_scroll` overscroll events for `PullToRefresh`. Fallback: ⋯ Refresh only.
2. `flet-camera` availability for Android and iOS. Fallback: the Camera tile is disabled with a ReasonChip, and Photos remains.
3. `NavigationDrawer.controls` hosting a `SearchBar`, chips and lists. Fallback: a custom left overlay `Container` with the same content.
4. `TextSpan.on_click` inside an `AppBar` title. Fallback: a `Row` of three compact `TextButton`s.
5. `flet_audio.Audio` seek and volume events on iOS for the AudioCard. Fallback: Open externally.
