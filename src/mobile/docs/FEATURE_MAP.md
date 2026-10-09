# Feature to mobile surface map

This map covers **all 1046 inventory features**: 1028 reader features plus 18 critic gaps. The critic's uncovered dialogs and assumed sub-settings are mapped as well.

**How to read it**
- Every Feature cell is the exact inventory name from `feature_inventory.toml`, grouped by reader key.
- Every row names a concrete mobile surface defined in `UI_SPEC.md`.
- `tests_host/test_feature_map.py` checks that every inventory feature appears in its section with a non-empty surface.
- When a feature is added to the inventory, add its row here.

The design decisions behind the surfaces are in `UI_SPEC.md`, with plan §5 as the authority. In short:
- Output modes are the composer's always-visible six-toggle row.
- Nothing is hidden.
- Routes carry opaque ids only.
- The dependency rule decides what ships.
- The default model `authgpt/gpt-6-luna` ships with ChatGPT sign-in.

## Legend

**Surfaces**

| Name | Meaning |
|---|---|
| Chat | Chat home |
| Composer | The radius-24 card at the bottom of the chat |
| Output-mode row | The composer's always-visible "Output: Text" label + six toggles 📝 👁️ 🖼️ 🎬 🔊 ✨ (UI_SPEC §2.3) |
| Mode options | `ModeOptionsSheet`, opened by tapping the active toggle; also shown inline in the ＋ sheet (UI_SPEC §2.6) |
| ＋ sheet | Attach tiles, output mode, tools, chat options |
| Chat settings | Per-chat sheet (This chat / All chats) |
| ModelSheet | Header subtitle → Model / Profile / Language tabs; `ModelPicker` in field mode |
| Plan / Job card | In-thread job card: Plan → Running → Result |
| PlanGlossarySheet | Glossary chip sheet on the Plan card, TranslateSheet and BatchPlanCard |
| Jobs | `/jobs`; LaunchBanner for interrupted jobs |
| JobStrip | Global mini-player |
| Library | Library home |
| Book › Overview / Chapters / Glossary / Output | Book page tabs |
| Reader | Full-screen reader |
| Aa | Reader typography sheet |
| Glossaries | Manager: Editor + General / Balanced / Minimal / Refinement tabs |
| Tools › X | Tool screens |
| Settings › X | Schema section (UI_SPEC §4.15) |
| Keys | Multi-Key Manager (`/settings/keys/<pool>`) |
| Models | Model Manager |
| Accounts | Sign-in providers; "Unavailable on mobile" section |
| Profiles | Profiles & prompts |
| Data › X | Storage / Backup / Import / Logs & diagnostics |
| About › X | Updates, About, guides, Danger zone |
| FileBrowser, TextEditor, ExportSheet, SourcePicker, MediaViewer, HtmlView | Shared components (UI_SPEC §5) |

**Notes keywords**
- **Automatic (no control)**: behaviour with no control of its own. The surface column says where its effect is visible.
- **Adapted**: the mobile form differs from desktop.
- **New**: a mobile addition.
- **Excluded**: on the plan's exclusion list, or native-impossible. The feature is shown as a **disabled row with a ReasonChip** where users would look for it, never hidden, and its config values are preserved.
- **Dependency rule**: ships when its packages resolve for Android and iOS (`check_mobile_wheels.py`). Otherwise it uses a REST equivalent, or shows a disabled row with "Needs <package> · not in this build". U9 outcomes: grpcio 1.81 + google-ai-generativelanguage, google-cloud-translate / -texttospeech / -vision are pinned; google-cloud-aiplatform is not installable (protobuf<7), so Vertex runs over REST; sentence-transformers and argostranslate stay disabled.
- **(U3)**, **(U9)**: the milestone that ships the surface. Every milestone (U0-U9) has shipped; Series (optional, mobile only) shipped in U9.

---

## 1. main-window (77)

| # | Feature | Mobile surface | Notes |
|---|---|---|---|
| 0 | Input File(s) field + drag & drop | Composer › ＋ sheet (Files / From Library / Photos / Camera / Clipboard) → FileChip; Share / Open-with → IntentRouter action sheet; ＋ › From Library / the empty-chat chip / `/library [title]` open the in-chat Library picker (search, newest first, covers); the picked raw is attached and Send defaults to Save to: Library (desktop Load for translation + Run) | Adapted: files are copied into Inbox or Library/Raw; user originals are never renamed. External drag & drop is replaced by Share / Open-with (Flet `DragTarget` accepts only in-app Draggables) |
| 1 | Browse menu: Select Files | ＋ sheet › Files (multi-select) | Several files give a BatchPlanCard |
| 2 | Browse menu: Select Folder (+ include subfolders) | ＋ sheet › Files long-press › "Pick folder…"; BatchPlanCard "Include subfolders" switch | Android SAF failure falls back to a ZIP |
| 3 | Browse menu: Glossary Parallel EPUB Pair | Glossaries ⋯ › Parallel EPUB pair (`/glossary/parallel-pair`) | Glossary-only job |
| 4 | Browse menu: Clear Selection | Attachment chip ×; BatchPlanCard "Clear" |  |
| 5 | File status label / selection summary | Attachment chip meta line + status caption ("Attached X · Vision enabled") |  |
| 6 | 💬 Direct Text (chat-style input/output translator) | **Chat home (root screen)** | The whole app shell is built on it |
| 7 | Asst. Prompt (assistant prefill) dialog with profiles | Settings › Profiles & prompts › Assistant prefill; ModelSheet › Profile › Prefill dropdown |  |
| 8 | GCloud Creds + Vertex AI location | ModelSheet route row (Vertex creds PathTile + location Dropdown); Settings › Endpoints › Vertex | Dependency rule (U9): google-cloud-aiplatform needs protobuf<7 and is not shipped, so Vertex runs through REST + google-auth (Gemini via google-genai `vertexai=True`, Claude via `AnthropicVertex`; the desktop code) |
| 9 | Model box (editable, autocomplete, poll ✓ markers) | Header subtitle → ModelSheet (search, favourites, provider groups, ✓ polled) |  |
| 10 | Automatic provider catalog polling | ModelSheet: shimmer on the provider group while polling (24 h TTL) | Automatic |
| 11 | Model right-click menu | ModelSheet: 🌐 refresh on each provider-group header + title-row ⋯ (Refresh online models · Manage models · Hide unpolled); model row long-press ActionSheet; same in ModelPicker field mode | Refresh ignores the 24 h TTL (desktop "🌐 Refresh Online Models") |
| 12 | Manage Models dialog (⚙ cog inside model box) | Settings › Models & keys › Model Manager |  |
| 13 | ℹ️ Model Provider Information | ModelSheet ⓘ → provider info sheet (bundled docs) |  |
| 14 | 🦙 Load Ollama (ollamapull/ route) | ModelSheet ollamapull/ rows + Settings › Endpoints "Load Ollama" row (disabled row + ReasonChip) | **Excluded**: ollamapull installs and pulls a desktop binary. ollama/ and lmstudio/ to a LAN host work via Settings › Endpoints |
| 15 | 🔐 ChatGPT Login + account-slot dropdown | Accounts › ChatGPT; LoginChip in ModelSheet; Welcome step 1 "Sign in with ChatGPT"; Send `blocked` fix action | Ships in U3: the default model is `authgpt/gpt-6-luna` |
| 16 | 🔐 Grok Login + slot dropdown with '+ N' | Accounts › Grok |  |
| 17 | 🔐 Claude Login + slot dropdown | Accounts › Claude | Browser OAuth + paste code; CLI import excluded |
| 18 | 🔐 Gemini Login + slot dropdown + 📊 status + GCP project combo | Accounts › Gemini (status sheet, project picker) |  |
| 19 | 🔐 OCAGY Login + 📊 quota | Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded**: ocagy (npm/bun) |
| 20 | 🔐 Z.AI Login (authza/authzaN/) | Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded**: authza |
| 21 | Arena Login + account dropdown (autharena/) | Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded**: autharena |
| 22 | 🔐 Antigravity Login + 📊 status + ♻️ reset + 🛸 dashboard | Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded**: antigravity |
| 23 | Model-route-driven control visibility | ModelSheet route row; Plan card chips; Send `blocked` reasons (`route_controls`) |  |
| 24 | Profile dropdown (system prompt profiles) | ModelSheet › Profile tab; Chat settings › Model & prompt; Settings › Profiles | Adapted: the chat pickers list every profile, the task-specific built-ins under "Specialised" |
| 25 | + New Profile | Profiles FAB "New profile"; ModelSheet › Profile › New; Chat settings › Model & prompt › New profile… (a copy of the current profile) |  |
| 26 | Save Profile | Profile editor › Save / Save as |  |
| 27 | Delete Profile / Reset Profile | Profile editor ⋯ Delete / Reset to default |  |
| 28 | Manage Profiles dialog | Settings › Profiles & prompts |  |
| 29 | Profiles import/export | Profiles ⋯ Import / Export; Data › Backup | FilePicker / Share |
| 30 | System Prompt editor | Profile editor (PromptEditor, full screen); Chat settings › Model & prompt › Edit prompt |  |
| 31 | System/User prompt toggle (↕️/🔀) | Profile editor role toggle; ModelSheet › Profile toggle |  |
| 32 | Output Mode combo (📝Text/👁️Vision/🖼️Image/🎬Video/🔊Audio/✨Refine) | Composer output-mode row (six toggles; tap the active one → Mode options; chat key `direct_text_output_mode`); Settings › Translation defaults › Output mode (global `output_mode` for book jobs; linked from Image & vision); Plan card mode chip |  |
| 33 | Output Token Limit button | Settings › Translation defaults; Plan card › Run options |  |
| 34 | Target Language (editable combo) | ModelSheet › Language; Plan card chip; Settings › Translation defaults |  |
| 35 | Open Output Folder 📁 | FileBrowser (Tools › Files, Book › Output › Files, Job card › Files); ExportSheet | Adapted |
| 36 | Run Translation / Stop Translation | Composer Send (text) / Plan card Start (files) / Send-Stop state machine; JobStrip Stop; notification Stop |  |
| 37 | Per-file-type dispatch inside a run | Plan card shows the detected type | Automatic |
| 38 | Threading delay | Settings › Translation defaults › Pacing |  |
| 39 | Chunk Size | Settings › Translation defaults; Plan card › Run options |  |
| 40 | API call delay | Settings › Translation defaults › Pacing |  |
| 41 | API Preflight | Settings › Translation defaults › Pacing |  |
| 42 | Chapter range + Spine Order + 🔍 preview | Plan card › Choose chapters (range field, spine switch, live preview list) |  |
| 43 | Input Token limit + Enable/Disable toggle | Settings › Translation defaults; Plan card › Run options |  |
| 44 | Generate Review + 📝 indicator | Tools › Review; ＋ sheet › Generate review; 📝 badge on Book › Output |  |
| 45 | Context Mode combo | Settings › Context & memory; Plan card › Run options |  |
| 46 | Translation History Limit / Summarize last / Retain | Settings › Context & memory (visibility depends on the mode) |  |
| 47 | RS Keys button | Settings › Context & memory › KeyPoolTile → Keys (Rolling summary pool) |  |
| 48 | Temperature + Disable temperature | Settings › Translation defaults; Plan card |  |
| 49 | Batch Translation + Batch Size | Settings › Translation defaults; Plan card |  |
| 50 | Multipass mode + refinement mode combo | Settings › Translation defaults › Multipass; Plan card; Mode options › Refine |  |
| 51 | Refine Prompt dialog | PromptEditor (Profiles › All prompts › Refine prompt; Mode options › Refine) |  |
| 52 | Glossary Mode dropdown | Settings › Glossary › General; Plan card glossary chip (effective mode); chat policy pill | Shows the effective mode (critic gap) |
| 53 | 🗑️ Delete glossary files / ↩️ Restore glossary backup | Book › Glossary ⋯; Glossaries file row ⋯; Library selection bar › More › Delete glossary files (N) / Restore glossary backup |  |
| 54 | Loaded-glossary status + ✕ clear + auto-mapping | Plan card glossary chip → PlanGlossarySheet (effective mode, Load file…, Use book glossary, Clear ✕, Map glossaries, Review); chat glossary pill |  |
| 55 | Post QA Scan + ⚙️ Scanner Settings | Settings › Processing › Post-translation scan; Plan card switch; link to Settings › QA |  |
| 56 | Remove AI Artifacts (Off/Low/Medium/High) | Settings › Translation defaults |  |
| 57 | API Key + Show/Hide | ModelSheet route row KeyField; Keys main pool; Welcome step 2 | Stored encrypted |
| 58 | Multi Key Manager button | Drawer / sidebar footer 🔑 API keys; Settings › Models & keys › Multi-Key Manager; KeyPoolTiles |  |
| 59 | ⚙️ Other Setting button | Settings home (searchable schema sections) |  |
| 60 | 📚 Library button | Drawer › Library |  |
| 61 | API watchdog progress bar | JobStrip subtitle "3 in flight"; Job card running line |  |
| 62 | Log panel | Jobs › job detail LogConsole; Job card › Log; Data › Logs |  |
| 63 | QA Scan (toolbar) | Tools › QA Scanner; ＋ sheet › QA scan; `/qa` |  |
| 64 | EPUB Converter (toolbar) | Tools › Converter; Book › Compile; Job card › Compile |  |
| 65 | Extract Glossary (toolbar) | ＋ sheet › Extract glossary; Book › Glossary › Extract; Glossaries ⋯ › Extract; `/glossary` |  |
| 66 | ⚙️ Glossary Settings / Progress Manager / 🖼️ Manga Translator / 📦 Async Translator (toolbar launchers) | Drawer › Glossaries; Tools › Progress / Manga / Async |  |
| 67 | 💾 Save Config (toolbar) | Auto-save (600 ms debounce) + Settings ⋯ Save now / Discard changes | Adapted |
| 68 | 📄 Load Glossary (toolbar) | Plan card glossary chip → PlanGlossarySheet › Load file…; Glossaries › Import → Use as manual glossary |  |
| 69 | Splash screen + parallel module preload | Native splash, then the chat shell immediately; background warm import; Send `blocked` "Preparing engine…" + drawer status chip until it ends | Adapted: native splash, then the shell at once; until the warm import ends Send shows `blocked` "Preparing engine…" and the drawer status chip says so (no separate Boot View); undecryptable keys: Settings home notice |
| 70 | First-run Welcome wizard | `/welcome` (step 1 = Sign in with ChatGPT for the default `authgpt/gpt-6-luna`); About › Welcome guide |  |
| 71 | Update checker | About › Updates (Check now · Check on startup · Skip this version · release notes; `update_core`) | (U9) Self-install excluded; links to the APK for the device ABI / the IPA when a release has one; mobile builds are never published (download the APK/IPA artifact from the Build Mobile run), so a release without a mobile build says so |
| 72 | Theme / scaling | Settings › Appearance (theme, accent, text scale) | DPI scaling excluded; the OS handles it |
| 73 | Keyboard: F11 fullscreen, Ctrl+/-/0 zoom | Chat ⋯ › Text size; Reader pinch → Aa font size; MediaViewer pinch zoom; Ctrl ± 0 on tablet keyboards | F11 n/a: mobile is always full screen |
| 74 | Config load/decrypt/sanitize/auto-encrypt/save/backup | Automatic (config_store); Data › Backup; "keys could not be decrypted" banner |  |
| 75 | Close/shutdown | Lifecycle flush on pause; LaunchBanner "N interrupted jobs · Resume / Review"; Jobs › Interrupted (Resume / Discard) | Adapted |
| 76 | Metadata-only and single-chapter runs | Book › Overview › Translate Metadata; Book › Chapters › Translate this chapter; Reader 🌐 |  |

## 2. gui-to-backend (62)

| # | Feature | Mobile surface | Notes |
|---|---|---|---|
| 0 | Run Translation (single input) | Chat: attach → Send → Plan card → Start; Library › Translate… (TranslateSheet) |  |
| 1 | Multiple input files / batch queue | ＋ Files multi-select → BatchPlanCard (one job, sub-items); Jobs queue |  |
| 2 | Browse folder + Deep scan | BatchPlanCard "Include subfolders" switch |  |
| 3 | Input normalization & workspace collision rename | FileBridge copies with a " (2)" suffix | Automatic; originals are never renamed |
| 4 | ZIP / CBZ / standalone HTML input conversion | Plan card "ZIP → EPUB on start" | Automatic |
| 5 | Subtitle & synced-lyrics ZIP bundles (SRT/ASS/LRC) | Attach .zip → Plan card; Book › Chapters subtitle rows |  |
| 6 | PDF translation | Attach PDF → Plan card; Settings › PDF |  |
| 7 | SDLXLIFF translation | Attach .sdlxliff → Plan card |  |
| 8 | TXT/MD/JSON/CSV translation | Paste or attach in chat |  |
| 9 | Standalone image / video translation | Photos / Camera / Files → Vision auto-switch |  |
| 10 | Generative-only mode (no input file) | Mode options (Image / Video / Audio) › "Generate from prompt (no input)"; the prompt is the composer text | Decided: a small run_env hook passes the composer text; the button is disabled with a reason when the composer is empty or an attachment is present |
| 11 | RPG Maker game translation (GTool) | Tools › RPG Maker (pick game folder or ZIP) | Adapted: no .exe filter |
| 12 | Auto Glossary modes (Off / Off Fuzzy / Manual Only / No Glossary / Minimal / Balanced / Full / Single Pass) | Settings › Glossary › General; Plan card chip; Chat settings › Glossary (Direct Text policy) |  |
| 13 | Require complete glossary before translation | Settings › Glossary › General |  |
| 14 | Extract Glossary (manual) | ＋ sheet › Extract glossary (Job card); Book › Glossary › Extract |  |
| 15 | Image-folder glossary extraction | Extract glossary with images / CBZ attached |  |
| 16 | Glossary auto-load / auto-mapping / per-file glossary map | BatchPlanCard / PlanGlossarySheet › Map glossaries | Automatic |
| 17 | Multipass refinement (Full / Full+raw / Failed / Partial / Partial.b / Partial.b2) | Settings › Translation defaults › Multipass; Mode options › Refine |  |
| 18 | Resolve single QA issue (targeted Partial.b) | Book › Chapters row ⋯ › ⚠️ Resolve QA issue → RESOLVE_QA job |  |
| 19 | Post-translation QA scan phase | Settings › Processing; Plan card switch | Skipped for chat-workspace runs (desktop parity) |
| 20 | QA Scan (manual, single or bulk folders) | Tools › QA Scanner (SourcePicker multi-select) |  |
| 21 | EPUB Converter / Compile EPUB | Book › Compile; Job card › Compile; Tools › Converter |  |
| 22 | PDF compile from workspace | Book › Compile ▾ PDF; Tools › Converter | Adapted: PyMuPDF renderer |
| 23 | Metadata-only translation (one or many EPUBs) | Book › Translate Metadata; Library bulk "Metadata" |  |
| 24 | Single-chapter translation (Library/Reader Translate) | Book › Chapters row; Reader 🌐 live |  |
| 25 | Direct Text (Input/Output chat) | Chat home |  |
| 26 | Stop / Graceful stop / Force stop | Send-Stop state machine; JobStrip; Job card; notification |  |
| 27 | Save partial / prohibited results on stop | Settings › Response handling › Failure saving |  |
| 28 | Crash/shutdown recovery of in-progress rows | LaunchBanner "N interrupted jobs · Resume / Review" → Jobs › Interrupted (Resume / Discard) |  |
| 29 | Live log console | Job detail LogConsole; Job card › Log; Data › Logs |  |
| 30 | API watchdog / in-flight progress bar | JobStrip; Job card |  |
| 31 | Streaming & thinking log toggles | Settings › Response handling › Streaming (one switch) | Adapted: one switch for the four toggles + Enable thoughts, on by default; off also stops chat / Reader streaming (the desktop forces it) |
| 32 | Chapter range & spine order | Plan card › Choose chapters |  |
| 33 | Input / output token limits | Settings › Translation defaults; Plan card |  |
| 34 | Context mode (Off / Contextual History / Rolling Summary Replace/Append) | Settings › Context & memory; Plan card |  |
| 35 | Batch translation / request merging | Settings › Translation defaults; Settings › Processing › Request merging |  |
| 36 | Output directory override | Data › Storage › Output folder | Adapted: output root limited to app storage / iOS Files-visible Documents (+ Android "Mirror outputs to Downloads/Glossarion"); arbitrary SAF folders are not supported |
| 37 | Save glossary copy in output | Settings › Glossary › General |  |
| 38 | Multi API key pools | Keys |  |
| 39 | Vertex AI credentials | ModelSheet route row; Settings › Endpoints › Vertex | Dependency rule (U9): google-cloud-aiplatform needs protobuf<7 and is not shipped, so Vertex runs through REST + google-auth (Gemini via google-genai `vertexai=True`, Claude via `AnthropicVertex`; the desktop code) |
| 40 | Custom endpoints & routing | Settings › Endpoints; Models › Custom prefixes |  |
| 41 | Thinking / reasoning controls | Settings › Thinking & reasoning; ModelSheet › Thinking; Chat settings "Disable all thinking" |  |
| 42 | Sampling / anti-duplicate parameters | Settings › Anti-duplicate |  |
| 43 | Output mode (Text / Vision / Image / Video / Audio / Refinement) | Composer output-mode row; Settings › Translation defaults › Output mode; Plan card mode chip |  |
| 44 | Profile-driven extraction override | Profile editor › Extraction override section |  |
| 45 | System prompt as user message / split-marker instruction | Profile role toggle; Settings › Processing › Request merging `{split_marker_instruction}` chip |  |
| 46 | Large-EPUB extraction tuning | Settings › Response handling › Parallel extraction | Automatic; threads only |
| 47 | Save Config / settings persistence | Auto-save |  |
| 48 | Debug payload capture | Data › Logs & diagnostics › Save payloads |  |
| 49 | Library raw-input registry & source resolution | Library › Scan for raw; Book ⋯ Clear saved raw link | Automatic |
| 50 | Retain source extension rename | Settings › EPUB output + "Rename files" action; Tools › Converter |  |
| 51 | Translation QA failure summary | Job card Result "N QA failed" chip → Book › Chapters filtered |  |
| 52 | Open Output Folder | FileBrowser / ExportSheet |  |
| 53 | Generate Review | Tools › Review |  |
| 54 | Standalone header/TOC translation | Tools › Headers & metadata |  |
| 55 | Async (batch API) processing | Tools › Async batch; Plan card Start ▾ "Run as async batch" |  |
| 56 | Parallel EPUB Pair (glossary-only) | Glossaries › Parallel EPUB pair |  |
| 57 | Manga translator launch | Tools › Manga; ＋ sheet › Manga; "Translate as manga" chip |  |
| 58 | Browser-backed free routes (authnd/, Gemini-free) | Accounts › Experimental (WebViewBridge); ModelSheet route chip | Experimental (U9): a hidden in-app WebView runs the desktop page scripts through `browser_driver`; limits shown in Accounts › Experimental; disabled row + ReasonChip where flet-webview is unavailable (Windows/Linux dev) |
| 59 | Subscription logins (ChatGPT/Claude/Gemini/Grok) | Accounts |  |
| 60 | Antigravity / GLM / Arena proxies, OcAgy | Accounts › Unavailable | **Excluded**: antigravity / authza / autharena / ocagy |
| 61 | Tor proxy routing | Settings › Response handling & retries › "Tor proxy routing" (disabled row + ReasonChip) | **Excluded**: Tor binary; value preserved |

## 3. translation-engine (60)

| # | Feature | Mobile surface | Notes |
|---|---|---|---|
| 0 | EPUB translation (full book) | Chat attach → Plan → Job; Library › Translate… |  |
| 1 | Plain text / CSV / JSON / Markdown translation | Chat paste or attach; Settings › Processing › Break split count |  |
| 2 | PDF input translation | Attach PDF; Settings › PDF › Input |  |
| 3 | PDF alignment / RTL layout options | Settings › PDF › Input layout |  |
| 4 | PDF output format (PDF vs EPUB) and PDF compile options | Settings › PDF |  |
| 5 | Subtitle translation (SRT/ASS/LRC) | Attach → chat; result file chip |  |
| 6 | Subtitle/lyrics ZIP bundle | Attach .zip → Plan card |  |
| 7 | SDLXLIFF translation round-trip | Attach .sdlxliff; output chip; Tools › SDLXLIFF reviewer |  |
| 8 | HTML / chapter ZIP / CBZ input → EPUB | Plan card note | Automatic |
| 9 | RPG Maker game translation (GTool) | Tools › RPG Maker |  |
| 10 | Metadata-only run | Book › Translate Metadata |  |
| 11 | Single-chapter translate | Book › Chapters; Reader 🌐 |  |
| 12 | Extraction mode & filtering | Settings › Processing › Extraction |  |
| 13 | Header/structure cleanup during extraction | Settings › Processing |  |
| 14 | Remote image download | Settings › EPUB output › Remote images |  |
| 15 | Special-file handling | Settings › Processing › Special files (keyword editor); Book › Chapters "Do not skip" |  |
| 16 | Chapter numbering controls | Settings › Processing › Chapter numbering |  |
| 17 | Chapter range / spine order | Plan card › Choose chapters |  |
| 18 | Custom CSS attach & fonts | Settings › EPUB output (PathTile imports into app data) | Adapted |
| 19 | System prompt / profile | Profiles; ModelSheet › Profile |  |
| 20 | Assistant prompt (prefill) | Profiles › Assistant prefill |  |
| 21 | Core sampling & limits | Settings › Translation defaults |  |
| 22 | Chunking & token budget | Settings › Translation defaults (chunk size); Response handling (compression factor) | Automatic |
| 23 | Chunk prompt & previous-chunk context | Settings › Processing › Configure chunk prompt; Context & memory |  |
| 24 | Chunk-level resume | Settings › Response handling; Job card "Resume" |  |
| 25 | Context mode: Off / Contextual history | Settings › Context & memory |  |
| 26 | Context mode: Rolling summary (replace/append) | Settings › Context & memory › Memory prompts |  |
| 27 | Batch (parallel) translation | Settings › Translation defaults |  |
| 28 | Request merging (Split-the-Merge) | Settings › Processing › Request merging |  |
| 29 | Retry policies | Settings › Response handling › Retries |  |
| 30 | Failure output handling | Settings › Response handling › Failure saving |  |
| 31 | AI artifact / thinking cleanup | Settings › Translation defaults / Processing |  |
| 32 | Emergency paragraph / image restore | Settings › Processing |  |
| 33 | Duplicate-content detection & retry (Basic / Cascading / AI Hunter) | Settings › Response handling › Duplicates + "Configure AI Hunter" subpage |  |
| 34 | Anti-duplicate sampling parameters | Settings › Anti-duplicate |  |
| 35 | Glossary injection into requests | Settings › Glossary › General (append format, compression) | Automatic |
| 36 | Auto glossary generation phase (inside translation) | Plan card glossary chip; GlossaryApprovalCard (chat runs) |  |
| 37 | Title tag & image-only title translation | Settings › Metadata, TOC & headers |  |
| 38 | Book title + metadata translation | Settings › Metadata (Configure All, Custom metadata, mode); Book › Translate Metadata |  |
| 39 | Chapter header & TOC translation | Settings › Metadata, TOC & headers |  |
| 40 | Standalone 'Translate Headers' tool | Tools › Headers & metadata |  |
| 41 | In-chapter image translation (web novels) | Settings › Image & vision › Process web novel images |  |
| 42 | Output mode: Text | Mode options 📝; Markdown card |  |
| 43 | Output mode: Vision (OCR) | Mode options 👁️ (OCR prompt, skip, batch, keep image); OCR section in card |  |
| 44 | Output mode: Image (EPUB/PDF passthrough + image edit) | Mode options 🖼️ (1K / 2K / 4K, batch); image gallery card |  |
| 45 | Output mode: Audio / TTS | Mode options 🔊 (voice); AudioCard |  |
| 46 | Output mode: Video | Mode options 🎬 (duration, resolution); VideoCard |  |
| 47 | Output mode: Refinement (standalone) | Mode options ✨ (mode, raw role, prompt); refined card + Compare |  |
| 48 | Multipass refinement after translation | Settings › Translation defaults › Multipass |  |
| 49 | Output naming & retain source extension | Settings › EPUB output |  |
| 50 | MD/TXT and SDLXLIFF sidecars | Settings › EPUB output › Sidecars (+ Generate actions); Tools › Converter |  |
| 51 | Output HTML post-processing | Settings › Processing |  |
| 52 | Progress tracking (translation_progress.json) | Book › Chapters; Tools › Progress; Job card progress |  |
| 53 | Stop: graceful vs immediate | Send-Stop state machine; Settings › Response handling › Stop logic |  |
| 54 | Post-translation QA scan | Settings › Processing; Plan card |  |
| 55 | Review generator | Tools › Review |  |
| 56 | Direct Text engine hooks | Chat (shared `direct_text_core` run scope) | Automatic |
| 57 | Multi-key / per-purpose key pools | Keys |  |
| 58 | Debug toggles | Data › Logs & diagnostics |  |
| 59 | Library origin lookup for raw EPUB | Automatic (no control); the raw link shows on Book › Output › Raw source row | Automatic (library_core) |

## 4. api-client (93)

| # | Feature | Mobile surface | Notes |
|---|---|---|---|
| 0 | Model-prefix provider routing | ModelSheet search (prefix-aware ranking) | Automatic |
| 1 | OpenAI native (gpt*, chatgpt*, o1/o3/o4, codex*) | ModelSheet › OpenAI; Keys |  |
| 2 | OpenAI image generation/editing | Mode options 🖼️; Settings › Endpoints › Image edit |  |
| 3 | Global Custom OpenAI Endpoint | Settings › Endpoints |  |
| 4 | Override Gemma routing for custom endpoint | Settings › Endpoints |  |
| 5 | Per-key Individual Endpoint | Keys › KeyEditor › Individual endpoint |  |
| 6 | Custom prefix routes | Models › Custom prefixes |  |
| 7 | Gemini native (gemini-*, gemma-*, palm*, bard*) | ModelSheet › Google; Settings › Thinking (Gemini); Provider options & safety |  |
| 8 | Gemini OpenAI-compatible endpoint | Settings › Endpoints › Gemini custom |  |
| 9 | Gemini raw gRPC transport | Settings › Endpoints › Gemini gRPC transport | Tier-B pin (U9): grpcio 1.81.0 + google-ai-generativelanguage 0.12.1 ship; the bootstrap sets `GRPC_DNS_RESOLVER=native` |
| 10 | Gemini Veo / Omni video and Lyria music generation | Mode options 🎬 / 🔊 |  |
| 11 | Anthropic native (claude*, sonnet*, opus*, haiku*) | ModelSheet › Anthropic; Settings › Thinking; Endpoints › Anthropic custom |  |
| 12 | Force Native Anthropic format | Settings › Endpoints |  |
| 13 | DeepSeek | ModelSheet; Settings › Thinking (DeepSeek) |  |
| 14 | xAI Grok (grok*) | ModelSheet › xAI; Keys |  |
| 15 | Mistral family (mistral, mixtral, codestral, devstral, pixtral, voxtral, magistral, ministral, labs-leanstral) | ModelSheet › Mistral |  |
| 16 | Mistral OCR | ModelSheet (Vision OCR model) |  |
| 17 | Cohere (command*, cohere*, aya*) | ModelSheet |  |
| 18 | Groq (groq/, llama-groq, mixtral-groq) | ModelSheet; Settings › Endpoints › Groq/Local base URL |  |
| 19 | OpenRouter (or/, openrouter/) | ModelSheet; Settings › Provider options & safety |  |
| 20 | LiteRouter (lr/) | ModelSheet |  |
| 21 | OpenCode Go (oc/, opencode/, opencode-go/) | ModelSheet; Settings › Thinking |  |
| 22 | OpenCode Zen free (ocz/) | ModelSheet ocz/ rows (disabled row + ReasonChip) | **Excluded**: ocz/ (npm/bun) |
| 23 | ElectronHub (eh/, electronhub/, electron/) | ModelSheet |  |
| 24 | Poe (poe/) | ModelSheet Poe route row → PoeSetupSheet (paste p-b cookie, deprecation guide link, Test) | Deprecated route |
| 25 | NVIDIA NIM (nd/) | ModelSheet; Settings › Thinking |  |
| 26 | Chutes (chutes/) | ModelSheet |  |
| 27 | Fireworks (fireworks/) | ModelSheet; Settings › Endpoints |  |
| 28 | Together AI (together/, llama*, alpaca, vicuna, wizardlm, openchat; bloom/opt/galactica/llama2-4/codellama fallback) | ModelSheet |  |
| 29 | Perplexity (perplexity, pplx, sonar) | ModelSheet |  |
| 30 | Z.AI API key (za/) | ModelSheet; Keys | Key route allowed; only the authza login is excluded |
| 31 | NanoGPT (nan/) | ModelSheet; Image / Video modes |  |
| 32 | SambaNova (sam/) | ModelSheet |  |
| 33 | Chinese and other legacy OpenAI-compatible providers | ModelSheet |  |
| 34 | AI21 / Replicate / Aleph Alpha / HuggingFace | ModelSheet |  |
| 35 | Azure OpenAI (azure*) | ModelSheet; Settings › Endpoints (Azure version); KeyEditor |  |
| 36 | Vertex AI Model Garden (vertex/, model@version) | ModelSheet route row; Settings › Endpoints › Vertex | Dependency rule (U9): google-cloud-aiplatform needs protobuf<7 and is not shipped, so Vertex runs through REST + google-auth (Gemini via google-genai `vertexai=True`, Claude via `AnthropicVertex`; the desktop code) |
| 37 | DeepL (deepl) | ModelSheet; Keys |  |
| 38 | Google Translate Free (google-translate-free) | ModelSheet |  |
| 39 | Google Cloud Translate (google-translate) | ModelSheet (Google Cloud Translate route) | Dependency rule (U9): google-cloud-translate 3.28.0 ships (its translate_v2 client is REST) |
| 40 | Local OpenAI-compatible routes (ollama/, lmstudio/) | ModelSheet › Local; Settings › Endpoints › Local LLM host | iOS local-network permission |
| 41 | Managed Ollama (ollamapull/) | ModelSheet ollamapull/ rows (disabled row + ReasonChip) | **Excluded**: desktop binary install |
| 42 | AuthGPT (authgpt/, authgptN/) ChatGPT subscription | Accounts › ChatGPT |  |
| 43 | AuthGrok (authgrok/, authgrokN/, authgrok0/ rotating pool) | Accounts › Grok |  |
| 44 | AuthCD (authcd/, authcdN/) Claude subscription | Accounts › Claude |  |
| 45 | AuthGem (authgem/, authgemN/, authgem-key/, authgem-vertex/, authgem-vertexN/) | Accounts › Gemini |  |
| 46 | AuthZA (authza/, authzaN/) Z.AI login plan / general API | Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded** |
| 47 | Antigravity (antigravity/) | Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded** |
| 48 | OcAgy (ocagy/, ocagy0/, ocagyN/) | Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded** |
| 49 | AuthArena (autharena/, autharena0/, autharenaN/) | Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded** |
| 50 | AuthND (authnd/, authndN/) NVIDIA Build browser route | Accounts › Experimental (WebViewBridge); Settings › Response handling › NIM/AuthND helpers | Experimental (U9): a hidden in-app WebView runs the desktop page scripts through `browser_driver`; limits shown in Accounts › Experimental; disabled row + ReasonChip where flet-webview is unavailable (Windows/Linux dev) |
| 51 | Gemini Free (search/, search/gemini) | Accounts › Experimental (WebViewBridge); Settings › Response handling › Gemini Free chunking | Experimental (U9): a hidden in-app WebView runs the desktop page scripts through `browser_driver`; limits shown in Accounts › Experimental; disabled row + ReasonChip where flet-webview is unavailable (Windows/Linux dev) |
| 52 | Opera Aria (search/opera) | ModelSheet search/opera rows + Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded** |
| 53 | Tor proxy for provider traffic | Settings › Response handling & retries › "Tor proxy" (disabled row + ReasonChip) | **Excluded**; value preserved |
| 54 | Streaming responses | Settings › Response handling › Streaming (one switch) |  |
| 55 | GPT-5 / OpenRouter / NIM / OpenCode thinking controls | Settings › Thinking; ModelSheet › Thinking |  |
| 56 | Reasoning-effort auto-repair | Log line in Job card | Automatic |
| 57 | Skip thinking for lightweight tasks | Settings › Thinking |  |
| 58 | Service tier (standard/flex/fast/priority) | Settings › Provider options & safety |  |
| 59 | API safety settings | Settings › Provider options & safety |  |
| 60 | Anti-duplicate sampling parameters | Settings › Anti-duplicate |  |
| 61 | Disable temperature / per-key temperature | Settings › Translation defaults; KeyEditor |  |
| 62 | Per-key request parameters | KeyEditor › 🧩 Request parameters |  |
| 63 | Multi-key rotation (main pool) | Keys › Rotation card |  |
| 64 | Per-context key routing | KeyEditor › Request contexts (tri-state chips) |  |
| 65 | Fallback keys | Keys › Fallback pool |  |
| 66 | Dedicated key pools | Keys pool chips; KeyPoolTiles in Settings |  |
| 67 | API key testing | Keys › Test selected / all; KeyField Test |  |
| 68 | Test API connections (endpoints) | Settings › Endpoints › Test connection |  |
| 69 | Retry, backoff and rate-limit handling | Settings › Response handling › Retries; Job card "Rate limited" chip |  |
| 70 | Timeouts and HTTP tuning | Settings › Response handling › HTTP |  |
| 71 | Truncation and empty-response retry | Settings › Response handling › Truncation |  |
| 72 | Refusal pattern detection | Keys › Refusal patterns; Settings › Response handling |  |
| 73 | Failure result handling | Settings › Response handling |  |
| 74 | System-prompt-to-user merge and preflight fixes | Profile role toggle | Automatic |
| 75 | Request pacing (API call delay / threading delay / stagger) | Settings › Translation defaults › Pacing |  |
| 76 | Stop / graceful stop / hard cancel | State machine; Settings › Stop logic |  |
| 77 | API watchdog (in-flight indicator) | JobStrip; Job card |  |
| 78 | Payload/response dumps and HTTP logging | Data › Logs & diagnostics | Redacted when shared |
| 79 | Text-to-speech | Mode options 🔊; Settings › Endpoints › TTS; Keys (TTS pool) |  |
| 80 | Image/video output mode and custom image-edit endpoint | Mode options; Settings › Endpoints |  |
| 81 | Image request encoding | Settings › Image & vision › Compression | Automatic |
| 82 | Static model catalog | ModelSheet |  |
| 83 | Provider model catalog polling | ModelSheet 🌐 per-provider refresh + title ⋯ Refresh online models; Models › Poll providers |  |
| 84 | Polled-model markers and Hide unpolled models | ModelSheet ✓; Models "Polled only" chip |  |
| 85 | Numbered account prefix completion | ModelSheet › Account aliases section |  |
| 86 | Login status / account slots / logout for auth routes | Accounts; LoginChip |  |
| 87 | API key encryption in config.json | Fernet key in SecureStorage; Data › Backup (passphrase export) | Automatic |
| 88 | OAuth token encryption | Automatic (no control); sign-in state shown in Accounts | Automatic: tokens encrypted with the SecureStorage key |
| 89 | Async batch processing (50% off) | Tools › Async batch |  |
| 90 | Model Provider Information | ModelSheet ⓘ |  |
| 91 | Chapter/request context labeling | Assistant header labels; Job card request rows |  |
| 92 | Gemini prohibited-use / safety detection | ErrorCard "Blocked by the provider's safety filter" | Automatic |

## 5. auth-routes (47)

| # | Feature | Mobile surface | Notes |
|---|---|---|---|
| 0 | ChatGPT Login (AuthGPT) button | Accounts › ChatGPT › Log in (LoginSheet) |  |
| 1 | ChatGPT logout | Accounts slot ⋯ › Log out |  |
| 2 | AuthGPT account-slot dropdown and pool rotation (authgpt0/) | Accounts › ChatGPT slots + rotation note; ModelSheet account aliases |  |
| 3 | AuthGPT translation requests | Automatic (no control); status in Accounts › ChatGPT and the LoginChip | Automatic |
| 4 | AuthGPT / AuthGrok / AuthCD / AuthGem token import from env (headless) | Data › Logs & diagnostics (env tokens honoured, shown redacted) | Automatic |
| 5 | Grok Login (AuthGrok) button | Accounts › Grok |  |
| 6 | Grok account slots '+ N' and authgrok0/ rotation | Accounts › Grok "＋ Add account" |  |
| 7 | Grok CLI credential import | Accounts › Unavailable | **Excluded**: CLI credential import |
| 8 | AuthGrok translation requests | Automatic (no control); status in Accounts › Grok | Automatic |
| 9 | Claude Login (AuthCD), Strategy 1: import an existing Claude Code login | Accounts › Unavailable | **Excluded**: CLI credential import |
| 10 | Claude Login (AuthCD), Strategy 2: automatic browser OAuth | Accounts › Claude › Log in |  |
| 11 | AuthCD manual paste-code flow | LoginSheet › "Paste redirect URL / code" | New in the UI (hidden on desktop) |
| 12 | Claude logout | Accounts |  |
| 13 | AuthCD account email lookup and requests | Accounts slot row shows the email |  |
| 14 | Gemini Login (AuthGem) button | Accounts › Gemini |  |
| 15 | Gemini logout | Accounts |  |
| 16 | AuthGem status 📊 (quota and verification) | Accounts › Gemini › Status sheet; LoginChip |  |
| 17 | AuthGem GCP project dropdown (authgem-vertex/) | Accounts › Gemini › Project; ModelSheet route row |  |
| 18 | AuthGem account slots and authgem-vertex0/ pool | Accounts › Gemini slots |  |
| 19 | AuthGem translation routes | Automatic (no control); status in Accounts › Gemini | Automatic |
| 20 | AuthND (authnd/), NVIDIA Build free browser route | Accounts › Experimental | Experimental (U9): hCaptcha tokens are minted in the hidden in-app WebView (`browser_driver`); an interactive challenge fails at the token timeout |
| 21 | NIM / AuthND token helper settings | Settings › Response handling › NIM/AuthND helpers | Adapted: the subprocess-limit field is shown disabled with a ReasonChip (no subprocesses on mobile) |
| 22 | AuthND model catalog polling | Models › Poll providers |  |
| 23 | Gemini Free (search/gemini) | Accounts › Experimental | Experimental (U9): AI Mode runs in the hidden in-app WebView; Google consent / verification pages surface as errors |
| 24 | Gemini Free browser chunking settings | Settings › Response handling | (U9) Every chunking setting applies: one hidden page per helper request (at most 4) |
| 25 | Opera Aria (search/opera, search/opera-think) | ModelSheet search/opera rows + Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded** |
| 26 | TOR proxy rotation (search/opera and ocz/) | Settings › Response handling & retries › "Tor rotation" (disabled row + ReasonChip) | **Excluded**; value preserved |
| 27 | Antigravity Login (antigravity/, antigravityN/) | Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded** |
| 28 | Antigravity status 📊 / reset rankings ♻️ / dashboard 🛸 | Accounts › Unavailable on mobile › Antigravity row (disabled row + ReasonChip) | **Excluded** |
| 29 | Antigravity/OcAgy encrypted account storage adapter | Accounts › Unavailable on mobile (Antigravity / OcAgy rows, disabled); stored account files round-trip untouched | **Excluded** |
| 30 | OCAGY Login (ocagy0/, ocagy/, ocagyN/) | Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded** |
| 31 | OcAgy quota status 📊 and requests | Accounts › Unavailable on mobile › OCAGY row (disabled row + ReasonChip) | **Excluded** |
| 32 | OpenCode Zen free models (ocz/) | ModelSheet ocz/ rows + Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded** |
| 33 | Z.AI Login (AuthZA, authza/, authzaN/) | Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded** |
| 34 | AuthZA / GLM access mode toggle | Settings › Endpoints › "AuthZA / GLM access mode" (disabled row + ReasonChip) | **Excluded**; value preserved |
| 35 | Arena Login (autharena/, autharenaN/, autharena0/ rotation) | Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded** |
| 36 | Arena account selector (#0..#N, + New) and catalog refresh | ModelSheet autharena/ rows + Accounts › Unavailable on mobile › Arena row (disabled row + ReasonChip) | **Excluded** |
| 37 | Google Translate Free (google-translate-free) | ModelSheet |  |
| 38 | Gemini gRPC endpoint transport | Settings › Endpoints › Gemini gRPC transport | Tier-B pin (U9): grpcio 1.81.0 + google-ai-generativelanguage 0.12.1 ship; the bootstrap sets `GRPC_DNS_RESOLVER=native` |
| 39 | 🦙 Load Ollama button (ollamapull/ route) | ModelSheet ollamapull/ rows + Settings › Endpoints "Load Ollama" row (disabled row + ReasonChip) | **Excluded**: ollamapull |
| 40 | Ollama settings dialog | Settings › Endpoints › Local LLM host (URL / port only) | Install / pull excluded |
| 41 | ollamapull/ chat requests | ModelSheet ollamapull/ rows (disabled row + ReasonChip) | **Excluded** |
| 42 | Forced-stream batch log toggle | Settings › Response handling › Streaming (one switch) | Its excluded routes show a ReasonChip on the switch |
| 43 | Login-button labels and status snapshot | Accounts slot rows; LoginChip "ChatGPT #2 ✓" |  |
| 44 | Login control visibility per model and key pools | ModelSheet route row (`route_controls`) |  |
| 45 | Provider model polling for subscription routes | Models › Poll providers; ModelSheet refresh |  |
| 46 | Encrypted token storage | Automatic (no control); sign-in state shown in Accounts | Automatic (SecureStorage key) |

## 6. glossary (81)

| # | Feature | Mobile surface | Notes |
|---|---|---|---|
| 0 | Glossary Mode selector (8 modes) | Settings › Glossary › General; Plan card chip; Chat settings › Glossary policy |  |
| 1 | Mode-locked toggles (🔒 purple) | Glossary settings tabs: purple lock chip + reason |  |
| 2 | Extract Glossary (manual run / Stop) | ＋ sheet › Extract glossary; Book › Glossary › Extract; Glossaries ⋯ |  |
| 3 | Balanced/Full pre-translation auto extraction | Plan card; GlossaryApprovalCard | Automatic |
| 4 | Minimal mode extraction during translation | Settings › Glossary › Minimal |  |
| 5 | Single Pass mode | Settings › Glossary › General + Single Pass header prompt |  |
| 6 | Image-folder glossary extraction | Extract glossary with images / CBZ |  |
| 7 | Append Glossary to System Prompt | Settings › Glossary › General |  |
| 8 | Auto-Mapping (Auto-Fill) | General |  |
| 9 | Fuzzy Auto-Mapping + Similarity slider | General |  |
| 10 | Output-folder glossary auto-load | Automatic (no control); the effective glossary shows on the Plan card glossary chip | Automatic |
| 11 | Auto-load glossary after extraction | Automatic (no control); the Plan card glossary chip switches to the extracted file | Automatic |
| 12 | Sync auto-mapped glossary into output folder | Automatic (no control); visible in Book › Output › Glossary files | Automatic |
| 13 | 📄 Load Glossary (main toolbar) | Glossaries › Import; Plan card glossary chip → PlanGlossarySheet › Load file… |  |
| 14 | Map Glossaries to EPUBs dialog | BatchPlanCard / PlanGlossarySheet › Map glossaries |  |
| 15 | Loaded-glossary status label + ✕ Clear | Plan card glossary chip → PlanGlossarySheet (status + Clear ✕); chat glossary pill (×) |  |
| 16 | 🗑️ Delete glossary for selected inputs | Book › Glossary ⋯; Library selection bar › More › Delete glossary files (N); Glossaries file row ⋯ |  |
| 17 | ↩️ Restore Glossary | Book › Glossary ⋯ › Restore backup; Library selection bar › More › Restore glossary backup |  |
| 18 | Glossary mode welcome dialog | Welcome step 3 |  |
| 19 | Require 100% glossary progress before translation | General |  |
| 20 | Skip glossary retries for API_ERROR | General |  |
| 21 | Add Additional Glossary + Load Additional Glossary | General |  |
| 22 | Enable Unified Glossary | Settings › Glossary › Unified |  |
| 23 | Generate Unified Glossary | Glossaries › Unified › Rebuild now (job) |  |
| 24 | Unified Glossary Settings dialog | `/glossary/unified` |  |
| 25 | Compress Glossary Prompt | General |  |
| 26 | Consider Translated Column | General |  |
| 27 | Precise Term Matching + 'Whole term for' (All/Gender Entries/Custom/None) + Configure… | General + Configure subpage |  |
| 28 | Multipass: Exclude Already-Applied Entries | General |  |
| 29 | Log Match Differences (shadow mode) + verdict allowlist | General › "Log Match Differences" switch | Adapted: the per-glossary verdict allowlist (always_keep / always_drop) and "apply verdicts" stay a desktop CLI tool (`tools/glossary_match_report.py`); no mobile surface, the files round-trip untouched |
| 30 | Save Glossary Backup in Output | General |  |
| 31 | Gender tracker controls: Skip Gender Tracking / Ignore rare gender flips slider / Bias | General |  |
| 32 | Glossary Append Format prompt | General (PromptTile) |  |
| 33 | Entry Type Configuration | Balanced/Full (ListEditor) |  |
| 34 | Entry Type Filtering (Strict/Loose/No Filtering) | Balanced/Full |  |
| 35 | Skip identical entries / Filter CJK script entries | Balanced/Full |  |
| 36 | Custom Fields (additional columns) | Balanced/Full (ListEditor) |  |
| 37 | Duplicate Detection settings | Balanced/Full (+ name-matching sub-options) |  |
| 38 | Output Format: legacy CSV / legacy JSON | Balanced/Full |  |
| 39 | Glossary Target Language | Balanced/Full & Minimal |  |
| 40 | Balanced/Full Extraction Prompt + prompt profiles | Balanced/Full PromptTile + profile dropdown |  |
| 41 | Balanced/Full Extraction Settings | Balanced/Full |  |
| 42 | Glossary Anti-Duplicate Parameters dialog | Balanced/Full › Anti-duplicate subpage |  |
| 43 | Minimal tab: gender / description / smart-filter toggles | Minimal |  |
| 44 | Minimal Extraction Settings | Minimal |  |
| 45 | Minimal Glossary Extraction Prompt + profiles | Minimal |  |
| 46 | Glossary translation prompt / format instructions (config-only) | General › Advanced prompts ("Show advanced") | New UI for a config-only key |
| 47 | Glossary Refinement tab | Glossaries › Refinement |  |
| 48 | Save All Settings / Cancel | Auto-save + ⋯ Discard changes since opening |  |
| 49 | Editor: file picker combo, ◀/▶ nav, Browse, Force Refresh, auto-reload | Editor top bar (file ▾, ◀ ▶, ⋯ Reload, auto-reload dot) |  |
| 50 | Editor: Load (set as manual glossary) | Editor ⋯ › Use as manual glossary |  |
| 51 | Editor: parse/load glossary (CSV token/legacy, JSON list/dict) | Automatic (no control); every supported format opens in the Glossary editor | Automatic |
| 52 | Editor: cell edit (double-click / context Edit) | EntrySheet (tap row) | Adapted |
| 53 | Editor: column header value filters | Editor › Filter sheet |  |
| 54 | Editor: Save (Ctrl+S) with gender resolution | Editor toolbar Save |  |
| 55 | Editor: Update output files on save | Editor ⋯ switch |  |
| 56 | Editor: Hide unused entries | Editor ⋯ switch |  |
| 57 | Editor: Save As / Export Selection | Editor ⋯ |  |
| 58 | Editor: Delete Selected (Del) | Selection bar Delete; swipe |  |
| 59 | Editor: Undo / Redo | Editor toolbar |  |
| 60 | Editor: 📂 Backups | Editor ⋯ › Backups |  |
| 61 | Editor: ✏️ Edit in Notepad | Editor ⋯ › Edit raw (CodeEditor) | Adapted |
| 62 | Editor: Find / Replace (Ctrl+F) | Find / Replace sheet |  |
| 63 | Editor: Resolve Gender… (tracked conflicts) | EntrySheet › Resolve gender |  |
| 64 | Editor Advanced: Reload | Editor ⋯ › Advanced |  |
| 65 | Editor Advanced: Clean Empty Fields | Advanced |  |
| 66 | Editor Advanced: Remove Duplicates | Advanced |  |
| 67 | Editor Advanced: Backup Settings | Advanced |  |
| 68 | Editor Advanced: Trim Entries | Advanced |  |
| 69 | Editor Advanced: Filter Entries | Advanced |  |
| 70 | Editor Advanced: Convert Format | Advanced |  |
| 71 | Editor Advanced: About Format | Advanced (info sheet) |  |
| 72 | Editor: font size zoom | Editor ⋯ › Text size |  |
| 73 | Parallel EPUB Pair (cross-check raw vs existing translation) | `/glossary/parallel-pair`: Auto-offset switch, Re-map, −/+ offset, row re-map sheet, Set unmapped, Pair wrapper prompt + profiles; the saved mapping is restored on reopen |  |
| 74 | Include book title in glossary / Auto-inject book title | Settings › Metadata, TOC & headers (linked from Glossary › General) |  |
| 75 | Emergency Glossary Compliance | Settings › Processing |  |
| 76 | Glossary / Refinement API key pools | Keys (Glossary, Glossary refinement); KeyPoolTiles |  |
| 77 | Translation-time glossary application | Automatic (no control); the per-message "Glossary terms used" sheet shows what was applied | Automatic |
| 78 | Glossary storage layout / legacy migration / book rename | Automatic (no control); glossary files appear in Glossaries and Book › Output | Automatic |
| 79 | Glossary Progress manager (cross-area) | Book › Glossary tab; Tools › Glossary progress |  |
| 80 | Direct Text glossary policy and approval (cross-area) | Chat settings › Glossary; GlossaryApprovalCard (+ Always accept); ManualGlossarySheet; `jobs.action` "Glossary ready" notification (Accept / Review) whenever the chat is not on screen; Settings › Notifications & background › Glossary review | New: the notification's Accept action and "Always accept generated glossaries" (mobile only; Prefs `chat_auto_accept_glossary` + sidecar `auto_accept_glossary`; `JobService._job_ask` answers the shared gate; tests_host/test_glossary_auto_accept.py). The desktop always asks (DISCREPANCIES "Device fixes 2026-10-08") |

## 7. other-settings (175)

| # | Feature | Mobile surface | Notes |
|---|---|---|---|
| 0 | Dialog: Save Settings | Auto-save + ⋯ Save now / Discard changes since opening | Adapted |
| 1 | Translate Book Title / Metadata | Settings › Metadata, TOC & headers › Metadata |  |
| 2 | Configure All (translation prompts dialog) | Metadata › "Configure all prompts" subpage |  |
| 3 | Custom Metadata (metadata fields dialog) | Metadata › Custom metadata (ListEditor) + Metadata translation mode |  |
| 4 | Metadata Keys (pool preview) | Metadata › KeyPoolTile → Keys (Metadata) |  |
| 5 | Skip .txt book title translation | Metadata |  |
| 6 | Skip .pdf book title translation | Metadata |  |
| 7 | Skip image title translation (use original filename) | Metadata |  |
| 8 | Skip title tag translation | Metadata |  |
| 9 | Configure Title Prompt (image-only title tag prompt) | Metadata › PromptTile |  |
| 10 | Include book title at top of glossary (during generation) | Metadata |  |
| 11 | Auto-inject book title (loaded glossaries only) | Metadata |  |
| 12 | Use & Translate TOC / PDF bookmarks | Metadata, TOC & headers › TOC |  |
| 13 | Skip duplicate TOC translation | TOC |  |
| 14 | Deduplicate TOC / Deduplicate using translated titles | TOC |  |
| 15 | Use TOC fallback titles (Legacy) | TOC |  |
| 16 | TOC entries per batch | TOC |  |
| 17 | Delete TOC Files | TOC ActionTile; Tools › Headers & metadata | Confirm |
| 18 | Batch Translate Headers + Headers per batch | Metadata, TOC & headers › Headers |  |
| 19 | Header translation help (ℹ️) | Headers ⓘ sheet |  |
| 20 | Update headers in HTML files | Headers |  |
| 21 | Save translations to .txt | Headers |  |
| 22 | Ignore header | Headers |  |
| 23 | Remove duplicate H1-H6+P pairs | Headers |  |
| 24 | ⚠️ Use Sorted Fallback | Headers (warning badge) |  |
| 25 | Translate Headers Now (toggles to Stop) | Headers ActionTile (job); Tools › Headers & metadata |  |
| 26 | Delete Header Files | Headers ActionTile; Tools › Headers |  |
| 27 | Failed-entry retry attempts (TOC + headers) | Metadata, TOC & headers |  |
| 28 | Validate EPUB Structure | Settings › EPUB output ActionTile; Tools › Converter |  |
| 29 | Use NCX-only Navigation (Compatibility Mode) | EPUB output |  |
| 30 | EPUB Layout (Auto / EPUB2 / EPUB3) | EPUB output |  |
| 31 | Attach CSS to Chapters + Load CSS… / Clear | EPUB output (PathTile) | Adapted: imported into app data |
| 32 | Custom Fonts: Load Font… | EPUB output (PathTile) | Adapted |
| 33 | Use HTML Method for EPUB | EPUB output |  |
| 34 | Retain source extension (no 'response_' prefix) + Rename Files | EPUB output + ActionTile |  |
| 35 | Download remote image URLs + Threads + Interval | EPUB output › Remote images |  |
| 36 | Include previous source text in history/memory (Not Recommended) | Settings › Context & memory |  |
| 37 | Rolling Summary: Role (user/system/both) | Context & memory › Rolling summary |  |
| 38 | Rolling Summary: Max tokens | Context & memory |  |
| 39 | Configure Memory Prompts | Context & memory › PromptTiles |  |
| 40 | RS Keys (rolling summary key pool) | Context & memory › KeyPoolTile |  |
| 41 | Application Updates: Check for Updates + Check on startup | About › Updates | (U9) Install excluded |
| 42 | Config Backup: Create Backup / Restore Backup | Data › Backup & restore |  |
| 43 | Default Output Folder Override | Data › Storage › Output folder | Adapted: app storage / iOS Documents (+ Android "Mirror outputs to Downloads/Glossarion") |
| 44 | Auto DPI Scale / GUI Scale Factor / GUI Font Scale | Settings › Appearance › "Auto DPI / GUI scale" (disabled row + ReasonChip); Appearance › Text scale replaces it | **Excluded**: DPI scaling (the OS handles it); value preserved |
| 45 | Enable streaming responses (OpenAI-compatible) | Settings › Response handling › Streaming (one switch) |  |
| 46 | Stream thinking/reasoning logs | Settings › Response handling › Streaming (one switch) |  |
| 47 | Allow streaming logs during batch mode | Settings › Response handling › Streaming (one switch) |  |
| 48 | Allow forced-stream batch log (AuthGPT/AuthGrok/AuthGem/AuthCD/AuthZA/Arena/Antigravity/OcAgy) | Settings › Response handling › Streaming (one switch) | Its excluded routes show a ReasonChip on the switch |
| 49 | GPT / OpenRouter / NIM / OpenCode Thinking: Enable + Effort | Settings › Thinking & reasoning; ModelSheet › Thinking |  |
| 50 | Use OR token budget instead of Effort + OR Thinking Tokens | Thinking |  |
| 51 | ⚠️ Force reasoning parameters on unknown routes | Thinking |  |
| 52 | Service Tier (off/standard/flex/fast/priority) + Force on unknown routes | Settings › Provider options & safety |  |
| 53 | Gemini Thinking: Enable + Budget + Level (Gemini 3) | Thinking |  |
| 54 | Enable thoughts (include model reasoning metadata) | Settings › Response handling › Streaming (one switch) | Follows Streaming (the desktop stream-thinking lock) |
| 55 | DeepSeek Thinking: Enable (DeepSeek & Chutes) + Effort (V4) + Use Responses API format | Thinking |  |
| 56 | Anthropic Extended Thinking: Enable + Budget + Force Adaptive + Effort | Thinking |  |
| 57 | NIM / AuthND Token Helpers (auto limits, token concurrency, subprocess limit, token timeout) | Settings › Response handling › NIM/AuthND helpers | Adapted: the subprocess-limit field is shown disabled with a ReasonChip (no subprocesses) |
| 58 | Dispatch order timeout (s) | Response handling |  |
| 59 | Enable TOR proxy rotation (search/opera, ocz/) | Settings › Response handling & retries › "Tor rotation" (disabled row + ReasonChip) | **Excluded**: Tor; value preserved |
| 60 | Gemini Free Browser Chunking (search/gemini) | Response handling (shown when the search/gemini experimental route is enabled) | Adapted: experimental WebView route (U9); every chunking setting applies; not in the exclusion list |
| 61 | Parallel Extraction: Enable + Workers | Response handling › Parallel extraction | Threads; capped by CPU |
| 62 | Enable GUI Responsiveness Yield | Settings › Response handling & retries › "GUI Responsiveness Yield" (disabled row + ReasonChip) | **Excluded**: Qt event-loop workaround; the mobile UI thread never blocks. Value preserved |
| 63 | Translation Keys (Main Pool) status + Configure API Keys | Response handling › KeyPoolTile → Keys |  |
| 64 | Translation Input->Output Compression Factor (Auto + manual) | Response handling |  |
| 65 | Resume incomplete EPUB/PDF chapters from saved chunks | Response handling |  |
| 66 | HTTP Timeouts & Connection Pooling | Response handling › HTTP |  |
| 67 | Stop Logic: Graceful Stop + Wait for all chunks | Response handling › Stop logic |  |
| 68 | API Request Retries: Maximum retry attempts + Indefinite Rate Limit Retry | Response handling › Retries |  |
| 69 | Auto-retry Truncated Responses + Token constraint + Truncated attempts + Truncation Keys | Response handling › Truncation |  |
| 70 | Auto-retry Silent Truncation (Char-ratio) | Truncation |  |
| 71 | Auto-retry Slow Processing (API Timeouts) | Truncation |  |
| 72 | Auto-retry Duplicate Content + Check last N chapters + Detection Method | Response handling › Duplicates |  |
| 73 | Configure AI Hunter (sub-dialog) | Duplicates › AI Hunter subpage |  |
| 74 | Save interrupted chapters | Response handling › Failure saving |  |
| 75 | Save blocked/prohibited responses | Failure saving |  |
| 76 | Disable empty response = safety filter check | Failure saving |  |
| 77 | Treat missing finish reason as prohibited content | Failure saving |  |
| 78 | Preserve Original Text on Failure | Failure saving |  |
| 79 | Disable QA marker checks + Marker length limit | Response handling › QA markers |  |
| 80 | Manage Refusal Patterns (Enabled/Disabled) | Response handling › Refusal patterns; Keys footer |  |
| 81 | Emergency Paragraph Restoration | Settings › Processing |  |
| 82 | Emergency Image Restoration | Processing |  |
| 83 | Emergency Glossary Compliance (mode, Configure…, min raw length) | Processing (+ Configure) |  |
| 84 | Enable Decimal Chapter Detection (EPUBs) | Processing |  |
| 85 | Fix Empty Attribute Tags (EPUB) / (Extraction) | Processing |  |
| 86 | Fix Stray p> Text (EPUB) | Processing |  |
| 87 | Number Spacing Tokenization Fix | Processing |  |
| 88 | Output SDLXLIFF | EPUB output › Sidecars |  |
| 89 | Output MD / Output TXT + Generate MD / Generate TXT | EPUB output › Sidecars + ActionTiles; Tools › Converter |  |
| 90 | Skip Thinking for Lightweight Tasks (Book Title / Metadata / TOC-Header) + Lightweight Thinking slider | Thinking |  |
| 91 | Configure Chunk Prompt (sub-dialog) | Processing › PromptTile |  |
| 92 | Break Split Count | Processing |  |
| 93 | Text Extraction Method: Standard (BeautifulSoup) / Enhanced (html2text) | Processing › Extraction (SegmentedTile) |  |
| 94 | [Standard] Fix Stray p> Text (BeautifulSoup) | Extraction (Standard only) |  |
| 95 | [Enhanced] Preserve Markdown Structure | Extraction (Enhanced only) |  |
| 96 | [Enhanced] SKIP markdown -> html tag conversion | Extraction |  |
| 97 | [Enhanced] Allow AI-created Markdown headers | Extraction |  |
| 98 | [Enhanced] Enable Single Line Break | Extraction |  |
| 99 | [Enhanced] Convert <br> tags to <p> paragraphs + Apply to Existing Outputs | Extraction + ActionTile; Tools › Converter |  |
| 100 | [Enhanced] Escape Markdown specials (escape_snob) | Extraction |  |
| 101 | [Enhanced] Keep asterisk-only lines as text (prevent <hr>) | Extraction |  |
| 102 | [Enhanced] Use markdown2 Converter (Legacy) | Extraction |  |
| 103 | EPUB File Filtering Level: Smart / Comprehensive / No Filtering | Extraction |  |
| 104 | Force BeautifulSoup for DeepL / Google Translate / Google Free | Extraction |  |
| 105 | Disable Section Merging | Extraction |  |
| 106 | Request Merging + Chapters per request + {split_marker_instruction} chip | Processing › Request merging |  |
| 107 | Split the Merge / Disable Fallback / Auto-retry Split Failures + Attempts | Request merging |  |
| 108 | Translate Special Files (Skip Override) + Edit Keywords panel | Processing › Special files (keyword ListEditor) |  |
| 109 | Translate All Numbered HTML Files | Special files |  |
| 110 | Never consider in between files as special | Special files |  |
| 111 | Disable Image Gallery in EPUB | EPUB output |  |
| 112 | Disable Automatic Cover Creation | EPUB output |  |
| 113 | Skip Non-Spine Special Files in EPUB | EPUB output |  |
| 114 | Skip Unreferenced Images in EPUB | EPUB output |  |
| 115 | PDF Input: Output format (pdf/epub) | Settings › PDF › Input |  |
| 116 | PDF Input: Async page threshold | PDF › Input (disabled row + ReasonChip) | Adapted: PDF extraction runs single-process on mobile; value preserved |
| 117 | PDF Input: Extraction workers (auto / 1..cores) + Auto | PDF › Input (disabled row + ReasonChip) | Adapted: PDF extraction runs single-process on mobile; value preserved |
| 118 | PDF Input: Use PDF table of contents for sections | PDF › Input |  |
| 119 | PDF Input: Render mode | PDF › Input |  |
| 120 | PDF Input: Paragraph alignment / Header alignment / Paragraph justification / RTL layout | PDF › Input layout |  |
| 121 | Disable 0-based Chapter Detection | Processing › Chapter numbering |  |
| 122 | Chapter Number Offset | Chapter numbering |  |
| 123 | Enable post-translation Scanning phase + Mode | Processing › Post-translation scan; Plan card |  |
| 124 | Batching Mode: Conservative (× multiplier) / Direct / No batching | Processing › Batching mode |  |
| 125 | Fallback Keys (pool preview) | Provider options & safety › KeyPoolTile |  |
| 126 | Disable API Safety Filters (Gemini, Groq, Fireworks…) + Threshold | Provider options & safety |  |
| 127 | Use HTTP-only for OpenRouter/NVIDIA (bypass SDK) | Provider options |  |
| 128 | Disable compression for OpenRouter (Accept-Encoding: identity) | Provider options |  |
| 129 | Preferred OpenRouter Provider (editable combo) | Provider options (editable Dropdown) |  |
| 130 | Output Mode: Text / Vision / Image / Video / Audio / Refinement | Settings › Translation defaults › Output mode (linked from Image & vision); composer output-mode row for chat |  |
| 131 | Process Long Images (Web Novel Style) | Image & vision; Mode options › Vision |  |
| 132 | Hide labels and remove OCR images | Image & vision; Mode options › Vision |  |
| 133 | [Image mode] Output Resolution 1K/2K/4K + Image Keys | Mode options › Image; Image & vision |  |
| 134 | [Video mode] Video Duration (5/10/15/20/30/60 s) + Video Resolution (360p/480p/720p/1080p) | Mode options › Video; Image & vision |  |
| 135 | [Vision] Configure Vision OCR Prompt + Vision Keys | Mode options › Vision; Image & vision |  |
| 136 | [Vision] Skip translation | Mode options › Vision |  |
| 137 | [Vision/Image] Batch Vision API requests + slot count | Mode options |  |
| 138 | [Vision] Keep OCR image | Mode options › Vision |  |
| 139 | Watermark Removal (Enable / Save Cleaned Images / Advanced FFT) | Image & vision › Watermark | Adapted: FFT may be slow; warning note |
| 140 | Image grid: Min image height / Max images per chapter / Chunk height / Chunk overlap % / Overlap floor px | Image & vision › Chunking |  |
| 141 | Direct image legacy mode (single multimodal request for tall chunks) | Chunking |  |
| 142 | Smart line-boundary chunking | Chunking |  |
| 143 | Fuzzy OCR chunk dedupe | Chunking |  |
| 144 | Configure Image Chunk Prompt (sub-dialog) | Image & vision › PromptTile |  |
| 145 | Configure GTool Scan Prompt (sub-dialog) | Image & vision › PromptTile; Tools › RPG Maker |  |
| 146 | Configure Image Compression (sub-dialog) | Image & vision › Compression subpage |  |
| 147 | Enable Anti-Duplicate Parameters | Settings › Anti-duplicate |  |
| 148 | Anti-Dup Core tab: Top-P, Min-P, Bypass Min-P allowlist, Top-K, Frequency/Presence penalty | Anti-duplicate › Core |  |
| 149 | Anti-Dup Advanced tab: Repetition Penalty, Candidate Count (Gemini) | Anti-duplicate › Advanced |  |
| 150 | Anti-Dup Stop Sequences tab | Anti-duplicate › Stop sequences (ListEditor) |  |
| 151 | Anti-Dup Logit Bias tab | Anti-duplicate › Logit bias (key/value) |  |
| 152 | Anti-Dup Reset to Defaults | Anti-duplicate ⋯ Reset |  |
| 153 | AuthZA / GLM Access Mode: Use auto-provisioned API key with Z.AI General API | Settings › Endpoints › "AuthZA / GLM access mode" (disabled row + ReasonChip) | **Excluded**: authza; value preserved |
| 154 | Enable Custom OpenAI Endpoint + Override API Endpoint (+Clear) + Azure API Version | Settings › Endpoints |  |
| 155 | Endpoint quick-paste shortcuts (double-click) | Endpoints: quick-paste chips under URL fields | Adapted: chips instead of double-click |
| 156 | Enable Custom Image Edit Endpoint + URL (+Clear) | Endpoints |  |
| 157 | TTS Voice/File | Endpoints › TTS; Mode options › Audio |  |
| 158 | Show More Fields: Groq/Local Base URL, Fireworks Base URL | Endpoints › More endpoints |  |
| 159 | Enable Gemini Custom Endpoint + URL (+Clear) + quick paste | Endpoints |  |
| 160 | Override Gemma routing | Endpoints |  |
| 161 | Enable Anthropic Custom Endpoint + Anthropic Base URL (+Clear) + quick paste | Endpoints |  |
| 162 | Test Connection | Endpoints › Test connection (live result list) |  |
| 163 | Debug Mode ON/OFF + Check Environment Variables | Data › Logs & diagnostics | Output redacted |
| 164 | Output Settings: Enable PDF Output | Settings › PDF › Output |  |
| 165 | Use Rapid Workspace Compiler | PDF › Output | PyMuPDF |
| 166 | Generate Table of Contents + Include Page Numbers in TOC | PDF › Output |  |
| 167 | Page Numbers + Alignment (left/center/right) | PDF › Output |  |
| 168 | Render Batch Size (25/50/100/150/200, editable) | PDF › Output |  |
| 169 | Fast Rendering | PDF › Output |  |
| 170 | Quality Settings: Enable Image Compression + Quality + Exclude Cover + Exclude .gif | PDF › Quality; EPUB output |  |
| 171 | Danger Zone: Reset Settings to Defaults | About › Danger zone (backup first) |  |
| 172 | [Main window, not in Other Settings] Context Mode combo + rolling-summary fields | Context & memory; Plan card |  |
| 173 | [Main window] Prompt profile management (bound from other_settings.py) | Profiles & prompts |  |
| 174 | [Main window] Other toolbar items touching settings | Settings home quick-action chips (Save now · Backup · Import / Export profiles) |  |

## 8. progress-retranslation (65)

| # | Feature | Mobile surface | Notes |
|---|---|---|---|
| 0 | Open Progress Manager (routing) | Book › Chapters; Tools › Progress (`?out=`) for chat workspaces, non-Library outputs, ZIPs, image folders; Job card › Progress |  |
| 1 | Output folder resolution and creation | Book › Chapters and Tools › Progress header: output-folder chip (tap → Files); "📁 Created: <folder>" snackbar | Automatic resolution |
| 2 | Workspace source link (source_epub.txt + Library registry) | Automatic (no control) on open; the raw link shows on Book › Output › Raw source row | Automatic |
| 3 | Auto-discovery of existing outputs | Automatic (no control) on full refresh (⟳ Refresh / pull-to-refresh) | Automatic. A desktop regex bug is recorded; it is fixed only in a separate, user-approved commit |
| 4 | Missing-file cleanup on open | Automatic (no control) during a read-write refresh | Automatic |
| 5 | EPUB OPF spine view (reading order) | Book › Chapters list |  |
| 6 | Non-OPF chapter list (TXT/CSV/JSON/PDF/HTML/Subtitles) | Chapters (fallback rows) |  |
| 7 | PDF bookmark section rows | Chapters "Section 007 · Pages 41-58" |  |
| 8 | Subtitle file and subtitle-ZIP rows | Chapters subtitle rows "Batches c/t" |  |
| 9 | Metadata (metadata.json) phase rows | Chapters pinned metadata rows |  |
| 10 | TOC.txt / translated_headers.txt artifact rows | Chapters artifact rows; RECYCLED 3-choice dialog |  |
| 11 | Chunk child rows | Expandable parent rows "↳ Chunk i/T" |  |
| 12 | PDF Vision OCR summary row | Pinned summary row |  |
| 13 | Image Generation summary row | Pinned summary row |  |
| 14 | Row status vocabulary, icons and colors | StatusAvatar + StatusChip (PM vocabulary) |  |
| 15 | Output-mode aware view (Text/Vision/Image/Video/Audio/Refinement) | Mode badge; labels Reset TTS / Not Refined / No TTS |  |
| 16 | Statistics bar with jump-to-status | StatusChipRow (tap filters, long-press jumps) |  |
| 17 | Show special files toggle | Chapters filter menu (persisted `epub_details_show_special_files`) |  |
| 18 | Show Model Info toggle | Chapters filter menu (persisted) |  |
| 19 | Selection controls | Selection top bar (Select all, Select ▾ by status, Clear) |  |
| 20 | Retranslate Selected | Bottom bar Retranslate (verbatim confirm) |  |
| 21 | Reset TTS Selected (Audio mode) | Bottom bar label swaps in audio mode | Label consistency fixed |
| 22 | Remove QA Failed Mark | Bottom bar |  |
| 23 | Remove Pending Mark | More ▾ |  |
| 24 | Remove refinement status | More ▾ |  |
| 25 | Restore In Progress Status | More ▾ |  |
| 26 | Refresh and live auto-refresh | 2 s signature poll while visible; ⟳ Refresh / pull-to-refresh = full reconcile | Adapted: no QFileSystemWatcher |
| 27 | Open File | Row ⋯ › 📂 Open file (TextEditor read-only / Share) | Adapted |
| 28 | Edit File (find QA issue) | Row ⋯ › TextEditor at the QA hit | Adapted: replaces Notepad++ |
| 29 | Copy QA issue | Row ⋯ |  |
| 30 | Open in EPUB reader | Row tap → Reader at chapter |  |
| 31 | Resolve QA issue - LLM token / empty-attribute repair | Row ⋯ › Resolve QA issue → Before / After sheet |  |
| 32 | Resolve QA issue - raw foreign text (single-entry Partial.b) | Row ⋯ › Resolve QA issue → RESOLVE_QA job | Engine-gated |
| 33 | Insert Missing Image | Row ⋯ / More ▾ |  |
| 34 | Do not skip (remove special-file keyword) | Skipped row ⋯ "⏭️ Do not skip (remove keyword '…')" |  |
| 35 | Open / Delete Audio File | Row ⋯ › Play audio (inline) / Delete audio |  |
| 36 | Manual editing toggle (source-only sidecars) | Chapters ⋯ › Manual editing ("Creating sidecars… i/N") |  |
| 37 | Edit Translation (open SDLXLIFF reviewer) | Chapters ⋯ / row ⋯ → Tools › SDLXLIFF reviewer |  |
| 38 | Multi-file Progress Manager | Tools › Progress output dropdown | Adapted: Library books get their own pages |
| 39 | Image folder / single image retranslation | Chapters thumbnail-grid variant (Mark as Skipped / Delete Selected) | Ported to v2.1 structure |
| 40 | Parallel EPUB pair Progress Manager | Tools › Progress (parallel context) |  |
| 41 | Glossary Progress dialog | Book › Glossary tab; Tools › Glossary progress |  |
| 42 | Glossary Progress: Mark as Completed / Remove from progress | Glossary tab selection bar + row ⋯ (Mark as completed / Remove from progress) | Writes use the extractor's lock + atomic replace (desktop plain dumps recorded as a bug) |
| 43 | Glossary Progress: Footnotes and completed summary | Glossary tab row ⋯ › Show footnote; selection More › Show glossary footnote(s) / Generate completed summary (share .md) |  |
| 44 | Glossary Progress: Manual glossary refinement | Glossary tab ✨ Refinement → preview sheet → job; row ⋯ › ✨ Refine this |  |
| 45 | Glossary Progress: Open Folder / Open Glossary / Select All | Glossary path row: Files / ✏️ Open Glossary / Select All |  |
| 46 | SDLXLIFF reviewer: book navigation and piece list | Tools › SDLXLIFF (Dropdown / side list) |  |
| 47 | SDLXLIFF reviewer: alignment and status analysis | Legend chips + row status colours |  |
| 48 | SDLXLIFF reviewer: Compact layout editing | Row cards |  |
| 49 | SDLXLIFF reviewer: Notepad layout (full-document editor) | Tablet: CodeEditor Notepad layout; phone: compact layout, Notepad toggle disabled row + ReasonChip "Tablet layout only" | Adapted |
| 50 | SDLXLIFF reviewer: Machine Translation previews | MT sheet (provider, preview, Inject MT) |  |
| 51 | SDLXLIFF reviewer: Flag Inaccurate | Flag button + threshold |  |
| 52 | SDLXLIFF reviewer: Mark as Completed / Undo | Piece ⋯ |  |
| 53 | SDLXLIFF reviewer: auto-refresh and sidecar regeneration | Refresh action + 2 s poll while visible |  |
| 54 | SDLXLIFF reviewer: manual-editing save semantics | Automatic (no control); row-card saves in Tools › SDLXLIFF use the shared semantics | Automatic (shared) |
| 55 | SDLXLIFF reviewer: window controls | Tools › SDLXLIFF reviewer is a full-screen route | n/a on mobile |
| 56 | SDLXLIFF input translation (.sdlxliff source files) | Attach .sdlxliff |  |
| 57 | HTML SDLXLIFF sidecar output during translation | Settings › EPUB output › Output SDLXLIFF |  |
| 58 | Review Generator dialog | Tools › Review (book picker, mode chips, Review system prompt PromptTile, Final prompt, Start / Review all / Stop) |  |
| 59 | Review modes: 50/50 Split, Full Review (chunked), Wrap Chunks, Volume Mode | Tools › Review mode chips |  |
| 60 | Volume file order dialog | Review › Volume order (ReorderableListView) |  |
| 61 | Final Prompt editor | Review › Final prompt PromptEditor (Review system prompt PromptTile above it) |  |
| 62 | Start Review / Review all Files / Stop | Review buttons (REVIEW jobs) |  |
| 63 | Review output pane, font settings, save/delete/restore | Review output pane (Markdown, Save / Delete / Restore) + ⋯ › Display sheet (review_font_family / size / line height / colour / header spacing / spacing / list gap, Reset) |  |
| 64 | Workspace reader manifest (PDF/HTML workspaces) | Reader PDF-workspace mode | Automatic |

## 9. qa-epub-pdf (68)

| # | Feature | Mobile surface | Notes |
|---|---|---|---|
| 0 | QA Scan launcher (toolbar 'QA Scan' button) | Tools › QA Scanner; ＋ sheet › QA scan; `/qa`; Book ⋯; Job card; chat Result card › QA scan (Quick Scan of the chat workspace, `qa_model.chat_qa_job`) |  |
| 1 | Detection mode: AI Hunter | QA mode card | Threads |
| 2 | Detection mode: Aggressive | Mode card |  |
| 3 | Detection mode: Quick Scan | Mode card (Recommended) |  |
| 4 | Detection mode: Custom + Custom Mode Settings dialog | Mode card → Custom settings sheet |  |
| 5 | Quick Scan duplicate sample size | QA home field; Settings › QA | Adapted: mobile default 0 (duplicate check off) when unset + one-time migration of a saved 1000; desktop keeps 1000; chat scans use the same value |
| 6 | Open QA Report | QA › Reports → QA report viewer `/tools/qa/report/<rid>` (WebView on Android/iOS; Markdown + "Open in browser" fallback on Windows/Linux dev) |  |
| 7 | Auto-search output folder | QA home switch |  |
| 8 | Source file selection for word-count/resource checks | SourcePicker; Settings › QA › Word count |  |
| 9 | Source/folder name mismatch warning | Dialog + setting |  |
| 10 | Bulk multi-folder scan | SourcePicker multi-select |  |
| 11 | Direct Text exclusion | Automatic (no control); Direct Text workspaces are skipped by post-translation QA; Tools › QA Scanner does not pick them; the chat's own QA scan opts in (`allow_direct_text`, owner 2026-10-08) | Automatic |
| 12 | Post-translation scanning phase | Settings › Processing; Plan card |  |
| 13 | Foreign character detection: source/target language, threshold | Settings › QA › Foreign characters |  |
| 14 | Whitelist emoticon patterns + phrase editor | QA › Foreign characters (ListEditor) |  |
| 15 | Exclude ruby annotation tags | QA › Foreign characters |  |
| 16 | Additional excluded characters | QA › Foreign characters |  |
| 17 | Check encoding issues | QA › Detection options |  |
| 18 | Check excessive repetition | Detection options |  |
| 19 | Check translation artifacts | Detection options |  |
| 20 | Check AI artifacts + phrase editor | Detection options (ListEditor) |  |
| 21 | Detect AI thinking preambles + editor | Detection options (ListEditor) |  |
| 22 | ?! punctuation mismatch + excess punctuation | Detection options |  |
| 23 | Quotation mark mismatch + sub-options | Detection options |  |
| 24 | Glossary leakage check | Detection options |  |
| 25 | Potential truncation (heuristic) | Detection options |  |
| 26 | SDLXLIFF source->output tag checks | Detection options |  |
| 27 | File processing thresholds | QA › File processing |  |
| 28 | Word-count cross-reference | QA › Word count |  |
| 29 | Per-language length multipliers editor | QA › Word count (NumberTile grid) |  |
| 30 | Source resources check (images/links/tables/graphics) | QA › Word count |  |
| 31 | Header/HTML structure checks | QA › Additional checks |  |
| 32 | Silent truncation detection (heuristic + embeddings) | Settings › QA › Additional checks | The heuristic always ships; the embeddings method (sentence-transformers) is native-impossible and shown disabled with a ReasonChip |
| 33 | AI Truncation Detection | QA › Additional checks + Keys (AI truncation pool) |  |
| 34 | Report settings | QA › Report |  |
| 35 | Progress-file QA marking (qa_failed / chunk QA) | Book › Chapters QA rows | Automatic |
| 36 | Performance cache settings | QA › Performance |  |
| 37 | Use threads instead of processes | QA › Performance (locked ON) | Adapted: forced on mobile |
| 38 | AI Hunter max parallel workers | QA › AI Hunter performance |  |
| 39 | QA settings Save / Cancel / Reset to Default | Auto-save + ⋯ Discard / Reset to default |  |
| 40 | Stop QA scan | State machine (Job card / JobStrip) |  |
| 41 | EPUB Converter (compile translated folder to EPUB) | Tools › Converter; Book Compile; Job card Compile |  |
| 42 | EPUB layout mode (Auto/EPUB2/EPUB3) | Settings › EPUB output |  |
| 43 | NCX-only navigation (compatibility) | EPUB output |  |
| 44 | Attach CSS to chapters | EPUB output |  |
| 45 | CSS override (Load CSS… / Clear) | EPUB output (PathTile) |  |
| 46 | Custom fonts loader | EPUB output (PathTile) |  |
| 47 | Use HTML method for EPUB serialization | EPUB output |  |
| 48 | Retain source extension + Rename Files | EPUB output; Tools › Converter |  |
| 49 | Validate EPUB Structure | Tools › Converter; EPUB output |  |
| 50 | Image gallery page | EPUB output |  |
| 51 | Cover handling | EPUB output |  |
| 52 | Skip non-spine special files / unreferenced images; translate special files | EPUB output; Processing |  |
| 53 | HTML cleanup at compile | Processing |  |
| 54 | Batch header translation during compile (OPF-based) | Metadata, TOC & headers |  |
| 55 | Translate Headers Now (standalone) + EPUB rebuild | Tools › Headers & metadata |  |
| 56 | Delete Header Files / Delete TOC.txt | Tools › Headers & metadata; Settings |  |
| 57 | Use source TOC (toc.ncx / nav.xhtml) + TOC translation | Metadata, TOC & headers › TOC |  |
| 58 | Metadata / book-title translation at compile | Metadata |  |
| 59 | Image compression (EPUB WebP) | EPUB output / PDF › Quality |  |
| 60 | Create PDF after EPUB | Settings › PDF › Output | Adapted: the PyMuPDF HTML renderer replaces WeasyPrint; layout may differ |
| 61 | PDF workspace compile (PDF input -> translated PDF) | Book › Compile ▾ PDF; Tools › Converter | Adapted: PyMuPDF |
| 62 | Image ZIP/CBZ -> EPUB | Automatic (no control); Plan card shows "CBZ → EPUB on start" | Automatic |
| 63 | HTML chapter ZIP / standalone HTML -> EPUB | Automatic (no control); Plan card shows "ZIP → EPUB on start" | Automatic |
| 64 | EPUB-in-ZIP passthrough | Automatic (no control); Plan card shows the detected EPUB | Automatic |
| 65 | Library organized-copy replacement | Automatic (no control) after compile; the Library card refreshes | Automatic |
| 66 | EPUB directory diagnostics (CLI) | Tools › Converter › Validate EPUB structure | The CLI itself is n/a |
| 67 | Standalone scanner CLI/Tk GUI | Tools › QA Scanner (the same scanner) | **Excluded**: Tk / CLI front end (desktop-only) |

## 10. manga (90)

| # | Feature | Mobile surface | Notes |
|---|---|---|---|
| 0 | Open Manga Translator panel | Tools › Manga; ＋ sheet › Manga; "Translate as manga" chip; `/manga` |  |
| 1 | Add Files (images + .cbz) | Manga › Files › Add files / Add ZIP-CBZ |  |
| 2 | Add Folder(s) | Manga › Files › Add folder (both platforms; `FilePicker.get_directory_path`, ZIP fallback when Android SAF fails) |  |
| 3 | Drag & drop files/folders/CBZ | Share / Open-with → IntentRouter "Manga translator" | Adapted |
| 4 | Remove Selected / Clear All | Files selection bar |  |
| 5 | Sort file list (Name / Number / Date / Reverse) and manual reorder | Files sort menu + ReorderableListView |  |
| 6 | Image Range filter | Files › Range field |  |
| 7 | Process Grouping (split first-level subfolders) | Manga › Files › Process grouping switch (split first-level subfolders) |  |
| 8 | Skip processing per file | Per-file "Process this image" switch |  |
| 9 | Persist selected files across sessions | Automatic (no control); the selection is restored when Files reopens | Automatic |
| 10 | OCR provider: custom-api (LLM vision OCR) | Manga › Settings › OCR provider |  |
| 11 | OCR provider: Google Cloud Vision | OCR provider (status chip; credentials PathTile) | Dependency rule (U9): google-cloud-vision 3.16.0 ships (the SDK path, like desktop); `google_vision_rest` stays the fallback |
| 12 | OCR provider: Azure Computer Vision | OCR provider (key / endpoint) |  |
| 13 | OCR provider: Azure Document Intelligence | OCR provider (status chip) | REST (`azure_document_intelligence_rest`, U8): the SDKs resolve (U9 check) but ocr_manager imports azure.ai.formrecognizer, never documentintelligence |
| 14 | OCR provider: RapidOCR | OCR provider + ONNX download (status chip) | Ships when pyclipper / shapely wheels resolve (check_mobile_wheels.py); otherwise disabled row + ReasonChip |
| 15 | OCR provider: manga-ocr (Japanese) | OCR provider list: manga-ocr row (disabled row + ReasonChip "Needs PyTorch") | **Excluded**: PyTorch model; no mobile wheels |
| 16 | OCR provider: Qwen2-VL (local) | OCR provider list: Qwen2-VL row (disabled row + ReasonChip) | **Excluded**: PyTorch |
| 17 | OCR provider: EasyOCR | OCR provider list: EasyOCR row (disabled row + ReasonChip) | **Excluded**: PyTorch |
| 18 | OCR provider: DocTR | OCR provider list: DocTR row (disabled row + ReasonChip) | **Excluded**: PyTorch |
| 19 | OCR provider: PaddleOCR (hidden) | OCR provider list: PaddleOCR row (disabled row + ReasonChip) | **Excluded**: PaddlePaddle is native-impossible; shown, never hidden |
| 20 | OCR provider status / Setup / Download model | OCR provider row status chip + model download manager |  |
| 21 | Edit OCR prompt and Full-Page-Context prompt | Settings › Context PromptTiles |  |
| 22 | Refresh from Main GUI (current main settings display) | Context › "Current translation settings" summary (always live) | Adapted |
| 23 | Full Page Context translation | Context |  |
| 24 | Per-region translation with rolling context/history | Automatic (no control); uses the main context settings (Settings › Context & memory) | Automatic |
| 25 | Visual context (include page image in translation requests) | Context |  |
| 26 | Image request quality override | Context |  |
| 27 | Manga glossary workflow (use loaded/generated glossary) | Settings › Glossary workflow |  |
| 28 | Load / Clear manga glossary; auto-load per source root | Glossary workflow |  |
| 29 | Compress Glossary Prompt (shared setting) | Glossary workflow |  |
| 30 | Save OCR/glossary debug subfolder | Glossary workflow |  |
| 31 | Create .cbz at translation end | Files switch + Output "Create CBZ" |  |
| 32 | Auto consolidate / Download Images | Files switch + Output "Download images" |  |
| 33 | Manga-specific output token limit | Context |  |
| 34 | Skip Inpainter | Settings › Inpainting |  |
| 35 | Inpaint method: Replicate API (cloud) | Inpainting (SecretTile key) |  |
| 36 | Inpaint method: Local - ONNX models (aot_onnx, anime_onnx, lama_onnx) | Inpainting + model download manager | onnxruntime |
| 37 | Inpaint method: Local - torch JIT/checkpoint models (aot, lama, anime, lama_official, mat) | Inpainting method list: torch JIT rows (disabled row + ReasonChip) | **Excluded**: PyTorch |
| 38 | Inpaint method: custom-image-edit (OpenAI-compatible image edit endpoint) | Inpainting (endpoint, prompt, batch) |  |
| 39 | Test custom image edit endpoint | Inpainting › Test (the desktop check and texts, `manga_env.test_custom_image_edit_endpoint`) · Image keys (`inpainter` pool) |  |
| 40 | Inpaint method: Hybrid | Inpainting method list: Hybrid row (disabled row + ReasonChip) | **Excluded**: depends on torch models |
| 41 | Local model file Browse / Load / Download / Model Info / status | Inpainting › Local / API model ⓘ (the desktop Model Information text, `manga_models.MODEL_INFO`) · "Import model file…" (`manga_<type>_model_path`, a private copy) · Model manager (download, delete) | Adapted: Browse is Import model file (U9) |
| 42 | Disable Performance Mode (local inpaint) | Inpainting |  |
| 43 | Inpainter preload / pool tracker | Inpainting method row status chip (Preloaded / Loading / Not downloaded / Needs key) | Adapted |
| 44 | Background settings (opacity, size, style Box/Circle/Wrap, preserve free text) | Settings › Rendering |  |
| 45 | Font size: algorithm, mode (Auto/Fixed/Multiplier), min/max, fit style, prefer larger, bubble scaling, line spacing, max lines, presets | Rendering |  |
| 46 | Constrain text to bubble / Safe area (mask/polygon) with scale | Rendering |  |
| 47 | Strict text wrapping / Force CAPS | Rendering |  |
| 48 | Font Style selection + Browse Custom Font | Rendering (FontTile, import) |  |
| 49 | Font color / Shadow (enable, color, offset X/Y, blur) | Rendering (ColorTile) |  |
| 50 | Reset rendering to defaults | Rendering › Reset (confirm; the desktop values, `manga_settings_defaults.RENDERING_RESET_VALUES`) | (U9) |
| 51 | Start / Stop batch translation (graceful and force stop) | Files › Start / state machine; Job card / JobStrip |  |
| 52 | Parallel panel translation | Settings › Advanced | Capped workers |
| 53 | Webtoon mode / format detection | Advanced / Preprocessing |  |
| 54 | Image preprocessing (enhancement, denoise, size limits) | Preprocessing |  |
| 55 | Inpainting HD strategy (original/resize/crop) and tiling | Preprocessing |  |
| 56 | Mask settings (dilation, kernel, per-type iterations, auto iterations, presets) | Settings › Mask; Inpainting › Mask presets (B&W Manga / Colored / Uniform, `manga_settings_defaults.MASK_PRESETS`) | Presets (U9) |
| 57 | OCR parameters (language hints, cloud confidence, detection mode, min region size, retries) | Settings › OCR params |  |
| 58 | Text region merging & filtering | OCR params |  |
| 59 | OCR batching, concurrency and ROI locality | OCR params |  |
| 60 | AI bubble detection: RT-DETR ONNX (default) | Manga › Settings › Detection (RT-DETR ONNX ModelDownloadRow: status · Download · Load) |  |
| 61 | AI bubble detection: RT-DETR (PyTorch/transformers) | Manga › Settings › Detection: RT-DETR PyTorch row (disabled row + ReasonChip) | **Excluded**: PyTorch |
| 62 | AI bubble detection: YOLOv8 (Speech/Text/Manga) and Custom model | Manga › Settings › Detection: YOLOv8 / custom detector rows (disabled row + ReasonChip) | **Excluded**: PyTorch / ultralytics |
| 63 | Detector model download / load / status | Manga › Settings › Detection (status chip, Download, Load / Unload, Delete) |  |
| 64 | Advanced: debug mode, concise logs, save intermediate images | Advanced › Debug |  |
| 65 | Advanced: performance (parallel processing, max workers, parallel rendering, RT-DETR concurrency, inpainting concurrency, cache, disable worker process) | Advanced › Performance | Adapted: capped for phones |
| 66 | Advanced: ONNX conversion, quantization, torch precision | Manga › Settings › Advanced: "ONNX conversion / quantization" (disabled row + ReasonChip) | **Excluded**: needs torch |
| 67 | Advanced: memory management and RAM cap | Advanced › Memory | Adapted |
| 68 | Experimental editing tools (Brush, Eraser) | Editor toolbar: Brush / Eraser (disabled + ReasonChip) | **Excluded**: experimental mask painting stays desktop only |
| 69 | Manual Edit settings (Translate This Text) | Settings › Manual edit |  |
| 70 | Settings dialog Save / Reset to defaults | Auto-save + ⋯ Reset |  |
| 71 | Preview: dual viewer (Source / Translated Output) with thumbnails and navigation | Manga › Editor (Source / Translated, page strip) |  |
| 72 | Preview: manual box/circle/lasso drawing, move/resize, delete, clear | Editor Edit mode tools; toolbar Clear boxes (confirm; the desktop's Clear Boxes halves, `manga_editor_core`) | Clear boxes (U9) |
| 73 | Preview workflow: Detect Text | Editor › Detect |  |
| 74 | Preview workflow: Clean | Editor › Clean |  |
| 75 | Preview workflow: Recognize Text | Editor › Recognize |  |
| 76 | Preview workflow: Translate / Translate All | Editor › Translate / Translate all |  |
| 77 | Preview per-box context menu | Long-press box → BoxSheet |  |
| 78 | Preview: drag translated text overlays and Save & Update Overlay (re-render) | Editor move tool; BoxSheet › Save & Update Overlay |  |
| 79 | Per-image editor state persistence | Automatic (no control); restored when a page reopens in the Editor | Automatic |
| 80 | Open output folder | Editor ⋯ › Files |  |
| 81 | Import OCR (batch OCR JSON) and drag-drop JSON | Editor ⋯ › Import OCR |  |
| 82 | Automatic OCR export per run + Open Auto-Saved OCR Files | Editor ⋯ › Auto-saved OCR files |  |
| 83 | Manual editor OCR export / import | Editor ⋯ |  |
| 84 | Translation log panel | Manga › Files LogConsole; Job detail |  |
| 85 | Multi-key / fallback keys / Vision / Image key pools for manga | Settings KeyPoolTiles → Keys (Vision, Inpainter, …) |  |
| 86 | Non-functional inpaint options 'ollama' and 'sd_local' | Inpainting method list: 'ollama' / 'sd_local' rows (disabled row + ReasonChip "Not functional on desktop either") | **Excluded**: non-functional on desktop as well; shown, never hidden |
| 87 | EPUB in-book image translation (Process Web Novel Images) | Settings › Image & vision › Process web novel images |  |
| 88 | Vision OCR source EPUB (OCR prepass EPUB) | Mode options › Vision; Settings › Image & vision |  |
| 89 | F11 fullscreen toggle for the manga dialog | Manga screens are full-screen Views | n/a: full screen by default |

## 11. library-reader (54)

| # | Feature | Mobile surface | Notes |
|---|---|---|---|
| 0 | Open Library (📚 button) | Drawer › Library |  |
| 1 | Dual library scan (In Progress + Completed) | Library (skeleton, "Scanning library…") |  |
| 2 | Auto-refresh library (2 s) | Quiet scan while visible and foregrounded; diffed by `card_signature` |  |
| 3 | Manual Refresh | ⋯ Refresh; pull-to-refresh |  |
| 4 | Search / filter cards | Library search "Filter title or tag…" |  |
| 5 | Sort | Filter sheet › Sort |  |
| 6 | Format filter | Filter sheet › Filter |  |
| 7 | Card size / view zoom | Filter sheet › Display › Density (all 11 desktop presets 2XS–6XL; `epub_library_card_size`) | Display clamp only; the stored value is preserved |
| 8 | Raw titles toggle (cards) | Filter sheet › Display |  |
| 9 | Library pagination | Paging by `epub_library_page_size` (20 / 50 / 100 / 250 / 500 / All) as the append increment | Adapted: no pager buttons |
| 10 | Tabs In Progress / Completed (+ remembered tab) | Shelf SegmentedButton (remembered) |  |
| 11 | Book card rendering | BookCard (exact ribbons, pills, badges) |  |
| 12 | Cover extraction & caching | Cover thumbs in `cache/covers` |  |
| 13 | Card selection (multi-select) | Long-press selection mode (`Container.on_long_press`) | Adapted |
| 14 | Open card (double-click) | Tap → Book page (both shelves, every type) | Adapted: desktop opens TXT in an editor and a PDF without a workspace in the system viewer; mobile shows the Book page, and ⋯ › ↗ Share hands the file to another app |
| 15 | Context menu: Open Book Details / Open in Reader / Open Translated EPUB / Open in EPUB reader / Open File | Card ⋯ ActionSheet (custom bottom sheet); Open File → ↗ Share; 📖 Open in Reader also for TXT books |  |
| 16 | Load for translation (single / N files) | Bulk "Translate" / ⋯ › Load for translation → TranslateSheet; in a chat: ＋ › From Library / "From Library" chip / `/library [title]` attach the book, Send → Plan "Save to: Library" (in-chat picker: `targets.order_library_rows`, the Library search + Date sort) | Output Folder Mismatch question first, as on desktop (`translate_sheet.confirm_output_root`, like `_ensure_output_override_matches`) |
| 17 | Translate Metadata (single / N EPUBs) | Bulk "Metadata"; Book › Overview |  |
| 18 | Compile EPUB / Compile PDF | Bulk "Compile"; Book › Overview / Output; Book page Compile uses the resolved workspace (organized books too) |  |
| 19 | Reveal source file / Reveal Translated File / Open Output Folder / Open Library Folder | FileBrowser; Share | Adapted |
| 20 | Copy Path | ⋯ Copy Path (developer setting) |  |
| 21 | Clear saved raw link | Bulk More; Book ⋯ |  |
| 22 | Delete (single / N) | Bulk Delete (two-level confirmation) |  |
| 23 | Import EPUB (raw, register in place) | In Progress FAB "Import EPUB" | Adapted: copies into Raw |
| 24 | Add Translation (register compiled EPUB) | Completed FAB "Add translation" | Adapted: copies into Translated |
| 25 | Drag-and-drop import | Share / Open-with → "Add to Library" | Adapted |
| 26 | Organize (N) | Automatic (no control): finished chat books move into the Library (auto-migrate); imports copy into Library/Raw | Automatic |
| 27 | Undo (N) Raw / Translated / All | No control: there is no manual move to undo (see #26) | Automatic |
| 28 | Legacy layout & registry migration | Automatic (no control); Library shows the migrated layout | Automatic |
| 29 | Scan for Raw (pair missing raws) | Chip "Scan for raw (N)" → `/library/scan-raw` |  |
| 30 | Library loading spinner / toast / F11 fullscreen | Skeletons, snackbars | F11 n/a |
| 31 | Book Details hero page | Book › Overview hero |  |
| 32 | Start reading / Read raw source | Overview primary / tonal buttons |  |
| 33 | Edit metadata.json | Overview › Edit → metadata form; ⋯ Edit raw JSON (CodeEditor) (the resolved workspace; an organized book edits its workspace's metadata.json) |  |
| 34 | Chapter list (Book Details) | Book › Chapters (PM parity; filters persisted via `epub_details_*` incl. rows per page); a book without a workspace lists its EPUB's own chapters (`BookDetailsModel.row_specs`) |  |
| 35 | Translate / Retranslate this chapter (Book Details) | Chapters row ⋯ (SINGLE_CHAPTER job) |  |
| 36 | Reader: open modes (plain / overlay / dual-path / PDF workspace) | Reader (plain / overlay / dual-path / PDF workspace) |  |
| 37 | Reader: EPUB loading & cache | Reader "Loading EPUB…"; reader cache |  |
| 38 | Reader: TOC sidebar & native TOC | Reader › Chapters drawer + "Native TOC" switch |  |
| 39 | Reader: layout modes | Aa › Layout (Single page / Scroll / Scroll all; Double on tablet) | Adapted |
| 40 | Reader: Raw / Translated toggle | Reader top bar Original · Translated · Bilingual | New: Bilingual |
| 41 | Reader: typography (font family, size, line spacing) | Aa › Text; pinch → font size |  |
| 42 | Reader: themes | Aa › Theme (6 themes + Follow app) |  |
| 43 | Reader: navigation | Tap zones / swipe; bottom bar prev / next + slider | Adapted |
| 44 | Reader: search across book | Reader search sheet |  |
| 45 | Reader: selection context menu (Google Translate / Define on web) | Selection chip row (+ Add to glossary, Ask in chat) |  |
| 46 | Reader: live 'Translate' current chapter (inline streaming translation) | Reader 🌐 → LivePanel (native) |  |
| 47 | Reader: images | Reader (served from localhost; tap image → viewer) |  |
| 48 | Reader: special files filtering | Chapters drawer "Show special files" |  |
| 49 | Reader: fullscreen & settings persistence | Reader full screen + `epub_reader_*` keys + per-book overrides, `reader_positions` and `reader_bookmarks` in `mobile_state.json` (Prefs) | New: reading positions |
| 50 | Reader: QtWebEngine prewarm & QTextBrowser fallback | Reader (flet-webview on Android/iOS; native fallback renderer elsewhere) | **Excluded** (Qt-specific); replaced |
| 51 | Open reader from Progress Manager / Direct Text attachments | Chapters row tap; Job card "Read" / "Open reader" |  |
| 52 | AuthND browser-token routing (shares the Library's QtWebEngine) | Accounts › Experimental (off-screen WebViewBridge) | Experimental (U9): `browser_driver` pages in the hidden WebView |
| 53 | Translation context history (HistoryManager) | Automatic (no control); context settings live in Settings › Context & memory | Automatic |

## 12. multikey-misc (52)

| # | Feature | Mobile surface | Notes |
|---|---|---|---|
| 0 | Open Multi API Key Manager (full) | Drawer / sidebar footer 🔑 API keys; Settings › Models & keys › Multi-Key Manager |  |
| 1 | One-pool preview windows | KeyPoolTiles → `/settings/keys/<pool>` | Adapted |
| 2 | Global rotation settings | Keys › Rotation card |  |
| 3 | Enable Translation Keys (main pool toggle) | Keys › Translation pool switch |  |
| 4 | Add API key (main pool) | Keys › ＋ Add key |  |
| 5 | Copy Current Key | Keys footer |  |
| 6 | Key list display and stats | Key cards |  |
| 7 | Reorder keys | Keys: ReorderableListView with drag handles (rotation order) |  |
| 8 | Inline per-key edits (double-click a column) | KeyEditor | Adapted |
| 9 | Bulk per-key context menu | Multi-select bulk bar |  |
| 10 | Per-key request context routing | KeyEditor › Request contexts |  |
| 11 | Individual (per-key) endpoint | KeyEditor |  |
| 12 | Custom request parameters per key | KeyEditor › 🧩 |  |
| 13 | Test Selected / Test All keys | Keys footer |  |
| 14 | Enable / Disable / Remove / Clear All keys | Bulk bar + pool ⋯ › Clear all keys (confirm) |  |
| 15 | Fallback Keys pool (prohibited-content fallback) | Keys › Fallback |  |
| 16 | Glossary Keys pool | Keys › Glossary |  |
| 17 | Dedicated pools (8, spec-driven) | Keys pool chips |  |
| 18 | Import keys (JSON) | Keys footer › Import |  |
| 19 | Export keys (all pools) | Keys footer › Export |  |
| 20 | Manage Refusal Patterns | Keys footer; Settings › Response handling |  |
| 21 | Searchable model picker in every key form | ModelPicker (field mode) |  |
| 22 | Model field context menu (Refresh Online Models / Manage Models) | ModelSheet title-row ⋯ and per-provider refresh; ModelPicker field mode |  |
| 23 | Per-key login slots (numbered account routes) | ModelPicker › Account aliases + Accounts | Supported routes only; excluded routes stay disabled |
| 24 | AuthZA (Z.AI) login button inside model field | ModelPicker authza/ rows + Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded**: authza |
| 25 | Ollama settings/install button next to each model field | Settings › Endpoints › Local LLM host; the install / pull action is shown disabled with a ReasonChip | **Excluded**: ollamapull install; host in Endpoints |
| 26 | Key tree zoom and height persistence | Keys app bar ⋯ › "Key list zoom" (disabled row + ReasonChip); the list follows Settings › Appearance › Text scale | n/a: replaced by app text scale; value preserved |
| 27 | Save / Close and live sync | Auto-save; `apply_key_pools_to_runtime` at job start |  |
| 28 | Model Manager: model list editing | Models › Models tab |  |
| 29 | Model Manager: Hide unpolled models | Models "Polled only" chip |  |
| 30 | Model Manager: Lock mouse wheel | Keys app bar ⋯ › "Lock mouse wheel" (disabled row + ReasonChip) | **Excluded**: desktop mouse-wheel guard; n/a on touch; value preserved |
| 31 | Model Manager: Poll Providers (online catalogs) | Models › 🌐 Poll providers |  |
| 32 | Auto-poll selected provider (24h TTL) | ModelSheet shimmer | Automatic |
| 33 | Model Manager: Custom Prefix routes | Models › Custom prefixes |  |
| 34 | Model Provider Information | ModelSheet ⓘ |  |
| 35 | Check for Updates (manual) | About › Updates › Check now | (U9) |
| 36 | Check for updates on startup | About › Updates | (U9) Once per session, only when a release has a file for this device |
| 37 | Install downloaded update | About › Updates "Download APK" / IPA link | **Excluded**: self-installing updates |
| 38 | Automatic config backup before every save | Data › Backup (list) | Automatic |
| 39 | Create config backup (manual) | Data › Backup › Create |  |
| 40 | Config Backup Manager (restore/delete/open folder) | Data › Backup (restore / delete) | Adapted: no "open folder" |
| 41 | Debug Mode toggle and env-var check | Data › Logs & diagnostics |  |
| 42 | enable_debug_mode.py / debug_env_vars.py CLI tools | Data › Logs & diagnostics (same functions) | CLI n/a |
| 43 | Target language list | ModelSheet › Language; all language fields |  |
| 44 | Emoticon whitelist for QA scanning | Settings › QA › Foreign characters |  |
| 45 | Gemini prohibited-use refusal detection | Automatic (no control); ErrorCard "Blocked by the provider's safety filter" | Automatic |
| 46 | EPUB metadata extraction/merge helpers | Automatic (no control); used by the Book › Overview metadata list | Automatic |
| 47 | Hard stop / shutdown progress restore | Jobs › Interrupted recovery | Automatic |
| 48 | Startup splash and module preloading | Native splash + immediate shell + warm import ("Preparing engine…" on Send and the drawer status) | Adapted: native splash, then the shell at once; until the warm import ends Send shows `blocked` "Preparing engine…" and the drawer status chip says so (no separate Boot View); undecryptable keys: Settings home notice |
| 49 | Dialog fade animations / spinning Halgakos icons | M3 motion tokens; Halgakos in the native splash, the drawer header, About and the empty states | Adapted |
| 50 | Memory usage logger | Data › Logs & diagnostics › Memory stats (off by default) | Adapted |
| 51 | API key encryption at rest | Automatic (no control); Data › Backup offers passphrase-encrypted export | Automatic (SecureStorage key) |

## 13. existing-android (48): the old Kivy/PySide app is deleted; each row shows its replacement

| # | Feature | Mobile surface | Notes |
|---|---|---|---|
| 0 | App shell: slide-in navigation drawer | ChatDrawer | Replaced |
| 1 | Dark Material theme + CJK font registration | M3 theme (System / Light / Dark / AMOLED); system CJK fallback | Replaced |
| 2 | Config persistence (config_android.json) | Shared config.json | Replaced |
| 3 | Storage permission request | No storage permissions (SAF / FilePicker) | Replaced |
| 4 | Notifications (progress + completion) | Native extension channels (spec §1.9): progress kept current while hidden, re-posted when swiped (Android 14+), glossary Accept / Review, real permission state on Settings › Notifications & background | Replaced |
| 5 | Open-with / Share intent import | IntentRouter | Replaced |
| 6 | Native file picker (SAF) with copy-to-Library | FileBridge | Replaced |
| 7 | SD card / external volume detection | Data › Storage | Adapted: SAF is used for picking only; the output root is app storage / iOS Documents (+ Android mirror to Downloads/Glossarion) |
| 8 | Library: scan and list books | Library | Replaced |
| 9 | Library: EPUB cover thumbnails | BookCard covers | Replaced |
| 10 | Library: import file FAB and add scan folder | Library FAB + Scan for raw | Replaced |
| 11 | Reader: EPUB/TXT loading | Reader; TXT: `ui/reader/text_book.py` (text mode + TXT workspace), device-fix batch 2026-10-08 | Replaced |
| 12 | Reader: chapter navigation + TOC | Reader bottom bar + Chapters drawer | Replaced |
| 13 | Reader: bookmarks | Reader ⋯ › Bookmarks | Carried forward |
| 14 | Reader: reading progress save/restore | `reader_positions` in `mobile_state.json` (Prefs) + Resume snackbar | Carried forward |
| 15 | Reader: typography/appearance settings panel | Aa sheet | Replaced |
| 16 | Reader: text sanitisation / paragraph splitting | `reader_core` (native fallback renderer) | Replaced |
| 17 | Reader: KO/EN view toggle | Original · Translated · Bilingual | Replaced |
| 18 | Reader: in-reader single-chapter streaming translation | Reader 🌐 LivePanel | Replaced |
| 19 | Reader: per-chapter translation cache | Workspace overlay (translation_progress) | Replaced |
| 20 | Reader: search (broken/unwired) | Reader search sheet | Replaced |
| 21 | Translate: input file selection | Composer attach | Replaced |
| 22 | Translate: model selection | ModelSheet | Replaced |
| 23 | Translate: API key | ModelSheet KeyField / Keys | Replaced |
| 24 | Translate: API-key-required check | Send `blocked` + ErrorCard | Replaced |
| 25 | Translate: AuthGPT/AuthGem OAuth login/logout | Accounts | Replaced |
| 26 | Translate: temperature | Settings › Translation defaults | Replaced |
| 27 | Translate: prompt profile + system prompt editor | Profiles / ModelSheet | Replaced |
| 28 | Translate: max output tokens | Settings › Translation defaults | Replaced |
| 29 | Translate: batch translation toggle + size | Settings / Plan card | Replaced |
| 30 | Translate: glossary mode | Settings › Glossary / Plan card | Replaced |
| 31 | Translate: output language | ModelSheet › Language | Replaced |
| 32 | Translate: 'Thinking' toggle (reader only) | Chat settings "Disable all thinking"; Settings › Thinking | Replaced |
| 33 | Translate: reader glossary CSV | Glossaries / manual glossary | Replaced |
| 34 | Translate: run/stop full translation | Plan card / state machine | Replaced |
| 35 | Multi-Key manager | Keys | Replaced |
| 36 | Extract Glossary: run/stop | ＋ sheet › Extract glossary | Replaced |
| 37 | Extract Glossary: generic settings editor | Glossaries settings tabs | Replaced |
| 38 | Progress manager | Book › Chapters / Glossary | Replaced |
| 39 | Other Settings: custom API endpoints card | Settings › Endpoints | Replaced |
| 40 | Other Settings: generic key/value editor | Schema-driven Settings | Replaced |
| 41 | Halgakos spinning FAB | Halgakos avatar / boot view | Replaced |
| 42 | Android stub modules | Real backend bundle (collector) | Removed |
| 43 | Experimental PySide6-on-Android launcher | Replaced by the whole Flet app | **Excluded / deleted**: superseded by the Flet app |
| 44 | CI: Build Android APK (Kivy/Buildozer) | New Flet CI (build-ci workstream) | Replaced (not a UI surface) |
| 45 | CI: Build Android PySide APK | New `build-mobile.yml` (Z_Glossarion-Android-APK / AAB artifacts) | Deleted |
| 46 | CI: emulator smoke test script | New host and device smoke tests | Replaced |
| 47 | CI: build-all.yml / other platform workflows (wiring reference) | Release concurrency rules (build-ci) | n/a for UI |

## 14. deps-audit (56)

| # | Feature | Mobile surface | Notes |
|---|---|---|---|
| 0 | EPUB novel translation (core pipeline) | Chat Plan / Job; Library Translate |  |
| 1 | TXT/MD/CSV/JSON translation | Chat |  |
| 2 | PDF input extraction (fast_semantic / fast_layout / xhtml / absolute / image render modes, TOC grouping, images, CSS) | Settings › PDF › Input (render mode) | PyMuPDF |
| 3 | PDF output generation | Settings › PDF › Output; Compile PDF | PyMuPDF |
| 4 | Subtitle translation (SRT/ASS/LRC, bundles) | Chat attach |  |
| 5 | SDLXLIFF translation | Chat attach; SDLXLIFF reviewer |  |
| 6 | In-EPUB image translation (vision) and Vision OCR source prepass | Settings › Image & vision; Mode options › Vision |  |
| 7 | Watermark detection/removal | Settings › Image & vision › Watermark | numpy / cv2 |
| 8 | Automatic glossary generation (during translation) | Plan card; approval card |  |
| 9 | Glossary extraction (Extract Glossary button) | ＋ sheet › Extract glossary |  |
| 10 | Glossary duplicate detection algorithms | Glossary › Balanced/Full | rapidfuzz |
| 11 | Unified glossary (merge/dedupe across books) | Glossaries › Unified |  |
| 12 | Glossary compression for prompts | Glossary › General |  |
| 13 | Glossary refinement | Glossary › Refinement; Glossary tab ✨ |  |
| 14 | Glossary matching/transliteration/usage footnotes/translation gate/paths | Glossary tab footnotes; automatic |  |
| 15 | Gender tracking | Glossary › General; EntrySheet Resolve gender |  |
| 16 | QA scanner (quick-scan / aggressive / ai-hunter / custom) | Tools › QA |  |
| 17 | QA silent truncation check | Settings › QA › Additional checks | The heuristic ships; embeddings (sentence-transformers) are native-impossible and shown disabled with a ReasonChip |
| 18 | QA AI truncation check | Settings › QA; Keys (AI truncation pool) |  |
| 19 | AI Hunter duplicate/retranslation detection | Settings › Response handling › Duplicates |  |
| 20 | EPUB compilation | Compile EPUB |  |
| 21 | EPUB image compression | Settings › EPUB output / PDF › Quality | Pillow |
| 22 | Metadata and chapter-header batch translation | Settings › Metadata, TOC & headers; Tools › Headers |  |
| 23 | Async batch API processing (50% discount: OpenAI/Anthropic/Gemini/Mistral/Groq) | Tools › Async batch |  |
| 24 | Multi API key pools and rotation (main, glossary, refinement, QA/vision, metadata, truncation, rolling summary, inpainter, TTS) | Keys |  |
| 25 | API key encryption at rest | Automatic (no control); Data › Backup offers passphrase-encrypted export | Automatic (SecureStorage key) |
| 26 | Model catalog polling and model dropdown | ModelSheet / Models |  |
| 27 | Token counting / chunk sizing | Composer token hint; automatic | tiktoken seeded offline |
| 28 | Translation history / rolling context | Settings › Context & memory |  |
| 29 | HTTP request/response logging | Data › Logs & diagnostics |  |
| 30 | Payload saving | Data › Logs & diagnostics |  |
| 31 | Provider: OpenAI and OpenAI-compatible endpoints (custom base URL, OpenRouter, DeepSeek, Groq, xAI, Fireworks, SambaNova, NVIDIA, chutes, ElectronHub, NanoGPT, etc.) | ModelSheet; Settings › Endpoints |  |
| 32 | Provider: Gemini native (google-genai) | ModelSheet |  |
| 33 | Provider: Gemini raw gRPC transport | Settings › Endpoints › Gemini gRPC transport | Tier-B pin (U9): grpcio 1.81.0 + google-ai-generativelanguage 0.12.1 ship; the bootstrap sets `GRPC_DNS_RESOLVER=native` |
| 34 | Provider: Anthropic / Mistral / Cohere | ModelSheet |  |
| 35 | Provider: DeepL | ModelSheet |  |
| 36 | Provider: Google Cloud Translate | ModelSheet (Google Cloud Translate route) | Dependency rule (U9): google-cloud-translate 3.28.0 ships (its translate_v2 client is REST) |
| 37 | Provider: Google Free Translate | ModelSheet |  |
| 38 | Provider: Vertex AI / Model Garden (incl. Claude on Vertex) | ModelSheet route row; Settings › Endpoints › Vertex | Dependency rule (U9): google-cloud-aiplatform needs protobuf<7 and is not shipped, so Vertex runs through REST + google-auth (Gemini via google-genai `vertexai=True`, Claude via `AnthropicVertex`; the desktop code) |
| 39 | Provider: Poe | ModelSheet | Deprecated |
| 40 | Google Cloud Text-to-Speech | Mode options › Audio (voice); Settings › Endpoints › TTS | Dependency rule (U9): google-cloud-texttospeech 2.38.0 ships (with the grpcio 1.81 pins) |
| 41 | NanoGPT image/video generation and media probing | Image / Video modes |  |
| 42 | OAuth subscription routes: AuthGPT (ChatGPT), AuthCD (Claude), AuthGem (Gemini/Code Assist), AuthGrok (xAI) | Accounts |  |
| 43 | Browser-backed keyless routes: AuthND (NVIDIA Build) and Search/Gemini Free | Accounts › Experimental |  |
| 44 | Opera Aria route (search/opera) | ModelSheet search/opera rows + Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded** |
| 45 | OcAgy / OpenCode Zen routes | ModelSheet ocagy*/ and ocz/ rows + Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded** |
| 46 | Antigravity Cloud Code proxy | ModelSheet antigravity/ rows + Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded** |
| 47 | AuthZA (Z.AI GLM) local proxy | ModelSheet authza*/ rows + Accounts › Unavailable on mobile (disabled row + ReasonChip) | **Excluded** |
| 48 | Local LLM servers (Ollama / LM Studio) and ollamapull | Settings › Endpoints › Local LLM host | ollamapull **excluded** |
| 49 | Async/subprocess chapter extraction mode | Settings › Processing & extraction: async chapter extraction row shown locked off with a ReasonChip "No subprocesses on mobile" | **Excluded**: no subprocess on mobile; gated at consumers, threads used |
| 50 | Enhanced text extraction (html2text) | Settings › Processing › Extraction |  |
| 51 | Markdown/TXT sidecar outputs and enhanced-text-to-HTML conversion | Settings › EPUB output |  |
| 52 | Remote image localization during extraction | Settings › EPUB output |  |
| 53 | Process priority / CPU affinity / child-process cleanup | Data › Logs & diagnostics: "Process priority / CPU affinity" rows (disabled row + ReasonChip) | **Excluded**: desktop process management |
| 54 | DPI/font scaling | Appearance › Text scale | **Excluded**: DPI scaling (OS-managed) |
| 55 | Stop / graceful stop / wait-for-chunks | State machine; Settings › Stop logic |  |

## 15. critic-missing (18): features the critic found missing from the inventory

| # | Feature | Mobile surface | Notes |
|---|---|---|---|
| 0 | Direct Text: chat sessions, sidebar and history persistence | ChatDrawer (Pinned / Series / Recents; Series is U9, optional: `mobile_series.json` + `series_id` in the sidecar), header rename, `direct_text_chats.json` v2 via ChatStore, per-chat drafts, auto-title | Sidecar for extras |
| 1 | Direct Text: file attachments that run the full pipeline | Composer attach → Plan card → Job card; attached-text prompt role; Vision / Image auto-switch |  |
| 2 | Direct Text: attachment workspace manager with Migrate | Automatic: a finished chat book moves into the Library by itself (`ChatFeature.auto_migrate`, the desktop Migrate); `/chat/<cid>/attachments` lists workspaces still waiting (⋯ Merge into Library… on a name clash); job card "Open in Library" | Automatic |
| 3 | Direct Text: per-chat settings panel and glossary override | Chat settings sheet (This chat / All chats) + ManualGlossarySheet |  |
| 4 | Direct Text: output editing, media playback, bookmarks and zoom | Output editor route; Image / Video / Audio cards (AudioCard volume 75% + Open externally); Jump-to sheet (▲/▼ steppers, Input / Output counters); Chat ⋯ › Text size; token hint |  |
| 5 | SDLXLIFF reviewer: machine-translation provider choice, credentials, Inject MT and score threshold | Tools › SDLXLIFF › MT sheet + Flag threshold | Argos shown disabled with a ReasonChip (ctranslate2 is native-impossible) |
| 6 | google-translate-free fallback chain (DeepL/Bing/Yandex/Argos offline) | Automatic (ModelSheet google-translate-free); SDLXLIFF MT sheet providers | The Argos step is skipped on mobile (native-impossible) |
| 7 | Ignore server Retry-After header (HTTP tuning) | Settings › Response handling › HTTP |  |
| 8 | Glossary: Single Pass Header Prompt | Glossary › Balanced/Full (PromptTile) |  |
| 9 | Glossary: 'Add minimal glossary pass' before Balanced/Full/Single Pass | Glossary › Balanced/Full switch; Glossary tab Minimal Pass row |  |
| 10 | Unified Glossary: 'Exclude gendered active entries' and 'Rebuild Now' | `/glossary/unified` |  |
| 11 | Glossary duplicate-detection name-matching sub-options | Glossary › Balanced/Full › Duplicate detection |  |
| 12 | Glossary editor Find/Replace that edits output HTML files directly | Glossary editor Find / Replace → "No glossary match — apply to output HTML files?" | Undoable |
| 13 | Metadata translation mode (together / title + others / per-field parallel) | Settings › Metadata, TOC & headers › Custom metadata › Mode |  |
| 14 | Manga: 'Disable all thinking' for custom-api OCR | Manga › Settings › OCR provider (custom-api) |  |
| 15 | Manga: Custom Image Edit Prompt and batch image-edit requests | Manga › Settings › Inpainting (custom image edit) |  |
| 16 | Manga preview: manual 'Create CBZ' and 'Download Images' buttons | Manga › Files output actions |  |
| 17 | Persistent logs and crash/freeze diagnostics | Data › Logs & diagnostics (crash.log, freeze log, share bundle, previous-crash banner) |  |

## 16. critic-dialogs (10): dialogs the critic found uncovered

| # | Dialog | Mobile surface | Notes |
|---|---|---|---|
| a | Direct Text sub-windows: Attachments manager, "Attachment folder already exists", "Provide Manual Glossary" | AttachmentsManager View; collision dialog (Merge and replace / Cancel); ManualGlossarySheet |  |
| b | `_DirectTextMediaPlayerFrame` | VideoCard (flet-video default controls, `aspect_ratio` 16/9) / AudioCard (flet-audio; volume 75%, Open externally) |  |
| c | `_DirectChatTitleButton` (rename) | Drawer row long-press › Rename; header title long-press |  |
| d | POE Authentication p-b cookie helper | ModelSheet Poe route row → PoeSetupSheet (paste p-b cookie, deprecation guide link, Test) | Route is deprecated |
| e | Install OpenCode Antigravity | Accounts › Unavailable on mobile › Antigravity / OCAGY rows (disabled row + ReasonChip) | **Excluded**: ocagy / antigravity |
| f | SDLXLIFF MT provider menu, credential prompts, MT tooltip | SDLXLIFF MT sheet; MT preview expands inline |  |
| g | Manga: Custom Image Edit Prompt, Replicate API Key, Model Information, '?' help, Qwen2-VL size | Manga Settings PromptTile / SecretTile / ⓘ info sheets | Qwen2-VL **excluded** (torch) |
| h | ImageRenderer per-box "📝 OCR Recognition Result" / "🌍 Translation Result" | Manga Editor BoxSheet tabs (Save / Save & Update Overlay) |  |
| i | GlossaryManager "No glossary match" prompt | Find / Replace dialog |  |
| j | autharena Page / View / Controls | Accounts › Unavailable on mobile › Arena row (disabled row + ReasonChip) | **Excluded**: autharena |

## 17. sub-settings: settings the critic found assumed under generic items

| Setting | Mobile surface | Notes |
|---|---|---|
| Disable Glossary History | Glossary › Balanced/Full & Minimal |  |
| Skip title / header-only chapters (GLOSSARY_SKIP_TITLE_HEADER_ONLY) | Glossary › Balanced/Full & Minimal |  |
| Glossary Request Merging | Glossary › Balanced/Full |  |
| Dynamic request splitting | Glossary › Balanced/Full |  |
| Dynamic Limit Expansion / Include All Characters | Glossary › Balanced/Full & Minimal |  |
| Remove honorifics | Glossary › Balanced/Full & Minimal |  |
| Reopen completed types | Glossary › Refinement |  |
| Do not refine until 100% | Glossary › Refinement |  |
| Skip deduplication after refinement | Glossary › Refinement |  |
| Manga "Process This Image" / Skip | Manga › Files per-file switch; Editor ⋯ |  |
| Manga / Manhwa / Large Text font presets | Manga › Settings › Rendering preset chips |  |
| PM image view: Mark as Skipped / Delete Selected | Book › Chapters image-grid variant |  |
