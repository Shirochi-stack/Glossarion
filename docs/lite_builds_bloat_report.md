# TurboLite, Lite and Standard builds: size diagnosis and slimming (v9.15.14)

Section 0 has the result: the low-risk items from section 5 are applied to the three specs and
measured on rebuilt exes. Sections 1-7 are the original diagnosis, kept as the reasoning behind
each item. This work follows on from the NoCuda slimming (`docs/nocuda_bloat_report.md`, PR 162).

## 0. Applied and measured

### 0.1 Sizes

"Before" is the unchanged origin/main specs (33fb2276), built in the same worktree and Python
3.12.9 / PyInstaller 6.19 environment as "after", so the only difference is the spec edits. The
before exes are about 40 KB bigger than the v9.15.14 release assets because of code merged after
the release.

| Build | Before (bytes) | After (bytes) | Saved | PYZ modules | PKG entries | Extracted per launch (u MiB) |
|---|---:|---:|---:|---|---|---|
| TurboLite | 102,355,055 | 96,040,105 | **6.3 MB (6.2%)** | 6,506 -> 6,002 | 1,944 -> 1,285 | 176.5 -> 170.3 |
| Lite | 151,138,994 | 122,996,072 | **28.1 MB (18.6%)**, and PDF export now works | 8,121 -> 7,834 | 2,179 -> 1,451 | 254.6 -> 194.0 |
| Standard | 383,600,743 | 271,976,625 | **111.6 MB (29.1%)** | 9,404 -> 8,367 | 5,329 -> 1,662 | 809.5 -> 513.7 |

The estimates in section 5 were about 95 / 108 / 253 MB for the same set of items. Lite and
Standard came out lower than that: the measured savings are 28.1 and 111.6 MB, against estimates of
about 21 (net) and 94-104 MB.

### 0.2 What changed

All the edits are filtering: `excludes`, the MSYS2 `binaries` list, a hidden-import filter, the
`collect_all` list and a post-Analysis filter on `a.binaries` / `a.datas`. `app_files`,
`app_modules`, the `EXE()` block and the console settings are untouched.

The NoCuda helpers (`_msys2_weasyprint_dlls`, `_slim_unused`, `_pyside6_unreachable_qt_dlls`) now
live in `src/spec_slim.py`, which every spec imports from `SPEC_DIR` the same way it imports
`windows_version_info`. `translator_NoCuda.spec` imports them from there instead of keeping its own
copy. The function bodies are unchanged (an AST comparison against the old spec shows them
identical), and a NoCuda rebuild was checked against PR 162's slim exe (0.4). `slim_analysis(a,
label, extra=...)` runs the NoCuda rules plus an optional per-spec predicate, then the Qt6 DLL PE
closure. `pyside6_unused_bindings([...])` builds that predicate from PySide6-relative names.

| Item | TurboLite | Lite | Standard | Measured saving (c MiB) |
|---|---|---|---|---|
| L0: allow `fontTools` back (WeasyPrint needs it) | n/a | yes | already shipped | Lite +1.3 |
| S2/L1: MSYS2 = WeasyPrint PE closure (29 DLLs instead of 101) | n/a | yes | yes | 23.3 each |
| S1: WebEngine DevTools + `.debug` resources | n/a | n/a | yes | 27.7 |
| S3 + S5: `qml/` tree, then the 101 Qt6 DLLs nothing left links | already filtered | already filtered | yes | 8.8 + about 22 |
| S4 + S6: Chromium locales except en-US, Qt `.qm` translations | already filtered | already filtered | yes | 13.3 |
| S9: QtOpenGL/QtQuick/QtQml/QtPositioning/QtQuickWidgets `.pyd`, pyside6qml, `plugins/qmltooling`, `plugins/position`, `plugins/platforminputcontexts` | n/a | n/a | yes | about 3 (in the PySide6 figure) |
| L5/T3: QtMultimedia(Widgets)/QtWebSockets `.pyd`, `plugins/multimedia`, swscale, the qpdf and virtual-keyboard plugins (their Qt6 DLLs were already filtered out) | yes | yes | n/a (the media player is shipped) | 1.0 |
| S10: `pygments`, `rich`, `redis`, `ttkbootstrap` (and ttkbootstrap out of `collect_all`) | already | already | yes | 3.0 |
| S13: `onnxruntime_extensions` | n/a | already | yes | 0.8 |
| S11/L3/T1: stray manga modules + `doctr` | yes | yes | yes | about 1.5 |
| S12/L4/T2: `mcp`, `werkzeug` | yes | yes | yes | about 1.0 |
| L6/T4: `xhtml2pdf`, `svglib`, `asn1crypto`, `arabic_reshaper` (reportlab is already excluded there) | yes | yes | not applied (S8) | 0.5 |
| S14/L7/T5: hidden imports `pytz`, `sqlite3`, `_sqlite3`, `lru` (with sqlite3.dll, pytz/zoneinfo) | yes | yes | yes | 1.3 (Lite/Standard also drop MSYS2's libsqlite3) |

The diagnosis put `setuptools`/`pkg_resources` in S14. They stay, because cffi pulls them in
regardless of the hidden import, and `pyi_rth_pkgres` needs `pkg_resources`.

The set differences between the before and after TOCs contain only what the items above name and
the packages reachable only through them. Every removed top-level package was checked against
the before build's xref graph:

- `google.cloud.vision*` (34 modules) and `azure.ai.vision.imageanalysis` (18 modules) were reached
  only from `manga_translator`, `manga_editor_core` and `ocr_manager`. So the diagnosis was wrong
  to mark vision_v1 as "used" in these tiers: no shipped non-manga module imports it.
- `pywintypes`/`win32` and `anyio.streams.*` came from `mcp.os.win32`.
- `markupsafe` and `code` came from `werkzeug`.
- `cProfile`/`pstats` and `bidi` came from `xhtml2pdf`.
- `pyclipper`, `validators` and `rapidocr_onnxruntime` came from `doctr` / `ocr_manager`.
- `markdown_it`, `mdurl` and `pty` came from `rich`.
- `cryptography.x509.ocsp` came from `redis`.

numpy, grpc, google.genai, lxml, PIL, the API SDKs and (Standard) QtWebEngine, QtMultimedia +
FFmpeg, Qt6Pdf, PyMuPDF and pdfminer are unchanged.

### 0.3 Smoke tests (after exes, isolated profile)

`GLOSSARION_APP_DIR`, `USERPROFILE`, `HOME`, `APPDATA` and `LOCALAPPDATA` pointed into a scratch
folder. Every test process was killed afterwards.

| Test | TurboLite | Lite | Standard |
|---|---|---|---|
| GUI starts and stays up for 3+ minutes (UI Automation) | pass (9 s to main window) | pass | pass |
| Dialogs open and close: Other Settings, Glossary Settings, QA Scanner Settings, Multi API Key Manager, Async Translator (its "No File Selected" prompt), Direct Text Translation, Glossarion Library, Assistant Prompt | all pass | all pass | all pass |
| `--run-chapter-extraction` on a 3-chapter Korean EPUB with an image | pass | pass | pass |
| `--run-pdf-worker` (WeasyPrint, Korean + English + images) | "WeasyPrint not installed", as designed | **pass**: 3 pages, 3 images, Korean and English extract back (the before exe fails with "WeasyPrint not installed") | pass |
| `--run-pdf-extraction` fast_semantic + pdfplumber | n/a | n/a | pass |
| `--autharena-qt-browser` offscreen: DevTools endpoint, CDP attach/navigate/evaluate, `navigator.language` = ko with only en-US.pak | n/a | n/a | pass |

Not exercised interactively:

- The EPUB reader view: the CDP test covers QtWebEngine.
- The media player: its Qt6Multimedia, FFmpeg and plugin files are all still shipped, and the Qt
  closure keeps whatever the plugins link.
- A real Vertex request.

tests/test_mobile_runtime.py (spec tiers) passes. Of the other tests that read the specs, only 4
in test_sdlxliff_support.py fail, and they fail the same way on a clean tree.

### 0.4 NoCuda check

`translator_NoCuda.spec`, now built through `spec_slim`, gives 582,559,337 bytes.

- Its PKG entry set equals PR 162's slim exe, apart from the onnxscript dist-info.
- Its PYZ adds only `onnxscript`, `onnx_ir`, `onnxslim` (and `wcwidth`). PR 162's review commit
  be2e0bbd restored those after that exe was built.
- So the extraction leaves the NoCuda output unchanged.

### 0.5 Needs owner decision (not applied)

- **Vertex SDK (S7 / L2 / T0)** needs a code change in `unified_api_client._send_vertex_model_garden`
  and `splash_utils`, not only a spec edit. It would save 13.4 MiB in Lite and Standard and make
  Vertex Model Garden work in TurboLite.
- **Lite PDF policy.** Lite now makes PDFs, as help/index.html implies ("TurboLite ... also drops
  ... PDF generation"). `run_specs_parallel.bat` says "Lite ... no PDF", though. Dropping
  WeasyPrint from Lite the way TurboLite does would save about another 11 MiB (the 29 MSYS2 DLLs,
  weasyprint, fontTools, pyphen and their CSS/zopfli/brotli helpers).
- **S8: the xhtml2pdf fallback chain in Standard** (about 4.2 MiB). It runs only when WeasyPrint
  fails, and PyMuPDF is the next fallback.
- **S15 / L8 / T6: the loose `app_files` .py copies** (about 4.1-4.2 MiB each). This edits
  `app_files`, which the desktop-manga-console branch also changes, and autharena needs three of
  the files.
- **S16: opengl32sw.dll in Standard** (7.3 MiB). It is the software-GL fallback for the EPUB
  reader on VMs and RDP sessions.
- **E1: build machine.** Uninstall caldav, niquests and urllib3-future, then reinstall urllib3
  (about 2.4 MiB per build, and requests goes back to stock urllib3).

---

The original diagnosis follows. It uses this method:

- The exes are the v9.15.14 release builds in `src/dist`. Their byte sizes match the draft-release assets.
- The CArchive and PYZ tables of contents were read with PyInstaller 6.19.
- Import chains come from each build's `build/<spec>/xref-<spec>.html`.
- DLL needs come from PE import tables (`PyInstaller.depend.bindepend.get_imports`).
- "c" means compressed MiB inside the exe, which is what the download pays for. "u" means
  uncompressed MiB, which is what gets extracted to `%TEMP%\_MEIxxxx` on every launch.

## 1. Sizes

| Build | Spec | Exe (bytes) | MB | PKG entries | PYZ modules (c / u MiB) |
|---|---|---:|---:|---:|---|
| TurboLite | translator_TurboLite.spec | 102,314,742 | 102.3 | 1,945 | 6,505 (32.2 / 81.7) |
| Lite | translator_lite.spec | 151,100,124 | 151.1 | 2,180 | 8,120 (46.6 / 152.3) |
| Standard | translator.spec | 383,562,225 | 383.6 | 5,330 | 9,403 (55.7 / 175.5) |

What each tier ships, per the spec comments and the tier note:

- **TurboLite**: no EPUB reader/Chromium, no PDF output or reading (no WeasyPrint, PyMuPDF or
  pdfminer), no Vertex AI, and no QtMultimedia/QML/opengl32sw.
- **Lite**: TurboLite plus Vertex AI and WeasyPrint PDF output, with its MSYS2 DLLs. Still no
  EPUB reader/Chromium and no PyMuPDF.
- **Standard**: Lite plus the EPUB library/reader (QtWebEngine), the browser-backed auth routes
  (autharena/ over CDP), PyMuPDF/pdfplumber PDF reading, the PyMuPDF HTML engine, and the
  QtMultimedia player for generated audio/video.
- None of the three has the manga/torch/cv2/onnxruntime stack. That is excluded on purpose.

## 2. Two functional bugs found

1. **PDF export does not work in the Lite exe.** `translator_lite.spec` carries over TurboLite's
   "PDF STACK" excludes, and that list includes `fontTools`. WeasyPrint imports fontTools at module
   import time (`weasyprint/pdf/fonts.py:10`, `weasyprint/text/fonts.py`). The build's
   `warn-translator_lite.txt` says "excluded module named fontTools - imported by
   weasyprint.pdf.fonts (top-level)".
   - Running `"L_Glossarion_Lite v9.15.14.exe" --run-pdf-worker` (isolated profile) returns
     `{"success": false, "error": "WeasyPrint not installed"}`. The Standard exe passes the same
     test: a 3-page PDF with images, and the Korean and English text extract back.
   - On host Python 3.12, with every top-level name in Lite's `excludes` hidden, WeasyPrint fails
     to import. With only `fontTools` allowed back, it writes the PDF.
   - So Lite ships 29.6 MiB of MSYS2 DLLs plus 2.9 MiB of WeasyPrint Python code that can never
     run. Fix: take `fontTools` out of Lite's excludes and its `a.pure` filter (adds 1.5 MiB).
2. **Vertex AI Model Garden cannot work in TurboLite, and in Lite and Standard it loads 13 MiB it
   never uses.** On desktop, `unified_api_client.py:16318-16336` (`_send_vertex_model_garden`)
   imports `google.cloud.aiplatform` and `vertexai` and re-raises the ImportError. Nothing after
   that uses either module:
   - Vertex Gemini goes through `google.genai` with `vertexai=True` (`:16630`).
   - Vertex Claude goes through `anthropic.AnthropicVertex`.
   - The code's own mobile comment already says so. `_build_vertex_content_list`, the only
     `vertexai.generative_models` user, is never called.

   TurboLite excludes the SDK, so this route raises ImportError there. If the import were optional
   on desktop too (as it already is on mobile), the route would work in TurboLite. Lite and
   Standard could then drop the SDK: 13.4 MiB c / 68 MiB u, about 1,470 PYZ modules, mostly the
   `aiplatform_v1` + `aiplatform_v1beta1` proto stubs. That also removes the 10-20 s, ~215 MB RAM
   import that `splash_utils.py:825-836` pre-warms. This one needs a code change, not only a spec
   edit.

The old **libwinpthread-1.dll PDF bug does not apply** to these three builds. Their binaries
lists are empty (`binaries.extend([])`), so the bundle root carries MSYS2's own
`libwinpthread-1`, `libgcc_s_seh-1` and `libstdc++-6` (checked in each `PKG-00.toc`), and the
Standard PDF smoke test passes. Only `translator_Heavy.spec` still lists the src/ copies, as the
NoCuda report noted.

## 3. Top 30 per build

"Status": **used** means that tier's code path needs it. **transitive** means it is reached only
through an optional or lazy import. **unused** means no code path in that tier loads it.
**duplicate** means it is a second copy of something already in the bundle.

### 3.1 TurboLite (102.3 MB)

| # | Item | c MiB | u MiB | Status |
|---|---|---:|---:|---|
| 1 | PySide6 Qt6 core DLLs (Core, Gui, Widgets, Network, Svg, MSVC runtime) | 11.9 | 28.9 | used (whole GUI) |
| 2 | numpy (+ numpy.libs OpenBLAS 6.3) | 9.6 | 29.5 | used: `scan_html_folder` -> datasketch MinHash (QA scanner / AI Hunter) |
| 3 | app modules (PYZ) | 9.5 | 22.6 | used. About 1.6 of it is manga modules plus doctr that the tier cannot run, see T1 |
| 4 | CPython runtime (python312.dll, libcrypto-3, sqlite3.dll, _ssl, ...) | 6.8 | 17.3 | used. sqlite3.dll (0.8) is hidden-imported only, see T5 |
| 5 | grpc | 4.7 | 11.1 | used: `grpc_gemini_client` (Antigravity), google vision/translate clients |
| 6 | PySide6 Python bindings (*.pyd) | 4.5 | 13.3 | used. QtMultimedia(Widgets).pyd and QtWebSockets.pyd are dead, see T3 |
| 7 | app .py copies (loose, also in PYZ) | 4.1 | 18.9 | duplicate (see T6) |
| 8 | lxml | 3.4 | 6.9 | used (EPUB/HTML) |
| 9 | cryptography | 3.3 | 9.1 | used (OAuth, token storage) |
| 10 | google (genai, generativelanguage, vision_v1, translate_v3) | 2.9 | 10.2 | used |
| 11 | PIL | 2.7 | 6.2 | used |
| 12 | PySide6 plugins | 2.3 | 5.3 | used, apart from 0.4 of multimedia plugins (T3) |
| 13 | qh3 | 2.0 | 5.8 | transitive: urllib3-future (build-env issue, see E1) |
| 14 | pydantic_core | 1.9 | 5.2 | used (openai, anthropic, genai) |
| 15 | openai | 1.3 | 3.0 | used |
| 16 | mistralai | 1.2 | 3.1 | used |
| 17 | rapidfuzz | 1.1 | 4.2 | used (glossary matching) |
| 18 | Crypto (pycryptodome) | 1.1 | 1.9 | used (`token_encryption`) |
| 19 | langdetect | 1.0 | 2.3 | used |
| 20 | tiktoken | 0.8 | 2.2 | used |
| 21 | setuptools | 0.8 | 1.8 | unused: spec hidden import, plus pyi_rth_pkgres (T5) |
| 22 | anthropic | 0.7 | 1.7 | used |
| 23 | pydantic | 0.7 | 1.7 | used |
| 24 | urllib3 (urllib3-future code) | 0.7 | 1.5 | used by requests (E1) |
| 25 | aiohttp | 0.6 | 1.4 | used |
| 26 | translator_gui entry script | 0.5 | 1.4 | used |
| 27 | cohere | 0.5 | 1.6 | used |
| 28 | azure | 0.5 | 1.1 | used (Azure OpenAI) |
| 29 | shiboken6 | 0.4 | 1.1 | used |
| 30 | fastavro | 0.4 | 1.0 | used: hard top-level import in `cohere/utils.py:9` |

TurboLite is already close to its floor. What remains is the Qt core, numpy (needed by the QA
scanner), the app's own code and the API SDKs.

### 3.2 Lite (151.1 MB)

Rows shared with TurboLite have the same status and are not repeated (numpy 9.6, app 9.5,
CPython 7.1, grpc 4.7, pyd 4.5, loose .py 4.1, lxml, cryptography, PIL, plugins, qh3,
pydantic_core, openai, mistralai, rapidfuzz, Crypto, langdetect, tiktoken, setuptools, anthropic,
pydantic, urllib3, aiohttp, cohere, azure, shiboken6).

| # | Item | c MiB | u MiB | Status |
|---|---|---:|---:|---|
| 1 | MSYS2 mingw64/bin glob (101 DLLs + fontconfig) | 29.6 | 71.3 | **unused today** (PDF broken, section 2). After the fontTools fix, 27 DLLs (6.3 c) are used and 74 are not (L1) |
| 2 | google | 16.3 | 78.2 | 13.4 of it is the Vertex SDK, imported but unused (section 2) |
| 3 | PySide6 Qt6 core DLLs | 11.9 | 28.9 | used |
| 15 | pyphen | 1.9 | 5.9 | used by WeasyPrint (CSS `hyphens`), so dead until the fix |
| 28 | weasyprint (PYZ) | 0.6 | 1.3 | dead until the fix |

### 3.3 Standard (383.6 MB)

| # | Item | c MiB | u MiB | Status |
|---|---|---:|---:|---|
| 1 | Qt6WebEngineCore.dll | 79.8 | 195.3 | used (EPUB reader, autharena/ CDP) |
| 2 | MSYS2 mingw64/bin glob (101 DLLs) | 29.6 | 71.3 | 27 used (WeasyPrint closure, 6.3 c); **74 unused** (23.3 c: librsvg, gtk3, libpython3.14, tcl/tk, isl, ...) |
| 3 | WebEngine DevTools frontend + `*.debug.pak/.bin` | 27.7 | 88.2 | **unused** (CDP is served by the core DLL; verified on NoCuda) |
| 4 | PySide6 Qt6 core DLLs (Core/Gui/Widgets/Network/Quick/Qml/Pdf/Positioning/WebChannel...) | 22.1 | 52.6 | used (closure of the shipped pyd, plugins and QtWebEngineProcess) |
| 5 | 98 Qt6 DLLs that no shipped module links (Quick3D, 3D, Charts, Graphs, Location, QuickControls2 styles...) | 21.8 | 56.6 | **unused** (only the QML plugins in `qml/` link them) |
| 6 | pymupdf | 17.7 | 41.0 | used (`fitz`: PDF import, covers, the PyMuPDF HTML engine, `_pdf_worker` shard merge) |
| 7 | google | 16.3 | 78.2 | 13.4 Vertex SDK imported but unused (section 2) |
| 8 | qtwebengine_locales (52 .pak) | 10.8 | 43.6 | only en-US.pak needed (verified on NoCuda: `navigator.language` still follows `--lang`) |
| 9 | numpy | 9.6 | 29.5 | used |
| 10 | app modules (PYZ) | 9.5 | 22.6 | used, apart from about 1.6 of manga modules plus doctr (S11) |
| 11 | PySide6 `qml/` tree (2,514 files) | 8.8 | 25.0 | **unused** (no QML in the app) |
| 12 | PySide6 FFmpeg (avcodec/avformat/avutil/sw*) | 7.9 | 17.9 | used: `translator_gui.py:1564-1669` media player |
| 13 | PySide6 Python bindings | 7.5 | 24.2 | 2.7 unused: QtOpenGL 2.06, QtQuick, QtQml, QtPositioning, QtQuickWidgets (S9) |
| 14 | CPython runtime | 7.5 | 18.8 | used |
| 15 | opengl32sw.dll | 7.3 | 19.7 | fallback only (software GL on VMs/RDP) |
| 16 | WebEngine resources (icudtl.dat, resources.pak, v8 snapshot) | 7.0 | 13.1 | used |
| 17 | pdfminer (incl. CJK cmaps) | 5.5 | 7.0 | used (pdfplumber extraction mode) |
| 18 | grpc | 4.7 | 11.1 | used |
| 19 | app .py copies (loose) | 4.2 | 19.6 | duplicate (S15) |
| 20 | lxml | 3.4 | 6.9 | used |
| 21 | cryptography | 3.3 | 9.2 | used |
| 22 | PySide6 plugins | 2.9 | 6.6 | used, apart from about 0.6 of qmltooling debug plugins plus position and virtual-keyboard plugins (S9) |
| 23 | PIL | 2.7 | 6.2 | used |
| 24 | PySide6 translations `*.qm` | 2.6 | 9.4 | **unused** (no QTranslator) |
| 25 | qh3 | 2.0 | 5.8 | transitive (E1) |
| 26 | pyphen | 1.9 | 5.9 | used (WeasyPrint) |
| 27 | pydantic_core | 1.9 | 5.2 | used |
| 28 | pygments | 1.7 | 3.6 | **transitive**: markdown2's optional code highlighting. TurboLite/Lite already exclude it |
| 29 | fontTools | 1.5 | 3.6 | used (WeasyPrint font subsetting) |
| 30 | reportlab | 1.3 | 2.9 | **transitive**: xhtml2pdf fallback only (S8) |

## 4. Which NoCuda trims apply

| NoCuda trim | TurboLite | Lite | Standard |
|---|---|---|---|
| WebEngine DevTools + .debug resources | n/a (no WebEngine) | n/a | **27.7** |
| qtwebengine_locales en-US only | n/a | n/a | **10.7** |
| PySide6 `translations/*.qm` | already stripped | already stripped | **2.6** |
| `qml/` tree + Qt6 DLL PE closure | already stripped by name list (closure finds 0 unreachable) | same | **8.8 + 21.8** |
| MSYS2 PE closure (27 instead of 101) | n/a (no MSYS2) | **22.5** | **22.5** |
| src/ libwinpthread-1 shadowing (PDF bug) | n/a | not present | not present (PDF smoke passes) |
| opencv videoio FFmpeg | n/a (no cv2) | n/a | n/a |
| onnxruntime DirectML.dll | n/a | n/a | n/a |
| torchvision python312.dll, protoc, vfcompat/appverifUI | none present | none | none |
| polars/av/pyarrow/pythainlp/pandas/yt_dlp/... | already excluded or absent | same | same |
| gradio / emoji / _pytest | absent | absent | absent |

All the "n/a: torch/cv2/onnxruntime" rows come down to the tier excludes. TurboLite and Lite
already filter Qt by name (`'Qt6Quick' in b[0]` ...), so the NoCuda closure finds nothing more
there.

## 5. Ranked candidates per build

Savings are compressed MiB (1 MiB = 1.049 MB) and come off the exe. They are estimates from the
TOCs and the import graph (`reach.py` drops whatever is no longer reachable from any root once
the package is cut). Rebuild to measure.

### 5.1 Standard (383.6 MB)

| Rank | Change (translator.spec unless noted) | Saves | Risk | Verify with |
|---|---|---:|---|---|
| S1 | Drop WebEngine `devtools_resources*.pak` + `*.debug.pak/.bin` (NoCuda `_slim_unused`) | 27.7 | Low | EPUB reader; `smoke.py web` (CDP) |
| S2 | MSYS2: ship the WeasyPrint PE closure + cairo/pixman (NoCuda `_msys2_weasyprint_dlls`) | 22.5 | Low | `smoke.py pdf` |
| S3 | Drop the 98 Qt6 DLLs no shipped module links (NoCuda `_pyside6_unreachable_qt_dlls`) | 21.8 | Low-Med | EPUB reader, browser auth windows, media player |
| S4 | Keep only `en-US.pak` in qtwebengine_locales | 10.7 | Low | EPUB reader; `smoke.py web` |
| S5 | Drop `PySide6/qml/**` | 8.8 | Low | same as S3 |
| S6 | Drop PySide6 `translations/*.qm` | 2.6 | Low | none |
| S7 | Make the aiplatform/vertexai imports optional on desktop (unified_api_client, splash_utils), then exclude `google.cloud.aiplatform*`, `google.cloud.resourcemanager*`, `google.cloud.bigquery*`, `vertexai` | 13.4 | Low-Med (code change) | a Vertex Gemini + a Vertex Claude request |
| S8 | Exclude the xhtml2pdf fallback stack: `xhtml2pdf`, `reportlab`, `rlPyCairo`, `cairo` (pycairo), `freetype` (freetype-py), `svglib`, `pyhanko`, `pyhanko_certvalidator`, `oscrypto`, `qrcode`. It runs only after WeasyPrint fails in `pdf_extractor.create_pdf_from_html` (`:2523`), and PyMuPDF is the next fallback. Keep `pypdf` and `asn1crypto` if anything else needs them | 4.2 | Low-Med | `smoke.py pdf` + pdfx |
| S9 | Drop unused Qt bindings `QtOpenGL/QtQuick/QtQml/QtPositioning/QtQuickWidgets.pyd` (+ pyside6qml), `plugins/qmltooling`, `plugins/position`, `plugins/platforminputcontexts` (+ Qt6VirtualKeyboard*, Qt6Quick3DUtils, Qt6SerialPort, which only those plugins link). Importing the WebEngine/Multimedia/WebSockets modules does not import these bindings (checked on the host). The Qt6 DLLs stay, because Qt6WebEngineCore links Quick/Qml/Positioning | 3.8 | Low-Med | as S3 |
| S10 | Exclude `pygments`, `rich`, `redis`, `ttkbootstrap` and drop `ttkbootstrap` from `collect_all` (TurboLite/Lite already do; tkinter is excluded, so ttkbootstrap cannot import) | 3.5 | Low | start-up, Markdown rendering |
| S11 | Exclude the manga modules PyInstaller follows from lazy imports (`manga_translator`, `ocr_manager`, `bubble_detector`, `local_inpainter`, `ImageRenderer`, `manga_editor_core`, `manga_env`, `manga_runner`, `manga_settings_dialog`) and `doctr`. Every importer in a shipped module is inside try/except (`stop_control.py:385,661`, `translator_gui.py:9825`, `translation_pipeline.py:765,4310`) | 1.6 | Low | Stop button during a run |
| S12 | Exclude `mcp` (try-guarded in `google/genai/types.py:66`; takes jsonschema/rpds/pydantic_settings with it) and `werkzeug` (only `websockets.asyncio.router`) | 1.1 | Low | a Gemini request |
| S13 | Exclude `onnxruntime_extensions` (Lite already does; onnxruntime is excluded, so it cannot run) | 0.8 | Low | none |
| S14 | Drop the unused hidden imports `pytz`, `sqlite3`, `lru`, `setuptools`, `pkg_resources` (+ sqlite3.dll). No src module imports them. pyi_rth_pkgres goes away with pkg_resources | 2.5 | Low-Med | start-up of the frozen exe |
| S15 | Stop copying `app_files` as loose .py beside the PYZ. Keep `autharena_proxy.py`, `autharena_browser.py`, `token_encryption.py` (`autharena_proxy.py:747` copies them into its runtime) | 4.2 | Medium | autharena/, multi-instance run, every `--run-*` worker flag |
| S16 | Drop `opengl32sw.dll` | 7.3 | Medium | EPUB reader on a VM/RDP session |

Totals:

- **S1-S6 (the NoCuda trims, spec only): about 94 MiB, so 383.6 -> about 285 MB.**
- Adding S7-S14: about 125 MiB, so **about 253 MB**.
- Adding S15-S16: about 137 MiB, so about 240 MB.

translatoronefileoff.spec (Structured) ships the same tree, so S1-S6 apply there too (out of scope
here).

### 5.2 Lite (151.1 MB)

| Rank | Change (translator_lite.spec unless noted) | Saves | Risk | Verify with |
|---|---|---:|---|---|
| L0 | **Fix:** remove `fontTools` from Lite's excludes and from the `a.pure` PDF strip, so WeasyPrint imports and PDF export works | -1.5 (adds) | Low | `smoke.py pdf` (fails today) |
| L1 | MSYS2 WeasyPrint closure instead of the glob (as S2) | 22.5 | Low | `smoke.py pdf` |
| L2 | Vertex SDK out (as S7, after the code change) | 13.4 | Low-Med | Vertex requests |
| L3 | Stray manga modules + doctr (as S11) | 1.6 | Low | Stop button |
| L4 | `mcp`, `werkzeug` (as S12) | 1.1 | Low | Gemini request |
| L5 | Drop dead QtMultimedia(Widgets).pyd, QtWebSockets.pyd, `plugins/multimedia/*`, `swscale-8.dll`, the qpdf image plugin and the virtual-keyboard plugin. Their Qt6 DLLs are already filtered out, so they can never load. The media player falls back to its "unavailable" label (`translator_gui.py:1591`), and autharena/ refuses without WebEngine | 1.0 | Low | start-up |
| L6 | Exclude `xhtml2pdf`, `svglib`, `asn1crypto`, `arabic_reshaper` (reportlab is already excluded, so xhtml2pdf cannot run) | 0.5 | Low | none |
| L7 | Unused hidden imports (as S14) | 2.5 | Low-Med | start-up |
| L8 | Loose app .py copies (as S15; Lite has no autharena) | 4.1 | Medium | worker flags, multi-instance |

Totals:

- **L0-L7: about 41 MiB net, so 151.1 -> about 108 MB, with working PDF export.**
- Adding L8: about 104 MB.
- Owner alternative: if Lite is not meant to make PDFs (its "PDF STACK" comment says "Lite has no
  PDF output or viewing", while the MSYS2 block says "WeasyPrint + PDF stack kept in Lite"), drop
  WeasyPrint and MSYS2 the way TurboLite does: about 32.5 MiB instead of L0+L1's 21.

### 5.3 TurboLite (102.3 MB)

| Rank | Change (translator_TurboLite.spec unless noted) | Saves | Risk | Verify with |
|---|---|---:|---|---|
| T0 | The S7 code change alone (no spec change) makes Vertex Model Garden work in TurboLite | 0 | Low-Med | a Vertex request in TurboLite |
| T1 | Stray manga modules + doctr (as S11) | 1.6 | Low | Stop button |
| T2 | `mcp`, `werkzeug` (as S12) | 1.1 | Low | Gemini request |
| T3 | Dead QtMultimedia/QtWebSockets bindings and plugins (as L5) | 1.0 | Low | start-up |
| T4 | `xhtml2pdf`, `svglib`, `asn1crypto`, `arabic_reshaper` (as L6) | 0.5 | Low | none |
| T5 | Unused hidden imports (as S14) | 2.5 | Low-Med | start-up |
| T6 | Loose app .py copies (as L8) | 4.1 | Medium | worker flags, multi-instance |

Totals:

- **T1-T5: about 6.7 MiB, so 102.3 -> about 95 MB.**
- Adding T6: about 91 MB.
- Adding E1: about 88 MB.

## 6. Build environment (affects all three)

- **E1. urllib3-future overlays urllib3.** `caldav` (not used by Glossarion) needs `niquests`,
  which needs `urllib3-future`. urllib3-future installs into the same `urllib3/` folder, so the
  exe's `requests` runs on urllib3-future code (`urllib3/contrib/hface`, HTTP/3 via `qh3`, HTTP/2
  via `jh2`). Uninstalling caldav, niquests and urllib3-future, then reinstalling `urllib3`, would
  remove qh3 + jh2 (2.4 MiB per build) and put `requests` back on stock urllib3. Check first that
  nothing else on the build host needs niquests.
- The onnxruntime/OpenCV overlays from the NoCuda report do not matter here, because these tiers
  exclude both.

## 7. Kept on purpose

- Standard:
  - Qt6WebEngineCore + icudtl + resources (87 MiB) for the EPUB reader and the browser routes.
  - PyMuPDF (17.7) for PDF import, covers and the PyMuPDF HTML engine.
  - pdfminer (5.5) for pdfplumber extraction, CJK cmaps included.
  - The QtMultimedia FFmpeg DLLs (7.9) for the media player.
- All three:
  - numpy (9.6) for datasketch / the QA scanner.
  - grpc (4.7) for the Antigravity gRPC client and the google vision/translate clients.
  - fastavro, which cohere imports at top level.
  - opentelemetry, which mistralai's client hooks import at top level.

## Method

The scripts are in the session scratchpad (`.../scratchpad/lite/` next to `.../scratchpad/nocuda/`),
not in the repo:

- `nocuda/dump_exe.py <exe> <json>` dumps each exe's TOC.
- `breakdown.py <json> <PKG-00.toc> N [unreachable.txt]` makes the top-N tables.
- `tri.py` compares one package across the three builds.
- `reach.py <xref.html> <json> prefix...` estimates the PYZ saving of an exclude by graph
  reachability, built on `nocuda/xref.py`.
- `nocuda_rules.py <json> <PKG-00.toc>` applies translator_NoCuda.spec's own `_slim_unused`, Qt
  closure and MSYS2 closure functions (read from the spec with `ast`) to another build.
- `qtclosure.py`, `msysclosure.py`, `dups.py` and `pyzq.py` are the NoCuda scripts with the build
  json taken from `BUILD_JSON`.
- `weasy_lite_check.py <spec> <out.pdf> [allow...]` renders a PDF with a spec's excludes hidden.
- `nocuda/smoke.py <exe> <workdir> pdf pdfx web` runs the worker-flag smoke tests with an isolated profile.

The slimming pass (section 0) adds these, in `.../scratchpad/slim3/`:

- `build.sh <tag> <spec>` builds into `src/dist/<tag>` with its own `PYINSTALLER_CONFIG_DIR`.
- `setdiff.py` / `gone.py` compare the before and after TOCs (removed or added packages and files).
- `guismoke.py <exe> <workdir> <secs> open:<button>...` is the UI Automation GUI smoke test.
- `epubsmoke.py <exe> <workdir>` runs the `--run-chapter-extraction` EPUB import.
- `nocuda_equiv.py` is the AST check of the extracted helpers.
