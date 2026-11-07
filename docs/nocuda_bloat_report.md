# NoCuda build: size diagnosis (v9.15.14)

Diagnosis only. No spec has been changed yet. All numbers come from the v9.15.14 release exe
(`src/dist/N_Glossarion_NoCuda v9.15.14.exe`, 861,211,156 bytes, the same size as the draft-release
asset `N_Glossarion_NoCuda.v9.15.14.exe`). The CArchive and PYZ tables of contents were read with
PyInstaller 6.19's `CArchiveReader` / `ZlibArchiveReader`. Nothing was extracted to disk.

- "c" = compressed size inside the exe. This is what the download pays for.
- "u" = uncompressed size, which is what is written to `%TEMP%\_MEIxxxx` on every launch.
- Import chains come from `build/translator_NoCuda/xref-translator_NoCuda.html`.
- DLL needs were worked out from PE import tables (`PyInstaller.depend.bindepend.get_imports`).

## 1. Build sizes (v9.15.14)

| Build | Spec | Size |
|---|---|---|
| TurboLite | translator_TurboLite.spec | 102.3 MB |
| Lite | translator_lite.spec | 151.1 MB |
| Full (novel) | translator.spec | 383.6 MB |
| **NoCuda (manga)** | translator_NoCuda.spec | **861.2 MB** (pkg 820 MB c / 2,064 MB u, 12,152 entries; PYZ 134 MB c / 374 MB u, 19,804 modules) |
| Structured (folder) | translatoronefileoff.spec | 876 MB on disk, 376.6 MB zip |

NoCuda is 477.6 MB bigger than the full build. The full build already carries the shared base:
PySide6 + WebEngine 206 MB, pymupdf, google, the MSYS2 DLLs and so on. So the 477 MB is the manga
stack plus everything that stack drags in.

## 2. What is inside (top 40 by compressed size)

| # | Package / item | c MB | u MB | In full build? | Status |
|---|---|---:|---:|---|---|
| 1 | PySide6 (whole tree) | 206.2 | 552.3 | yes (206.2) | mixed, see 3.3 |
| 1a | - Qt6WebEngineCore.dll | 79.8 | 195.3 | yes | used (EPUB reader, browser auth routes) |
| 1b | - WebEngine devtools .pak (normal + .debug) | 24.9 | 83.4 | yes | **unused** |
| 1c | - WebEngine .debug.pak / .debug.bin duplicates | 2.8 | 4.8 | yes | **unused** |
| 1d | - qtwebengine_locales (67 .pak) | 10.9 | 43.8 | yes | only en-US needed |
| 1e | - qml tree (2,514 files) | 8.8 | 25.0 | yes | **unused** (no QML in the app) |
| 1f | - 99 Qt6*.dll reachable only from QML plugins (Quick3D, Charts, Graphs, Location, 3D, QuickControls2 styles...) | 21.8 | 56.6 | yes | **unused** |
| 1g | - opengl32sw.dll | 7.3 | 19.7 | yes | fallback only (software GL) |
| 1h | - avcodec/avformat/avutil/sw* (QtMultimedia FFmpeg) | 7.9 | 17.9 | yes | used (translator_gui video player) |
| 2 | torch | 106.0 | 360.4 | no | used |
| 2a | - torch_cpu.dll | 72.8 | 253.5 | | used |
| 2b | - torch .py source copies (pyz+py hook mode) | 8.7 | 39.7 | | duplicate of PYZ |
| 3 | cv2 | 46.8 | 123.4 | no | used, apart from: |
| 3a | - opencv_videoio_ffmpeg4130_64.dll + ..._ffmpeg490_64.dll | 21.6 | 52.4 | | **unused** (no VideoCapture/VideoWriter anywhere in src) and duplicated (two OpenCV wheels share cv2/) |
| 4 | root MSYS2 mingw64/bin glob (122 DLLs) | 28.7 | 68.8 | yes | 74 of them (23.3 MB) are not reachable from WeasyPrint |
| 5 | polars | 37.2 | 130.1 | no | **transitive only** |
| 6 | scipy + scipy.libs | 29.6 | 84.4 | no | used (easyocr, ultralytics, doctr) |
| 7 | PyAV (av + av.libs) | 29.6 | 83.6 | no | **transitive only** |
| 8 | transformers | 27.5 | 90.3 | no | used; 9.6 MB of it is .py source copies |
| 9 | pyarrow | 24.4 | 75.5 | no | **transitive only** |
| 10 | onnxruntime | 21.9 | 55.4 | no | used, apart from: |
| 10a | - capi/DirectML.dll | 8.9 | 17.7 | | **dead**: the shipped onnxruntime.dll is the 1.24.4 CPU build (providers: Azure, CPU) |
| 11 | pythainlp | 17.9 | 59.7 | no | **transitive only** |
| 12 | pymupdf | 17.7 | 41.0 | yes | used |
| 13 | google (PYZ) | 16.3 | 78.2 | yes | used |
| 14 | root files (2 PNGs, app .py, base_library) | 14.4 | 34.9 | 5.3 | used (PNGs 8.4 MB) |
| 15 | rapidocr_onnxruntime (bundled PP-OCRv4 models) | 14.3 | 15.6 | no | used (`RapidOCR()` default models) |
| 16 | numpy + numpy.libs | 9.9 | 30.2 | yes | used |
| 17 | pandas | 8.9 | 24.5 | no | **transitive only** |
| 18 | matplotlib | 7.4 | 16.1 | no | used by doctr (eager `import matplotlib.pyplot` in doctr.io.elements) |
| 19 | PIL (incl. _avif 4.1) | 6.8 | 13.7 | 2.7 | used |
| 20 | sympy | 5.8 | 15.4 | no | used (onnxruntime.quantization.shape_inference -> symbolic_shape_infer, plus torch) |
| 21 | pdfminer | 5.5 | 7.0 | yes | used (pdfplumber) |
| 22 | yt_dlp | 5.3 | 10.6 | no | **transitive only** |
| 23 | torchvision | 4.9 | 14.1 | no | used; contains a stray `torchvision\python312.dll` (2.5) |
| 24 | grpc | 4.7 | 11.1 | yes | used |
| 25 | root onnxruntime.dll (from src/, older ORT) | 4.2 | 11.0 | no | opt-in C++ inpainter only |
| 26 | lxml | 3.4 | 6.9 | yes | used |
| 27 | cryptography | 3.3 | 9.2 | yes | used |
| 28 | sqlalchemy | 2.9 | 7.7 | no | **transitive only** |
| 29 | Pythonwin (mfc140u.dll, win32ui) | 2.9 | 6.4 | no | **transitive only** |
| 30 | pypdfium2_raw | 2.7 | 5.3 | no | used by doctr.io.pdf |
| 31 | hf_xet | 2.7 | 6.5 | no | used (HF downloads) |
| 32 | h5py | 2.6 | 6.5 | no | used by doctr.datasets (eager) |
| 33 | tokenizers | 2.5 | 7.3 | no | used |
| 34 | ultralytics | 2.5 | 6.8 | no | used |
| 35 | PySide6 translations *.qm (non-WebEngine) | 2.5 | 9.3 | yes | unused (no QTranslator) |
| 36 | qh3 / pyphen / pydantic_core / pygments | 7.5 | 20.5 | yes | shared base |
| 37 | curl_cffi | 1.7 | 3.4 | no | **transitive only** (yt_dlp) |
| 38 | onnx | 1.6 | 4.2 | no | used |
| 39 | networkx | 1.5 | 3.9 | no | transitive (torch), lazy |
| 40 | Cryptodome | 1.2 | 2.2 | no | **transitive only** (yt_dlp, py7zr); the app uses `Crypto` |

## 3. Who pulls in what

### 3.1 Features of the NoCuda build (kept)

NoCuda is the full manga tier (see the memory note on spec tiers). Unlike the Mac NoCuda specs, it
deliberately keeps CPU torch. The code paths that need each heavy package:

- **torch / torch_cpu.dll**
  - `bubble_detector.py:213,265` (YOLO via ultralytics, RT-DETR via transformers)
  - `local_inpainter.py:78-92` (JIT inpainting models)
  - `ocr_manager.py:874-1532` (manga-ocr, Qwen2-VL, EasyOCR)
  - `manga_env.py`
- **transformers**
  - `bubble_detector.py:241,2460`: RTDetrV2 / RTDetr
  - `ocr_manager.py:939-1426`: VisionEncoderDecoder, BertJapaneseTokenizer, AutoModelForVision2Seq
- **ultralytics**: `bubble_detector.py:228,257` (YOLO).
- **easyocr** (`ocr_manager.py:1739`) and **doctr** (`ocr_manager.py:2086`; shown in the UI at
  `manga_integration.py:3696`). doctr's eager imports pin **matplotlib**, **h5py**, **pypdfium2**,
  **scipy** and **shapely**.
- **onnxruntime / onnx**
  - `bubble_detector.py:298`, `local_inpainter.py:107-119`, `ocr_manager.py:1220`, `hayai_ocr.py`, `ner_names.py`
  - `onnxruntime.quantization.quantize_dynamic`: `bubble_detector.py:712,993`, `local_inpainter.py:1598`.
    Its `shape_inference` module imports `onnxruntime.tools.symbolic_shape_infer`, which imports
    **sympy** at top level.
- **rapidocr_onnxruntime**: `ocr_manager.py:2241,2331` (default bundled models).
- **accelerate**: `ocr_manager.py:36`, lazy check, needed for Qwen2-VL device maps.
- **cv2** (core): ImageRenderer, bubble_detector, local_inpainter, manga_editor_core.
- **QtWebEngine**: epub_library reader, autharena_proxy, the browser-backed auth routes.
- **QtMultimedia** + its FFmpeg DLLs: `translator_gui.py:1564-1669` (QMediaPlayer/QVideoWidget), `:30102` (QSoundEffect).
- **WeasyPrint** + MSYS2 Pango stack: `_pdf_worker`, `epub_converter`, `pdf_extractor`, `pdf_workspace_compiler`.

### 3.2 Transitive only (no src module imports them, and every importer imports them lazily or optionally)

| Package | c MB | Shortest chain from app code | Why it is safe to drop |
|---|---:|---|---|
| polars | 37.2 | bubble_detector -> ultralytics -> ultralytics.utils -> polars | ultralytics imports it only inside `to_df()/to_csv()`, export and trainer (`utils/__init__.py:182,196`). Inference never calls them. Other importers: datasets, gradio. |
| av (PyAV) + av.libs | 29.6 | bubble_detector -> transformers -> transformers.video_utils -> av | `video_utils.py:470` is a lazy import behind `requires_backends`. The other importer, imageio, uses a plugin that loads on demand. |
| pyarrow | 24.4 | translator_gui -> google.cloud.aiplatform.featurestore -> pyarrow | Lazy in aiplatform. The full and Lite builds ship Vertex AI with no pyarrow at all. Other importers: datasets, pandas, polars, fsspec. |
| pythainlp | 17.9 | transformers.models.xlm.tokenization_xlm -> pythainlp | try-guarded Thai word split (`tokenization_xlm.py:421`) |
| pandas | 8.9 | translator_gui -> onnxruntime.transformers.metrics -> pandas | It comes in through `collect_all('onnxruntime')` (spec line 113), which hidden-imports `onnxruntime.transformers.*` / `tools.*` / `quantization.matmul_nbits_quantizer`. Every other importer (datasets, google, gradio, openai, pdfplumber, pymupdf, torch, tqdm, transformers) is optional. The full build has no pandas. |
| yt_dlp + curl_cffi + Cryptodome | 8.2 | transformers.video_utils -> yt_dlp | Lazy YouTube video loading, also used by ultralytics.data.loaders. |
| sqlalchemy | 2.9 | onnxruntime.quantization.matmul_nbits_quantizer -> datasets -> sqlalchemy | `datasets` is lazy in accelerate (`data_loader.py:1205`, `state.py:490`) and in transformers. `matmul_nbits_quantizer` is not imported by `onnxruntime.quantization.__init__`. |
| datasets + multiprocess + joblib | 1.8 | same as above | same as above |
| Pythonwin / win32ui (mfc140u.dll) | 2.9 | torch._appdirs -> win32com.shell -> win32com.client.makepy -> win32ui | makepy (COM type-library code generation) is never run |
| gradio + gradio_client (+ groovy, spaces) | 1.2+ | bubble_detector -> huggingface_hub._webhooks_server -> gradio | Webhooks server only. The app's Gradio UI (`app.py`) is not in this spec. |
| emoji | 0.6 | transformers.models.bertweet -> emoji | optional |
| _pytest / py | 0.5 | anyio._backends._asyncio -> _pytest.outcomes | The spec excludes `pytest` but not `_pytest`. |
| onnxscript / onnx_ir / onnxslim | 1.2 | `collect_all('onnxruntime')`, ultralytics.engine.exporter, torch.onnx dynamo | Only used for exporting; the app loads ONNX models. |

### 3.3 Unused or duplicate files inside packages that are used

| Item | c MB | Evidence |
|---|---:|---|
| WebEngine devtools .pak (`qtwebengine_devtools_resources(.debug).pak`) | 24.9 | These hold the DevTools inspector UI (the HTML/JS frontend). `autharena_proxy.py:352` does set `QTWEBENGINE_REMOTE_DEBUGGING`, but it only talks CDP over the `ws://.../devtools/browser/...` websocket (`:297,389`), which the core DLL serves without the frontend. That is still to be confirmed with an autharena/ run. |
| WebEngine `*.debug.pak`, `v8_context_snapshot.debug.bin` | 2.8 | These belong to the debug Qt6WebEngineCore. The release DLL loads the non-.debug files. |
| qtwebengine_locales except en-US.pak | ~10.7 | Chromium falls back to en-US. These files only localise context-menu and error-page strings. |
| PySide6 `qml/` + the Qt6 DLLs that only QML plugins link | 8.8 + 21.8 | The PE closure was computed from the 16 shipped .pyd files (QtCore/Gui/Widgets/Network/WebEngine*/Multimedia*/WebSockets/Quick/Qml/QuickWidgets/OpenGL/Positioning/PrintSupport/WebChannel), the plugins and QtWebEngineProcess.exe. It needs 36 PySide6 DLLs; 99 are never reached (Quick3D*, 3D*, Charts, Graphs, DataVisualization, Location, QuickControls2*, ShaderTools, RemoteObjects, SpatialAudio, Scxml, Sensors, Test, Sql, TextToSpeech, WebEngineQuick...). The PySide6 hook ships the whole qml tree because QtWebEngineCore imports QtQuick. |
| PySide6 non-WebEngine translations (*.qm) | 2.5 | No `QTranslator` anywhere in src |
| `cv2\opencv_videoio_ffmpeg490_64.dll` and `..._ffmpeg4130_64.dll` | 21.6 | Two OpenCV wheels (opencv-python 4.9.0.80 and opencv-python-headless 4.13.0.92) are installed over the same `cv2/` directory, so both FFmpeg plugins are collected. cv2 loads the plugin only for video I/O, and no src module uses VideoCapture/VideoWriter. |
| `onnxruntime\capi\DirectML.dll` | 8.9 | Three onnxruntime distributions are overlaid in one directory: onnxruntime 1.24.4, onnxruntime-directml 1.22.0 and onnxruntime-gpu 1.22.0. The active `onnxruntime.dll` is the 1.24.4 CPU build, and `get_available_providers()` returns only Azure and CPU. DirectML.dll is left over from the directml wheel and is never loaded. The DML branch at `bubble_detector.py:2248` never triggers in this build. |
| 74 MSYS2 DLLs not reachable from WeasyPrint | 23.3 | The spec globs all of `C:\msys64\mingw64\bin\*.dll` (lines 96-104). The closure was computed from WeasyPrint's six `ffi.dlopen` roots (gobject, pango, harfbuzz, harfbuzz-subset, fontconfig, pangoft2). It needs 27 DLLs (6.2 MB). Not reachable: librsvg 4.6, libgtk-3 2.8, libcrypto-3-x64 2.0, **libpython3.14.dll** 2.0, libisl 0.9, libsqlite3 0.8, tcl86/tk86 1.5, libgdk-3, libepoxy, gmp/mpfr/mpc, ncurses/readline, girepository, and others. libcairo-2 + libpixman (0.9) are not reachable either, but are worth keeping in case cairocffi falls back to them (pycairo's .pyd links cairo statically). |
| `torchvision\python312.dll` | 2.5 | A second python312.dll that came with the torchvision install. The interpreter uses the root python312.dll. |
| root `onnxruntime.dll`, `onnxruntime_providers_shared.dll`, `onnx_inpainter.dll`, `libonnx_inpainter.dll`, `onnx_inpainter.cpp` | 4.2 | Spec lines 62-73. Only the opt-in C++ inpainting backend uses them (`local_inpainter.py:1496`, `enable_onnx_cpp_backend` / `ENABLE_ONNX_CPP_BACKEND`, default off, not in any settings UI). This is an older ORT DLL in the extraction root, next to the 1.24.4 one in `onnxruntime\capi\`. Windows DLL search can resolve the wrong one. |
| `vfcompat.dll`, `appverifUI.dll` | 0.1 | Windows Application Verifier DLLs. Nothing references them. |
| torch / transformers `.py` source copies | 18.3 | PyInstaller hooks collect these two packages as `pyz+py`: every module is both in the PYZ and as a loose .py. torch needs the .py for TorchScript scripting and the dynamo source lookups. transformers needs it for `inspect.getsource` in its docstring and auto utilities. The app only `jit.load`s and runs models, so this may be removable, but it needs testing. |
| `torch\bin\protoc.exe`, `torch\testing\_internal` (PYZ 1.5) | 2.6 | build and test tooling |
| `PySide6\opengl32sw.dll` | 7.3 | Mesa software GL. Qt loads it only when no usable GPU OpenGL is present (VMs, RDP, broken drivers). |
| ttkbootstrap | ~0.5 | In the `collect_all` list, but the spec excludes tkinter, so it can never import. |

### 3.4 Duplicates that must stay

- `numpy.libs\libscipy_openblas64_` (6.1) and `scipy.libs\libscipy_openblas` (6.3) are different
  builds (ILP64 vs LP64). Each wheel links its own, so neither can be dropped.
- The FFmpeg copies in PySide6 (avcodec-61 etc.) and av.libs are separate builds. Removing PyAV
  removes the av.libs copy.
- MKL is not present. torch 2.11.0+cpu ships only `libiomp5md.dll`.
- No CUDA, cuDNN or TensorRT DLLs are in the exe. The spec's GPU filter works: providers_cuda and
  providers_tensorrt are stripped.

## 4. Ranked candidate removals

Savings are compressed MB, so they come off the .exe size. The uncompressed savings, which come off
the start-up extraction, are about 3x.

| Rank | Change (in translator_NoCuda.spec only) | Saves (c MB) | Risk | How to verify |
|---|---|---:|---|---|
| 1 | Exclude `polars` | 37.2 | Low | YOLO + RT-DETR detection run |
| 2 | Exclude `av` (and av.libs goes with it) | 29.6 | Low | manga-ocr / RT-DETR load (transformers imports) |
| 3 | Drop WebEngine devtools + .debug resources from `a.datas` | 27.7 | Low-Med | open the EPUB reader; one autharena/ request (CDP over QTWEBENGINE_REMOTE_DEBUGGING) |
| 4 | Exclude `pyarrow` | 24.4 | Low | Vertex AI translation (the full build already ships without it) |
| 5 | Keep only the 27 WeasyPrint-reachable MSYS2 DLLs (+ cairo, pixman) instead of the whole `mingw64/bin` glob | 22.4 | Low-Med | PDF export via `_pdf_worker` (WeasyPrint) |
| 6 | Filter `opencv_videoio_ffmpeg*.dll` from `a.binaries` | 21.6 | Low | manga pipeline (cv2 imread/inpaint) |
| 7 | Exclude `pythainlp` | 17.9 | Low | none needed (Thai XLM tokenizer only) |
| 8 | Exclude `pandas`, `sqlalchemy`, `datasets`, `multiprocess`, `joblib`; replace `collect_all('onnxruntime')` with `collect_dynamic_libs` + `collect_submodules` that skips `onnxruntime.transformers.models`, `onnxruntime.tools` (keep `symbolic_shape_infer`), `onnxruntime.quantization.matmul_nbits_quantizer` | 13.4+ | Low-Med | ONNX quantize path (`quantize_dynamic`) for bubble_detector/local_inpainter, PDF import (pdfplumber) |
| 9 | Keep only `en-US.pak` in `qtwebengine_locales` | 10.7 | Low | EPUB reader |
| 10 | Filter `onnxruntime\capi\DirectML.dll` | 8.9 | Low | ONNX sessions start (CPU provider) |
| 11 | Exclude `yt_dlp`, `curl_cffi`, `Cryptodome` | 8.2 | Low | token encryption (uses `Crypto`, not `Cryptodome`) |
| 12 | Drop `PySide6/qml/**` and the 99 QML-only Qt6 DLLs | 30.6 | Medium | EPUB reader, browser auth windows (authchatgpt/, authnd/, gemini_free), video player |
| 13 | Exclude `win32ui`, `pythonwin` | 2.9 | Low | torch import |
| 14 | Filter `torchvision\python312.dll`, `torch\bin\protoc.exe`, `vfcompat.dll`, `appverifUI.dll` | 3.6 | Low | start-up |
| 15 | Drop PySide6 `translations/*.qm` (non-WebEngine) | 2.5 | Low | none |
| 16 | Exclude `gradio`, `gradio_client`, `emoji`, `_pytest`, `py`, `onnxscript`, `onnx_ir`, `onnxslim` | ~3.5 | Low | HF model download (huggingface_hub) |
| 17 | Drop `opengl32sw.dll` | 7.3 | Medium | Qt start-up on a VM/RDP session without GPU GL |
| 18 | Drop the root C++ inpainter DLL set (`onnxruntime.dll`, `onnx_inpainter*.dll`, `.cpp`) | 4.2 | Medium | only if the hidden `enable_onnx_cpp_backend` option is retired; also removes the ORT version-shadowing risk |
| 19 | Collect torch/transformers as `pyz` only (hooksconfig / module_collection_mode) | 18.3 | Med-High | RT-DETR, manga-ocr, Qwen2-VL, LaMa JIT all load and run |
| 20 | Prune `transformers.models` to the families used (rt_detr, rt_detr_v2, vision_encoder_decoder, vit, bert, bert_japanese, qwen2_vl, qwen2_5_vl, auto, plus the trocr/deit encoders) | ~20 | High | custom Qwen2-VL model IDs can name any architecture |

Totals:

- **Low and Low-Med (ranks 1-11, 13-16): about 233 MB off the exe**, roughly 861 -> 628 MB.
- **Adding ranks 12, 17 and 18: about 275 MB**, roughly 586 MB.
- **Adding ranks 19 and 20: about 313 MB**, roughly 548 MB.

Each excluded package also takes away the PYZ modules that only it pulls in. The real savings
should therefore be equal to or slightly above these estimates; measure after rebuilding.

## 5. Not removable (kept)

| Item | c MB | Reason |
|---|---:|---|
| torch_cpu.dll + torch_python.dll | 78.2 | YOLO/RT-DETR/manga-ocr/EasyOCR/Qwen2-VL/LaMa |
| Qt6WebEngineCore.dll + icudtl + resources.pak | 86.7 | EPUB reader and browser-backed routes |
| cv2.pyd | 25.2 | manga image pipeline |
| onnxruntime capi (dll + pybind) | 11.1 | ONNX detectors/inpainters/OCR |
| scipy | 29.6 | easyocr (`craft_utils`, `utils` top level), ultralytics ops/matching, doctr |
| rapidocr models | 14.2 | RapidOCR provider default models |
| matplotlib, h5py, pypdfium2 | 12.7 | doctr imports them eagerly |
| sympy (+ mpmath) | 6.5 | onnxruntime.quantization.shape_inference and torch |

## 6. Build-environment problems (affect size and correctness)

1. **Three onnxruntime wheels are overlaid** in the 3.12 user site: onnxruntime 1.24.4,
   onnxruntime-directml 1.22.0 and onnxruntime-gpu 1.22.0. The 1.22 wheels put
   providers_cuda/tensorrt and DirectML.dll into the directory, and `collect_all` picks them up. The
   GPU filter strips cuda/tensorrt, but DirectML.dll gets through. Uninstalling
   onnxruntime-directml and onnxruntime-gpu, then reinstalling onnxruntime 1.24.4, would leave a
   consistent capi folder.
2. **Two OpenCV wheels are overlaid**: opencv-python 4.9.0.80 and opencv-python-headless 4.13.0.92.
   The cv2.pyd that ends up in the exe is whichever was installed last, and both FFmpeg plugins ship.
   Keeping one wheel (headless is enough, since the app has no cv2 GUI) removes the duplicate even
   without a spec filter.
3. **MSYS2 glob**: the bundle contents depend on whatever is installed in `C:\msys64\mingw64\bin`,
   which currently includes Python 3.14, GTK3, Tcl/Tk and gcc's isl. An explicit list in the spec
   makes the build reproducible.
4. The warn file (`build/translator_NoCuda/warn-translator_NoCuda.txt`) lists 1,605 missing
   modules. All of them are optional or conditional imports, mostly numpy, torch and scipy
   platform branches. None of them is a size issue.

## 7. Shared-spec note

Ranks 3, 5, 9, 12, 15 and 17 (the PySide6/WebEngine and MSYS2 trims) apply just as much to
translator.spec (full, 383.6 MB) and the Structured build. Those builds ship the same 206 MB PySide6
tree and the same MSYS2 glob. The full build would drop roughly 100 MB with the same edits. This
report keeps the proposed edits to translator_NoCuda.spec. Keeping them to the
excludes/binaries/datas filtering lets them merge cleanly with the `console=False` /
`console_window` edits on branch desktop-manga-console.

## Method (for re-measuring after changes)

The scripts live in the session scratchpad (`.../scratchpad/nocuda/`), not in the repo:

- `dump_exe.py <exe> <out.json>` reads the CArchive TOC plus a per-module PYZ TOC, with sizes.
- `agg.py`, `cats.py` and `top.py` group sizes by package or category.
- `xref.py <xref.html> <src> pkg...` prints the shortest import chain from a project module to a package.
- `qtclosure.py` and `msysclosure.py` compute the PE import closure for the PySide6 and MSYS2 DLLs.

Run them with `python -P` (host Python 3.12, user site needed for PyInstaller).
