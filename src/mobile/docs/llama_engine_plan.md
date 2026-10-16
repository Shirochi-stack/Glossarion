# Built-in llama.cpp engine: implementation plan

Status: design, 2026-10-16. Nothing here is implemented yet. Branch for CI iteration: `llama-engine`.

Goal (owner): remove "Ollama in Termux" and replace it with a built-in llama.cpp engine like
PocketPal / ChatterUI. One tap on a model downloads its GGUF from Hugging Face and runs it inside
the app; the model then works for chat, translation, glossary and QA like any other model.

## 0. Findings that drive the decisions

Everything below was checked on 2026-10-16.

1. **pypi.flet.dev already ships llama-cpp-python for every target we build.** The index lists
   `llama_cpp_python-0.3.32-2-cp313-cp313-{android_24_arm64_v8a, android_24_x86_64,
   ios_13_0_arm64_iphoneos, ios_13_0_arm64_iphonesimulator, ios_13_0_x86_64_iphonesimulator}.whl`
   (build tag 2; build 1 also exists and lacks the loader fixes). They are built by
   flet-dev/mobile-forge `recipes/llama-cpp-python` at the commit we already pin
   (`0fadbfab0d63d11164030dfddc955ed980e32e2c`), PRs #91 and #106. The Flet 0.86 loader fixes and
   the on-device tests (Android arm64 and x86_64 emulators, iOS arm64 simulator) went in with #106.
   - The recipe's CMake args: `GGML_NATIVE=OFF`, `GGML_OPENMP=OFF`, `GGML_LLAMAFILE=OFF`,
     BLAS/Accelerate/Metal/CUDA/HIP/Vulkan/OpenCL/RPC all OFF, `LLAVA_BUILD=OFF`,
     `CMAKE_INSTALL_LIBDIR=llama_cpp/lib`. Android also gets `ANDROID_STL=c++_shared` and
     `-Wl,-z,max-page-size=16384`. iOS uses the Unix Makefiles generator, Metal is off, and
     `GGML_METAL_EMBED_LIBRARY=OFF`.
   - Its `mobile.patch` does four things. It stops the CMakeLists from forcing Metal on iOS. It
     skips the `common` library. It removes SONAME versioning. It rewrites the ctypes loader: on
     iOS it resolves the `.fwork` framework markers, on Android it falls back to bare-soname
     `CDLL("libllama.so")` from jniLibs, it accepts `sys.platform == "android"` / `"ios"`, and it
     preloads ggml-base, ggml-cpu, ggml and llama with `RTLD_GLOBAL`.
2. **These wheels pass our own static checks.** I downloaded the x86_64 Android wheel (sha256
   `8d0499ae…f965f`, 2.0 MB) and the iphoneos wheel (`4956370e…ff87`, 1.7 MB) and parsed them with
   `ci/wheels/wheels.py` `parse_elf` / `parse_macho`. Results:
   - Android: four libraries (`libggml-base.so`, `libggml-cpu.so`, `libggml.so`, `libllama.so`).
     Every LOAD segment is aligned to 0x4000 (16 KB). The NEEDED entries are the siblings plus
     `libc++_shared.so`, `libc.so`, `libm.so` and `libdl.so`. No Python C API imports (pure ctypes).
   - iOS: four dylibs with `LC_BUILD_VERSION` platform IOS and minos 13.0. They link only
     `@rpath/lib{ggml*,llama}.dylib`, `/usr/lib/libc++.1.dylib` and `/usr/lib/libSystem.B.dylib`.
   - Wheel METADATA: `Requires-Dist` lists typing-extensions, numpy, diskcache and jinja2. The
     Android wheels also list `flet-libcpp-shared>=27.2.12479018`, which ships `libc++_shared.so`
     for the jniLibs.
3. **0.3.32 runs Qwen3.5.** The architecture strings `qwen35` / `qwen35moe` appear in
   `libllama.so` and `libllama.dylib`, and HF's GGUF metadata for the unsloth Qwen3.5 repos reports
   architecture `qwen35`. (PyPI has newer releases up to 0.3.36 of 2026-10-01, sdist only.)
4. **The iOS simulator can load ctypes dylibs with Flet 1.0.3.** serious-python PR #223
   ("framework-ize ctypes .dylib libs so they load on the iOS simulator") was merged on 2026-07-01,
   and flet-cli 1.0.3's `build_base.py` needs serious_python >= 5.0.0 (released 2026-09-26).
5. **A dependency dry run passes.** I ran `check_mobile_wheels.py --wheelhouse-plan --platform all`
   on a scratch copy of pyproject with `llama-cpp-python==0.3.32` added to
   `[tool.flet.android].dependencies` and `[tool.flet.ios].dependencies`. Result: 0 errors.
   llama-cpp-python resolves on all 5 targets from flet's index. diskcache 5.6.3, jinja2 3.1.6 and
   typing-extensions 4.16.0 resolve as pure wheels and markupsafe 3.0.4 as native wheels, on all
   five targets. One gap: the tool reads dependencies from PyPI's JSON, and PyPI's metadata lacks
   `flet-libcpp-shared`, so the tool never sees that Android dependency (package A fixes this).
6. **Hugging Face accepts resumable downloads.** For
   `https://huggingface.co/<repo>/resolve/<rev>/<file>`, a HEAD returns 302 to the Xet CDN with
   `X-Repo-Commit`, `X-Linked-Size` and `X-Linked-ETag` (the sha256 of the LFS file) and
   `Accept-Ranges: bytes`. A `Range: bytes=0-1023` GET on the redirect target returns
   `206 Partial Content`. `GET /api/models/<repo>/tree/main` lists `path`, `size` and `lfs.oid`
   (the sha256). The official `Qwen/Qwen3.5-*-GGUF` repos return 401 (gated or missing); the
   unsloth, bartowski and lmstudio-community repos are public.
7. **The existing code already has the seams.** `UnifiedClient._LOCAL_OPENAI_ROUTES`
   (`ollama/`, `lmstudio/`) is a keyless OpenAI-compatible route family. `model_options` has
   `ProviderCatalogSpec` (with `base_url_env`) and `_PREFIX_PROVIDER_MAP`. The mobile
   `model_catalog` already groups `ollama` / `lmstudio` / `ollamapull` under the "Local" chip.
   `services/reader_server.py` is a hardened loopback `ThreadingHTTPServer` (random port, path
   token, Host guard). `ollama_termux.PullManager` is the progress plumbing to reuse.
8. **A prebuilt Windows CPU wheel exists for the host check:**
   `llama_cpp_python-0.3.32-py3-none-win_amd64.whl` on
   `https://abetlen.github.io/llama-cpp-python/whl/cpu/`.

## 1. Engine packaging (decision)

**Decision: use llama-cpp-python 0.3.32 from pypi.flet.dev (build tag 2) on all five targets.
Build no wheel ourselves. The own-recipe path below is the written-down contingency.**

Pins in `src/mobile/pyproject.toml`. These are device-only, the same pattern as rapidocr, so
`uv sync` / `uv.lock` on the host never builds the 70 MB sdist:

```toml
[tool.flet.android]
dependencies = [
  "psutil==7.2.2",
  "rapidocr-onnxruntime==1.2.3", "pyclipper==1.4.0", "shapely==2.1.2",
  "llama-cpp-python==0.3.32",   # built-in engine; pypi.flet.dev cp313 build 2 (forge recipe, ctypes)
]
[tool.flet.ios]
dependencies = ["rapidocr-onnxruntime==1.2.3", "pyclipper==1.4.0", "shapely==2.1.2", "llama-cpp-python==0.3.32"]
```

pip installs jinja2, markupsafe, diskcache and typing-extensions transitively, plus
flet-libcpp-shared on Android. numpy is already pinned at 2.4.6. Expected size cost: about 2 MB of
native libraries per ABI, plus jinja2, markupsafe and diskcache (about 1 MB of Python).

### Alternatives considered

| Option | Android | iOS | Our pipeline | Verdict |
|---|---|---|---|---|
| **A. llama-cpp-python wheel (index build 2)** | prebuilt, verified (16 KB, NEEDED ok) | prebuilt, simulator fixed by serious-python 5.0 | zero new build; PIP resolves from the flet index like grpcio / onnxruntime | **chosen** |
| A'. same, own forge recipe (e.g. 0.3.36) | ~20 min forge build per ABI | ~30 min per slice | new `[[wheel]]` in wheelhouse.toml, cache keys in mobile-wheels.yml, verify-native profile | contingency (below) |
| B. llama.cpp in the native extension (Kotlin JNI / Swift with `llama.xcframework`) | NDK `externalNativeBuild` in the plugin's Gradle on every `flet build` | xcframework via CocoaPods/SPM in the plugin | tokens would cross Dart→Python per token through the extension channel; Python would still need its own server for UnifiedClient | rejected: two native codebases, slow builds, bridges every token |
| C. `llama-server` binary shipped as `lib*.so` in jniLibs and exec'd | needs `extractNativeLibs=true` (legacy packaging) and breaks `GLOSSARION_NO_PROCESSES=1` | **impossible**: iOS apps cannot exec processes | — | rejected: Android-only |

Option A is the only one that works on both platforms with no new build infrastructure. Option A'
keeps the same code path if we ever need a different version.

### Contingency A' (only if a gate in section 5 fails on an index wheel)

Package A adds `src/mobile/ci/wheels/recipes/llama-cpp-python/`, an overlay of the upstream recipe
at forge `0fadbfab`. Its `meta.yaml` keeps upstream `CMAKE_ARGS` and `patches/mobile.patch` byte
for byte, with three changes:

- `version: "0.3.36"`, with sdist sha256
  `832db0699007f1be95a7e41ef12e88926b02ba836461e36a36372db2760c1a2e` (PyPI JSON; re-check against
  uv.lock if it is ever locked);
- `number: 1000`;
- `IPHONEOS_DEPLOYMENT_TARGET: '13.0'` in the iOS branch.

wheelhouse.toml gets a `[[wheel]]` for all five targets. `requires_dist` differs per platform
(`flet-libcpp-shared` is Android-only), so the manifest needs a per-platform `requires_dist` key
(`requires_dist_android` / `requires_dist_ios`): a small schema change in `check_mobile_wheels.py`
and wheels.py. mobile-wheels.yml gets the restore/save cache steps for the package. uv.lock needs
the sdist hash, so the pin then moves to `[project].dependencies` with marker
`platform_system == "Android" or platform_system == "iOS"` (flet's sitecustomize fakes
`platform.system()`, and host uv skips it). Performance follow-up (not part of the contingency):
measure `-DGGML_CPU_KLEIDIAI=ON` on arm64. It dispatches dotprod / i8mm kernels at run time without
`GGML_BACKEND_DL`. Never set `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod` globally: it would SIGILL on
Cortex-A53-only phones, and minSdk 26 still admits those.

### Verification checks (package A)

1. **`check_mobile_wheels.py`: read flet-index Requires-Dist from the wheel.** For files whose
   source is pypi.flet.dev, read Requires-Dist from the wheel's own METADATA. Use the PEP 658
   `data-dist-info-metadata` / `.metadata` file when the index serves one; otherwise do a ranged
   GET of the wheel's central directory plus the METADATA member, cached in `--cache-dir`. Then
   `flet-libcpp-shared` gets resolved and reported. Add a unit test with a fake index file.
2. **New `wheels.py verify-site-packages --platform <p> --package llama-cpp-python <build/site-packages>`.**
   It runs after every `flet build` in build-mobile.yml, next to `--verify-installed`, and reuses
   `parse_elf`, `parse_macho`, `elf_dynsym`, `macho_platforms` and `NdkStubs`. Per arch directory it
   checks:
   - `llama_cpp_python-0.3.32.dist-info` exists, its WHEEL says `Build: 2` (or higher), and
     `llama_cpp/lib` holds exactly libggml-base, libggml-cpu, libggml and libllama;
   - Android: ELF machine per ABI; every LOAD aligned >= 0x4000; NEEDED a subset of
     {the four siblings, `libc++_shared.so`, `libc.so`, `libm.so`, `libdl.so`, `liblog.so`}; every
     strong undefined symbol exported by a sibling, by `libc++_shared.so` (from the installed
     flet-libcpp-shared) or by the NDK API-24 stubs; exactly one `libc++_shared.so` across the
     whole arch directory (a second copy from another package would collide in Gradle's jniLibs
     merge);
   - iOS: per slice the right cputype and platform (IOS / IOSSIMULATOR); minos <= 13.0 (14.0
     allowed on the arm64 simulator); dylibs only `@rpath/lib{ggml-base,ggml-cpu,ggml,llama}.dylib`
     and `/usr/lib/*`; no `DYNAMIC_LOOKUP` binds.
3. **On device (both smoke jobs).** The smoke suite's new `llama_engine` check (package B) imports
   `llama_cpp`, compares `llama_cpp.__version__` with `ENGINE_VERSION` and calls
   `llama_print_system_info()`, which loads all four libraries. A host test asserts
   `ENGINE_VERSION` equals the pyproject pin. The iOS simulator smoke (policy `required`) thereby
   proves the framework-ized dylibs load on the simulator.
4. **Real inference on the emulator.** A new `llama` self-test suite (package B) driven by
   `android_smoke.sh` (package A), behind the optional-suite policy line
   `ANDROID_LLAMA_SUITE_POLICY: optional`. Promote it to `required` once it has passed once in CI
   (README "Promoting a suite").

## 2. App integration: how requests reach the engine (decision)

**Decision: an in-app OpenAI-compatible endpoint on 127.0.0.1, reached through a new shared keyless
local route `local/` in `UnifiedClient._LOCAL_OPENAI_ROUTES`.** No direct route handler.

Why the endpoint instead of a direct `_send_local` route:
- every job kind (translation, glossary, QA, chat, retranslate, review) already goes through
  `UnifiedClient`'s OpenAI-compatible path, with its streaming, retries, Stop/cancel (closing the
  connection), timeouts, think-tag handling, token accounting and error mapping. A direct route
  would re-implement all of it in the 30k-line client, a mobile-only copy of desktop logic;
- model discovery reuses `ProviderCatalogSpec` polling (`/v1/models`) exactly like ollama/lmstudio;
- the same route works on the desktop against a user's own `llama-server` (default
  `http://127.0.0.1:8080/v1`), so the shared change is meaningful there too;
- the engine thread model stays isolated: one generation at a time behind one lock, whatever the
  callers do.

### Shared changes (`src/unified_api_client.py`, `src/model_options.py`)

- `_LOCAL_OPENAI_ROUTES['local/'] = ('local', 'http://127.0.0.1:8080/v1')`.
  `_local_openai_route()` returns the base URL from `os.environ['GLOSSARION_LOCAL_LLM_URL']` when
  set, otherwise the default. Replace the direct dict read at ~line 23968
  (`self._LOCAL_OPENAI_ROUTES[f'{provider}/'][1]`) with the resolved `local_route[2]`.
- Add `'local/': 'local'` to `MODEL_PROVIDERS` and `'local/'` to `_NO_API_KEY_PREFIXES`.
  Add a class constant `_LOCAL_OPENAI_PROVIDERS = ('ollama', 'lmstudio', 'local')` and use it in
  place of each literal `('ollama', 'lmstudio')` tuple (lines ~23964, 24033, 24209, 24243, 24286,
  26581) and in the `_send_openai_provider_router` table (`'local': self._send_openai_provider_router`,
  ~19324).
- `model_options`: add
  `ProviderCatalogSpec("local", "local/", "http://127.0.0.1:8080/v1/models", public=True, base_url_env="GLOSSARION_LOCAL_LLM_URL")`
  and `("local/", "local")` to `_PREFIX_PROVIDER_MAP`. Add `"local": 8080` to
  `local_catalog_available`'s ports; when `GLOSSARION_LOCAL_LLM_URL` is set, probe that URL's port
  instead. **No static `local/` suggestions** go into the default model list: the desktop list
  stays unchanged.
- Desktop tests: `tests/test_model_options_catalog.py` and `tests/test_openai_token_compatibility.py`
  (or a new `tests/test_local_llm_route.py`) cover the route resolution, the env override, the
  keyless rule and that `local/` is stripped from the request model.

### Model id format

`local/<stem>`, where `<stem>` is the GGUF file name without `.gguf`, e.g.
`local/Qwen3.5-2B-Q4_K_M`. Ids come from files, not preset names, so renaming a preset never breaks
a saved model choice in config.json. If a second repository brings the same file name, the second
file gets the stem `<stem>-<repo owner>`. The server resolves stems case-insensitively against its
registry.

### The server (`services/local_llm/server.py`, class `LocalLLMServer`)

Modelled on `ReaderServer`: stdlib `ThreadingHTTPServer` on `("127.0.0.1", 0)`, daemon thread
`gl-llm-http`, a random 192-bit path token compared with `hmac.compare_digest`, a Host-header guard
and bare 404s for everything else. `base_url = http://127.0.0.1:<port>/<token>/v1`, published as
`GLOSSARION_LOCAL_LLM_URL`. The OpenAI SDK appends `/chat/completions`, so the token travels in the
path and the dummy Bearer key `UnifiedClient` sends is ignored. Endpoints:

- `GET /<token>/v1/models`: the downloaded models, `{"object":"list","data":[{"id":"<stem>","object":"model","owned_by":"local","meta":{"size":…,"loaded":bool}}]}`;
- `POST /<token>/v1/chat/completions`, streaming (`text/event-stream`, `data: {...}`,
  `data: [DONE]`) and non-streaming, OpenAI response schema with `usage`:
  - the model name with or without `local/` maps to a stem. Unknown or not downloaded gives
    `404 {"error":{"code":"model_not_found","message":"local/X is not downloaded. Download it in Model manager › On-device."}}`;
  - image parts in messages give `400` "This on-device model has no vision support";
  - prompt tokens over `n_ctx` give `400 {"error":{"code":"context_length_exceeded","message":"This model's maximum context length is N tokens…"}}`
    (the wording the client already maps to a context error). `max_tokens` /
    `max_completion_tokens` are clamped to `n_ctx - prompt_tokens`;
  - sampling: the request's `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`,
    `stop` and `seed` are used. Defaults when absent: `top_k=20`, `top_p=0.8`, `min_p=0.0`
    (Qwen3.5 non-thinking defaults). `tools`, `response_format` and `logprobs` are ignored;
  - one generation at a time: a FIFO `threading.Lock`. At most 8 waiters; beyond that, `503`
    "On-device model busy". While a job is running on model A, a request for another local model B
    gets `409` "The on-device engine is busy with A for a running job" instead of swapping (no
    thrash);
  - cancel: a `StoppingCriteria` checks every 8 tokens whether the client is gone
    (`select.select([sock],[],[],0)` readable plus `recv(1, MSG_PEEK) == b""`). In streaming mode
    a failed write (`BrokenPipeError` / `ConnectionResetError`) also stops generation. Stop in
    the app closes the client connection, so generation stops within a few tokens;
- `GET /<token>/health`: `{"state":"idle|loading|ready|generating","model":stem|null,"queue":n}`.

### Chat template and thinking

`LlamaEngine` builds `Llama(..., chat_handler=handler)`. The handler is
`NoThinkFormatter(Jinja2ChatFormatter).to_chat_handler()`. It uses the GGUF's
`tokenizer.chat_template` with eos/bos taken from the model's token ids, the same way llama.py does.
`__call__` passes `enable_thinking=False` through `**kwargs`: 0.3.32's
`Jinja2ChatFormatter.__call__` forwards `**kwargs` to `render`, as checked in the wheel source.
A request may override it with `chat_template_kwargs: {"enable_thinking": true}` (the
llama-server / vLLM convention). As a fallback, any leading `<think>…</think>` block is removed from
the completion before it is returned or streamed.

### Catalog and pickers (mobile)

- `services/model_catalog.py`: provider `local`, chip "Local", label "On this device". After each
  download, delete, import or start, `LocalEngineService` calls
  `catalog.refresh(provider="local")`, which polls `/v1/models` through the shared spec. Only
  downloaded models are listed.
- The ModelSheet's Local group gets one extra row: "Get on-device models…" opens
  Model manager › On-device.
- `model_block` / job preflight: a `local/<stem>` model that is not downloaded is blocked with the
  reason "Download it in Model manager › On-device". For a `local/` job model, the job adapters force
  batch translation off and the parallel extraction workers to 1 (the requests serialize at the
  server anyway, and queue waits would eat the client timeout). Package B finds the exact keys in
  `job_kinds/translate.py` / `glossary.py`, which build the env, and checks the `local/` timeout in
  `UnifiedClient` (it should be the ollama/lmstudio one).

## 3. Model download, storage and lifecycle

### Presets: editable data in Prefs

New key `local_model_presets`. On first load, the old `ollama_presets` key is dropped: its values
are Ollama tags and cannot be converted. Row: `{"name", "repo", "file", "note", "n_ctx"}`. `file`
may be empty (then the auto-pick below applies) and `n_ctx` may be 0 (the default applies).
`DEFAULT_PRESETS`, each repo and file verified on 2026-10-16 (sha256 = HF `lfs.oid`, recorded for
the tests and the self-test, re-read from HF at download time):

| name | repo | file | size | sha256 | note |
|---|---|---|---|---|---|
| Garbage | `unsloth/Qwen3.5-0.8B-GGUF` | `Qwen3.5-0.8B-Q4_K_M.gguf` | 532,517,120 | `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` | Qwen3.5 0.8B · fastest, rough translations · ~1 GB RAM |
| Light | `unsloth/Qwen3.5-2B-GGUF` | `Qwen3.5-2B-Q4_K_M.gguf` | 1,280,835,840 | `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223` | Qwen3.5 2B · quick drafts on most phones · ~2 GB RAM |
| Medium | `unsloth/Qwen3.5-4B-GGUF` | `Qwen3.5-4B-Q4_K_M.gguf` | 2,740,937,888 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` | Qwen3.5 4B · better quality · needs ~4 GB free RAM |
| Heavy | `unsloth/Qwen3.5-9B-GGUF` | `Qwen3.5-9B-Q4_K_M.gguf` | 5,680,522,464 | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` | Qwen3.5 9B · best of the four · needs ~8 GB free RAM |

(All four are Apache-2.0, architecture `qwen35`, 262,144 trained context. Each repo also has
`mmproj-*.gguf` vision projectors and an `imatrix` file, which the auto-pick must skip.)

`load_presets`, `save_presets`, `restore_defaults`, `validate_preset` and `MAX_PRESETS` are kept from
`ollama_termux` and adapted to these fields. `validate_preset` checks the name as today and runs
`repo` / `file` through the HF normaliser.

### Hugging Face references (`services/local_llm/hf.py`)

`parse_ref(text) -> HfRef(repo, file|None, revision="main", quant|None)` accepts:
- `user/repo`, `user/repo:Q4_K_M`, `hf.co/user/repo[:quant]` (the old Ollama-style form);
- `user/repo/path/file.gguf`;
- `https://huggingface.co/user/repo`, `…/tree/<rev>`, `…/blob/<rev>/<file>.gguf`,
  `…/resolve/<rev>/<file>.gguf`.

Anything else is rejected with a precise message. Repo names follow HF's rules
(`[A-Za-z0-9][A-Za-z0-9._-]{0,95}` per part); files must end in `.gguf`; `..` and absolute paths
are refused.

`list_files(ref)` calls `GET https://huggingface.co/api/models/<repo>/tree/<rev>?recursive=1`
(requests, 30 s timeout) and returns `(path, size, sha256)` for `*.gguf`, excluding `mmproj*`,
`*imatrix*` and split shards beyond `-00001-of-`. Split GGUFs (`-0000N-of-0000M`) are rejected in v1
with "Split GGUF files are not supported yet". A 401/403 gives "This repository is gated or private;
Glossarion cannot sign in to Hugging Face".

`pick_file(files, quant)`: an exact `quant` (case-insensitive) when given; otherwise Q4_K_M, then
Q4_K_S, Q4_0, IQ4_XS, Q5_K_M, Q8_0; otherwise the panel shows a file chooser sheet listing name and
size.

`resolve_url = https://huggingface.co/<repo>/resolve/<commit>/<file>`. The commit comes from the
tree call's `X-Repo-Commit` header or `GET /api/models/<repo>/revision/<rev>`, so a resumed download
fetches the same bytes.

### Downloads (`services/local_llm/downloads.py`, `DownloadManager`)

`ollama_termux.PullManager` is renamed and generalized. Kept: the worker thread through
`run_io`, the progress marshalled to the loop with `call_soon_threadsafe`, the 0.25 s throttle,
subscribe/unsubscribe and one transfer per model. `DownloadState` adds `bytes_done`, `bytes_total`,
`speed`, `eta`, `paused` and `phase` (`checking|downloading|verifying|done`).

- Storage: `GLOSSARION_MODELS_DIR = <data>/models`, set in `runtime_bootstrap` (paths contract and
  README table) and created on demand. Files go to `<models>/<file>.gguf`. The registry is
  `<models>/models.json`, `{stem: {repo, file, revision, sha256, size, downloaded_at}}`, written
  atomically: temp file, then fsync, then `os.replace`, as Prefs does.
  - iOS: the extension gets a new method `exclude_from_backup(path)`
    (`URLResourceValues.isExcludedFromBackup = true` in Swift; returns False elsewhere), called on
    `<models>`.
  - Android: the files stay in app-internal storage. Auto Backup's 25 MB quota just stops backing
    up the app, which is acceptable; this is noted in the README.
- Partial files: `<file>.gguf.part` plus `<file>.gguf.part.json` (`{url, repo, revision, size, sha256}`).
- Resume: `GET resolve_url` with `Range: bytes=<have>-` (requests streaming, 1 MiB chunks, connect /
  read timeout 30 s / 60 s). `206` appends. `200` means the server ignored the range: truncate and
  start over. `416` with `have == size` goes straight to verify. Transient errors (connection
  reset, timeout, 5xx, 429) retry 5 times with 2, 4, 8, 16 and 30 s backoff, resuming each time.
  A resumed transfer first re-hashes the existing part (phase `checking`, with progress), so the
  final sha256 always covers every byte.
- Verify: the sha256 must equal HF's `lfs.oid` / `X-Linked-ETag`, and the size must equal
  `X-Linked-Size`. On a mismatch the part is deleted and the download reports "Failed: checksum
  mismatch". On success the file is renamed into place and registered, and the catalog refreshes.
- Free space: before starting or resuming, `shutil.disk_usage(<models>).free` must be at least
  `size - have + 256 MiB`; otherwise "Not enough storage: X needed, Y free".
- Pause keeps the part and sidecar (the row shows Resume). Cancel deletes both. Delete (a finished
  model) unloads it if loaded, removes the file and the registry entry, and refreshes the catalog.
- Downloads over 1 GB ask once: "Download 5.7 GB from Hugging Face?" (size and repo shown).
- Downloads run **outside JobService**. A 5.7 GB download must not queue behind (or block) a
  two-hour translation, and resume makes a killed download cheap. A part found on app start is
  listed as paused; nothing resumes on its own.

### Engine and lifecycle (`services/local_llm/engine.py`, `service.py`)

`Engine` protocol: `load(path, n_ctx, n_threads) / unload() / chat(messages, params, stop_check) -> iterator of deltas / tokenize(text) -> int / loaded`.
- `LlamaEngine`: imports `llama_cpp` lazily, only on load.
- `FakeEngine`: deterministic echo, configurable delay; used by the host tests and by the
  on-device offline e2e check.
- `ENGINE_VERSION = "0.3.32"`.

`Llama(...)` parameters: `model_path`, `n_ctx`, `n_threads=t`, `n_threads_batch=t`, `n_batch=512`,
`n_ubatch=512`, `n_gpu_layers=0`, `use_mmap=True`, `use_mlock=False`, `verbose=False`, the chat
handler above.

Defaults:
- `n_threads`: `t = max(2, min(4, cpus - 2))`, where `cpus = len(os.sched_getaffinity(0))` when
  available, else `os.cpu_count()`. That gives 4 on 8-core and 6-core phones and 2 on 4-core ones.
- `n_ctx`: 8192 for files under 4 GB and 4096 above. The preset's `n_ctx` wins, and a global Prefs
  `local_llm_settings` `{n_ctx, n_threads}` sets defaults. Both are editable in the panel's Engine
  card.

RAM gate before load:
`need = size * 1.15 + n_ctx * 64 KiB + 400 MiB` (KV estimate per token; Qwen3.5's hybrid layers keep
the real number below it).
- Android: `avail = psutil.virtual_memory().available` (psutil is already an Android-only pin).
  If `need > total`, refuse ("This phone has X GB RAM; Heavy needs about Y"). If `need > avail`,
  show a confirm sheet: "Android may close Glossarion while this model runs. Load anyway?" Jobs do
  not confirm; they fail with that message.
- iOS: `total = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')`. Refuse when
  `need > 0.5 * total` (the jetsam limit for a foreground app without the
  increased-memory-limit entitlement, which unsigned / AltStore IPAs cannot carry).

`LocalEngineService` (one per app, created by the Models & keys feature install):
- at install it starts `LocalLLMServer`, sets `GLOSSARION_LOCAL_LLM_URL`, scans the registry and
  refreshes the catalog. The server is cheap when idle; no model is loaded at start;
- **one model loaded at a time.** A request for an unloaded stem loads it lazily, on the server
  thread under the generation lock: state `loading`, then `ready`; load time is logged. A request
  for another stem unloads the current one first, unless a job is pinned to it (the 409 above);
- explicit Load / Unload in the panel;
- **unload policy:**
  - idle for 10 minutes with no request and no running job: unload;
  - lifecycle `hide`/`pause` on iOS with no active job: unload at once (jetsam);
  - on Android: unload after 60 s in the background unless a job is active (the job's foreground
    service keeps the process alive);
  - `detach`: unload and stop the server;
  - wired from `GlossarionApp.on_lifecycle` (`app.py`) through `local_engine.on_lifecycle(name)`,
    like `cloud_sync`;
- **JobService interplay:** jobs still run one at a time (unchanged). The active job's model is read
  from `JobService` snapshots through a `pinned_model()` callable. A chat while a local job runs
  waits in the server queue; the chat composer shows "Waiting for the on-device model…" when the
  server reports `queue > 0`.

### UI: Model manager › On-device (replaces the Ollama tab)

`TABS` changes from `("ollama", "Ollama")` to `("local", "On-device")`. The panel is
`ui/screens/local_models_panel.py`, `LocalModelsPanel` (a `git mv` of `ollama_panel.py`, keeping
`SectionCard`, `ConfirmDialog`, `close_dialog`, the preset editor sheet and the 360 dp layout).
Cards:

1. **Engine:** state chip (Not loaded / Loading… / Ready · `<stem>` · RAM / Generating). Unload
   button. Context and Threads fields (Prefs `local_llm_settings`). One line of system info from
   `llama_print_system_info()`, shown on first load.
2. **Presets:** each row shows name, note and size. State and action:
   - **Download**: one tap downloads, then loads, then sets the current model through
     `use_model("local/<stem>")`;
   - a progress bar with percent, MB/s, ETA, Pause and Cancel;
   - **Use** (downloaded);
   - **In use** (current).
   The overflow menu has Edit, Rename, Remove preset, Delete file. The card ends with Add preset
   and Restore defaults (the existing editor; fields: Name, "Hugging Face model (repo, repo:quant or
   URL)", File (optional), Note, Context).
3. **Add from Hugging Face:** a field accepting any id or URL (hint
   `unsloth/Qwen3.5-2B-GGUF · …/resolve/main/x.gguf`) and a Download button. It runs `list_files`,
   then the auto-pick or the file chooser, then the download. An option saves the result as a
   preset.
4. **Downloaded models:** files not covered by a preset, with Use, Load and Delete. Paused parts are
   listed here with Resume and Cancel.

Keys follow the existing scheme: `local-panel`, `local-preset-<i>`, `local-download-field`,
`local-engine-state`, …

## 4. Termux removal

Delete:
- `src/mobile/app/glossarion_mobile/services/ollama_termux.py`. Its keepers move to
  `services/local_llm/presets.py` (preset load/save/restore/validate, `MAX_PRESETS`, `_clean`) and
  `services/local_llm/downloads.py` (`PullManager` → `DownloadManager`, `PullState` →
  `DownloadState`, `_PERCENT_RE` dropped since progress is in bytes). `installed()`, `is_loopback`,
  `route_model` (now `local/<stem>`) and `normalize_model` (replaced by `hf.parse_ref`) are not
  needed any more.
  Gone: `TERMUX_*`, `INSTALL_COMMAND`, `SERVE_COMMAND`, `SETUP_COMMAND`,
  `FIRST_RUN_SETUP_COMMAND`, `manual_steps`, `pull_command`, `TermuxOutcome`, `run_in_termux`,
  `_open_termux`, `server_status`.
- `ui/screens/ollama_panel.py` becomes `ui/screens/local_models_panel.py`. The server card,
  "Termux steps", "Set up in Termux", "Open Termux", "Get Termux", "Use this phone" and the
  Ollama pull card are removed.
- `ui/screens/model_manager.py`: the docstring paragraph, `TABS`, `ollama_pulls` / `self.ollama` /
  `ollama_box` / `ollama_panel()` / `_show_ollama` become `local_engine` / `local` / `local_box` /
  `local_panel()` / `_show_local`. The FAB label "Add preset" stays. `ModelsKeysFeature.__init__`
  builds `LocalEngineService` instead of `PullManager`.
- `ui/screens/endpoints.py` lines 89–91: the reason text no longer says "Model manager › Ollama to
  run Ollama in Termux". It now says "…or Model manager › On-device to run a model on this phone".
- `services/native.py`: the stub's and the bridge's `termux_status`, `termux_run` and `open_termux`
  (~195–205, ~565–582). Add `exclude_from_backup`.
- `extensions/flet_glossarion_native`:
  - `src/flet_glossarion_native/native.py`: the Termux block (~540–569); add `exclude_from_backup`;
  - `…/android/src/main/kotlin/…/GlossarionNativePlugin.kt`: the method-table entries (363–365),
    the Termux section (569–650), the constants (869–875) and the header comment (52–53); drop
    imports that become unused (`ActivityCompat`, if nothing else uses it);
  - `…/android/src/main/AndroidManifest.xml`: the comment (22–26),
    `<uses-permission android:name="com.termux.permission.RUN_COMMAND" />` and
    `<package android:name="com.termux" />` (remove `<queries>` if it becomes empty);
  - `…/lib/src/native_service.dart`: the cases at 358–364 and the defaults at 418–422; add
    `exclude_from_backup` (iOS: method channel; others: false);
  - iOS Swift plugin: `exclude_from_backup`;
  - `tests/test_native_defaults.py`: drop the Termux rows (109–111) and the
    `test_termux_calls_forward_on_android_and_match_the_platform_side` test (262–295); add
    `exclude_from_backup` defaults and a regression test that `termux` / `RUN_COMMAND` appear in
    none of the manifest, Kotlin, Dart, Python bridge or `src/mobile/pyproject.toml`.
- `src/mobile/pyproject.toml` (package A): the `"com.termux.permission.RUN_COMMAND" = true` line
  and its two comment lines.
- `src/mobile/tests_host/test_ollama_termux.py`: delete it and replace it with the new tests in
  section 5.

Unchanged: the shared `ollamapull/` route, the Local AI screen's "Managed Ollama on your PC" card
(`remote_routes` `ollamapull`), and LAN `ollama/` / `lmstudio/`.

## 5. Testing

### Host unit tests (package B, `src/mobile/tests_host/`)

- `test_local_llm_presets.py`:
  - the four defaults are data with the exact repo, file, size and sha256 above;
  - rename / edit / remove / restore persist in Prefs;
  - the legacy `ollama_presets` key is dropped;
  - validation messages.
- `test_local_llm_hf.py`: `parse_ref` over every accepted form and the rejects; `pick_file` order;
  the exclusion of mmproj, imatrix and split files. The tree API is a JSON fixture shaped like the
  real `unsloth/Qwen3.5-0.8B-GGUF` listing (no network).
- `test_local_llm_downloads.py` runs against a local fake HF: a stdlib `ThreadingHTTPServer` in the
  test with `/api/models/...` and `/resolve/...` honouring `Range` and serving a small random
  payload. It covers a fresh download, pause then resume (206, re-hash), a server that ignores
  Range (200 restarts), a checksum mismatch, the free-space refusal (monkeypatched `disk_usage`),
  cancel deleting the part, progress throttling on the loop, and the registry written atomically.
  HF base URLs are injectable (`hf.HF_ENDPOINT`, default `https://huggingface.co`, also from env
  `HF_ENDPOINT`).
- `test_local_llm_server.py`: `LocalLLMServer` + `FakeEngine`, covering:
  - token and Host guard 404s;
  - `/v1/models`;
  - non-streaming and SSE responses through the **real** `openai` SDK and through `UnifiedClient`
    with model `local/<stem>` and `GLOSSARION_LOCAL_LLM_URL` set (proves the shared route end to
    end on the host);
  - 404 for a missing model, 400 for context overflow and for images, 409 when pinned, 503 when the
    queue is full;
  - a client disconnect stops generation (FakeEngine counts tokens after the close);
  - `<think>` stripping;
  - `enable_thinking` passed to the formatter.
- `test_local_llm_service.py`:
  - one model loaded at a time;
  - the unload policies, driven by fake lifecycle names and a fake clock;
  - the RAM gate (psutil and sysconf monkeypatched);
  - thread and ctx defaults;
  - the catalog refresh after a download;
  - `ENGINE_VERSION` equals the pyproject pin.
- `test_local_models_panel.py` (replaces the panel tests of `test_ollama_termux.py`):
  - the tab is labelled "On-device";
  - one tap on Download goes through download (FakeEngine + fake HF), load and `use_model`;
  - progress rows, Pause / Resume / Cancel / Delete;
  - the editor; layout at 360 dp and 320 dp (no Row overflow, as `test_ui_flows` checks).
- Update `test_models_keys.py` / `test_model_manager_perf.py` / `test_u9_gap_round*.py` /
  `test_devfix_issue19.py` wherever they use `ollama_pulls`, the `ollama` tab or the old panel keys.
- Desktop `tests/`: the route tests from section 2.

Run them with
`cd src/mobile && PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_local_llm_*.py tests_host/test_local_models_panel.py tests_host/test_models_keys.py`,
and from the repo root
`python -m pytest -p no:cacheprovider -W ignore tests/test_model_options_catalog.py tests/test_openai_token_compatibility.py tests/test_local_llm_route.py`,
plus the extension's `tests/test_native_defaults.py`.

### One real local check on Windows (package B; scratch venv, never `src/mobile/.venv`)

```bash
SCR="$LOCALAPPDATA/Temp/claude/llama-host"            # scratch, outside the repo
py -3.12 -m venv --system-site-packages "$SCR/venv"    # host 3.12 only; system site gives requests/openai/etc.
"$SCR/venv/Scripts/python.exe" -m pip install --only-binary :all: \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.3.32 diskcache jinja2              # py3-none-win_amd64 CPU wheel
```

The driver is `$SCR/drive_llama.py`, a real file. It sets `GLOSSARION_LIBRARY_DIR`,
`OUTPUT_DIRECTORY`, `HOME`, `USERPROFILE` and `GLOSSARION_MODELS_DIR` to `$SCR/...`, puts
`src/mobile/app` and `src` on `sys.path`, and then:

1. downloads `unsloth/SmolLM2-135M-Instruct-GGUF` `SmolLM2-135M-Instruct-Q4_K_M.gguf` (about
   100 MB) with `DownloadManager`. It kills the transfer at about 40% and resumes it, checking the
   206 path and the sha256;
2. downloads the Garbage preset (533 MB) the same way;
3. for each model: `LocalEngineService` load, then `UnifiedClient(model="local/<stem>")` sends one
   chat (non-streaming and streaming) and one short `TranslateKRtoEN`-style system+user request;
4. asserts non-empty output with no `<think>` text for Qwen3.5 (this proves the Qwen3.5 GGUF
   template renders in the jinja2 sandbox with `enable_thinking=False`), and prints load seconds
   and tokens/s;
5. checks Stop: closing the client mid-stream drops server-side generation within 1 s.

src/config.json is never touched: the driver passes model and settings explicitly.

### Emulator check (package A drives it; the suite itself is package B)

- **Self-test additions** (`diagnostics/selftest.py`, package B):
  - smoke: a `llama_engine` check (section 1, check 3). On the host it raises `CheckSkipped` when
    `llama_cpp` is not importable;
  - e2e: a `local_route_fake_engine` check: `LocalLLMServer` + `FakeEngine` + `UnifiedClient`
    `local/fake-echo`, offline;
  - new suite `llama`, reached by the deep link `glossarion://app/__selftest__?suite=llama`. It
    downloads the preset named by `GLOSSARION_LLAMA_SUITE_PRESET` (default "Garbage") into
    `<models>`, reusing a verified file. It loads with `n_ctx=2048`, generates at most 16 tokens
    through `UnifiedClient` → server, and unloads. It prints
    `GLOSSARION_LLAMA {"model":…,"sha256_ok":true,"load_s":…,"tokens":n,"tok_s":…,"system_info":"…"}`
    and passes when the text is non-empty and the sha256 matches.
- **`ci/android_smoke.sh`** (package A): after the e2e step, add `RUN_LLAMA` (default 0),
  `LLAMA_URL` (default `glossarion://app/__selftest__?suite=llama`), `LLAMA_TIMEOUT` (1800 s) and
  `run_suite llama …`, and save the `GLOSSARION_LLAMA` line to `llama_result.txt`.
- **`build-mobile.yml`** (package A): dispatch input `llama_suite` (auto/on/off), policy line
  `ANDROID_LLAMA_SUITE_POLICY: optional`, and `RUN_LLAMA=1` passed to the android-smoke job when
  the suite is on. The emulator already has 4 GB RAM and a 6 GB disk (APK ~0.3 GB + model
  0.53 GB). `test_ci_release.py` pins the input, the policy line and the one-line promotion.
- **Locally:** `gh run download <run> -n X_Glossarion-Android-APK` from the branch's Build Mobile
  run. Start the AVD (`glossarion`, API 35 x86_64, 2 GB RAM; enough for 0.8B at `n_ctx` 2048), then
  run `RUN_LLAMA=1 PATH="$LOCALAPPDATA/Android/Sdk/platform-tools:$PATH" bash src/mobile/ci/android_smoke.sh <x86_64.apk> <scratch logdir>`.
  Then check by hand with `adb`:
  - `adb shell run-as com.glossarion.app ls -l files/…/models` shows the GGUF and `models.json`;
  - `adb logcat -s flet.python | grep GLOSSARION_LLAMA`.
- The iOS simulator smoke job runs the smoke suite unchanged, so it covers `llama_engine` (dylib
  load). The `llama` suite on the iOS simulator stays off in v1.

## 6. Work split (two parallel packages, disjoint files)

The interface contract between them, fixed here:
- the self-test suite name `llama`;
- the markers `GLOSSARION_LLAMA` and the `llama_engine` check name;
- the env names `GLOSSARION_LOCAL_LLM_URL`, `GLOSSARION_MODELS_DIR` and
  `GLOSSARION_LLAMA_SUITE_PRESET`;
- the pin `llama-cpp-python==0.3.32` and `ENGINE_VERSION = "0.3.32"`.

### Package A: packaging, wheel checks, CI

Files (only these):
- `src/mobile/pyproject.toml`: the two pins; remove the RUN_COMMAND permission and its comment;
- `src/mobile/tools/check_mobile_wheels.py`: Requires-Dist from flet-index wheel METADATA
  (section 1, check 1);
- `src/mobile/ci/wheels/wheels.py`: `verify-site-packages` with the llama profile (check 2);
- `src/mobile/tests_host/test_check_mobile_wheels_wheelhouse.py`: tests for both;
- `.github/workflows/build-mobile.yml`: verify step after every `flet build` (android, aab, ios,
  ios-signed, ios-simulator); `llama_suite` input; `ANDROID_LLAMA_SUITE_POLICY`; `RUN_LLAMA` in
  android-smoke; upload `llama_result.txt` with the smoke logs;
- `src/mobile/ci/android_smoke.sh`: the llama suite step;
- `src/mobile/tests_host/test_ci_release.py`: pins for the new input, policy and suite;
- `src/mobile/README.md`: the on-device engine section (pin, size, storage, the RAM notes), the CI
  table row and the paths-contract entry `GLOSSARION_MODELS_DIR`;
- contingency only: `src/mobile/ci/wheels/recipes/llama-cpp-python/**`,
  `src/mobile/ci/wheels/wheelhouse.toml`, `.github/workflows/mobile-wheels.yml`.

A's acceptance:
- `check_mobile_wheels.py --wheelhouse-plan` shows flet-libcpp-shared on Android, 0 errors;
- the branch's Build Mobile run is green with the verify steps;
- the `llama` suite passes on the CI emulator with `llama_suite=on`.

### Package B: engine, server, downloads, UI, Termux removal, tests

Files (only these):
- shared: `src/unified_api_client.py`, `src/model_options.py`,
  `tests/test_model_options_catalog.py`, `tests/test_openai_token_compatibility.py`,
  `tests/test_local_llm_route.py` (new);
- new package `src/mobile/app/glossarion_mobile/services/local_llm/` (`__init__.py`,
  `presets.py`, `hf.py`, `downloads.py`, `engine.py`, `server.py`, `service.py`);
- `src/mobile/app/glossarion_mobile/services/ollama_termux.py` (delete),
  `services/native.py`, `services/model_catalog.py`;
- `src/mobile/app/glossarion_mobile/runtime_bootstrap.py` (`GLOSSARION_MODELS_DIR`), `app.py`
  (lifecycle hook);
- `src/mobile/app/glossarion_mobile/ui/screens/model_manager.py`, `ui/screens/ollama_panel.py` →
  `ui/screens/local_models_panel.py` (`git mv`), `ui/screens/endpoints.py`,
  `ui/sheets/model_sheet.py` (the "Get on-device models…" row);
- `src/mobile/app/glossarion_mobile/job_kinds/translate.py`, `job_kinds/glossary.py` (only if the
  batch/worker override lands there);
- `src/mobile/app/glossarion_mobile/diagnostics/selftest.py`, `diagnostics/e2e.py`;
- `src/mobile/extensions/flet_glossarion_native/**` (Python bridge, Kotlin, manifest, Dart, Swift,
  `tests/test_native_defaults.py`);
- `src/mobile/tests_host/test_ollama_termux.py` (delete), the new
  `tests_host/test_local_llm_{presets,hf,downloads,server,service}.py` and
  `tests_host/test_local_models_panel.py`, plus the updates to `test_models_keys.py`,
  `test_model_manager_perf.py`, `test_u9_gap_round3.py`, `test_u9_gap_round4.py` and
  `test_devfix_issue19.py`;
- `src/mobile/docs/UI_SPEC.md` (§4.11 Model Manager › On-device) and
  `src/mobile/tests_host/` feature-map fixtures, if the tab list is pinned there.

B's acceptance:
- the host tests above are green;
- the Windows real check passes for SmolLM2-135M and Qwen3.5-0.8B;
- `python tools/collect_backend.py --check` is still green;
- `grep -ri termux src/mobile` matches only this plan and the new regression test.

### Integration order

1. Both packages land on `llama-engine`; neither edits the other's files.
2. Run the full mobile and desktop test sets (known unrelated failures excepted).
3. Dispatch `gh workflow run build-mobile.yml --ref llama-engine -f targets=all -f llama_suite=on`.
4. Run the local emulator check on that run's x86_64 APK.
5. The review step merges to main.

## 7. Risks and what decides them

| Risk | Detected by | Response |
|---|---|---|
| The index wheel fails to load on device (loader, libc++ duplicate, simulator dylib) | `llama_engine` in the Android smoke and iOS simulator smoke; `verify-site-packages` | contingency A' (own recipe), or a `[tool.flet.cleanup]` / extract fix for the specific failure |
| Qwen3.5 template does not render or think tags leak | Windows real check (step 4) and the `llama` suite | adjust `NoThinkFormatter`; the strip fallback is already in the server |
| CPU-baseline speed too slow on phones (no dotprod) | tok/s in `GLOSSARION_LLAMA` and the owner's phone | the KleidiAI experiment in an own-recipe build (A'), measured before adoption |
| iOS memory kills for 4B/9B | owner's device check; the RAM gate refuses above 50% of RAM | preset notes say so; the gate prevents a load that would be killed |
| Client timeout during long CPU generations or queue waits | Windows check with a long chapter; e2e | the local route uses the ollama/lmstudio timeouts; batch off for `local/` jobs |
| HF changes the resolve/redirect behaviour | download tests against the real HF in the Windows check and the `llama` suite | `hf.py` isolates the URLs and `HF_ENDPOINT` is overridable |
