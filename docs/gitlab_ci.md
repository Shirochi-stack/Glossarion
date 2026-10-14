# GitLab alternatives alongside GitHub Actions

The GitHub workflow files stay in place and retain their original execution
behavior. GitLab does not execute GitHub Actions YAML. `.gitlab-ci.yml` adds an
opt-in alternative: only a pipeline created through **Run pipeline** (`web`)
or the authenticated Pipelines API (`api`) is admitted. Pushes, mirrors, merge requests, schedules, and tags alone start
no GitLab jobs. No runner, secret, project setting, or live pipeline was
configured as part of this source change.

Choose the typed `task` input in the Run pipeline form. This also works when
the project disables pipeline-variable overrides; no access setting change
is needed. The default `source-check`
checks source syntax and the GitLab helper tests; it does not claim to run the
application test suites. A requested job fails when its prerequisites or
required tests are absent. Builds and dependency downloads can consume runner
time/storage. Android APK and wheel jobs now bootstrap their dependencies on
the existing `saas-linux-medium-amd64` hosted runner in a disposable Python
3.13 container. The SDK download is checked against Google's published digest,
and the Rust/NDK pins come from the existing wheelhouse manifest. This installs
no persistent runner and needs no signing or publishing credentials.

Select `task=mobile` for the original **Build Mobile (Android + iOS)** graph.
Its `targets` (`all`, `android`, `ios`), `emulator_smoke` (true),
`ios_simulator_smoke` (`auto`, `on`, `off`), `ui_tests` (`auto`, `on`, `off`),
and `android_legacy_packaging` (false) inputs preserve the GitHub defaults.
`auto` runs the required iOS simulator suite and leaves the optional Android
UI suite off. An explicitly enabled optional Android UI suite may fail without
failing the pipeline, matching the current GitHub policy. Android device smoke
remains blocking. The combined mobile task adds an explicit host preparation
job before the platform build jobs. Wheel compilation runs concurrently with
preparation, as it does on GitHub. The prepared backend/assets are transferred
to the build jobs only after host smoke, offline regression and E2E validation.
Build helpers require matching commit/pipeline provenance and verify the
bundle again before packaging. Individual platform tasks without this artifact
still perform all preparation checks themselves.

macOS/Xcode jobs use `saas-macos-medium-m1` and `macos-26-xcode-26`, with
Python 3.13, pinned Rust and GNU timeout bootstrapped in the disposable job.
This requires existing hosted macOS entitlement (no subscription is purchased).
Android device jobs start an isolated API-35 emulator on hosted Linux. They
use KVM when available, otherwise software acceleration, and fail with logs
if the emulator does not boot within 600 seconds. `targets=android` with `emulator_smoke=false` is
the build-only APK selection for hosted Linux; it does not validate a device.
`publish` and `signing_approved` are separate typed boolean inputs defaulting
to false; the signed tasks and existing-release upload still require their
blocking manual job and protected-ref gates. The ordinary `mobile` task never
selects signing or release publication.

## Workflow mapping

| Existing GitHub workflow | GitLab selection | Result |
| --- | --- | --- |
| `python-app.yml` | `desktop-tests` | Original dependency install, syntax/import checks, shared-module test list, import hygiene |
| `python-app.yml`, mobile prepare job | `mobile-check` | Backend/assets, host smoke, included mobile/extension tests and offline E2E |
| `build-windows.yml` | `windows` | Lite, NoCuda, TurboLite, Main, Heavy executables, unsigned |
| `build-linux.yml` | `linux` | Standard, NoCuda, TurboLite ZIPs with existing bundle checks |
| `build-macos.yml` | `macos` | ARM and Intel Lite/NoCuda DMGs |
| `build-all.yml` | `desktop` | All the desktop jobs above (12 variants) |
| `mobile-wheels.yml` | `wheels-android`, `wheels-ios` | Pinned self-built wheelhouses with structure/native/provenance checks |
| `build-mobile.yml` | `android`, `ios` | Verified debug-signed APKs or unsigned IPA, preceded by wheel builds and included tests |
| `build-mobile.yml`, workflow dispatch | `mobile` + platform/smoke inputs | Combined preparation, wheelhouses, APK/IPA and selected device suites |
| `build-mobile.yml` | `android-signed`, `ios-signed` | Blocking manual signing job after the platform's wheel build |
| `build-mobile.yml` | `ios-simulator` | Simulator app archive plus actual simulator smoke/cryptography checks |
| `build-mobile.yml` | `android-ui`, `android-smoke` | UI suite or APK smoke/E2E on an already running emulator/device |
| Linux/macOS tag release steps; mobile existing-release upload | `GLOSSARION_PUBLISH=true` on a protected version tag | Separate blocking manual upload to an **existing GitLab release** |

`.gitlab/ci.py` deliberately reuses named `run` steps from the checked-in
GitHub workflows, PyInstaller specs, and `src/mobile/tools` / `src/mobile/ci`
scripts. It is not a general Actions interpreter: marketplace `uses` steps
are not executed, unknown `${{ ... }}` expressions fail, and selected shell
steps must retain their expected names. This keeps build/verification logic
shared while GitHub remains usable. `--plan` lists the selected commands
without running them, checking runners, installing anything, or signing.

GitLab replaces checkout, artifact transfer, matrices and conditional jobs
with native checkout, `artifacts`, `needs`, `parallel:matrix`, and `rules`.
The small runner maps the command-output files used by reused shell steps
into a workspace scratch directory. GitHub permissions, tokens, trigger
events, and marketplace actions are not silently translated.

## Runner prerequisites

The desktop tags below are intentional custom tags. No matching runner means a job
waits; it does not fall back to an arbitrary shared machine. Use isolated,
disposable build environments. Installing/registering a runner, purchasing
capacity, or granting credentials is a separate owner action.

| Runner tag | Required environment |
| --- | --- |
| `glossarion-linux-py310` | Linux x64, Python 3.10 as `python`, Bash, Git, Cairo/Pango/GTK and headless Qt native libraries from `python-app.yml` |
| `glossarion-linux-py312` | Linux x64, Python 3.12, Bash, Git; for desktop builds: compiler/python headers, zip, pkg-config, cmake, and the exact apt-library list in `build-linux.yml` |
| `glossarion-windows-py312` | Windows x64, Python 3.12, PowerShell **7.4+** as `pwsh`, Git, MSYS2 MINGW64 with the GTK/Pango/Cairo package list from `build-windows.yml`; set `MSYS2_ROOT` if not `C:/msys64` |
| `glossarion-macos-arm64-py312` | Native ARM Python 3.12 on Apple Silicon, Bash, Git, create-dmg, pango, cairo, libffi, gdk-pixbuf, gobject-introspection, pkg-config, cmake |
| `glossarion-macos-x64-py312` | Same dependencies with x86_64 Python on an Intel-compatible Mac; an ARM interpreter does not produce the Intel variant |
| `saas-linux-medium-amd64` | Existing hosted Linux runner; Python 3.13 container, uv and disposable Android SDK/JDK 17/Rust setup are supplied by the jobs |
| `saas-macos-medium-m1` | Existing hosted macOS entitlement, `macos-26-xcode-26`; uv/Python 3.13, Rust targets, pkg-config and GNU timeout setup supplied by the jobs; CocoaPods/Xcode/simulators from the image |
| `saas-linux-medium-amd64`, device jobs | Android toolchain above plus isolated API-35 x86_64 emulator; setup supplies `ANDROID_SERIAL`, boot timeout and emulator log; KVM preferred, software acceleration fallback |

Provision Flet/Flutter-compatible native tools and accept required SDK/Xcode
licenses yourself. The mobile jobs use the locked project environment and
the existing Flet build driver. Rust version/targets, NDK and Python support
pins come from `src/mobile/ci/wheels/wheelhouse.toml`; the runner rejects a
wrong Rust version/target set. Wheel builds reuse the hash-checked forge
scripts. No GitLab wheel cache is trusted or persisted yet: requested wheels
are built and verified, then passed as artifacts through `needs`.

On GitLab.com, hosted macOS availability is plan-dependent and its default
image is not a substitute for these checked prerequisites. Windows hosted
runners use PowerShell. See [GitLab migration guidance](https://docs.gitlab.com/ci/migration/github_actions/),
[Windows runners](https://docs.gitlab.com/ci/runners/hosted_runners/windows/),
and [macOS runners](https://docs.gitlab.com/ci/runners/hosted_runners/macos/).

## Tests and fresh history

Reviewed-safe tests are included in Git. Drive copies are backups, not the
normal source of test files. No external-test skip path is enabled. An absent
suite fails the requested test/build job, and pytest failures propagate.

Desktop/mobile parity suites use fixed old commit IDs as `git show` oracles.
The included bounded oracle package now supports these reads from a fresh
history. GitLab jobs verify and expose it through a temporary Git object
directory before testing; they do not import ancestors or modify repository
history. Missing/corrupt fixtures fail. See [parity oracle instructions](parity_oracles.md)
for the exact scope, review, standalone wrapper, and reproducible local export.
The retained original history and all-refs bundle stay local and unchanged.

## Signing and release prerequisites

Unsigned desktop and mobile jobs need no release credentials. Existing GitHub
SignPath test-signing remains available there. It is **not** replaced with a
fake GitLab signing result: that action requires a GitHub artifact ID and the
provider's approved provenance integration. A GitLab SignPath integration is
a separate provider/configuration decision. macOS notarization is not added.

For a signed mobile job, use a protected ref, select the signed task, set
`GLOSSARION_SIGNING_APPROVED=true`, then explicitly start the blocking manual
job. Use `signing_approved=true`. Provision protected/masked variables through the platform's secure
settings, not the public YAML or the manual-job free-text variable form:

- Android: `FLET_ANDROID_SIGNING_KEY_STORE` (file variable or private runner
  path), `FLET_ANDROID_SIGNING_KEY_STORE_PASSWORD`,
  `FLET_ANDROID_SIGNING_KEY_PASSWORD`, `FLET_ANDROID_SIGNING_KEY_ALIAS`.
- iOS: an existing signing identity in the isolated runner's keychain and an
  installed matching profile; `IOS_TEAM_ID`, `IOS_PROFILE_UUID`,
  `IOS_SIGNING_CERTIFICATE`, `IOS_EXPORT_METHOD`. This helper neither imports
  certificates nor creates/modifies keychains or provisioning profiles.

Release upload additionally needs an existing GitLab release for the checked
out protected `v...` tag, permission for the job token to use the Generic
Package Registry and Release Links API, and `GLOSSARION_PUBLISH=true`.
Protect the `gitlab-release` environment if your plan supports it. The manual
job verifies every build manifest/commit/hash, rejects debug/simulator output,
refuses existing package paths or asset links, uploads the selected binaries
to the Generic Package Registry, and attaches release links. It does not
create a release, replace an existing asset, contact GitHub, or upload source
archives. A partially completed upload fails and needs owner review before a
retry; it does not silently clobber prior output. Build-only selections keep
ordinary job artifacts for seven days; registry assets have a separate
retention policy set by the owner.

The current application updater remains pointed at its existing release
source. Publishing a GitLab release does not change the updater's provider.
See [manual jobs](https://docs.gitlab.com/ci/jobs/job_control/),
[generic packages](https://docs.gitlab.com/user/packages/generic_packages/),
and [release API](https://docs.gitlab.com/api/releases/).

## Validation and limits

Run `python tests/test_gitlab_ci.py` after installing `.gitlab/requirements.txt`.
Plan examples: `python .gitlab/ci.py desktop --platform linux --variant nocuda --plan`
and `python .gitlab/ci.py mobile --target ipa --plan`.

Local validation covers the job graph/configuration, existing step references,
all build plans, honest missing-prerequisite failures, and release gating/hash
checks with mocked HTTP. It does not establish successful multi-platform
builds or signing. Run GitLab CI Lint and the selected jobs on the provisioned
runners before relying on release output. No remote lint/pipeline, full build,
signing operation, or release upload was executed during the initial migration.
The follow-up mobile wiring is covered by all task dependency graphs, all
platform/smoke/UI input combinations, and checks against the original dispatch
defaults. Server lint and actual pipeline status are separate runtime evidence;
successful static checks do not prove an APK/IPA was produced.
