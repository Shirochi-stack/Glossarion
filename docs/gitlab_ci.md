# GitLab alternatives alongside GitHub Actions

The GitHub workflow files stay in place and retain their original execution
behavior. GitLab does not execute GitHub Actions YAML. `.gitlab-ci.yml` adds an
opt-in alternative: only a pipeline created through **Run pipeline** (`web`)
is admitted. Pushes, mirrors, merge requests, schedules, and tags alone start
no GitLab jobs. No runner, secret, project setting, or live pipeline was
configured as part of this source change.

Choose `GLOSSARION_TASK` in the Run pipeline form. The default `source-check`
checks source syntax and the GitLab helper tests; it does not claim to run the
application test suites. A requested job fails when its prerequisites or
required tests are absent. Builds and dependency downloads can consume runner
time/storage; the user must choose and provision appropriate runners first.

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

The tags below are intentional custom tags. No matching runner means a job
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
| `glossarion-linux-mobile-py313` | Linux x64, Python 3.13, uv, Bash, Git, pkg-config, SQLite, Rust/Cargo; for Android: JDK 17, keytool, Android SDK tools/adb/sdkmanager and `ANDROID_HOME` |
| `glossarion-macos-mobile-py313` | macOS/Xcode compatible with the current `macos-26` workflow, Python 3.13, uv, Bash, Git, CocoaPods, Xcode command-line tools, simulator runtimes, Rust/Cargo, pkg-config, GNU `timeout` on PATH |
| `glossarion-android-emulator-py313` | Android toolchain above plus an available API-35 x86_64 emulator/device; `ANDROID_SERIAL` identifies it; emulator host needs working hardware acceleration/KVM |

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
job. Provision protected/masked variables through the platform's secure
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
signing operation, or release upload was executed during implementation.
