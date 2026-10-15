# Update providers and SHA-256 verification

The desktop updater checks GitHub first, then uses stable version tags in public
GitLab project `87623444` (`shirochi-stack/Glossarion`) if GitHub cannot supply
release metadata. GitHub URLs, version comparisons, skipped-version settings,
startup preferences and the 30-minute cache remain supported. No client token is
required or embedded.

## Provider selection

Fallback covers connection failures/timeouts, HTTP 403/404/429 and
5xx server errors, and malformed release JSON/schema. A certificate failure is a
hard error. A healthy current/skipped GitHub release does not select a newer
GitLab release. When GitHub has no build for the current OS, the checker may use
the **same tag** on GitLab. Release history cannot replace the selected release.

GitLab discovery requests version-ordered `v*` tags, follows up to 20 pages of
100 tags, validates and sorts versions with `packaging.version.Version`, and
excludes prerelease/dev versions and tags without releases. Exceeding this
bounded discovery window is reported as an error, not “up to date.” Future
scheduled releases are not an update channel; publish releases only when ready.
GitLab source archives are never update binaries.

Desktop checks run on a GUI-free worker with a snapshot of update configuration.
Only queued GUI-thread slots save configuration, open results, or display errors.
The loading dialog is modeless and cancellable; repeated clicks share one active
check. Closing/canceling it suppresses late results and leaves the check cache
unchanged. An in-flight socket read may take up to five seconds to return before
the worker notices cancellation; the main window remains usable throughout.
Metadata checks have a 20-second overall budget, three-second connection and
five-second read limits, bounded redirects, and cancellation checks between
requests/chunks/pages. Empty history starts an asynchronous check instead of
fetching releases on the GUI thread.

If a GitHub binary transfer fails with one of the network/HTTP conditions above,
the updater can retry once against GitLab. The mirror must have the same tag,
exact filename, exact size and SHA-256. Different variants/architectures or
independently rebuilt binaries with different hashes are not interchangeable.
Publish identical final bytes to both providers for download mirroring. A
checksum mismatch, conflicting/missing checksum metadata, or user cancellation
never triggers a bypass through another provider. Partial files are removed.

## Required release assets

Publish each final binary and its matching `<binary filename>.sha256.json` asset.
The sidecar format is:

```json
{
  "schema_version": 1,
  "tag": "v9.14.4",
  "name": "Glossarion v9.14.4.exe",
  "size": 60000000,
  "sha256": "<64 hexadecimal characters for the final binary>"
}
```

Generate it **after** packaging and any code signing:

```powershell
python tools/generate_update_metadata.py "src/dist/*.exe"
```

The same tool accepts quoted DMG/ZIP or other artifact globs. By default it uses
the GitHub tag when running on a tag, otherwise `src/app_version.py`'s version.
An explicit `--tag v9.14.4` is supported; a tag different from the packaged source
version is rejected by the CLI. It does not upload or sign anything. Never reuse
the unsigned binary's metadata for a signed binary.

GitHub's native asset `digest: sha256:...` plus its exact API size is also
accepted. When a sidecar and native digest both exist, they must agree. Sidecars
are size bounded and must match the selected tag, filename, size and schema.
On GitLab the sidecar supplies the size absent from release links. Attach both
files as release asset links pointing to durable, anonymously downloadable HTTPS
packages on `gitlab.com`. A tag alone, private package, expiring job artifact, or
release containing only source code is insufficient.

All original GitHub workflows are preserved. Their native SHA-256 asset digests
are accepted by this updater; when manually publishing a checksum sidecar, run
the generator against the final chosen binary and upload both. Do not reuse a
sidecar from before signing or from a different build. GitLab release automation
generates and attaches the required sidecars without changing GitHub workflows.

`.gitlab/release.py` retains its existing protected-tag/manual publication gates.
After validating the build manifests, it creates per-file consumer metadata,
preflights binary and sidecar package paths before writing, and links both to
the existing release. The build's `manifest.json` is internal CI evidence; it
is not the consumer checksum format. The separate automatic release-created
workflow and its append-only publisher are documented in [GitLab CI](gitlab_ci.md).

## Acceptance and legacy releases

The desktop keeps the requested `Glossarion` filename and **60,000,000 byte**
minimum gates. It additionally requires a safe basename, exact published size,
and SHA-256 match. TLS certificate checking remains enabled in frozen and source
builds; HTTP downgrade redirects are blocked before following them. A packaged
runtime missing working CA support fails securely instead of disabling TLS.
Downloads use `.part` files; verified bytes are promoted to the selected file,
then hashed again immediately before installation.

Older releases without a native SHA-256 digest or matching sidecar remain
visible in release notes/history, but automatic download/install is blocked with
an explanation. There is no “skip verification” control or size-only verified
state. Older clients do not acquire this protection until updated to this code.

SHA-256 here checks integrity against metadata delivered over verified HTTPS.
It does **not** prove the publisher's identity or defend against a compromised
release account that replaces both binary and checksum. No signed manifest,
publisher pinning, signing keys or credentials are introduced.

OS/architecture and build-variant selection remain supported, including macOS
ARM/Intel. A single foreign-platform asset is no longer selected automatically.
Windows retains its replacement/restart behavior after verification; macOS/Linux
packages are opened for manual installation without running Windows commands.
Existing Linux ZIP packaging remains outside the installable formats accepted
by this updater; adding ZIP extraction/install behavior is separate work.

Mobile shares provider discovery/cache/skip handling and still opens links in
the system browser. Those external browser downloads are **not** claimed to be
SHA-256-verified by the app and are never installed by the desktop updater.
