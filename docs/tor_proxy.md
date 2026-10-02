# Tor for OCZ and Opera

`ocz/` translation calls and `search/opera` (including `search/opera-think`)
chat POSTs optionally use a managed Tor HTTP CONNECT proxy. Tor is disabled by
default. Enable **TOR proxy rotation** in Other Settings, in the **search/opera
and ocz/ TOR Proxy Rotation** section above Gemini Free Browser Chunking.
The toggle is saved with settings and passed to translation subprocesses.
When disabled, these routes retain their normal transport without Tor setup.
A unique proxy
authentication identity isolates each translation call onto a separate circuit.
Opera's token-rejection retry receives another identity. Concurrent calls do not
change one another's identity, and an OpenCode call keeps its identity for its
CLI lifetime, including internal requests and retries. Separate circuits may
still use the same exit IP; a different IP on every call is not guaranteed.
Every Opera chat POST logs its Tor proxy address and a short circuit identity
label, even when Tor was already running. The label identifies the isolation
identity, not the exit IP; proxy credentials and bearer tokens are not logged.
OCZ also logs the proxy address and fresh circuit identity on every streaming or
buffered call. Each call uses a new loopback relay port and authentication
identity, including parallel calls. The relay forwards the connection to the
selected Tor instance while preserving that identity. The port stays open until
the response stream or CLI call finishes, then closes. Application retries rotate
ports and instances; requests internal to one CLI invocation retain its proxy.

Tor is discovered on PATH, in the cached expert bundle, or in common Tor Browser
locations. When absent, the official stable expert bundle for the current
platform and architecture is downloaded over HTTPS from the Tor Project and
installed under `~/.glossarion/tor/bundle`. Archive extraction prevents traversal
outside the installation directory. Downloads rely on HTTPS; detached release
signatures are not verified. Unsupported architectures produce a setup error.

The app rotates through four client-only Tor instances, starting each on demand.
Each has its own data directory, HTTP proxy port with `IsolateSOCKSAuth`, and
cookie-authenticated loopback control port. Each caller waits up to 120 seconds
for its instance to bootstrap. Instances are reused, restarted if they exit, and
terminated at normal app exit. Cancellation is checked during installation and
bootstrap. Installation is serialized; different instances can bootstrap in
parallel. Requests on a ready instance proceed concurrently. Console output
is captured directly so early Windows configuration errors remain visible.

On Opera HTTP 403/429 or OCZ rate-limit/access-block errors, the affected instance
receives `SIGNAL NEWNYM`. Accepted renewal commands have a ten-second cooldown
per instance. Existing streams continue; normal provider retry/backoff still
applies, and the next attempt rotates to the next pool instance. Control failures
are logged without replacing the original API error. Renewal does not guarantee
a distinct exit IP.
The application prints bootstrap percentages and stages, reports when startup
stays at the same stage for ten seconds, and announces when the proxy is ready.
Timeout errors include the last bootstrap stage and recent Tor console output.
If that wait expires, the Tor process and its downloaded relay information are
kept; later requests resume waiting on the same instance instead of restarting
bootstrap from zero. A proxy URL is returned only after full bootstrap. Discovery
logs the existing executable path and skips installation when one is found.
Setup failures are reported rather than silently sending chat requests directly.

Optional environment settings:

- `GLOSSARION_TOR_ENABLED`: `1` enables Tor and per-request ports; default `0`.
- `GLOSSARION_TOR_BINARY`: explicit path to a Tor executable.
- `GLOSSARION_TOR_DIR`: override the installation/cache directory.
- `GLOSSARION_TOR_INSTANCES`: instance pool size (1–8, default 4).

Paid `oc/`, `opencode/`, and `opencode-go/` aliases retain their existing routing.
Tor installation downloads, Opera browser token minting, and local CLI/CDP
control connections retain their existing transport. Existing SOCKS listeners
are not reused because their circuit isolation configuration cannot be assumed.
No additional SOCKS Python dependency is needed.

OCZ captures error-level CLI diagnostics so internal model lookup failures are
reported directly. Missing or unavailable models are configuration errors rather
than retryable transport failures. OCZ model polling uses `opencode models opencode`
and marks only its free model IDs as polled. Older HTTP-catalog confirmations are
ignored. CLI recognition does not guarantee upstream availability. Tor does not
resolve upstream model errors.

References: [Tor HTTP CONNECT isolation](https://spec.torproject.org/http-connect.html),
[Bun proxy environment support](https://bun.sh/docs/runtime/networking/fetch),
and [official expert bundles](https://download.torproject.org/tor/).
