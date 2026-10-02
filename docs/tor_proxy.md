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
shared Tor process while preserving that identity. The port stays open until
the response stream or CLI call finishes, then closes. The internal Tor listener
is shared so changing request ports does not require bootstrapping Tor again.

Tor is discovered on PATH, in the cached expert bundle, or in common Tor Browser
locations. When absent, the official stable expert bundle for the current
platform and architecture is downloaded over HTTPS from the Tor Project and
installed under `~/.glossarion/tor/bundle`. Archive extraction prevents traversal
outside the installation directory. Downloads rely on HTTPS; detached release
signatures are not verified. Unsupported architectures produce a setup error.

The app starts its own client-only Tor instance on a random loopback HTTP proxy
port with `IsolateSOCKSAuth`. Each caller waits up to 120 seconds for full bootstrap,
reuses that instance for later calls, restarts it if it exits, and terminates it
at normal app exit. Cancellation is checked during download and bootstrap.
Callers waiting behind another request's bootstrap can also cancel. Only one
thread installs or starts Tor; once ready, parallel callers receive independent
proxy identities and their HTTPS transfers proceed concurrently. Console output
is captured directly so early Windows configuration errors remain visible.
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

Paid `oc/`, `opencode/`, and `opencode-go/` aliases retain their existing routing.
Tor installation downloads, Opera browser token minting, and local CLI/CDP
control connections retain their existing transport. Existing SOCKS listeners
are not reused because their circuit isolation configuration cannot be assumed.
No additional SOCKS Python dependency is needed.

OCZ captures error-level CLI diagnostics so internal model lookup failures are
reported directly. Missing or unavailable models are configuration errors rather
than retryable transport failures. A model listed by Zen may still be deprecated
in OpenCode or unavailable at its upstream provider. Tor does not resolve that.

References: [Tor HTTP CONNECT isolation](https://spec.torproject.org/http-connect.html),
[Bun proxy environment support](https://bun.sh/docs/runtime/networking/fetch),
and [official expert bundles](https://download.torproject.org/tor/).
