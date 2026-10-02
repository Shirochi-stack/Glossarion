# Tor for OCZ and Opera

`ocz/` translation calls and `search/opera` (including `search/opera-think`)
chat POSTs automatically use a managed Tor HTTP CONNECT proxy. A unique proxy
authentication identity isolates each translation call onto a separate circuit.
Opera's token-rejection retry receives another identity. Concurrent calls do not
change one another's identity, and an OpenCode call keeps its identity for its
CLI lifetime, including internal requests and retries. Separate circuits may
still use the same exit IP; a different IP on every call is not guaranteed.

Tor is discovered on PATH, in the cached expert bundle, or in common Tor Browser
locations. When absent, the official stable expert bundle for the current
platform and architecture is downloaded over HTTPS from the Tor Project and
installed under `~/.glossarion/tor/bundle`. Archive extraction prevents traversal
outside the installation directory. Downloads rely on HTTPS; detached release
signatures are not verified. Unsupported architectures produce a setup error.

The app starts its own client-only Tor instance on a random loopback HTTP proxy
port with `IsolateSOCKSAuth`. It waits up to 120 seconds for full bootstrap,
reuses that instance for later calls, restarts it if it exits, and terminates it
at normal app exit. Cancellation is checked during download and bootstrap.
Setup failures are reported rather than silently sending chat requests directly.

Optional environment settings:

- `GLOSSARION_TOR_BINARY`: explicit path to a Tor executable.
- `GLOSSARION_TOR_DIR`: override the installation/cache directory.

Paid `oc/`, `opencode/`, and `opencode-go/` aliases retain their existing routing.
Tor installation downloads, Opera browser token minting, and local CLI/CDP
control connections retain their existing transport. Existing SOCKS listeners
are not reused because their circuit isolation configuration cannot be assumed.
No additional SOCKS Python dependency is needed.

References: [Tor HTTP CONNECT isolation](https://spec.torproject.org/http-connect.html),
[Bun proxy environment support](https://bun.sh/docs/runtime/networking/fetch),
and [official expert bundles](https://download.torproject.org/tor/).
