# Arena Login

`autharena_proxy.py` installs uv, managed Python 3.12, and the pinned
CloudWaddie/LMArenaBridge runtime and its Chromium browser under
`~/.glossarion/autharena_proxy`.
Neither system Python nor an extension is required. Frozen builds include the
worker's source and the existing token-encryption module as data files.

Use **Arena Login** beside the model selector or in a multi-key model field.
For login, the app opens installed Chrome with a fresh, app-owned regular
profile (not an incognito context). If Chrome is unavailable, it uses the
automatically installed Chromium executable with the same regular-profile flow. It navigates
to Arena's homepage, and activates its sign-in control. No existing browser or
remote-debugging setup is needed. Finish sign-in and any website challenge in
that window; the app captures the Arena session automatically. Google may still
reject sign-in from a connected browser; this change does not guarantee Google
login acceptance. This is session
authentication, not an official Arena OAuth API.

| Model prefix | Account |
| --- | --- |
| `autharena/` | Stored account `0` |
| `autharena0/` | Rotate all saved accounts |
| `autharena1/` | Stored account `0` |
| `autharena2/` | Stored account `1` |
| `autharenaN/` | Stored account `N - 1` |

The inline selector is hidden for the bare and rotating routes. Numbered routes
show numeric account labels and **+ New**. Pool-mode Arena Login opens account
selection. Each login opens a fresh isolated browser session, so a new account
does not inherit another account's sign-in. A login keeps
its original target if the model or another key row changes while it is open.

Translation requests have separate conversation state and isolated internal
browser contexts per saved account. The app owns and closes this browser;
personal browser profiles are not used. Credentials use Glossarion's existing encrypted
storage. The loopback service requires a generated authentication key.

Arena always streams. Outside batch mode, `LOG_STREAM_CHUNKS` controls log
visibility. During batch mode, `ALLOW_AUTHGPT_BATCH_STREAM_LOGS` controls it.
As with Antigravity, visible forced streams include reasoning. The optional
streaming and generic batch-streaming toggles do not disable this transport.
Interrupted streams are errors rather than completed translations.

`AUTHARENA_PROXY_DATA_DIR` changes the runtime and browser installation location.
Personal browser cookie databases are never read. The temporary login profile
is removed after its browser closes; only captured Arena credentials are retained
in the encrypted account store. Translation contexts remain isolated per account.

For standalone use, run `python src/autharena_proxy.py` to keep the proxy running,
`python src/autharena_proxy.py --status` for a read-only health check, or
`python src/autharena_proxy.py --login autharena1/` to sign into stored account `0`.

Windows runtime installation, authenticated startup, and a minimal onefile
executable bootstrap have been exercised. An offline test against the pinned
bridge checks streaming, concurrent account isolation, rotation, fresh request
IDs, and cancellation before dispatch using a simulated managed browser.
Windows automatic Chromium installation and launch have also been exercised.
macOS/Linux installation paths are implemented but require native
validation. Live signed-in Arena generation also requires end-to-end validation.
