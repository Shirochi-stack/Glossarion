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
| `autharena/` | Account `#0` (first saved account) |
| `autharena0/` | Rotate all saved accounts |
| `autharena1/` | Account `#1` (second saved account) |
| `autharena2/` | Account `#2` (third saved account) |
| `autharenaN/` | Account `#N`, for N ≥ 1 |

The inline selector is hidden for the bare and rotating routes. Numbered routes
show **#0**, **#1**, etc., matching AuthGPT, and **+ New**. Pool-mode Arena Login opens account
selection. Each login opens a fresh isolated browser session, so a new account
does not inherit another account's sign-in. A login keeps
its original target if the model or another key row changes while it is open.

Translation requests have separate conversation state and isolated internal
browser contexts per saved account. New requests use Arena's current
`direct-battle` creation mode (the Direct UI route), retaining the selected
model for the first turn; the pinned bridge's legacy `direct` value is adapted.
Translation runs headlessly; only an explicit
Arena Login opens a visible browser. Upstream HTTP errors include Arena's response
details when provided, rather than only the status code. The app owns and closes this browser;
personal browser profiles are not used. Credentials use Glossarion's existing encrypted
storage. The loopback service requires a generated authentication key.

Arena always streams. Outside batch mode, `LOG_STREAM_CHUNKS` controls log
visibility. During batch mode, `ALLOW_AUTHGPT_BATCH_STREAM_LOGS` controls it.
As with Antigravity, visible forced streams include reasoning. The optional
streaming and generic batch-streaming toggles do not disable this transport.
Interrupted streams are errors rather than completed translations.

Token generation, loader initialization, readiness waits, and cache/refresh now
use the pinned LMArenaBridge `recaptcha.py` helpers. The adapter binds them to
each account's request page and forces a fresh token before submission. A rejected
CAPTCHA gets a bounded fresh-token attempt in the app-owned browser. The custom
CAPTCHA widget has been removed. Successful streams and partial output are never
replayed. Account credentials remain encrypted.

Token requested/received, submission, and response-header stages are logged
without token values. The application acknowledges dispatch before the worker
sends to Arena, starting the watchdog and API-call progress at that boundary
rather than during browser and CAPTCHA preparation. The acknowledgment no longer
has a separate 30-second expiry; request timeout and cancellation still apply.
Browser tasks are stopped before an account's request state can be reused.
Dispatch failures are provider errors unless a real cancellation was requested.
Upstream errors retain Arena's HTTP status and Retry-After value when supplied,
including HTTP 429, instead of reporting them as CAPTCHA failures.

If Arena serves a security interstitial instead of its homepage, the proxy opens
an app-owned browser and waits for interactive verification before obtaining a
CAPTCHA token or submitting a translation. An interstitial is no longer treated
as a missing reCAPTCHA loader. The wait is cancellable and bounded to three minutes,
within the upstream five-minute total browser-setup budget;
blocked challenge scripts or DNS failures still require working network access.

`AUTHARENA_PROXY_DATA_DIR` changes the runtime and browser installation location.
The proxy stores full model IDs and metadata in `models.enc`, separately from
the GUI's display-name catalog. It reuses this cache across restarts for 24 hours
and retains the last successful records if a refresh is blocked or fails.
Arena Login also captures these records from its browser. A cache failure now
reports whether the page was blocked or its model data could not be parsed.
Personal browser cookie databases are never read. The temporary login profile
is removed after its browser closes; only captured Arena credentials are retained
in the encrypted account store. Translation contexts remain isolated per account.

For standalone use, run `python src/autharena_proxy.py` to keep the proxy running,
`python src/autharena_proxy.py --status` for a read-only health check, or
`python src/autharena_proxy.py --login autharena/` to sign into account `#0`.

Windows runtime installation, authenticated startup, and a minimal onefile
executable bootstrap have been exercised. An offline test against the pinned
bridge checks streaming, concurrent account isolation, rotation, fresh request
IDs, and cancellation before dispatch using a simulated managed browser.
Windows automatic Chromium installation and launch have also been exercised.
macOS/Linux installation paths are implemented but require native
validation. Live signed-in Arena generation also requires end-to-end validation.
