# AuthArena user guide

`src/autharena.py` uses Arena's website Direct chat flow through your existing
browser session. In Glossarion, select an `autharena/` model, such as
`autharena/gpt-6-astra-medium`. Available models come from Arena's current website
catalog and can change.

This route uses your Arena website login without an API key. Each request starts
a fresh conversation. It does not resume a previous Arena chat or use Arena's
separate API gateway credentials.

## Set up your current browser

The **Glossarion Arena Browser Companion** extension connects Glossarion to your
normal Chrome, Edge, Brave, or other compatible Chromium browser profile. The
extension requires Chromium 120 or newer. You install it once in each browser
profile you want to use.

1. Select an Arena model and click **Arena Login**. Glossarion opens a local
   connection page using the system's default external browser and prepares the
   companion folder at `~/.glossarion/autharena_extension`.
2. Open that browser's extensions page, enable **Developer mode**, choose
   **Load unpacked**, and select the generated `autharena_extension` folder.
3. Refresh the local connection page, or click **Arena Login** again. The
   companion pairs this browser profile with the selected account number and
   opens Arena's own sign-in dialog. Choose Google or email there if sign-in is
   needed, and complete Arena's verification and first-use terms.
4. Return to Glossarion after login is verified. The button stays labeled
   **Arena Login**; its tooltip reports the selected account's status.

The repository template is in `assets/autharena_extension`. Load the generated
folder above: Glossarion adds the required `arena_page.js`, which is not present
in the template alone. After a companion update, reload the extension from the
browser's extensions page and reconnect.

Existing Arena login cookies stay inside that browser profile. The companion
requests access to Arena and Glossarion's local broker, with no cookies or
debugger permission. It creates its own Arena tab and retains your other tabs
and chats. Arena login opens its native sign-in dialog without selecting a
hardcoded model. Qt WebEngine and a separate browser profile created by
Glossarion are not used for this route, including in Lite and TurboLite builds.

## Account numbers and rotation

The main account selector uses plain numbers: **0**, **1**, **2**, and so on.
Choose **+ New** to add another account number. In **Multi API Key Manager**,
**Arena Login** appears in each Arena model field, including fallback and
dedicated key pools. Arena rows do not require an API key.

| Model prefix | Account selection |
| --- | --- |
| `autharena/` | Physical account slot `0`, initially paired with your current browser profile. |
| `autharena1/`, `autharena2/`, … | The corresponding numbered browser-profile binding. |
| `autharena0/` | Rotation through verified, connected accounts, including physical slot `0`. |

For another account, use a different existing browser profile, install the
companion there, and open that account number's connection page in that profile.
Each browser profile binds to one physical slot. Two tabs in the same profile
share the same Arena login and cannot hold separate Arena accounts. The
**Arena Login** action for `autharena0/` lets you choose a physical account number
or **+ New**; logging in does not create a separate pooled identity.

Keep the intended browser profiles connected while using rotation. A cached
login marker alone does not establish that a browser is connected or that its
Arena session is still valid. Sign-in and terms are verified in the browser;
sessions can expire and require another **Arena Login**.

Local slot and verification metadata remains under
`~/.glossarion/autharena_browser/<account-id>`. Browser bindings and local pairing
credentials are stored in `~/.glossarion/autharena-bridge.enc`. Neither contains
a copy of the current browser's cookies. A `chromium` directory left by the
older transport is not used by the current-browser companion.

Explicit sign-in or rate-limit rejections can advance to another eligible
account. An uncertain submitted request stops with an error instead of being
sent a second time. Arena's normal usage limits and interactive checks still
apply.

## Standalone commands

Run these from the repository root. The same companion setup is required for
standalone requests.

```powershell
python src/autharena.py --login
python src/autharena.py --model gpt-6-astra-medium --prompt "Reply with exactly: ok"
python src/autharena.py --account-id 1 --login
python src/autharena.py --account-id 1 --prompt "Hello"
python src/autharena.py --model autharena0/gpt-6-astra-medium --prompt "Hello"
```

An account's requests run one at a time. The timeout includes waiting for that
account and generating the response.

```powershell
python src/autharena.py --prompt-file prompt.txt
python src/autharena.py --system "Translate into English." --prompt-file chapter.txt
python src/autharena.py --messages messages.json
```

`--messages` accepts a JSON list, or an object containing a `messages` list.
Use `--messages -` to read JSON from stdin:

```json
[
  {"role": "system", "content": "Translate into English."},
  {"role": "user", "content": "Bonjour."}
]
```

System instructions and supplied message history are combined into one website
prompt. This route does not expose separate system-role, sampling, or maximum
output-token controls.

## Models and live output

```powershell
python src/autharena.py --list-models
python src/autharena.py --list-models --json
python src/autharena.py --prompt-file prompt.txt --timeout 300 --json
python src/autharena.py --prompt "Hello" --quiet > answer.txt
```

Model listing reads Arena's public catalog without opening a browser or starting
a chat. Selecting an Arena model triggers automatic polling when its 24-hour
catalog cache expires. The Model Manager's **Poll** button refreshes the selected
Arena catalog immediately. Catalog membership does not guarantee current access
for your account.

Arena always receives responses as a stream. The main log and Direct Text show
individual text fragments as they arrive, preserving word boundaries and line
breaks. Reasoning is displayed separately from answer text.

In **Other Settings**, **Stream thinking/reasoning logs** controls visible
reasoning, and **Allow forced-stream batch log** controls Arena's live output
during batch translation. Batch stream logs are off by default. The general
**Enable streaming responses** setting does not disable Arena's required stream.
Direct Text enables its live output for the active run; its skip-thinking control
can hide reasoning. The environment overrides `LOG_STREAM_CHUNKS` and
`AUTHARENA_LOG_STREAM_CHUNKS` can hide live output outside batch mode.

The standalone script prints the final answer to stdout, or a result object with
`--json`. Progress and live fragments go to stderr; `--quiet` suppresses them.
The default model is `gpt-6-astra-medium`, account number `0`, and timeout
`180` seconds. Use `python src/autharena.py --help` for all options.

## Troubleshooting

If the connection page cannot find the companion, check that the generated
extension folder is loaded in that exact browser profile, then refresh the page.
Leave Glossarion running while connecting. If pairing a new account number fails
because the browser profile is already bound, open the connection page in a
different browser profile instead of another tab in the same profile.

Complete Arena's sign-in, terms, or browser challenge in the normal Arena tab.
If an operation times out, allow more time with `--timeout` and check that the
browser profile is still connected. Closing the companion's active Arena tab
cancels its request. Restarting the companion does not remove Arena's limits.

Arena can change its website request format. Local automated tests do not verify
that a particular signed-in account and model can complete a live request; try a
short prompt before a long job.
