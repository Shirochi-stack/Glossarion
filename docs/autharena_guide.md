# AuthArena user guide

`src/autharena.py` sends requests through Arena's website Direct chat flow using
an installed Chrome or Edge browser. In Glossarion, select a model with the `autharena/` prefix,
for example `autharena/gpt-6-astra-medium`. Model availability comes from Arena's
current website catalog; a saved model name may later become unavailable.

This route uses your Arena website login, without an API key. It is separate
from Arena's documented API gateway and its API keys. Each request starts a new
conversation, so previous website chats are not added to the request.

## Requirements and login

Install Chrome or Edge and the Glossarion Python dependencies, including
`websocket-client`, or use the packaged desktop app. Arena opens in an external
browser window with a separate profile for each account slot. Qt WebEngine is
not required for this provider, including in Lite and TurboLite builds.
On Windows the default Chrome-compatible browser is preferred, with installed
Chrome or Edge as fallbacks. To choose another installation, set
`AUTHARENA_BROWSER` to its executable path. Each account has a separate browser
profile, so sign in once there even if Arena is already open in your usual profile.

In the main window, select an Arena model and click **Arena Default Login**
(or **Arena #N Login** for a numbered account). Use the adjacent account selector
and **+ New** to add another physical account slot. In the
**Multi API Key Manager**, a **Login** button appears inside every Arena model
field, including the model editor shared by translation, fallback and dedicated
key pools. Type a numbered prefix to sign into that account slot. These routes
do not require an API key in the manager.

Run these commands from the repository root:

```powershell
python src/autharena.py --login
python src/autharena.py --model gpt-6-astra-medium --prompt "Reply with exactly: ok"
```

Direct chat requires Arena sign-in and first-use consent. Complete these and
any browser challenge interactively in the external browser. Login is recorded
only after the website confirms the signed-in account and consent. Requests use
the website's streaming flow and may require reCAPTCHA or a renewed login;
the adapter does not bypass those checks. Arena's own limits still apply.
The login window closes automatically once verification succeeds.

Account state persists locally under `~/.glossarion/autharena_browser/<account-id>`;
the external browser stores its cookies and profile in that directory's `chromium` subfolder.
Normal application shutdown retains them. Use the same `--account-id` for login
and requests when using a separate profile:

```powershell
python src/autharena.py --account-id 1 --login
python src/autharena.py --account-id 1 --prompt "Hello"
```

| Model prefix | Account selection |
| --- | --- |
| `autharena/` | Default physical account slot `0`. |
| `autharena1/`, `autharena2/`, … | The corresponding numbered account slot. |
| `autharena0/` | Round-robin rotation across all successfully signed-in accounts, including the default slot. |

`autharena1/gpt-6-astra-medium` selects the same account profile as
`--account-id 1`. Each call still starts a new conversation. For `autharena0/`,
the manager's **Login** button asks which physical slot to sign in: the default,
an existing or configured slot, or a new numbered slot. Sign into each intended
account once before using rotation. A folder or saved cookies alone do not
qualify an account for the pool; a verified login does. Sessions can expire and
may need another login.
Pool requests run without opening login windows. Explicit sign-in or rate-limit
rejections can advance to the next account; an uncertain submitted request is
reported as an error to avoid duplicating it.

## Prompts and messages

Requests using the same account run one at a time to protect its browser
profile. The timeout includes time spent waiting for that profile, signing in,
and generating the response.

```powershell
python src/autharena.py --prompt-file prompt.txt
python src/autharena.py --system "Translate into English." --prompt-file chapter.txt
python src/autharena.py --messages messages.json
```

`--messages` reads a JSON list of messages, or an object containing a `messages`
list. Pass `--messages -` to read that JSON from stdin. For example:

```json
[
  {"role": "system", "content": "Translate into English."},
  {"role": "user", "content": "Bonjour."}
]
```

System instructions and message history are flattened into one website prompt.
The website route does not expose separate system-role, sampling, or maximum
output-token controls. Supplying history does not resume an existing Arena chat.

## Model discovery and output

```powershell
python src/autharena.py --list-models
python src/autharena.py --list-models --json
python src/autharena.py --prompt-file prompt.txt --timeout 300 --json
python src/autharena.py --prompt "Hello" --quiet > answer.txt
```

Model listing reads Arena's website catalog without starting a chat. Glossarion's
AuthArena model polling uses that catalog to discover `autharena/` model IDs.
Select or type an `autharena/` model to trigger automatic catalog polling when
its 24-hour cache expires. Use the Model Manager's **Poll** button to refresh
the selected Arena catalog immediately.
Listing a model does not guarantee that your account can use it at that moment.

Normal request output prints the answer to stdout; `--json` prints the result
object. Progress goes to stderr, and `--quiet` suppresses progress logs. The
default model is `gpt-6-astra-medium`, the default account ID is `0`, and the
default timeout is `180` seconds. Run `python src/autharena.py --help` for the
complete command options.

## Troubleshooting

If the browser asks you to sign in or solve a challenge, complete it in the
external browser window. If it times out, allow more time with `--timeout` and
check that Arena loads normally. A rate-limit response requires waiting before
retrying; restarting the helper does not remove the website's limits.

The adapter depends on Arena's website request format, which can change. Local
automated tests cannot establish that a real account, CAPTCHA, or model will
complete a live request; verify a short prompt before starting a long job.
