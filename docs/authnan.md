# NanoGPT subscription accounts (`authnan/`)

In the desktop model field, select `authnan/` and click **NanoGPT Login**. Approve
`api.use models.read` in your browser, return to Glossarion, and poll the model
catalog to choose a model. Leave the API key field blank. Login works with the
default route, a numbered account, or an enabled key pool containing these routes.

| Route | Account |
| --- | --- |
| `authnan/model` | Default saved account |
| `authnanN/model` | Saved slot N, from 1 through 9999 |
| `authnan0/model` | Round-robin rotation through signed-in default and numbered accounts |

Prefixes are case-insensitive. With `authnan0/`, the account selector manages saved
slots; **+ N** opens a fresh signed-out browser window for a new slot, so the website
and Google login cannot silently reuse your previous browser account. This window
uses a temporary profile and closes when login finishes, is cancelled, or times out.
Chrome, Edge, Chromium, or Firefox must be installed; `AUTHNAN_BROWSER_BINARY` can
select their executable. Normal login still uses your default browser. NanoGPT's
regular logout page requires confirmation, so + N uses a separate empty session.
An account already saved in another slot is rejected when adding a new account.
The selected slot is captured when login
starts, so switching models or slots during approval does not move the saved key.
Click the signed-in status button to log out of the displayed slot. Logout removes
the local credential; revoke its key separately in NanoGPT API settings if needed.

## Requests and billing

Chat always uses
`https://nano-gpt.com/api/subscription/v1/chat/completions`, including SDK, HTTP,
compatibility repairs, and retries. There is no automatic fallback to the standard
chat endpoint. Text suggestions come from
`https://nano-gpt.com/api/subscription/v1/models?detailed=true` for the saved account.
Catalog polling never starts browser login.

Images use `https://nano-gpt.com/api/v1/images/generations`; video submission and
polling use `/api/generate-video` and `/api/generate-video/status`. Their suggestions
come from the separate `/api/v1/image-models` and `/api/v1/video-models` catalogs.
Thinking, streaming, and supported OpenAI/Gemini service tiers follow the existing
NanoGPT transport rules. Generic custom endpoints do not redirect `authnan/` keys.

**NanoGPT account and API-key settings govern paid overage and extras.** The
subscription URL does not guarantee that every request is free. Explicit provider
selection always uses pay-as-you-go balance and bypasses subscription coverage.
Images have separate coverage and allowances; video and other extras may be billed
separately. Check your NanoGPT settings before using overage, providers, or media.
See [NanoGPT support and billing guidance](https://nano-gpt.com/support) and the
[chat endpoint documentation](https://docs.nano-gpt.com/api-reference/endpoint/chat-completion).

The existing `nan/` API-key route and default model are unchanged. Browser login UI
is desktop-only; Android, the extension, and Gradio login UI are outside this feature.

## Saved credentials and recovery

NanoGPT's [S256 PKCE browser handoff](https://nano-gpt.com/blog/sign-in-with-nanogpt-oauth-pkce)
uses an available loopback port, random state, `/auth`, and code exchange at
`/api/v1/auth/keys`. The callback listener closes after approval, cancellation, or
five minutes. The returned key, identity, scope, and expiry metadata use Glossarion's
existing encrypted credential storage; no plaintext fallback is written.

Files are `authnan_tokens.json` and `authnan_tokens_N.json` in the shared token
directory (`GLOSSARION_TOKEN_DIR`, otherwise `~/.glossarion`). `AUTHNAN_TOKEN_FILE`
overrides only the default slot. NanoGPT issues an API key rather than a refresh
token. An expired or revoked key needs another browser approval.

Default and pinned requests can initiate login when missing credentials. A confirmed
HTTP 401 invalidates only the rejected credential and permits one reauthentication.
Concurrent requests reuse a newer replacement key when one is already saved.
Rotation deduplicates known NanoGPT user IDs, skips missing/expired credentials,
and advances to the next account on 401 or 429 without opening login. Retry timing
is retained on errors; fixed-account retries honor the provider's Retry-After.
An empty pool asks you to sign in first. Once a video job is submitted, status
polling keeps its submitting account and never opens login or submits a new job.
