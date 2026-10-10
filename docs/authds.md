# DeepSeek web login (`authds/`)

Select `authds/flash`, `authds/pro`, `authds/flash-thinking`, or
`authds/pro-thinking` in the desktop model picker. The **Enable DeepSeek and
Chutes Thinking** toggle in Other Settings also controls DeepThink for every
`authds/` model, including the `-thinking` aliases. Effort and Responses API
format settings apply to API routes; web chat uses the DeepThink on/off setting.
Click **DeepSeek Login** and
sign in on DeepSeek's own website; choose its Google login option if preferred.
The first translation also opens login automatically when needed. Leave the API
key blank. Flash and Pro select the website's default and expert modes, rather
than promising a particular model version.

Chrome or Edge must be installed. Glossarion uses a separate profile under
`~/.glossarion/authds_browser`; it does not read your everyday browser profile,
store your Google password, or export the session token. Keep this profile private.
Once signed in, translation uses that profile in a background browser. Expired
sessions trigger one interactive re-login. Each chunk gets a new chat, and requests
sharing this profile run sequentially. Stop cancels queued work and active requests.

This route uses DeepSeek's web-chat service and its account limits. Temperature,
sampling parameters, and requested output-token limits are not sent; the website
controls these. Thinking text is excluded from translated output. Incomplete
streams fail rather than saving a partial chapter. Translation conversations
appear in the DeepSeek account's chat history.

The route currently requires desktop Chrome/Edge. Mobile shows an explicit
unsupported-route message; use the existing `deepseek/` API-key route there.

Advanced settings through environment variables:

- `AUTHDS_BROWSER_BINARY`: full path to Chrome/Edge if automatic detection fails.
- `AUTHDS_PROFILE_DIR`: dedicated browser profile directory.
- `AUTHDS_POW_WORKER_URL`: override the site's proof worker URL if a site update
  changes the asset or worker discovery fails. Normally the route observes the
  site's worker URL, with a known asset as a fallback.

The web protocol is unofficial and can change. Local tests cover routing, stream
parsing, cancellation, expired-session recovery, and the mobile process guard.
A successful Google login and translation require live verification with a
DeepSeek account; mocked tests do not establish that.
