# Glossarion Arena Browser Companion

Click **Arena Login** in Glossarion, then **Install in browser**. On Windows, setup opens an Extensions tab in that same Chrome/Edge window and profile, then attempts the normal Developer mode and Load unpacked steps. Keep Arena Login in front until Extensions opens, then keep that window in front. The setup page shows progress and any remaining manual step. **Copy folder path**, **Open folder**, and **Connect helper** are also available there.

The companion folder is generated at `~/.glossarion/autharena_extension`, including when Glossarion runs as a one-file `.exe`. Keep this folder in place: the browser loads the extension from there after app restarts. It includes `arena_page.js`; the repository template folder by itself is incomplete. Other Chromium browsers can load the generated folder manually.

Click **Arena Login** in Glossarion. Its local connection page pairs this browser profile, then the companion opens Arena's normal login dialog. Existing Arena sign-in stays in this browser. Choose Google or email in Arena and complete any normal verification or first-use terms.

The companion creates its own Arena tab. It keeps your other tabs and existing chats intact. Each translation uses a fresh conversation and sends stream chunks back to Glossarion as they arrive. Closing its active tab cancels that request.

Each browser profile can pair with one numbered account slot. To use another slot, install and pair the companion in a different browser profile. Two tabs in one profile share the same Arena login. If pairing fails, open a new Arena Login link from Glossarion; the old link cannot be reused.

The extension uses only Arena and the local Glossarion broker at `127.0.0.1:18874`. It does not request cookies or debugger permissions. Login credentials stay in the browser; the pairing token stays in extension storage. If Glossarion disconnects, active work stops without replaying a potentially submitted request. The companion reconnects when Glossarion becomes available again.
