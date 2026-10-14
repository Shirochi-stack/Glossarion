# Antigravity and OcAgy credential storage

Glossarion adapts the JavaScript account stores to the `GLSE1` encrypted format
used by `token_encryption.py`. The `proxy_token_storage.py` adapter is included
in every desktop PyInstaller specification; generated JavaScript stays in the
downloaded runtime directory.

- Windows uses the current user's DPAPI protection. JavaScript calls Windows
  PowerShell's built-in ProtectedData API with secret bytes over stdin, never
  as command-line arguments. No system Python is required by the runtime.
- macOS uses the shared token-encryption key in Keychain and Fernet encryption.
- Linux uses the shared key file with owner-only permissions and Fernet.
  As with the existing token store, access to both that key and the encrypted
  files permits decryption. Encryption does not isolate credentials from
  malicious software already running as the same user.

Account files retain their existing names but contain encrypted data. The
runtime decrypts only into memory. Account selection, token refresh, and
rotation metadata are preserved. New writes use encrypted temporary files
and atomic replacement. Encryption failures do not fall back to plain JSON.

Antigravity applies its storage adapter before migrating accounts during
managed proxy startup. OcAgy patches its installed OAuth plugin before use
and migrates its account file. Numbered OcAgy requests also receive encrypted
isolated account files. OpenCode's Google OAuth refresh field is encrypted
separately; other provider entries are left unchanged. Persisted Google access
tokens are cleared and regenerated in memory when needed. Login callbacks
likewise return no access token for OpenCode to persist.

Restart Glossarion and its managed proxy/OpenCode processes after upgrading.
An already-running process still has its previous storage implementation.
An independently updated OpenCode plugin may replace the adapter; launch
OcAgy through Glossarion again before using migrated accounts. Unsupported
upstream storage layouts stop setup with a compatibility error.

Validation covers Python/JavaScript round trips on Windows, interoperability
with the Unix Fernet format, encrypted writes through copies of both installed
upstream storage modules, OAuth refresh serialization, migration, failed-write
preservation, isolated account routing, and packaged module inclusion.
Native macOS/Keychain and Linux execution have not been tested on those systems.
