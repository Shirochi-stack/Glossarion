# AuthGem OAuth provenance and repository migration

Reviewed on 2026-10-14. This review concerns the default client pair in
`src/authgem_auth.py`, not user tokens or environment overrides.

## Finding and evidence

The embedded `_GEMINI_CLI_CLIENT_SECRET` is Google's publicly distributed
installed/native-app client value. It is named `client_secret` by the OAuth
protocol, but it is not a confidential web-server secret, a service-account
key, an API key, or a user's access/refresh token. Both default constants were
compared locally, without displaying them, with the corresponding
`OAUTH_CLIENT_ID` and `OAUTH_CLIENT_SECRET` declarations in Google's
[Gemini CLI source at commit 42c51822084ac25cd5d97ac0023822c5235f88c8](https://github.com/google-gemini/gemini-cli/blob/42c51822084ac25cd5d97ac0023822c5235f88c8/packages/core/src/code_assist/oauth2.ts).
Both match exactly; upstream explicitly documents this as installed-app
material that is not treated as secret.

[Google's installed-app OAuth overview](https://developers.google.com/identity/protocols/oauth2#installed)
explains why this client type can embed the shared pair in distributed code.
[RFC 8252 sections 8.4 and 8.5](https://www.rfc-editor.org/rfc/rfc8252.html#section-8.5)
classify such native apps as public clients; an embedded shared value cannot
authenticate a particular installation. User authorization is still required.
This conclusion relies on the exact upstream match and the documented client
type, not merely on the value appearing in a public repository. Google's
private client-registration console was not accessed.

SHA-256 fingerprints for a narrowly scoped recheck (values intentionally omitted):

| Python constant | Reviewed SHA-256 |
| --- | --- |
| `_GEMINI_CLI_CLIENT_ID` | `636d7183f3ca4ad92c7803811a3f635643f1da657a22eeedf280283161cb6e17` |
| `_GEMINI_CLI_CLIENT_SECRET` | `6a5f78b8b99dd4025e41ba11bf54c304c6af29f924c5569cc7865b2428ce03a9` |

All 90 objects listed by the earlier historical OAuth finding report were
read from the local Git object database. Every client ID and client-secret
constant matched this reviewed pair. This resolves those 90 findings; it is
not a certification of unrelated history, ignored files, generated builds,
local settings, or token stores.

## Source and functional compatibility

Keep the existing default constants, source module, local tests, and Git history.
No credential extraction, rotation, replacement, environment configuration,
or reauthentication is needed for this finding. Removing the default or
substituting another client could break sign-in or refresh of tokens bound
to the existing client. No runtime code or credential value was changed.

`AUTHGEM_CLIENT_ID` and `AUTHGEM_CLIENT_SECRET` keep their existing import-time,
independent override behavior, including explicit empty values. An override
is not covered by the public-default finding; privately registered secrets
and all user tokens must stay local. No credentials were created, entered,
configured, refreshed, or sent to Google during this review.

Offline regression coverage lives in `tests/test_mobile_auth_splits.py`,
which is included in Git under the [test-storage policy](test_storage_policy.md).
GitHub CI runs this suite normally. It checks the exact public
pair by digest, consent-URL fields, exchange and refresh payloads, expiry,
custom/empty overrides, and token-error propagation. HTTP is faked and new
tests neither launch a browser nor instantiate a token store. An intentional
change to either default requires a new provenance review before changing
the pinned digest. The tests do not contact upstream or perform live login.

The existing flow's absence of PKCE is unchanged. Public-client provenance
does not establish the security of the entire OAuth flow; PKCE modernization
would be a separate change requiring its own compatibility testing.

## Upload and history handling

For a scanner flagging this exact default, record an evidence-backed false
positive using the source path, constant name, and fingerprints above. Do
not exempt the whole module or all OAuth secret patterns, obscure the value,
or suppress unrelated findings. No scanner bypass or suppression was added
by this review. The complete `authgem_auth.py` can remain part of the source
set. This report does not itself authorize an upload or assert that a remote
scanner has accepted the classification.

The existing all-refs bundle and other original backups remain local and
unchanged. This public pair alone does not warrant rewriting history or
starting a new repository. Some desktop-parity tests use historical commit
IDs, so a fresh history or rewrite can also break validation.

If a separate genuine secret is found, stop transfer of the affected
artifact and keep the complete original locally. The user can have its owner
revoke/rotate it through the provider and configure any replacement privately;
those are separate actions requiring action-time approval. Source removal
alone does not remove historical copies. After preserving the original,
options requiring a separate explicit decision are a reviewed rewrite in an
isolated copy, or a clearly labeled source snapshot without history. Neither
option silently replaces the original, and a snapshot is not a full-history
migration. Do not upload actual secrets merely to satisfy file completeness.
