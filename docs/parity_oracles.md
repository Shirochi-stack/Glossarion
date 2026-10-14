# Parity oracles in fresh migrations

The included tests compare current code with specific old sources. A fresh
repository does not contain those commits. `tests/parity/oracle_objects.zip`
provides a bounded, reviewed fixture instead of fetching all old history.

The pinned manifest contains 13 commit objects, their root/source lookup trees,
and 251 distinct Python source blobs. These cover 107 candidate source paths at
the pinned revisions (1,131 revision/path entries). The source-path set is a
conservative union of literal module/file references in the existing oracle
consumers, intersected with those revisions. It includes shared freezer helpers
whose availability changes across revisions. No ancestor walk, branches, tags,
remote configuration, other historical file contents, or account credentials
are included. Exact object IDs, types, sizes, SHA-256 hashes, aliases, consumers,
and source paths are recorded in `oracle_manifest.json`.

The archive is approximately 17.8 MB; its objects expand to approximately
88 MB. Commit/tree metadata retains original hashes and therefore references
parents and other objects that are deliberately absent. This is a source-read
fixture, not a complete repository or a history-restoration bundle. Do not
import it into `.git/objects`, add refs to it, or use it as a clone/backup.

## Use

From a complete checkout, with the normal test dependencies installed:

```sh
python tests/parity/oracle_package.py verify
python tests/parity/oracle_package.py run -- python -m pytest tests/test_mobile_auth_splits.py
```

Replace the pytest selection with the suite being validated. The wrapper sets
`CI=true` and passes the command's exit status through. Existing test assertions
and their `git show` calls remain intact. GitLab desktop/mobile test jobs prepare
the same fixture automatically before their historical-oracle preflight.
GitHub workflows remain unchanged; their original history continues to work.

The wrapper creates a temporary object directory and sets `GIT_OBJECT_DIRECTORY`
for the child process, using this checkout's actual object store as an alternate.
It does not write into the checkout's object store, change HEAD or refs, add
remotes, fetch, or attach the original repository. The temporary data is cleaned
up after the command; GitLab jobs keep equivalent scratch data only inside their
disposable workspace. Missing, corrupt, unexpected or oversized package members
fail. The archive digest, each SHA-256, and each Git object identity are checked.

## Review and reproduction

The selected source objects and commit metadata were scanned locally for strong
credential formats and literal credential assignments. All 47 literal matches
were resolved: the public native OAuth client already documented in
`authgem_oauth_migration.md`, reviewed current-source literals, and two provider
sentinels moved between modules. No unresolved candidate or strong-format hit
remains. This is a scoped review, not a claim that regex scanning can detect
every arbitrary secret. Candidate values are omitted from the review metadata.

To reproduce the exact object selection from the retained original repository:

```sh
python tests/parity/oracle_package.py export --output /path/to/new/oracle_objects.zip
```

Export reads only manifest-listed object IDs. It refuses to overwrite a file or
accept missing/changed objects. ZIP member order, timestamps, permissions and
compression settings are fixed. Exact ZIP reproduction was checked with the
current Python 3.12/zlib toolchain; a different compressor that changes bytes is
rejected rather than silently updating the trusted digest. The original local
history and all-refs bundle remain the restoration sources.

When a test adds another historical reference, update and review the bounded
manifest/package explicitly, including its integrity pin and regressions. Do
not fall back to an unreviewed remote, silently skip the comparison, or broaden
the package to all ancestors.
