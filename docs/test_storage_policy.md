# Tests included in Git; Drive copies are backups

The user's decision on 2026-10-14 supersedes the earlier external-tests-only
policy. All previously tracked, reviewed-safe tests and fixtures remain in the
source repository, including the new OAuth regression coverage. The temporary
test-exclusion rules and index removals were reversed. Preexisting ignore rules
are preserved, so this does not force-add previously ignored local experiments,
generated output, or arbitrary credentials.

GitHub CI runs the included suites normally. No missing-archive skip guards
remain. GitLab alternatives must also fail if a required suite is missing.
The separate Google Drive copies remain backups, not a prerequisite for normal
test execution from a complete checkout.

Some desktop-parity tests use `git show` with fixed historical commit IDs.
Fresh migrations use the reviewed [bounded oracle package](parity_oracles.md)
to supply only the pinned source data in an isolated temporary object store.
Test files remain included, comparisons still run, and missing/corrupt oracles
fail. Original repository history and the all-refs bundle remain local and
unchanged; no full-history import is needed for these oracle reads.
