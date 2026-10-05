"""Desktop parity oracle (local safety net; gitignored).

Modules:
    freeze_legacy   - freeze the methods scheduled to move (U1-U3) from ``git show <sha>``
    fakes           - fake widgets, FakeState and recorders used to drive frozen/new owners
    normalize       - sandbox, deterministic process state and env capture
    scenarios       - the 12 golden scenarios (config dicts + widget values + run attrs)
    capture_golden  - ``capture(owner_factory, scenario)`` and the golden writer/loader

Reusable tiers for every extraction step (U2+), wired by test_parity_tiers.py:
    moved_functions - registry of moved methods/attrs, shared modules, mixins, hooks
    fuzz_moved      - tier D: differential fuzz, legacy oracle vs working-tree desktop
    roundtrip       - tier R: live desktop -> collected settings -> rebuilt owner
    owner_contract  - unguarded ``self.<attr>`` reads of the shared mixins (AST)
    import_hygiene  - tier I: PySide6-blocked subprocess imports + Python 3.10 parse

Run from the repository root::

    python tests/parity/freeze_legacy.py            # freeze at HEAD
    python tests/parity/capture_golden.py           # write goldens for the frozen SHA
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests/parity/test_legacy_self_consistency.py tests/parity/test_parity_tiers.py
"""
