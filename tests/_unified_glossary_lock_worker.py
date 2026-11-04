"""Child-process side of tests/test_unified_glossary_lock.py.

Started with ``multiprocessing`` (spawn): each process plays one Glossarion instance that merges
its own book into the shared unified glossary many times. Lives in its own module (not the test
file, not stdin) so the spawned interpreter can import it.
"""

import os
import sys

KEY = "test-english"
#: The dedup settings the merges run with: exact raw-name matches only, so the only way an entry
#: can disappear is a lost update.
MERGE_ENV = {
    "GLOSSARY_FUZZY_THRESHOLD": "1.0",
    "GLOSSARY_DEDUPE_TRANSLATIONS": "0",
    "GLOSSARY_DUPLICATE_ALGORITHM": "basic",
    "UNIFIED_GLOSSARY_EXCLUDE_GENDER_ENTRIES": "1",
    "UNIFIED_GLOSSARY_SUBPROCESS_MIN_ENTRIES": "1000000",
    "GLOSSARY_OUTPUT_LEGACY_JSON": "0",
    "GLOSSARION_HTTP_LOG": "0",
}


def raw_name(book, iteration, index):
    return f"{book}{iteration:03d}n{index:02d}"


def book_entries(book, iteration, per_iteration):
    return [
        {"type": "term", "raw_name": raw_name(book, iteration, index),
         "translated_name": f"Tr {book} {iteration:03d} {index:02d}"}
        for index in range(per_iteration)
    ]


def merge_many(src_dir, shared_dir, book, iterations, per_iteration, start, use_lock=True):
    """Merge ``iterations`` batches of ``book`` into ``shared_dir``'s unified glossary.

    ``start`` (a multiprocessing Barrier) lines the processes up after their slow imports so the
    merges really overlap. ``use_lock=False`` is the control experiment that shows the race.
    """
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    os.environ.update(MERGE_ENV)
    os.environ.pop("TRANSLATION_CANCELLED", None)
    os.chdir(os.path.dirname(os.path.abspath(shared_dir)))
    import contextlib

    import unified_glossary as ug

    if not use_lock:
        ug.unified_lock = lambda *_args, **_kwargs: contextlib.nullcontext()
    ug._extractor()  # the heavy glossary machinery, before the start line
    start.wait(timeout=600)
    for iteration in range(iterations):
        ug.merge_book(shared_dir, KEY, book_entries(book, iteration, per_iteration),
                      log=lambda *_args: None)


if __name__ == "__main__":  # pragma: no cover - manual control experiment
    import multiprocessing
    import tempfile

    use_lock = "--no-lock" not in sys.argv
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    shared = os.path.join(tempfile.mkdtemp(prefix="ug_lock_"), "Glossary")
    os.makedirs(shared)
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    procs = [ctx.Process(target=merge_many, args=(src, shared, book, 25, 4, barrier, use_lock))
             for book in ("A", "B")]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    csv_path = os.path.join(shared, "Unified Glossary", KEY, "glossary_unified.csv")
    with open(csv_path, encoding="utf-8") as handle:
        text = handle.read()
    expected = [raw_name(b, i, j) for b in ("A", "B") for i in range(25) for j in range(4)]
    missing = [name for name in expected if name not in text]
    print(f"use_lock={use_lock}: {len(expected) - len(missing)}/{len(expected)} entries kept, "
          f"{len(missing)} lost")
