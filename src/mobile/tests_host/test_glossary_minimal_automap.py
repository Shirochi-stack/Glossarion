"""Owner issues 7 and 8 (glossary on mobile): "Auto mapping doesn't seem to work in mobile" and
"Minimal auto glossary seems to get stuck on mobile".

Issue 8, what the owner saw: the log of a Minimal automatic glossary over 434,141 characters of Korean
(12,710 sentences) ended at "Using ThreadPoolExecutor for sentence processing (dataset < 5000 sentences)".
Diagnosis (host, GLOSSARION_MOBILE=1 GLOSSARION_NO_PROCESSES=1, 434,141 chars of a real Korean book,
16,025 sentences): ``GlossaryManager._filter_text_for_glossary`` was not dead-locked but drowning in its
own stop checks. ``_check_filter_stop`` (``is_stop_requested``: three env reads, a stop-file ``stat`` and
the ``unified_api_client`` flags, ~10 us) ran inside the innermost loops: once per honorific (656 of them
for Korean) for every candidate term, and once per (sentence x frequent term) in the sentence-coverage
scan - 91 million checks for that book, ~28-42 sentences/s in step 5 and a step 7 of ~87 million checks.
Desktop hid step 5 behind its process pool (12 processes); mobile runs it on threads under the GIL.
Fixed: the stop is checked per sentence / per candidate term, never per honorific or per term inside a
sentence; a large dataset without process pools runs the desktop's large-dataset batch worker on
threads and merges batches in batch order (as ``map_async`` does), so mobile extracts what the desktop
extracts; the log names the real reason for threads and reports every batch.

Issue 7, what "auto mapping" is: the desktop "Auto-Mapping (Auto-Fill)" switch
(``append_glossary_auto_load``; on in the Off / Off (Fuzzy Mapping) / Balanced / Full / Single Pass modes,
fuzzy name matching with ``fuzzy_auto_mapping``): at run start ``_sync_automapped_glossaries_to_output``
picks ``Glossary/<book>/<book>_glossary.*`` for the input (``_guess_glossary_for_input_file``) and
appends it. The guess searches ``<OUTPUT_DIRECTORY>/Glossary``, ``<owner.base_dir>/Glossary`` and the
app folder's ``Glossary``. A chat send (``direct_text`` job, the mobile translate path) points
``OUTPUT_DIRECTORY`` at its run root, like the desktop Input/Output dialog; the desktop still finds its
shared glossaries through ``base_dir`` (the app folder), but on mobile they live under the platform
output root (``<Library>/Glossary/<book>/``), which no searched folder covered: nothing was ever mapped.
Fixed: the direct_text job gives its owner that root as ``base_dir`` before the run environment.

Run from src/mobile with the mobile venv:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_glossary_minimal_automap.py
"""

from __future__ import annotations

import os
import random
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for _entry in (str(APP_DIR), str(SRC_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


@pytest.fixture
def mobile_env(tmp_path, monkeypatch):
    """Mobile process gates + every user path in tmp; os.environ restored as a whole afterwards
    (HeadlessOwner's init writes ~40 variables)."""
    saved = dict(os.environ)
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "TMP", "TEMP", "TMPDIR"):
        folder = tmp_path / "isolated" / name.lower()
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    monkeypatch.setenv("GLOSSARION_DATA_DIR", str(data))
    monkeypatch.setenv("GLOSSARION_APP_DIR", str(data))
    monkeypatch.setenv("CONFIG_FILE", str(data / "config.json"))
    monkeypatch.setenv("GLOSSARY_STOP_FILE", str(tmp_path / "glossary.stop"))
    for flag in ("TRANSLATION_CANCELLED", "GRACEFUL_STOP", "GRACEFUL_STOP_COMPLETED", "MANUAL_GLOSSARY",
                 "GLOSSARY_INCLUDE_GENDER_CONTEXT", "GLOSSARY_INCLUDE_ALL_CHARACTERS", "GLOSSARY_MAX_SENTENCES"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    monkeypatch.undo()
    os.environ.clear()
    os.environ.update(saved)


# ==========================================================================
# Issue 8: Minimal automatic glossary filter
# ==========================================================================

def _korean_text(sentences=6000, seed=7):
    """Deterministic Korean prose: recurring 3-syllable names, honorific forms, filler words."""
    rng = random.Random(seed)
    first, mid, last = "김이박최정강조윤장임한오", "서민지현수영하도준우예은", "연호우진아빈희율원린경석"
    names = [a + b + c for a in first for b in mid for c in last]
    rng.shuffle(names)
    names = names[:400]
    places = ["마탑", "황궁", "제국", "왕국", "던전", "검술관", "학원", "신전"]
    verbs = ["말했다", "웃었다", "대답했다", "고개를 끄덕였다", "검을 뽑았다", "한숨을 쉬었다"]
    out = []
    for _ in range(sentences):
        a, b = rng.choice(names), rng.choice(names)
        out.append(f"{a}{rng.choice(['님은', '은', '이', ''])} {rng.choice(places)}에서 {b}에게 "
                   f"{rng.choice(verbs)}")
    return ". ".join(out) + "."


def _filter(GM, text):
    return GM._filter_text_for_glossary(text, min_frequency=2, max_sentences=40)


def test_stop_checks_are_per_sentence_not_per_honorific_or_term(mobile_env, monkeypatch):
    import GlossaryManager as GM

    monkeypatch.setenv("EXTRACTION_WORKERS", "4")
    calls = [0]
    lock = threading.Lock()

    def counting_check():
        with lock:
            calls[0] += 1

    monkeypatch.setattr(GM, "_check_filter_stop", counting_check)
    text = _korean_text()
    result, _ = _filter(GM, text)
    assert result
    sentences = 6001  # the split leaves one trailing empty piece
    # Before the fix: one check per honorific (656 for Korean) per candidate term in step 5 and one per
    # (sentence x frequent term) in step 7, millions here. Now: per sentence / term / batch / log line.
    assert calls[0] < 25 * sentences, calls[0]


def test_stop_is_still_honoured_inside_a_batch(mobile_env, monkeypatch):
    """The checks that stay still stop a running extraction batch (desktop graceful stop semantics)."""
    import GlossaryManager as GM

    def sentences():
        yield "김서연님은 마탑에서 이민호에게 말했다"
        monkeypatch.setenv("GRACEFUL_STOP", "1")
        yield "이민호는 황궁에서 김서연에게 웃었다"

    with pytest.raises(GM._GlossaryFilteringStopped):
        GM._process_sentence_batch_for_extraction(
            (sentences(), 0, r"[가-힣]{2,4}", (list(GM.PM.CJK_HONORIFICS["korean"]), [], set(), set())))


def test_large_dataset_without_processes_runs_the_desktop_batch_worker(mobile_env, monkeypatch, capsys):
    """> 5000 sentences: desktop's process pool runs ``_process_sentence_batch_for_extraction``; without
    processes the same worker runs on threads (it used to switch to the small-dataset local worker,
    whose sentence de-duplication differs), and the log says why threads are used."""
    import GlossaryManager as GM

    monkeypatch.setenv("EXTRACTION_WORKERS", "4")
    seen = []
    real = GM._process_sentence_batch_for_extraction

    def spy(args):
        seen.append(args[1])
        return real(args)

    monkeypatch.setattr(GM, "_process_sentence_batch_for_extraction", spy)
    _filter(GM, _korean_text())
    out = capsys.readouterr().out
    assert "Using ThreadPoolExecutor for sentence processing (process pools unavailable on this platform)" in out
    assert "dataset < 5000" not in out
    batches = int(out.split("📑 Processing ")[1].split(" batches")[0])
    assert sorted(seen) == list(range(batches))
    # every batch is reported (the owner's log showed nothing after the executor line)
    for done in range(1, batches + 1):
        assert f"| Batch {done}/{batches} |" in out


def test_small_dataset_keeps_the_thread_worker_and_message(mobile_env, monkeypatch, capsys):
    import GlossaryManager as GM

    monkeypatch.setenv("EXTRACTION_WORKERS", "4")
    monkeypatch.setattr(GM, "_process_sentence_batch_for_extraction",
                        lambda args: pytest.fail("small datasets use the local thread worker"))
    _filter(GM, _korean_text(sentences=2000))
    assert "Using ThreadPoolExecutor for sentence processing (dataset < 5000 sentences)" in capsys.readouterr().out


@pytest.mark.parametrize("sentences", [2000, 6000])
def test_result_does_not_depend_on_thread_completion_order(mobile_env, monkeypatch, sentences):
    """Batches merge in batch order (the process pool's ``map_async`` order), so the same input gives
    the same filtered text however the threads finish."""
    import GlossaryManager as GM
    from concurrent.futures import wait

    monkeypatch.setenv("EXTRACTION_WORKERS", "4")
    text = _korean_text(sentences=sentences)

    def completed_in(order):
        def completed(futures):
            futures = list(futures)
            wait(futures)
            yield from order(futures)
        return completed

    monkeypatch.setattr(GM, "_filter_completed", completed_in(lambda fs: fs))
    forward = _filter(GM, text)
    monkeypatch.setattr(GM, "_filter_completed", completed_in(lambda fs: list(reversed(fs))))
    backward = _filter(GM, text)
    assert forward[0] and forward == backward


# ==========================================================================
# Issue 7: auto-mapping in a chat (direct_text) run
# ==========================================================================

def _chat_setup(root: Path, *, input_stem: str, glossary_stem: str):
    library = root / "Output"
    raw = library / "_raw"
    raw.mkdir(parents=True)
    epub = raw / f"{input_stem}.epub"
    epub.write_bytes(b"PK")
    book_dir = library / "Glossary" / glossary_stem
    book_dir.mkdir(parents=True)
    glossary = book_dir / f"{glossary_stem}_glossary.csv"
    glossary.write_text("type,raw_name,translated_name\ncharacter,김서연,Kim Seoyeon\n", encoding="utf-8")
    return library, epub, glossary


@pytest.mark.parametrize("mode,fuzzy,input_stem", [
    ("off", False, "Book"),
    ("balanced", False, "Book"),
    ("off_fuzzy_automap", True, "Book Vol 1"),
])
def test_chat_run_auto_maps_the_library_glossary(mobile_env, monkeypatch, mode, fuzzy, input_stem):
    import headless_owner
    from glossarion_mobile.job_kinds import direct_text as dt_job
    from glossarion_mobile.ui.chat import run_request
    from glossarion_mobile.ui.chat.direct_text_rules import DirectTextSettings

    library, epub, glossary = _chat_setup(mobile_env, input_stem=input_stem, glossary_stem="Book")
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(library))  # the platform output root (env contract)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(library))
    config = {"auto_glossary_mode": mode, "append_glossary": True, "append_glossary_auto_load": True,
              "fuzzy_auto_mapping": fuzzy, "fuzzy_auto_mapping_threshold": 50}
    logs = []
    owner = headless_owner.HeadlessOwner(config, host=SimpleNamespace(log=logs.append, emit=lambda *a, **k: None))

    run = run_request.prepare_direct_text_run(text="", attachment={"path": str(epub), "name": epub.name},
                                              temp_dir=str(mobile_env / "data" / "runs"))
    params = {"input_path": run.source_path, "output_root": run.temp_root, "is_attachment": True,
              "run": run.as_dict(),
              "options": run_request.run_options_dict(run, DirectTextSettings(glossary_override_mode="none"))}
    seen = {}

    def fake_run_translation(ctx, files):
        # what the shared worker does at run start, on the job's own owner and environment
        seen["output_directory"] = os.environ.get("OUTPUT_DIRECTORY")
        seen["guess"] = owner._guess_glossary_for_input_file(files[0])
        seen["synced"] = owner._sync_automapped_glossaries_to_output()
        seen["manual"] = os.environ.get("MANUAL_GLOSSARY")
        seen["env"] = owner._get_environment_variables(files[0], "key").get("MANUAL_GLOSSARY")
        return {"ok": True, "outputs": []}

    monkeypatch.setattr(dt_job, "run_translation", fake_run_translation)
    ctx = SimpleNamespace(params=params, inputs=[run.source_path], owner=owner, set_result=lambda **kw: None)
    assert dt_job.run(ctx) == {"ok": True, "outputs": []}

    assert os.path.normcase(seen["output_directory"]) == os.path.normcase(run.temp_root)  # unchanged run root
    assert os.path.normcase(seen["guess"] or "") == os.path.normcase(str(glossary))
    assert seen["synced"] == 1
    assert os.path.normcase(seen["manual"] or "") == os.path.normcase(str(glossary))
    assert os.path.normcase(seen["env"] or "") == os.path.normcase(str(glossary))
    assert any("Auto-mapped glossary" in line for line in logs), logs


def test_manual_glossary_only_still_maps_nothing(mobile_env, monkeypatch):
    import headless_owner
    from glossarion_mobile.job_kinds import direct_text as dt_job

    library, epub, _glossary = _chat_setup(mobile_env, input_stem="Book", glossary_stem="Book")
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(library))
    owner = headless_owner.HeadlessOwner({"auto_glossary_mode": "off_no_automap", "append_glossary": True},
                                         host=SimpleNamespace(log=lambda m: None, emit=lambda *a, **k: None))
    dt_job.keep_shared_glossary_root(owner)
    owner.selected_files = [str(epub)]
    assert owner._sync_automapped_glossaries_to_output() == 0
    assert not owner.manual_glossary_path


def test_keep_shared_glossary_root_keeps_an_existing_base_dir(mobile_env, monkeypatch):
    from glossarion_mobile.job_kinds import direct_text as dt_job

    monkeypatch.setenv("OUTPUT_DIRECTORY", str(mobile_env / "Output"))
    owner = SimpleNamespace(base_dir=str(mobile_env / "app"))
    dt_job.keep_shared_glossary_root(owner)
    assert owner.base_dir == str(mobile_env / "app")
    blank = SimpleNamespace()
    dt_job.keep_shared_glossary_root(blank)
    assert blank.base_dir == os.path.abspath(str(mobile_env / "Output"))
