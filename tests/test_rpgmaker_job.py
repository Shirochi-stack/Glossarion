"""rpgmaker_job: the RPG Maker (GTool) runner and the mobile game-folder entry (Glossarion mobile rewrite, U7).

``TranslatorGUI._process_rpgmaker_game`` moved verbatim into ``rpgmaker_job.RpgMakerJobMixin``
(``translation_pipeline.TranslationPipelineMixin`` inherits it in place of the U3 placeholder).
The desktop dispatches ``.exe`` inputs to it; mobile picks a game folder or a ZIP, which
``prepare_rpgmaker_game`` makes writable (``apply_translations`` patches the game's data files)
and ``_register_rpgmaker_game_input`` registers for ``run_translation_direct``.

What is checked here:

* verbatim: the moved method equals its text at ``runner_parity.LEGACY_SHA`` after the one
  documented edit (``rpgmaker_game_dir``); TranslatorGUI no longer defines it;
* runner-level trace parity (``tests/parity/runner_parity.py``) on fixture MV / MZ games: the
  legacy method and the mixin run the real ``rpgmaker_handler`` extraction / chunking / parsing /
  apply with a recording ``UnifiedClient`` (and a recording ``translate_game_images`` in image
  mode); logs, client calls with their environment, the final environment and the whole game
  tree (``GTool_Translation/translation_map.json`` + progress + ``originals_backup``, patched data)
  must be equal, including resume / scrub, re-runs over patched data, stops and failures;
* the mobile entry: game root detection, writable-folder / read-only copy / ZIP extraction (with a
  zip-slip guard), registration, and a HeadlessOwner run through the shared worker whose game
  output equals the desktop ``.exe`` path's; an unregistered folder stays unsupported.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_rpgmaker_job.py
"""

from __future__ import annotations

import ast
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(TESTS), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from parity import runner_parity as rp  # noqa: E402
from parity.runner_parity import PNG_BYTES, RunnerScenario, rpg_echo  # noqa: E402

EDITS = [("            game_dir = os.path.dirname(os.path.abspath(exe_path))\n",
          "            game_dir = rpgmaker_game_dir(exe_path)\n", 1)]


def _legacy_or_skip(name):
    try:
        return rp.legacy_method_text(name)
    except rp.Unavailable as exc:  # pragma: no cover - shallow clone
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))


# ---------------------------------------------------------------------------
# fixture games
# ---------------------------------------------------------------------------

SYSTEM = {"gameTitle": "勇者の冒険",
          "terms": {"basic": ["レベル", "HP", ""], "commands": ["戦う", "逃げる", None],
                    "params": ["最大HP"], "messages": {"actionFailure": "%1には効かなかった！", "empty": ""}}}
ACTORS = [None, {"id": 1, "name": "留奈", "nickname": "", "profile": "元気な少女。\n剣が得意。"},
          {"id": 2, "name": "留奈", "nickname": "影", "profile": ""}]
ITEMS = [None, {"id": 1, "name": "ポーション", "description": "HPを\\c[2]50\\c[0]回復する。"},
         {"id": 2, "name": "\\I[64]", "description": "123"}]
COMMON_EVENTS = [None, {"id": 1, "name": "宿屋", "list": [
    {"code": 401, "parameters": ["一泊50Gです。"]}, {"code": 102, "parameters": [["泊まる", "やめる"], 1]},
    {"code": 0, "parameters": []}]}]
MAP001 = {"displayName": "始まりの村", "events": [None, {"id": 1, "name": "村長", "pages": [{"list": [
    {"code": 101, "parameters": ["", 0, 0, 2, "村長"]},
    {"code": 401, "parameters": ["ようこそ、旅の方。"]},
    {"code": 401, "parameters": ["\\N[1]殿、お待ちしておりました。"]},
    {"code": 102, "parameters": [["はい", "いいえ"], 1]},
    {"code": 320, "parameters": [1, "勇者"]},
    {"code": 0, "parameters": []}]}]}]}


def _dump(value):
    return json.dumps(value, ensure_ascii=False)


def mv_game(prefix="game", *, mz=False, exe=True, layout="www"):
    """{relpath: content} of a small MV (www/data + www/js) or MZ (data + js/rmmz_core.js) game."""
    data = f"{prefix}/www/data" if layout == "www" else f"{prefix}/data"
    js = f"{prefix}/www/js" if layout == "www" else f"{prefix}/js"
    files = {
        f"{data}/System.json": _dump(SYSTEM),
        f"{data}/Actors.json": _dump(ACTORS),
        f"{data}/Items.json": _dump(ITEMS),
        f"{data}/CommonEvents.json": _dump(COMMON_EVENTS),
        f"{data}/Map001.json": _dump(MAP001),
        f"{data}/MapInfos.json": _dump([None, {"id": 1, "name": "MAP001"}]),
        f"{js}/{'rmmz_core.js' if mz else 'rpg_core.js'}": "// core\n",
        f"{prefix}/img/pictures/title.png": PNG_BYTES,
    }
    if exe:
        files[f"{prefix}/Game.exe"] = b"MZ\x90\x00fake-exe"
    return files


GAME_EXE = "@/game/Game.exe"


def _cfg(**extra):
    cfg = {"model": "gpt-4o", "api_key": "sk-rpg-0000", "output_language": "English",
           "prompt_text": "Translate this RPG Maker text into {target_lang}.\n{split_marker_instruction}\nKeep codes."}
    cfg.update(extra)
    return cfg


_TEXT_ATTRS = {"batch_translation_var": True, "batch_size_var": 2, "compression_factor_var": "3.0",
               "enable_image_translation_var": False}
_ECHO = (rpg_echo(),) * 12


def _rpg(name, *, files=None, config=None, attrs=None, env=None, args=(GAME_EXE,), **kw):
    return RunnerScenario(name, "_process_rpgmaker_game", args=args,
                          files=dict(mv_game() if files is None else files), config=_cfg(**(config or {})),
                          attrs=dict(_TEXT_ATTRS, **(attrs or {})), env=dict(env or {}), **kw)


def _progress(entries):
    return {"game/GTool_Translation/progress.json": _dump(entries)}


def _patched_rerun_files():
    """A game translated before: data files already patched, originals in the backup folder."""
    files = mv_game()
    patched = dict(SYSTEM, gameTitle="Hero's Adventure (old)")
    files["game/www/data/System.json"] = _dump(patched)
    files["game/GTool_Translation/originals_backup/www/data/System.json"] = _dump(SYSTEM)
    return files


_ALL_POOLS = {
    "use_multi_api_keys": True, "multi_api_keys": [{"api_key": "k1"}], "force_key_rotation": False,
    "rotation_frequency": 2, "use_fallback_keys": True, "fallback_keys": [{"api_key": "f1"}],
    "use_main_key_fallback": False, "fallback_key_shuffle": True,
    "use_glossary_keys": True, "glossary_keys": [{"api_key": "g1"}],
    "use_glossary_refinement_keys": True, "glossary_refinement_keys": [{"api_key": "r1"}],
    "use_metadata_keys": True, "metadata_keys": [{"api_key": "m1"}],
    "use_qa_scan_keys": True, "qa_scan_keys": [{"api_key": "v1"}],
    "use_rolling_summary_keys": True, "rolling_summary_keys": [{"api_key": "s1"}],
    "use_truncation_retry_keys": True, "truncation_retry_keys": [],
    "use_inpainter_keys": True, "inpainter_keys": [{"api_key": "i1"}],
}


def _stop(owner):
    owner.stop_requested = True


def _not_loaded(owner):
    owner._modules_loaded = False


RPG_SCENARIOS = [
    _rpg("mv_text_translate", send=_ECHO),
    _rpg("mz_text_translate", files=mv_game(mz=True, layout="root"), send=_ECHO),
    _rpg("mz_in_www_layout", files=mv_game(mz=True), send=_ECHO),
    _rpg("mv_sequential_no_batch", attrs={"batch_translation_var": False}, env={"BATCH_TRANSLATION": "0"},
         send=_ECHO),
    _rpg("mv_small_output_budget", attrs={"compression_factor_var": "0"}, config={}, send=_ECHO,
         before=lambda owner: setattr(owner, "max_output_tokens", 900)),
    _rpg("mv_resume_and_scrub", files=dict(mv_game(), **_progress({
        "www/data/System.json::gameTitle": "Hero's Adventure",
        "www/data/Actors.json::1_name": "",
        "www/data/Items.json::1_name": "\\c[2]",
        "www/data/Map001.json::displayName": "Starting Village",
        "www/data/Gone.json::0_name": "stale"})), send=_ECHO),
    _rpg("mv_rerun_restores_originals", files=_patched_rerun_files(), send=_ECHO),
    _rpg("mv_key_pools", config=_ALL_POOLS, send=_ECHO),
    _rpg("mv_key_pools_enabled_without_keys",
         config={"use_qa_scan_keys": True, "use_rolling_summary_keys": True, "use_truncation_retry_keys": True,
                 "use_inpainter_keys": True}, send=_ECHO),
    _rpg("mv_modules_not_loaded", before=_not_loaded, send=_ECHO),
    _rpg("mv_stop_requested", before=_stop, send=_ECHO),
    _rpg("mv_chunk_send_fails", send=(RuntimeError("rate limited"),) * 12),
    _rpg("mv_unparsable_response", send=("Sorry, I cannot help with that.",) * 12),
    _rpg("mv_empty_response", send=("",) * 12),
    _rpg("mv_image_mode", attrs={"enable_image_translation_var": True,
                                 "prompt_profiles": {"RPGMaker_GTool_Image": "Translate the image text."},
                                 "gtool_filter_user_prompt_var": "Is there text?",
                                 "gtool_scan_user_prompt_var": "List the text."}, images=3),
    _rpg("mv_image_mode_env_flag_defaults", attrs={"enable_image_translation_var": False,
                                                   "default_prompts": {"RPGMaker_GTool_Image": "Default image prompt"}},
         env={"ENABLE_IMAGE_TRANSLATION": "1"}, images=0),
    _rpg("vxace_image_mode_refused", files={"game/Game.exe": b"MZ", "game/Data/Actors.rvdata2": b"\x04\x08[\x00"},
         attrs={"enable_image_translation_var": True}),
    _rpg("unknown_game", files={"game/Game.exe": b"MZ", "game/readme.txt": "not a game"}),
    _rpg("no_translatable_strings", files={"game/Game.exe": b"MZ", "game/www/data/System.json": "{}",
                                           "game/www/js/rpg_core.js": "//"}),
    _rpg("missing_exe_folder", args=("@/nowhere/Game.exe",)),
]


@pytest.fixture(scope="module", autouse=True)
def _legacy_available():
    _legacy_or_skip("_process_rpgmaker_game")


def test_moved_method_is_the_legacy_text_plus_the_documented_edit():
    legacy = _legacy_or_skip("_process_rpgmaker_game")
    for old, new, count in EDITS:
        assert legacy.count(old) == count
        legacy = legacy.replace(old, new)
    text = rp.module_text("rpgmaker_job")
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RpgMakerJobMixin")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert set(methods) == {"_process_rpgmaker_game", "_register_rpgmaker_game_input"}
    assert rp.node_text(text, methods["_process_rpgmaker_game"]) == legacy


def test_dispatch_edit_is_the_only_change_to_run_translation_direct():
    import translation_pipeline as tp

    text = rp.module_text("translation_pipeline")
    assert text.count("elif ext == '.exe' or file_path in (getattr(self, RPGMAKER_GAME_INPUTS_ATTR, None) or ()):") == 1
    assert "elif ext == '.exe':" not in text
    import rpgmaker_job

    assert tp.RPGMAKER_GAME_INPUTS_ATTR == rpgmaker_job.RPGMAKER_GAME_INPUTS_ATTR == "_rpgmaker_game_inputs"


@pytest.mark.parametrize("scenario", RPG_SCENARIOS, ids=lambda s: s.name)
def test_runner_matches_the_legacy_desktop_method(scenario, tmp_path):
    legacy, new = rp.run_pair(scenario, tmp_path)
    problems = rp.compare(legacy, new)
    assert not problems, "\n".join(problems)
    assert legacy["error"] is None, legacy["error"]


def test_the_matrix_reaches_its_paths(tmp_path):
    by_name = {s.name: s for s in RPG_SCENARIOS}

    def run(name):
        return rp.run_side("legacy", by_name[name], tmp_path / name)

    def logs(result):
        return [e[1] for e in result["events"] if e[0] == "log"]

    def calls(result, name):
        return [e[1] for e in result["events"] if e[0] == name]

    r = run("mv_text_translate")
    assert r["result"] is True, logs(r)[-10:]
    patched = r["fs"]["game/www/data/System.json"]["json"]
    assert patched["gameTitle"] == "EN(勇者の冒険)"
    assert "game/GTool_Translation/originals_backup/www/data/System.json" in r["fs"]
    assert r["fs"]["game/GTool_Translation/translation_map.json"]["json"]["www/data/System.json"]["gameTitle"][
        "translated"] == "EN(勇者の冒険)"
    send = calls(r, "UnifiedClient.send")[0]
    assert send["messages"][0]["content"].startswith("Translate this RPG Maker text into English.")
    assert send["env"]["set"]["IS_TEXT_FILE_TRANSLATION"] == "1" and send["env"]["set"]["BATCH_TRANSLATION"] == "1"
    r = run("mz_text_translate")
    assert any("Detected RPG Maker MZ" in line for line in logs(r))
    r = run("mv_resume_and_scrub")
    assert any("Cleaned 2 invalid translations" in line for line in logs(r)), logs(r)
    r = run("mv_rerun_restores_originals")
    assert any("Restoring 1 original files" in line for line in logs(r))
    r = run("mv_image_mode")
    (images,) = calls(r, "rpgmaker_handler.translate_game_images")
    assert images["system_prompt"] == "Translate the image text." and images["batch_size"] == 2
    assert r["result"] is True
    r = run("vxace_image_mode_refused")
    assert r["result"] is False and any("only supported for MV/MZ" in line for line in logs(r))
    r = run("mv_chunk_send_fails")
    assert any("failed: rate limited" in line for line in logs(r))


# ---------------------------------------------------------------------------
# the mobile game-folder entry
# ---------------------------------------------------------------------------


def _write(root: Path, files: dict):
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)


def _zip(files: dict, extra=()) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for rel, content in files.items():
            zf.writestr(rel, content)
        for rel, content in extra:
            zf.writestr(rel, content)
    return buf.getvalue()


def test_game_dir_and_root_detection(tmp_path):
    import rpgmaker_job

    _write(tmp_path, mv_game("wrap/My Game", exe=False))
    game = tmp_path / "wrap" / "My Game"
    assert rpgmaker_job.rpgmaker_game_dir(str(game / "Game.exe")) == str(game)
    assert rpgmaker_job.rpgmaker_game_dir(str(game)) == str(game)
    assert rpgmaker_job.find_rpgmaker_game_root(str(tmp_path)) == str(game)
    # a web / Android deployment (index.html + data/ + js/) is a game root of its own
    assert rpgmaker_job.find_rpgmaker_game_root(str(game / "www")) == str(game / "www")
    assert rpgmaker_job.find_rpgmaker_game_root(str(game / "img")) is None
    assert rpgmaker_job.find_rpgmaker_game_root(str(tmp_path / "missing")) is None


def test_prepare_uses_a_writable_folder_in_place(tmp_path):
    import rpgmaker_job

    _write(tmp_path, mv_game("pick/game"))
    logs = []
    root = rpgmaker_job.prepare_rpgmaker_game(str(tmp_path / "pick"), str(tmp_path / "work"), log=logs.append)
    assert root == str(tmp_path / "pick" / "game") and not (tmp_path / "work").exists() and logs == []
    assert rpgmaker_job.prepare_rpgmaker_game(str(tmp_path / "pick" / "game" / "Game.exe"), None) == root


def test_prepare_copies_a_read_only_folder_once(tmp_path, monkeypatch):
    import rpgmaker_job

    _write(tmp_path, mv_game("ro/game", exe=False))
    source = str(tmp_path / "ro" / "game")
    real_access = os.access
    monkeypatch.setattr(rpgmaker_job.os, "access",
                        lambda p, mode: False if str(p).startswith(source) else real_access(p, mode))
    logs = []
    root = rpgmaker_job.prepare_rpgmaker_game(source, str(tmp_path / "work"), log=logs.append)
    container = Path(root).parent
    assert Path(root).name == "game" and container.parent == tmp_path / "work" / rpgmaker_job.COPY_SUBFOLDER
    assert container.name.startswith("game-") and (container / rpgmaker_job.SOURCE_MARKER).is_file()
    assert len(logs) == 1 and "read-only" in logs[0]
    assert json.loads((Path(root) / "www" / "data" / "System.json").read_text(encoding="utf-8")) == SYSTEM
    (Path(root) / "GTool_Translation").mkdir()
    assert rpgmaker_job.prepare_rpgmaker_game(source, str(tmp_path / "work"), log=logs.append) == root
    assert len(logs) == 1 and (Path(root) / "GTool_Translation").is_dir()  # kept: resume progress lives there
    with pytest.raises(ValueError):
        rpgmaker_job.prepare_rpgmaker_game(str(tmp_path / "missing-folder"), str(tmp_path / "work2"))


def test_prepare_extracts_a_zip_once_into_the_work_dir(tmp_path):
    import rpgmaker_job

    game = {rel.replace("game/", "Hero Quest/", 1): content for rel, content in mv_game().items()}
    archive = tmp_path / "Hero Quest.zip"
    archive.write_bytes(_zip(game))
    work = tmp_path / "work"
    logs = []
    root = rpgmaker_job.prepare_rpgmaker_game(str(archive), str(work), log=logs.append)
    target = Path(root).parent
    assert Path(root).name == "Hero Quest" and target.parent == work / rpgmaker_job.ZIP_SUBFOLDER
    assert target.name.startswith("Hero Quest-") and (Path(root) / "Game.exe").is_file()
    marker = Path(root) / "GTool_Translation" / "progress.json"
    marker.parent.mkdir()
    marker.write_text("{}", encoding="utf-8")
    assert rpgmaker_job.prepare_rpgmaker_game(str(archive), str(work), log=logs.append) == root
    assert marker.is_file() and len(logs) == 1  # reused: no second extraction
    # another copy of the same archive (a picker hands over a copy) resumes from the same folder
    copy = tmp_path / "picked" / "Hero Quest.zip"
    copy.parent.mkdir()
    copy.write_bytes(archive.read_bytes())
    assert rpgmaker_job.prepare_rpgmaker_game(str(copy), str(work), log=logs.append) == root
    assert marker.is_file() and len(logs) == 1
    assert rpgmaker_job.prepare_rpgmaker_game(str(archive), str(work), log=logs.append, fresh=True) == root
    assert not marker.exists() and len(logs) == 2
    with pytest.raises(ValueError):
        rpgmaker_job.prepare_rpgmaker_game(str(archive), None)


def _titled_zip(path: Path, title: str, top: str = "game") -> Path:
    system = dict(SYSTEM, gameTitle=title)
    files = {rel.replace("game/", f"{top}/", 1): content for rel, content in mv_game().items()}
    files[f"{top}/www/data/System.json"] = json.dumps(system, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_zip(files))
    return path


def _title(root: str) -> str:
    return json.loads((Path(root) / "www" / "data" / "System.json").read_text(encoding="utf-8"))["gameTitle"]


def test_prepare_never_reuses_another_games_extraction_or_copy(tmp_path, monkeypatch):
    """Two different games with the same ZIP / folder name get their own work folders."""
    import rpgmaker_job

    work = str(tmp_path / "work")
    first = _titled_zip(tmp_path / "downloads_a" / "game.zip", "FIRST GAME")
    second = _titled_zip(tmp_path / "downloads_b" / "game.zip", "SECOND GAME")
    root_a = rpgmaker_job.prepare_rpgmaker_game(str(first), work, log=lambda *_a: None)
    root_b = rpgmaker_job.prepare_rpgmaker_game(str(second), work, log=lambda *_a: None)
    assert root_a != root_b and (_title(root_a), _title(root_b)) == ("FIRST GAME", "SECOND GAME")
    assert rpgmaker_job.prepare_rpgmaker_game(str(first), work, log=lambda *_a: None) == root_a
    assert _title(root_a) == "FIRST GAME"
    # read-only folders with the same name (two SD-card games both called "game")
    folders = []
    for side, title in (("sd_a", "FIRST FOLDER"), ("sd_b", "SECOND FOLDER")):
        _write(tmp_path / side, mv_game(exe=False))
        system = tmp_path / side / "game" / "www" / "data" / "System.json"
        system.write_text(json.dumps(dict(SYSTEM, gameTitle=title), ensure_ascii=False), encoding="utf-8")
        folders.append(str(tmp_path / side / "game"))
    monkeypatch.setattr(rpgmaker_job, "_is_writable_game", lambda game_dir: False)
    copy_a = rpgmaker_job.prepare_rpgmaker_game(folders[0], work, log=lambda *_a: None)
    copy_b = rpgmaker_job.prepare_rpgmaker_game(folders[1], work, log=lambda *_a: None)
    assert copy_a != copy_b and (_title(copy_a), _title(copy_b)) == ("FIRST FOLDER", "SECOND FOLDER")
    assert {Path(p).name for p in (copy_a, copy_b)} == {"game"}
    # ZIP extractions and folder copies live in separate subfolders
    assert Path(root_a).parents[1] != Path(copy_a).parents[1]


def test_prepare_rebuilds_an_interrupted_extraction_and_serialises_callers(tmp_path):
    import threading

    import rpgmaker_job

    work = str(tmp_path / "work")
    archive = _titled_zip(tmp_path / "game.zip", "WHOLE GAME")
    logs = []
    root = rpgmaker_job.prepare_rpgmaker_game(str(archive), work, log=logs.append)
    target = Path(root).parent
    # an extraction that never finished has no marker: rebuilt, not reused
    (target / rpgmaker_job.SOURCE_MARKER).unlink()
    (Path(root) / "www" / "data" / "System.json").unlink()
    assert rpgmaker_job.prepare_rpgmaker_game(str(archive), work, log=logs.append) == root
    assert len(logs) == 2 and _title(root) == "WHOLE GAME"
    # a scan and a job preparing the same ZIP at once: one extraction, the same root
    other = _titled_zip(tmp_path / "other" / "game.zip", "RACED GAME")
    logs.clear()
    barrier = threading.Barrier(2)
    results = []

    def prepare():
        barrier.wait()
        results.append(rpgmaker_job.prepare_rpgmaker_game(str(other), work, log=logs.append))

    threads = [threading.Thread(target=prepare) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert len(results) == 2 and results[0] == results[1] and _title(results[0]) == "RACED GAME"
    assert len(logs) == 1


def test_prepare_refuses_unsafe_or_gameless_archives(tmp_path):
    import rpgmaker_job

    evil = tmp_path / "evil.zip"
    evil.write_bytes(_zip(mv_game(), extra=[("../escaped.txt", "x")]))
    with pytest.raises(ValueError, match="unsafe path"):
        rpgmaker_job.prepare_rpgmaker_game(str(evil), str(tmp_path / "work"))
    assert not (tmp_path / "escaped.txt").exists()
    plain = tmp_path / "plain.zip"
    plain.write_bytes(_zip({"notes/readme.txt": "hello"}))
    with pytest.raises(ValueError, match="No RPG Maker game"):
        rpgmaker_job.prepare_rpgmaker_game(str(plain), str(tmp_path / "work"))
    zips = tmp_path / "work" / rpgmaker_job.ZIP_SUBFOLDER
    assert not zips.exists() or not any(zips.iterdir())  # no half-made or gameless extraction is kept
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="Not a valid ZIP"):
        rpgmaker_job.prepare_rpgmaker_game(str(broken), str(tmp_path / "work"))
    (tmp_path / "folder").mkdir()
    with pytest.raises(ValueError, match="No RPG Maker game"):
        rpgmaker_job.prepare_rpgmaker_game(str(tmp_path / "folder"), str(tmp_path / "work"))


class _EchoClient:
    """UnifiedClient stand-in for the HeadlessOwner runs (same echo as the parity scenarios)."""

    calls: list = []

    def __init__(self, *args, **kwargs):
        pass

    def send(self, messages=None, temperature=None, max_tokens=None, **_kw):
        type(self).calls.append(messages)
        return rpg_echo()(None, {"messages": messages}), "stop"

    @classmethod
    def _model_needs_api_key(cls, model):
        return True


def _run_headless(tmp_path, monkeypatch, inputs, *, register=None):
    import job_runner
    import rpgmaker_handler
    import stop_control
    import unified_api_client
    from _headless_env import headless_owner

    monkeypatch.setattr(unified_api_client, "UnifiedClient", _EchoClient)
    monkeypatch.setattr(rpgmaker_handler._get_tiktoken_encoder, "_enc", None, raising=False)
    monkeypatch.setitem(sys.modules, "epub_library", None)
    logs = []
    host = type("H", (), {"log": lambda self, m, **k: logs.append(str(m)), "emit": lambda self, k, **d: None,
                          "ask": lambda self, k, **d: False, "is_stop_requested": lambda self: False,
                          "is_graceful_stop": lambda self: False})()
    cfg = {"model": "gpt-4o", "api_key": "sk-test-0000", "output_directory": str(tmp_path / "out"),
           "auto_update_check": False, "batch_translation": True, "batch_size": 2}
    with headless_owner(tmp_path / "app", monkeypatch, cfg, host=host) as owner:
        os.environ["GLOSSARION_LIBRARY_DIR"] = str(tmp_path / "Library")
        files = list(inputs)
        if register is not None:
            files = [owner._register_rpgmaker_game_input(register, str(tmp_path / "work"))]
        with job_runner.job_process_state(host.log):
            stop_control.reset_for_new_run("translation")
            request = owner._prepare_translation_run(files)
            outcome = owner._translation_worker(request) if request is not None else None
        registered = list(getattr(owner, "_rpgmaker_game_inputs", None) or [])
    return outcome, logs, registered


def _game_tree(game: Path) -> dict:
    out = {}
    for path in sorted(game.rglob("*")):
        if path.is_file() and path.name != "Game.exe":
            data = path.read_bytes()
            out[path.relative_to(game).as_posix()] = (json.loads(data.decode("utf-8"))
                                                      if path.suffix == ".json" else data)
    return out


def test_headless_owner_folder_entry_matches_the_exe_path(tmp_path, monkeypatch):
    """Mobile: a game folder without an .exe, registered, runs through the shared worker and
    leaves exactly the game output of the desktop .exe dispatch on the same game."""
    _write(tmp_path / "exe_side", mv_game())
    _write(tmp_path / "dir_side", mv_game(exe=False))
    exe_outcome, exe_logs, none_registered = _run_headless(
        tmp_path / "a", monkeypatch, [str(tmp_path / "exe_side" / "game" / "Game.exe")])
    dir_outcome, dir_logs, registered = _run_headless(
        tmp_path / "b", monkeypatch, [], register=str(tmp_path / "dir_side" / "game"))
    assert exe_outcome is True, exe_logs[-15:]
    assert dir_outcome is True, dir_logs[-15:]
    assert none_registered == [] and registered == [str(tmp_path / "dir_side" / "game")]
    assert any("🎮 GTool: Translation complete!" in line for line in dir_logs)
    exe_tree = _game_tree(tmp_path / "exe_side" / "game")
    dir_tree = _game_tree(tmp_path / "dir_side" / "game")
    assert "GTool_Translation/progress.json" in exe_tree
    assert exe_tree == dir_tree


def test_unregistered_folder_input_stays_unsupported(tmp_path, monkeypatch):
    """The desktop dispatch is unchanged: a folder nobody registered is not an RPG Maker input."""
    _write(tmp_path / "plain", mv_game(exe=False))
    outcome, logs, registered = _run_headless(tmp_path / "c", monkeypatch, [str(tmp_path / "plain" / "game")])
    assert registered == []
    assert any(line.startswith("⚠️ Unsupported file type") for line in logs), logs[-15:]
    assert not (tmp_path / "plain" / "game" / "GTool_Translation").exists()
    del outcome


def test_register_is_idempotent_and_reports_bad_sources(tmp_path):
    import rpgmaker_job

    class Owner(rpgmaker_job.RpgMakerJobMixin):
        def __init__(self):
            self.logs = []

        def append_log(self, message):
            self.logs.append(message)

    _write(tmp_path, mv_game(exe=False))
    owner = Owner()
    first = owner._register_rpgmaker_game_input(str(tmp_path / "game"))
    second = owner._register_rpgmaker_game_input(str(tmp_path / "game" / "www" / ".." ))
    assert first == second == str(tmp_path / "game")
    assert owner._rpgmaker_game_inputs == [first]
    with pytest.raises(ValueError):
        owner._register_rpgmaker_game_input(str(tmp_path / "missing"))
