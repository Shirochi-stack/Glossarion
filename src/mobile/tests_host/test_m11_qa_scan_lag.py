"""v9.15.11 batch, owner issues 4 and 5: the QA scanner's workers and the lag while it scans.

Issue 4: "Increase the default parallel cores on mobile for the QA scanner ... just assign half of all cores in
the device. Actually I don't even see the setting for this in the QA scanner, so it should be added."
Issue 5: "The mobile GUI was lagging a lot during the QA scanning."

MEASURED (``tools/qa_scan_bench.py``, the before / after tables are in the batch report): the scan's workers are
threads next to the Flet loop, so 1, 2, 4 and 8 of them take about the same time (the GIL) while a burst of work on
the loop takes (workers + 1) times as long and its timers wait for the 5 ms switch interval. Log lines and repaints
are about 1 % of the loop's time. So a scan's CPU workers now take turns (one competes with the loop, however many
there are) and it runs with a 1 ms switch interval. Half the cores stays the default the owner asked for.

The budgets compare numbers of the same run, so a slow runner does not fail them.
"""

from __future__ import annotations

import asyncio
import ctypes
import importlib.util
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for _entry in (str(APP_DIR), str(SRC_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def _load(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(alias, module)
    spec.loader.exec_module(module)
    return module


bench = _load(MOBILE_DIR / "tools" / "qa_scan_bench.py", "_glossarion_qa_scan_bench")
_TB = _load(Path(__file__).with_name("test_bootstrap.py"), "_glossarion_tb_helpers_m11_qa_lag")
storage = _TB.storage  # noqa: F811  (pytest fixtures)
app_env = _TB.app_env


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _backend_error() -> str:
    try:
        import qa_scan_runtime  # noqa: F401
        import scan_html_folder  # noqa: F401
    except Exception as exc:  # pragma: no cover - a venv without the backend dependencies
        return f"the shared QA scanner is not importable here ({exc})"
    return ""


_BACKEND_ERROR = _backend_error()
needs_backend = pytest.mark.skipif(bool(_BACKEND_ERROR), reason=_BACKEND_ERROR or "backend importable")
needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")


@pytest.fixture
def phone(tmp_path, monkeypatch):
    """The phone's gates, every path in tmp, no workers left in the env by an earlier run."""
    for name in ("GLOSSARION_MOBILE", "GLOSSARION_NO_PROCESSES", "GLOSSARION_HTTP_LOG"):
        monkeypatch.setenv(name, "0" if name == "GLOSSARION_HTTP_LOG" else "1")
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"), ("LOCALAPPDATA", "local"),
                      ("GLOSSARION_LIBRARY_DIR", "library"), ("OUTPUT_DIRECTORY", "out"), ("GLOSSARION_DATA_DIR", "data")):
        (tmp_path / sub).mkdir(exist_ok=True)
        monkeypatch.setenv(name, str(tmp_path / sub))
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "data" / "config.json"))
    monkeypatch.delenv("AI_HUNTER_MAX_WORKERS", raising=False)
    return tmp_path


class ScanCtx:
    """What ``job_kinds.qa.scan`` needs of a job context, over the real ``HeadlessOwner``."""

    def __init__(self, config: dict) -> None:
        from headless_owner import HeadlessOwner

        self.config = config
        self.owner = HeadlessOwner(json.loads(json.dumps(config)))
        self.lines: list = []
        self.results: dict = {}

    def log(self, text) -> None:
        self.lines.append(str(text))

    def phase(self, _label) -> None:
        pass

    def stop_requested(self) -> bool:
        return False

    def add_outputs(self, _paths) -> None:
        pass

    def set_result(self, **data) -> None:
        self.results.update(data)


def _scan(folder, config: dict, *, env_workers=None, mode: str = "quick-scan") -> ScanCtx:
    from glossarion_mobile.job_kinds import qa as qa_kind

    ctx = ScanCtx(config)
    if env_workers is not None:  # after the owner, which mirrors its config into the env
        os.environ["AI_HUNTER_MAX_WORKERS"] = env_workers
    result = qa_kind.scan(ctx, [(str(folder), None, False)], mode, disable_word_count=True)
    assert result["ok"] is True, ctx.lines[-20:]
    return ctx


def _book_config(workers=None, sample: int = 1000) -> dict:
    config: dict = {"output_language": "English", "qa_scanner_settings": {"quick_scan_sample_size": sample}}
    if workers is not None:
        config["ai_hunter_config"] = {"ai_hunter_max_workers": workers}
    return config


# ---- issue 4: half the cores by default, up to the cores ---------------------------------------------------


@needs_backend
@pytest.mark.parametrize("cores,configured,expected", [
    (8, None, 4), (8, "", 4), (8, 0, 4), (8, "0", 4), (8, "x", 4), (8, -3, 4),  # no setting; never the desktop's "all"
    (8, 1, 1), (8, 3, 3), (8, "6", 6), (8, 8, 8), (8, 16, 8),  # the setting, up to the cores
    (1, None, 1), (2, None, 1), (3, None, 1), (12, None, 6), (1, 4, 1),
])
def test_workers_are_half_the_cores_unless_set_and_never_above_them(phone, monkeypatch, cores, configured, expected):
    import qa_scan_runtime as qsr

    monkeypatch.setattr(os, "cpu_count", lambda: cores)
    monkeypatch.setenv("AI_HUNTER_MAX_WORKERS", "" if configured is None else str(configured))
    assert qsr.mobile_qa_max_workers() == expected
    assert qsr.mobile_qa_env_overrides() == {"QA_USE_THREAD_EXECUTOR": "1", "AI_HUNTER_MAX_WORKERS": str(expected)}


@needs_backend
def test_desktop_never_applies_the_mobile_workers(monkeypatch):
    import qa_scan_runtime as qsr

    for name in ("GLOSSARION_MOBILE", "GLOSSARION_NO_PROCESSES", "FLET_PLATFORM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AI_HUNTER_MAX_WORKERS", "16")
    assert qsr.mobile_qa_env_overrides() == {}
    previous = qsr.apply_qa_scan_env_from_settings({})
    try:
        assert "AI_HUNTER_MAX_WORKERS" not in previous and os.environ["AI_HUNTER_MAX_WORKERS"] == "16"
    finally:
        qsr.restore_env(previous)


@needs_backend
@pytest.mark.parametrize("saved,expected", [(None, 4), (3, 3), ("5", 5), (16, 8), (0, 4)])
def test_a_real_scan_runs_with_those_workers(phone, monkeypatch, saved, expected):
    """config.json decides, not a mirror an earlier run left in the env; the env is put back after the scan."""
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setenv("AI_HUNTER_MAX_WORKERS", "2")
    folder = bench.make_book(phone / "out" / "Book", chapters=12, words=300)
    ctx = _scan(folder, _book_config(saved), env_workers="2")
    assert f"   🖥️ Using {expected} CPU cores for file processing (configured limit)" in ctx.lines
    assert not any("Using ALL" in line for line in ctx.lines), "the desktop's all-cores path ran"
    assert os.environ["AI_HUNTER_MAX_WORKERS"] == "2"


# ---- issue 4: the setting on the QA scanner -----------------------------------------------------------------


@needs_flet
@needs_backend
def test_tools_qa_scanner_shows_the_workers_and_lets_them_change(phone, monkeypatch):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.tools.qa_screen import THREADS_REASON, QaScannerScreen

    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    ui = _load(Path(__file__).with_name("test_tools_ui.py"), "_glossarion_tools_ui_helpers_m11_qa_lag")
    target = ui.tool_target(phone, "Alpha")

    def open_screen(store):
        _conn, session = _TB._fake_session("android")
        ctx = ui._ctx(session.page, store=store)
        screen = QaScannerScreen(parse_route("/tools/qa"), ctx)
        ui._mount(session.page, screen.get_body())
        return screen, ctx

    def under(control, key):
        """The control with ``key`` under ``control``."""
        if getattr(control, "key", None) == key:
            return control
        children = list(getattr(control, "controls", None) or ())
        if getattr(control, "content", None) is not None:
            children.append(control.content)
        return next((found for found in (under(child, key) for child in children) if found is not None), None)

    async def scenario():
        fresh = {"qa_scanner_settings": {"check_word_count_ratio": False}}
        screen, ctx = open_screen(fresh)
        field = screen.workers_field
        # in the Scan options card, half of the 8 cores, with the range in its help
        assert under(under(screen.body_list, "qa-fields"), "qa-workers") is field and field.value == "4"
        assert field.label == "Maximum parallel workers:" and "1 to 8" in field.helper and "default 4" in field.helper
        # the untouched default is not written, also at Start
        assert screen._on_workers() and "ai_hunter_config" not in fresh
        screen.set_targets([target])
        assert await screen.start() and "ai_hunter_config" not in fresh
        # a typed number is saved under the desktop's key
        field.value = "6"
        assert screen._on_workers() and fresh["ai_hunter_config"]["ai_hunter_max_workers"] == 6
        # the device's cores are the limit; nothing else is saved, and Start refuses the bad text
        for bad in ("9", "0", "-1", "two", ""):
            field.value = bad
            assert not screen._on_workers() and field.error == "Whole number from 1 to 8", bad
            assert fresh["ai_hunter_config"]["ai_hunter_max_workers"] == 6
        specs = len(ctx.jobs.specs)
        assert await screen.start() is None and len(ctx.jobs.specs) == specs
        field.value = "8"
        assert screen._on_workers() and field.error is None and fresh["ai_hunter_config"]["ai_hunter_max_workers"] == 8
        # a reopened screen shows the saved number; one saved above this device's cores (a desktop import) shows
        # what the scan uses and stays as it is in config.json
        assert open_screen(fresh)[0].workers_field.value == "8"
        imported = {"ai_hunter_config": {"ai_hunter_max_workers": 32}}
        again = open_screen(imported)[0]
        assert again.workers_field.value == "8" and again._on_workers()
        assert imported["ai_hunter_config"]["ai_hunter_max_workers"] == 32

    asyncio.run(scenario())
    assert "at most 2" not in THREADS_REASON and "Maximum parallel workers" in THREADS_REASON


@needs_flet
@needs_backend
def test_settings_ai_hunter_shows_the_number_a_scan_uses(storage, app_env):
    """Settings › AI Hunter: half the cores when nothing is saved, its note says what mobile does; not written."""
    import settings_schema

    from glossarion_mobile.job_kinds.qa import WORKERS_KEY, default_scan_workers
    from glossarion_mobile.ui.settings.section_page import SectionPage

    note = settings_schema.MOBILE_NOTES["ai_hunter_config.ai_hunter_max_workers"]
    assert "at most 2" not in note and "half" in note and "MOBILE_QA_MAX_WORKERS" not in note
    tf = _load(Path(__file__).with_name("test_ui_foundations.py"), "_glossarion_tf_helpers_m11_qa_lag")

    async def scenario():
        _m, _conn, _session, _page, app = await tf._start("android")
        try:
            await tf._wait(lambda: app.state.engine_ready, timeout=15)
            app.navigate_to("settings.section", {"section": "qa.ai_hunter"})
            assert await tf._wait(lambda: isinstance(app.shell.top_screen, SectionPage), 10)
            section = app.shell.top_screen
            assert await tf._wait(lambda: section.keys, 10)
            key = ".".join(WORKERS_KEY)
            assert key in section.keys
            tile = section.tile(key)
            assert tile.field.value == str(default_scan_workers()) and tile.editable and not tile.stored
            assert app.config_store.get(WORKERS_KEY) is None
        finally:
            await tf._stop(app)

    asyncio.run(scenario())


# ---- issue 5: the switch interval around a scan --------------------------------------------------------------


class _FakeRuntime:
    """The shared loop's surface for ``scan``; records what a running scan sees."""

    def __init__(self, error=None) -> None:
        self.error = error
        self.seen: dict = {}

    is_direct_text_qa_path = staticmethod(lambda _path: False)
    reset_qa_cancel_flags = staticmethod(lambda: None)
    load_current_qa_settings = staticmethod(lambda _config: {})
    mobile_qa_max_workers = staticmethod(lambda configured=None: 3)

    def run_bulk_qa_scan(self, folders, **_kwargs):
        self.seen = {"interval": sys.getswitchinterval(), "workers": os.environ.get("AI_HUNTER_MAX_WORKERS")}
        if self.error is not None:
            raise self.error
        return 1, 0


class _FakeCtx:
    config = {}
    owner = None
    host = None

    def log(self, _text):
        pass

    def phase(self, _label):
        pass

    def stop_requested(self):
        return False

    def add_outputs(self, _paths):
        pass

    def set_result(self, **_data):
        pass


def test_a_scan_shortens_the_gil_switch_interval_and_restores_it(tmp_path, monkeypatch):
    from glossarion_mobile.job_kinds import qa as qa_kind

    monkeypatch.delenv("AI_HUNTER_MAX_WORKERS", raising=False)
    before = sys.getswitchinterval()
    try:
        sys.setswitchinterval(0.005)
        runtime = _FakeRuntime()
        assert qa_kind.scan(_FakeCtx(), [(str(tmp_path), None, False)], "quick-scan", qa=runtime)["ok"] is True
        assert runtime.seen == {"interval": pytest.approx(qa_kind.SCAN_SWITCH_INTERVAL), "workers": "3"}
        assert sys.getswitchinterval() == pytest.approx(0.005) and "AI_HUNTER_MAX_WORKERS" not in os.environ
        # also when the scan raises
        failing = _FakeRuntime(error=RuntimeError("scan died"))
        with pytest.raises(RuntimeError):
            qa_kind.scan(_FakeCtx(), [(str(tmp_path), None, False)], "quick-scan", qa=failing)
        assert failing.seen["interval"] == pytest.approx(qa_kind.SCAN_SWITCH_INTERVAL)
        assert sys.getswitchinterval() == pytest.approx(0.005) and "AI_HUNTER_MAX_WORKERS" not in os.environ
        # never above what the process already runs with
        sys.setswitchinterval(0.0005)
        runtime = _FakeRuntime()
        qa_kind.scan(_FakeCtx(), [(str(tmp_path), None, False)], "quick-scan", qa=runtime)
        assert runtime.seen["interval"] == pytest.approx(0.0005) and sys.getswitchinterval() == pytest.approx(0.0005)
    finally:
        sys.setswitchinterval(before)


def _turns_pool(batches: int, items: int, stop=lambda: False, workers: int = 4, item_seconds: float = 0.002):
    """Real pool threads through ``_taking_turns``: how many ran at once between two stop checks, and what ran."""
    from concurrent.futures import ThreadPoolExecutor

    from glossarion_mobile.job_kinds import qa as qa_kind

    lock = threading.Lock()
    state = {"running": 0, "most": 0, "items": 0}

    def worker(args, should_stop):
        for _ in range(items):
            if should_stop is not None and should_stop():
                break
            with lock:
                state["running"] += 1
                state["most"] = max(state["most"], state["running"])
                state["items"] += 1
            time.sleep(item_seconds)
            with lock:
                state["running"] -= 1
        return args

    def submit(pool, work, args, should_stop):  # scan_html_folder._submit_worker
        return pool.submit(work, args, should_stop)

    taking_turns = qa_kind._taking_turns(submit)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [taking_turns(pool, worker, index, stop) for index in range(batches)]
        results = [future.result(timeout=60) for future in futures]
    return state, results


def test_workers_take_turns_and_every_batch_finishes():
    state, results = _turns_pool(batches=12, items=6)
    assert state["most"] == 1 and state["items"] == 72 and results == list(range(12))


def test_turns_are_not_given_up_for_a_batch_that_raises():
    from concurrent.futures import ThreadPoolExecutor

    from glossarion_mobile.job_kinds import qa as qa_kind

    def worker(args, should_stop):
        should_stop()
        if args == 0:
            raise ValueError("a batch died")
        return args

    taking_turns = qa_kind._taking_turns(lambda pool, work, args, stop: pool.submit(work, args, stop))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [taking_turns(pool, worker, index, lambda: False) for index in range(6)]
        assert futures[0].exception(timeout=30) is not None
        assert [future.result(timeout=30) for future in futures[1:]] == [1, 2, 3, 4, 5]


def test_a_stop_frees_the_workers_waiting_for_their_turn():
    """A worker waits for its turn only until the Stop: it sees it within the wait's short poll."""
    from glossarion_mobile.job_kinds.qa import _WorkerTurns

    turns = _WorkerTurns()
    stop = threading.Event()
    holding, freed = threading.Event(), threading.Event()

    def holder():
        turns.take(stop.is_set)
        holding.set()
        stop.wait(30)
        turns.leave()

    def waiter():
        turns.take(stop.is_set)
        freed.set()

    threading.Thread(target=holder, daemon=True).start()
    assert holding.wait(10)
    threading.Thread(target=waiter, daemon=True).start()
    assert not freed.wait(0.3)  # it waits while the other holds the turn
    stop.set()
    assert freed.wait(5)


@needs_backend
def test_only_the_cpu_batches_take_turns_and_the_scanner_is_put_back(tmp_path, monkeypatch):
    """Every CPU batch goes through ``_submit_worker``; the API pools (their workers wait on the network) submit
    directly, so they keep running at once. Outside a mobile scan the shared function is the desktop's."""
    import scan_html_folder as shf

    from glossarion_mobile.job_kinds import qa as qa_kind

    source = (SRC_DIR / "scan_html_folder.py").read_text(encoding="utf-8-sig")
    for batch in ("process_html_file_batch", "process_enhance_duplicate_batch", "process_deep_similarity_batch",
                  "process_comparison_batch_fast"):
        assert f"_submit_worker(executor, {batch}," in source, batch
    assert "trunc_pool.submit(_check_one, r)" in source and "_ai_pool.submit(_ai_check_one, r)" in source
    real = shf._submit_worker
    seen = {}

    class Runtime(_FakeRuntime):
        def run_bulk_qa_scan(self, folders, **kwargs):
            seen["during"] = shf._submit_worker
            return super().run_bulk_qa_scan(folders, **kwargs)

    qa_kind.scan(_FakeCtx(), [(str(tmp_path), None, False)], "quick-scan", qa=Runtime())
    assert seen["during"] is not real and shf._submit_worker is real
    monkeypatch.setattr(qa_kind, "SCAN_WORKER_TURNS", False)
    qa_kind.scan(_FakeCtx(), [(str(tmp_path), None, False)], "quick-scan", qa=Runtime())
    assert seen["during"] is real and shf._submit_worker is real


@needs_backend
def test_the_workers_and_the_interval_do_not_change_the_findings(phone, monkeypatch):
    """The way a scan ran before (Python's interval, all workers running) against the scan's own: same report."""
    from glossarion_mobile.job_kinds import qa as qa_kind

    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    runs = {}
    for name, workers, interval, turns in (("one", 1, 0.005, False), ("free", 4, 0.005, False),
                                           ("half", None, qa_kind.SCAN_SWITCH_INTERVAL, True),
                                           ("all", 8, qa_kind.SCAN_SWITCH_INTERVAL, True)):
        monkeypatch.setattr(qa_kind, "SCAN_SWITCH_INTERVAL", interval)
        monkeypatch.setattr(qa_kind, "SCAN_WORKER_TURNS", turns)
        folder = bench.make_book(phone / "out" / name / "Book", chapters=60, words=700)
        ctx = _scan(folder, _book_config(workers))
        assert any(f"Using {workers or 4} CPU cores" in line or "sequential" in line for line in ctx.lines)
        runs[name] = bench.report_findings(folder)
    assert runs["one"] == runs["free"] == runs["half"] == runs["all"]
    flagged = {row[0] for row in runs["one"] if row[1]}
    assert {"response_0039_ch0039.html", "response_0040_ch0040.html", "response_0041_ch0041.html"} <= flagged


# ---- performance budgets (relative: a slow runner does not fail them) ---------------------------------------


def _spin(units: int) -> None:
    for _ in range(units):
        total = 0
        for i in range(1000):
            total += i * i % 7


def _spin_units(target: float = 0.02) -> int:
    units = 1
    while True:
        started = time.perf_counter()
        _spin(units)
        took = time.perf_counter() - started
        if took >= target / 2:
            return max(1, int(units * target / took))
        units *= 2


def _ui_while(run, units: int) -> tuple:
    """Run ``run()`` on a thread while this thread acts like the UI loop: sleep a frame, how late it wakes; then a
    fixed burst of Python, how long it takes. Returns (wake-up lateness ms, burst ms, seconds)."""
    finished = threading.Event()
    box: dict = {}

    def target() -> None:
        try:
            box["value"] = run()
        except BaseException as exc:  # noqa: BLE001 - reported on the test's thread
            box["error"] = exc
        finally:
            finished.set()

    timer = getattr(getattr(ctypes, "windll", None), "winmm", None)  # Windows: 15.6 ms ticks would hide the wait
    if timer is not None:
        timer.timeBeginPeriod(1)
    late: list = []
    bursts: list = []
    started = time.perf_counter()
    threading.Thread(target=target, daemon=True).start()
    while not finished.is_set():
        slept = time.perf_counter()
        time.sleep(0.016)
        late.append((time.perf_counter() - slept - 0.016) * 1000)
        begun = time.perf_counter()
        _spin(units)
        bursts.append((time.perf_counter() - begun) * 1000)
    if timer is not None:
        timer.timeEndPeriod(1)
    if "error" in box:
        raise box["error"]
    return late, bursts, time.perf_counter() - started


@needs_backend
def test_budget_the_loop_wakes_sooner_and_the_scan_takes_no_longer(phone, monkeypatch):
    """The same real scan with 4 workers: the way it ran before (Python's interval, all of them running) against the
    job's own. The loop's timers must wake at least a quarter sooner (host: 5.5 to 1.5 ms, median) and the scan must
    not take half as long again (host: about the same)."""
    from glossarion_mobile.job_kinds import qa as qa_kind

    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    previous = sys.getswitchinterval()
    sys.setswitchinterval(0.005)
    fast = qa_kind.SCAN_SWITCH_INTERVAL
    late: dict = {"before": [], "after": []}
    seconds: dict = {"before": 0.0, "after": 0.0}
    try:
        _scan(bench.make_book(phone / "out" / "warm" / "Book", chapters=6, words=200), _book_config(2))  # imports
        for round_index in range(2):
            for name, interval, turns in (("before", 0.005, False), ("after", fast, True)):
                monkeypatch.setattr(qa_kind, "SCAN_SWITCH_INTERVAL", interval)
                monkeypatch.setattr(qa_kind, "SCAN_WORKER_TURNS", turns)
                # a book of its own each time: the scanner's text caches would make a repeated one free
                seed = 100 + round_index * 2 + (name == "after")
                folder = bench.make_book(phone / "out" / f"{name}{round_index}" / "Book", chapters=30, words=1500, seed=seed)
                woke, _took, spent = _ui_while(lambda: _scan(folder, _book_config(4), mode="aggressive"), 0)
                late[name].extend(woke)
                seconds[name] += spent
    finally:
        sys.setswitchinterval(previous)
    report = (f"late {statistics.median(late['before']):.2f} -> {statistics.median(late['after']):.2f} ms, "
              f"scan {seconds['before']:.1f} -> {seconds['after']:.1f} s")
    assert statistics.median(late["after"]) <= statistics.median(late["before"]) * 0.75, report
    assert seconds["after"] <= seconds["before"] * 1.5, report


@needs_backend
def test_budget_work_on_the_loop_is_not_stretched_by_every_worker(monkeypatch):
    """CPU-bound batches on 4 pool threads through the scanner's own ``_submit_worker`` (like its all-pairs stage), next
    to a fixed burst of Python on the "loop". With all of them running the burst shares the GIL with 4 threads (host:
    about 5 times its idle time); taking turns it shares it with one (about twice). The batches all finish."""
    from concurrent.futures import ThreadPoolExecutor

    import qa_scan_runtime as qsr
    import scan_html_folder as shf

    from glossarion_mobile.job_kinds import qa as qa_kind

    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    item = _spin_units(0.002)
    burst = _spin_units(0.02)

    def worker(_args, should_stop):
        for _ in range(10):
            if should_stop is not None and should_stop():
                break
            _spin(item)
        return True

    def bursts() -> list:
        def run() -> None:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [shf._submit_worker(pool, worker, index, lambda: False) for index in range(120)]
                assert all(future.result(timeout=300) for future in futures)

        return _ui_while(run, burst)[1]

    free = bursts()
    with qa_kind._scan_runtime({"ai_hunter_config": {"ai_hunter_max_workers": 4}}, qsr):
        turns = bursts()
    free += bursts()
    before, after = statistics.median(free), statistics.median(turns)
    assert after <= before * 0.75, f"burst {before:.1f} -> {after:.1f} ms"


@needs_flet
@needs_backend
def test_budget_log_lines_and_repaints_stay_a_small_part_of_a_scan(phone):
    """The cost the batch ruled out, kept ruled out: a job's lines reach a mounted LogConsole through the pump's
    fixed cadence, at few lines per chapter, and the pump takes a small part of the scan's time."""
    from glossarion_mobile.services.dispatcher import DEFAULT_INTERVAL, LogBuffer, UiDispatcher
    from glossarion_mobile.ui.components.log_console import LogConsole

    chapters = 60
    folder = bench.make_book(phone / "out" / "Book", chapters=chapters, words=800)

    async def scenario():
        _conn, session = _TB._fake_session("android")
        dispatcher = UiDispatcher(session.page)
        dispatcher.bind()
        spent = [0.0]
        real_flush = dispatcher.flush

        def flush() -> int:
            started = time.perf_counter()
            try:
                return real_flush()
            finally:
                spent[0] += time.perf_counter() - started

        dispatcher.flush = flush
        dispatcher.start()
        buffer = LogBuffer(5000, name="job:bench")
        session.page.controls.append(LogConsole(buffer=buffer, dispatcher=dispatcher, backlog=2000))
        session.page.update()
        ctx = ScanCtx(_book_config(2))
        ctx.log = buffer.append

        def run() -> None:
            from glossarion_mobile.job_kinds import qa as qa_kind

            assert qa_kind.scan(ctx, [(str(folder), None, False)], "quick-scan", disable_word_count=True)["ok"] is True

        started = time.perf_counter()
        await asyncio.get_running_loop().run_in_executor(None, run)
        total = time.perf_counter() - started
        await asyncio.sleep(DEFAULT_INTERVAL * 2)
        updates = dispatcher.updates_sent
        await dispatcher.stop()
        return total, spent[0], updates, buffer.last_seq

    total, pump, updates, lines = asyncio.run(scenario())
    assert lines <= 200 + chapters * 3, lines
    assert updates <= total / DEFAULT_INTERVAL + 5, (updates, total)
    assert pump <= total * 0.10, (pump, total)
