"""Owner 2026-10-30, "Forced Truncation": an answer repeating the same text for longer than allowed is stopped and ends
with finish reason length, unless the source repeats like that (the QA scanner's check). On by default on device."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import forced_truncation as ft  # noqa: E402
import ollamapull  # noqa: E402

LINE_A = 'character,영주,Duke,,"A white-haired elderly man who appears to be around 70-100 years old"\n'
LINE_B = 'character,프레이,Frey,,"A human male adventurer who has been working as a Squire"\n'


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _watch(source="", seconds=10):
    clock = Clock()
    return ft.RepetitionWatch(source, seconds, clock=clock), clock


def _run(watch, clock, unit, seconds, step=0.5):
    """Feed ``unit`` every ``step`` seconds of fake time; True once the watch stops the answer."""
    for _ in range(int(seconds / step)):
        clock.now += step
        if watch.feed(unit):
            return True
    return False


def test_the_end_of_a_loop_is_found_with_its_length_and_count():
    assert ft.repeat_run((LINE_A + LINE_B) * 5) == (len(LINE_A + LINE_B), 5)
    assert ft.repeat_run("Some ordinary text that never comes back again, however long it is.") == (0, 0)
    assert ft.repeat_run("short") == (0, 0)
    period, count = ft.repeat_run("<p>\n" * 40)
    assert period == 4 and count == 40


def test_a_repeat_is_stopped_only_after_the_allowed_time():
    watch, clock = _watch(seconds=10)
    watch.feed("Opening text that is not a loop at all, and goes on for a little while. ")
    assert not _run(watch, clock, LINE_A + LINE_B, 8)  # looping, not for long enough yet
    assert _run(watch, clock, LINE_A + LINE_B, 5)
    assert watch.fired


def test_new_text_starts_the_time_again():
    watch, clock = _watch(seconds=10)
    assert not _run(watch, clock, LINE_A, 8)
    for index in range(4):  # the model got out of the loop
        clock.now += 0.5
        assert not watch.feed(f"Fresh sentence number {index} that has not been seen before in this answer at all. ")
    assert not _run(watch, clock, LINE_B, 8)  # another repeat: its own full time
    assert _run(watch, clock, LINE_B, 5)


def test_a_slow_trickle_of_tokens_of_a_loop_still_counts():
    watch, clock = _watch(seconds=3)
    text = (LINE_A + LINE_B) * 12
    fired = False
    for index in range(0, len(text), 7):
        clock.now += 0.1
        if watch.feed(text[index:index + 7]):
            fired = True
            break
    assert fired


def test_source_repetition_is_left_alone_until_the_answer_passes_it():
    source = "\n".join(["그는 끝없이 같은 말을 되풀이하고 있었다."] * 200)
    watch, clock = _watch(source, seconds=2)
    assert not _run(watch, clock, LINE_A, 20, step=0.5)  # 40 repeats: the source has 200 of them
    assert ft.source_repeats(source, 200) and not ft.source_repeats(source, 201)


def test_short_runs_in_the_source_count_and_markup_does_not():
    assert ft.source_repeats("ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 그는 웃었다.", 8)
    assert ft.source_repeats("아 아 아 아 아 아", 5)
    html = "".join(f"<p>문장 번호 {i} 입니다.</p>\n" for i in range(50))
    assert not ft.source_repeats(html, 10)
    assert not ft.source_repeats("", 3)


def test_trim_ends_on_a_whole_line_when_the_answer_stopped_mid_line():
    line = "character,Duke,The white-haired elderly man rules the city with great magical power.\n"
    assert ft.trim_repeats(line * 22 + "character,Duke,") == line


def test_trim_keeps_the_first_time_only():
    answer = "Chapter text.\n" + (LINE_A + LINE_B) * 9
    assert ft.trim_repeats(answer) == "Chapter text.\n" + LINE_A + LINE_B
    assert ft.trim_repeats("Nothing repeats in this short answer at all, not even once or twice.") == \
        "Nothing repeats in this short answer at all, not even once or twice."


class _Response:
    ok = True

    def __init__(self, lines):
        self.lines = lines
        self.closed = False

    def iter_lines(self):
        return iter(self.lines)

    def close(self):
        self.closed = True


def test_ollama_answer_ends_with_length_when_the_watch_fires(monkeypatch):
    import json

    chunk = json.dumps({"message": {"content": LINE_A}, "done": False}).encode()
    response = _Response([chunk] * 40)
    posted = {}

    def post(url, json=None, **kw):
        posted.update(json or {})
        return response

    monkeypatch.setattr(ollamapull.requests, "post", post)
    monkeypatch.setattr(ollamapull, "model_settings", lambda _n: {})
    monkeypatch.delenv("OLLAMAPULL_BASE_URL", raising=False)
    monkeypatch.setattr(ollamapull, "_stop_ban_server", lambda: None)
    clock = Clock()
    watch = ft.RepetitionWatch("", 2, clock=clock)
    real_feed = watch.feed

    def feed(text):
        clock.now += 0.5
        return real_feed(text)

    watch.feed = feed
    result = ollamapull.chat("ollamapull/m", [{"role": "user", "content": "x"}], stream=False, repeat_watch=watch)
    assert posted["stream"] is True  # watched answers always stream
    assert result["finish_reason"] == "length" and response.closed and watch.fired


def test_client_watches_only_when_the_toggle_is_on(monkeypatch):
    from unified_api_client import UnifiedClient

    client = UnifiedClient.__new__(UnifiedClient)
    monkeypatch.delenv("FORCED_TRUNCATION", raising=False)
    assert client._repeat_watch([{"role": "user", "content": "x"}]) is None
    monkeypatch.setenv("FORCED_TRUNCATION", "1")
    monkeypatch.setenv("FORCED_TRUNCATION_SECONDS", "7")
    watch = client._repeat_watch([{"role": "system", "content": "no"}, {"role": "user", "content": "source"}])
    assert watch.seconds == 7 and watch.source == "source"
    monkeypatch.setattr(UnifiedClient, "_is_on_device_endpoint", lambda self, url: True)
    assert client._repeat_watch([], "http://127.0.0.1:1/v1") is None  # its engine watches itself


def _stopped_client():
    from unified_api_client import UnifiedClient

    client = UnifiedClient.__new__(UnifiedClient)
    client.model = "ollamapull/m"
    client._get_thread_local_client = lambda tls=__import__("types").SimpleNamespace(_forced_truncation=True): tls
    return client


@pytest.mark.parametrize("env, context, finish", [
    ("1", "glossary", "stop"),  # the toggle: a stopped glossary request is complete
    ("1", "glossary_refinement", "stop"),
    ("0", "glossary", "length"),  # off by default
    ("1", "translation", "length"),  # only glossary requests
])
def test_mark_glossary_forced_truncation_as_complete(monkeypatch, env, context, finish):
    monkeypatch.setenv("MARK_GLOSSARY_FORCED_TRUNCATION_COMPLETE", env)
    answer = LINE_A + LINE_B * 9
    content, reason = _stopped_client()._trim_forced_truncation(answer, "length", context)
    assert reason == finish and content == LINE_A + LINE_B


def test_a_real_length_finish_stays_truncated_with_the_toggle_on(monkeypatch):
    monkeypatch.setenv("MARK_GLOSSARY_FORCED_TRUNCATION_COMPLETE", "1")
    client = _stopped_client()
    client._get_thread_local_client()._forced_truncation = False
    assert client._trim_forced_truncation("A long answer cut off", "length", "glossary") == ("A long answer cut off", "length")
    # on device the engine chose the finish reason (its own switch): the client only takes the copies off
    client.model = "local/qwen"
    assert client._trim_forced_truncation("A long answer cut off", "length", "glossary") == ("A long answer cut off", "length")
    assert client._trim_forced_truncation(LINE_A * 9, "length", "glossary") == (LINE_A, "length")
    assert client._trim_forced_truncation(LINE_A * 9, "stop", "glossary") == (LINE_A, "stop")


def test_glossary_runs_read_stop_as_complete():
    source = (SRC / "extract_glossary_from_epub.py").read_text(encoding="utf-8")
    assert "def _glossary_issue_from_finish_reason" in source


def test_a_length_finish_marks_the_chapter_truncated():
    """The Progress Manager's mark: this path is what the finish reason has to reach."""
    source = (SRC / "TransateKRtoEN.py").read_text(encoding="utf-8")
    at = source.index('if finish_reason in ["length", "max_tokens", "max_length"')
    assert 'qa_issues_found=["TRUNCATED"]' in source[at:at + 1200] and 'status="qa_failed"' in source[at:at + 1200]


def test_the_setting_is_wired_through_desktop_schema_and_specs():
    other = (SRC / "other_settings.py").read_text(encoding="utf-8")
    assert '_create_styled_checkbox("Forced Truncation")' in other and "forced_spin.setRange(1, 600)" in other
    run_env = (SRC / "run_env.py").read_text(encoding="utf-8")
    assert run_env.count("'FORCED_TRUNCATION'") == 3 and run_env.count("'FORCED_TRUNCATION_SECONDS'") == 3
    import settings_schema_data as data
    assert data.SETTINGS["forced_truncation_enabled"]["default"] is False
    assert data.SETTINGS["forced_truncation_seconds"]["default"] == 3
    for spec in SRC.glob("*.spec"):
        text = spec.read_text(encoding="utf-8")
        assert "('forced_truncation.py', '.')" in text and "'forced_truncation'," in text, spec.name
    assert '"forced_truncation"' in (SRC / "mobile" / "backend_manifest.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize("seconds, expected", [(0, 1.0), ("abc", 3.0), (None, 3.0), (15, 15.0)])
def test_seconds_are_read_safely(seconds, expected):
    assert ft.RepetitionWatch("", seconds).seconds == expected
