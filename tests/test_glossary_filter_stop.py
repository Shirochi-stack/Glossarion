import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import GlossaryManager as GM


@pytest.fixture(autouse=True)
def clear_stop(monkeypatch):
    for flag in ("TRANSLATION_CANCELLED", "GRACEFUL_STOP", "GRACEFUL_STOP_COMPLETED"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setattr(GM, "_stop_requested", False)
    monkeypatch.setattr(GM, "_get_stop_file_path", lambda: None)
    monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")


@pytest.mark.parametrize("flag", ["module", "file", "TRANSLATION_CANCELLED",
                                 "GRACEFUL_STOP", "GRACEFUL_STOP_COMPLETED"])
def test_filter_stops_before_html_parsing(monkeypatch, capsys, flag):
    if flag == "module":
        monkeypatch.setattr(GM, "_stop_requested", True)
    elif flag == "file":
        monkeypatch.setattr(GM, "_get_stop_file_path", lambda: __file__)
    else:
        monkeypatch.setenv(flag, "1")
    def unexpected_html(_):
        raise AssertionError("filter started after stop")
    monkeypatch.setattr(GM, "_html_soup", unexpected_html)
    assert GM._filter_text_for_glossary("Alice talks to Robert.") == ("", [])
    assert capsys.readouterr().out == ""


def test_stop_after_html_does_not_advance_filter(monkeypatch, capsys):
    def soup(_):
        monkeypatch.setenv("GRACEFUL_STOP", "1")
        return SimpleNamespace(get_text=lambda: "Alice talks to Robert.")
    monkeypatch.setattr(GM, "_html_soup", soup)
    assert GM._filter_text_for_glossary("source") == ("", [])
    assert "Step 2" not in capsys.readouterr().out


def test_sentence_worker_stops_inside_batch(monkeypatch):
    def sentences():
        yield "Alice talks to Robert."
        monkeypatch.setattr(GM, "_stop_requested", True)
        yield "Alice talks again."
    with pytest.raises(GM._GlossaryFilteringStopped):
        GM._check_sentence_batch_for_terms((sentences(), {"Alice"}))


def test_waiting_for_unfinished_future_is_interruptible(monkeypatch):
    future = Future()
    timer = threading.Timer(0.05, lambda: monkeypatch.setattr(GM, "_stop_requested", True))
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(GM._GlossaryFilteringStopped):
            list(GM._filter_completed([future]))
        assert time.monotonic() - started < 1.0
        assert not future.done()
    finally:
        timer.join()


def test_cancelled_executor_does_not_wait_and_terminates_owned_processes():
    calls = []
    process = SimpleNamespace(is_alive=lambda: True, terminate=lambda: calls.append("terminate"))
    executor = SimpleNamespace(_processes={1: process},
        shutdown=lambda **kwargs: calls.append(kwargs))
    with pytest.raises(GM._GlossaryFilteringStopped):
        with GM._filter_executor(lambda **_: executor):
            raise GM._GlossaryFilteringStopped()
    assert calls == [{"wait": False, "cancel_futures": True}, "terminate"]


@pytest.mark.parametrize("worker", ["scoring", "extraction"])
def test_parallel_workers_stop_during_sentence_iteration(monkeypatch, worker):
    def sentences():
        yield "Alice speaks to Robert beside the castle."
        monkeypatch.setenv("GRACEFUL_STOP_COMPLETED", "1")
        yield "Robert speaks to Alice beside the castle."
    with pytest.raises(GM._GlossaryFilteringStopped):
        if worker == "scoring":
            GM._score_sentence_batch(((0, sentences()), ["Alice"], "", [], False))
        else:
            GM._process_sentence_batch_for_extraction(
                (sentences(), 0, r"[A-Z][a-z]+", ([], [], set(), set())))


def test_normal_filter_still_returns_text(monkeypatch):
    monkeypatch.setenv("EXTRACTION_WORKERS", "1")
    monkeypatch.setenv("GLOSSARY_INCLUDE_ALL_CHARACTERS", "0")
    result = GM._filter_text_for_glossary(
        "Alice met Robert beside the castle. Robert spoke to Alice about the castle. " * 20,
        min_frequency=2, max_sentences=10)
    assert result[0]
    assert isinstance(result[1], (list, set, dict))
