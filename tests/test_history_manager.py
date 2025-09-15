import os
import tempfile
import json
import shutil
import pytest

from history_manager import HistoryManager

@pytest.fixture
def temp_payloads_dir(tmp_path):
    d = tmp_path / "payloads"
    d.mkdir()
    return str(d)

def test_history_manager_empty_load_when_missing(temp_payloads_dir):
    hm = HistoryManager(temp_payloads_dir)
    assert hm.load_history() == []

def test_history_manager_save_and_load(temp_payloads_dir):
    hm = HistoryManager(temp_payloads_dir)
    data = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
    hm.save_history(data)
    loaded = hm.load_history()
    assert loaded == data

def test_history_manager_append_with_limit_reset(temp_payloads_dir):
    hm = HistoryManager(temp_payloads_dir)
    # Start with 1 exchange
    hm.save_history([{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}])
    # hist_limit = 1 => next append should reset
    history = hm.append_to_history("u2", "a2", hist_limit=1, reset_on_limit=True, rolling_window=False)
    assert len(history) == 2
    assert history[0]["content"] == "u2"

def test_history_manager_append_with_rolling_window(temp_payloads_dir):
    hm = HistoryManager(temp_payloads_dir)
    # Start with 2 exchanges
    hm.save_history([
        {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}
    ])
    # hist_limit = 2 => keep last 1 exchange and add new => 2 exchanges total
    history = hm.append_to_history("u3", "a3", hist_limit=2, reset_on_limit=False, rolling_window=True)
    assert len(history) == 4
    assert history[0]["content"] == "u2"
    assert history[-1]["content"] == "a3"

def test_history_manager_no_history_when_limit_zero(temp_payloads_dir):
    hm = HistoryManager(temp_payloads_dir)
    history = hm.append_to_history("u", "a", hist_limit=0)
    assert history == []
