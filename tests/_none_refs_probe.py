"""Temporary CI probe: where sys.getrefcount(None) drops (a C extension decref'ing None it does not own)."""
import collections
import gc
import sys

import pytest

_per_file = collections.Counter()
_per_test = []
_start = []
_file_mark = {}


def _file(item):
    return item.nodeid.split("::")[0]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    name = _file(item)
    if name not in _file_mark:
        gc.collect()
        _file_mark[name] = sys.getrefcount(None)
        if not _start:
            _start.append(_file_mark[name])
    before = sys.getrefcount(None)
    yield
    _per_test.append((sys.getrefcount(None) - before, item.nodeid))
    if nextitem is None or _file(nextitem) != name:
        gc.collect()
        _per_file[name] = sys.getrefcount(None) - _file_mark[name]


def pytest_terminal_summary(terminalreporter):
    gc.collect()
    w = terminalreporter.write_line
    w(f"NONEPROBE start={_start[:1]} end={sys.getrefcount(None)}")
    for name, delta in sorted(_per_file.items(), key=lambda kv: kv[1])[:30]:
        w(f"NONEPROBE file {delta:+d} {name}")
    for delta, nodeid in sorted(_per_test)[:30]:
        w(f"NONEPROBE test {delta:+d} {nodeid}")
    sys.stdout.flush()
