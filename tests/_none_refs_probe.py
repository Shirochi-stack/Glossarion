"""Temporary CI probe: where sys.getrefcount(None) drops (a C extension decref'ing None it does not own)."""
import collections
import gc
import sys

import pytest

_per_file = collections.Counter()
_per_test = []
_start = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    gc.collect()
    before = sys.getrefcount(None)
    if not _start:
        _start.append(before)
    yield
    gc.collect()
    delta = sys.getrefcount(None) - before
    _per_file[item.nodeid.split("::")[0]] += delta
    _per_test.append((delta, item.nodeid))


def pytest_terminal_summary(terminalreporter):
    gc.collect()
    w = terminalreporter.write_line
    w(f"NONEPROBE start={_start[:1]} end={sys.getrefcount(None)}")
    for name, delta in sorted(_per_file.items(), key=lambda kv: kv[1])[:25]:
        w(f"NONEPROBE file {delta:+d} {name}")
    for delta, nodeid in sorted(_per_test)[:25]:
        w(f"NONEPROBE test {delta:+d} {nodeid}")
