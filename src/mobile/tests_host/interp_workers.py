"""Workers for ``test_interpreters.py``, imported by name inside child interpreters (a test module would bring pytest)."""

from __future__ import annotations

import os
import subprocess
import sys
import time

CALLS = 0  # a child's own: counts the calls this interpreter has done


def echo(args):
    return args


def count_calls(_args=None):
    global CALLS
    CALLS += 1
    return CALLS


def where(_args=None):
    """Something that differs between interpreters and stays the same within one."""
    return id(sys.modules[__name__])


def get_env(names):
    return {name: os.environ.get(name) for name in names}


def echo_checked(args, _should_stop):
    return args


def similarity_cache_sizes(_args=None):
    import scan_html_folder

    outer = scan_html_folder.calculate_similarity_ratio
    return [outer.cache_info().currsize, outer.__wrapped__.cache_info().currsize]


def fill_similarity_cache(count):
    """Entries in the caches of ``calculate_similarity_ratio``, from nothing: the texts' lengths differ, so it is quick."""
    import scan_html_folder

    outer = scan_html_folder.calculate_similarity_ratio
    outer.cache_clear()
    outer.__wrapped__.cache_clear()
    for index in range(count):
        outer(f"a{index}", "b" * 40)
    return similarity_cache_sizes()


def environ_type(_args=None):
    return type(os.environ).__name__


def set_env(pairs):
    os.environ.update(pairs)
    return True


def recursion_limit(_args=None):
    return sys.getrecursionlimit()


def boom(message):
    raise ValueError(message)


class _NoRebuild(Exception):
    def __init__(self, first, second):  # pickles, but cannot be rebuilt from its args
        super().__init__(f"{first} {second}")


def boom_no_rebuild(_args=None):
    raise _NoRebuild("one", "two")


def speak(text):
    print(text)
    print("to stderr", file=sys.stderr)
    return len(text)


def items(args, should_stop):
    """``count`` items of ``seconds``, a file per item, checking for a stop between them like the scan workers."""
    count, seconds, folder = args
    done = 0
    for done in range(count):
        if should_stop():
            return {"stopped_at": done}
        if folder:
            with open(os.path.join(folder, f"item-{os.getpid()}-{id(args)}-{done}"), "w", encoding="utf-8"):
                pass
        time.sleep(seconds)
    return {"stopped_at": None, "done": done + 1}


def cpu(count):
    total = 0
    for i in range(count):
        total += i * i % 7
    return total


def loaded(names):
    return {name: sys.modules.get(name) is not None for name in names}


def blocked(names):
    return [name for name in names if name in sys.modules and sys.modules[name] is None]


def speak_until(path):
    """Prints a line, waits until someone read it (the caller creates ``path``), prints the rest."""
    print("한글 first")
    deadline = time.monotonic() + 30
    while not os.path.exists(path):
        if time.monotonic() > deadline:
            raise TimeoutError("nobody read the line while the call ran")
        time.sleep(0.01)
    sys.stdout.write("then a piece without its line end")
    return "done"


def spawn(_args=None):
    return subprocess.Popen([sys.executable, "-c", "pass"])


def clear_marker(_args=None):
    """A pool's ``after_call``."""
    os.environ["INTERP_AFTER_CALL"] = str(int(os.environ.get("INTERP_AFTER_CALL", "0")) + 1)


def fail_after_call():
    raise RuntimeError("the after_call hook failed")


def not_at_the_top_level():
    def inner():
        return None

    return inner
