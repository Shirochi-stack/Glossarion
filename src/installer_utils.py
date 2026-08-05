"""Shared helpers for streaming dependency-installer output into the app log."""

from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional


_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_progress_line(value: str) -> str:
    line = _ANSI_RE.sub("", str(value or ""))
    line = _CONTROL_RE.sub("", line).strip()
    # Package-manager spinners frequently emit one punctuation-only line per
    # frame. Keep real progress bars/percentages, but suppress those frames.
    if line and not any(char.isalnum() for char in line) and len(line) < 12:
        return ""
    return line[:2000]


def run_logged_subprocess(
    command: List[str],
    *,
    log_fn: Optional[Callable[[str], None]],
    timeout: float,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    popen_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a command while forwarding merged stdout/stderr lines to ``log_fn``."""
    logger = log_fn or (lambda _message: None)
    kwargs: Dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
        "cwd": cwd,
        "env": env,
    }
    kwargs.update(popen_kwargs or {})
    process = subprocess.Popen(command, **kwargs)

    output_queue: queue.Queue = queue.Queue()
    reader_finished = object()

    def read_output() -> None:
        try:
            stream = process.stdout
            if stream is not None:
                for raw_line in iter(stream.readline, ""):
                    output_queue.put(raw_line)
        finally:
            output_queue.put(reader_finished)

    threading.Thread(target=read_output, daemon=True).start()

    tail = deque(maxlen=160)
    last_logged = ""
    reader_done = False
    timed_out = False
    deadline = time.monotonic() + max(0.1, float(timeout))

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 and process.poll() is None:
            timed_out = True
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

        try:
            item = output_queue.get(timeout=0.1)
        except queue.Empty:
            item = None

        if item is reader_finished:
            reader_done = True
        elif item is not None:
            # Universal-newline mode handles ordinary \r progress updates; the
            # split also covers tools that embed several updates in one chunk.
            for raw_line in re.split(r"[\r\n]+", str(item)):
                line = _clean_progress_line(raw_line)
                if not line:
                    continue
                tail.append(line)
                if line != last_logged:
                    logger(f"   ↳ {line}")
                    last_logged = line

        if process.poll() is not None and reader_done and output_queue.empty():
            break

    try:
        returncode = int(process.wait(timeout=1))
    except Exception:
        returncode = int(process.returncode if process.returncode is not None else -1)

    output = "\n".join(tail)
    return {
        "returncode": returncode,
        "output": output[-6000:] if output else "No installer output was returned.",
        "timed_out": timed_out,
    }
