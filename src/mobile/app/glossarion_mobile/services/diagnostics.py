"""Self-test runner shared by the ``/__selftest__`` deep link and the Diagnostics screen.

``diagnostics.selftest.run_selftest`` is blocking, so it runs on a worker
thread through ``UiDispatcher.run_in_thread``; the result lands in
``AppState.selftest_result`` on the loop thread. The self-test prints the
``GLOSSARION_SELFTEST PASS|FAIL <json>`` marker itself (CI contract with
``ci/android_smoke.sh``). Only one run at a time; a second request while one
is running is ignored (and logged), exactly like the U0 spike. ``before_run``
(the app passes "encryption keys installed") is awaited before each run.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, Optional

from glossarion_mobile.services.dispatcher import UiDispatcher
from glossarion_mobile.state.app_state import AppState

__all__ = ["SelfTestRunner", "summarize"]

log = logging.getLogger("glossarion.diagnostics")


def summarize(result: Optional[dict]) -> str:
    if not result:
        return "Not run yet."
    head = (
        f"{'PASS' if result.get('ok') else 'FAIL'} · {result.get('passed', 0)} passed · "
        f"{result.get('failed', 0)} failed · {result.get('skipped', 0)} skipped · {result.get('secs', 0)} s"
    )
    if result.get("error"):
        head += f"\n{result['error']}"
    return head


class SelfTestRunner:
    def __init__(
        self,
        dispatcher: UiDispatcher,
        state: AppState,
        *,
        before_run: Optional[Callable[[], Awaitable[Any]]] = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.state = state
        self.before_run = before_run
        self.runs = 0
        self.current_suite: Optional[str] = None  # the suite being run ("smoke", "e2e")

    @property
    def running(self) -> bool:
        return bool(self.state.selftest_running.value)

    async def run(self, suite: str = "smoke", *, source: str = "button") -> Optional[dict[str, Any]]:
        if self.running:
            log.info("self-test already running; ignored request from %s", source)
            return None
        from glossarion_mobile import runtime_bootstrap as rb
        from glossarion_mobile.diagnostics import selftest

        rb.emit_marker(rb.MARKER_SELFTEST_START, {"suite": suite, "source": source})
        self.current_suite = suite
        self.state.selftest_running.set(True)
        self.runs += 1
        t0 = time.monotonic()
        try:
            if self.before_run is not None:
                await self.before_run()
            # Attribute looked up at call time on the worker thread.
            result = await self.dispatcher.run_in_thread(lambda: selftest.run_selftest(suite), name="gl-selftest")
        except Exception as exc:
            log.exception("self-test crashed")
            result = {
                "suite": suite,
                "ok": False,
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "secs": round(time.monotonic() - t0, 2),
                "error": f"self-test crashed: {type(exc).__name__}: {exc}",
                "checks": [],
            }
        finally:
            self.state.selftest_running.set(False)
        result = dict(result)
        result["source"] = source
        self.state.selftest_result.set(result)
        log.info("self-test (%s): %s", source, summarize(result).splitlines()[0])
        return result
