"""Offline checks for Arena stream inactivity, independent of browser setup."""

import asyncio
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import autharena_proxy as arena


class ArenaStreamTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_reasoning_and_text_can_exceed_total_timeout(self):
        activity = asyncio.Event()
        received = []

        async def upstream():
            # Both reasoning and answer data keep the same response alive.
            for phase in ("thinking", "text"):
                for _ in range(12):
                    await asyncio.sleep(0.025)
                    received.append(phase)
                    activity.set()
            return "finished"

        result = await arena._run_stream_with_idle_timeout(upstream(), activity, 0.2)
        self.assertEqual(result, "finished")
        self.assertEqual(received, ["thinking"] * 12 + ["text"] * 12)

    async def test_idle_after_reasoning_reports_timeout_and_drains_upstream(self):
        activity = asyncio.Event()
        drained = asyncio.Event()
        never = asyncio.Event()

        async def upstream():
            try:
                activity.set()
                await never.wait()
            finally:
                # Returning before asynchronous cleanup would leak browser work.
                await asyncio.sleep(0.01)
                drained.set()

        with self.assertRaisesRegex(RuntimeError, "no upstream data"):
            await arena._run_stream_with_idle_timeout(upstream(), activity, 0.05)
        self.assertTrue(drained.is_set())

    async def test_missing_first_data_is_also_bounded(self):
        drained = asyncio.Event()

        async def upstream():
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()

        with self.assertRaisesRegex(RuntimeError, "no upstream data"):
            await arena._run_stream_with_idle_timeout(upstream(), asyncio.Event(), 0.05)
        self.assertTrue(drained.is_set())

    async def test_caller_cancellation_propagates_after_upstream_cleanup(self):
        started = asyncio.Event()
        drained = asyncio.Event()

        async def upstream():
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0.01)
                drained.set()

        task = asyncio.create_task(
            arena._run_stream_with_idle_timeout(upstream(), asyncio.Event(), 0.2)
        )
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(drained.is_set())

    async def test_none_allows_inactivity_until_upstream_completes(self):
        async def upstream():
            await asyncio.sleep(0.12)
            return "completed without a deadline"

        result = await arena._run_stream_with_idle_timeout(upstream(), asyncio.Event(), None)
        self.assertEqual(result, "completed without a deadline")

    async def test_upstream_failure_is_preserved(self):
        failure = ValueError("upstream protocol failure")

        async def upstream():
            await asyncio.sleep(0)
            raise failure

        with self.assertRaises(ValueError) as caught:
            await arena._run_stream_with_idle_timeout(upstream(), asyncio.Event(), 0.2)
        self.assertIs(caught.exception, failure)


if __name__ == "__main__":
    unittest.main()
