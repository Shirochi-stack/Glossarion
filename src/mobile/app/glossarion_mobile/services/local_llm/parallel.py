"""Two independent inference slots; never run two requests on the same llama context.

The second context is lazy and freed after its response. Turning parallel mode off
drains admitted responses before admitting another serial request. The original
single-context path remains unchanged until parallel mode is first enabled.
"""
from __future__ import annotations

import collections
import threading

from .server import ChatRun, LocalLLMError

PARALLEL_KEY = "local_llm_parallel_requests"


class RequestSlots:
    def __init__(self, owner):
        self.owner = owner
        self.secondary = None
        self.used = False
        self.active = {}
        self.queue = collections.deque()
        self.condition = threading.Condition()

    def _secondary(self):
        if self.secondary is None:
            from .service import LocalEngineService
            owner = self.owner
            self.secondary = LocalEngineService(
                prefs=owner.prefs, registry=owner.registry, downloads=owner.downloads,
                engine_factory=owner._engine_factory, active_job_model=owner.active_job_model,
                platform=owner.platform, clock=owner.clock, memory=owner.memory,
                stop_state=owner.stop_state, gpu_backend=owner.gpu_backend or "",
                watch=False, _parallel_child=True)
            self.secondary.subscribe(lambda _snap: owner._notify())
        return self.secondary

    def chat(self, model, body, should_stop):
        owner = self.owner
        if not self.used and not owner.parallel_requests():
            return owner._chat_serial(model, body, should_stop)
        self.used = True
        stem = owner.registry.resolve(model)
        if stem is None:
            return owner._chat_serial(model, body, should_stop)  # normal validation error
        pinned = owner._pinned_stem()
        if pinned and pinned.casefold() != stem.casefold():
            raise LocalLLMError(409, "engine_busy", f"The on-device engine is busy with {pinned} for a running job")
        arrived = owner._stops
        ticket = object()
        worker = None
        with self.condition:
            if len(self.queue) >= owner.gate.max_waiters:
                raise LocalLLMError(503, "server_busy", "Too many on-device requests are waiting", "server_error")
            self.queue.append(ticket)
            try:
                while worker is None:
                    if owner._stops != arrived or should_stop() or owner.stop_state():
                        raise LocalLLMError(499, "client_closed", "The request was cancelled while waiting")
                    parallel = owner.parallel_requests()
                    # Model changes wait for both contexts. An already-running serial
                    # request may predate the toggle and is represented by its gate.
                    stems = list(self.active.values())
                    if owner.gate.busy:
                        stems.append(owner._stem)
                    same_model = all(value == stem for value in stems)
                    draining = not parallel and any(w is not owner for w in self.active)
                    if self.queue[0] is ticket and same_model and not draining:
                        if owner not in self.active and not owner.gate.busy:
                            worker = owner
                        elif parallel and (self.secondary is None or self.secondary not in self.active):
                            worker = self._secondary()
                    if worker is None:
                        self.condition.wait(.1)
                self.queue.popleft()
                self.active[worker] = stem
                self.condition.notify_all()
            except BaseException:
                self.queue.remove(ticket)
                self.condition.notify_all()
                raise
        owner._notify()

        def release():
            try:
                if worker is not owner:
                    worker.unload()  # the extra model/context never sits idle in RAM
            finally:
                with self.condition:
                    self.active.pop(worker, None)
                    self.condition.notify_all()
                owner._notify()

        try:
            run = worker._chat_serial(model, body, lambda: owner._stops != arrived or should_stop())
        except BaseException:
            release()
            raise

        def close():
            try:
                run.close()
            finally:
                release()

        return ChatRun(run.stem, run.deltas, usage=lambda: run.usage,
                       finish=lambda: run.finish_reason, release=close)

    def changed(self):
        with self.condition:
            self.condition.notify_all()

    def busy_with(self, stem):
        return stem in self.active.values() or bool(self.secondary and self.secondary.busy_with(stem))

    def stop_generation(self):
        if self.secondary is not None:
            self.secondary.stop_generation()
        self.changed()

    def unload(self, force=False):
        return bool(self.secondary and self.secondary.unload(force=force))
