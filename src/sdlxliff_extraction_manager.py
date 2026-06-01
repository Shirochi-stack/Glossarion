"""SDLXLIFF extraction subprocess manager."""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


class SdlxliffExtractionManager:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.process = None
        self.result = None
        self.output_queue = queue.Queue()
        self.error_queue = queue.Queue()
        self.is_running = False
        self.stop_requested = False

    def extract_async(self, sdlxliff_path, output_dir, progress_callback=None, completion_callback=None):
        if self.is_running:
            self._log("SDLXLIFF extraction already in progress")
            return False
        self.is_running = True
        self.stop_requested = False
        self.result = None
        thread = threading.Thread(
            target=self._run_subprocess,
            args=(sdlxliff_path, output_dir, progress_callback, completion_callback),
            daemon=True,
        )
        thread.start()
        return True

    def _build_cmd(self, sdlxliff_path, output_dir):
        if getattr(sys, "frozen", False):
            return [sys.executable, "--run-sdlxliff-extraction", sdlxliff_path, output_dir]
        worker_script = Path(__file__).parent / "sdlxliff_extraction_worker.py"
        return [sys.executable, str(worker_script), sdlxliff_path, output_dir]

    def _run_subprocess(self, sdlxliff_path, output_dir, progress_callback, completion_callback):
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONLEGACYWINDOWSSTDIO"] = "0"
            try:
                cpu_count = multiprocessing.cpu_count()
                max_safe_workers = max(2, cpu_count - 2)
                workers = int(env.get("EXTRACTION_WORKERS", "2"))
                if workers > max_safe_workers:
                    self._log(f"Reducing SDLXLIFF workers from {workers} to {max_safe_workers} based on {cpu_count} CPUs")
                    workers = max_safe_workers
                env["EXTRACTION_WORKERS"] = str(max(1, workers))
            except Exception:
                env["EXTRACTION_WORKERS"] = "2"

            self._log("Starting SDLXLIFF extraction subprocess...")
            self._log(f"SDLXLIFF: {os.path.basename(sdlxliff_path)}")
            self._log(f"Output: {output_dir}")
            self.process = subprocess.Popen(
                self._build_cmd(sdlxliff_path, output_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                universal_newlines=True,
                env=env,
            )

            while True:
                if self.stop_requested:
                    self._terminate_process()
                    break
                if self.process.poll() is not None:
                    break
                line = self.process.stdout.readline()
                if not line:
                    continue
                self._handle_line(line.strip(), progress_callback)

            if not self.stop_requested:
                remaining_output, remaining_error = self.process.communicate(timeout=1)
                for line in (remaining_output or "").splitlines():
                    self._handle_line(line.strip(), progress_callback)
                if remaining_error and self.process.returncode not in (0, None):
                    for line in remaining_error.splitlines():
                        if line:
                            self._log(line)
                if self.process.returncode != 0 and self.result is None:
                    self._log(f"SDLXLIFF extraction exited with code {self.process.returncode}")
            else:
                try:
                    self.process.communicate(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
        except Exception as exc:
            if not self.stop_requested:
                self._log(f"SDLXLIFF subprocess error: {exc}")
            self.result = {"success": False, "error": str(exc)}
        finally:
            self.is_running = False
            process_ref = self.process
            self.process = None
            if process_ref and process_ref.poll() is None:
                try:
                    process_ref.terminate()
                    time.sleep(0.1)
                    if process_ref.poll() is None:
                        process_ref.kill()
                except Exception:
                    pass
            if self.result is None:
                self.result = {
                    "success": False,
                    "error": "SDLXLIFF extraction stopped by user" if self.stop_requested else "SDLXLIFF extraction ended unexpectedly",
                }
            if completion_callback:
                completion_callback(self.result)

    def _handle_line(self, line, progress_callback=None):
        if not line:
            return
        if line.startswith("[PROGRESS]"):
            message = line[10:].strip()
            if progress_callback:
                progress_callback(message)
            else:
                self._log(message)
        elif line.startswith("[INFO]"):
            self._log(line[6:].strip())
        elif line.startswith("[ERROR]"):
            message = line[7:].strip()
            self.error_queue.put(message)
            self._log(message)
        elif line.startswith("[RESULT]"):
            try:
                self.result = json.loads(line[8:].strip())
                if self.result.get("success"):
                    self._log(f"SDLXLIFF extraction completed: {self.result.get('chapters', 0)} segment(s)")
                else:
                    self._log(f"SDLXLIFF extraction failed: {self.result.get('error', 'Unknown error')}")
            except Exception as exc:
                self._log(f"Failed to parse SDLXLIFF extraction result: {exc}")
        elif not line.startswith("["):
            self._log(line)

    def stop_extraction(self):
        if not self.is_running:
            return False
        self.stop_requested = True
        self._log("Stopping SDLXLIFF extraction...")
        process_ref = self.process
        time.sleep(0.5)
        if process_ref:
            self._terminate_process_ref(process_ref)
        return True

    def _terminate_process(self):
        if self.process:
            self._terminate_process_ref(self.process)

    def _terminate_process_ref(self, process_ref):
        try:
            if process_ref and process_ref.poll() is None:
                process_ref.terminate()
                time.sleep(0.5)
                if process_ref.poll() is None:
                    process_ref.kill()
        except Exception as exc:
            self._log(f"Error terminating SDLXLIFF worker: {exc}")

    def _log(self, message):
        if self.stop_requested and "Stopping" not in str(message):
            return
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def is_extraction_running(self):
        return self.is_running
