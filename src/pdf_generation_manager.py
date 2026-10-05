#!/usr/bin/env python3
"""
PDF Generation Manager - Manages PDF generation in a subprocess to prevent GUI freezing.
Follows the same pattern as ChapterExtractionManager.

Where subprocesses are unavailable (Glossarion Mobile) the same _pdf_worker
runs in-process on the manager's thread; its protocol lines are parsed by
the same handler and stop() becomes cooperative: the caller then uses
wait() so the render thread (and PyMuPDF) is done before it moves on.
"""

import subprocess
import sys
import os
import json
import threading
import time
import traceback

import mobile_runtime
from shutdown_utils import subprocess_no_window_kwargs, terminate_subprocess_tree


class PdfGenerationManager:
    """Manages PDF generation in a separate process to prevent GUI freezing."""

    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.process = None
        self.result = None
        self.is_running = False
        self.stop_requested = False
        self._thread = None

    def _log(self, message):
        if self.log_callback:
            try:
                self.log_callback(message)
            except Exception:
                pass
        else:
            print(message, flush=True)

    def generate_pdf_async(self, config_path, completion_callback=None):
        """Start PDF generation in a subprocess.

        Args:
            config_path: Path to the JSON config file with all PDF parameters.
            completion_callback: Called with (success: bool, result: dict) when done.
        """
        if self.is_running:
            self._log("⚠️ PDF generation already in progress")
            return False

        self.is_running = True
        self.stop_requested = False
        self.result = None

        thread = threading.Thread(
            target=(self._run_pdf_subprocess if mobile_runtime.subprocesses_available()
                    else self._run_pdf_inprocess),
            args=(config_path, completion_callback),
            daemon=True
        )
        self._thread = thread
        thread.start()
        return True

    def wait(self, timeout=None):
        """Block until the generation thread has finished; True when it has.

        Used after stop() where the worker runs in-process (stop is cooperative
        there), so no render thread outlives the caller's compile step.
        """
        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _run_pdf_subprocess(self, config_path, completion_callback):
        """Run the PDF generation subprocess and handle its output."""
        try:
            # Build command for frozen vs dev mode
            if getattr(sys, 'frozen', False):
                cmd = [sys.executable, '--run-pdf-worker', config_path]
            else:
                cmd = [sys.executable, '_pdf_worker.py', config_path]

            # Copy current environment
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUNBUFFERED'] = '1'

            self._log("🚀 Starting PDF generation subprocess...")

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True,
                env=env,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                **subprocess_no_window_kwargs(),
            )

            # Heartbeat while waiting for subprocess (covers WeasyPrint import delay)
            _hb_stop = threading.Event()
            _hb_start = time.time()
            _got_first_output = threading.Event()
            def _startup_heartbeat():
                while not _hb_stop.is_set():
                    if _hb_stop.wait(3.0):
                        break
                    if _got_first_output.is_set():
                        break
                    elapsed = time.time() - _hb_start
                    self._log(f"  ⏳ PDF subprocess starting... ({elapsed:.0f}s elapsed)")
            _hb_thread = threading.Thread(target=_startup_heartbeat, daemon=True)
            _hb_thread.start()

            # Read output in real-time
            while True:
                if self.stop_requested:
                    self._terminate_process()
                    break

                if self.process.poll() is not None:
                    break

                try:
                    line = self.process.stdout.readline()
                    if not line:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                except UnicodeDecodeError:
                    continue

                if self.stop_requested:
                    continue

                # Stop startup heartbeat once we get real output
                _got_first_output.set()

                self._handle_worker_line(line)

            # Stop startup heartbeat
            _hb_stop.set()
            _hb_thread.join(timeout=1)

            # Read any remaining output
            if not self.stop_requested:
                try:
                    remaining_output, remaining_error = self.process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    remaining_output, remaining_error = "", ""

                if remaining_output:
                    for line in remaining_output.strip().split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("[PROGRESS]"):
                            self._log(line[10:].strip())
                        elif line.startswith("[INFO]"):
                            self._log(f"ℹ️ {line[6:].strip()}")
                        elif line.startswith("[ERROR]"):
                            self._log(f"❌ {line[7:].strip()}")
                        elif line.startswith("[RESULT]"):
                            if not self.result:
                                try:
                                    self.result = json.loads(line[8:].strip())
                                except Exception:
                                    pass
                        elif not line.startswith("["):
                            self._log(line)

                if remaining_error and not self.stop_requested:
                    for line in remaining_error.strip().split('\n'):
                        if line.strip():
                            self._log(f"⚠️ [stderr] {line.strip()}")

                if self.process.returncode != 0 and not self.stop_requested:
                    self._log(f"⚠️ PDF subprocess exited with code {self.process.returncode}")
            else:
                try:
                    self.process.communicate(timeout=0.5)
                except (subprocess.TimeoutExpired, Exception):
                    pass

        except Exception as e:
            if not self.stop_requested:
                self._log(f"❌ PDF subprocess error: {e}")
            self.result = {
                "success": False,
                "error": str(e) if not self.stop_requested else "PDF generation stopped by user"
            }

        finally:
            self.is_running = False
            process_ref = self.process
            self.process = None

            if process_ref and process_ref.poll() is None:
                try:
                    terminate_subprocess_tree(process_ref, kill=False, timeout=1)
                    process_ref.wait(timeout=3)
                except Exception:
                    try:
                        terminate_subprocess_tree(process_ref, kill=True, timeout=1)
                    except Exception:
                        pass

            self._notify_completion(completion_callback)

    def _handle_worker_line(self, line):
        """Dispatch one stripped worker protocol line ([PROGRESS]/[INFO]/[ERROR]/[RESULT]/plain)."""
        if line.startswith("[PROGRESS]"):
            message = line[10:].strip()
            self._log(message)
        elif line.startswith("[INFO]"):
            message = line[6:].strip()
            self._log(f"ℹ️ {message}")
        elif line.startswith("[ERROR]"):
            message = line[7:].strip()
            self._log(f"❌ {message}")
        elif line.startswith("[RESULT]"):
            try:
                json_str = line[8:].strip()
                self.result = json.loads(json_str)
            except json.JSONDecodeError as e:
                self._log(f"⚠️ Failed to parse result: {e}")
        elif not line.startswith("["):
            self._log(line)

    def _notify_completion(self, completion_callback):
        if completion_callback:
            success = self.result.get("success", False) if self.result else False
            try:
                completion_callback(success, self.result or {"success": False, "error": "No result received"})
            except Exception as e:
                self._log(f"⚠️ Completion callback error: {e}")

    def _handle_worker_output(self, text):
        """In-process emit callback: split into lines like the subprocess pipe reader."""
        for line in str(text).split('\n'):
            line = line.strip()
            if not line or self.stop_requested:
                continue
            self._handle_worker_line(line)

    def _run_pdf_inprocess(self, config_path, completion_callback):
        """Run _pdf_worker on this thread (platforms without subprocesses).

        Emits the same protocol lines as the subprocess, parsed by the same
        handler; stop() is cooperative (checked between render phases).
        """
        try:
            self._log("🚀 Starting PDF generation in-process (subprocesses unavailable)...")
            import _pdf_worker
            try:
                _pdf_worker.run_pdf_generation(
                    config_path,
                    emit=self._handle_worker_output,
                    should_stop=lambda: self.stop_requested,
                )
            except _pdf_worker.PdfGenerationStopped:
                pass
            except Exception as e:
                for line in _pdf_worker.failure_protocol_lines(e, traceback.format_exc()):
                    self._handle_worker_output(line)
            if self.stop_requested:
                self.result = {"success": False, "error": "PDF generation stopped by user"}
        except Exception as e:
            if not self.stop_requested:
                self._log(f"❌ PDF in-process error: {e}")
            self.result = {
                "success": False,
                "error": str(e) if not self.stop_requested else "PDF generation stopped by user"
            }
        finally:
            self.is_running = False
            self._notify_completion(completion_callback)

    def stop(self):
        """Request stop of the PDF generation subprocess (cooperative when in-process)."""
        self.stop_requested = True
        self._terminate_process()

    def _terminate_process(self):
        """Terminate the subprocess."""
        if self.process and self.process.poll() is None:
            try:
                terminate_subprocess_tree(self.process, kill=False, timeout=1)
            except Exception:
                pass
