#!/usr/bin/env python3
"""
PDF Extraction Manager - Manages PDF extraction in a subprocess to prevent GUI freezing.
Follows the same pattern as PdfGenerationManager / ChapterExtractionManager.
"""

import subprocess
import sys
import os
import json
import threading
import time


class PdfExtractionManager:
    """Manages PDF extraction in a separate process to prevent GUI freezing."""

    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.process = None
        self.result = None
        self.is_running = False
        self.stop_requested = False

    def _log(self, message):
        if self.log_callback:
            try:
                self.log_callback(message)
            except Exception:
                pass
        else:
            print(message, flush=True)

    def extract_pdf_subprocess(self, config_path, completion_callback=None):
        """Start PDF extraction in a subprocess.

        Args:
            config_path: Path to the JSON config file with extraction parameters.
            completion_callback: Called with (success: bool, result: dict) when done.
        """
        if self.is_running:
            self._log("⚠️ PDF extraction already in progress")
            return False

        self.is_running = True
        self.stop_requested = False
        self.result = None

        thread = threading.Thread(
            target=self._run_extraction_subprocess,
            args=(config_path, completion_callback),
            daemon=True
        )
        thread.start()
        return True

    def extract_pdf_sync(self, config_path, timeout=300):
        """Run PDF extraction in subprocess and wait for result.

        Args:
            config_path: Path to the JSON config file.
            timeout: Maximum seconds to wait.

        Returns:
            dict with extraction results, or None on failure.
        """
        if self.is_running:
            self._log("⚠️ PDF extraction already in progress")
            return None

        completed = threading.Event()
        result_holder = [None]

        def on_complete(success, result):
            result_holder[0] = result
            completed.set()

        self.extract_pdf_subprocess(config_path, completion_callback=on_complete)

        if completed.wait(timeout=timeout):
            return result_holder[0]
        else:
            self._log("⚠️ PDF extraction timed out")
            self.stop()
            return None

    def _run_extraction_subprocess(self, config_path, completion_callback):
        """Run the PDF extraction subprocess and handle its output."""
        try:
            # Build command for frozen vs dev mode
            if getattr(sys, 'frozen', False):
                cmd = [sys.executable, '--run-pdf-extraction', config_path]
            else:
                worker_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    '_pdf_extraction_worker.py'
                )
                cmd = [sys.executable, worker_path, config_path]

            # Copy current environment
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUNBUFFERED'] = '1'

            self._log("🚀 Starting PDF extraction subprocess...")

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
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            # Heartbeat while waiting for subprocess
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
                    self._log(f"  ⏳ PDF extraction subprocess starting... ({elapsed:.0f}s elapsed)")

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
                    self._log(f"⚠️ PDF extraction subprocess exited with code {self.process.returncode}")
            else:
                try:
                    self.process.communicate(timeout=0.5)
                except (subprocess.TimeoutExpired, Exception):
                    pass

        except Exception as e:
            if not self.stop_requested:
                self._log(f"❌ PDF extraction subprocess error: {e}")
            self.result = {
                "success": False,
                "error": str(e) if not self.stop_requested else "PDF extraction stopped by user"
            }

        finally:
            self.is_running = False
            process_ref = self.process
            self.process = None

            if process_ref and process_ref.poll() is None:
                try:
                    process_ref.terminate()
                    process_ref.wait(timeout=3)
                except Exception:
                    try:
                        process_ref.kill()
                    except Exception:
                        pass

            if completion_callback:
                success = self.result.get("success", False) if self.result else False
                try:
                    completion_callback(success, self.result or {"success": False, "error": "No result received"})
                except Exception as e:
                    self._log(f"⚠️ Completion callback error: {e}")

    def stop(self):
        """Request stop of the PDF extraction subprocess."""
        self.stop_requested = True
        self._terminate_process()

    def _terminate_process(self):
        """Terminate the subprocess."""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
            except Exception:
                pass
