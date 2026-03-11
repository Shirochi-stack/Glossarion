#!/usr/bin/env python3
"""
PDF Generation Manager - Manages PDF generation in a subprocess to prevent GUI freezing.
Follows the same pattern as ChapterExtractionManager.
"""

import subprocess
import sys
import os
import json
import threading
import time


class PdfGenerationManager:
    """Manages PDF generation in a separate process to prevent GUI freezing."""

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
            target=self._run_pdf_subprocess,
            args=(config_path, completion_callback),
            daemon=True
        )
        thread.start()
        return True

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
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

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
                    # Only log non-trivial stderr
                    for line in remaining_error.strip().split('\n'):
                        if line and 'WARNING' not in line.upper():
                            self._log(f"⚠️ {line}")

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
        """Request stop of the PDF generation subprocess."""
        self.stop_requested = True
        self._terminate_process()

    def _terminate_process(self):
        """Terminate the subprocess."""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
            except Exception:
                pass
