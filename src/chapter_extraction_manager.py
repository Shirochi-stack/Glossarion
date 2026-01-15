#!/usr/bin/env python3
"""
Chapter Extraction Manager - Manages chapter extraction in subprocess to prevent GUI freezing
"""

import subprocess
import sys
import os
import json
import threading
import queue
import time
from pathlib import Path


class ChapterExtractionManager:
    """
    Manages chapter extraction in a separate process to prevent GUI freezing
    Similar to GlossaryManager but for chapter extraction
    """
    
    def __init__(self, log_callback=None):
        """
        Initialize the extraction manager
        
        Args:
            log_callback: Function to call with log messages (for GUI integration)
        """
        self.log_callback = log_callback
        self.process = None
        self.output_queue = queue.Queue()
        self.error_queue = queue.Queue()
        self.result = None
        self.is_running = False
        self.stop_requested = False
        
    def extract_chapters_async(self, epub_path, output_dir, extraction_mode="smart", 
                              progress_callback=None, completion_callback=None):
        """
        Start chapter extraction in a subprocess
        
        Args:
            epub_path: Path to EPUB file
            output_dir: Output directory for extracted content
            extraction_mode: Extraction mode (smart, comprehensive, full, enhanced)
            progress_callback: Function to call with progress updates
            completion_callback: Function to call when extraction completes
        """
        if self.is_running:
            self._log("⚠️ Chapter extraction already in progress")
            return False
        
        self.is_running = True
        self.stop_requested = False
        self.result = None
        
        # Start extraction in a thread that manages the subprocess
        thread = threading.Thread(
            target=self._run_extraction_subprocess,
            args=(epub_path, output_dir, extraction_mode, progress_callback, completion_callback),
            daemon=True
        )
        thread.start()
        
        return True
    
    def _run_extraction_subprocess(self, epub_path, output_dir, extraction_mode, 
                                   progress_callback, completion_callback):
        """
        Run the extraction subprocess and handle its output
        """
        try:
            # Build command differently for frozen vs dev mode
            if getattr(sys, 'frozen', False):
                # In a frozen one-file build, sys.executable is our GUI .exe, not Python.
                # Use an internal worker-mode flag handled by translator_gui.py to run the worker.
                cmd = [
                    sys.executable,
                    '--run-chapter-extraction',
                    epub_path,
                    output_dir,
                    extraction_mode
                ]
            else:
                # In dev mode, invoke the worker script with the Python interpreter
                base_dir = Path(__file__).parent
                worker_script = base_dir / "chapter_extraction_worker.py"
                cmd = [
                    sys.executable,
                    str(worker_script),
                    epub_path,
                    output_dir,
                    extraction_mode
                ]
            
            # Set environment to force UTF-8 encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSSTDIO'] = '0'  # Use new Windows console API
            
            # Cap worker count for subprocess mode based on CPU count
            # Very high worker counts can cause access violations on Windows
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            max_safe_workers = max(2, cpu_count - 2)  # Leave 2 cores for system
            
            current_workers = env.get('EXTRACTION_WORKERS', '2')
            try:
                workers = int(current_workers)
                # Cap based on CPU count for stability
                if workers > max_safe_workers:
                    self._log(f"⚠️ Reducing workers from {workers} to {max_safe_workers} (based on {cpu_count} CPUs)")
                    workers = max_safe_workers
                env['EXTRACTION_WORKERS'] = str(workers)
            except ValueError:
                env['EXTRACTION_WORKERS'] = '2'
            
            self._log(f"🚀 Starting chapter extraction subprocess...")
            self._log(f"📚 EPUB: {os.path.basename(epub_path)}")
            self._log(f"📂 Output: {output_dir}")
            self._log(f"⚙️ Mode: {extraction_mode}")
            
            # Start the subprocess with UTF-8 encoding
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace invalid chars instead of failing
                bufsize=1,
                universal_newlines=True,
                env=env  # Pass the environment with UTF-8 settings
            )
            
            # Read output in real-time
            while True:
                if self.stop_requested:
                    self._terminate_process()
                    break
                
                # Check if process is still running
                if self.process.poll() is not None:
                    break
                
                # Read stdout line by line with error handling
                try:
                    line = self.process.stdout.readline()
                    if not line:
                        continue
                    
                    line = line.strip()
                    if not line:
                        continue
                except UnicodeDecodeError as e:
                    self._log(f"⚠️ Encoding error reading output: {e}")
                    continue
                
                # Skip all processing if stop is requested to suppress logs
                if self.stop_requested:
                    continue
                
                # Parse output based on prefix
                if line.startswith("[PROGRESS]"):
                    # Progress update - format as progress bar
                    message = line[10:].strip()
                    
                    # Try to format as progress bar
                    import re
                    match = re.search(r'(\d+)/(\d+)', message)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        percent = int(100 * current / total)
                        
                        # Determine prefix and track last percent for this type
                        if "Scanning files" in message:
                            prefix = "📂 Scanning files"
                            prog_type = "scan"
                        elif "Extracting resources" in message:
                            prefix = "📦 Extracting resources"
                            prog_type = "extract"
                        elif "Processing chapters" in message:
                            prefix = "📚 Processing chapters"
                            prog_type = "process"
                        elif "Processed" in message:
                            prefix = "📊 Processing metadata"
                            prog_type = "processed"
                        else:
                            prefix = "📊 Progress"
                            prog_type = "other"
                        
                        # Only show progress every 10% or at completion
                        if not hasattr(self, '_last_percent'):
                            self._last_percent = {}
                        
                        last_percent = self._last_percent.get(prog_type, -1)
                        
                        # Show if: crossed a 10% threshold, or reached 100%
                        should_show = (percent // 10 > last_percent // 10) or (percent == 100)
                        
                        if should_show:
                            self._last_percent[prog_type] = percent
                            
                            # Create progress bar
                            bar_length = 20
                            filled = int(bar_length * current / total)
                            bar = '█' * filled + '░' * (bar_length - filled)
                            
                            formatted_message = f"{prefix}: [{bar}] {current}/{total} ({percent}%)"
                            
                            # Only log once - _log will call log_callback if it exists
                            self._log(formatted_message)
                    else:
                        # Not a progress message with numbers
                        self._log(f"📊 {message}")
                    
                elif line.startswith("[INFO]"):
                    # Information message
                    message = line[6:].strip()
                    self._log(f"ℹ️ {message}")
                    
                elif line.startswith("[ERROR]"):
                    # Error message
                    message = line[7:].strip()
                    self._log(f"❌ {message}")
                    self.error_queue.put(message)
                    
                elif line.startswith("[RESULT]"):
                    # Final result as JSON
                    try:
                        json_str = line[8:].strip()
                        self.result = json.loads(json_str)
                        
                        if self.result.get("success"):
                            self._log(f"✅ Extraction completed successfully!")
                            self._log(f"📚 Extracted {self.result.get('chapters', 0)} chapters")
                        else:
                            error = self.result.get("error", "Unknown error")
                            self._log(f"❌ Extraction failed: {error}")
                            
                    except json.JSONDecodeError as e:
                        self._log(f"⚠️ Failed to parse result: {e}")
                        
                elif line.startswith("["):
                    # Other prefixed messages - skip
                    pass
                else:
                    # Regular output - only log if not too verbose
                    if not any(skip in line for skip in ["📁     Searching for", "📁     Found", "📁   ✓", "📁   ✗"]):
                        self._log(line)
            
            # Get any remaining output - but only process if not stopped
            if not self.stop_requested:
                remaining_output, remaining_error = self.process.communicate(timeout=1)
                
                # Process any remaining output
                if remaining_output:
                    for line in remaining_output.strip().split('\n'):
                        if line and not line.startswith("["):
                            self._log(line)
                
                # Check for errors
                if remaining_error:
                    for line in remaining_error.strip().split('\n'):
                        if line:
                            self._log(f"⚠️ {line}")
                
                # Check final status
                if self.process.returncode != 0:
                    self._log(f"⚠️ Process exited with code {self.process.returncode}")
            else:
                # If stopped, just clean up without processing output
                try:
                    self.process.communicate(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass  # Ignore timeout when cleaning up
            
        except subprocess.TimeoutExpired:
            if not self.stop_requested:
                self._log("⚠️ Subprocess communication timeout")
            self._terminate_process()
            
        except Exception as e:
            # Only log errors if not stopping (unless it's a critical error)
            if not self.stop_requested or "Subprocess error" in str(e):
                self._log(f"❌ Subprocess error: {e}")
            self.result = {
                "success": False,
                "error": str(e) if not self.stop_requested else "Extraction stopped by user"
            }
            
        finally:
            self.is_running = False
            # Store process reference before clearing it in case termination is needed
            process_ref = self.process
            self.process = None
            
            # If process is still running, try to clean it up
            if process_ref and process_ref.poll() is None:
                try:
                    process_ref.terminate()
                    time.sleep(0.1)  # Brief wait
                    if process_ref.poll() is None:
                        process_ref.kill()
                except Exception:
                    pass  # Ignore cleanup errors in finally block
            
            # Ensure result is never None
            if self.result is None:
                if self.stop_requested:
                    self.result = {
                        "success": False,
                        "error": "Extraction stopped by user"
                    }
                else:
                    self.result = {
                        "success": False,
                        "error": "Extraction process ended unexpectedly"
                    }
            
            # Call completion callback
            if completion_callback:
                completion_callback(self.result)
    
    def stop_extraction(self):
        """Stop the extraction process"""
        if not self.is_running:
            return False
        
        # Set stop flag first to suppress subsequent logs
        self.stop_requested = True
        self._log("🛑 Stopping chapter extraction...")
        
        # Store process reference to avoid race condition
        process_ref = self.process
        
        # Give it a moment to stop gracefully
        time.sleep(0.5)
        
        # Force terminate if still running and process still exists
        if process_ref:
            self._terminate_process_ref(process_ref)
        
        return True
    
    def _terminate_process(self):
        """Terminate the subprocess using current process reference"""
        if self.process:
            self._terminate_process_ref(self.process)
    
    def _terminate_process_ref(self, process_ref):
        """Terminate a specific process reference"""
        if not process_ref:
            return
            
        try:
            # Check if process is still alive before attempting termination
            if process_ref.poll() is None:
                process_ref.terminate()
                # Give it a moment to terminate
                time.sleep(0.5)
                
                # Force kill if still running
                if process_ref.poll() is None:
                    process_ref.kill()
                    time.sleep(0.1)  # Brief wait after kill
                    
                # Only log termination if not stopping (user already knows they stopped it)
                if not self.stop_requested:
                    self._log("✅ Process terminated")
            else:
                # Only log if not stopping
                if not self.stop_requested:
                    self._log("✅ Process already terminated")
        except Exception as e:
            # Always log termination errors as they might indicate a problem
            self._log(f"⚠️ Error terminating process: {e}")
    
    def _log(self, message):
        """Log a message using the callback if available"""
        # Suppress logs when stop is requested (except for stop/termination messages)
        if self.stop_requested and not any(keyword in message for keyword in ["🛑", "✅ Process terminated", "❌ Subprocess error"]):
            return
            
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
    
    def is_extraction_running(self):
        """Check if extraction is currently running"""
        return self.is_running
    
    def get_result(self):
        """Get the extraction result if available"""
        return self.result


# Example usage
if __name__ == "__main__":
    from shutdown_utils import run_cli_main
    def _main():
        # Tkinter test code disabled - migrated to PySide6
        # import tkinter as tk
        # from tkinter import filedialog
        
        # def test_extraction():
        #     """Test the extraction manager"""
        #     
        #     # Create a simple GUI for testing
        #     root = tk.Tk()
        #     root.title("Chapter Extraction Test")
        #     root.geometry("800x600")
        #     
        #     # Text widget for logs
        #     text = tk.Text(root, wrap=tk.WORD)
        #     text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        #     
        #     # Log callback
        #     def log_message(msg):
        #         text.insert(tk.END, msg + "\n")
        #         text.see(tk.END)
        #         root.update_idletasks()
        #     
        #     # Progress callback
        #     def progress_update(msg):
        #         log_message(f"📊 Progress: {msg}")
        #     
        #     # Completion callback
        #     def extraction_complete(result):
        #         if result and result.get("success"):
        #             log_message(f"✅ Extraction completed!")
        #             log_message(f"   Chapters: {result.get('chapters', 0)}")
        #         else:
        #             log_message(f"❌ Extraction failed!")
        #     
        #     # Create manager
        #     manager = ChapterExtractionManager(log_callback=log_message)
        #     
        #     # File selection
        #     epub_path = filedialog.askopenfilename(
        #         title="Select EPUB file",
        #         filetypes=[("EPUB files", "*.epub"), ("All files", "*.*")]
        #     )
        #     
        #     if epub_path:
        #         output_dir = os.path.splitext(os.path.basename(epub_path))[0]
        #         
        #         # Start extraction
        #         manager.extract_chapters_async(
        #             epub_path,
        #             output_dir,
        #             extraction_mode="smart",
        #             progress_callback=progress_update,
        #             completion_callback=extraction_complete
        #         )
        #     
        #     # Button to stop
        #     stop_btn = tk.Button(
        #         root,
        #         text="Stop Extraction",
        #         command=lambda: manager.stop_extraction()
        #     )
        #     stop_btn.pack(pady=5)
        #     
        #     root.mainloop()
        # 
        # # Run test
        # test_extraction()
        pass  # Test code disabled for PySide6 migration
        return 0
    run_cli_main(_main)
