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
            # Path to worker script
            worker_script = Path(__file__).parent / "chapter_extraction_worker.py"
            
            # Build command
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
                
                # Parse output based on prefix
                if line.startswith("[PROGRESS]"):
                    # Progress update
                    message = line[10:].strip()
                    if progress_callback:
                        progress_callback(message)
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
                    if not any(skip in line for skip in ["📑     Searching for", "📑     Found", "📑   ✓", "📑   ✗"]):
                        self._log(line)
            
            # Get any remaining output
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
            if self.process.returncode != 0 and not self.stop_requested:
                self._log(f"⚠️ Process exited with code {self.process.returncode}")
            
        except subprocess.TimeoutExpired:
            self._log("⚠️ Subprocess communication timeout")
            self._terminate_process()
            
        except Exception as e:
            self._log(f"❌ Subprocess error: {e}")
            self.result = {
                "success": False,
                "error": str(e)
            }
            
        finally:
            self.is_running = False
            self.process = None
            
            # Call completion callback
            if completion_callback:
                completion_callback(self.result)
    
    def stop_extraction(self):
        """Stop the extraction process"""
        if not self.is_running:
            return False
        
        self._log("🛑 Stopping chapter extraction...")
        self.stop_requested = True
        
        # Give it a moment to stop gracefully
        time.sleep(0.5)
        
        # Force terminate if still running
        if self.process:
            self._terminate_process()
        
        return True
    
    def _terminate_process(self):
        """Terminate the subprocess"""
        if self.process:
            try:
                self.process.terminate()
                # Give it a moment to terminate
                time.sleep(0.5)
                
                # Force kill if still running
                if self.process.poll() is None:
                    self.process.kill()
                    
                self._log("✅ Process terminated")
            except Exception as e:
                self._log(f"⚠️ Error terminating process: {e}")
    
    def _log(self, message):
        """Log a message using the callback if available"""
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
    import tkinter as tk
    from tkinter import filedialog
    
    def test_extraction():
        """Test the extraction manager"""
        
        # Create a simple GUI for testing
        root = tk.Tk()
        root.title("Chapter Extraction Test")
        root.geometry("800x600")
        
        # Text widget for logs
        text = tk.Text(root, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Log callback
        def log_message(msg):
            text.insert(tk.END, msg + "\n")
            text.see(tk.END)
            root.update_idletasks()
        
        # Progress callback
        def progress_update(msg):
            log_message(f"📊 Progress: {msg}")
        
        # Completion callback
        def extraction_complete(result):
            if result and result.get("success"):
                log_message(f"✅ Extraction completed!")
                log_message(f"   Chapters: {result.get('chapters', 0)}")
            else:
                log_message(f"❌ Extraction failed!")
        
        # Create manager
        manager = ChapterExtractionManager(log_callback=log_message)
        
        # File selection
        epub_path = filedialog.askopenfilename(
            title="Select EPUB file",
            filetypes=[("EPUB files", "*.epub"), ("All files", "*.*")]
        )
        
        if epub_path:
            output_dir = os.path.splitext(os.path.basename(epub_path))[0]
            
            # Start extraction
            manager.extract_chapters_async(
                epub_path,
                output_dir,
                extraction_mode="smart",
                progress_callback=progress_update,
                completion_callback=extraction_complete
            )
        
        # Button to stop
        stop_btn = tk.Button(
            root,
            text="Stop Extraction",
            command=lambda: manager.stop_extraction()
        )
        stop_btn.pack(pady=5)
        
        root.mainloop()
    
    # Run test
    test_extraction()