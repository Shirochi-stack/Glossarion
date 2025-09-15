"""
Process-safe glossary generation worker
========================================
This module provides a pickleable function for glossary generation
that can be run in a separate process using ProcessPoolExecutor.
"""

import os
import sys
import json
import time

def generate_glossary_in_process(output_dir, chapters_data, instructions, env_vars, log_queue=None):
    """
    Generate glossary in a separate process to avoid GIL blocking.
    
    Args:
        output_dir: Output directory path
        chapters_data: Serialized chapters data
        instructions: Glossary instructions
        env_vars: Environment variables to set
        log_queue: Queue to send logs back to main process
    
    Returns:
        Dictionary with glossary results or error info
    """
    import io
    import sys
    from io import StringIO
    
    # Capture ALL output - both stdout and stderr
    captured_logs = []
    
    class LogCapture:
        def __init__(self, queue=None):
            self.queue = queue
            self.buffer = ""
            
        def write(self, text):
            if text:
                # Buffer text and send complete lines
                self.buffer += text
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    if line:
                        captured_logs.append(line)
                        if self.queue:
                            try:
                                self.queue.put(line)
                            except:
                                pass
        
        def flush(self):
            if self.buffer:
                captured_logs.append(self.buffer)
                if self.queue:
                    try:
                        self.queue.put(self.buffer)
                    except:
                        pass
                self.buffer = ""
    
    try:
        # Redirect BOTH stdout and stderr to capture ALL output
        log_capture = LogCapture(log_queue)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = log_capture
        sys.stderr = log_capture
        
        # Set environment variables from parent process
        for key, value in env_vars.items():
            os.environ[key] = str(value)
        
        # Import here to avoid circular imports
        from TransateKRtoEN import GlossaryManager
        
        # Create glossary manager instance
        glossary_manager = GlossaryManager()
        
        # Generate glossary
        print(f"📑 Starting glossary generation in subprocess...")
        result = glossary_manager.save_glossary(output_dir, chapters_data, instructions)
        
        print(f"📑 Glossary generation completed")
        
        # Flush any remaining output
        log_capture.flush()
        
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        return {
            'success': True,
            'result': result,
            'pid': os.getpid(),
            'logs': captured_logs
        }
        
    except Exception as e:
        import traceback
        
        # Restore stdout and stderr if needed
        if 'old_stdout' in locals():
            sys.stdout = old_stdout
        if 'old_stderr' in locals():
            sys.stderr = old_stderr
        
        error_msg = f"Glossary generation error: {str(e)}"
        captured_logs.append(f"📑 ❌ {error_msg}")
        
        return {
            'success': False,
            'error': error_msg,
            'traceback': traceback.format_exc(),
            'pid': os.getpid(),
            'logs': captured_logs
        }

def generate_glossary_async(output_dir, chapters, instructions, extraction_workers=None):
    """
    Generate glossary asynchronously using ProcessPoolExecutor.
    
    This function completely bypasses the GIL by running in a separate process,
    ensuring the GUI remains fully responsive.
    """
    import concurrent.futures
    import multiprocessing
    
    # Determine worker count
    if extraction_workers is None:
        extraction_workers = int(os.getenv("EXTRACTION_WORKERS", "1"))
    
    if extraction_workers == 1:
        # Auto-detect optimal workers
        extraction_workers = min(multiprocessing.cpu_count() or 4, 4)
        print(f"📑 Auto-detected {extraction_workers} CPU cores for glossary generation")
    
    # Collect relevant environment variables
    env_vars = {}
    important_vars = [
        'EXTRACTION_WORKERS', 'GLOSSARY_MIN_FREQUENCY', 'GLOSSARY_MAX_NAMES',
        'GLOSSARY_MAX_TITLES', 'GLOSSARY_BATCH_SIZE', 'GLOSSARY_STRIP_HONORIFICS',
        'GLOSSARY_FUZZY_THRESHOLD', 'GLOSSARY_MAX_TEXT_SIZE', 'AUTO_GLOSSARY_PROMPT',
        'GLOSSARY_USE_SMART_FILTER', 'GLOSSARY_USE_LEGACY_CSV', 'GLOSSARY_PARALLEL_ENABLED',
        'GLOSSARY_FILTER_MODE', 'GLOSSARY_SKIP_FREQUENCY_CHECK', 'GLOSSARY_SKIP_ALL_VALIDATION',
        'MODEL', 'API_KEY', 'OPENAI_API_KEY', 'GEMINI_API_KEY', 'MAX_OUTPUT_TOKENS',
        'GLOSSARY_TEMPERATURE', 'MANUAL_GLOSSARY', 'ENABLE_AUTO_GLOSSARY'
    ]
    
    for var in important_vars:
        if var in os.environ:
            env_vars[var] = os.environ[var]
    
    # Use ProcessPoolExecutor for true parallelism
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        # Submit the task
        future = executor.submit(
            generate_glossary_in_process,
            output_dir,
            chapters,
            instructions,
            env_vars
        )
        
        # Return the future for the caller to monitor
        return future

def check_glossary_completion(future, timeout=0.01):
    """
    Check if glossary generation is complete without blocking.
    
    Args:
        future: Future object from generate_glossary_async
        timeout: Timeout in seconds for checking
    
    Returns:
        Tuple of (is_done, result_or_none)
    """
    try:
        if future.done():
            result = future.result(timeout=timeout)
            return True, result
        else:
            # Not done yet
            return False, None
    except concurrent.futures.TimeoutError:
        return False, None
    except Exception as e:
        # Error occurred
        return True, {'success': False, 'error': str(e)}