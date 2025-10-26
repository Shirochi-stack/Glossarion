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
        
        # ALSO capture logging module output
        import logging
        
        # Create a custom logging handler that writes to our log_capture
        class QueueLogHandler(logging.Handler):
            def __init__(self, log_capture):
                super().__init__()
                self.log_capture = log_capture
                
            def emit(self, record):
                try:
                    msg = self.format(record)
                    self.log_capture.write(msg + '\n')
                except Exception:
                    pass
        
        # Add our handler to the root logger and all existing loggers
        queue_handler = QueueLogHandler(log_capture)
        queue_handler.setLevel(logging.DEBUG)
        
        # Store original handlers so we can restore them later
        original_handlers = {}
        
        # Redirect root logger
        root_logger = logging.getLogger()
        original_handlers['root'] = root_logger.handlers[:]
        root_logger.handlers = [queue_handler]
        
        # Redirect unified_api_client logger specifically
        api_logger = logging.getLogger('unified_api_client')
        original_handlers['unified_api_client'] = api_logger.handlers[:]
        api_logger.handlers = [queue_handler]
        api_logger.setLevel(logging.DEBUG)
        api_logger.propagate = False  # Prevent propagation to root logger (avoid duplicates)
        
        # Set environment variables from parent process
        for key, value in env_vars.items():
            os.environ[key] = str(value)
            if key == 'GLOSSARY_MAX_SENTENCES':
                print(f"🔍 [DEBUG] Worker process setting GLOSSARY_MAX_SENTENCES to: '{value}'")
        
        # Import here to avoid circular imports
        import GlossaryManager
        
        # Create a log callback that uses the queue
        def queue_log_callback(msg):
            if log_queue:
                try:
                    log_queue.put(str(msg))
                except:
                    pass
            # Also capture locally
            captured_logs.append(str(msg))
        
        # Generate glossary using module function (not a class)
        print(f"📑 Starting glossary generation in subprocess...")
        result = GlossaryManager.save_glossary(output_dir, chapters_data, instructions, log_callback=queue_log_callback)
        
        print(f"📑 Glossary generation completed")
        
        # Flush any remaining output
        log_capture.flush()
        
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Restore logging handlers
        if 'original_handlers' in locals():
            root_logger = logging.getLogger()
            root_logger.handlers = original_handlers.get('root', [])
            api_logger = logging.getLogger('unified_api_client')
            api_logger.handlers = original_handlers.get('unified_api_client', [])
        
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
    
    # Ensure freeze support for Windows frozen executables
    try:
        multiprocessing.freeze_support()
    except Exception:
        pass
    
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
        'GLOSSARY_FUZZY_THRESHOLD', 'GLOSSARY_MAX_TEXT_SIZE', 'GLOSSARY_MAX_SENTENCES',
        'AUTO_GLOSSARY_PROMPT', 'GLOSSARY_USE_SMART_FILTER', 'GLOSSARY_USE_LEGACY_CSV',
        'GLOSSARY_PARALLEL_ENABLED', 'GLOSSARY_FILTER_MODE', 'GLOSSARY_SKIP_FREQUENCY_CHECK',
        'GLOSSARY_SKIP_ALL_VALIDATION', 'MODEL', 'API_KEY', 'OPENAI_API_KEY', 'GEMINI_API_KEY',
        'MAX_OUTPUT_TOKENS', 'GLOSSARY_TEMPERATURE', 'MANUAL_GLOSSARY', 'ENABLE_AUTO_GLOSSARY'
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