"""stop_control: the desktop run-start reset and Stop protocol, shared by TranslatorGUI and the mobile JobService.

Shared GUI-free core (Glossarion mobile rewrite, milestone U3). The statements below
moved verbatim out of ``TranslatorGUI`` (``translator_gui.py`` @ 1719fb59):

* run-start resets: ``run_translation_thread`` (stop env 29150-29155, run id
  29157-29162, client cancellation 29254-29286, stop file 29294-29304) and
  ``run_glossary_extraction_thread`` (32571-32579, 32596-32606), split into
  ``reset_stop_env`` / ``make_run_id`` / ``clear_client_cancellation`` /
  ``prepare_glossary_stop_file`` so the desktop calls each part at its original
  position; ``reset_for_new_run`` runs them in desktop order for a mobile job;
* the Stop protocol of ``stop_translation`` (34865-34876 force-stop flags,
  34924-35164 env flags -> latch -> stop files -> module flags -> background
  cleanup), as ``apply_force_stop_flags`` and ``request_stop``. The widget code
  (button text, progress bars, message boxes) and the double-click detection stay
  in ``stop_translation``; ``register_stop_click`` / ``StopClickTracker`` hold the
  click-window rule both front ends use;
* the glossary extractor's stop callback (``_extract_glossary_from_text_file``),
  ``make_glossary_stop_callback``;
* the GUI-free half of ``_reset_api_watchdog_progress`` (counters + watchdog
  files), ``reset_api_watchdog``;
* the non-widget tail of ``stop_translation`` (EPUB converter stop flag, HTTP logger
  silencing on a graceful stop, the stop-mode log line) as ``stop_epub_converter`` /
  ``announce_stop``, and ``run_translation_thread``'s wait for the previous stop's
  cleanup thread as ``wait_for_stop_cleanup`` (U3 fix pass; both front ends call them).

Flag order matters (see the race comment kept in ``request_stop``): the environment
mode flags are published before the ``set_stop_requested`` latch, so a graceful first
click is never mistaken for an immediate stop by a worker polling both.

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import os
import threading
import time

from mobile_runtime import subprocesses_available

__all__ = [
    "HTTP_LOGGER_NAMES",
    "STOP_CLICK_WINDOW",
    "StopClickTracker",
    "announce_stop",
    "apply_force_stop_flags",
    "clear_client_cancellation",
    "clear_module_stop_flags",
    "kill_helper_subprocesses",
    "make_glossary_stop_callback",
    "make_run_id",
    "prepare_glossary_stop_file",
    "register_stop_click",
    "request_stop",
    "reset_api_watchdog",
    "reset_for_new_run",
    "reset_stop_env",
    "silence_http_loggers",
    "stop_epub_converter",
    "wait_for_stop_cleanup",
]

#: Two Stop clicks closer than this (seconds) force an immediate stop.
STOP_CLICK_WINDOW = 1.0

#: HTTP client loggers a graceful stop silences (stop_translation).
HTTP_LOGGER_NAMES = ['httpx', 'openai', 'google', 'google.api_core', 'google.generativeai', 'urllib3']


# ---------------------------------------------------------------------------
# Run start
# ---------------------------------------------------------------------------


def reset_stop_env(kind='translation'):
    """Clear the previous run's stop state from os.environ (run start, before the run id).

    ``kind='translation'``: run_translation_thread; ``kind='glossary'``:
    run_glossary_extraction_thread (which keeps TRANSLATION_CANCELLED / WAIT_FOR_CHUNKS).
    """
    if kind == 'glossary':
        os.environ['GRACEFUL_STOP'] = '0'  # Reset graceful stop env var
        os.environ['GRACEFUL_STOP_COMPLETED'] = '0'  # Reset completion flag
        os.environ['GRACEFUL_STOP_API_ACTIVE'] = '0'  # Reset API active flag
        return
    os.environ.pop('TRANSLATION_CANCELLED', None)  # Clear hard-stop state from previous run
    os.environ.pop('TRANSLATION_ANTI_DUPLICATE_LOGGED', None)
    os.environ['GRACEFUL_STOP'] = '0'  # Reset graceful stop env var
    os.environ['GRACEFUL_STOP_COMPLETED'] = '0'  # Reset completion flag
    os.environ['WAIT_FOR_CHUNKS'] = '0'  # Reset graceful batch-stop mode
    os.environ['GRACEFUL_STOP_API_ACTIVE'] = '0'  # Reset API active flag


def make_run_id(kind='translation'):
    """A new GLOSSARION_RUN_ID value (transport logs of stale runs are suppressed by it).

    Translation: ten hex digits (``str(int(time.time()))`` fallback); glossary:
    ``glossary-<hex10>`` (``glossary-<ms>`` fallback). The caller exports it.
    """
    if kind == 'glossary':
        try:
            import uuid as _uuid
            return f"glossary-{_uuid.uuid4().hex[:10]}"
        except Exception:
            return f"glossary-{int(time.time() * 1000)}"
    try:
        import uuid as _uuid
        return _uuid.uuid4().hex[:10]
    except Exception:
        return str(int(time.time()))


def clear_module_stop_flags(kind='translation'):
    """Reset the backend module's own stop flag for a new run.

    Glossary: ``extract_glossary_from_epub.set_stop_flag(False)`` exactly like
    run_glossary_extraction_thread. Translation: ``TransateKRtoEN.set_stop_flag(False)``
    when the module is loaded (the desktop calls the lazily loaded ``translation_stop_flag``).
    """
    if kind == 'glossary':
        # IMPORTANT: Also reset the module's internal stop flag
        try:
            import extract_glossary_from_epub
            extract_glossary_from_epub.set_stop_flag(False)
        except:
            pass
        return
    try:
        import sys as _sys
        module = _sys.modules.get('TransateKRtoEN')
        if module is not None and hasattr(module, 'set_stop_flag'):
            module.set_stop_flag(False)
    except Exception:
        pass


def clear_client_cancellation():
    """Close the previous run's lingering streams and reset the client's cancellation flag."""
    # CRITICAL: Before starting a new run, hard-cancel any lingering streams
    # from the previous run.  A graceful stop does NOT call hard_cancel_all(),
    # so glossary/translation streams can still be alive and printing output.
    # Close them NOW before we reset the stop flags (otherwise they'd keep running
    # after we clear _global_cancelled and leak into the new translation phase).
    try:
        import unified_api_client
        # Close lingering streams/sessions from any previous run
        # (hard_cancel_all also resets the watchdog internally)
        if hasattr(unified_api_client, 'hard_cancel_all'):
            try:
                # Use the provider-aware wrapper. The class method does not
                # own Antigravity's adapter-level HTTP response registry.
                unified_api_client.hard_cancel_all()
            except Exception:
                pass
        elif hasattr(unified_api_client, 'UnifiedClient'):
            try:
                unified_api_client.UnifiedClient.hard_cancel_all()
            except Exception:
                pass
    except Exception:
        pass

    # Reset unified_api_client global cancellation (streaming stop) for new runs
    try:
        import unified_api_client
        if hasattr(unified_api_client, 'set_stop_flag'):
            unified_api_client.set_stop_flag(False)
        elif hasattr(unified_api_client, 'UnifiedClient'):
            unified_api_client.UnifiedClient.set_global_cancellation(False)
    except Exception:
        pass


def prepare_glossary_stop_file():
    """Export GLOSSARY_STOP_FILE (shared with glossary/translation workers) and delete a stale one."""
    try:
        import tempfile, os as _os
        stop_file = os.environ.get('GLOSSARY_STOP_FILE') or _os.path.join(
            tempfile.gettempdir(), f"glossarion_glossary_stop_{_os.getpid()}.flag"
        )
        os.environ['GLOSSARY_STOP_FILE'] = stop_file
        if _os.path.exists(stop_file):
            _os.remove(stop_file)
    except Exception:
        pass
    return os.environ.get('GLOSSARY_STOP_FILE')


def wait_for_stop_cleanup(cleanup_thread, log=print, timeout=3.0):
    """Wait for the previous immediate stop's cleanup thread before a new run starts.

    ``run_translation_thread``'s preflight (moved verbatim): the cleanup closes HTTP
    transports on a helper thread, so a new run must not clear the cancellation state
    while it can still close transports underneath it. Returns False when the thread
    is still alive after *timeout* seconds (the new run must not start yet).
    """
    # Immediate Stop performs connection teardown on a helper thread. Do
    # not clear cancellation state for a new run while that older cleanup
    # can still close transports underneath it.
    previous_stop_cleanup = cleanup_thread
    if previous_stop_cleanup is not None and previous_stop_cleanup.is_alive():
        log("⏳ Waiting for the previous translation HTTP session to close...")
        previous_stop_cleanup.join(timeout=timeout)
        if previous_stop_cleanup.is_alive():
            log("⏹️ Previous translation is still stopping; try Start again shortly.")
            return False
    return True


def reset_for_new_run(kind='translation'):
    """All run-start resets in desktop order; returns the new GLOSSARION_RUN_ID.

    The desktop calls the parts at their own positions inside run_translation_thread /
    run_glossary_extraction_thread (owner attributes are reset in between); a mobile job
    (one fresh HeadlessOwner per run) calls this once on the job thread.
    """
    reset_stop_env(kind)
    run_id = make_run_id(kind)
    os.environ['GLOSSARION_RUN_ID'] = run_id
    clear_module_stop_flags(kind)
    if kind != 'glossary':
        clear_client_cancellation()
    prepare_glossary_stop_file()
    return run_id


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


def register_stop_click(times, now, window=STOP_CLICK_WINDOW):
    """Add a Stop click at *now* to *times* and return the clicks still inside *window*.

    ``stop_translation``: ``self._stop_click_times = register_stop_click(self._stop_click_times, t)``;
    two clicks left means "force stop".
    """
    # Add current click
    times.append(now)
    # Remove clicks older than 1 second
    return [t for t in times if now - t < window]


class StopClickTracker:
    """Double-click (force stop) detection for a Stop control.

    ``register()`` returns True on the second click inside the window and then
    forgets the clicks, like the desktop Stop button.
    """

    def __init__(self, window=STOP_CLICK_WINDOW, clicks=2):
        self.window = float(window)
        self.clicks = int(clicks)
        self.times = []

    def register(self, now=None):
        current_time = time.time() if now is None else now
        self.times = register_stop_click(self.times, current_time, self.window)
        force = len(self.times) >= self.clicks
        if force:
            self.times = []
        return force

    def reset(self):
        self.times = []


def apply_force_stop_flags():
    """Double-click force stop: hard-abort env flags and client cancellation, set at once."""
    os.environ['TRANSLATION_CANCELLED'] = '1'
    os.environ['GRACEFUL_STOP'] = '0'
    os.environ['GRACEFUL_STOP_COMPLETED'] = '0'
    os.environ['WAIT_FOR_CHUNKS'] = '0'
    try:
        import unified_api_client
        if hasattr(unified_api_client, 'set_stop_flag'):
            unified_api_client.set_stop_flag(True)
        if hasattr(unified_api_client, 'UnifiedClient'):
            unified_api_client.UnifiedClient._global_cancelled = True
    except Exception:
        pass


def reset_api_watchdog(clear_stale_external_files=True):
    """Reset the API watchdog counters and (on Stop) delete every cross-process watchdog file."""
    # 1) Reset watchdog counters in unified_api_client (best effort)
    try:
        import unified_api_client
        if hasattr(unified_api_client, '_api_watchdog_reset'):
            unified_api_client._api_watchdog_reset()
    except Exception:
        pass

    # 2) Clear cross-process watchdog files so aggregation can't keep the bar "busy".
    # On Stop clicks we intentionally delete ALL watchdog files (including .tmp), even if the
    # originating process is still alive, because the user explicitly requested a hard reset.
    if clear_stale_external_files:
        try:
            watchdog_dir = os.environ.get("GLOSSARION_WATCHDOG_DIR")
            if watchdog_dir and os.path.isdir(watchdog_dir):
                import glob
                for fp in glob.glob(os.path.join(watchdog_dir, "api_watchdog_*.json*")):
                    try:
                        os.remove(fp)
                    except Exception:
                        continue
        except Exception:
            pass


def kill_helper_subprocesses():
    """Terminate the known helper subprocesses this process started (chapter/PDF extraction,
    browser token helpers). No-op where subprocesses are unavailable (mobile)."""
    if not subprocesses_available():
        return
    # Best-effort: terminate only known helper subprocesses we started.
    try:
        import psutil
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)

        # Collect inpainter worker PIDs to protect from termination.
        _protected_pids = set()
        try:
            from manga_translator import MangaTranslator
            if hasattr(MangaTranslator, '_inpaint_pool') and MangaTranslator._inpaint_pool:
                for _key, _rec in MangaTranslator._inpaint_pool.items():
                    if _rec and 'spares' in _rec:
                        for _inp in _rec['spares']:
                            if _inp and getattr(_inp, '_mp_worker', None):
                                try:
                                    _pid = _inp._mp_worker.pid
                                    if _pid:
                                        _protected_pids.add(_pid)
                                except Exception:
                                    pass
        except Exception:
            pass

        def _cmdline_s(proc) -> str:
            try:
                cmd = proc.cmdline()
                return " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            except Exception:
                return ""

        def _is_mp_internal(cmd_s: str) -> bool:
            cs = (cmd_s or "")
            return ("--multiprocessing-fork" in cs) or ("spawn_main" in cs) or ("multiprocessing.spawn" in cs)

        # Only terminate explicit helper modes (safe).
        processes_to_terminate = []
        for child in children:
            try:
                if child.pid in _protected_pids:
                    continue
                cmd_s = _cmdline_s(child)
                if _is_mp_internal(cmd_s):
                    continue

                # Known helper flags / scripts
                #
                # Browser-backed routes (AuthND / Gemini-Free) spawn a
                # short-lived QtWebEngine helper subprocess to mint a
                # captcha/search token. If a stop races the spawn, the
                # helper (and its Chromium children) can linger holding
                # PyInstaller _MEIPASS DLLs, which makes the bootloader's
                # temp-dir cleanup fail with "Failed to remove temporary
                # directory". Kill them here so they don't outlive Stop.
                if ("--run-chapter-extraction" in cmd_s or "chapter_extraction_worker" in cmd_s
                        or "--run-pdf-extraction" in cmd_s or "_pdf_extraction_worker" in cmd_s
                        or "pdf_extraction_manager" in cmd_s or "pdf_extractor" in cmd_s
                        or "--authnd-mint-token" in cmd_s or "--mint-token" in cmd_s
                        or "--gemini-free-search" in cmd_s or "--search-helper" in cmd_s):
                    processes_to_terminate.append(child)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if processes_to_terminate:
            for proc in processes_to_terminate:
                try:
                    proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            try:
                gone, alive = psutil.wait_procs(processes_to_terminate, timeout=1)
            except Exception:
                alive = processes_to_terminate

            for proc in alive:
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except Exception as e:
        print(f"Error terminating helper child processes: {e}")


def request_stop(*, graceful, wait_for_chunks, force=False, set_stop_requested, log=print,
                 clear_watchdog=None, stop_flag_hook=None, cleanup_thread_created=None,
                 thread_name="translation-stop-cleanup"):
    """Stop the running translation job, in the exact desktop flag order.

    Order (stop_translation): env mode flags (TRANSLATION_CANCELLED for an immediate
    stop, GRACEFUL_STOP, GRACEFUL_STOP_COMPLETED) -> graceful only: cancel queued sends
    and pending watchdog rows -> WAIT_FOR_CHUNKS -> reset the API-call stagger ->
    ``set_stop_requested()`` (the latch) -> PDF stop file, glossary stop file (immediate
    only) -> immediate only: module stop flags, then a background thread that hard-cancels
    HTTP sessions, resets the watchdog (``clear_watchdog``) and kills helper subprocesses.

    * ``graceful``: the "graceful stop" setting; ``force=True`` (double click) first
      applies ``apply_force_stop_flags()`` and stops immediately.
    * ``wait_for_chunks``: the "wait for chunks" setting (only effective with graceful).
    * ``set_stop_requested()``: sets the owner's stop latch (desktop: graceful_stop_active,
      the stop timestamps and stop_requested).
    * ``log``: user-facing log lines; ``stop_flag_hook()``: extra module stop flag the
      desktop calls first (its lazily loaded ``translation_stop_flag``);
      ``cleanup_thread_created(thread)`` runs before the cleanup thread starts.

    Returns the cleanup thread (immediate stop) or None.
    """
    if force:
        apply_force_stop_flags()
        graceful = False
    graceful_stop = graceful
    wait_for_chunks_var = wait_for_chunks

    # Set environment variable to suppress multi-key logging and signal hard abort.
    # During graceful stop, do NOT set TRANSLATION_CANCELLED — it would kill
    # in-flight AuthGem/AuthGPT SSE streams via _is_externally_stopped().
    if not graceful_stop:
        os.environ['TRANSLATION_CANCELLED'] = '1'

    # Set graceful stop mode in environment so API client knows to show logs
    os.environ['GRACEFUL_STOP'] = '1' if graceful_stop else '0'
    if not graceful_stop:
        os.environ['GRACEFUL_STOP_COMPLETED'] = '0'

    # A graceful first click preserves only requests which have crossed the
    # real provider boundary.  Cancel admitted-but-waiting wrappers and
    # remove their queued/cooldown watchdog rows in one snapshot so the GUI
    # immediately reflects the queue being closed.
    if graceful_stop:
        cancelled_queued = 0
        cleared_watchdog = 0
        try:
            import TransateKRtoEN as _translation_module
            cancel_queued = getattr(
                _translation_module,
                'cancel_queued_translation_sends',
                None,
            )
            if callable(cancel_queued):
                cancelled_queued = int(cancel_queued() or 0)
        except Exception:
            cancelled_queued = 0
        try:
            import unified_api_client as _uac
            clear_pending = getattr(
                _uac,
                '_api_watchdog_clear_pending_requests',
                None,
            )
            if callable(clear_pending):
                cleared_watchdog = int(clear_pending() or 0)
        except Exception:
            cleared_watchdog = 0
        queued_count = max(cancelled_queued, cleared_watchdog)
        if queued_count:
            log(
                f"⏹️ Graceful stop cleared {queued_count} queued API "
                "request(s); active calls will finish."
            )

    # Set wait for chunks mode - only applies when graceful stop is also enabled
    wait_for_chunks = wait_for_chunks_var and graceful_stop
    os.environ['WAIT_FOR_CHUNKS'] = '1' if wait_for_chunks else '0'

    # Drop any queued API-delay reservations now. Stopped worker threads may
    # have already reserved future send slots, and keeping those slots makes
    # the next translation inherit a stacked "Sending API call in ..." timer.
    try:
        import unified_api_client
        if hasattr(unified_api_client, 'reset_api_call_stagger'):
            unified_api_client.reset_api_call_stagger()
        elif hasattr(unified_api_client, 'UnifiedClient') and hasattr(unified_api_client.UnifiedClient, 'reset_api_call_stagger'):
            unified_api_client.UnifiedClient.reset_api_call_stagger()
    except Exception:
        pass

    # Debug: Log the stop settings being applied
    print(f"🔧 Stop triggered: graceful_stop={graceful_stop}, wait_for_chunks_var={wait_for_chunks_var}, WAIT_FOR_CHUNKS={os.environ.get('WAIT_FOR_CHUNKS')}")

    # Publish the environment mode before the shared callback latch. Workers
    # poll both; setting stop_requested first creates a race where a graceful
    # first click can be mistaken for an immediate/force stop.
    set_stop_requested()

    # PDF extraction has no API request to preserve. Graceful and immediate
    # stops both halt it at the next page/image boundary, including child
    # process workers.
    try:
        pdf_stop_file = os.environ.get('PDF_EXTRACTION_STOP_FILE')
        if pdf_stop_file:
            os.makedirs(os.path.dirname(pdf_stop_file) or '.', exist_ok=True)
            with open(pdf_stop_file, 'w', encoding='utf-8') as f:
                f.write('stop')
    except Exception:
        pass

    # Touch stop file for cross-process glossary workers (only for immediate stop)
    if not graceful_stop:
        try:
            stop_file = os.environ.get('GLOSSARY_STOP_FILE')
            if stop_file:
                with open(stop_file, 'w', encoding='utf-8') as f:
                    f.write('stop')
        except Exception:
            pass

    # For graceful stop: DON'T abort in-flight API calls, let them finish
    # For immediate stop: abort everything aggressively
    if not graceful_stop:
        # The desktop's lazily imported translation_stop_flag (TransateKRtoEN.set_stop_flag)
        if stop_flag_hook is not None:
            stop_flag_hook()

        # Also try to call it directly on the module if imported
        try:
            import TransateKRtoEN
            if hasattr(TransateKRtoEN, 'set_stop_flag'):
                TransateKRtoEN.set_stop_flag(True)
        except:
            pass

        try:
            import unified_api_client
            if hasattr(unified_api_client, 'set_stop_flag'):
                unified_api_client.set_stop_flag(True)
            # If there's a global client instance, stop it too
            if hasattr(unified_api_client, 'global_stop_flag'):
                unified_api_client.global_stop_flag = True

            # Set the _cancelled flag on the UnifiedClient class itself
            if hasattr(unified_api_client, 'UnifiedClient'):
                unified_api_client.UnifiedClient._global_cancelled = True
        except Exception as e:
            print(f"Error setting stop flags: {e}")

        # ── SLOW PATH (background thread): close connections, kill procs ──
        def _stop_heavy_work():
            # Hard cancel: close active HTTP sessions to abort in-flight requests
            try:
                import unified_api_client as _uac
                if hasattr(_uac, 'hard_cancel_all'):
                    _uac.hard_cancel_all()
                # Also reset watchdog counts so progress bar clears immediately
                if hasattr(_uac, '_api_watchdog_reset'):
                    _uac._api_watchdog_reset()
            except Exception:
                pass

            # Delete watchdog files again after stop flags/hard-cancel
            if clear_watchdog is not None:
                try:
                    clear_watchdog()
                except Exception:
                    pass

            kill_helper_subprocesses()

        stop_cleanup_thread = threading.Thread(
            target=_stop_heavy_work,
            daemon=True,
            name=thread_name,
        )
        if cleanup_thread_created is not None:
            cleanup_thread_created(stop_cleanup_thread)
        stop_cleanup_thread.start()
        return stop_cleanup_thread
    return None


def stop_epub_converter():
    """Raise the EPUB converter's stop flag (stop_translation, while the converter runs).

    Raises when epub_converter cannot be imported; the callers swallow it like the desktop.
    """
    import epub_converter
    if hasattr(epub_converter, 'set_stop_flag'):
        epub_converter.set_stop_flag(True)


def silence_http_loggers():
    """A graceful stop silences the HTTP client loggers (``HTTP_LOGGER_NAMES``)."""
    try:
        import logging
        for logger_name in HTTP_LOGGER_NAMES:
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    except Exception:
        pass


def announce_stop(graceful_stop, log=print):
    """The tail of stop_translation: HTTP log suppression (graceful) and the stop-mode log line.

    Reads WAIT_FOR_CHUNKS as ``request_stop`` published it.
    """
    # Suppress HTTP logs during graceful stop
    if graceful_stop:
        silence_http_loggers()

    # Log message depends on stop mode
    if graceful_stop:
        try:
            wait_for_chunks = os.environ.get('WAIT_FOR_CHUNKS') == '1'
        except Exception:
            wait_for_chunks = False
        if wait_for_chunks:
            log("⏳ Graceful stop — waiting for in-flight API calls to complete...")
        else:
            log("⏳ Graceful stop — waiting for the current in-flight API call; queued work will not continue (WAIT_FOR_CHUNKS=0)")
    else:
        log("🛑 Force stop requested — aborting queued/in-flight API calls")


def make_glossary_stop_callback(is_stop_requested, is_graceful):
    """The stop callback glossary extraction polls (``stop_callback`` of extract_glossary_from_epub.main).

    ``is_stop_requested()`` is the owner's stop latch, ``is_graceful()`` its
    graceful-stop state (GRACEFUL_STOP=1 in the environment counts too).
    """
    def enhanced_stop_callback():
        """Stop callback for glossary extraction.

        Immediate stop: abort quickly.
        Graceful stop: do NOT abort an in-flight API call; stop only once in-flight is idle.
        """
        # Check GUI stop flag
        if is_stop_requested():
            graceful = bool(is_graceful()) or (os.environ.get('GRACEFUL_STOP') == '1')
            if graceful:
                # If any API calls are currently in flight, keep going so they can finish.
                try:
                    import unified_api_client
                    st = unified_api_client.get_api_watchdog_state() or {}
                    if int(st.get('in_flight', 0) or 0) > 0:
                        return False
                except Exception:
                    return False
                # No in-flight calls: safe to stop before starting a new one.
                return True
            return True

        # Also check if the glossary extraction module has its own stop flag
        try:
            import extract_glossary_from_epub
            if hasattr(extract_glossary_from_epub, 'is_stop_requested') and extract_glossary_from_epub.is_stop_requested():
                return True
        except:
            pass

        return False

    return enhanced_stop_callback
