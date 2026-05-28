"""
Shutdown utilities to ensure child processes are terminated and the interpreter exits.
Designed for Windows-first behavior, with best-effort cross-platform fallbacks.
"""

from __future__ import annotations

import os
import sys
import json
import time
import subprocess
import tempfile
import shutil
import concurrent.futures
from contextlib import contextmanager
from typing import Callable, Iterable, Optional


_last_qt_shutdown_drain_at = 0.0


def _normalize_exit_code(code) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _cpu_worker_cap() -> int:
    """Return the user's logical CPU count as a safe worker ceiling."""
    try:
        return max(1, int(os.cpu_count() or 1))
    except Exception:
        return 1


def _run_cleanup_fns(cleanup_fns: Optional[Iterable[Callable[[], None]]]) -> None:
    if not cleanup_fns:
        return
    for fn in cleanup_fns:
        try:
            if callable(fn):
                fn()
        except Exception:
            # Best-effort cleanup
            pass


def drain_qt_events_for_shutdown(duration_ms: int = 350, slice_ms: int = 50) -> None:
    """Let Qt process close/deleteLater work before a hard process exit.

    QtWebEngine keeps native GPU/VSync helper threads alive behind the
    widgets. A tiny event drain after closing WebEngine views gives those
    threads a chance to unwind before ``os._exit`` cuts the process off.
    """
    if duration_ms <= 0:
        return
    try:
        from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QThread
    except Exception:
        return

    global _last_qt_shutdown_drain_at
    try:
        app = QCoreApplication.instance()
        if app is None:
            return
        try:
            if QThread.currentThread() != app.thread():
                return
        except Exception:
            pass

        deadline = time.monotonic() + (duration_ms / 1000.0)
        slice_ms = max(1, int(slice_ms or 1))
        while time.monotonic() < deadline:
            try:
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                QCoreApplication.sendPostedEvents(None, 0)
                app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, slice_ms)
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            except Exception:
                try:
                    app.processEvents()
                except Exception:
                    return
            time.sleep(0.01)
        _last_qt_shutdown_drain_at = time.monotonic()
    except Exception:
        pass


def _terminate_multiprocessing_children(timeout: float = 1.5) -> None:
    try:
        import multiprocessing as mp
        children = mp.active_children()
        for p in children:
            try:
                p.terminate()
            except Exception:
                pass
        for p in children:
            try:
                p.join(timeout=timeout)
            except Exception:
                pass
        # Anything still alive after terminate/join: hard kill so it can
        # release file handles to DLLs under _MEIPASS before the bootloader
        # tries to clean the temp directory.
        for p in children:
            try:
                if p.is_alive():
                    if hasattr(p, "kill"):
                        p.kill()
                    else:
                        p.terminate()
                    p.join(timeout=timeout)
            except Exception:
                pass
    except Exception:
        pass


def _ensure_safe_tempdir() -> None:
    """
    Ensure a writable temp directory is set to avoid hangs when the default temp
    location is inaccessible (common on locked-down Windows environments).
    """
    try:
        candidates = []
        for var in ("TMPDIR", "TEMP", "TMP"):
            val = os.environ.get(var)
            if val:
                candidates.append(val)
        try:
            candidates.append(os.path.join(os.getcwd(), "_tmp"))
        except Exception:
            pass
        try:
            candidates.append(os.path.join(os.path.expanduser("~"), ".glossarion_tmp"))
        except Exception:
            pass

        for path in candidates:
            try:
                if not path:
                    continue
                os.makedirs(path, exist_ok=True)
                test_path = os.path.join(path, ".__tmp_test__")
                with open(test_path, "wb") as f:
                    f.write(b"x")
                try:
                    os.remove(test_path)
                except Exception:
                    pass
                os.environ["TMPDIR"] = path
                os.environ["TEMP"] = path
                os.environ["TMP"] = path
                tempfile.tempdir = path
                return
            except Exception:
                continue
    except Exception:
        pass


def _terminate_psutil_children(timeout: float = 1.5) -> None:
    """Terminate every descendant of the current process.

    Any grandchild that still has DLLs from `_MEIPASS` mapped will cause the
    PyInstaller bootloader's final `rmtree` to fail and pop the
    "Failed to remove temporary directory" dialog, so we err on the side of
    killing aggressively and then waiting for handles to drop.
    """
    try:
        import psutil
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        for proc in children:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            gone, alive = psutil.wait_procs(children, timeout=timeout)
        except Exception:
            gone, alive = [], children
        for proc in alive:
            try:
                proc.kill()
            except Exception:
                pass
        # Re-check after kill to make sure handles to _MEIPASS DLLs are gone.
        try:
            still = [p for p in alive if p.is_running()]
            if still:
                psutil.wait_procs(still, timeout=timeout)
        except Exception:
            pass
    except Exception:
        pass


def _taskkill_self_tree() -> None:
    if os.name != "nt":
        return
    if os.environ.get("GLOSSARION_TASKKILL_SELF_ON_EXIT", "").strip().lower() not in ("1", "true", "yes", "on"):
        return
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            ["taskkill", "/F", "/T", "/PID", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception:
        pass


def request_hard_stop_for_shutdown(owner=None, translation_stop_flag=None, glossary_stop_flag=None) -> None:
    """Set all known immediate-stop flags before shutdown cleanup starts."""
    try:
        if owner is not None:
            owner.stop_requested = True
            owner.graceful_stop_active = False
        os.environ["TRANSLATION_CANCELLED"] = "1"
        os.environ["GRACEFUL_STOP"] = "0"
        os.environ["GRACEFUL_STOP_COMPLETED"] = "0"
        os.environ["WAIT_FOR_CHUNKS"] = "0"
    except Exception:
        pass

    try:
        stop_file = os.environ.get("GLOSSARY_STOP_FILE")
        if stop_file:
            with open(stop_file, "w", encoding="utf-8") as f:
                f.write("stop")
    except Exception:
        pass

    for setter in (translation_stop_flag, glossary_stop_flag):
        try:
            if setter:
                setter(True)
        except Exception:
            pass

    for module_name in ("TransateKRtoEN", "extract_glossary_from_epub", "GlossaryManager"):
        try:
            module = sys.modules.get(module_name)
            if hasattr(module, "set_stop_flag"):
                module.set_stop_flag(True)
        except Exception:
            pass

    try:
        import unified_api_client
        if hasattr(unified_api_client, "set_stop_flag"):
            unified_api_client.set_stop_flag(True)
        if hasattr(unified_api_client, "global_stop_flag"):
            unified_api_client.global_stop_flag = True
        if hasattr(unified_api_client, "UnifiedClient"):
            unified_api_client.UnifiedClient._global_cancelled = True
            if hasattr(unified_api_client.UnifiedClient, "set_global_cancellation"):
                unified_api_client.UnifiedClient.set_global_cancellation(True)
    except Exception:
        pass


def _dedupe_paths(paths):
    seen = set()
    unique = []
    for path in paths:
        if not path:
            continue
        try:
            norm = os.path.normcase(os.path.abspath(path))
        except Exception:
            norm = str(path)
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(path)
    return unique


def _selected_input_files_for_shutdown(input_files=None, entry_file=None):
    files = list(input_files or [])
    if entry_file and os.path.isfile(str(entry_file)):
        files.append(str(entry_file))
    return [
        file_path for file_path in files
        if file_path and file_path != "__generative_mode__"
    ]


def translation_progress_paths_for_shutdown(input_files=None, entry_file=None, output_dir_resolver=None):
    paths = []
    for file_path in _selected_input_files_for_shutdown(input_files, entry_file):
        try:
            output_dir = output_dir_resolver(file_path) if output_dir_resolver else None
            if output_dir:
                paths.append(os.path.join(str(output_dir), "translation_progress.json"))
        except Exception:
            pass
    return _dedupe_paths(paths)


def glossary_progress_paths_for_shutdown(
    input_files=None,
    entry_file=None,
    output_dir_resolver=None,
    config=None,
    app_dir=None,
    output_path=None,
):
    paths = []
    files = _selected_input_files_for_shutdown(input_files, entry_file)
    roots = []

    def _add_root(root):
        if not root:
            return
        try:
            roots.append(os.path.abspath(str(root)))
        except Exception:
            pass

    try:
        override_dir = os.environ.get("OUTPUT_DIRECTORY") or (config or {}).get("output_directory")
    except Exception:
        override_dir = os.environ.get("OUTPUT_DIRECTORY")
    if override_dir:
        _add_root(os.path.join(override_dir, "Glossary"))
    _add_root(os.environ.get("GLOSSARY_SHARED_DIR"))
    _add_root(app_dir and os.path.join(app_dir, "Glossary"))
    if files:
        try:
            _add_root(os.path.join(os.path.dirname(os.path.abspath(files[0])), "Glossary"))
        except Exception:
            pass

    for file_path in files:
        try:
            base = os.path.splitext(os.path.basename(file_path))[0]
            for root in roots:
                paths.append(os.path.join(root, base, f"{base}_glossary_progress.json"))
                paths.append(os.path.join(root, f"{base}_glossary_progress.json"))
            try:
                output_dir = output_dir_resolver(file_path) if output_dir_resolver else None
                if output_dir:
                    paths.append(os.path.join(output_dir, "glossary_progress.json"))
                    paths.append(os.path.join(output_dir, "Glossary", f"{base}_glossary_progress.json"))
            except Exception:
                pass
        except Exception:
            pass

    try:
        output_path = (output_path or os.environ.get("OUTPUT_PATH", "")).strip()
        if output_path:
            out_dir = os.path.dirname(os.path.abspath(output_path))
            base = os.path.splitext(os.path.basename(output_path))[0]
            if base.lower().endswith("_glossary"):
                base = base[:-len("_glossary")]
            paths.append(os.path.join(out_dir, f"{base}_glossary_progress.json"))
    except Exception:
        pass

    try:
        extract_module = sys.modules.get("extract_glossary_from_epub")
        module_progress = getattr(extract_module, "PROGRESS_FILE", "")
        if module_progress and (
            os.path.isabs(str(module_progress))
            or os.path.basename(str(module_progress)).lower() != "glossary_progress.json"
        ):
            paths.append(module_progress)
    except Exception:
        pass

    return _dedupe_paths(paths)


def _unique_int_list(values):
    seen = set()
    result = []
    for value in values or []:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result


def _remove_idx(values, idx):
    return [value for value in _unique_int_list(values) if value != idx]


def _add_idx(values, idx):
    values = _unique_int_list(values)
    if idx not in values:
        values.append(idx)
    return values


def _progress_entry_index(info, key=None):
    if isinstance(info, dict):
        for idx_key in ("chapter_index", "idx", "index"):
            try:
                return int(info.get(idx_key))
            except (TypeError, ValueError):
                pass
    try:
        return int(key)
    except (TypeError, ValueError):
        return None


def _progress_chapter_key(idx):
    try:
        return str(int(idx))
    except (TypeError, ValueError):
        return str(idx)


def _progress_filename(value):
    name = os.path.basename(str(value or "").strip())
    if not name:
        return ""
    if os.path.splitext(name)[1].lower() in (".epub", ".pdf", ".zip", ".cbz"):
        return ""
    return name


def _progress_field_value(progress_data, field_name, idx):
    values = progress_data.get(field_name, {})
    if not isinstance(values, dict):
        return None
    return values.get(str(idx), values.get(idx))


def _restore_previous_progress_entry(info):
    if not isinstance(info, dict):
        return None
    previous_status = str(info.get("previous_status", "") or "").lower()
    previous_entry = info.get("previous_progress_entry")
    if isinstance(previous_entry, dict):
        restored = dict(previous_entry)
        restored_status = str(restored.get("status", previous_status) or previous_status).lower()
        if restored_status and restored_status not in ("in_progress", "not_completed", "not translated", "not_translated"):
            restored.pop("previous_status", None)
            restored.pop("previous_progress_entry", None)
            restored.pop("previous_status_unknown", None)
            return restored

    if previous_status in ("qa_failed", "failed", "error", "pending", "merged", "completed"):
        restored = dict(info)
        restored["status"] = "failed" if previous_status == "error" else previous_status
        restored.pop("previous_status", None)
        restored.pop("previous_progress_entry", None)
        restored.pop("previous_status_unknown", None)
        return restored

    if info.get("previous_status_unknown"):
        restored = dict(info)
        restored["status"] = "failed"
        restored.pop("previous_status", None)
        restored.pop("previous_progress_entry", None)
        restored.pop("previous_status_unknown", None)
        return restored

    if previous_status in ("not_completed", "not translated", "not_translated", ""):
        if previous_status:
            return None
        if info.get("output_file"):
            restored = dict(info)
            restored["status"] = "failed"
            restored.pop("previous_status", None)
            restored.pop("previous_progress_entry", None)
            restored.pop("previous_status_unknown", None)
            return restored
    return None


def _failed_from_in_progress_entry(info):
    if not isinstance(info, dict):
        info = {}
    failed = dict(info)
    failed["status"] = "failed"
    failed.pop("previous_status", None)
    failed.pop("previous_progress_entry", None)
    failed.pop("previous_status_unknown", None)
    return failed


def _atomic_replace_progress_file(temp_path, target_path, max_retries=3, delay=0.5):
    last_err = None
    for attempt in range(max_retries):
        try:
            os.replace(temp_path, target_path)
            return
        except (PermissionError, OSError) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(delay)
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass
    if last_err:
        raise last_err


@contextmanager
def _locked_progress_file(progress_file):
    if not progress_file:
        yield
        return

    progress_dir = os.path.dirname(progress_file) or "."
    os.makedirs(progress_dir, exist_ok=True)
    lock_path = f"{progress_file}.lock"
    lock_f = open(lock_path, "a+b")
    locked = False
    try:
        if lock_f.seek(0, os.SEEK_END) == 0:
            lock_f.write(b"\0")
            lock_f.flush()
            os.fsync(lock_f.fileno())
        lock_f.seek(0)

        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    lock_f.seek(0)
                    msvcrt.locking(lock_f.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                    break
                except OSError:
                    time.sleep(0.05)
        else:
            import fcntl
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            locked = True

        yield
    finally:
        try:
            if locked:
                lock_f.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock_f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        finally:
            lock_f.close()


def restore_glossary_in_progress_for_hard_stop(progress_file):
    """Resolve persisted glossary in-progress rows before an immediate shutdown."""
    if not progress_file or not os.path.exists(progress_file):
        return 0

    with _locked_progress_file(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
        except Exception:
            return 0
        if not isinstance(progress_data, dict):
            return 0

        chapters = progress_data.get("chapters", {})
        if not isinstance(chapters, dict):
            chapters = {}

        completed_clean = _unique_int_list(progress_data.get("completed", []))
        failed_clean = _unique_int_list(progress_data.get("failed", []))
        merged_clean = _unique_int_list(progress_data.get("merged_indices", []))
        in_progress_clean = _unique_int_list(progress_data.get("in_progress", []))

        changed = 0
        processed_indices = set()

        def _apply_resolved_status(idx, resolved_status):
            nonlocal completed_clean, failed_clean, merged_clean
            if idx is None:
                return
            completed_clean = _remove_idx(completed_clean, idx)
            failed_clean = _remove_idx(failed_clean, idx)
            merged_clean = _remove_idx(merged_clean, idx)
            if resolved_status == "completed":
                completed_clean = _add_idx(completed_clean, idx)
            elif resolved_status == "merged":
                merged_clean = _add_idx(merged_clean, idx)
            elif resolved_status in ("failed", "qa_failed", "error"):
                failed_clean = _add_idx(failed_clean, idx)

        for chapter_key, chapter_info in list(chapters.items()):
            if not isinstance(chapter_info, dict):
                continue
            if str(chapter_info.get("status", "")).lower() != "in_progress":
                continue
            idx = _progress_entry_index(chapter_info, chapter_key)
            if idx is not None:
                processed_indices.add(idx)

            restored = _restore_previous_progress_entry(chapter_info)
            if restored:
                chapters[chapter_key] = restored
                resolved_status = str(restored.get("status", "") or "").lower()
            else:
                failed = _failed_from_in_progress_entry(chapter_info)
                chapters[chapter_key] = failed
                resolved_status = str(failed.get("status", "failed") or "failed").lower()
            _apply_resolved_status(idx, resolved_status)
            changed += 1

        for idx in in_progress_clean:
            if idx in processed_indices:
                continue
            chapter_key = _progress_chapter_key(idx)
            existing_info = chapters.get(chapter_key)
            if isinstance(existing_info, dict) and str(existing_info.get("status", "")).lower() == "in_progress":
                continue
            if isinstance(existing_info, dict) and existing_info:
                existing_status = str(existing_info.get("status", "") or "").lower()
                processed_indices.add(idx)
                _apply_resolved_status(idx, existing_status)
                changed += 1
                continue

            actual_num = _progress_field_value(progress_data, "chapter_numbers", idx)
            try:
                actual_num = int(actual_num)
            except (TypeError, ValueError):
                actual_num = idx + 1
            chapter_info = {
                "chapter_index": idx,
                "actual_num": actual_num,
                "chapter_num": actual_num,
                "status": "failed",
                "last_updated": time.time(),
            }
            chapter_file = _progress_filename(_progress_field_value(progress_data, "chapter_filenames", idx))
            if chapter_file:
                chapter_info["output_file"] = chapter_file
            for key in ("model_name", "key_identifier", "key_pool"):
                if progress_data.get(key):
                    chapter_info[key] = progress_data[key]
            chapters[chapter_key] = chapter_info
            processed_indices.add(idx)
            _apply_resolved_status(idx, "failed")
            changed += 1

        if not changed:
            return 0

        progress_data["chapters"] = chapters
        progress_data["completed"] = completed_clean
        progress_data["failed"] = failed_clean
        progress_data["merged_indices"] = merged_clean
        progress_data["in_progress"] = [idx for idx in in_progress_clean if idx not in processed_indices]

        temp_path = None
        try:
            progress_dir = os.path.dirname(progress_file) or "."
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=progress_dir, delete=False, suffix=".tmp") as temp_f:
                temp_path = temp_f.name
                json.dump(progress_data, temp_f, ensure_ascii=False, indent=2)
                temp_f.flush()
                os.fsync(temp_f.fileno())
            _atomic_replace_progress_file(temp_path, progress_file)
        except Exception:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise

        return changed


def _restore_image_progress_manager_for_shutdown(manager) -> int:
    try:
        progress = getattr(manager, "prog", None)
        images = progress.get("images", {}) if isinstance(progress, dict) else {}
        if not isinstance(images, dict):
            return 0
        changed = 0
        now = time.time()
        for content_hash, image_info in list(images.items()):
            if not isinstance(image_info, dict):
                continue
            if str(image_info.get("status", "")).lower() != "in_progress":
                continue
            updated = dict(image_info)
            updated["status"] = "cancelled"
            updated["last_updated"] = now
            updated["cancelled_at"] = now
            images[content_hash] = updated
            changed += 1
        if changed and hasattr(manager, "save"):
            manager.save()
        return changed
    except Exception:
        return 0


def restore_in_progress_rows_for_shutdown(
    input_files=None,
    entry_file=None,
    output_dir_resolver=None,
    config=None,
    app_dir=None,
    image_progress_managers=None,
    max_workers: int = 8,
) -> None:
    """Resolve persisted in-progress rows in parallel before a forced exit."""
    cleanup_tasks = []

    try:
        from TransateKRtoEN import ProgressManager
    except Exception:
        ProgressManager = None

    def _restore_translation_progress(progress_file):
        if not progress_file or not os.path.exists(progress_file) or ProgressManager is None:
            return 0
        manager = ProgressManager(os.path.dirname(progress_file))
        changed = manager.restore_all_in_progress_for_hard_stop()
        if changed:
            manager.save()
        return changed

    for progress_file in translation_progress_paths_for_shutdown(input_files, entry_file, output_dir_resolver):
        if progress_file and os.path.exists(progress_file) and ProgressManager is not None:
            cleanup_tasks.append((
                f"translation progress {progress_file}",
                _restore_translation_progress,
                progress_file,
            ))

    def _restore_glossary_progress(progress_file):
        if not progress_file or not os.path.exists(progress_file):
            return 0
        return restore_glossary_in_progress_for_hard_stop(progress_file)

    for progress_file in glossary_progress_paths_for_shutdown(
        input_files=input_files,
        entry_file=entry_file,
        output_dir_resolver=output_dir_resolver,
        config=config,
        app_dir=app_dir,
    ):
        if progress_file and os.path.exists(progress_file):
            cleanup_tasks.append((
                f"glossary progress {progress_file}",
                _restore_glossary_progress,
                progress_file,
            ))

    for name, manager in image_progress_managers or []:
        if manager is not None:
            cleanup_tasks.append((
                f"image progress manager {name}",
                _restore_image_progress_manager_for_shutdown,
                manager,
            ))

    if not cleanup_tasks:
        return

    cpu_cap = _cpu_worker_cap()
    requested_workers = max(1, int(max_workers or 1))
    workers = min(requested_workers, len(cleanup_tasks), cpu_cap)
    print(
        f"[CLOSE] Shutdown progress cleanup workers: {workers} "
        f"(cpu_cap={cpu_cap}, tasks={len(cleanup_tasks)}, "
        f"requested={requested_workers})"
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="shutdown-progress-cleanup",
        ) as pool:
            future_map = {
                pool.submit(task_fn, task_arg): task_name
                for task_name, task_fn, task_arg in cleanup_tasks
            }
            for future in concurrent.futures.as_completed(future_map):
                task_name = future_map[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[CLOSE] Warning: could not restore {task_name}: {e}")
    except Exception as e:
        print(f"[CLOSE] Warning: parallel shutdown progress cleanup failed: {e}")


def _cleanup_pyinstaller_temp_dir(retries: int = 4, delay: float = 0.2) -> None:
    """Intentional no-op.

    Previously this tried to `shutil.rmtree(sys._MEIPASS)` from inside the
    frozen child process. That is actively harmful: DLLs/PYDs loaded from
    `_MEIPASS` are still mapped into this process, so the tree ends up only
    partially deleted. The PyInstaller bootloader then runs its own cleanup
    after we exit, walks the half-deleted tree, fails on the mapped files,
    and shows the

        "Failed to remove temporary directory: ...\\_MEIxxxxxx"

    warning dialog. There is no way to suppress that dialog from Python
    because the bootloader shows it *after* our interpreter is already gone.

    The correct fix is to leave `_MEIPASS` alone and make sure every child
    process is dead (so its mapped DLLs get unmapped) before we exit. Then
    the bootloader's own cleanup succeeds and no dialog is ever shown.
    """
    return


def force_shutdown(exit_code: int = 0, cleanup_fns: Optional[Iterable[Callable[[], None]]] = None) -> None:
    """
    Best-effort cleanup then forcefully exit the current process.
    Terminates child processes (multiprocessing + psutil if available) and
    can optionally fall back to taskkill on Windows via
    GLOSSARION_TASKKILL_SELF_ON_EXIT=1.

    Ordering note: children must be terminated BEFORE we exit so that any
    DLLs they loaded from PyInstaller's `_MEIPASS` are unmapped. Otherwise
    the bootloader's final temp-dir cleanup pops a warning dialog.
    We deliberately do NOT touch `_MEIPASS` from Python; see
    `_cleanup_pyinstaller_temp_dir` for the rationale.
    """
    code = _normalize_exit_code(exit_code)
    _ensure_safe_tempdir()
    _run_cleanup_fns(cleanup_fns)
    if time.monotonic() - _last_qt_shutdown_drain_at > 0.75:
        drain_qt_events_for_shutdown(duration_ms=350)
    else:
        print("[CLOSE] Skipping duplicate Qt shutdown event drain")
    # Kill descendants first so their handles to _MEIPASS drop before the
    # bootloader tries to rmtree it after we return.
    _terminate_multiprocessing_children()
    _terminate_psutil_children()
    _cleanup_pyinstaller_temp_dir()  # no-op, kept for backwards compatibility
    # Disabled by default. Killing our own process tree with taskkill can
    # interrupt Qt/PySide/native DLL teardown at an arbitrary instruction and
    # show Windows' "memory could not be read" dialog. Child processes are
    # already handled above; this remains available as an opt-in last resort.
    _taskkill_self_tree()
    try:
        os._exit(code)
    except Exception:
        sys.exit(code)


def run_cli_main(main_fn: Callable[[], Optional[int]], cleanup_fns: Optional[Iterable[Callable[[], None]]] = None) -> None:
    """
    Run a CLI-style main function and always force shutdown with the appropriate exit code.
    """
    exit_code = 0
    try:
        result = main_fn()
        if isinstance(result, int):
            exit_code = result
    except SystemExit as e:
        if isinstance(e.code, str):
            try:
                print(e.code, file=sys.stderr)
            except Exception:
                pass
        exit_code = _normalize_exit_code(e.code)
    except Exception as e:
        try:
            print(f"❌ Unhandled error: {e}", file=sys.stderr)
        except Exception:
            pass
        exit_code = 1
    finally:
        force_shutdown(exit_code, cleanup_fns=cleanup_fns)
