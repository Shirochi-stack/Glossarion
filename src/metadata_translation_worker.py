"""Metadata-only EPUB translation helpers.

``run_metadata_translation_job`` is the fast in-process path used by the GUI's
thread pool.  The legacy command-line entry point remains available for older
bundles and tests, but normal batch translation no longer starts one complete
Glossarion process per book.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import threading
import traceback
from typing import Any, Callable, Mapping

from epub_metadata_utils import extract_epub_metadata_file
from metadata_progress import (
    normalize_metadata_mode,
    resolve_metadata_field_settings,
)


LogCallback = Callable[[str], None]
StopCheck = Callable[[], bool]


def _emit(callback: LogCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _json_mapping(value: Any) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _float_setting(env: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except (TypeError, ValueError):
        return default


def _int_setting(env: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(float(env.get(key, default)))
    except (TypeError, ValueError):
        return default


def _extract_epub_metadata(source_path: str) -> dict:
    """Keep existing callers on the shared lightweight OPF parser."""
    return extract_epub_metadata_file(source_path)


def _load_metadata(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as metadata_file:
            value = json.load(metadata_file)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _save_metadata(path: str, metadata: Mapping[str, Any]) -> None:
    """Atomically replace metadata.json so the Library never sees half a file."""
    temp_path = (
        f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        with open(temp_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
            metadata_file.flush()
            os.fsync(metadata_file.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass


def _source_language(metadata: Mapping[str, Any]) -> str:
    language = str(metadata.get("language") or "").strip().lower()
    if language.startswith("ko") or "korean" in language:
        return "Korean"
    if language.startswith("ja") or "japanese" in language:
        return "Japanese"
    if language.startswith("zh") or "chinese" in language:
        return "Chinese"
    return ""


def _translated_keys(field_name: str) -> tuple[str, str]:
    if field_name == "title":
        return "original_title", "title_translated"
    return f"original_{field_name}", f"{field_name}_translated"


def _translate_title(
    title: Any,
    client,
    env: Mapping[str, Any],
    stop_check_fn: StopCheck,
) -> tuple[Any, bool]:
    """Run the dedicated title request without consulting per-book globals."""
    title_text = str(title or "").strip()
    if not title_text:
        return title, True
    if stop_check_fn():
        return title, False

    output_language = str(env.get("OUTPUT_LANGUAGE") or "English")
    user_prompt = str(env.get("BOOK_TITLE_PROMPT") or "").replace(
        "{target_lang}", output_language
    )
    system_prompt = str(
        env.get("BOOK_TITLE_SYSTEM_PROMPT")
        or (
            "Translate this book title to English while retaining any "
            "acronyms. Do not output anything other than the translated text."
        )
    ).replace("{target_lang}", output_language)
    from TransateKRtoEN import prepare_glossary_aware_request

    system_prompt, compliant_title, _ = prepare_glossary_aware_request(
        system_prompt,
        title_text,
        output_dir=getattr(client, "output_dir", None),
        source_path=(
            env.get("GLOSSARY_SOURCE_PATH") or env.get("EPUB_PATH")
        ),
        source_text=title_text,
        chapter_ref={"use_storage_gender": True},
        settings=dict(env),
    )

    client_type = getattr(client, "client_type", "")
    if client_type in {"deepl", "google_translate"}:
        messages = [{"role": "user", "content": compliant_title}]
    else:
        user_content = (
            f"{user_prompt}\n\n{compliant_title}"
            if user_prompt.strip() else compliant_title
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    from TransateKRtoEN import _skip_thinking_env, send_with_interrupt

    with _skip_thinking_env("BOOK_TITLE", quiet=True):
        response = send_with_interrupt(
            messages=messages,
            client=client,
            temperature=_float_setting(
                env, "TRANSLATION_TEMPERATURE", 0.3
            ),
            max_tokens=_int_setting(env, "MAX_OUTPUT_TOKENS", 8192),
            stop_check_fn=stop_check_fn,
            context="book_title",
        )

    if hasattr(response, "content"):
        translated = response.content
    elif isinstance(response, tuple):
        translated = response[0] if response else ""
    else:
        translated = str(response or "")
    translated = str(translated or "").strip()
    if (
        len(translated) >= 2
        and translated[0] == translated[-1]
        and translated[0] in {"'", '"'}
    ):
        translated = translated[1:-1].strip()
    invalid = (
        not translated
        or "\n" in translated
        or "{" in translated
        or "}" in translated
        or '"role":' in translated
        or '"content":' in translated
        or ("[[" in translated and "]]" in translated)
        or any(
            tag in translated.lower()
            for tag in ("<p>", "</p>", "<h1>", "</h1>", "<html")
        )
    )
    return (title, False) if invalid else (translated, True)


def run_metadata_translation_job(
    source_path: str,
    env_vars: Mapping[str, Any] | None,
    *,
    log_callback: LogCallback | None = None,
    stop_check_fn: StopCheck | None = None,
    client_factory=None,
    translator_factory=None,
) -> bool:
    """Translate one EPUB's metadata inside the caller's worker thread.

    All book-specific values are passed explicitly.  This is important because
    a process-wide environment and ``sys.argv`` cannot safely be swapped by
    several concurrent threads.
    """
    env = {
        str(key): "" if value is None else str(value)
        for key, value in dict(env_vars or {}).items()
    }
    source_path = os.path.abspath(str(source_path or ""))
    stop_check = stop_check_fn or (lambda: False)
    if not source_path or not os.path.isfile(source_path):
        _emit(log_callback, "❌ Source EPUB was not found")
        return False

    output_root = str(
        env.get("OUTPUT_DIRECTORY") or env.get("OUTPUT_DIR") or ""
    ).strip()
    output_root = os.path.abspath(output_root) if output_root else os.getcwd()
    output_folder = os.path.join(
        output_root,
        os.path.splitext(os.path.basename(source_path))[0],
    )
    metadata_path = os.path.join(output_folder, "metadata.json")
    progress_manager = None
    plan = []
    phase_by_field: dict[str, str] = {}

    def set_progress(status: str, *, key=None, error=None) -> None:
        if progress_manager is None:
            return
        progress_manager.update_metadata_status(
            status,
            metadata_path,
            error=error,
            key=key,
        )
        progress_manager.save()

    try:
        if stop_check():
            return False
        os.makedirs(output_folder, exist_ok=True)
        with open(
            os.path.join(output_folder, "source_epub.txt"),
            "w",
            encoding="utf-8",
        ) as source_pointer:
            source_pointer.write(source_path)

        _emit(log_callback, "⚡ Reading metadata directly from the source EPUB")
        existing_metadata = _load_metadata(metadata_path)
        metadata = _extract_epub_metadata(source_path)
        _emit(
            log_callback,
            f"📋 Found {len(metadata)} source metadata field(s); "
            "chapter extraction skipped",
        )

        field_settings = resolve_metadata_field_settings(
            _json_mapping(env.get("TRANSLATE_METADATA_FIELDS")),
            source_path,
        )
        if not any(field_settings.values()):
            field_settings["title"] = True

        # Explicit Library metadata translation is a regeneration. Selected
        # fields start with current OPF values; unselected translations survive.
        for field_name, should_translate in field_settings.items():
            if should_translate or field_name == "_per_epub":
                continue
            original_key, translated_key = _translated_keys(field_name)
            for key in (field_name, original_key, translated_key):
                if key in existing_metadata:
                    metadata[key] = copy.deepcopy(existing_metadata[key])
        for key in ("chapter_count", "chapter_titles"):
            if key in existing_metadata:
                metadata[key] = copy.deepcopy(existing_metadata[key])

        mode = normalize_metadata_mode(
            env.get("METADATA_TRANSLATION_MODE", "together")
        )
        try:
            from TransateKRtoEN import ProgressManager

            progress_manager = ProgressManager(output_folder)
            plan = progress_manager.configure_metadata_progress(
                mode,
                metadata,
                field_settings,
                metadata_path,
                title_allowed=True,
                source_path=source_path,
                reset_all=True,
            )
            for phase in plan:
                for field_name in phase["fields"]:
                    phase_by_field[field_name] = phase["key"]
            progress_manager.save()
        except Exception as progress_error:
            _emit(
                log_callback,
                f"⚠️ Progress tracking unavailable: {progress_error}",
            )
            progress_manager = None

        if client_factory is None:
            from unified_api_client import UnifiedClient

            client_factory = UnifiedClient
        if translator_factory is None:
            from metadata_batch_translator import MetadataTranslator

            translator_factory = MetadataTranslator

        model = str(env.get("MODEL") or "")
        api_key = str(
            env.get("API_KEY")
            or env.get("OPENAI_OR_Gemini_API_KEY")
            or env.get("OPENAI_API_KEY")
            or "dummy-key-not-required"
        )
        client = client_factory(
            api_key=api_key,
            model=model,
            output_dir=output_folder,
        )

        mt_config = {
            "_prefer_explicit_config": True,
            "_glossary_settings": dict(env),
            "quiet": True,
            "output_dir": output_folder,
            "source_path": source_path,
            "glossary_path": str(env.get("MANUAL_GLOSSARY") or ""),
            "metadata_system_prompt": str(
                env.get("METADATA_SYSTEM_PROMPT") or ""
            ),
            "metadata_field_prompts": _json_mapping(
                env.get("METADATA_FIELD_PROMPTS")
            ),
            "metadata_batch_prompt": str(
                env.get("METADATA_BATCH_PROMPT") or ""
            ),
            "output_language": str(
                env.get("OUTPUT_LANGUAGE") or "English"
            ),
            "lang_prompt_behavior": str(
                env.get("LANG_PROMPT_BEHAVIOR") or "auto"
            ),
            "forced_source_lang": str(
                env.get("FORCED_SOURCE_LANG") or "Korean"
            ),
            "source_language": _source_language(metadata),
            "temperature": _float_setting(
                env, "TRANSLATION_TEMPERATURE", 0.3
            ),
            "max_tokens": _int_setting(
                env, "MAX_OUTPUT_TOKENS", 4096
            ),
        }

        selected_fields = {
            field_name: True
            for field_name, should_translate in field_settings.items()
            if (
                field_name != "_per_epub"
                and should_translate
                and field_name in metadata
                and metadata.get(field_name)
            )
        }
        _emit(
            log_callback,
            "🌐 Translating fields: "
            + (", ".join(selected_fields) if selected_fields else "none"),
        )
        failed_fields: set[str] = set()

        if (
            mode != "together"
            and "title" in selected_fields
            and not stop_check()
        ):
            title_phase = phase_by_field.get("title")
            if title_phase:
                set_progress("in_progress", key=title_phase)
            translated_title, title_succeeded = _translate_title(
                metadata["title"],
                client,
                env,
                stop_check,
            )
            if title_succeeded:
                original_title = metadata["title"]
                if translated_title != original_title:
                    metadata["original_title"] = original_title
                    metadata["title"] = translated_title
                metadata["title_translated"] = True
                if title_phase:
                    set_progress("completed", key=title_phase)
            else:
                failed_fields.add("title")
                if title_phase:
                    set_progress(
                        "failed",
                        key=title_phase,
                        error="Title translation request failed",
                    )
            selected_fields.pop("title", None)

        if stop_check():
            if progress_manager is not None:
                progress_manager.reset_in_progress_metadata()
                progress_manager.save()
            return False

        if selected_fields:
            grouped_phase = next(
                (
                    phase_by_field.get(field_name)
                    for field_name in selected_fields
                    if phase_by_field.get(field_name)
                ),
                None,
            )
            if mode != "parallel" and grouped_phase:
                set_progress("in_progress", key=grouped_phase)

            def field_progress(field_name, status, error=None):
                phase_key = phase_by_field.get(field_name)
                if phase_key:
                    set_progress(status, key=phase_key, error=error)

            translator = translator_factory(
                client,
                mt_config,
                stop_check_fn=stop_check,
                progress_callback=(
                    field_progress if mode == "parallel" else None
                ),
            )
            translator_mode = "parallel" if mode == "parallel" else "together"
            translated_metadata = translator.translate_metadata(
                metadata,
                selected_fields,
                mode=translator_mode,
            )
            completed_fields = set(
                getattr(translator, "last_completed_fields", set())
            )
            for field_name in completed_fields.intersection(selected_fields):
                translated_value = translated_metadata.get(
                    field_name, metadata.get(field_name)
                )
                original_value = metadata.get(field_name)
                original_key, translated_key = _translated_keys(field_name)
                if translated_value != original_value:
                    metadata[original_key] = copy.deepcopy(original_value)
                    metadata[field_name] = translated_value
                metadata[translated_key] = True
                if mode == "parallel":
                    field_progress(field_name, "completed")

            unresolved = set(selected_fields) - completed_fields
            failed_fields.update(unresolved)
            if unresolved:
                error = (
                    "Metadata fields did not complete: "
                    + ", ".join(sorted(unresolved))
                )
                if mode == "parallel":
                    for field_name in unresolved:
                        field_progress(field_name, "failed", error)
                elif grouped_phase:
                    set_progress("failed", key=grouped_phase, error=error)
            elif mode != "parallel" and grouped_phase:
                set_progress("completed", key=grouped_phase)

        if stop_check():
            if progress_manager is not None:
                progress_manager.reset_in_progress_metadata()
                progress_manager.save()
            return False

        _save_metadata(metadata_path, metadata)
        if (
            progress_manager is not None
            and progress_manager.refresh_metadata_content_hash(metadata_path)
        ):
            progress_manager.save()
        _emit(log_callback, "💾 Saved metadata.json")
        if failed_fields:
            _emit(
                log_callback,
                "❌ Incomplete fields: " + ", ".join(sorted(failed_fields)),
            )
            return False
        return True
    except Exception as exc:
        if progress_manager is not None:
            try:
                for phase_key, phase_entry in progress_manager.metadata_entries():
                    if str(phase_entry.get("status", "")).lower() == "in_progress":
                        set_progress("failed", key=phase_key, error=exc)
            except Exception:
                pass
        if stop_check():
            _emit(log_callback, "⏹️ Metadata translation stopped")
        else:
            _emit(log_callback, f"❌ Metadata translation failed: {exc}")
        return False


def main(job_path: str | None = None) -> int:
    if not job_path:
        print("[ERROR] Metadata worker job file was not provided", flush=True)
        return 2

    try:
        with open(job_path, "r", encoding="utf-8") as job_file:
            job = json.load(job_file)

        source_path = os.path.abspath(str(job.get("source_path") or ""))
        env_vars = job.get("env") or {}
        if not source_path or not os.path.isfile(source_path):
            print(
                f"[ERROR] Metadata worker EPUB was not found: {source_path}",
                flush=True,
            )
            return 2
        if not isinstance(env_vars, dict):
            raise ValueError("Metadata worker environment must be an object")

        # large_env keeps oversized prompts/API settings out of the platform's
        # small environment-value limits while preserving normal os.getenv()
        # behavior inside the translation backend.
        import large_env

        large_env.update_env({
            str(key): "" if value is None else str(value)
            for key, value in env_vars.items()
        })
        os.environ["METADATA_ONLY"] = "1"
        os.environ["EPUB_PATH"] = source_path
        os.environ.pop("TRANSLATION_CANCELLED", None)
        os.environ["GRACEFUL_STOP"] = "0"
        os.environ["GRACEFUL_STOP_COMPLETED"] = "0"

        output_root = str(env_vars.get("OUTPUT_DIRECTORY") or "").strip()
        output_folder = os.path.join(
            os.path.abspath(output_root) if output_root else os.getcwd(),
            os.path.splitext(os.path.basename(source_path))[0],
        )
        try:
            os.makedirs(output_folder, exist_ok=True)
            with open(
                os.path.join(output_folder, "source_epub.txt"),
                "w",
                encoding="utf-8",
            ) as source_pointer:
                source_pointer.write(source_path)
        except Exception as exc:
            print(
                f"⚠️ Could not save source EPUB reference: {exc}",
                flush=True,
            )

        sys.argv = ["TransateKRtoEN.py", source_path]
        from TransateKRtoEN import main as translation_main

        result = translation_main()
        return 1 if result is False else 0
    except Exception as exc:
        print(f"[ERROR] Metadata worker failed: {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
