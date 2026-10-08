"""From a chat send to a ``direct_text`` JobSpec (pure Python, no Flet, Python 3.10).

The desktop ``_InputOutputDialog._start_translation`` (translator_gui.py) does,
on the GUI thread, before it calls ``run_translation_thread()``:

1. records the user turn (``("user", text)`` or the ``user_file`` tuple), auto-titles
   the chat, clears the draft/attachment and saves the history;
2. makes ``tempfile.mkdtemp("glossarion_input_output_")``, materialises the per-run
   manual glossary, and writes typed text to ``direct_text_<ts>_<uuid8>.txt`` (or
   passes the attached file straight through / adapts markup to ``<stem>.txt``);
3. builds ``DirectTextRunOptions(...)`` and applies it to the main window;
4. sets ``OUTPUT_DIRECTORY``/``OUTPUT_DIR`` to the temp root plus ``DIRECT_TEXT_*``
   / ``ORDER_BATCH_REQUESTS_BY_SPINE``, then ``_apply_forced_streaming_environment()``
   and ``_apply_direct_text_runtime_environment()``.

On mobile step 1 runs in the chat (``ChatStoreAdapter``), step 2 here through the
dialog's own code (``direct_text_store.prepare_direct_text_input``), and steps 3-4 in
the ``direct_text`` job adapter on the job thread (``DirectTextRunOptions.apply_to``,
then ``direct_text_store.apply_direct_text_run_environment``), from the
JSON-serialisable ``params`` built by :func:`job_params`:

    params = {
        "chat_id": 3,                              # v2 session id
        "user_index": 7,                           # index of the user turn in the session
        "input_path": ..., "output_root": ...,     # what job_kinds/direct_text.py reads
        "is_attachment": True,
        "run": DirectTextRun.as_dict(),            # temp_root, source_path, expected_output, ...
        "options": {...},                          # DirectTextRunOptions(**options).apply_to(owner)
        "config_overrides": {"model": ..., ...},   # per-chat model/profile/target, applied to the
                                                   # config snapshot before HeadlessOwner is built
        "auto_accept_glossary": False,             # "Always accept generated glossaries" (mobile);
                                                   # JobService._job_ask answers the gate Yes when set
    }

The job adapter then runs ``owner._translation_worker(owner._prepare_translation_run())``
(the shared ``run_translation_thread`` body) and the chat finishes the run with
``direct_text_store.ChatStore.finish_run`` (the dialog's ``_finish_translation``).
Everything in ``params`` is plain JSON so the job can be checkpointed and resumed after
the app was killed.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional

from glossarion_mobile.services.jobs import AUTO_ACCEPT_GLOSSARY_PARAM
from glossarion_mobile.ui.chat.direct_text_rules import (
    MOBILE_EXTRA_ATTACHMENT_EXTENSIONS,
    DirectTextSettings,
    ManualGlossarySource,
    force_no_glossary_for_mode,
)

__all__ = [
    "DIRECT_TEXT_JOB_KIND",
    "GENERATE_MEDIA_JOB_KIND",
    "GENERATIVE_SENTINEL",
    "DirectTextRun",
    "attachment_record",
    "build_run_options",
    "job_params",
    "prepare_direct_text_run",
    "run_options_dict",
    "user_turn",
]

DIRECT_TEXT_JOB_KIND = "direct_text"
#: "Generate from prompt (no input)" (UI_SPEC §2.6): the desktop generative-only run as a chat job.
GENERATE_MEDIA_JOB_KIND = "generate_media"
#: ``translation_pipeline``'s no-input file sentinel (``selected_files == ["__generative_mode__"]``).
GENERATIVE_SENTINEL = "__generative_mode__"
TEMP_PREFIX = "glossarion_input_output_"  # the dialog's mkdtemp prefix


def attachment_record(path: Any) -> Optional[dict]:
    """The v2 attachment record for a picked file (desktop ``_set_attachment``)."""
    path = os.path.abspath(os.path.expanduser(str(path or "")))
    if not path or not os.path.isfile(path):
        return None
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    return {
        "path": path,
        "name": os.path.basename(path),
        "extension": os.path.splitext(path)[1].lower(),
        "size": size,
    }


def user_turn(text: str, attachment: Optional[Mapping[str, Any]], prompt_role: str) -> tuple:
    """The v2 message the send records (``user`` or ``user_file``)."""
    if attachment:
        return (
            "user_file",
            str(attachment.get("name") or os.path.basename(str(attachment.get("path") or ""))),
            str(attachment.get("path") or ""),
            int(attachment.get("size") or 0),
            str(text or ""),
            prompt_role,
        )
    return ("user", str(text or ""))


@dataclass
class DirectTextRun:
    """Run state the dialog keeps on ``self._run_*`` / ``self._temp_root`` (JSON-safe)."""

    temp_root: str
    source_path: str
    source_extension: str
    is_attachment: bool
    expected_output: str
    manual_glossary_path: str = ""
    started_at: float = 0.0
    output_mode: str = "text"
    attachment_prompt: str = ""
    attachment_prompt_role: str = "user"
    display_name: str = ""
    created_at: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DirectTextRun":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in dict(data or {}).items() if k in known})


def prepare_direct_text_run(
    *,
    text: str,
    attachment: Optional[Mapping[str, Any]],
    output_mode: str = "text",
    attachment_prompt_role: str = "user",
    manual_glossary: Optional[ManualGlossarySource] = None,
    temp_dir: Optional[str] = None,
) -> DirectTextRun:
    """Temp root, temp input and manual glossary, made by the dialog's own code.

    ``direct_text_store.prepare_direct_text_input`` is the ``_start_translation`` block
    (typed text -> ``direct_text_<ts>_<uuid8>.txt``, pass-through types handed over
    as-is, markup/subtitle/log files adapted to ``<stem>.txt``, the per-run manual
    glossary materialised in the temp root). Mobile passes ``temp_dir`` (an app folder
    that survives a restart, so Resume continues from ``translation_progress.json``) and
    the extra attachment types the chat accepts (ZIP, SDLXLIFF, MP4).

    Blocking file I/O: call it off the UI loop. Raises ``FileNotFoundError`` when the
    attachment or the selected manual glossary is gone (desktop messages).
    """
    from direct_text_store import prepare_direct_text_input  # shared (U3)

    text = str(text or "").strip()
    if attachment and not os.path.isfile(str(attachment.get("path") or "")):
        raise FileNotFoundError(f"The attached file no longer exists:\n{attachment.get('path')}")
    prepared = prepare_direct_text_input(
        text,
        dict(attachment) if attachment else None,
        manual_glossary.as_dict() if manual_glossary is not None else None,
        temp_parent=temp_dir,
        extra_pass_through_extensions=MOBILE_EXTRA_ATTACHMENT_EXTENSIONS,
    )
    display_name = ""
    if attachment:
        display_name = str(attachment.get("name") or os.path.basename(str(attachment.get("path") or "")))
    return DirectTextRun(
        temp_root=prepared["temp_root"],
        source_path=prepared["source_path"],
        source_extension=prepared["source_extension"],
        is_attachment=bool(prepared["is_attachment"]),
        expected_output=prepared["expected_output"],
        manual_glossary_path=prepared["manual_glossary_path"] or "",
        started_at=float(prepared["started_at"] or time.time()),
        output_mode=output_mode,
        attachment_prompt=text if attachment else "",
        attachment_prompt_role=attachment_prompt_role,
        display_name=display_name,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )


def run_options_dict(run: DirectTextRun, settings: DirectTextSettings) -> dict:
    """``DirectTextRunOptions`` fields exactly as ``_start_translation`` passes them."""
    return {
        "selected_files": [run.source_path],
        "force_stream_all": True,
        "archive_conversion_dir": os.path.join(run.temp_root, "_archive_input"),
        "force_multipass_off": bool(settings.force_multipass_off),
        "force_no_glossary": bool(
            force_no_glossary_for_mode(settings.glossary_override_mode, run.is_attachment)
            and not run.manual_glossary_path
        ),
        "manual_glossary_path": run.manual_glossary_path,
        "skip_thinking": bool(settings.disable_thinking),
        "attachment_prompt": run.attachment_prompt if run.is_attachment else "",
        "attachment_prompt_role": run.attachment_prompt_role,
        "skip_prompt_profile": bool(settings.skip_prompt_profile),
        "output_mode": run.output_mode,
    }


def build_run_options(run: DirectTextRun, settings: DirectTextSettings) -> Any:
    """``headless_owner.DirectTextRunOptions`` for this run (the shared U2 contract)."""
    from headless_owner import DirectTextRunOptions  # backend module (on sys.path after bootstrap)

    return DirectTextRunOptions(**run_options_dict(run, settings))


#: Per-chat overrides -> the config.json key the owner reads.
OVERRIDE_CONFIG_KEYS = {"model": "model", "profile": "active_profile", "target_language": "output_language"}


def config_overrides(overrides: Optional[Mapping[str, Any]]) -> dict:
    out = {}
    for name, key in OVERRIDE_CONFIG_KEYS.items():
        value = (overrides or {}).get(name)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
    if out.get("active_profile"):
        # a per-chat profile switches the extraction method like the desktop profile combo
        # (prompt_profiles.extraction_method_for_profile: *_BeautifulSoup / *_html2text)
        from glossarion_mobile.state.setting_writes import extraction_method_for_profile

        method = extraction_method_for_profile(out["active_profile"])
        if method is not None:
            out["text_extraction_method"] = method
    return out


def job_params(
    *,
    chat_id: Any,
    user_index: int,
    run: DirectTextRun,
    settings: DirectTextSettings,
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict:
    """``JobSpec.params`` of a ``direct_text`` job (see the module docstring).

    ``auto_accept_glossary`` is the chat's "Always accept generated glossaries" value captured at Send
    (a queued job, a Resume or a Retry keeps it): ``JobService._job_ask`` answers the glossary gate Yes
    for the job instead of showing the approval card."""
    return {
        "chat_id": chat_id,
        "user_index": int(user_index),
        "input_path": run.source_path,
        "output_root": run.temp_root,
        "is_attachment": bool(run.is_attachment),
        "run": run.as_dict(),
        "options": run_options_dict(run, settings),
        "config_overrides": config_overrides(overrides),
        AUTO_ACCEPT_GLOSSARY_PARAM: bool(settings.auto_accept_glossary),
    }


def job_title(run: DirectTextRun, chat_title: str = "") -> str:
    """The JobSpec title: the attachment name or the chat title (the strip prepends the verb,
    "Translating · Book.epub", UI_SPEC §1.7)."""
    if run.is_attachment and run.display_name:
        return run.display_name
    return chat_title or "Direct Text"
