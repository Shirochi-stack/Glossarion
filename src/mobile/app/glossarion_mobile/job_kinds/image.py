"""TRANSLATE_IMAGE: standalone image / video translation outside a chat.

Desktop "Standalone image / video translation" (main window, an image or video selected):
the run's file loop hands every image / video input to ``_process_image_file`` (now
``image_job.ImageJobMixin``, moved verbatim), with one combined output folder when several
images are selected. The output mode decides what the client does with each image (Vision
OCR + translation, image edit / generation, video, audio). This adapter selects that output
mode on the job's owner the way the desktop selector does (``settings_rules.output_mode_flags``
= ``other_settings._set_output_mode``: the owner attributes, then the environment it exports)
and runs the shared ``run_translation_thread`` body (``job_kinds.translate.run_translation``),
so the per-image progress (``ImageProgressManager``), payloads and generated media follow the
desktop exactly. Generated media in the output folders are listed as job outputs.

The chat does not use this kind: an image sent in the chat is a ``direct_text`` run (the same
pipeline, Direct Text environment). It serves surfaces that translate picked images directly
(Tools / Open-with).

params: ``output_mode`` (default ``vision``); inputs: image / video paths.
"""

from __future__ import annotations

import os
from typing import Any

from glossarion_mobile.job_kinds.translate import check_inputs, run_translation

__all__ = ["KINDS", "VIDEO_INPUT_EXTENSIONS", "apply_output_mode", "generated_media", "media_input_extensions", "run"]

#: The worker's ``image_extensions | video_extensions`` (translation_pipeline: images go to
#: ``_process_image_file``): ``IMAGE_ATTACHMENT_EXTENSIONS`` plus its video literal.
VIDEO_INPUT_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})


def _image_extensions() -> frozenset:
    try:
        from translation_pipeline import IMAGE_ATTACHMENT_EXTENSIONS  # shared (U3)

        return frozenset(IMAGE_ATTACHMENT_EXTENSIONS)
    except Exception:
        from glossarion_mobile.ui.chat.output_modes import IMAGE_ATTACHMENT_EXTENSIONS

        return frozenset(IMAGE_ATTACHMENT_EXTENSIONS)


def media_input_extensions() -> frozenset:
    return _image_extensions() | VIDEO_INPUT_EXTENSIONS


def apply_output_mode(owner: Any, mode: Any) -> str:
    """Select ``mode`` on ``owner`` like the desktop output-mode selector (``settings_rules``)."""
    from settings_rules import output_mode_flags  # shared (U4)

    flags = output_mode_flags(str(mode or "vision"))
    for name, value in flags.var_values.items():
        setattr(owner, name, value)
    config = getattr(owner, "config", None)
    if isinstance(config, dict):
        config.update(flags.config_values)
    os.environ.update(flags.env)  # the job scope restores the environment afterwards
    return flags.mode


def generated_media(owner: Any) -> list:
    """Media the run generated: ``_process_image_file`` records every saved image / video in the
    owner's ``generated_images`` (decoded ``data:`` images and moved ``[GENERATED_IMAGE]`` files)."""
    found: list = []
    for path in getattr(owner, "generated_images", None) or ():
        path = os.path.abspath(str(path))
        if os.path.isfile(path) and path not in found:
            found.append(path)
    return found


def run(ctx: Any) -> dict:
    from glossarion_mobile.services.jobs import JobError

    files = check_inputs(ctx.inputs or ctx.params.get("inputs"))
    allowed = media_input_extensions()
    unsupported = [os.path.basename(p) for p in files if os.path.splitext(p)[1].lower() not in allowed]
    if unsupported:
        raise JobError(f"Not an image or video: {', '.join(unsupported)}")
    apply_output_mode(ctx.owner, ctx.params.get("output_mode") or "vision")
    result = run_translation(ctx, files)
    for path in generated_media(ctx.owner):
        if path not in result["outputs"]:
            result["outputs"].append(path)
    return result


KINDS = {
    "translate_image": {"verb": "Translating", "icon": "IMAGE", "stop_kind": "translation", "run": run},
}
