"""RPGMAKER: translate an RPG Maker game through GTool (Tools › RPG Maker, UI_SPEC §4.10).

The desktop picks the game's ``.exe``; mobile picks the game folder or a ZIP of it. The
shared mobile entry of ``rpgmaker_job`` turns that into a writable game folder (a ZIP is
extracted, a read-only folder copied, into ``params["work_dir"]``) and registers it for the
run; the run itself is the desktop one - ``_prepare_translation_run([game_dir])`` +
``_translation_worker`` route the registered folder to
``RpgMakerJobMixin._process_rpgmaker_game`` (``rpgmaker_handler`` extraction, chunked
translation with resume in ``<game>/GTool_Translation``, ``apply_translations`` into the
game's data folder; Image output mode translates the game's images instead):

    game_dir = owner._register_rpgmaker_game_input(source, work_dir)
    request = owner._prepare_translation_run([game_dir])
    owner._translation_worker(request)

params: ``work_dir`` (the writable folder games are extracted / copied into). Input: the
game folder or ZIP. A resumed job continues from ``GTool_Translation`` (the extracted
folder is kept between runs).
"""

from __future__ import annotations

import os
from typing import Any

from glossarion_mobile.job_kinds import owner_method
from glossarion_mobile.job_kinds.translate import check_inputs, run_translation

__all__ = ["KINDS", "run"]


def run(ctx: Any) -> dict:
    from glossarion_mobile.services.jobs import JobError

    files = check_inputs(ctx.inputs)
    if len(files) != 1:
        raise JobError("RPG Maker translation takes one game folder or ZIP.")
    source = files[0]
    if not (os.path.isdir(source) or source.lower().endswith((".zip", ".exe"))):
        raise JobError("Pick the game's folder or a ZIP of it.")
    work_dir = str((ctx.params or {}).get("work_dir") or "") or None
    register = owner_method(ctx.owner, "_register_rpgmaker_game_input")
    ctx.phase("Preparing game")
    try:
        game_dir = register(source, work_dir)
    except ValueError as exc:
        raise JobError(str(exc)) from exc
    ctx.log(f"🎮 Game folder: {game_dir}")
    ctx.set_result(rpgmaker_game_dir=str(game_dir))
    result = run_translation(ctx, [str(game_dir)])
    translation_dir = os.path.join(str(game_dir), "GTool_Translation")
    if os.path.isdir(translation_dir):
        ctx.set_output_dir(translation_dir)
    return result


KINDS = {
    "rpgmaker": {"verb": "Translating game", "icon": "VIDEOGAME_ASSET", "stop_kind": "translation", "run": run},
}
