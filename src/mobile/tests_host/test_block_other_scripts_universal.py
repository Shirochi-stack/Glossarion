"""Universal "Block other scripts in translations (adaptive, all models)" on mobile (owner 2026-10-28):
the on-device ban and the API setting share one table (src/script_ban_core.py), and the config setting
``block_other_scripts`` reaches the mobile Settings through the generated schema (Anti-Duplicate
Parameters · Logit Bias, with its warning as the tile's help line) and the run env.

Run from src/mobile:
    .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore tests_host/test_block_other_scripts_universal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import script_ban_core  # noqa: E402
from glossarion_mobile.services.local_llm import script_ban  # noqa: E402
from glossarion_mobile.services.local_llm import service as svc  # noqa: E402


def test_on_device_ban_uses_the_shared_desktop_table():
    assert script_ban.SCRIPTS is script_ban_core.SCRIPTS
    assert script_ban.banned_scripts is script_ban_core.banned_scripts
    assert script_ban.target_scripts is script_ban_core.target_scripts
    # the on-device switch stays its own Prefs key
    assert svc.BLOCK_CJK_KEY == "local_llm_block_cjk"


def test_universal_setting_is_in_the_mobile_schema():
    import settings_schema

    spec = settings_schema.spec("block_other_scripts")
    assert spec.label == "Block other scripts in translations (adaptive, all models)"
    assert settings_schema.effective_default("block_other_scripts") is False
    assert settings_schema.section_of("block_other_scripts").id == "other.anti_duplicate.logit_bias"
    assert "BLOCK_OTHER_SCRIPTS" in settings_schema.env_names("block_other_scripts")

    from glossarion_mobile.ui.settings.model import help_line

    line = help_line(spec)
    assert line.startswith("⚠️ Can cause errors or retries")
    assert not line.endswith("…")


def test_universal_setting_reaches_the_run_env():
    from run_env import RunEnvMixin

    import inspect
    source = inspect.getsource(RunEnvMixin)
    assert "'BLOCK_OTHER_SCRIPTS': '1' if getattr(self, 'block_other_scripts_var', False) else '0'" in source
