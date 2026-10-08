"""Desktop pin (devfix4 item 14): Other Settings › Response Handling shows the same streaming notes.

The two notes of the "Real-time Translation (Streaming)" group now come from shared constants
(``settings_rules.STREAMING_TRUNCATION_WARNING`` / ``FORCED_STREAM_NOTE``), which Glossarion Mobile's
one Streaming switch shows too (owner-approved move, 2026-10-08). The desktop dialog must render
exactly the text it rendered before: the constants equal the old literals, other_settings keeps no
second copy, and the real section builder (offscreen Qt, in a subprocess with the user folders
pointed at pytest's tmp dir) shows both labels with their old styles.
"""

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
#: The literal texts other_settings.py had before the move (HEAD 55c46555).
WARNING = "⚠️ Enabling this may result in silent truncation"
NOTE = ("\U0001f510 AuthGPT, AuthGrok, AuthGem, AuthCD, Arena, Antigravity, and OcAgy always stream "
        "— this controls batch log visibility")


def test_shared_constants_keep_the_desktop_texts():
    import settings_rules

    assert settings_rules.STREAMING_TRUNCATION_WARNING == WARNING
    assert settings_rules.FORCED_STREAM_NOTE == NOTE


def test_other_settings_reads_them_from_settings_rules_only():
    source = (SRC / "other_settings.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    builder = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "_create_response_handling_section")
    label_args = [ast.unparse(call.args[0]) for call in ast.walk(builder)
                  if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "QLabel"
                  and call.args]
    assert "settings_rules.STREAMING_TRUNCATION_WARNING" in label_args
    assert "settings_rules.FORCED_STREAM_NOTE" in label_args
    # one copy of each text: the shared constant
    assert "may result in silent truncation" not in source
    assert "this controls batch log visibility" not in source


_RENDER = textwrap.dedent(r"""
    import json, os, sys
    sys.path.insert(0, sys.argv[1])
    from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QWidget
    app = QApplication.instance() or QApplication([])
    import other_settings

    class Owner(QWidget):
        def __init__(self):
            super().__init__()
            self.config = {}

    owner = Owner()
    other_settings.setup_other_settings_methods(owner)
    parent = QWidget()
    QGridLayout(parent)
    other_settings._create_response_handling_section(owner, parent)
    labels = [{"text": w.text(), "style": w.styleSheet(), "wrap": w.wordWrap()} for w in parent.findChildren(QLabel)]
    print("LABELS=" + json.dumps(labels))
""")


def test_the_response_handling_section_renders_the_same_notes(tmp_path):
    pytest.importorskip("PySide6")
    env = dict(os.environ)
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "GLOSSARION_LIBRARY_DIR", "OUTPUT_DIRECTORY",
                 "GLOSSARION_DATA_DIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        folder = tmp_path / name.lower()
        folder.mkdir()
        env[name] = str(folder)
    env.update(QT_QPA_PLATFORM="offscreen", GLOSSARION_HTTP_LOG="0", PYTHONIOENCODING="utf-8")
    script = tmp_path / "render_response_section.py"
    script.write_text(_RENDER, encoding="utf-8")
    result = subprocess.run([sys.executable, str(script), str(SRC)], cwd=str(tmp_path), env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    line = next((ln for ln in result.stdout.splitlines() if ln.startswith("LABELS=")), None)
    assert line is not None, (result.returncode, result.stdout[-2000:], result.stderr[-2000:])
    labels = json.loads(line[len("LABELS="):])
    warning = [label for label in labels if label["text"] == WARNING]
    note = [label for label in labels if label["text"] == NOTE]
    assert len(warning) == 1 and warning[0]["style"] == "color: #f59e0b; font-size: 9pt;"
    assert len(note) == 1 and note[0]["style"] == "color: #6b7280; font-size: 9pt; font-style: italic;"
    assert note[0]["wrap"] is True
    texts = [label["text"] for label in labels]
    assert texts.index("Real-time Translation (Streaming)") < texts.index(WARNING) < texts.index(NOTE)
