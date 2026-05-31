#!/usr/bin/env python3
"""Export Glossarion's Python model catalog for the browser extension."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "extension" / "glossarion" / "modelOptions.js"


def main() -> int:
    sys.path.insert(0, str(SRC))
    from model_options import get_model_options

    models = get_model_options()
    OUT.write_text(
        "// Generated from ../../src/model_options.py. Do not edit by hand.\n"
        f"export const MODEL_OPTIONS = {json.dumps(models, ensure_ascii=False, indent=2)};\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT} with {len(models)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
