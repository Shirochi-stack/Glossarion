"""Isolated metadata-only translation worker.

Each worker owns its environment and translation module state, allowing the
main GUI to translate metadata for several EPUBs concurrently without races
over ``os.environ``, ``sys.argv``, or the backend's module globals.
"""

from __future__ import annotations

import json
import os
import sys
import traceback


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
