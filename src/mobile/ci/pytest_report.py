"""List every skipped pytest case in CI logs and the GitHub job summary."""

from __future__ import annotations

import argparse
import glob
import html
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def report(paths: list[Path]) -> str:
    lines = ["## Pytest skip report", ""]
    if not paths:
        print("No pytest XML reports were produced; see the test step for errors.")
        return "\n".join(lines + ["No pytest XML reports were produced; see the test step for errors.", ""])
    for path in paths:
        cases = list(ET.parse(path).getroot().iter("testcase"))
        skipped = [(case, case.find("skipped")) for case in cases if case.find("skipped") is not None]
        print(f"{path.name}: {len(cases)} test cases, {len(skipped)} skipped")
        lines.extend([f"### {html.escape(path.name)}", "", f"{len(cases)} test cases; {len(skipped)} skipped.", ""])
        for case, skip in skipped:
            name = "::".join(filter(None, (case.get("classname"), case.get("name"))))
            reason = skip.get("message") or (skip.text or "").strip() or "No reason supplied"
            print(f"SKIPPED {name}: {reason}")
            lines.extend([f"- <code>{html.escape(name)}</code>: {html.escape(reason)}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="JUnit XML paths or glob patterns")
    args = parser.parse_args()
    paths = sorted({Path(path) for pattern in args.reports for path in glob.glob(pattern)})
    summary = report(paths)
    if destination := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(destination).open("a", encoding="utf-8") as stream:
            stream.write(summary + "\n")


if __name__ == "__main__":
    main()
