# -*- coding: utf-8 -*-
"""Shadow-mode recorder for glossary term matching.

When GLOSSARY_MATCH_ENGINE=shadow, both the legacy and the tiered matcher run
on every entry, the *legacy* answer is the one that takes effect, and every
disagreement is appended here as JSONL.  That makes the change reviewable:
you can see exactly which entries the new matcher would drop, and why, on
real books, before anything about translation output changes.

Kept separate from glossary_matching so that module stays pure and free of
file I/O and global state.
"""

import atexit
import collections
import csv
import json
import os
import threading
import time

_FLUSH_EVERY = 200
SAMPLES_PER_TERM = 3
_LOG_SUBDIR = "glossary_match_shadow"


def default_log_dir():
    """Where shadow logs go when nothing overrides it.

    translator_gui resolves a writable logs directory at startup (next to the
    executable when frozen, LOCALAPPDATA otherwise) and exports it as
    GLOSSARION_LOG_DIR for helper modules. Honour it: a bare relative "logs/"
    resolves against the current working directory, which in a frozen build
    is wherever the user happened to launch from and may not be writable.

    Resolved per call, not at import, because the GUI sets the variable after
    this module may already have been imported.
    """
    base = os.environ.get("GLOSSARION_LOG_DIR")
    if base:
        return os.path.join(os.path.expanduser(base), _LOG_SUBDIR)
    return os.path.join("logs", _LOG_SUBDIR)


# Kept for callers that want the path without a run in progress.
DEFAULT_LOG_DIR = default_log_dir()


class ShadowRecorder:
    """Buffered JSONL appender. Safe to call from translation worker threads."""

    def __init__(self, log_dir=None, run_id=None):
        self.log_dir = log_dir or default_log_dir()
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self._lock = threading.Lock()
        self._buffer = []
        self._path = None
        self._failed = False
        self.counts = {"decisions": 0, "agree": 0, "newly_dropped": 0, "newly_kept": 0}

    @property
    def path(self):
        if self._path is None:
            self._path = os.path.join(self.log_dir, f"{self.run_id}.jsonl")
        return self._path

    def record(self, *, legacy, new, term, translated_name="", entry_type="",
               is_character=False, tier=-1, rule="", reject_reason="",
               source_text="", chapter_ref=None, glossary_path=None):
        """Log one decision.

        Call this for EVERY decision, not only disagreements: the agreement
        count is what makes the report's denominator meaningful ("42 of 900
        entries would change" rather than just "42 entries"). Agreements are
        counted and dropped; only disagreements are written.
        """
        with self._lock:
            self.counts["decisions"] += 1
            if legacy == new:
                self.counts["agree"] += 1
                return
            self.counts["newly_dropped" if legacy else "newly_kept"] += 1
            # Deferred until we know it is needed: scanning the chapter for a
            # context snippet on every agreeing entry would be pure waste.
            context = context_around(source_text, term)
            self._buffer.append({
                "ts": int(time.time()),
                "chapter": _chapter_fields(chapter_ref),
                "glossary_path": str(glossary_path or ""),
                "entry_type": str(entry_type or ""),
                "raw_name": str(term or ""),
                "translated_name": str(translated_name or ""),
                "is_character": bool(is_character),
                "legacy": bool(legacy),
                "new": bool(new),
                "new_tier": int(tier),
                "new_rule": str(rule or ""),
                "new_reject_reason": str(reject_reason or ""),
                "context": str(context or ""),
            })
            if len(self._buffer) >= _FLUSH_EVERY:
                self._flush_locked()

    def flush(self):
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        if not self._buffer or self._failed:
            self._buffer.clear()
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                for record in self._buffer:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            # Shadow logging must never break a translation run.
            self._failed = True
            print(f"⚠️ Glossary shadow log disabled ({exc})")
        finally:
            self._buffer.clear()


def _chapter_fields(chapter_ref):
    if isinstance(chapter_ref, dict):
        return {
            "num": chapter_ref.get("chapter_num"),
            "file": os.path.basename(str(chapter_ref.get("chapter_file") or "")),
        }
    return {"num": chapter_ref, "file": ""}


_recorder = None
_recorder_lock = threading.Lock()


def finish(recorder=None):
    """Flush the log and refresh report.md / verdicts.csv beside it.

    Registered with atexit so that turning on the checkbox, running a book and
    opening logs/glossary_match_shadow/ is the whole workflow — the raw JSONL
    is machine format and nobody should have to read it, or remember to run a
    separate command, to see what the tiered matcher would change.
    """
    recorder = recorder or _recorder
    if recorder is None:
        return None
    recorder.flush()
    try:
        return write_report(recorder.log_dir)
    except Exception as exc:  # pragma: no cover - reporting must not break exit
        print(f"⚠️ Glossary shadow report not written ({exc})")
        return None


def get_recorder(log_dir=None):
    """Process-wide recorder, created on first use."""
    global _recorder
    with _recorder_lock:
        if _recorder is None:
            _recorder = ShadowRecorder(log_dir=log_dir)
            atexit.register(finish, _recorder)
        return _recorder


def reset_recorder():
    """Drop the process-wide recorder. For tests."""
    global _recorder
    with _recorder_lock:
        if _recorder is not None:
            _recorder.flush()
        _recorder = None


# ─── Report generation ───────────────────────────────────────────────────
# Lives here rather than in tools/ so the report can be written
# automatically when a shadow run finishes: the point of the checkbox is
# that you turn it on, run a book, and open the folder.


def load_records(log_dir):
    records = []
    if not os.path.isdir(log_dir):
        return records
    for name in sorted(os.listdir(log_dir)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(log_dir, name)
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    print(f"⚠️ skipping malformed line {path}:{line_no}")
    return records


def group_by_term(records):
    groups = collections.defaultdict(lambda: {
        "count": 0, "entry_type": "", "translated_name": "",
        "reasons": collections.Counter(), "samples": [], "chapters": set(),
        "glossary_path": "",
    })
    for record in records:
        group = groups[record.get("raw_name", "")]
        group["count"] += 1
        group["entry_type"] = group["entry_type"] or record.get("entry_type", "")
        group["translated_name"] = group["translated_name"] or record.get("translated_name", "")
        group["glossary_path"] = group["glossary_path"] or record.get("glossary_path", "")
        reason = record.get("new_reject_reason") or record.get("new_rule") or "?"
        group["reasons"][reason] += 1
        chapter = record.get("chapter") or {}
        group["chapters"].add(str(chapter.get("file") or chapter.get("num") or ""))
        if len(group["samples"]) < SAMPLES_PER_TERM and record.get("context"):
            group["samples"].append(record["context"])
    return groups


def _table(groups, title, empty_note):
    lines = [f"## {title}", ""]
    if not groups:
        lines += [empty_note, ""]
        return lines
    lines += ["| term | type | occurrences | chapters | reason |",
              "| --- | --- | --- | --- | --- |"]
    for term, group in sorted(groups.items(), key=lambda kv: -kv[1]["count"]):
        reason = group["reasons"].most_common(1)[0][0] if group["reasons"] else "?"
        lines.append(
            f"| `{term}` | {group['entry_type'] or '-'} | {group['count']} | "
            f"{len(group['chapters'])} | {reason} |"
        )
    lines.append("")
    lines.append("<details><summary>Context samples</summary>")
    lines.append("")
    for term, group in sorted(groups.items(), key=lambda kv: -kv[1]["count"]):
        if not group["samples"]:
            continue
        lines.append(f"**`{term}`**")
        for sample in group["samples"]:
            lines.append(f"- {sample}")
        lines.append("")
    lines.append("</details>")
    lines.append("")
    return lines


def build_report(records):
    dropped = group_by_term([r for r in records if r.get("legacy") and not r.get("new")])
    kept = group_by_term([r for r in records if r.get("new") and not r.get("legacy")])

    chapters = collections.defaultdict(lambda: {"dropped": 0, "kept": 0})
    for record in records:
        chapter = record.get("chapter") or {}
        key = str(chapter.get("file") or chapter.get("num") or "?")
        chapters[key]["dropped" if record.get("legacy") else "kept"] += 1

    lines = [
        "# Glossary match shadow report",
        "",
        "Every row is an entry the two matchers disagreed about. Agreements are",
        "not logged, so this is the complete set of changes that turning on",
        "**Precise Term Matching** would make.",
        "",
        "## Headline",
        "",
        f"- disagreements logged: **{len(records)}**",
        f"- distinct terms newly **dropped**: **{len(dropped)}**",
        f"- distinct terms newly **kept**: **{len(kept)}**",
        f"- chapters touched: **{len(chapters)}**",
        "",
    ]

    heavy = sorted(chapters.items(), key=lambda kv: -kv[1]["dropped"])[:10]
    lines += ["## Cliff watch", "",
              "Chapters losing the most entries. If one of these would end up at",
              "zero matches, the relaxation ladder catches it — but a chapter near",
              "zero is worth reading before switching engines.", "",
              "| chapter | dropped | kept |", "| --- | --- | --- |"]
    for name, counts in heavy:
        lines.append(f"| {name} | {counts['dropped']} | {counts['kept']} |")
    lines.append("")

    lines += _table(dropped, "Newly dropped (precision wins, or regressions)",
                    "_Nothing would be dropped._")
    lines += _table(kept, "Newly kept (recall wins from normalization/spacing/honorifics)",
                    "_Nothing new would be kept._")

    reasons = collections.Counter()
    for record in records:
        if record.get("legacy") and not record.get("new"):
            reasons[record.get("new_reject_reason") or "?"] += 1
    lines += ["## Reject-reason histogram", "",
              "Which gate is doing the work. A reason dominating unexpectedly is",
              "the signal that its table needs tuning.", "",
              "| reason | count |", "| --- | --- |"]
    for reason, count in reasons.most_common():
        lines.append(f"| {reason} | {count} |")
    lines.append("")

    return "\n".join(lines), dropped, kept


def write_verdicts(path, dropped, kept):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["term", "entry_type", "direction", "occurrences",
                         "reason", "sample", "verdict", "glossary_path"])
        for direction, groups in (("dropped", dropped), ("kept", kept)):
            for term, group in sorted(groups.items(), key=lambda kv: -kv[1]["count"]):
                reason = group["reasons"].most_common(1)[0][0] if group["reasons"] else ""
                writer.writerow([
                    term, group["entry_type"], direction, group["count"], reason,
                    group["samples"][0] if group["samples"] else "", "",
                    group["glossary_path"],
                ])


def write_report(log_dir=None, out_dir=None):
    """Aggregate a shadow log directory into report.md + verdicts.csv."""
    log_dir = log_dir or default_log_dir()
    records = load_records(log_dir)
    if not records:
        return None
    out_dir = out_dir or log_dir
    os.makedirs(out_dir, exist_ok=True)
    report, dropped, kept = build_report(records)
    report_path = os.path.join(out_dir, 'report.md')
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write(report)
    write_verdicts(os.path.join(out_dir, 'verdicts.csv'), dropped, kept)
    return report_path


def context_around(text, term, width=30):
    """The source text surrounding a hit, so a reviewer can judge it.

    This is what makes the shadow report usable without opening the chapter:
    it shows *why* the legacy matcher kept the entry.
    """
    if not text or not term:
        return ""
    idx = text.find(term)
    if idx < 0:
        # Legacy may have matched on a token rather than the whole term.
        for token in str(term).split():
            idx = text.find(token)
            if idx >= 0:
                term = token
                break
        else:
            return ""
    start = max(0, idx - width)
    end = min(len(text), idx + len(term) + width)
    snippet = text[start:end].replace("\n", " ")
    return ("…" if start else "") + snippet + ("…" if end < len(text) else "")


# ─── Reviewed overrides ──────────────────────────────────────────────────────

def read_verdicts(path):
    """Rows of verdicts.csv that carry a decision."""
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            verdict = (row.get("verdict") or "").strip().lower()
            if verdict in ("keep", "drop"):
                rows.append({
                    "term": (row.get("term") or "").strip(),
                    "verdict": verdict,
                    "direction": (row.get("direction") or "").strip(),
                    "glossary_path": (row.get("glossary_path") or "").strip(),
                })
    return [r for r in rows if r["term"]]


def apply_verdicts(verdicts_path, default_glossary=None):
    """Turn reviewed verdicts into per-glossary override files.

    `keep` means "this entry is relevant, whatever the matcher decides";
    `drop` means the opposite. Rows left blank change nothing, so you can
    review a few terms at a time rather than all of them at once.

    Written beside each glossary, grouped by the glossary the term came from,
    because two books can legitimately disagree about the same string.
    Returns {allowlist_path: {"always_keep": [...], "always_drop": [...]}}.
    """
    from glossary_matching import allowlist_path_for

    by_glossary = collections.defaultdict(
        lambda: {"always_keep": set(), "always_drop": set()})
    skipped = []
    for row in read_verdicts(verdicts_path):
        glossary = row["glossary_path"] or default_glossary or ""
        if not glossary:
            skipped.append(row["term"])
            continue
        bucket = "always_keep" if row["verdict"] == "keep" else "always_drop"
        by_glossary[glossary][bucket].add(row["term"])

    written = {}
    for glossary, buckets in by_glossary.items():
        path = allowlist_path_for(glossary)
        if not path:
            continue
        existing = {"always_keep": [], "always_drop": []}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    existing.update(json.load(handle))
            except (OSError, ValueError):
                pass  # a corrupt file is replaced, not inherited
        merged = {
            "note": (
                "Reviewed glossary-match overrides. Terms here are forced "
                "regardless of what the matcher decides. Generated by "
                "tools/glossary_match_report.py --apply-verdicts; safe to edit "
                "by hand."
            ),
            "glossary": os.path.basename(glossary),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for bucket in ("always_keep", "always_drop"):
            other = "always_drop" if bucket == "always_keep" else "always_keep"
            # A later verdict wins over an earlier opposite one.
            kept = [t for t in existing.get(bucket) or []
                    if t not in buckets[other]]
            merged[bucket] = sorted(set(kept) | buckets[bucket])
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(merged, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            written[path] = merged
        except OSError as exc:
            print(f"⚠️ Could not write {path}: {exc}")

    if skipped:
        print(f"⚠️ {len(skipped)} verdict(s) had no glossary_path and were "
              f"skipped: {', '.join(skipped[:5])}")
        print("   Re-run with --glossary to say which glossary they belong to.")
    return written
