"""Readable payload summary: "Payload Summary/logistics.html" built from the saved API payloads.

Other Settings > Payloads > "Create Readable Payload Summary". Reads every request payload
(``Payloads/<context>/<thread>/*_payload_*.json``) and every failed request
(``Payloads/failed_requests/*.json``), joins them with ``Logistics.json`` (API key, output folder),
and writes, beside the Payloads folder (so the Payloads cleanup never counts it):

* ``Payload Summary/logistics.html`` - one row per request: time, context, model, provider, key,
  parameters, anti-duplicate settings, retry, token usage, output folder and previews of the system /
  user / assistant text; search, filters, sorting, double-click a row for the details, right-click it for its files.
* ``Payload Summary/details/<n>.html`` - one page per request: every field, the full messages, the
  parameters / usage / debug data, and a link to the raw JSON file.

Old payloads are fine: a missing value shows N/A and a column no request has is left out. Pages are
built on worker threads (``workers``: the Parallel Extraction setting, never more than the CPU has).
"""

import html
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

SUMMARY_DIR = "Payload Summary"
INDEX_FILE = "logistics.html"
PREVIEW_CHARS = 160
DETAIL_MAX_CHARS = 200_000  # per message; the raw JSON link has the rest

# (key, column label): the index columns, in order
COLUMNS = (
    ("time", "Time"), ("kind", "Type"), ("context", "Context"), ("chapter", "Chapter"), ("model", "Model"),
    ("provider", "Provider"), ("key", "API key"), ("temperature", "Temp"), ("max_tokens", "Max tokens"),
    ("anti_duplicate", "Anti-duplicate"), ("retry", "Retry"), ("tokens", "Tokens in / out"),
    ("output_dir", "Output folder"), ("system", "System prompt"), ("user", "User prompt"),
    ("assistant", "Assistant"), ("error", "Error"),
)


def default_workers(config=None):
    """Threads for the summary: Other Settings > Parallel Extraction workers when it is on (1 when off),
    never more than the CPU's cores."""
    config = config if isinstance(config, dict) else {}
    cpus = os.cpu_count() or 1
    if not config.get("enable_parallel_extraction", True):
        return 1
    try:
        wanted = int(config.get("extraction_workers", 2))
    except (TypeError, ValueError):
        wanted = 2
    return max(1, min(wanted, cpus))


def summary_dir(payloads_dir):
    return os.path.join(os.path.dirname(os.path.abspath(payloads_dir)), SUMMARY_DIR)


def collect_files(payloads_dir):
    """Request payloads and failed requests, oldest first."""
    found = []
    for root, _dirs, files in os.walk(payloads_dir):
        for name in files:
            if not name.endswith(".json") or name.startswith("Logistics"):
                continue
            path = os.path.join(root, name)
            if "_payload_" in name or os.path.basename(root) == "failed_requests":
                found.append(path)
    found.sort(key=lambda p: (os.path.getmtime(p) if os.path.exists(p) else 0, p))
    return found


def _load_logistics(payloads_dir):
    """Logistics.json records by payload path (normcased)."""
    records = {}
    path = os.path.join(payloads_dir, "Logistics.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return records
    for record in data if isinstance(data, list) else ():
        if isinstance(record, dict) and record.get("payload"):
            records[os.path.normcase(os.path.abspath(record["payload"]))] = record
    return records


def _text(content):
    """Message content as text: strings, multimodal parts (text / image), dicts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("image_url", "image", "input_image") or "image_url" in part:
                    parts.append("[image]")
                else:
                    parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or json.dumps(content, ensure_ascii=False))
    return str(content)


def _preview(text):
    text = " ".join(str(text or "").split())
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS - 1] + "…"


def _first(*values):
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _anti_duplicate_text(block):
    if not isinstance(block, dict):
        return None
    if not block.get("enabled"):
        return f"off ({block.get('section', '?')})"
    sent = block.get("sent") or block.get("settings") or {}
    values = ", ".join(f"{k}={v}" for k, v in sent.items()) or "on"
    return f"{block.get('section', '?')}: {values}"


def _tokens_text(usage):
    if not isinstance(usage, dict):
        return None
    given = _first(usage.get("prompt_tokens"), usage.get("input_tokens"), usage.get("promptTokenCount"))
    taken = _first(usage.get("completion_tokens"), usage.get("output_tokens"), usage.get("candidatesTokenCount"))
    if given is None and taken is None:
        return None
    return f"{given if given is not None else 'N/A'} / {taken if taken is not None else 'N/A'}"


def _time_text(value):
    """'2026-10-23 19:55:03' from ISO strings, '20261023_195503' (failed requests) or a datetime."""
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d_%H%M%S_%f"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return text


def _failed_context(name):
    """'failed_glossary_refinement_anthropic_20261023_193505.json' -> 'glossary_refinement'."""
    stem = name[len("failed_"):-len(".json")] if name.startswith("failed_") and name.endswith(".json") else ""
    pieces = stem.split("_")
    # drop the trailing date, time and provider
    context = "_".join(pieces[:-3]) if len(pieces) > 3 else ""
    return None if context in ("", "None") else context


def _chapter_text(info):
    """'Ch 5 · chunk 1/2' from the payload's chapter_info."""
    if not isinstance(info, dict):
        return None if info in (None, "") else str(info)
    chapter = _first(info.get("chapter"), info.get("chapter_num"), info.get("number"))
    chunk, total = info.get("chunk"), info.get("total_chunks")
    pieces = []
    if chapter is not None:
        pieces.append(f"Ch {chapter}")
    if chunk not in (None, ""):
        pieces.append(f"chunk {chunk}/{total}" if total not in (None, "") else f"chunk {chunk}")
    return " · ".join(pieces) or None


def read_row(path, payloads_dir, logistics):
    """The summary row for one payload / failed-request file (None when it cannot be read)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    rel = os.path.relpath(path, payloads_dir)
    parts = rel.split(os.sep)
    failed = parts[0] == "failed_requests"
    debug = data.get("debug") if isinstance(data.get("debug"), dict) else {}
    params = data.get("request_params") if isinstance(data.get("request_params"), dict) else {}
    record = logistics.get(os.path.normcase(os.path.abspath(path)), {})
    messages = data.get("messages") if isinstance(data.get("messages"), list) else []
    texts = {"system": [], "user": [], "assistant": []}
    for message in messages:
        if isinstance(message, dict) and message.get("role") in texts:
            texts[message["role"]].append(_text(message.get("content")))
    retry_info = data.get("retry_info") if isinstance(data.get("retry_info"), dict) else {}
    chapter = _chapter_text(debug.get("chapter_info"))
    row = {
        "time": _time_text(_first(data.get("timestamp"), debug.get("timestamp"), record.get("time"),
                                  datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"))),
        "kind": "Failed" if failed else "Request",
        "context": (_first(data.get("context") if data.get("context") not in ("None",) else None,
                           record.get("context"), _failed_context(os.path.basename(path)))
                    if failed else _first(data.get("context"), record.get("context"),
                                          parts[0] if len(parts) > 1 else None)),
        "chapter": chapter,
        "model": _first(data.get("model"), record.get("model")),
        "provider": _first(data.get("client_type"), record.get("provider")),
        "key": _first(data.get("key_identifier"), debug.get("key_identifier"), record.get("key")),
        "api_key": record.get("api_key"),
        "temperature": _first(params.get("temperature"), (record.get("params") or {}).get("temperature")),
        "max_tokens": _first(params.get("effective_max_tokens"), params.get("max_tokens"),
                             params.get("max_completion_tokens"), (record.get("params") or {}).get("max_tokens")),
        "anti_duplicate": _anti_duplicate_text(_first(data.get("anti_duplicate"), record.get("anti_duplicate"))),
        "retry": _first(retry_info.get("reason"), debug.get("retry_reason"), record.get("retry")),
        "tokens": _tokens_text(data.get("usage")),
        "output_dir": _first(data.get("output_dir"), record.get("output_dir")),
        "system": _preview("\n".join(texts["system"])) or None,
        "user": _preview("\n".join(texts["user"])) or None,
        "assistant": _preview("\n".join(texts["assistant"])) or None,
        "error": _preview(data.get("error")) if failed and data.get("error") else None,
        "file": rel,
    }
    if row["key"] and row["api_key"]:
        row["key"] = f"{row['key']} · {row['api_key']}"
    row["output_dir_uri"] = _folder_uri(row["output_dir"])
    row["raw_uri"] = _folder_uri(os.path.abspath(path))
    row["folder_uri"] = _folder_uri(os.path.dirname(os.path.abspath(path)))
    return row, data


def _folder_uri(folder):
    """file:/// link for an absolute path (output folder, raw JSON, payload subfolder)."""
    try:
        return Path(folder).as_uri() if folder and os.path.isabs(str(folder)) else None
    except (TypeError, ValueError):
        return None


# ---- pages -----------------------------------------------------------------------------------------

STYLE = """
:root { color-scheme: dark; }
body { font-family: Segoe UI, Arial, sans-serif; background: #1e1e1e; color: #ddd; margin: 20px; }
a { color: #7bb3e0; } h1 { font-size: 20px; margin: 0 0 4px; } .dim { color: #888; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border-bottom: 1px solid #333; padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: #2a2a2a; position: sticky; top: 0; cursor: pointer; white-space: nowrap; }
th.dragover { box-shadow: inset 3px 0 0 #7bb3e0; } th .arrow { color: #7bb3e0; }
button { background: #2d2d2d; color: #ddd; border: 1px solid #4a5568; border-radius: 4px; padding: 5px 10px;
         cursor: pointer; } button:hover { border-color: #7bb3e0; }
.panel { background: #252525; border: 1px solid #3a3a3a; border-radius: 6px; padding: 10px 12px; margin-bottom: 12px;
         max-width: 420px; }
.panel ul { list-style: none; padding: 0; margin: 8px 0; } .panel li { display: flex; align-items: center; gap: 6px;
         padding: 2px 0; } .panel li label { flex: 1; } .panel li button { padding: 1px 7px; }
.resizer { position: absolute; top: 0; right: -3px; width: 7px; height: 100%; cursor: col-resize; z-index: 2; }
.resizer:hover, .resizing .resizer { background: #7bb3e0; opacity: .5; }
td.sized { overflow: hidden; text-overflow: ellipsis; }
.menu { position: absolute; z-index: 10; background: #2b2b2b; border: 1px solid #5a9fd4; border-radius: 4px;
        padding: 4px; min-width: 200px; box-shadow: 0 4px 14px rgba(0,0,0,.5); }
.menu div { padding: 6px 12px; cursor: pointer; border-radius: 3px; } .menu div:hover { background: #17a2b8; }
.menu hr { border: none; border-top: 1px solid #444; margin: 4px 0; }
.pager { display: flex; gap: 8px; align-items: center; margin: 10px 0; }
tr.row:hover { background: #2c3440; cursor: default; user-select: text; }
.na { color: #666; } .failed td:nth-child(2) { color: #ff8a8a; }
.bar { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; align-items: center; }
input, select { background: #2d2d2d; color: #ddd; border: 1px solid #4a5568; border-radius: 4px; padding: 5px 8px; }
.preview { min-width: 220px; max-width: 360px; color: #bbb; }
.clamp { display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
td.time { white-space: nowrap; }
pre { white-space: pre-wrap; word-break: break-word; background: #252525; border: 1px solid #333;
      border-radius: 6px; padding: 10px; font-size: 13px; }
.msg { margin: 12px 0; } .role { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px;
       background: #34495e; } .role.system { background: #5a3d12; } .role.user { background: #1f4d2b; }
.role.assistant { background: #3d2a5a; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; } dt { color: #999; }
"""


def _detail_page(row, data, payload_path):
    esc = html.escape

    def value(v):
        return esc(str(v)) if v not in (None, "") else "<span class='na'>N/A</span>"

    def cell(key):
        if key == "output_dir" and row.get("output_dir_uri"):
            return f"<a href=\"{esc(row['output_dir_uri'])}\">{esc(str(row['output_dir']))}</a>"
        return value(row.get(key))

    fields = "".join(f"<dt>{esc(label)}</dt><dd>{cell(key)}</dd>"
                     for key, label in COLUMNS if key not in ("system", "user", "assistant"))
    messages = []
    for index, message in enumerate(data.get("messages") or [], 1):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "?")
        text = _text(message.get("content"))
        cut = len(text) > DETAIL_MAX_CHARS
        shown = text[:DETAIL_MAX_CHARS]
        messages.append(
            f"<div class='msg'><span class='role {esc(role)}'>{esc(role)}</span> "
            f"<span class='dim'>#{index} · {len(text):,} characters</span>"
            f"<pre>{esc(shown)}{'…' if cut else ''}</pre>"
            + ("<div class='dim'>Cut here; the raw JSON has the full text.</div>" if cut else "") + "</div>")

    def block(title, obj):
        if obj in (None, "", {}, []):
            return ""
        body = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        return f"<h3>{esc(title)}</h3><pre>{esc(body)}</pre>"

    raw_uri = Path(os.path.abspath(payload_path)).as_uri()
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(row.get('file', 'Request'))}</title>
<style>{STYLE}</style></head><body>
<p><a href="../{INDEX_FILE}">← All requests</a> · <a href="{esc(raw_uri)}">Raw JSON</a></p>
<h1>{esc(str(row.get('kind')))} · {esc(str(row.get('context') or 'N/A'))} · {esc(str(row.get('model') or 'N/A'))}</h1>
<p class="dim">{esc(row.get('file', ''))}</p>
<dl>{fields}</dl>
<h2>Messages</h2>{''.join(messages) or "<p class='na'>N/A</p>"}
{block('Request parameters', data.get('request_params'))}
{block('Anti-duplicate', data.get('anti_duplicate'))}
{block('Token usage', data.get('usage'))}
{block('Retry', data.get('retry_info'))}
{block('Error', data.get('error'))}
{block('Traceback', data.get('traceback'))}
{block('Response', data.get('response'))}
{block('Debug', data.get('debug'))}
</body></html>
"""


def _index_page(rows, payloads_dir, generated):
    shown = [(key, label) for key, label in COLUMNS if any(r.get(key) not in (None, "") for r in rows)]
    data = json.dumps(rows, ensure_ascii=False, default=str).replace("</", "<\\/")
    columns = json.dumps(shown, ensure_ascii=False)
    esc = html.escape
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Payload summary</title>
<style>{STYLE}</style></head><body>
<h1>Payload summary</h1>
<p class="dim">{len(rows):,} requests from {esc(os.path.abspath(payloads_dir))} · generated {esc(generated)}.
Double-click a row for the full request; right-click it for its raw JSON and folders.</p>
<div class="bar">
  <input id="q" type="search" placeholder="Search everything…" size="34">
  <select id="kind"><option value="">All types</option></select>
  <select id="context"><option value="">All contexts</option></select>
  <select id="model"><option value="">All models</option></select>
  <button id="colbtn" type="button">Columns ▾</button>
  <span id="count" class="dim"></span>
</div>
<div id="colpanel" class="panel" hidden>
  <div class="dim">Show, hide and order the columns (or drag a column header). Saved in this browser.</div>
  <ul id="collist"></ul>
  <button id="colreset" type="button">Reset</button>
</div>
<div class="pager" id="pagerTop"></div>
<table><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table>
<div class="pager" id="pagerBottom"></div>
<div id="menu" class="menu" hidden></div>
<script>
const ROWS = {data};
const COLS = {columns};
const PREVIEW = new Set(["system", "user", "assistant", "error"]);
let sortKey = "time", sortDir = -1;
// Column order and hidden columns: kept in localStorage, so they survive closing the page and regenerating
const PREFS_KEY = "glossarion.payloadSummary.columns";
const LABELS = Object.fromEntries(COLS);
function loadPrefs() {{
  try {{ return JSON.parse(localStorage.getItem(PREFS_KEY)) || {{}}; }} catch (e) {{ return {{}}; }}
}}
function savePrefs() {{
  try {{ localStorage.setItem(PREFS_KEY, JSON.stringify({{order, hidden: [...hidden], widths, pageSize}})); }}
  catch (e) {{}}
}}
let prefs = loadPrefs();
// saved order first (columns this summary has), then any new column in its default place
let order = (prefs.order || []).filter(k => k in LABELS);
COLS.forEach(([k]) => {{ if (!order.includes(k)) order.push(k); }});
let hidden = new Set((prefs.hidden || []).filter(k => k in LABELS));
let widths = Object.fromEntries(Object.entries(prefs.widths || {{}}).filter(([k]) => k in LABELS));
let pageSize = Number.isInteger(prefs.pageSize) ? prefs.pageSize : 250;  // 0 = all rows
let page = 1;
const sizeStyle = (el, key) => {{ const w = widths[key]; if (!w) return;
  el.style.width = el.style.minWidth = el.style.maxWidth = w + "px"; }};
const visible = () => order.filter(k => !hidden.has(k));
const head = document.getElementById("head");
let dragKey = null, resized = false;
// right-click a header: hide it, show all, choose columns, reset widths
const menu = document.getElementById("menu");
function openMenu(e, build) {{
  menu.innerHTML = "";
  const item = (text, action, disabled) => {{ const d = document.createElement("div"); d.textContent = text;
    if (disabled) d.style.opacity = .4; else d.onclick = () => {{ menu.hidden = true; action(); }};
    menu.appendChild(d); }};
  build(item);
  menu.hidden = false;
  menu.style.left = Math.min(e.pageX, window.scrollX + window.innerWidth - menu.offsetWidth - 8) + "px";
  menu.style.top = Math.min(e.pageY, window.scrollY + window.innerHeight - menu.offsetHeight - 8) + "px";
}}
// right-click a request: its raw JSON, its payload subfolder, its output folder (when it has one)
function showRowMenu(e, r) {{
  openMenu(e, item => {{
    item("View raw JSON", () => window.open(r.raw_uri, "_blank"), !r.raw_uri);
    item("Open payloads subfolder", () => window.open(r.folder_uri, "_blank"), !r.folder_uri);
    if (r.output_dir_uri) item("Open output folder", () => window.open(r.output_dir_uri, "_blank"));
  }});
}}
function showMenu(e, key) {{ openMenu(e, item => {{
  item("Hide \u201c" + LABELS[key] + "\u201d", () => {{ hidden.add(key); savePrefs(); renderPanel(); render(); }},
       visible().length <= 1);
  item("Show all columns", () => {{ hidden = new Set(); savePrefs(); renderPanel(); render(); }}, !hidden.size);
  menu.appendChild(document.createElement("hr"));
  item("Choose columns\u2026", () => {{ document.getElementById("colpanel").hidden = false; renderPanel();
    document.getElementById("colpanel").scrollIntoView({{block: "nearest"}}); }});
  item("Reset column widths", () => {{ widths = {{}}; savePrefs(); render(); }}, !Object.keys(widths).length);
}}); }}
document.addEventListener("click", e => {{ if (!menu.contains(e.target)) menu.hidden = true; }});
document.addEventListener("keydown", e => {{ if (e.key === "Escape") menu.hidden = true; }});
window.addEventListener("scroll", () => {{ menu.hidden = true; }}, true);
function renderHead() {{
  head.innerHTML = "";
  visible().forEach(key => {{
    const th = document.createElement("th"); th.textContent = LABELS[key]; th.draggable = true;
    sizeStyle(th, key);
    if (key === sortKey) {{ const a = document.createElement("span"); a.className = "arrow";
      a.textContent = sortDir > 0 ? " ▲" : " ▼"; th.appendChild(a); }}
    th.onclick = () => {{ if (resized) {{ resized = false; return; }}
      sortDir = sortKey === key ? -sortDir : 1; sortKey = key; page = 1; render(); }};
    th.oncontextmenu = e => {{ e.preventDefault(); showMenu(e, key); }};
    // drag the right edge to resize; the width is remembered
    const grip = document.createElement("span"); grip.className = "resizer";
    grip.onmousedown = e => {{
      e.preventDefault(); e.stopPropagation(); th.draggable = false; th.classList.add("resizing");
      const startX = e.pageX, startW = th.getBoundingClientRect().width;
      const move = ev => {{ widths[key] = Math.max(50, Math.round(startW + ev.pageX - startX));
        sizeStyle(th, key); document.querySelectorAll("td[data-key='" + key + "']").forEach(td => sizeStyle(td, key)); }};
      const up = () => {{ document.removeEventListener("mousemove", move); document.removeEventListener("mouseup", up);
        th.classList.remove("resizing"); th.draggable = true; resized = true; savePrefs(); }};
      document.addEventListener("mousemove", move); document.addEventListener("mouseup", up);
    }};
    th.appendChild(grip);
    th.ondragstart = e => {{ dragKey = key; e.dataTransfer.effectAllowed = "move"; }};
    th.ondragover = e => {{ e.preventDefault(); th.classList.add("dragover"); }};
    th.ondragleave = () => th.classList.remove("dragover");
    th.ondrop = e => {{ e.preventDefault(); th.classList.remove("dragover"); moveBefore(dragKey, key); }};
    head.appendChild(th);
  }});
}}
function moveBefore(moving, target) {{
  if (!moving || moving === target) return;
  order.splice(order.indexOf(moving), 1);
  order.splice(order.indexOf(target), 0, moving);
  savePrefs(); renderPanel(); render();
}}
function renderPanel() {{
  const list = document.getElementById("collist"); list.innerHTML = "";
  order.forEach((key, i) => {{
    const li = document.createElement("li");
    const label = document.createElement("label"); const box = document.createElement("input");
    box.type = "checkbox"; box.checked = !hidden.has(key);
    box.onchange = () => {{ box.checked ? hidden.delete(key) : hidden.add(key);
      if (!visible().length) {{ hidden.delete(key); box.checked = true; }} savePrefs(); render(); }};
    label.append(box, " " + LABELS[key]);
    const up = document.createElement("button"); up.textContent = "↑"; up.disabled = i === 0;
    up.onclick = () => {{ [order[i - 1], order[i]] = [order[i], order[i - 1]]; savePrefs(); renderPanel(); render(); }};
    const down = document.createElement("button"); down.textContent = "↓"; down.disabled = i === order.length - 1;
    down.onclick = () => {{ [order[i + 1], order[i]] = [order[i], order[i + 1]]; savePrefs(); renderPanel(); render(); }};
    li.append(label, up, down); list.appendChild(li);
  }});
}}
document.getElementById("colbtn").onclick = () => {{
  const panel = document.getElementById("colpanel"); panel.hidden = !panel.hidden; renderPanel();
}};
document.getElementById("colreset").onclick = () => {{
  order = COLS.map(([k]) => k); hidden = new Set(); widths = {{}}; savePrefs(); renderPanel(); render();
}};
for (const id of ["kind", "context", "model"]) {{
  const sel = document.getElementById(id);
  [...new Set(ROWS.map(r => r[id]).filter(v => v))].sort().forEach(v => {{
    const o = document.createElement("option"); o.value = o.textContent = v; sel.appendChild(o);
  }});
  sel.onchange = () => {{ page = 1; render(); }};
}}
document.getElementById("q").oninput = () => {{ page = 1; render(); }};
function renderPager(id, total, pages) {{
  const bar = document.getElementById(id); bar.innerHTML = "";
  const btn = (text, to, disabled) => {{ const b = document.createElement("button"); b.textContent = text;
    b.disabled = disabled; b.onclick = () => {{ page = to; render(); window.scrollTo(0, 0); }}; bar.appendChild(b); }};
  btn("\u00ab First", 1, page <= 1); btn("\u2039 Prev", page - 1, page <= 1);
  const info = document.createElement("span"); info.className = "dim";
  info.textContent = "Page " + page + " of " + pages; bar.appendChild(info);
  btn("Next \u203a", page + 1, page >= pages); btn("Last \u00bb", pages, page >= pages);
  const label = document.createElement("span"); label.className = "dim"; label.textContent = "Rows per page:";
  const sel = document.createElement("select");
  [[100, "100"], [250, "250"], [500, "500"], [1000, "1000"], [0, "All"]].forEach(([v, t]) => {{
    const o = document.createElement("option"); o.value = v; o.textContent = t; o.selected = v === pageSize; sel.appendChild(o); }});
  sel.onchange = () => {{ pageSize = parseInt(sel.value, 10); page = 1; savePrefs(); render(); }};
  bar.append(label, sel);
}}
function render() {{
  renderHead();
  const cols = visible();
  const q = document.getElementById("q").value.toLowerCase();
  const pick = id => document.getElementById(id).value;
  const rows = ROWS.filter(r => (!pick("kind") || r.kind === pick("kind")) &&
    (!pick("context") || r.context === pick("context")) && (!pick("model") || r.model === pick("model")) &&
    (!q || JSON.stringify(r).toLowerCase().includes(q)));
  rows.sort((a, b) => {{
    const x = a[sortKey] ?? "", y = b[sortKey] ?? "";
    return (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y), undefined, {{numeric: true}})) * sortDir;
  }});
  const pages = pageSize ? Math.max(1, Math.ceil(rows.length / pageSize)) : 1;
  page = Math.min(Math.max(1, page), pages);
  const shown = pageSize ? rows.slice((page - 1) * pageSize, page * pageSize) : rows;
  const body = document.getElementById("body"); body.innerHTML = "";
  const frag = document.createDocumentFragment();
  shown.forEach(r => {{
    const tr = document.createElement("tr"); tr.className = "row" + (r.kind === "Failed" ? " failed" : "");
    tr.ondblclick = () => window.open(r.detail, "_blank");  // double-click opens the request details
    tr.oncontextmenu = e => {{ e.preventDefault(); showRowMenu(e, r); }};
    cols.forEach(key => {{
      const td = document.createElement("td"); const v = r[key]; td.dataset.key = key;
      if (widths[key]) {{ sizeStyle(td, key); td.classList.add("sized"); }}
      if (v === null || v === undefined || v === "") {{ td.textContent = "N/A"; td.classList.add("na"); }}
      else if (PREVIEW.has(key)) {{ td.classList.add("preview"); const d = document.createElement("div");
        d.className = "clamp"; d.textContent = v; td.appendChild(d); }}
      else if (key === "output_dir" && r.output_dir_uri) {{ const a = document.createElement("a");
        a.href = r.output_dir_uri; a.textContent = v; a.title = "Open " + v; a.target = "_blank";
        a.onclick = e => e.stopPropagation(); td.appendChild(a); }}
      else {{ td.textContent = v; td.title = v; if (key === "time") td.classList.add("time"); }}
      tr.appendChild(td);
    }});
    frag.appendChild(tr);
  }});
  body.appendChild(frag);
  renderPager("pagerTop", rows.length, pages); renderPager("pagerBottom", rows.length, pages);
  const first = rows.length ? (pageSize ? (page - 1) * pageSize + 1 : 1) : 0;
  document.getElementById("count").textContent = (rows.length === ROWS.length ? "" : rows.length.toLocaleString() +
    " match · ") + "showing " + first.toLocaleString() + "\u2013" + (first + shown.length - (rows.length ? 1 : 0)).toLocaleString() +
    " of " + rows.length.toLocaleString();
}}
render();
</script></body></html>
"""


def build(payloads_dir, *, out_dir=None, workers=1, progress=None, cancel=None):
    """Write the summary; returns the index path (None when cancelled). ``progress(done, total)``
    is called as files are read; ``cancel()`` stops between files."""
    out_dir = out_dir or summary_dir(payloads_dir)
    details = os.path.join(out_dir, "details")
    if os.path.isdir(details):
        shutil.rmtree(details, ignore_errors=True)  # the previous summary's pages
    os.makedirs(details, exist_ok=True)
    files = collect_files(payloads_dir)
    logistics = _load_logistics(payloads_dir)
    total = len(files)
    done = [0]
    lock = threading.Lock()

    def one(numbered):
        number, path = numbered
        if cancel is not None and cancel():
            return None
        row, data = read_row(path, payloads_dir, logistics)
        if row is not None:
            row["detail"] = f"details/{number:05d}.html"
            with open(os.path.join(details, f"{number:05d}.html"), "w", encoding="utf-8") as f:
                f.write(_detail_page(row, data, path))
        with lock:
            done[0] += 1
            count = done[0]
        if progress is not None:
            progress(count, total)
        return row

    workers = max(1, int(workers or 1))
    if workers > 1 and total > 1:
        with ThreadPoolExecutor(max_workers=min(workers, total), thread_name_prefix="payload-summary") as pool:
            rows = list(pool.map(one, enumerate(files, 1)))
    else:
        rows = [one(item) for item in enumerate(files, 1)]
    if cancel is not None and cancel():
        return None
    rows = [r for r in rows if r is not None]
    index = os.path.join(out_dir, INDEX_FILE)
    with open(index, "w", encoding="utf-8") as f:
        f.write(_index_page(rows, payloads_dir, datetime.now().isoformat(timespec="seconds")))
    return index
