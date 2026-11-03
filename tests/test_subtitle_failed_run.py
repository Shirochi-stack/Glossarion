"""A subtitle run that created no file, or not all of them, reports a failed run instead of success.

Real runs through ``_translation_worker`` (a child process each: main() rewires print, the environment and module
state) against a local fake OpenAI-compatible server. Found on 2026-11-02 when a local model returned no text.
"""
import http.server
import json
import os
import re
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {
    "one.vtt": "WEBVTT\n\n00:01.000 --> 00:03.000\n안녕 친구 ONE\n",
    "two.ssa": ("[Script Info]\nScriptType: v4.00\n\n[Events]\nFormat: Marked, Start, End, Style, Name, MarginL, "
                "MarginR, MarginV, Effect, Text\nDialogue: Marked=0,0:00:01.00,0:00:03.00,Default,,0,0,0,,안녕 친구 TWO\n"),
    "three.sbv": "0:00:01.000,0:00:03.000\n안녕 친구 THREE\n",
}


def _serve(reply_empty):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            content = body["messages"][-1]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content)
            found = re.search(r"\[\s*\{.*\}\s*\]", content, re.S)
            rows = json.loads(found.group(0)) if found else []
            if reply_empty(content):
                text, finish = "", "length"
            else:
                text = json.dumps([{"id": row["id"], "target": row["source"].replace("안녕 친구", "Hello")}
                                   for row in rows], ensure_ascii=False)
                finish = "stop"
            raw = json.dumps({"id": "x", "object": "chat.completion", "model": "m", "created": 1,
                              "choices": [{"index": 0, "finish_reason": finish,
                                           "message": {"role": "assistant", "content": text}}],
                              "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}/v1"


def _drive(work, shape, empty_for):
    """Child process: one run, written to results.json."""
    sys.path.insert(0, str(ROOT / "src"))
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    if shape == "zip":
        source = work / "Show.zip"
        with zipfile.ZipFile(source, "w") as archive:
            for name, text in SAMPLES.items():
                archive.writestr(name, text)
    else:
        source = work / "one.vtt"
        source.write_text(SAMPLES["one.vtt"], encoding="utf-8")
    url = _serve(lambda content: empty_for == "all" or (empty_for and empty_for in content))
    os.environ.update({"GLOSSARION_HTTP_LOG": "0", "OUTPUT_DIRECTORY": str(work / "outputs"), "OPENAI_API_KEY": "sk-test",
                       "OPENAI_CUSTOM_BASE_URL": url, "USE_CUSTOM_OPENAI_ENDPOINT": "1"})
    from headless_owner import HeadlessOwner

    logs = []

    class Host:
        def log(self, message): logs.append(str(message))
        def emit(self, *args, **kwargs): pass
        def ask(self, *args, **kwargs): return None
        def stop_requested(self): return False

    config = {"batch_translation": True, "batch_size": 3, "delay": 0.0, "max_output_tokens": 65536,
              "output_directory": str(work / "outputs"), "use_custom_openai_endpoint": True, "openai_base_url": url,
              "auto_glossary_mode": "off", "enable_auto_glossary": False, "enable_post_translation_scan": False,
              "contextual": False, "retry_truncated": False}
    owner = HeadlessOwner(config, host=Host(), api_key="sk-test", model="gpt-4o-mini")
    owner.selected_files = [str(source)]
    request = owner._prepare_translation_run([str(source)])
    owner.batch_translation_var = True
    owner.batch_size_var = 3
    result = owner._translation_worker(request)
    outputs = sorted(path.name for path in (work / "outputs").rglob("*") if path.is_file()
                     and path.suffix in {".vtt", ".ssa", ".sbv"} and ".glossarion" not in str(path))
    (work / "results.json").write_text(json.dumps({"result": result, "outputs": outputs, "logs": logs}), encoding="utf-8")
    os._exit(0)


def _run(tmp_path, shape, empty_for):
    env = dict(os.environ)
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "GLOSSARION_LIBRARY_DIR", "GLOSSARION_DATA_DIR"):
        env[name] = str(tmp_path)
    env["PYTHONIOENCODING"] = "utf-8"
    work = tmp_path / "work"
    done = subprocess.run([sys.executable, __file__, str(work), shape, empty_for], env=env, capture_output=True,
                          encoding="utf-8", errors="replace", timeout=300)
    assert (work / "results.json").exists(), (done.stdout + done.stderr)[-3000:]
    return json.loads((work / "results.json").read_text(encoding="utf-8"))


def test_a_zip_that_translated_reports_success(tmp_path):
    run = _run(tmp_path, "zip", "")
    assert run["result"] is True
    assert run["outputs"] == ["one.vtt", "three.sbv", "two.ssa"]


@pytest.mark.parametrize("shape", ["zip", "file"])
def test_a_run_with_no_output_file_fails(tmp_path, shape):
    run = _run(tmp_path, shape, "all")
    assert run["result"] is False
    assert run["outputs"] == []
    assert not any("Translation completed successfully" in line for line in run["logs"])


def test_a_zip_missing_one_file_fails_and_keeps_the_others(tmp_path):
    run = _run(tmp_path, "zip", "TWO")
    assert run["result"] is False
    assert run["outputs"] == ["one.vtt", "three.sbv"]


if __name__ == "__main__":
    _drive(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
