"""The ``search/gemini`` route (src/gemini_free.py): google.com with AI Mode on.

* Request building: what a chat request puts on the AI Mode page (prompt box text, attached
  files and images, the model menu entry), without the old browser chunking.
* Answer extraction from canned page snapshots (``AI_MODE_SNAPSHOT_SCRIPT`` results): the HTML
  code block, plain and Markdown answers, and the pages that are not an answer (consent,
  sign-in, verification, "not available", web results only, a code block that ended early).
* The page flow on a scripted fake page, the model list, the Google sign-in switch and the
  error kind that crosses the helper process.
* The chunking settings are gone from every table and old config keys are ignored.

No Qt, no browser and no network. Python 3.10 compatible.
Run:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_gemini_free_ai_mode.py
"""
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import browser_driver  # noqa: E402
import gemini_free as gf  # noqa: E402

MODE_ENV = ("GEMINI_FREE_MODE", "GEMINI_FREE_SEARCH_PARAMS", "GEMINI_FREE_SEARCH_BASE_URL",
            "GEMINI_FREE_PROFILE_DIR", "TEXT_EXTRACTION_METHOD", "EXTRACTION_MODE", "USE_HTML2TEXT",
            "TRANSLATION_CANCELLED", "GLOSSARION_MOBILE", "GLOSSARION_NO_SUBPROCESS")


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    for name in MODE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("GEMINI_FREE_WAIT_AFTER_LOAD_MS", "0")
    monkeypatch.setenv("GEMINI_FREE_STABLE_MS", "0")
    monkeypatch.setenv("GEMINI_FREE_SETTLE_MS", "0")
    browser_driver.unregister_driver()
    gf._cancel_event.clear()
    yield monkeypatch
    browser_driver.unregister_driver()
    gf._cancel_event.clear()


def png_part(data=b"\x89PNG picture"):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}


def names(request):
    return [(item["name"], item["kind"]) for item in request["files"]]


# ============================================================================ request building
def test_page_address_is_google_com_with_ai_mode_on(clean):
    assert gf._build_search_base_url() == "https://www.google.com/search?udm=50&hl=en"
    assert gf._build_search_url("안녕 hi") == "https://www.google.com/search?udm=50&hl=en&q=%EC%95%88%EB%85%95+hi"
    clean.setenv("GEMINI_FREE_SEARCH_PARAMS", "gl=us&q=ignored")
    assert gf._build_search_base_url() == "https://www.google.com/search?udm=50&hl=en&gl=us"


def test_short_plain_request_is_typed_in_as_it_is():
    request = gf._build_ai_mode_request([{"role": "system", "content": "Answer briefly."},
                                         {"role": "user", "content": "What is 2+2?"}])
    assert request["prompt"] == "Answer briefly.\n\nWhat is 2+2?"
    assert request["files"] == [] and request["model"] == ""
    assert not request["expects_html"] and not request["instructions_attached"]
    assert gf._build_ai_mode_request([{"role": "user", "content": "hi"}])["prompt"] == "hi"


def test_html_chapter_is_attached_and_never_typed_in():
    chapter = "<h1>제1장</h1>\n<p>안녕하세요.</p>"
    request = gf._build_ai_mode_request([{"role": "system", "content": "Translate to English."},
                                         {"role": "user", "content": chapter}])
    assert names(request) == [("chapter.html", "file")]
    attached = request["files"][0]
    assert attached["data"].decode("utf-8") == chapter and attached["mime"] == "text/plain"
    assert request["expects_html"] is True
    assert request["prompt"] == (
        "Translate to English.\n\n"
        "Apply the instructions above to the attached file chapter.html. Reply with only the complete "
        "result as HTML inside one ```html code block, with no follow-up questions or commentary.")
    # no role labels: with an attachment AI Mode answers those with "no response available"
    for label in ("System instructions", "User:", "Assistant:"):
        assert label not in request["prompt"]


def test_chapter_of_one_long_line_gets_one_block_per_line():
    paragraph = "<p>가나다라마바사 아자차카타파하.<br/></p>"
    chapter = "<h1>1화</h1>" + paragraph * 120
    assert "\n" not in chapter and len(chapter) > gf.MAX_HTML_LINE_CHARS
    request = gf._build_ai_mode_request([{"role": "system", "content": "Translate."},
                                         {"role": "user", "content": chapter}])
    sent = request["files"][0]["data"].decode("utf-8")
    assert sent.replace("\n", "") == chapter                                   # only line breaks were added
    assert max(len(line) for line in sent.split("\n")) < 100
    assert sent.count("\n") == 120                                             # after every block but the last
    short = "<p>a</p><p>b</p>"
    assert gf._one_block_per_line(short) == short                             # short lines are left alone


def test_long_plain_text_is_attached_as_text():
    text = "문장입니다. " * 400
    request = gf._build_ai_mode_request([{"role": "system", "content": "Extract the names."},
                                         {"role": "user", "content": text}])
    assert names(request) == [("text.txt", "file")] and not request["expects_html"]
    assert request["prompt"].endswith(
        "Apply the instructions above to the attached file text.txt. Reply with only the complete result, "
        "with no follow-up questions or commentary.")
    alone = gf._build_ai_mode_request([{"role": "user", "content": text}])
    assert alone["prompt"].startswith("Read the attached file text.txt and do what it asks.")


def test_images_are_attached_as_images():
    picture = b"\x89PNG one"
    request = gf._build_ai_mode_request([
        {"role": "system", "content": "Translate the text in the image."},
        {"role": "user", "content": [{"type": "text", "text": "Page 3"}, png_part(picture),
                                     {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                                  "data": base64.b64encode(b"jpeg").decode()}},
                                     {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}])
    assert names(request) == [("image001.png", "image"), ("image002.jpg", "image")]
    assert request["files"][0]["data"] == picture and request["files"][0]["mime"] == "image/png"
    assert request["files"][1]["data"] == b"jpeg" and request["files"][1]["mime"] == "image/jpeg"
    assert request["prompt"] == ("Translate the text in the image.\n\nPage 3\n\n"
                                 "The 2 attached images belong to the request.")
    assert "[image omitted]" not in request["prompt"]


def test_earlier_messages_go_into_a_context_file():
    request = gf._build_ai_mode_request([
        {"role": "system", "content": "Translate."},
        {"role": "user", "content": "<p>하나</p>"}, {"role": "assistant", "content": "<p>One</p>"},
        {"role": "user", "content": "<p>둘</p>"}])
    assert names(request) == [("context.txt", "file"), ("chapter.html", "file")]
    assert request["files"][0]["data"].decode("utf-8") == (
        "Earlier request:\n<p>하나</p>\n\nEarlier reply:\n<p>One</p>")
    assert request["files"][1]["data"].decode("utf-8") == "<p>둘</p>"
    assert "context.txt holds the earlier messages" in request["prompt"]


def test_long_system_prompt_moves_into_an_instructions_file():
    glossary = "\n".join(f"김현우{index} = Kim Hyun-woo {index}" for index in range(900))
    system = "Translate with this glossary:\n" + glossary
    assert len(gf._build_search_url(system)) > gf.MAX_PROMPT_URL_CHARS
    request = gf._build_ai_mode_request([{"role": "system", "content": system},
                                         {"role": "user", "content": "<p>안녕</p>"}])
    assert names(request) == [("instructions.txt", "file"), ("chapter.html", "file")]
    assert request["files"][0]["data"].decode("utf-8") == system and request["instructions_attached"]
    assert request["prompt"].startswith("Follow the instructions in the attached file instructions.txt.")
    assert "Apply them to the attached file chapter.html." in request["prompt"]
    assert len(gf._build_search_url(request["prompt"])) < 2000


@pytest.mark.parametrize("model, choice", [
    ("search/gemini", ""), ("gemini", ""), ("search2/gemini", ""),
    ("search/gemini-4-carbon", "4 carbon"), ("search/gemini-3.8-flash", "3.8 flash"),
    ("gemini-carbon", "carbon"), ("search/Gemini-4-Hydrogen", "4 hydrogen"),
])
def test_model_names(model, choice):
    assert gf._ai_mode_model_choice(model) == choice
    assert gf._build_ai_mode_request([{"role": "user", "content": "hi"}], model)["model"] == choice


def test_unknown_model_is_refused_before_any_browser(clean):
    clean.setattr(gf.subprocess, "Popen", lambda *a, **k: pytest.fail("a browser was started"))
    with pytest.raises(RuntimeError, match="search/gemini-<AI Mode model>; got search/llama"):
        gf.send_chat_completion(messages=[{"role": "user", "content": "hi"}], model="search/llama")


def test_model_menu_labels_become_model_names():
    assert gf._model_ids(["Auto", "3.8 Flash", "4 Carbon"]) == [
        "search/gemini", "search/gemini-3.8-flash", "search/gemini-4-carbon"]
    assert gf._model_ids(["4 Hydrogen • Everyday help", "4 Carbon • Most complex work"]) == [
        "search/gemini", "search/gemini-4-hydrogen", "search/gemini-4-carbon"]
    assert gf._model_ids([]) == ["search/gemini"]
    for name in gf._model_ids(["3.8 Flash", "4 Carbon • Most complex work"])[1:]:
        label = {"3.8 flash": "3.8 Flash", "4 carbon": "4 Carbon"}[gf._ai_mode_model_choice(name)]
        assert gf._ai_mode_model_choice(name) in label.lower()                 # the name finds its menu entry


# ============================================================================ answer extraction
def snapshot(answer="", **extra):
    data = {"url": "https://www.google.com/search?udm=50&hl=en&q=x", "title": "x - Google Search",
            "ready": "complete", "text": "AI Mode\n" + answer, "aiMode": True, "promptBox": True,
            "answerFound": bool(answer), "answerText": answer, "answerMarkdown": answer, "codeBlocks": [],
            "complete": True, "pending": 0, "htmlLength": 5000}
    data.update(extra)
    return data


def test_html_answer_is_the_code_block():
    code = '<h1>Chapter 1</h1>\n<p>He looked up.<br/></p>\n<p><b>Park</b> asked.</p>\n'
    page = snapshot("html\n\n```\n" + code.rstrip() + "\n```\n\nUse code with caution.", codeBlocks=[code])
    assert gf._extract_rendered_content(page, prefer_html=True) == code.strip()


def test_code_block_closed_late_is_cut_at_the_models_fence():
    code = "<p>One</p>\n<p>Two</p>\n```\n\nThe image says RED MOON.\n\n<FollowUp>\nWant more?\n</FollowUp>\n"
    page = snapshot("html\n\n```\n" + code + "```", codeBlocks=[code])
    assert gf._extract_rendered_content(page, prefer_html=True) == "<p>One</p>\n<p>Two</p>"


def test_code_block_that_ended_early_is_refused():
    code = "<p>First half.</p>\n" * 40
    rest = "\n\n".join(f"Paragraph {index} of the second half, shown without its tags." for index in range(40))
    page = snapshot("html\n\n```\n" + code + "```\n\nUse code with caution.\n\n" + rest, codeBlocks=[code])
    with pytest.raises(gf.AIModeError, match="did not keep the HTML answer inside its code block") as caught:
        gf._extract_rendered_content(page, prefer_html=True)
    assert caught.value.kind == gf.KIND_INCOMPLETE and caught.value.kind in gf.FATAL_KINDS
    assert "sign in with Google Login" in str(caught.value)
    with pytest.raises(gf.AIModeError) as signed_in:
        gf._extract_rendered_content(dict(page, signed_in=True), prefer_html=True)
    assert "Google Login" not in str(signed_in.value)
    # an empty code block with the whole text beside it (an answer of one very long line)
    with pytest.raises(gf.AIModeError) as empty:
        gf._extract_rendered_content(snapshot("html\n\n```\n\n```\n\nUse code with caution.\n\n" + rest,
                                              codeBlocks=[""]), prefer_html=True)
    assert empty.value.kind == gf.KIND_INCOMPLETE


def test_tail_outside_a_long_code_block_is_refused():
    # the allowance for text outside the block is fixed: it does not grow with the block
    code = ("<p>" + "x" * 80 + "</p>\n") * 100
    tail = ("word " * 400)[:1519]
    assert len(code) == 8800 and len(tail) == 1519
    page = snapshot("html\n\n```\n" + code + "```\n\nUse code with caution.\n\n" + tail, codeBlocks=[code])
    with pytest.raises(gf.AIModeError, match="1,519 after it without their tags") as caught:
        gf._extract_rendered_content(page, prefer_html=True)
    assert caught.value.kind == gf.KIND_INCOMPLETE
    # one character over the allowance is refused, the allowance itself is not
    limit = gf.MAX_TEXT_OUTSIDE_CODE_BLOCK
    for extra, refused in ((limit, False), (limit + 1, True)):
        page = snapshot("html\n\n```\n" + code + "```\n\n" + "y" * extra, codeBlocks=[code])
        if refused:
            with pytest.raises(gf.AIModeError) as over:
                gf._extract_rendered_content(page, prefer_html=True)
            assert over.value.kind == gf.KIND_INCOMPLETE
        else:
            assert gf._extract_rendered_content(page, prefer_html=True) == code.strip()


def test_lead_in_and_closing_line_around_the_code_block_are_accepted():
    code = ("<p>" + "x" * 80 + "</p>\n") * 100
    page = snapshot("Here is the translated chapter:\n\nhtml\n\n```\n" + code + "```\n\nUse code with caution."
                    "\n\nLet me know if you want any changes.", codeBlocks=[code])
    assert gf._extract_rendered_content(page, prefer_html=True) == code.strip()


def test_plain_answer_loses_ai_modes_own_lines():
    answer = ("Paris is the capital.\n\npython\n\n```\nprint(1)\n```\n\nUse code with caution.\n\n"
              "<FollowUp>\nShall I continue?\n</FollowUp>\n\nAI can make mistakes, so double-check responses")
    assert gf._extract_rendered_content(snapshot(answer)) == "Paris is the capital.\n\n```python\nprint(1)\n```"
    # the fallback container starts with AI Mode's own heading
    assert gf._extract_rendered_content(snapshot("AI Mode reply for hi\n\nHello!")) == "Hello!"


def test_markdown_answer_when_the_request_works_in_markdown(clean):
    page = snapshot("Title\n\nBold text", answerMarkdown="# Title\n\n**Bold** text")
    assert gf._extract_rendered_content(page) == "Title\n\nBold text"
    assert gf._extract_rendered_content(page, prefer_markdown=True) == "# Title\n\n**Bold** text"
    clean.setenv("TEXT_EXTRACTION_METHOD", "enhanced")
    assert gf._build_ai_mode_request([{"role": "user", "content": "x" * 3000}])["expects_markdown"] is True


@pytest.mark.parametrize("page, kind, words", [
    (snapshot(url="https://www.google.com/sorry/index?continue=x", text="Our systems have detected unusual traffic"),
     gf.KIND_VERIFICATION, "verification page"),
    (snapshot(url="https://consent.google.com/m?continue=x", text="Before you continue to Google"),
     gf.KIND_CONSENT, "cookie consent page"),
    (snapshot(url="https://accounts.google.com/v3/signin/identifier", text="Sign in"),
     gf.KIND_SIGN_IN, "asked to sign in"),
    (snapshot(text="AI Mode isn't available in your country or language"),
     gf.KIND_UNAVAILABLE, "not available"),
    (snapshot(notAiMode=True, aiMode=False, title="hi - Google Search"),
     gf.KIND_UNAVAILABLE, "AI Mode did not open"),
    (snapshot("It looks like there's no response available for this search. Try asking something else."),
     gf.KIND_NO_ANSWER, "no response"),
    (snapshot("10 sites\nHere are top web results for exploring this topic:\nGoogle Translate"),
     gf.KIND_NO_ANSWER, "web results only"),
    (snapshot("Something went wrong and the content wasn't generated."),
     gf.KIND_GENERATION_FAILURE, "generation failure"),
])
def test_pages_that_are_not_an_answer(page, kind, words):
    with pytest.raises(gf.AIModeError, match=words) as caught:
        gf._extract_rendered_content(page)
    assert caught.value.kind == kind
    assert (kind in gf.FATAL_KINDS) == (kind not in (gf.KIND_NO_ANSWER, gf.KIND_GENERATION_FAILURE))


def test_web_results_for_an_oversized_prompt_box_say_so():
    page = snapshot("10 sites\nHere are top web results for exploring this topic:")
    with pytest.raises(gf.AIModeError, match="prompt box text is too long") as caught:
        gf._extract_rendered_content(page, prompt_url_chars=53000)
    assert caught.value.kind == gf.KIND_TOO_LONG
    with pytest.raises(RuntimeError, match="empty answer"):
        gf._extract_rendered_content(snapshot(""))


# ============================================================================ the page flow
class Page:
    """A scripted AI Mode page: ``run_js`` answers the route's scripts and records them."""

    def __init__(self, answer="Bonjour", models=("Auto", "3.8 Flash", "4 Carbon"), code_blocks=(), first=None,
                 chips=True, uploads=True, lose=0, lose_answers=False):
        self.answer, self.models, self.code_blocks = answer, list(models), list(code_blocks)
        self.first, self.chips, self.uploads = first, chips, uploads
        self.lose, self.lose_answers = lose, lose_answers                     # the staging run that is lost
        self.calls, self.staged, self.attached = [], {}, []
        self.sent, self.selected, self.prompt, self.waited = False, None, None, 0

    def wait(self, ms):
        self.waited += ms

    def run_js(self, script, js_timeout_ms=15000):
        if script == gf.AI_MODE_SNAPSHOT_SCRIPT:
            self.calls.append("snapshot")
            if self.first is not None and not self.sent:
                return json.dumps(self.first)
            return json.dumps(snapshot(self.answer if self.sent else "", codeBlocks=self.code_blocks if self.sent else [],
                                       aiMode=self.sent, complete=self.sent))
        if script == gf.AI_MODE_MENU_SCRIPT:
            self.calls.append("menu")
            return json.dumps({"ok": True, "opener": True, "clicked": False, "inputs": ["image/png", ""],
                               "models": [{"label": label, "checked": index == 0, "disabled": False}
                                          for index, label in enumerate(self.models)]})
        if "const want = " in script:
            want = json.loads(re.search(r'const want = (".*?");', script).group(1))
            self.calls.append(f"select:{want}")
            label = next((item for item in self.models if want in item.lower().split("•")[0]), None)
            if label is None:
                return json.dumps({"ok": False, "error": "not listed", "models": self.models})
            self.selected = label
            return json.dumps({"ok": True, "label": label, "already": False, "models": self.models})
        if "b64 +=" in script:
            self.calls.append("stage")
            index = int(re.search(r"const index = (\d+);", script).group(1))
            name, mime, kind, chunk = (json.loads(value) for value in re.search(
                r'\{name: (".*?"), mime: (".*?"), kind: (".*?"), b64: ""\}; staged\[index\]\.b64 \+= (".*?");',
                script).groups())
            entry = self.staged.setdefault(index, {"name": name, "mime": mime, "kind": kind, "b64": ""})
            if self.calls.count("stage") == self.lose:
                return len(entry["b64"]) if self.lose_answers else None       # the piece is not in the page
            entry["b64"] += chunk
            return float(len(entry["b64"]))                                   # QtWebEngine numbers are floats
        if "new DataTransfer()" in script:
            kind = json.loads(re.search(r'const kind = (".*?");', script).group(1))
            self.calls.append(f"attach:{kind}")
            found = [entry["name"] for _index, entry in sorted(self.staged.items()) if entry["kind"] == kind]
            self.attached += found
            return json.dumps({"ok": True, "names": found})
        if "crupload" in script:
            self.calls.append("uploads")
            asked = json.loads(re.search(r"const names = (\[.*?\]);", script).group(1))
            assert asked == [entry["name"] for _index, entry in sorted(self.staged.items())]
            return json.dumps({"chips": self.attached if self.chips else [],
                               "uploads": len(self.attached) if self.uploads else 0})
        if "AI Mode textarea not found" in script:
            self.calls.append("prompt")
            self.prompt = json.loads(re.search(r'const promptText = (".*?");', script).group(1))
            return json.dumps({"ok": True})
        if "AI Mode send button not ready" in script:
            self.calls.append("send")
            self.sent = True
            return json.dumps({"ok": True, "label": "Send"})
        raise AssertionError(f"unexpected script: {script[:80]}")


def submit(page, request, **kwargs):
    return gf._submit_ai_mode_request(request, run_js=page.run_js, wait_ms=page.wait,
                                      base_url=gf._build_search_base_url(), timeout=30, **kwargs)


def test_flow_of_a_plain_request_asks_for_auto():
    page = Page()
    result = submit(page, gf._build_ai_mode_request([{"role": "user", "content": "Say hello in French."}]))
    assert page.calls == ["snapshot", "menu", "select:auto", "prompt", "send", "snapshot"]
    assert page.selected == "Auto" and result["model_label"] == "Auto"
    assert page.prompt == "Say hello in French." and result["answerText"] == "Bonjour"
    assert gf._extract_rendered_content(result) == "Bonjour"
    # a page without "Auto" (signed out: one model) keeps the model it has
    signed_out = Page(models=("4 Hydrogen • Everyday help",))
    assert submit(signed_out, gf._build_ai_mode_request([{"role": "user", "content": "hi"}]))["model_label"] == ""
    assert signed_out.selected is None and signed_out.sent


def test_flow_attaches_files_then_images_before_typing(clean):
    clean.setattr(gf, "FILE_CHUNK_CHARS", 16)                                 # files reach the page in pieces
    chapter = "<p>안녕하세요, 형님.</p>\n<p>둘째 문단입니다.</p>"
    picture = bytes(range(256)) * 3
    code = "<p>Hello, hyung-nim.</p>\n<p>The second paragraph.</p>\n"
    page = Page(answer="html\n\n```\n" + code + "```", code_blocks=[code])
    request = gf._build_ai_mode_request(
        [{"role": "system", "content": "Translate to English."},
         {"role": "user", "content": [{"type": "text", "text": chapter}, png_part(picture)]}],
        "search/gemini-4-carbon")
    result = submit(page, request)
    order = [call for call in page.calls if call != "stage"]
    assert order == ["snapshot", "menu", "select:4 carbon", "menu", "attach:file", "menu", "attach:image",
                     "uploads", "prompt", "send", "snapshot"]
    assert page.calls.count("stage") > 40 and page.calls.index("stage") < page.calls.index("attach:file")
    assert page.selected == "4 Carbon"
    assert base64.b64decode(page.staged[0]["b64"]).decode("utf-8") == chapter
    assert base64.b64decode(page.staged[1]["b64"]) == picture
    assert (page.staged[0]["name"], page.staged[0]["mime"]) == ("chapter.html", "text/plain")
    assert (page.staged[1]["name"], page.staged[1]["mime"]) == ("image001.png", "image/png")
    assert page.attached == ["chapter.html", "image001.png"]
    assert "안녕" not in page.prompt
    assert gf._extract_rendered_content(result, prefer_html=True) == code.strip()
    assert result["submit_state"]["upload"] == {"chips": ["chapter.html", "image001.png"], "uploads": 2}


def test_flow_refuses_a_model_the_page_does_not_offer():
    page = Page(models=("4 Hydrogen • Everyday help",))
    with pytest.raises(gf.AIModeError, match="does not offer the model '4 carbon'") as caught:
        submit(page, gf._build_ai_mode_request([{"role": "user", "content": "hi"}], "search/gemini-4-carbon"))
    assert caught.value.kind == gf.KIND_SIGN_IN and "Google Login" in str(caught.value)
    assert "4 Hydrogen" in str(caught.value) and not page.sent


def test_flow_stops_at_a_page_that_needs_the_user():
    consent = snapshot(url="https://consent.google.com/m?continue=x", text="Before you continue to Google",
                       aiMode=False, promptBox=False)
    page = Page(first=consent)
    result = submit(page, gf._build_ai_mode_request([{"role": "user", "content": "hi"}]))
    assert page.calls == ["snapshot"] and not page.sent                       # nothing typed, nothing clicked
    with pytest.raises(gf.AIModeError) as caught:
        gf._extract_rendered_content(result)
    assert caught.value.kind == gf.KIND_CONSENT


def test_flow_fails_when_an_attachment_is_not_taken(clean):
    clock = {"now": 1000.0}
    clean.setattr(gf.time, "time", lambda: clock["now"])
    page = Page(chips=False)
    page.wait = lambda ms: clock.__setitem__("now", clock["now"] + ms / 1000.0)
    with pytest.raises(gf.AIModeError, match="did not take the attachment chapter.html") as caught:
        submit(page, gf._build_ai_mode_request([{"role": "system", "content": "Translate."},
                                                {"role": "user", "content": "<p>안녕</p>"}]))
    assert caught.value.kind == gf.KIND_UPLOAD and not page.sent


@pytest.mark.parametrize("answers", [False, True])
def test_flow_fails_when_a_piece_of_an_attachment_is_lost(clean, answers):
    # a script that failed or timed out (run_js gives None), or a page that did not add the piece
    clean.setattr(gf, "FILE_CHUNK_CHARS", 16)                                 # a multiple of 4, like the real size:
    page = Page(lose=3, lose_answers=answers)                                 # what is left still decodes
    request = gf._build_ai_mode_request([{"role": "system", "content": "Translate."},
                                         {"role": "user", "content": "<p>안녕하세요, 형님.</p>\n<p>둘째 문단입니다.</p>"}])
    with pytest.raises(gf.AIModeError, match="chapter.html did not reach the AI Mode page whole") as caught:
        submit(page, request)
    assert caught.value.kind == gf.KIND_UPLOAD and caught.value.kind not in gf.FATAL_KINDS
    assert ("did not answer" in str(caught.value)) is (not answers)
    assert ("holds 32 of 48 characters" in str(caught.value)) is answers
    assert page.calls.count("stage") == 3                                     # nothing after the lost piece,
    assert not page.attached and page.prompt is None and not page.sent        # nothing attached, typed or sent
    assert [gf._staged_length(value) for value in (48, 48.0, "48", None, "", "x")] == [48, 48, 48, -1, -1, -1]


def test_flow_lists_the_models_of_the_page():
    page = Page(models=("Auto", "3.8 Flash", "4 Carbon"))
    result = submit(page, {"list_models": True})
    assert page.calls == ["snapshot", "menu"] and not page.sent
    assert gf._model_ids(result["models"]) == ["search/gemini", "search/gemini-3.8-flash", "search/gemini-4-carbon"]


# ============================================================================ entry points
def test_signed_out_requests_use_the_helper_and_signed_in_the_google_profile(clean, tmp_path):
    used = []
    clean.setattr(gf, "_run_search_subprocess_once", lambda **kw: used.append("helper") or {"content": "a"})
    clean.setattr(gf, "_send_chat_completion_chrome", lambda **kw: used.append("chrome") or {"content": "b"})
    ask = {"messages": [{"role": "user", "content": "hi"}], "model": "search/gemini"}
    assert gf.profile_dir() == tmp_path / ".glossarion" / "search_gemini_google"
    assert not gf.has_session()
    assert gf.send_chat_completion(**ask)["content"] == "a"
    gf._mark_session(True)
    assert gf.has_session() and gf.send_chat_completion(**ask)["content"] == "b"
    gf._mark_session(False)
    assert not gf.has_session() and gf.send_chat_completion(**ask)["content"] == "a"
    assert used == ["helper", "chrome", "helper"]


def test_list_models_reads_the_helper_page(clean):
    asked = {}

    def helper(**kwargs):
        asked.update(kwargs)
        return {"content": "", "models": ["4 Hydrogen • Everyday help"], "url": "https://www.google.com/search"}

    clean.setattr(gf, "_run_search_subprocess_once", helper)
    assert gf.list_models(timeout=60) == ["search/gemini", "search/gemini-4-hydrogen"]
    assert asked["list_models"] is True and asked["messages"] == []
    clean.setattr(gf, "_run_search_subprocess_once", lambda **kw: {
        "models": [], "url": "https://consent.google.com/m", "text": "Before you continue to Google"})
    with pytest.raises(gf.AIModeError) as caught:
        gf.list_models()
    assert caught.value.kind == gf.KIND_CONSENT


def test_helper_error_kind_crosses_the_process(clean):
    commands = []

    class Proc:
        returncode = 1

        def __init__(self, cmd, **kwargs):
            commands.append(cmd)

        def communicate(self, timeout=None):
            return json.dumps({"content": "", "finish_reason": "error", "error": "Google showed its cookie consent page",
                               "error_kind": gf.KIND_CONSENT}), ""

        def poll(self):
            return 1

    clean.setattr(gf.subprocess, "Popen", Proc)
    with pytest.raises(gf.AIModeError, match="cookie consent page") as caught:
        gf.send_chat_completion(messages=[{"role": "user", "content": "hi"}], model="search/gemini")
    assert caught.value.kind == gf.KIND_CONSENT
    assert "--list-models" not in commands[0]
    with pytest.raises(gf.AIModeError):
        gf.list_models()
    assert "--list-models" in commands[1]


def test_unified_client_reports_fatal_kinds_without_retrying():
    source = (SRC_DIR / "unified_api_client.py").read_text(encoding="utf-8")
    handler = source[source.index("    def _send_search_gemini("):source.index("    def _send_search_opera(")]
    assert 'error_kind in ("consent", "sign_in")' in handler
    assert 'error_kind in ("unavailable", "too_long", "incomplete")' in handler
    for kind in (gf.KIND_CONSENT, gf.KIND_SIGN_IN, gf.KIND_UNAVAILABLE, gf.KIND_TOO_LONG, gf.KIND_INCOMPLETE):
        assert kind in gf.FATAL_KINDS and f'"{kind}"' in handler


# ============================================================================ the removed settings
REMOVED_KEYS = (
    "gemini_free_adaptive_split", "gemini_free_html_text_node_transport", "gemini_free_subchunk_prompt_chars",
    "gemini_free_subchunk_url_chars", "gemini_free_subchunk_safety_chars", "gemini_free_min_subchunk_body_chars",
    "gemini_free_subchunk_concurrency", "gemini_free_subchunk_start_delay", "gemini_free_subchunk_timeout",
    "gemini_free_subchunk_payload_format", "gemini_free_html_splitter", "gemini_free_subchunk_balancer",
)


def test_chunking_settings_are_gone_everywhere():
    import settings_schema as ss

    assert not [key for key in ss.keys() if key.startswith("gemini_free_")]
    for name in ("other_settings.py", "run_env.py", "settings_persistence.py", "owner_state.py", "gemini_free.py",
                 "settings_schema_data.py", "mobile/tools/frozen_desktop_tables.py",
                 "mobile/app/glossarion_mobile/ui/settings/model.py"):
        source = (SRC_DIR / name).read_text(encoding="utf-8-sig")
        for key in REMOVED_KEYS:
            assert key not in source and key.upper() not in source, (name, key)
        assert "Gemini Free Browser Chunking" not in source and "SUBCHUNK" not in source.upper(), name
    for name in ("_split_messages_for_search_budget", "_build_html_text_node_transport", "_adaptive_split_enabled",
                 "_run_search_subprocess_sequential", "_send_chat_completion_split"):
        assert not hasattr(gf, name), name


def test_old_config_keys_are_ignored(clean):
    """A config written by an older version still holds the keys: nothing reads them, and their
    env values, were they still set, change nothing in the request."""
    plain = gf._build_ai_mode_request([{"role": "system", "content": "Translate."},
                                       {"role": "user", "content": "<p>" + "가" * 9000 + "</p>"}])
    for key in REMOVED_KEYS:
        clean.setenv(key.upper(), "1")
    clean.setenv("GEMINI_FREE_SUBCHUNK_PROMPT_CHARS", "300")
    again = gf._build_ai_mode_request([{"role": "system", "content": "Translate."},
                                       {"role": "user", "content": "<p>" + "가" * 9000 + "</p>"}])
    assert again == plain and names(again) == [("chapter.html", "file")]      # one request, never split

    # the settings owner starts on such a config (its own process: startup writes os.environ)
    old_config = {key: 1 for key in REMOVED_KEYS}
    old_config.update(gemini_free_subchunk_payload_format="html", gemini_free_html_splitter="regex",
                      gemini_free_subchunk_start_delay="oops", model="search/gemini")
    code = (
        "import json, os, sys, types\n"
        "from headless_owner import HeadlessOwner\n"
        f"config = json.loads({json.dumps(json.dumps(old_config))})\n"
        "owner = HeadlessOwner(config, host=types.SimpleNamespace(log=lambda _m: None))\n"
        "assert owner.config['model'] == 'search/gemini'\n"
        "assert not [name for name in vars(owner) if name.startswith('gemini_free_')]\n"
        "assert not [name for name in os.environ if name.startswith('GEMINI_FREE_')], sorted(os.environ)\n"
        "print('ok')\n"
    )
    env = {name: value for name, value in os.environ.items() if not name.startswith("GEMINI_FREE_")}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(SRC_DIR), os.environ.get("PYTHONPATH"))))
    env["GLOSSARION_HTTP_LOG"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"  # startup prints emoji; a Windows pipe is cp1252 otherwise
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300,
                          cwd=str(SRC_DIR))
    assert proc.returncode == 0 and proc.stdout.strip().endswith("ok"), proc.stderr[-2000:]
