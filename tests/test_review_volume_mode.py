import os
import sys
import time
import types

import pytest

import review_generator


def test_extract_review_chapters_preserves_manual_file_order(monkeypatch):
    calls = []

    def fake_extract(path, log_fn=print):
        calls.append(path)
        return [("chapter.xhtml", f"text from {os.path.basename(path)}")]

    monkeypatch.setattr(review_generator, "extract_chapter_texts", fake_extract)
    paths = [os.path.join("books", "Novel 10.epub"), os.path.join("books", "Novel 2.epub")]

    chapters = review_generator.extract_review_chapters(paths, log_fn=lambda _message: None)

    assert calls == paths
    assert chapters == [
        ("Volume 1/2: Novel 10.epub / chapter.xhtml", "text from Novel 10.epub"),
        ("Volume 2/2: Novel 2.epub / chapter.xhtml", "text from Novel 2.epub"),
    ]


def test_extract_review_chapters_keeps_legacy_names_for_one_file(monkeypatch):
    expected = [("chapter-1.xhtml", "chapter text")]
    monkeypatch.setattr(
        review_generator,
        "extract_chapter_texts",
        lambda path, log_fn=print: expected,
    )

    assert review_generator.extract_review_chapters(["book.epub"]) == expected


def test_count_review_tokens_totals_every_volume(monkeypatch):
    monkeypatch.setattr(
        review_generator,
        "extract_review_chapters",
        lambda paths, log_fn=print: [("one", "abcd"), ("two", "abcdef")],
    )
    monkeypatch.setattr(review_generator, "_get_encoder", lambda: object())
    monkeypatch.setattr(review_generator, "count_tokens", len)

    assert review_generator.count_review_tokens(["v1.epub", "v2.epub"]) == 10


def test_generate_review_sends_ordered_files_in_one_request(tmp_path, monkeypatch):
    paths = ["Novel 2.epub", "Novel 10.epub"]
    observed_inputs = []
    sent_messages = []

    def fake_extract(review_input, log_fn=print):
        observed_inputs.append(list(review_input))
        return [("Volume 1", "first text"), ("Volume 2", "second text")]

    class FakeClient:
        _multi_key_mode = False
        context = None

    unified_module = types.ModuleType("unified_api_client")
    unified_module.UnifiedClient = object
    extractor_module = types.ModuleType("extract_glossary_from_epub")
    extractor_module.create_client_with_multi_key_support = (
        lambda *_args, **_kwargs: FakeClient()
    )
    translator_module = types.ModuleType("TransateKRtoEN")

    def fake_send(messages, _client, **_kwargs):
        sent_messages.append(messages)
        return "one combined review", "stop", None

    translator_module.send_with_interrupt = fake_send
    monkeypatch.setitem(sys.modules, "unified_api_client", unified_module)
    monkeypatch.setitem(sys.modules, "extract_glossary_from_epub", extractor_module)
    monkeypatch.setitem(sys.modules, "TransateKRtoEN", translator_module)
    monkeypatch.setattr(review_generator, "extract_review_chapters", fake_extract)
    monkeypatch.setattr(review_generator, "count_tokens", lambda text: len(text))

    result = review_generator.generate_review(
        epub_path=paths,
        output_dir=str(tmp_path),
        api_key="key",
        model="model",
        endpoint="",
        system_prompt="review",
        input_token_limit=10_000,
        spoiler_mode=False,
        temperature=0.3,
        config={},
        log_fn=lambda _message: None,
    )

    assert result == "one combined review"
    assert observed_inputs == [paths]
    assert len(sent_messages) == 1
    user_content = sent_messages[0][1]["content"]
    assert user_content.index("first text") < user_content.index("second text")
    assert (tmp_path / "review" / "review.md").read_text(encoding="utf-8") == result


def test_volume_filename_sort_is_numerical():
    pytest.importorskip("PySide6")
    from review_dialog import _natural_path_key

    paths = ["Novel 10.epub", "novel 2.epub", "Novel 1.epub"]
    assert sorted(paths, key=_natural_path_key) == [
        "Novel 1.epub",
        "novel 2.epub",
        "Novel 10.epub",
    ]


def test_volume_mode_ui_uses_combined_token_total_and_reorder_controls(
    tmp_path, monkeypatch
):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication
    import review_dialog

    first = tmp_path / "Novel 10.txt"
    second = tmp_path / "Novel 2.txt"
    first.write_text("ten", encoding="utf-8")
    second.write_text("two", encoding="utf-8")

    monkeypatch.setattr(review_dialog, "count_epub_tokens", lambda _path: 3)
    monkeypatch.setattr(review_dialog, "count_review_tokens", lambda paths: 6)

    class FakeGui:
        def __init__(self):
            self.selected_files = [str(first), str(second)]
            self.config = {}
            self.base_dir = str(tmp_path)

        def save_config(self, show_message=False):
            return None

    app = QApplication.instance() or QApplication([])
    gui = FakeGui()
    dialog = review_dialog.ReviewDialog(None, gui, str(first))
    try:
        assert dialog._all_epub_paths == [str(first), str(second)]
        assert dialog._volume_paths == [str(second), str(first)]

        dialog.volume_mode_checkbox.setChecked(True)
        deadline = time.monotonic() + 2
        while "Volume tokens" not in dialog.token_label.text() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)

        assert dialog._review_input() == [str(second), str(first)]
        assert dialog.token_label.text() == "📚 Volume tokens (2 files): 6"
        assert gui.review_volume_mode_var is True
        assert not dialog._volume_order_btn.isHidden()
        assert dialog.generate_all_btn.isHidden()
        assert dialog._get_review_path().endswith(
            os.path.join("Novel 2_Volume", "review", "review.md")
        )

        order_dialog = review_dialog.VolumeOrderDialog(dialog, dialog._volume_paths)
        order_dialog.file_list.setCurrentRow(1)
        order_dialog._move_selected(-1)
        assert order_dialog.ordered_paths() == [str(first), str(second)]
        order_dialog.close()
    finally:
        dialog.close()
