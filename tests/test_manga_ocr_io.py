from dataclasses import dataclass

import manga_ocr_io


@dataclass
class FakeTextRegion:
    text: str
    vertices: list
    bounding_box: tuple
    confidence: float
    region_type: str
    translated_text: str | None = None
    bubble_bounds: tuple | None = None


def test_document_round_trip_preserves_unicode_and_detector_metadata(tmp_path):
    image = tmp_path / "chapter" / "001.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    region = FakeTextRegion(
        text="こんにちは",
        vertices=[(10, 20), (30, 20), (30, 40), (10, 40)],
        bounding_box=(10, 20, 20, 20),
        confidence=0.98,
        region_type="free_text",
        translated_text="Hello",
        bubble_bounds=(8, 18, 24, 24),
    )
    region.bubble_type = "free_text"
    region.should_inpaint = False

    page = manga_ocr_io.make_page(
        str(image), [region], index=1, source_root=str(tmp_path)
    )
    document = manga_ocr_io.create_document(
        [page], workflow="automatic", source_root=str(tmp_path)
    )
    output = tmp_path / "OCR Text" / "chapter_ocr.json"
    manga_ocr_io.write_document(str(output), document)

    loaded = manga_ocr_io.load_document(str(output))
    loaded_region = loaded["pages"][0]["regions"][0]
    assert loaded_region["text"] == "こんにちは"
    assert loaded_region["translated_text"] == "Hello"
    assert loaded_region["bubble_type"] == "free_text"
    assert loaded_region["should_inpaint"] is False
    assert loaded_region["bounding_box"] == [10, 20, 20, 20]


def test_page_matching_survives_moved_root_and_duplicate_filenames(tmp_path):
    old_root = tmp_path / "old"
    old_a = old_root / "volume-a" / "001.png"
    old_b = old_root / "volume-b" / "001.png"
    old_a.parent.mkdir(parents=True)
    old_b.parent.mkdir(parents=True)
    old_a.write_bytes(b"a")
    old_b.write_bytes(b"b")
    document = manga_ocr_io.create_document(
        [
            manga_ocr_io.make_page(str(old_a), [{"text": "A", "bbox": [1, 2, 3, 4]}], index=1, source_root=str(old_root)),
            manga_ocr_io.make_page(str(old_b), [{"text": "B", "bbox": [5, 6, 7, 8]}], index=2, source_root=str(old_root)),
        ],
        workflow="manual-editor",
        source_root=str(old_root),
    )

    new_root = tmp_path / "moved"
    new_a = new_root / "volume-a" / "001.png"
    new_b = new_root / "volume-b" / "001.png"
    new_a.parent.mkdir(parents=True)
    new_b.parent.mkdir(parents=True)
    new_a.write_bytes(b"a")
    new_b.write_bytes(b"b")

    matches = manga_ocr_io.match_document_pages(document, [str(new_b), str(new_a)])
    assert matches[str(new_a)]["regions"][0]["text"] == "A"
    assert matches[str(new_b)]["regions"][0]["text"] == "B"


def test_editor_state_merges_ocr_text_with_rectangle_mapping():
    state = {
        "viewer_rectangles": [
            {
                "x": 10,
                "y": 20,
                "width": 100,
                "height": 50,
                "shape": "ellipse",
                "bubble_type": "text_bubble",
            }
        ],
        "recognized_texts": [
            {"region_index": 0, "bbox": [10, 20, 100, 50], "text": "원문"}
        ],
        "translated_texts": [
            {
                "original": {"region_index": 0, "text": "원문"},
                "translation": "Original",
                "bbox": [10, 20, 100, 50],
            }
        ],
    }

    regions = manga_ocr_io.canonical_regions_from_editor_state(state)
    assert regions[0]["text"] == "원문"
    assert regions[0]["translated_text"] == "Original"
    assert regions[0]["bounding_box"] == [10, 20, 100, 50]
    assert regions[0]["shape"] == "ellipse"


def test_pipeline_page_can_be_imported_into_manual_editor():
    page = {
        "regions": [
            {
                "text": "縦書き",
                "bounding_box": [12, 18, 30, 90],
                "confidence": 0.9,
                "region_type": "text_bubble",
                "bubble_type": "text_bubble",
                "translated_text": None,
            }
        ]
    }
    state = manga_ocr_io.editor_state_from_page(page)
    assert state["viewer_rectangles"][0] == {
        "x": 12,
        "y": 18,
        "width": 30,
        "height": 90,
        "shape": "rect",
        "bubble_type": "text_bubble",
        "region_type": "text_bubble",
    }
    assert state["recognized_texts"][0]["text"] == "縦書き"


def test_region_record_rehydrates_pipeline_object_metadata():
    record = {
        "text": "text",
        "bbox": [1, 2, 30, 40],
        "coords": [[1, 2], [31, 2], [31, 42], [1, 42]],
        "confidence": 0.75,
        "region_type": "free_text",
        "bubble_type": "free_text",
        "should_inpaint": False,
        "bubble_bounds": [0, 1, 32, 42],
    }
    region = manga_ocr_io.region_record_to_text_region(record, FakeTextRegion)
    assert region.bounding_box == (1, 2, 30, 40)
    assert region.bubble_bounds == (0, 1, 32, 42)
    assert region.bubble_type == "free_text"
    assert region.should_inpaint is False


def test_import_merges_region_translation_into_ocr_only_editor_snapshot():
    page = {
        "regions": [
            {
                "text": "Source text",
                "translated_text": "Translated text",
                "bounding_box": [10, 20, 100, 50],
            }
        ],
        "editor_state": {
            "viewer_rectangles": [
                {"x": 10, "y": 20, "width": 100, "height": 50, "shape": "ellipse"}
            ],
            "recognized_texts": [
                {"region_index": 0, "bbox": [10, 20, 100, 50], "text": "Source text"}
            ],
        },
    }

    state = manga_ocr_io.editor_state_from_page(page)

    assert state["viewer_rectangles"][0]["shape"] == "ellipse"
    assert state["recognized_texts"][0]["text"] == "Source text"
    assert state["translated_texts"][0]["translation"] == "Translated text"
