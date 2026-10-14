import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from sdlxliff_converter import convert_sdlxliff
from sdlxliff_extractor import extract_sdlxliff_to_chapters, iter_eligible_segments


XLIFF_NS = "urn:oasis:names:tc:xliff:document:1.2"
SDL_NS = "http://sdl.com/FileTypes/SdlXliff/1.0"
NS = {"x": XLIFF_NS, "sdl": SDL_NS}


def _sdlxliff(units):
    body = "\n".join(units)
    return f'''<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="{XLIFF_NS}" xmlns:sdl="{SDL_NS}" version="1.2">
  <file>
    <body>
{body}
    </body>
  </file>
</xliff>
'''


def _unit(unit_id, mid, source):
    return f'''      <trans-unit id="{unit_id}">
        <source><mrk mtype="seg" mid="{mid}">{source}</mrk></source>
        <seg-source><mrk mtype="seg" mid="{mid}">{source}</mrk></seg-source>
        <target><mrk mtype="seg" mid="{mid}"></mrk></target>
        <sdl:seg-defs><sdl:seg id="{mid}" conf="Draft"/></sdl:seg-defs>
      </trans-unit>'''


def _write_progress(output_dir, filename, status="completed"):
    with open(os.path.join(output_dir, "translation_progress.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "chapters": {
                    "1": {
                        "actual_num": 1,
                        "output_file": filename,
                        "status": status,
                    }
                }
            },
            f,
        )


class SdlxliffJsonBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = self.tmp.name
        self.old_available = os.environ.get("SDLXLIFF_AVAILABLE_TOKENS")

    def tearDown(self):
        if self.old_available is None:
            os.environ.pop("SDLXLIFF_AVAILABLE_TOKENS", None)
        else:
            os.environ["SDLXLIFF_AVAILABLE_TOKENS"] = self.old_available

    def _write_input(self, xml):
        path = os.path.join(self.base, "sample.sdlxliff")
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml)
        return path

    def test_json_batch_updates_multiple_segments_and_preserves_inline_text(self):
        os.environ["SDLXLIFF_AVAILABLE_TOKENS"] = "100000"
        input_path = self._write_input(
            _sdlxliff(
                [
                    _unit("u1", "1", 'Hello <g id="b">bold source</g> world'),
                    _unit("u2", "2", ' leading <x id="x1"/> text '),
                ]
            )
        )
        out = os.path.join(self.base, "out")
        result = extract_sdlxliff_to_chapters(input_path, out)
        self.assertEqual(result["chapters"], 1)

        with open(result["chapters_path"], encoding="utf-8") as f:
            chapters = json.load(f)
        records = json.loads(chapters[0]["body"])
        self.assertEqual([record["id"] for record in records], ["1", "2"])
        self.assertIn("bold source", records[0]["source"])
        self.assertTrue(records[1]["source"].startswith(" leading "))
        self.assertTrue(records[1]["source"].endswith(" text "))

        p0, p1 = records[0]["source"].split("bold source")
        p0 = p0.split("Hello ", 1)[1]
        p1 = p1.split(" world", 1)[0]
        p2 = records[1]["source"].split(" leading ", 1)[1].split(" text ", 1)[0]
        translated = [
            {"id": "1", "target": f"Bonjour {p0}gras{p1} monde"},
            {"id": "2", "target": f" avant {p2} texte "},
        ]
        with open(os.path.join(out, "response_section_1.txt"), "w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False)
        _write_progress(out, "response_section_1.txt")

        converted = convert_sdlxliff(out)
        self.assertEqual(converted["updated"], 2)
        tree = etree.parse(converted["output_path"])
        targets = tree.xpath("//x:target/x:mrk", namespaces=NS)
        self.assertEqual("".join(targets[0].itertext()), "Bonjour gras monde")
        self.assertEqual(targets[0][0].text, "gras")
        self.assertEqual(targets[1].text, " avant ")
        self.assertEqual(targets[1][0].tail, " texte ")

    def test_placeholder_only_segments_are_excluded_from_batches_and_auto_inserted(self):
        os.environ["SDLXLIFF_AVAILABLE_TOKENS"] = "100000"
        input_path = self._write_input(
            _sdlxliff(
                [
                    _unit("tag1", "1", '<x id="x1"/>'),
                    _unit("tag2", "2", '<x id="x2"/>'),
                    _unit("text", "3", "Translate me"),
                ]
            )
        )
        out = os.path.join(self.base, "out")
        result = extract_sdlxliff_to_chapters(input_path, out)
        self.assertEqual(result["segments"], 3)
        self.assertEqual(result["translatable_segments"], 1)
        self.assertEqual(result["auto_insert_segments"], 2)
        self.assertEqual(result["chapters"], 1)

        with open(result["manifest_path"], encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["auto_insert_segment_count"], 2)
        self.assertEqual([segment["auto_insert"] for segment in manifest["segments"]], [True, True, False])

        with open(result["chapters_path"], encoding="utf-8") as f:
            chapters = json.load(f)
        records = json.loads(chapters[0]["body"])
        self.assertEqual(records, [{"id": "3", "source": "Translate me"}])

        with open(os.path.join(out, "response_section_1.txt"), "w", encoding="utf-8") as f:
            json.dump([{"id": "3", "target": "Traduisez-moi"}], f)
        _write_progress(out, "response_section_1.txt")

        converted = convert_sdlxliff(out)
        self.assertEqual(converted["updated"], 3)
        self.assertEqual(converted["skipped"], 0)
        tree = etree.parse(converted["output_path"])
        targets = tree.xpath("//x:target/x:mrk", namespaces=NS)
        self.assertEqual(etree.QName(targets[0][0]).localname, "x")
        self.assertEqual(targets[0][0].get("id"), "x1")
        self.assertEqual(etree.QName(targets[1][0]).localname, "x")
        self.assertEqual(targets[1][0].get("id"), "x2")
        self.assertEqual("".join(targets[2].itertext()), "Traduisez-moi")

    def test_duplicate_placeholder_is_rejected(self):
        os.environ["SDLXLIFF_AVAILABLE_TOKENS"] = "100000"
        input_path = self._write_input(_sdlxliff([_unit("u1", "1", 'Hello <x id="x1"/> world')]))
        out = os.path.join(self.base, "out")
        result = extract_sdlxliff_to_chapters(input_path, out)
        with open(result["chapters_path"], encoding="utf-8") as f:
            records = json.loads(json.load(f)[0]["body"])
        placeholder = records[0]["source"].split("Hello ", 1)[1].split(" world", 1)[0]
        with open(os.path.join(out, "response_section_1.txt"), "w", encoding="utf-8") as f:
            json.dump([{"id": "1", "target": f"Bonjour {placeholder} monde {placeholder}"}], f)
        _write_progress(out, "response_section_1.txt")

        converted = convert_sdlxliff(out)
        self.assertEqual(converted["updated"], 0)
        self.assertEqual(converted["skipped"], 1)

    def test_malformed_json_batch_fails_safely(self):
        os.environ["SDLXLIFF_AVAILABLE_TOKENS"] = "100000"
        input_path = self._write_input(_sdlxliff([_unit("u1", "1", "Hello")]))
        out = os.path.join(self.base, "out")
        extract_sdlxliff_to_chapters(input_path, out)
        with open(os.path.join(out, "response_section_1.txt"), "w", encoding="utf-8") as f:
            f.write("not json")
        _write_progress(out, "response_section_1.txt")

        converted = convert_sdlxliff(out)
        self.assertEqual(converted["updated"], 0)
        self.assertEqual(converted["skipped"], 1)
        self.assertEqual(converted["invalid_batches"], 1)

    def test_missing_extra_duplicate_and_reordered_ids_are_rejected(self):
        os.environ["SDLXLIFF_AVAILABLE_TOKENS"] = "100000"
        input_path = self._write_input(_sdlxliff([_unit("u1", "1", "One"), _unit("u2", "2", "Two")]))

        cases = [
            ([{"id": "1", "target": "Un"}], {"updated": 0, "skipped": 2, "invalid_batches": 1}),
            ([{"id": "1", "target": "Un"}, {"id": "2", "target": "Deux"}, {"id": "99", "target": "Extra"}], {"updated": 0, "skipped": 2, "invalid_batches": 1}),
            ([{"id": "1", "target": "Un"}, {"id": "1", "target": "Encore"}], {"updated": 0, "skipped": 2, "invalid_batches": 1}),
            ([{"id": "2", "target": "Deux"}, {"id": "1", "target": "Un"}], {"updated": 0, "skipped": 2, "invalid_batches": 1}),
        ]

        for idx, (payload, expected) in enumerate(cases):
            out = os.path.join(self.base, f"out_ids_{idx}")
            extract_sdlxliff_to_chapters(input_path, out)
            with open(os.path.join(out, "response_section_1.txt"), "w", encoding="utf-8") as f:
                json.dump(payload, f)
            _write_progress(out, "response_section_1.txt")
            converted = convert_sdlxliff(out)
            for key, value in expected.items():
                self.assertEqual(converted[key], value, f"{key} mismatch for case {idx}")

    def test_non_completed_progress_output_is_not_inserted(self):
        os.environ["SDLXLIFF_AVAILABLE_TOKENS"] = "100000"
        input_path = self._write_input(_sdlxliff([_unit("u1", "1", "Hello")]))
        out = os.path.join(self.base, "out")
        extract_sdlxliff_to_chapters(input_path, out)
        with open(os.path.join(out, "response_section_1.txt"), "w", encoding="utf-8") as f:
            json.dump([{"id": "1", "target": "Bonjour"}], f)
        _write_progress(out, "response_section_1.txt", status="qa_failed")

        converted = convert_sdlxliff(out)
        self.assertEqual(converted["updated"], 0)
        self.assertEqual(converted["skipped"], 1)

    def test_batch_cache_reuses_count_until_source_changes(self):
        long_text = " ".join(f"word{i}" for i in range(1300))
        input_path = self._write_input(
            _sdlxliff(
                [
                    _unit("u1", "1", long_text),
                    _unit("u2", "2", long_text),
                    _unit("u3", "3", long_text),
                ]
            )
        )
        out = os.path.join(self.base, "out")
        os.environ["SDLXLIFF_AVAILABLE_TOKENS"] = "1000"
        first = extract_sdlxliff_to_chapters(input_path, out)
        self.assertGreater(first["chapters"], 1)

        os.environ["SDLXLIFF_AVAILABLE_TOKENS"] = "100000"
        cached = extract_sdlxliff_to_chapters(input_path, out)
        self.assertEqual(cached["chapters"], first["chapters"])

        input_path = self._write_input(
            _sdlxliff(
                [
                    _unit("u1", "1", "short one"),
                    _unit("u2", "2", "short two"),
                    _unit("u3", "3", "short three"),
                ]
            )
        )
        changed = extract_sdlxliff_to_chapters(input_path, out)
        self.assertEqual(changed["chapters"], 1)


if __name__ == "__main__":
    unittest.main()
