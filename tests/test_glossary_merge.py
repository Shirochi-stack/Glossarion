"""One glossary block per request: book glossary + extension + unified glossary (merge_glossary_texts)."""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from glossary_compressor import merge_glossary_texts  # noqa: E402

HEADER = "Glossary Columns: raw_name, translated_name, gender, description"


def test_sections_merge_and_blank_runs_go():
    # what compression leaves: every section's blank separator, also for sections with no match
    book = HEADER + "\n\n\n\n\n=== SURNAMES ===\n* 정 = Jung: a surname\n\n=== TERMS ===\n* 공지 = Notice: x\n\n\n"
    extension = HEADER + "\n\n\n\n\n=== TERMS ===\n* 수술 = Surgery: y\n\n=== TITLES ===\n* 어머니 = Mother [Female]: z\n\n\n"
    assert merge_glossary_texts([book, extension, ""]) == (
        HEADER + "\n\n=== SURNAMES ===\n* 정 = Jung: a surname\n\n=== TERMS ===\n* 공지 = Notice: x\n"
        "* 수술 = Surgery: y\n\n=== TITLES ===\n* 어머니 = Mother [Female]: z")


def test_one_glossary_is_tidied_and_nothing_is_empty():
    assert merge_glossary_texts([HEADER + "\n\n\n\n=== TERMS ===\n* a = A\n\n\n"]) == HEADER + "\n\n=== TERMS ===\n* a = A"
    assert merge_glossary_texts(["", "  \n", None]) == ""


def test_other_formats_are_joined_as_they_were():
    csv_book = "type,raw_name,translated_name\ncharacter,강,Kang\n\n\n\n"
    extension = HEADER + "\n\n=== TERMS ===\n* a = A\n"
    assert merge_glossary_texts([csv_book, extension]) == (
        "type,raw_name,translated_name\ncharacter,강,Kang\n\n" + HEADER + "\n\n=== TERMS ===\n* a = A")
    # different columns are not forced under one header
    other = "Glossary Columns: raw_name, translated_name\n\n=== TERMS ===\n* b = B\n"
    assert merge_glossary_texts([extension, other]).count("Glossary Columns:") == 2
