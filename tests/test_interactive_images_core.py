"""Interactive Images core: block numbering, plan parsing, quote re-location, injection + text check,
file naming, reference slots, chunking, placeholders and the prompt profiles (GUI-free)."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import interactive_images_core as core  # noqa: E402

CHAPTER = (
    '<?xml version="1.0" encoding="utf-8"?>\r\n'
    '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Chapter 12</title><style>p{}</style></head>\r\n'
    '<body>\r\n'
    '<h1>Chapter 12</h1>\r\n'
    '<p class="mine">Ha-eun pushed the iron door open.</p>\r\n'
    '<p>Second &amp; paragraph <b>bold</b>.</p>\r\n'
    '<blockquote><p>Quoted one</p><p>Quoted two</p></blockquote>\r\n'
    '<ul><li>first item</li><li>second item</li></ul>\r\n'
    '<table><tr><td><p>Name: Ha-eun</p></td><td>Level 3</td></tr></table>\r\n'
    '<p>[System] A quest begins.</p>\r\n'
    '<svg viewBox="0 0 1 1"><rect/></svg>\r\n'
    '</body></html>'
)


def _blocks():
    return core.number_blocks(CHAPTER)


# ---- numbering ------------------------------------------------------------------------------

def test_blocks_are_leaf_blocks_in_document_order():
    blocks = _blocks()
    assert [(b.n, b.tag, b.text) for b in blocks] == [
        (1, 'h1', 'Chapter 12'),
        (2, 'p', 'Ha-eun pushed the iron door open.'),
        (3, 'p', 'Second & paragraph bold .'),
        (4, 'p', 'Quoted one'),
        (5, 'p', 'Quoted two'),
        (6, 'li', 'first item'),
        (7, 'li', 'second item'),
        (8, 'table', 'Name: Ha-eun Level 3'),
        (9, 'p', '[System] A quest begins.'),
    ]
    for block in blocks:
        assert CHAPTER[block.start:block.start + len(block.tag) + 1] == '<' + block.tag
        assert CHAPTER[:block.end].endswith(f'</{block.tag}>')
    # a figure after a list item goes after the whole list
    assert CHAPTER[:blocks[5].insert_at].endswith('</ul>')
    assert core.format_paragraphs(blocks[:2]) == "[1] Chapter 12\n[2] Ha-eun pushed the iron door open."


def test_numbering_is_stable_across_injection_and_strip():
    blocks = _blocks()
    new = core.apply_plan(CHAPTER, blocks, [(blocks[1], 'Interactive_Images/Glossarion_ch012_1.png')],
                          [(blocks[8], 'glx-system-message')], book_class=True)
    assert [b.text for b in core.number_blocks(new)] == [b.text for b in blocks]
    assert core.strip_injections(new) == CHAPTER


# ---- plan parsing -----------------------------------------------------------------------------

NAMES = core.glossary_name_map([
    {'type': 'character', 'raw_name': '서하은', 'translated_name': 'Seo Ha-eun', 'aliases': ['Ha-eun']},
    {'type': 'character', 'raw_name': '백도윤', 'translated_name': 'Baek Do-yun'},
    {'type': 'term', 'raw_name': '검', 'translated_name': 'Sword'},
])


def test_parse_plan_takes_the_fenced_json_after_prose():
    reply = ('Here is the plan:\n```json\n{"images":[{"after_paragraph":2,"quote":"Ha-eun pushed",'
             '"characters":["seo ha-eun","Baek Do-yun","Stranger"],"prompt":"An iron door"}],'
             '"classes":[{"paragraph":9,"quote":"[System]","class":"glx-system-message"},'
             '{"paragraph":3,"class":"evil-class"}]}\n```\nDone.')
    plan = core.parse_plan(reply, 9, ['glx-system-message'], 3, glossary_names=NAMES)
    assert len(plan.images) == 1
    image = plan.images[0]
    assert (image.after_paragraph, image.quote, image.prompt) == (2, 'Ha-eun pushed', 'An iron door')
    assert image.characters == ['seo ha-eun', 'Baek Do-yun', 'Stranger']
    assert image.matched == {'seo ha-eun': 'Seo Ha-eun', 'Baek Do-yun': 'Baek Do-yun'}
    assert [(c.paragraph, c.css_class) for c in plan.classes] == [(9, 'glx-system-message')]
    assert any('evil-class' in w for w in plan.warnings)


def test_parse_plan_bare_json_with_leading_prose_and_braces_in_text():
    reply = 'Sure {not json} here: {"images": [], "classes": []}'
    plan = core.parse_plan(reply, 3, [], 2)
    assert plan.images == [] and plan.classes == []


def test_parse_plan_invalid_json_raises():
    with pytest.raises(core.PlanError, match="plan was not JSON"):
        core.parse_plan("I could not find anything to illustrate.", 3, [], 1)


def test_parse_plan_caps_images_and_drops_ones_without_prompt():
    items = ','.join('{"after_paragraph": %d, "prompt": "p%d"}' % (i, i) for i in range(1, 6))
    plan = core.parse_plan('{"images": [{"after_paragraph": 1}, %s]}' % items, 9, [], 2)
    assert [i.prompt for i in plan.images] == ['p1', 'p2']
    assert any('without a prompt' in w for w in plan.warnings)


def test_alias_and_case_name_matching():
    assert core.match_name('Ha-eun', NAMES) == 'Seo Ha-eun'
    assert core.match_name('서하은', NAMES) == 'Seo Ha-eun'
    assert core.match_name('BAEK DO-YUN', NAMES) == 'Baek Do-yun'
    assert core.match_name('Nobody', NAMES) is None
    assert core.character_names([{'type': 'character', 'translated_name': 'A'}, {'type': 'term',
                                 'translated_name': 'B'}, {'raw_name': 'C'}]) == ['A', 'C']


# ---- quote re-location --------------------------------------------------------------------------

def test_quote_fixes_an_off_by_two_paragraph_number():
    blocks = _blocks()
    assert core.locate_block(blocks, 2, 'Ha-eun pushed').n == 2
    assert core.locate_block(blocks, 4, 'Ha-eun pushed the iron').n == 2    # off by two
    assert core.locate_block(blocks, 9, '[system] a quest').n == 9          # case / punctuation
    assert core.locate_block(blocks, 1, 'Ha-eun pushd the iron door').n == 2  # near match
    assert core.locate_block(blocks, 3, 'Completely unrelated words here') is None
    assert core.locate_block(blocks, 5, '').n == 5


# ---- injection + text check ---------------------------------------------------------------------

def test_injection_keeps_the_visible_text_and_rest_of_the_file():
    blocks = _blocks()
    src = 'Interactive_Images/Glossarion_ch012_1.png'
    new = core.apply_plan(CHAPTER, blocks, [(blocks[1], src), (blocks[6], 'x.png')],
                          [(blocks[1], 'glx-dialogue'), (blocks[7], 'glx-status-window')], book_class=True)
    assert core.check_text_unchanged(CHAPTER, new) == (True, '')
    assert ('<p class="mine glx-dialogue" data-glx-cls="glx-dialogue">Ha-eun pushed the iron door open.</p>'
            f'<figure class="glx-figure" data-glx="img"><img class="glx-illustration" src="{src}" alt=""/></figure>') in new
    assert '</ul><figure class="glx-figure"' in new
    assert '<table class="glx-status-window" data-glx-cls="glx-status-window">' in new
    assert '<body>' + core.BOOK_WRAPPER_OPEN in new and core.BOOK_WRAPPER_CLOSE + '</body>' in new
    assert '<svg viewBox="0 0 1 1">' in new  # untouched bytes elsewhere
    assert core.strip_injections(new) == CHAPTER
    assert core.has_injections(new) and not core.has_injections(CHAPTER)


def test_text_check_rejects_any_text_change():
    ok, hint = core.check_text_unchanged(CHAPTER, CHAPTER.replace('iron door', 'wooden door'))
    assert not ok and 'iron' in hint
    ok, _hint = core.check_text_unchanged(CHAPTER, CHAPTER.replace('<p>Second', '<p>Extra</p><p>Second'))
    assert not ok


def test_strip_keeps_user_classes_and_only_removes_glossarions():
    html = '<body><p class="a glx-dialogue b" data-glx-cls="glx-dialogue">x</p><p class="glx-dialogue">y</p></body>'
    assert core.strip_injections(html) == '<body><p class="a b">x</p><p class="glx-dialogue">y</p></body>'


def test_visible_text_hash_matches_the_translators_content_hash():
    from TransateKRtoEN import ContentProcessor

    blocks = _blocks()
    new = core.apply_plan(CHAPTER, blocks, [(blocks[0], 'Interactive_Images/Glossarion_ch012_1.png')],
                          [(blocks[1], 'glx-dialogue')], book_class=True)
    samples = [CHAPTER, new, '<p>plain</p>', '<html><body><script>x()</script><p>a  b\n c</p></body></html>']
    for html in samples:
        assert core.visible_text_hash(html) == ContentProcessor.get_content_hash(html)
    assert core.visible_text_hash(new) == core.visible_text_hash(CHAPTER)


# ---- names --------------------------------------------------------------------------------------

def test_file_names_are_fixed_per_chapter_and_index():
    assert core.image_filename(12, 1) == 'Glossarion_ch012_1.png'
    assert core.image_filename('7', 3, 'jpeg') == 'Glossarion_ch007_3.jpg'
    assert core.image_filename(1234, 2, 'webp') == 'Glossarion_ch1234_2.webp'
    assert core.image_src('Glossarion_ch012_1.png') == 'Interactive_Images/Glossarion_ch012_1.png'
    for src in ('Interactive_Images/Glossarion_ch012_1.png', '../Interactive_Images/Glossarion_ch012_1.png',
                'Glossarion_ch3_2.jpg', 'interactive_images/anything.png'):
        assert core.is_interactive_image_src(src), src
    for src in ('images/ch12.png', 'Glossarion_logo.png', 'data:image/png;base64,AAA', '', None):
        assert not core.is_interactive_image_src(src), src


def test_chapter_images_on_disk(tmp_path):
    folder = tmp_path / core.INTERACTIVE_FOLDER
    folder.mkdir()
    for name in ('Glossarion_ch012_2.png', 'Glossarion_ch012_1.jpg', 'Glossarion_ch013_1.png', 'other.png'):
        (folder / name).write_bytes(b'x')
    assert core.chapter_images_on_disk(str(tmp_path), 12) == ['Glossarion_ch012_1.jpg', 'Glossarion_ch012_2.png']


# ---- references ---------------------------------------------------------------------------------

def test_reference_slots_characters_round_robin_then_style():
    refs = {'characters': {'A': ['a1', 'a2', 'a3'], 'B': ['b1']}, 'style': ['s1', 's2']}
    assert core.select_references(refs, ['A', 'B'], 5) == [('A', 'a1'), ('B', 'b1'), ('A', 'a2'), ('A', 'a3'),
                                                            (None, 's1')]
    assert core.select_references(refs, ['b', 'A'], 2) == [('b', 'b1'), ('A', 'a1')]  # names match case-insensitively
    assert core.select_references(refs, [], 1) == [(None, 's1')]
    assert core.select_references(refs, ['A'], 0) == []
    # the art-style label is a placeholder text (core.placeholder_text 'style_reference'), never a word in code
    assert core.reference_list_text([('A', 'a1'), (None, 's1')], 'Image {n}: {name}', 'art style') == \
        "Image 1: A\nImage 2: art style"
    assert core.reference_list_text([(None, 's1')], 'Image {n}: {name}', 'style') == "Image 1: style"
    assert core.reference_list_text([(None, 's1')], 'Image {n}: {name}') == "Image 1: "


def test_reference_capability_table():
    assert core.reference_limit('dall-e-3') == 0
    assert core.reference_limit('openai/dall-e-2') == 1
    assert core.reference_limit('gpt-image-1') == 1
    assert core.reference_limit('imagen-4.0') == 0
    assert core.reference_limit('gemini-3-pro-image-preview') == 14
    assert core.reference_limit('nano-banana') == 14
    assert core.reference_limit('nan/some-model') is None
    assert core.reference_limit('unknown-model') is None
    assert core.effective_reference_count('dall-e-3', 5) == 0
    assert core.effective_reference_count('gpt-image-1', 5) == 1
    assert core.effective_reference_count('unknown', 5) == 5


def test_reference_store_copies_images(tmp_path):
    source = tmp_path / 'pic.png'
    source.write_bytes(b'png')
    out = tmp_path / 'book'
    out.mkdir()
    refs = core.empty_references()
    rel = core.add_reference_image(str(out), refs, str(source), 'Seo Ha-eun')
    style = core.add_reference_image(str(out), refs, None, None, data=b'jpg', ext='.jpg')
    assert rel == 'references/seo_ha-eun_1.png' and style == 'references/style_1.jpg'
    core.save_references(str(out), refs)
    assert core.load_references(str(out)) == {'version': 1, 'characters': {'Seo Ha-eun': [rel]}, 'style': [style]}
    assert core.remove_reference_image(str(out), refs, rel)
    assert not (out / core.INTERACTIVE_FOLDER / rel).exists()


# ---- chunking / placeholders ----------------------------------------------------------------------

def test_chunks_are_block_aligned_with_global_numbers():
    blocks = [core.Block(n, 'p', 'word ' * 10, 0, 0, 0, 0) for n in range(1, 11)]
    count = lambda text: len(text.split())  # noqa: E731
    chunks = core.chunk_blocks(blocks, 40, count, overhead_tokens=5)
    assert [[b.n for b in c] for c in chunks] == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
    assert core.chunk_blocks(blocks, 0, count) == [blocks]
    assert core.chunk_image_share(2, 3, 10) == 1
    assert core.chunk_image_share(5, 3, 10) == 2
    merged = core.merge_plans([core.Plan(images=[core.PlannedImage(9, '', [], 'b')]),
                               core.Plan(images=[core.PlannedImage(2, '', [], 'a'), core.PlannedImage(5, '', [], 'c')])],
                              2)
    assert [i.prompt for i in merged.images] == ['a', 'c']


def test_placeholders_fill_by_name_only():
    import prompt_defaults

    template = prompt_defaults.DEFAULT_INTERACTIVE_IMAGES_PLANNER_PROMPT
    values = {'images_per_chapter': 2, 'character_names': 'A\nB', 'css_classes': 'glx-dialogue',
              'chapter_title': 'Ch 1', 'paragraphs': '[1] x'}
    filled = core.fill_placeholders(template, values)
    expected = template
    for name, value in values.items():
        expected = expected.replace('{' + name + '}', str(value))
    assert filled == expected
    assert '{"images":[{"after_paragraph":n,' in filled  # JSON braces stay literal
    assert core.fill_placeholders('{a} {b} {"c": 1}', {'a': 'x'}) == 'x {b} {"c": 1}'


# ---- settings / profiles --------------------------------------------------------------------------

def test_settings_defaults():
    settings = core.settings_from_config({})
    assert settings.per_chapter == 1 and settings.max_references == 5
    assert settings.text_model == '' and settings.reference_line == 'Image {n}: {name}'
    assert not settings.auto_after_translation and settings.recompile_after
    assert core.parse_per_chapter('99') == 10 and core.parse_per_chapter('x') == 1 and core.parse_per_chapter('-3') == 0


def test_profiles_seed_builtins_write_runtime_keys_and_reset():
    owner = SimpleNamespace(config={})
    saves = []
    profiles = core.InteractiveImagesProfiles(owner, persist=lambda: saves.append(1) or True)
    assert profiles.names == ['Default', 'Illustrated', 'Styles only']
    assert owner.config['interactive_images_per_chapter'] == '1'
    profiles.select('Styles only')
    assert owner.config['interactive_images_per_chapter'] == '0'
    assert owner.config['active_interactive_images_prompt_profile'] == 'Styles only'
    texts = dict(profiles.texts, planner='My planner', per_chapter='42')
    assert profiles.save('Styles only', texts) is None
    assert owner.config['interactive_images_planner_prompt'] == 'My planner'
    assert owner.config['interactive_images_per_chapter'] == '10'  # clamped
    assert profiles.reset('Styles only') is None
    import prompt_defaults
    assert owner.config['interactive_images_planner_prompt'] == prompt_defaults.DEFAULT_INTERACTIVE_IMAGES_PLANNER_PROMPT
    assert core.settings_from_config(owner.config).per_chapter == 0
    assert profiles.delete('Illustrated')[0] == "Built-in Profile"
    assert saves


# ---------------------------------------------------------------------------------------------
# Shared helpers the desktop dialog and the mobile screens both use
# ---------------------------------------------------------------------------------------------

def test_reference_note_and_settings_values():
    assert core.reference_note('dall-e-3') == ('warning', core.NO_REFERENCE_WARNING)
    assert core.reference_note('gpt-image-1') == (
        'info', "This image model takes up to 1 reference image; more are not sent")
    assert core.reference_note('nano-banana')[1].startswith("This image model takes up to 14 reference images")
    assert core.reference_note('') == ('', '') and core.reference_note('unknown-model') == ('', '')
    values = core.settings_values({})
    assert values == {key: core.DEFAULTS[key] for key in values}
    assert core.coerce_setting('interactive_images_max_references', '99') == core.MAX_REFERENCES
    assert core.coerce_setting('interactive_images_max_references', 'x') == 5
    assert core.coerce_setting('interactive_images_resolution', '8K') == '1K'
    assert core.coerce_setting('interactive_images_aspect_ratio', '16:9') == '16:9'
    assert core.coerce_setting('interactive_css_enabled', 'true') is True
    assert core.PER_CHAPTER_CHOICES == tuple(str(n) for n in range(0, 11))


def test_settings_schema_choices_match_the_core():
    import settings_schema as schema

    assert schema.CHOICES_OVERRIDES['interactive_images_resolution'] == core.RESOLUTIONS
    assert schema.CHOICES_OVERRIDES['interactive_images_aspect_ratio'] == core.ASPECT_RATIOS


def test_reference_helpers(tmp_path):
    from io import BytesIO
    from PIL import Image

    good = tmp_path / 'a.png'
    buffer = BytesIO()
    Image.new('RGB', (4, 4), (1, 2, 3)).save(buffer, format='PNG')
    good.write_bytes(buffer.getvalue())
    bad = tmp_path / 'b.png'
    bad.write_bytes(b'not an image')
    assert core.is_reference_image(str(good)) and not core.is_reference_image(str(bad))
    assert not core.is_reference_image(str(tmp_path / 'c.gif'))
    refs = core.empty_references()
    assert core.add_reference_names(refs, ['  Seo  Ha-eun ', 'baek', 'baek', '']) == 2
    assert [name for name, _paths in core.reference_rows(refs)] == ['baek', 'Seo Ha-eun']
    rel = core.add_reference_image(str(tmp_path), refs, str(good), 'baek')
    assert (tmp_path / 'Interactive_Images' / rel).is_file()
    core.clear_reference_character(str(tmp_path), refs, 'baek')
    assert 'baek' not in refs['characters'] and not (tmp_path / 'Interactive_Images' / rel).exists()