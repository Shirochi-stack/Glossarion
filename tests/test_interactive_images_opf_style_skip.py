"""Interactive Images: OPF reading order and unique image names, the art style, the {glossary_entries}
placeholder, the old-default prompt migration, and the glossary's chapter skip rules (GUI-free)."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import interactive_images_core as core  # noqa: E402
import interactive_images_progress_core as ipc  # noqa: E402
import interactive_images_runner as runner  # noqa: E402
import prompt_defaults  # noqa: E402
from test_interactive_images_round2 import PLAN, _factory  # noqa: E402

NORMAL = ('<html><head><title>T</title></head><body>\n<h1>Chapter</h1>\n'
          '<p>Mira walked to the lantern gate.</p>\n<p>"Hello," said Tomas.</p>\n</body></html>')
IMAGE_ONLY = '<html><head><title>T</title></head><body><img src="a.png"/></body></html>'
TITLE_ONLY = '<html><head><title>T</title></head><body><h1>Chapter 3</h1></body></html>'
EMPTY = '<html><head></head><body></body></html>'


def _opf(names):
    items = ''.join(f'<item id="i{n}" href="{name}" media-type="application/xhtml+xml"/>' for n, name in enumerate(names))
    refs = ''.join(f'<itemref idref="i{n}"/>' for n in range(len(names)))
    return (f'<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0">'
            f'<manifest>{items}</manifest><spine>{refs}</spine></package>')


def _book(tmp_path, chapters, spine=None):
    """``chapters``: [(output file, source file, html, actual_num)] in progress-file order."""
    out = tmp_path / 'book'
    out.mkdir(parents=True)
    progress = {}
    for index, (output, source, html, number) in enumerate(chapters):
        (out / output).write_text(html, encoding='utf-8')
        progress[str(index)] = {'actual_num': number, 'output_file': output, 'status': 'completed',
                                'original_basename': source}
    (out / 'translation_progress.json').write_text(json.dumps({'chapters': progress}), encoding='utf-8')
    if spine:
        (out / 'content.opf').write_text(_opf(spine), encoding='utf-8')
    return out


# ---------------------------------------------------------------------------------------------
# A. OPF order and unique names
# ---------------------------------------------------------------------------------------------

NOTICE_BOOK = [
    ('response_chapter_notice0001.html', 'notice_a.xhtml', NORMAL, 0),
    ('response_002_chap2.html', 'chap2.xhtml', NORMAL, 2),
    ('response_chapter_notice0002.html', 'notice_b.xhtml', NORMAL, 0),
    ('response_001_chap1.html', 'chap1.xhtml', NORMAL, 1),
]
NOTICE_SPINE = ['chap1.xhtml', 'notice_b.xhtml', 'notice_a.xhtml', 'chap2.xhtml']


def test_chapters_follow_the_opf_spine_and_get_unique_positions(tmp_path):
    book = _book(tmp_path, NOTICE_BOOK, NOTICE_SPINE)
    chapters = ipc.translated_chapters(str(book))
    assert [c.key for c in chapters] == ['response_001_chap1.html', 'response_chapter_notice0002.html',
                                         'response_chapter_notice0001.html', 'response_002_chap2.html']
    assert [c.position for c in chapters] == [1, 2, 3, 4]
    assert [c.number for c in chapters] == [1, 0, 0, 2]            # the real chapter number stays (gender tracker)
    names = {core.image_filename(c.position, 1) for c in chapters}
    assert len(names) == 4
    rows = ipc.image_rows(ipc.open_progress(str(book)))
    assert [row.key for row in rows] == [c.key for c in chapters]
    assert rows[1].display.count('[002]') == 1                        # the Translation tab's [003] style


def test_two_notice_chapters_with_number_zero_do_not_share_an_image_file(tmp_path):
    book = _book(tmp_path, NOTICE_BOOK, NOTICE_SPINE)
    result = runner.run_interactive_images({'model': 'gpt-test', 'interactive_images_image_model': 'gemini-3-pro-image-preview'},
                                           str(book), client_factory=_factory([], PLAN))
    assert result.completed == 4
    files = sorted(os.listdir(book / core.INTERACTIVE_FOLDER))
    assert [f for f in files if f.endswith('.png')] == [f'Glossarion_ch00{n}_1.png' for n in (1, 2, 3, 4)]
    data = ipc.read_progress_file(core.progress_path(str(book)))
    assert data['chapters']['response_chapter_notice0001.html']['chapter'] == 3
    assert data['chapters']['response_chapter_notice0002.html']['chapter'] == 2


def test_a_book_without_an_opf_still_gets_unique_positions(tmp_path):
    book = _book(tmp_path, [('response_chapter_notice0001.html', 'a.html', NORMAL, 0),
                            ('response_chapter_notice0002.html', 'b.html', NORMAL, 0)])
    chapters = ipc.translated_chapters(str(book))
    assert len({c.position for c in chapters}) == 2 and all(c.position >= 1 for c in chapters)
    numbered = _book(tmp_path / 'n', [('response_003_c.html', 'c.html', NORMAL, 3),
                                      ('response_001_a.html', 'a.html', NORMAL, 1)])
    assert [(c.key, c.position) for c in ipc.translated_chapters(str(numbered))] == [
        ('response_001_a.html', 1), ('response_003_c.html', 3)]       # unchanged names where numbers are distinct


# ---------------------------------------------------------------------------------------------
# B. Art style
# ---------------------------------------------------------------------------------------------

def test_art_style_default_and_the_default_image_prompt_text():
    assert core.DEFAULTS['interactive_images_art_style'] == 'Anime illustration, clean line art, cel shading'
    assert len(core.DEFAULTS['interactive_images_art_style']) == 47
    prompt = prompt_defaults.DEFAULT_INTERACTIVE_IMAGES_IMAGE_PROMPT
    assert prompt == 'Art style: {art_style}\n{scene_prompt}\nCharacters: {characters}\n{reference_list}'
    assert len(prompt) == 79
    assert 'art_style' in core.IMAGE_PLACEHOLDERS and 'glossary_entries' in core.IMAGE_PLACEHOLDERS
    assert core.describe_placeholder('art_style', {}, field='image')['text'] == core.DEFAULTS['interactive_images_art_style']
    assert core.describe_placeholder('glossary_entries', {}, field='image')['per_request']


def test_art_style_fills_the_prompt_and_an_empty_style_drops_its_line():
    settings = core.settings_from_config({})
    values = {'scene_prompt': 'A gate', 'characters': 'Mira', 'art_style': settings.art_style,
              'glossary_entries': '', 'reference_list': ''}
    text = core.fill_placeholders(settings.image_prompt, values, core.LINE_DROP_PLACEHOLDERS)
    assert text == 'Art style: Anime illustration, clean line art, cel shading\nA gate\nCharacters: Mira\n'
    cleared = core.settings_from_config({'interactive_images_art_style': ''})
    assert cleared.art_style == ''
    values['art_style'] = ''
    assert core.fill_placeholders(cleared.image_prompt, values, core.LINE_DROP_PLACEHOLDERS) == 'A gate\nCharacters: Mira\n'


def test_art_style_is_a_profile_field_with_the_default_for_old_profiles():
    owner = SimpleNamespace(config={core.InteractiveImagesProfiles.PROFILES_KEY: {
        'Mine': {'planner': 'p', 'image_system': '', 'image': 'my image {scene_prompt}', 'css': 'c', 'per_chapter': '1'}}})
    profiles = core.InteractiveImagesProfiles(owner, persist=lambda: True)
    assert profiles.profile_texts('Mine')['art_style'] == 'Anime illustration, clean line art, cel shading'
    profiles.select('Mine')
    assert owner.config['interactive_images_art_style'] == 'Anime illustration, clean line art, cel shading'
    assert profiles.save('Mine', dict(profiles.texts, art_style='Watercolor')) is None
    assert owner.config['interactive_images_art_style'] == 'Watercolor'
    assert core.settings_from_config(owner.config).art_style == 'Watercolor'


def test_a_saved_old_default_image_prompt_becomes_the_new_default_but_an_edited_one_stays():
    old = core.OLD_DEFAULT_IMAGE_PROMPT
    assert old == '{scene_prompt}\nCharacters: {characters}\n{reference_list}' and len(old) == 56
    new = prompt_defaults.DEFAULT_INTERACTIVE_IMAGES_IMAGE_PROMPT
    assert core.settings_from_config({'interactive_images_image_prompt': old}).image_prompt == new
    edited = old + '\nNo text.'
    assert core.settings_from_config({'interactive_images_image_prompt': edited}).image_prompt == edited
    owner = SimpleNamespace(config={core.InteractiveImagesProfiles.PROFILES_KEY: {
        'Old': {'planner': 'p', 'image_system': '', 'image': old, 'css': 'c', 'per_chapter': '1'},
        'Edited': {'planner': 'p', 'image_system': '', 'image': edited, 'css': 'c', 'per_chapter': '1'}},
        core.InteractiveImagesProfiles.DEFAULT_KEY: {'planner': 'p', 'image_system': '', 'image': old, 'css': 'c',
                                                      'per_chapter': '1'}})
    profiles = core.InteractiveImagesProfiles(owner, persist=lambda: True)
    assert profiles.profile_texts('Old')['image'] == new
    assert profiles.profile_texts('Edited')['image'] == edited
    assert profiles.profile_texts('Default')['image'] == new


# ---------------------------------------------------------------------------------------------
# C. {glossary_entries}
# ---------------------------------------------------------------------------------------------

ENTRIES = [
    {'type': 'character', 'raw_name': '미라', 'translated_name': 'Mira', 'description': 'a young mage'},
    {'type': 'character', 'raw_name': '토마스', 'translated_name': 'Tomas', 'description': ''},
    {'type': 'places', 'raw_name': '등대', 'translated_name': 'Lantern Gate', 'description': 'the old city gate'},
    {'type': 'terms', 'raw_name': '검', 'translated_name': 'Moon Blade', 'description': 'a silver sword'},
    {'type': 'character', 'raw_name': '미라2', 'translated_name': 'Mira', 'description': 'duplicate'},
    {'type': 'character', 'raw_name': 'Hana', 'translated_name': '', 'description': 'raw only'},
]


def test_glossary_entries_are_the_scenes_characters_and_terms_named_in_the_prompt():
    settings = core.settings_from_config({})
    text = core.relevant_glossary_entries_text(ENTRIES, ['Mira', 'Tomas'], 'A dusk at the lantern gate with a Moon Blade.',
                                               settings)
    assert text == 'Mira: a young mage\nTomas\nLantern Gate: the old city gate\nMoon Blade: a silver sword'
    # whole words only, case-insensitive; a raw name of a scene character counts
    assert core.relevant_glossary_entries_text(ENTRIES, ['Hana'], 'The MOON BLADES glow.', settings) == 'Hana: raw only'
    assert core.relevant_glossary_entries_text(ENTRIES, [], 'nothing here', settings) == ''


def test_the_glossary_entry_line_text_is_editable_and_an_empty_text_sends_nothing():
    assert core.placeholder_default('glossary_entry_line') == '{name}: {description}'
    config = {}
    core.set_placeholder_text(config, 'glossary_entry_line', '- {name} ({description})')
    settings = core.settings_from_config(config)
    assert core.relevant_glossary_entries_text(ENTRIES, ['Mira', 'Tomas'], '', settings) == '- Mira (a young mage)\nTomas'
    core.set_placeholder_text(config, 'glossary_entry_line', '')
    assert core.relevant_glossary_entries_text(ENTRIES, ['Mira'], '', core.settings_from_config(config)) == ''


def test_glossary_entries_line_is_dropped_when_empty_and_reaches_the_image_request(tmp_path):
    prompt = 'Art style: {art_style}\n{scene_prompt}\nGlossary:\n{glossary_entries}\nCharacters: {characters}'
    values = {'scene_prompt': 'A gate', 'characters': 'Mira', 'art_style': 'X', 'glossary_entries': ''}
    assert core.fill_placeholders(prompt, values, core.LINE_DROP_PLACEHOLDERS) == 'Art style: X\nA gate\nGlossary:\nCharacters: Mira'
    book = _book(tmp_path, [('response_001_ch1.html', 'ch1.xhtml', NORMAL, 1)], ['ch1.xhtml'])
    (book / 'glossary.csv').write_text(
        'type,raw_name,translated_name,gender,description\n'
        'character,미라,Mira,Female,a young mage\ncharacter,토마스,Tomas,Male,\n', encoding='utf-8')
    sent = []
    config = {'model': 'gpt-test', 'interactive_images_image_model': 'gemini-3-pro-image-preview',
              'interactive_images_image_prompt': '{scene_prompt}\n{glossary_entries}'}
    runner.run_interactive_images(config, str(book), client_factory=_factory(sent, PLAN))
    images = [r['messages'][-1]['content'][0]['text'] for r in sent if r['kind'] == 'image']
    assert images[0] == 'A lantern gate at dusk\nMira: a young mage\nTomas'


# ---------------------------------------------------------------------------------------------
# E. The glossary's skip rules
# ---------------------------------------------------------------------------------------------

SKIP_SPINE = ['titlepage.xhtml', 'chap1.xhtml', 'chap2.xhtml', 'chap3.xhtml', 'chap4.xhtml', 'chap5.xhtml',
              'copyright.xhtml']
SKIP_BOOK = [
    ('response_titlepage.html', 'titlepage.xhtml', NORMAL, 0),
    ('response_001_chap1.html', 'chap1.xhtml', NORMAL, 1),
    ('response_002_chap2.html', 'chap2.xhtml', IMAGE_ONLY, 2),
    ('response_003_chap3.html', 'chap3.xhtml', TITLE_ONLY, 3),
    ('response_004_chap4.html', 'chap4.xhtml', EMPTY, 4),
    ('response_005_chap5.html', 'chap5.xhtml', NORMAL, 5),
    ('response_copyright.html', 'copyright.xhtml', NORMAL, 0),
]


def _listed(book, config=None, owner=None):
    model = ipc.open_progress(str(book), ipc.glossary_skip_rules(config or {}, owner))
    return [row.key for row in ipc.image_rows(model)], {c.key: c.skip_reason for c in model.chapters if c.skip_reason}


def test_image_only_empty_and_special_chapters_are_skipped_like_the_glossary(tmp_path):
    book = _book(tmp_path, SKIP_BOOK, SKIP_SPINE)
    listed, reasons = _listed(book)
    assert listed == ['response_001_chap1.html', 'response_005_chap5.html']
    assert reasons == {'response_titlepage.html': 'special file', 'response_002_chap2.html': 'image-only',
                       'response_003_chap3.html': 'title/header-only', 'response_004_chap4.html': 'empty',
                       'response_copyright.html': 'special file'}
    stats = ipc.image_stats(ipc.open_progress(str(book)))
    assert stats['total'] == 2 and stats['not_built'] == 2


def test_title_header_only_is_skipped_only_while_the_glossary_setting_is_on(tmp_path):
    book = _book(tmp_path, SKIP_BOOK, SKIP_SPINE)
    on, _ = _listed(book, {'glossary_skip_title_header_only': True})
    off, reasons = _listed(book, {'glossary_skip_title_header_only': False})
    assert 'response_003_chap3.html' not in on
    assert 'response_003_chap3.html' in off and 'response_003_chap3.html' not in reasons
    # image-only and empty chapters are always skipped
    assert 'response_002_chap2.html' not in off and 'response_004_chap4.html' not in off


def test_special_files_are_built_when_the_glossary_includes_them(tmp_path):
    book = _book(tmp_path, SKIP_BOOK, SKIP_SPINE)
    listed, _ = _listed(book, {'translate_special_files': True})
    assert listed == ['response_titlepage.html', 'response_001_chap1.html', 'response_005_chap5.html',
                      'response_copyright.html']


def test_the_skip_rules_read_the_owner_the_way_the_glossary_run_does():
    owner = SimpleNamespace(config={}, translate_special_files_var=True, special_file_keywords_var='',
                            special_file_exact_var='', _glossary_skip_title_header_only_env_value=lambda: '0')
    rules = ipc.glossary_skip_rules({}, owner)
    assert rules['skip_title_header_only'] is False and rules['special']['translate_special_files'] is True


def test_the_runner_never_plans_a_chapter_the_glossary_skips(tmp_path):
    book = _book(tmp_path, SKIP_BOOK, SKIP_SPINE)
    sent, lines = [], []
    config = {'model': 'gpt-test', 'interactive_images_image_model': 'gemini-3-pro-image-preview'}
    result = runner.run_interactive_images(config, str(book), log=lines.append, client_factory=_factory(sent, PLAN))
    planner = [r for r in sent if r['kind'] == 'text']
    assert len(planner) == 2 and result.completed == 2
    data = ipc.read_progress_file(core.progress_path(str(book)))
    assert sorted(data['chapters']) == ['response_001_chap1.html', 'response_005_chap5.html']
    log = '\n'.join(lines)
    assert 'response_002_chap2.html skipped: image-only' in log and 'skipped: special file' in log
    assert 'skipped: empty' in log and 'skipped: title/header-only' in log
    # asked for by name, a skipped chapter is still not planned
    sent.clear()
    runner.run_interactive_images(config, str(book), ['response_002_chap2.html'], log=lines.append,
                                  client_factory=_factory(sent, PLAN))
    assert not [r for r in sent if r['kind'] == 'text']
