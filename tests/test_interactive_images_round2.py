"""Interactive Images, round 2: gender (confident only), image subjects, Auto aspect ratio, the editable placeholder
texts (and a test that no request holds text that is not a template or a placeholder text), and the gallery model
with its multi-delete (GUI-free)."""
import json
import os
import re
import sys
import threading
from io import BytesIO
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import gender_tracking as gt  # noqa: E402
import interactive_css_presets as presets  # noqa: E402
import interactive_images_core as core  # noqa: E402
import interactive_images_progress_core as ipc  # noqa: E402
import interactive_images_runner as runner  # noqa: E402
import prompt_defaults  # noqa: E402


def _png(color=(200, 20, 20)):
    from PIL import Image
    buffer = BytesIO()
    Image.new('RGB', (8, 8), color).save(buffer, format='PNG')
    return buffer.getvalue()


# ---------------------------------------------------------------------------------------------
# Gender: the confidence rule
# ---------------------------------------------------------------------------------------------

def _tracker_entry(*occurrences, decision=None):
    """A gender-tracker entry from ``(gender, chapter_num)`` pairs."""
    entry = {'occurrences': [{'gender': g, 'chapter_num': n, 'chapter_file': f'ch{n}.xhtml'} for g, n in occurrences]}
    if decision:
        entry['decision'] = decision
    return entry


@pytest.mark.parametrize("glossary,trackers,expected", [
    (['Female'], [], 'female'),                                   # the glossary alone
    (['male'], [], 'male'),
    (['f'], [], 'female'),                                        # aliases resolve
    (['Female'], [_tracker_entry(('female', 1), ('Female', 2))], 'female'),     # both agree
    ([''], [_tracker_entry(('female', 1))], 'female'),            # the tracker alone, nothing against it
    (['Unknown'], [_tracker_entry(('male', 4))], 'male'),
    (['Female'], [_tracker_entry(('male', 1), ('male', 2))], ''),               # glossary vs tracker disagree
    (['Female'], [_tracker_entry(('female', 1), ('male', 2))], ''),             # the tracker flip-flops
    ([''], [_tracker_entry(('female', 1), ('male', 2), ('female', 3))], ''),
    (['Female'], [_tracker_entry(('female', 1), ('male', 2), decision='female')], 'female'),   # the user's decision wins
    (['Female'], [_tracker_entry(('female', 1), decision='male')], ''),         # ... even against the glossary
    (['Neutral'], [], ''),                                        # never sent
    (['Nonbinary'], [_tracker_entry(('female', 1))], ''),
    (['Unknown'], [], ''), ([''], [], ''), ([None], [], ''), ([], [], ''),
    (['Female', 'Male'], [], ''),                                 # two rows of one name disagree
    (['Female', 'female'], [], 'female'),
    ([''], [_tracker_entry(('female', 1), ('Non-binary', 2))], ''),             # a binary mixed with another value
    ([''], [_tracker_entry(('female', 1)), _tracker_entry(('male', 1))], ''),   # two entries disagree
])
def test_confident_gender_rule(glossary, trackers, expected):
    assert core.confident_gender(glossary, trackers) == expected


def test_a_dated_gender_change_counts_only_when_unambiguous():
    entry = _tracker_entry(('male', 1), ('male', 2), ('female', 10), ('female', 11))
    assert core.confident_gender([''], [entry], chapter_number=1) == 'male'
    assert core.confident_gender([''], [entry], chapter_number=2) == 'male'
    assert core.confident_gender([''], [entry], chapter_number=10) == 'female'
    assert core.confident_gender([''], [entry], chapter_number=99) == 'female'
    assert core.confident_gender([''], [entry], chapter_number=5) == ''          # in the gap: unknown
    assert core.confident_gender([''], [entry]) == ''                            # no chapter: unknown
    assert core.confident_gender(['Male'], [entry], chapter_number=1) == 'male'  # the glossary agrees with that side
    assert core.confident_gender(['Female'], [entry], chapter_number=1) == ''    # ... and disagrees with this one
    lone = _tracker_entry(('female', 1), ('female', 2), ('female', 3), ('male', 9))   # one outlier is not a change
    assert core.confident_gender([''], [lone], chapter_number=20) == ''
    twice = _tracker_entry(('male', 1), ('male', 2), ('female', 5), ('female', 6), ('male', 8), ('male', 9))
    assert core.confident_gender([''], [twice], chapter_number=1) == ''          # two changes: ambiguous


def test_character_genders_match_by_exact_raw_name_never_a_similar_name():
    entries = [
        {'type': 'character', 'raw_name': '미라', 'translated_name': 'Mira', 'gender': 'Female'},
        {'type': 'character', 'raw_name': '미라야', 'translated_name': 'Mirae', 'gender': ''},
        {'type': 'character', 'raw_name': '토마스', 'translated_name': 'Tomas', 'gender': ''},
        {'type': 'character', 'raw_name': '하나', 'translated_name': 'Hana', 'gender': 'Female'},
        {'type': 'place', 'raw_name': '성', 'translated_name': 'Castle', 'gender': 'Female'},
    ]
    tracker = {'entries': {
        '미라': _tracker_entry(('female', 1)),
        '토마스': _tracker_entry(('male', 1)),
        '하나': _tracker_entry(('male', 1)),       # disagrees with the glossary: no gender
        '미라야': None,
    }}
    assert core.character_genders(entries, tracker) == {'Mira': 'female', 'Tomas': 'male'}
    # Mirae has no entry of its own: Mira's tracker data is not borrowed for a similar name
    assert 'Mirae' not in core.character_genders(entries, tracker)
    assert 'Castle' not in core.character_genders(entries, tracker)       # not a character
    # two rows with one display name must agree
    twin = [{'type': 'character', 'raw_name': 'A', 'translated_name': 'Kim', 'gender': 'Male'},
            {'type': 'character', 'raw_name': 'B', 'translated_name': 'Kim', 'gender': 'Female'}]
    assert core.character_genders(twin, {}) == {}


def test_gender_is_written_only_through_the_placeholder_texts():
    settings = core.settings_from_config({})
    assert settings.include_gender is True
    assert core.format_character('Mira', 'female', settings) == 'Mira (female)'
    assert core.format_character('Mira', '', settings) == 'Mira'
    custom = core.settings_from_config({core.PLACEHOLDER_TEXTS_KEY: {
        'character_gender': '{name} [{gender}]', 'gender_female': 'she/her'}})
    assert core.format_character('Mira', 'female', custom) == 'Mira [she/her]'
    assert core.format_character('Tomas', 'male', custom) == 'Tomas [male]'
    empty = core.settings_from_config({core.PLACEHOLDER_TEXTS_KEY: {'character_gender': ''}})
    assert core.format_character('Mira', 'female', empty) == 'Mira'
    off = core.settings_from_config({'interactive_images_include_gender': False})
    assert core.format_character('Mira', 'female', off) == 'Mira'


# ---------------------------------------------------------------------------------------------
# Subjects, aspect ratio, settings
# ---------------------------------------------------------------------------------------------

def test_subject_texts_are_editable_and_the_default_adds_nothing():
    default = core.settings_from_config({})
    assert default.subjects == 'characters' and core.subjects_text(default) == ''
    assert core.subjects_text(core.settings_from_config({'interactive_images_subjects': 'backgrounds'})) == \
        core.placeholder_default('image_subjects_backgrounds') != ''
    assert core.subjects_text(core.settings_from_config({'interactive_images_subjects': 'both'})) == \
        core.placeholder_default('image_subjects_both')
    edited = core.settings_from_config({'interactive_images_subjects': 'both',
                                        core.PLACEHOLDER_TEXTS_KEY: {'image_subjects_both': 'Scenery too.'}})
    assert core.subjects_text(edited) == 'Scenery too.'
    assert core.coerce_setting('interactive_images_subjects', 'nonsense') == 'characters'
    assert '{image_subjects}' in prompt_defaults.DEFAULT_INTERACTIVE_IMAGES_PLANNER_PROMPT
    assert 'image_subjects' in core.PLANNER_PLACEHOLDERS


def test_the_default_planner_request_reads_as_before_the_image_subjects_line():
    settings = core.settings_from_config({})
    blocks = [core.Block(1, 'p', 'Hello there.', 0, 0, 0, 0)]
    (prompt, _chunk, _share), = runner.build_planner_prompts(settings, blocks, 'Ch', ['Mira'], ['glx-dialogue'])
    old = prompt_defaults.DEFAULT_INTERACTIVE_IMAGES_PLANNER_PROMPT.replace('{image_subjects}\n', '')
    expected = core.fill_placeholders(old, {'images_per_chapter': 1, 'character_names': 'Mira',
                                            'css_classes': 'glx-dialogue', 'chapter_title': 'Ch',
                                            'paragraphs': '[1] Hello there.'})
    assert prompt == expected
    both = core.settings_from_config({'interactive_images_subjects': 'both'})
    (prompt, _c, _s), = runner.build_planner_prompts(both, blocks, 'Ch', ['Mira'], [])
    assert core.placeholder_default('image_subjects_both') in prompt


def test_fill_placeholders_drops_the_line_of_an_empty_listed_placeholder():
    template = 'a\nCharacters: {characters}\nb {x}'
    assert core.fill_placeholders(template, {'characters': '', 'x': '1'}, ('characters',)) == 'a\nb 1'
    assert core.fill_placeholders(template, {'characters': 'A', 'x': '1'}, ('characters',)) == 'a\nCharacters: A\nb 1'
    assert core.fill_placeholders(template, {'characters': '', 'x': '1'}) == 'a\nCharacters: \nb 1'


def test_aspect_ratio_auto_picks_from_the_ticked_ratios_and_is_stable():
    fixed = core.settings_from_config({'interactive_images_aspect_ratio': '16:9'})
    assert core.pick_aspect_ratio(fixed, 3, 1) == '16:9'
    auto = core.settings_from_config({'interactive_images_aspect_ratio': 'auto',
                                      'interactive_images_aspect_choices': ['1:1', '21:9']})
    picks = {core.pick_aspect_ratio(auto, chapter, k) for chapter in range(1, 40) for k in range(1, 4)}
    assert picks == {'1:1', '21:9'}                                   # only ticked ones, both get used
    assert [core.pick_aspect_ratio(auto, 7, k) for k in (1, 2, 3)] == [core.pick_aspect_ratio(auto, 7, k)
                                                                       for k in (1, 2, 3)]   # stable per chapter+index
    none = core.settings_from_config({'interactive_images_aspect_ratio': 'auto',
                                      'interactive_images_aspect_choices': []})
    assert core.pick_aspect_ratio(none, 1, 1) == ''                    # nothing ticked: the model decides
    default = core.settings_from_config({'interactive_images_aspect_ratio': 'auto'})
    assert set(default.aspect_choices) == {'3:4', '4:3', '16:9', '1:1'}
    assert core.ASPECT_RATIOS[0] == 'auto' and '21:9' in core.ASPECT_RATIOS
    assert core.coerce_setting('interactive_images_aspect_choices', ['16:9', 'bogus', '1:1']) == ['1:1', '16:9']
    assert core.coerce_setting('interactive_images_aspect_choices', 'nonsense') == []


def test_placeholder_text_store_round_trip():
    config = {}
    assert core.placeholder_text(config, 'gender_male') == 'male'
    assert core.set_placeholder_text(config, 'gender_male', 'he') is True
    assert config[core.PLACEHOLDER_TEXTS_KEY] == {'gender_male': 'he'}
    assert core.placeholder_text(config, 'gender_male') == 'he'
    assert core.set_placeholder_text(config, 'gender_male', 'male') is True        # back to the default: override gone
    assert config[core.PLACEHOLDER_TEXTS_KEY] == {}
    core.set_placeholder_text(config, 'reference_line', 'Ref {n} = {name}')
    assert config['interactive_images_reference_line'] == 'Ref {n} = {name}'
    assert core.placeholder_text(config, 'reference_line') == 'Ref {n} = {name}'
    assert core.stored_placeholder_texts({core.PLACEHOLDER_TEXTS_KEY: {'bogus': 'x', 'gender_male': 3}}) == {}
    assert [item['name'] for item in core.placeholder_text_defs()] == [
        'character_gender', 'gender_male', 'gender_female', 'image_subjects_characters', 'image_subjects_backgrounds',
        'image_subjects_both', 'style_reference', 'reference_line']


# ---------------------------------------------------------------------------------------------
# The runner: gender, subjects and the aspect ratio reach the requests
# ---------------------------------------------------------------------------------------------

CH1 = ('<html><head><title>Ch 1</title></head><body>\n<h1>Chapter 1</h1>\n'
       '<p>Mira walked to the lantern gate.</p>\n<p>"Hello," said Tomas.</p>\n</body></html>')
PLAN = ('{"images":[{"after_paragraph":2,"quote":"Mira walked","characters":["Mira","Tomas"],'
        '"prompt":"A lantern gate at dusk"},{"after_paragraph":3,"quote":"\\"Hello,\\"","characters":["Mira"],'
        '"prompt":"Mira at the gate"}],"classes":[]}')


class RecordingClient:
    def __init__(self, kind, sent, plan=PLAN):
        self.kind, self.sent, self.plan = kind, sent, plan
        self.tls = threading.local()
        self._forced_image_output_aspect_ratio = 'unset'

    def _get_thread_local_client(self):
        return self.tls

    def send(self, messages, temperature=None, max_tokens=None, context=None):
        self.sent.append({'kind': self.kind, 'context': context, 'messages': messages,
                          'ratio': self._forced_image_output_aspect_ratio})
        if self.kind == 'text':
            return self.plan, 'stop'
        self.tls._last_generated_image_bytes = _png()
        return '', 'stop'


def _factory(sent, plan=PLAN):
    clients = {}

    def make(kind, model, config, output_dir):
        clients.setdefault(kind, RecordingClient(kind, sent, plan))
        return clients[kind]
    return make


@pytest.fixture
def book(tmp_path):
    out = tmp_path / 'book'
    out.mkdir()
    (out / 'response_001_ch1.html').write_text(CH1, encoding='utf-8')
    (out / 'translation_progress.json').write_text(json.dumps({'chapters': {
        'a': {'actual_num': 1, 'output_file': 'response_001_ch1.html', 'status': 'completed'}}}), encoding='utf-8')
    (out / 'glossary.csv').write_text(
        'type,raw_name,translated_name,gender\ncharacter,미라,Mira,Female\ncharacter,토마스,Tomas,Male\n'
        'character,하나,Hana,Female\n', encoding='utf-8')
    tracker = {'entries': {
        '미라': _tracker_entry(('female', 1)),
        '토마스': _tracker_entry(('male', 1), ('female', 2)),      # a conflict: no gender for Tomas
        '하나': _tracker_entry(('male', 1)),                       # disagrees with the glossary: none
    }}
    tracker_path = gt.tracker_path_for_glossary(str(out / 'glossary.csv'))
    Path(tracker_path).write_text(json.dumps(tracker), encoding='utf-8')
    return out


CONFIG = {'model': 'gpt-test', 'interactive_images_image_model': 'gemini-3-pro-image-preview'}


def test_confident_genders_reach_the_planner_and_the_image_prompt(book):
    sent = []
    result = runner.run_interactive_images(dict(CONFIG, interactive_images_per_chapter='2'), str(book),
                                           client_factory=_factory(sent))
    assert result.completed == 1
    planner = [r for r in sent if r['kind'] == 'text'][0]['messages'][-1]['content']
    assert "Characters in this book:\nMira (female)\nTomas\nHana\n" in planner
    images = [r['messages'][-1]['content'][0]['text'] for r in sent if r['kind'] == 'image']
    # (the template's own empty {reference_list} line leaves the final line break, as before)
    assert images[0] == "A lantern gate at dusk\nCharacters: Mira (female), Tomas\n"
    assert images[1] == "Mira at the gate\nCharacters: Mira (female)\n"


def test_gender_off_sends_bare_names(book):
    sent = []
    runner.run_interactive_images(dict(CONFIG, interactive_images_include_gender=False,
                                       interactive_images_per_chapter='2'), str(book), client_factory=_factory(sent))
    planner = [r for r in sent if r['kind'] == 'text'][0]['messages'][-1]['content']
    assert "Characters in this book:\nMira\nTomas\nHana\n" in planner and '(female)' not in planner
    assert [r['messages'][-1]['content'][0]['text'] for r in sent if r['kind'] == 'image'][0] == \
        "A lantern gate at dusk\nCharacters: Mira, Tomas\n"


def test_backgrounds_mode_clears_characters_and_drops_the_characters_line(book):
    sent = []
    runner.run_interactive_images(dict(CONFIG, interactive_images_subjects='backgrounds',
                                       interactive_images_per_chapter='2'), str(book), client_factory=_factory(sent))
    planner = [r for r in sent if r['kind'] == 'text'][0]['messages'][-1]['content']
    assert core.placeholder_default('image_subjects_backgrounds') in planner
    images = [r['messages'][-1]['content'] for r in sent if r['kind'] == 'image']
    assert [part[0]['text'] for part in images] == ["A lantern gate at dusk\n", "Mira at the gate\n"]
    entry = ipc.open_progress(str(book)).data['chapters']['response_001_ch1.html']
    assert all(image['characters'] == [] for image in entry['images'])


def test_auto_aspect_ratio_is_set_per_image_from_the_ticked_ratios(book):
    sent = []
    config = dict(CONFIG, interactive_images_per_chapter='2', interactive_images_aspect_ratio='auto',
                  interactive_images_aspect_choices=['1:1', '21:9'])
    runner.run_interactive_images(config, str(book), client_factory=_factory(sent))
    ratios = [r['ratio'] for r in sent if r['kind'] == 'image']
    settings = core.settings_from_config(config)
    assert ratios == [core.pick_aspect_ratio(settings, 1, 1), core.pick_aspect_ratio(settings, 1, 2)]
    assert set(ratios) <= {'1:1', '21:9'}
    entry = ipc.open_progress(str(book)).data['chapters']['response_001_ch1.html']
    assert [i['aspect_ratio'] for i in entry['images']] == ratios
    sent.clear()
    runner.run_interactive_images(dict(CONFIG, interactive_images_per_chapter='1',
                                       interactive_images_aspect_ratio='9:16'), str(book), ['response_001_ch1.html'],
                                  client_factory=_factory(sent))
    assert [r['ratio'] for r in sent if r['kind'] == 'image'] == ['9:16']


# ---------------------------------------------------------------------------------------------
# No hardcoded request text: everything is a template or a placeholder text
# ---------------------------------------------------------------------------------------------

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.U)


def _words(text):
    return {w.casefold() for w in _WORD.findall(str(text))}


def test_every_word_of_every_request_comes_from_a_template_a_placeholder_text_or_the_book(book):
    """The templates and every placeholder text are replaced by made-up words; any other word in a planner, image
    or CSS request is a hardcoded one."""
    texts = {item['name']: 'zq' + item['name'].replace('_', '') for item in core.placeholder_text_defs()}
    texts['character_gender'] = 'zqname {name} zqgender {gender}'
    texts['reference_line'] = 'zqref {n} zqwho {name}'
    config = dict(CONFIG, interactive_images_per_chapter='2', interactive_images_subjects='both',
                  interactive_images_planner_prompt='zqplan {images_per_chapter} {character_names} {image_subjects} '
                                                    '{css_classes} {chapter_title} {paragraphs}',
                  interactive_images_image_system_prompt='zqsystem',
                  interactive_images_image_prompt='zqimg {scene_prompt} | {characters} | {reference_list}',
                  interactive_images_css_prompt='zqcss {user_request} {css_classes}',
                  interactive_css_enabled=True)
    config[core.PLACEHOLDER_TEXTS_KEY] = {k: v for k, v in texts.items() if k != 'reference_line'}
    config['interactive_images_reference_line'] = texts['reference_line']
    refs = core.empty_references()
    (book / 'style.png').write_bytes(_png((0, 0, 200)))
    (book / 'mira.png').write_bytes(_png((0, 200, 0)))
    core.add_reference_image(str(book), refs, str(book / 'mira.png'), 'Mira')
    core.add_reference_image(str(book), refs, str(book / 'style.png'), None)
    core.save_references(str(book), refs)
    sent = []
    runner.run_interactive_images(config, str(book), client_factory=_factory(sent))
    css_sent = []
    runner.generate_stylesheet(config, 'a calm sea', client_factory=_factory(css_sent, plan='```css\n.glx-a {}\n```'))
    settings = core.settings_from_config(config)
    allowed = set()
    for template in (settings.planner_prompt, settings.image_system_prompt, settings.image_prompt, settings.css_prompt,
                     settings.reference_line):
        allowed |= _words(re.sub(r'\{[A-Za-z_]+\}', ' ', template))
    for text in list(texts.values()) + [config[core.PLACEHOLDER_TEXTS_KEY]['character_gender']]:
        allowed |= _words(re.sub(r'\{[A-Za-z_]+\}', ' ', text))
    # the book's own data: the chapter, the glossary names, the plan, the class names, numbers
    allowed |= _words(CH1) | _words(PLAN) | {'mira', 'tomas', 'hana', 'ch', '1', '2', 'json'}
    allowed |= _words(' '.join(presets.CLASS_VOCABULARY)) | _words('a calm sea')
    checked = 0
    for request in sent + css_sent:
        content = request['messages'][-1]['content']
        body = content if isinstance(content, str) else ' '.join(p['text'] for p in content if p['type'] == 'text')
        system = ' '.join(m['content'] for m in request['messages'] if m['role'] == 'system')
        for text in (body, system):
            unknown = _words(text) - allowed
            assert not unknown, f"hardcoded words in a {request['kind']} request: {sorted(unknown)} in {text!r}"
            checked += 1
    assert checked >= 6
    first_image = [r for r in sent if r['kind'] == 'image'][0]['messages'][-1]['content'][0]['text']
    assert 'zqstylereference' in first_image          # the art-style reference label is the placeholder text


def test_no_request_text_literals_remain_in_the_runner_and_core():
    """A guard on the two modules that build requests: the old hardcoded art-style label is gone from the code."""
    source = (SRC / 'interactive_images_runner.py').read_text(encoding='utf-8')
    core_source = (SRC / 'interactive_images_core.py').read_text(encoding='utf-8')
    assert "or 'art style'" not in source and "or 'art style'" not in core_source
    # the only place the default wording lives is the placeholder default
    assert core.placeholder_default('style_reference') == 'art style'


# ---------------------------------------------------------------------------------------------
# The gallery model
# ---------------------------------------------------------------------------------------------

CH_A = ('<html><head><title>Ch 1</title></head><body>\n<h1>Chapter 1</h1>\n'
        '<p>Mira walked to the gate.</p>\n<p>"Hello," said Tomas.</p>\n</body></html>')
CH_B = ('<html><head><title>Ch 2</title></head><body>\n<h1>Chapter 2</h1>\n<p>The rain kept falling.</p>\n</body></html>')


@pytest.fixture
def gallery_book(tmp_path):
    out = tmp_path / 'gbook'
    out.mkdir()
    (out / 'translation_progress.json').write_text(json.dumps({'chapters': {
        'a': {'actual_num': 1, 'output_file': 'response_001_a.html', 'status': 'completed', 'title': 'The Gate'},
        'b': {'actual_num': 2, 'output_file': 'response_002_b.html', 'status': 'completed'}}}), encoding='utf-8')
    folder = out / core.INTERACTIVE_FOLDER
    folder.mkdir()
    figure = lambda name: core.figure_html(core.image_src(name))  # noqa: E731
    (out / 'response_001_a.html').write_text(CH_A.replace('gate.</p>', 'gate.</p>' + figure('Glossarion_ch001_1.png')).replace(
        'Tomas.</p>', 'Tomas.</p>' + figure('Glossarion_ch001_2.png')), encoding='utf-8')
    (out / 'response_002_b.html').write_text(CH_B.replace('falling.</p>', 'falling.</p>' + figure('Glossarion_ch002_1.png')),
                                             encoding='utf-8')
    for name in ('Glossarion_ch001_1.png', 'Glossarion_ch001_2.png', 'Glossarion_ch002_1.png'):
        (folder / name).write_bytes(_png())
    progress = {'version': 1, 'chapters': {
        'response_001_a.html': {'status': 'completed', 'images': [
            {'file': 'Glossarion_ch001_1.png', 'status': 'completed', 'characters': ['mira'], 'scene': ['Mira'],
             'prompt': 'gate', 'aspect_ratio': '3:4'},
            {'file': 'Glossarion_ch001_2.png', 'status': 'completed', 'characters': ['Mira', 'Tomas'],
             'scene': ['Mira', 'Tomas'], 'prompt': 'hello'},
            {'file': '', 'status': 'failed', 'characters': ['Mira']}], 'classes': []},
        'response_002_b.html': {'status': 'completed', 'images': [
            {'file': 'Glossarion_ch002_1.png', 'status': 'completed', 'characters': [], 'prompt': 'rain'}],
            'classes': []}}}
    (folder / core.PROGRESS_FILENAME).write_text(json.dumps(progress), encoding='utf-8')
    return out


def test_gallery_lists_filters_and_names_the_characters(gallery_book):
    model = ipc.open_progress(str(gallery_book))
    items = ipc.gallery_images(model)
    assert [(i.chapter_number, i.file) for i in items] == [(1, 'Glossarion_ch001_1.png'), (1, 'Glossarion_ch001_2.png'),
                                                           (2, 'Glossarion_ch002_1.png')]
    assert [i.caption for i in items] == ['Ch. 1 · Mira', 'Ch. 1 · Mira, Tomas', 'Ch. 2']
    assert items[0].aspect_ratio == '3:4' and items[0].chapter_title == 'The Gate'
    assert [i.file for i in ipc.gallery_filter(items, character='tomas')] == ['Glossarion_ch001_2.png']
    assert [i.file for i in ipc.gallery_filter(items, chapter_key='response_002_b.html')] == ['Glossarion_ch002_1.png']
    assert ipc.gallery_filter(items, character='Mira', chapter_key='response_002_b.html') == []
    assert len(ipc.gallery_filter(items)) == 3
    assert ipc.gallery_characters(items, ['Hana', 'mira']) == ['Hana', 'mira', 'Tomas']
    assert ipc.gallery_chapters(items) == [('response_001_a.html', 'Ch. 1 The Gate'), ('response_002_b.html', 'Ch. 2')]
    os.remove(items[2].path)                                              # a file that is gone is not listed
    assert len(ipc.gallery_images(ipc.open_progress(str(gallery_book)))) == 2


def test_gallery_multi_delete_removes_files_figures_and_records_text_checked(gallery_book):
    model = ipc.open_progress(str(gallery_book))
    items = ipc.gallery_images(model)
    result = ipc.delete_gallery_images(str(gallery_book), [items[0], items[2]])
    assert (result.deleted, result.chapters, result.skipped) == (2, 2, [])
    folder = gallery_book / core.INTERACTIVE_FOLDER
    assert not (folder / 'Glossarion_ch001_1.png').exists() and not (folder / 'Glossarion_ch002_1.png').exists()
    assert (folder / 'Glossarion_ch001_2.png').exists()
    html_a = (gallery_book / 'response_001_a.html').read_text(encoding='utf-8')
    assert 'Glossarion_ch001_1.png' not in html_a and 'Glossarion_ch001_2.png' in html_a
    html_b = (gallery_book / 'response_002_b.html').read_text(encoding='utf-8')
    assert 'data-glx' not in html_b and core.check_text_unchanged(CH_B, html_b) == (True, '')
    data = ipc.read_progress_file(core.progress_path(str(gallery_book)))
    assert [i['file'] for i in data['chapters']['response_001_a.html']['images'] if i['file']] == \
        ['Glossarion_ch001_2.png']
    assert 'response_002_b.html' not in data['chapters']          # nothing left: the chapter is "not built" again
    rows = {r.key: r for r in ipc.image_rows(ipc.open_progress(str(gallery_book)))}
    assert rows['response_002_b.html'].status == 'not_built'
    assert rows['response_001_a.html'].status == 'completed'      # still has its figure, no stale flag


def test_gallery_delete_skips_a_chapter_whose_text_would_change(gallery_book, monkeypatch):
    items = ipc.gallery_images(ipc.open_progress(str(gallery_book)))
    monkeypatch.setattr(core, 'check_text_unchanged', lambda a, b: (False, 'forced'))
    result = ipc.delete_gallery_images(str(gallery_book), items[:2])
    assert result.deleted == 0 and len(result.skipped) == 2 and 'forced' in result.skipped[0][1]
    folder = gallery_book / core.INTERACTIVE_FOLDER
    assert (folder / 'Glossarion_ch001_1.png').exists()
    assert 'Glossarion_ch001_1.png' in (gallery_book / 'response_001_a.html').read_text(encoding='utf-8')
    assert ipc.delete_gallery_images(str(gallery_book), []).deleted == 0
    assert '2 image' in ipc.gallery_delete_confirmation(2)


def test_a_saved_draft_planner_prompt_is_upgraded_but_an_edited_one_is_not():
    current = prompt_defaults.DEFAULT_INTERACTIVE_IMAGES_PLANNER_PROMPT
    draft = current.replace('{image_subjects}\n', '')
    assert draft != current and core.upgrade_draft_planner(draft) == current
    assert core.upgrade_draft_planner(current) == current
    edited = draft + '\nMy extra rule.'
    assert core.upgrade_draft_planner(edited) == edited
    # the runner reads the runtime key: a config that saved the draft gets the Image subjects line
    assert core.settings_from_config({'interactive_images_planner_prompt': draft}).planner_prompt == current
    assert core.settings_from_config({'interactive_images_planner_prompt': edited}).planner_prompt == edited
    # ... and so does the profile engine (the saved Default / named profiles)
    from types import SimpleNamespace
    owner = SimpleNamespace(config={core.InteractiveImagesProfiles.DEFAULT_KEY: {
        'planner': draft, 'image_system': '', 'image': 'img', 'css': 'css', 'per_chapter': '1'}})
    profiles = core.InteractiveImagesProfiles(owner)
    assert profiles.texts['planner'] == current and owner.config['interactive_images_planner_prompt'] == current