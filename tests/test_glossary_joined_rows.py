"""Last-resort recovery of AI glossary CSV whose row breaks were omitted."""
import json

import pytest

import extract_glossary_from_epub as full
import GlossaryManager as minimal


@pytest.fixture(autouse=True)
def glossary_settings(monkeypatch):
    monkeypatch.setenv('GLOSSARY_CUSTOM_ENTRY_TYPES', json.dumps({
        'character': {'enabled': True, 'has_gender': True},
        'term': {'enabled': True, 'has_gender': False},
        'faction': {'enabled': True, 'has_gender': False},
    }))
    monkeypatch.setenv('GLOSSARY_CUSTOM_FIELDS', '[]')
    monkeypatch.setenv('GLOSSARY_CJK_SCRIPT_FILTER', '0')
    monkeypatch.setenv('GLOSSARY_ENTRY_TYPE_FILTER_MODE', 'loose')
    monkeypatch.setenv('GLOSSARY_INCLUDE_GENDER_CONTEXT', '1')
    monkeypatch.setenv('GLOSSARY_INCLUDE_DESCRIPTION', '0')
    monkeypatch.setenv('GLOSSARY_SKIP_ALL_VALIDATION', '0')


def parse(parser, text, **kwargs):
    if parser == 'balanced/full':
        return full.parse_api_response(text, **kwargs)
    lines = minimal._process_ai_response(text, '', 1, False, 90, 'korean', 'all', **kwargs)
    header = lines[0].split(minimal.GLOSSARY_SEP)
    return [dict(zip(header, line.split(minimal.GLOSSARY_SEP))) for line in lines[1:]]


@pytest.mark.parametrize('parser', ['balanced/full', 'minimal'])
@pytest.mark.parametrize('text', [
    'character,철수,Cheolsu,male,character,영희,Yeonghee,female',
    'character,철수,Cheolsu,malecharacter,영희,Yeonghee,female',
    'type,raw_name,translated_name,gender\ncharacter,철수,Cheolsu,male,character,영희,Yeonghee,female',
    'type,raw_name,translated_name,gendercharacter,철수,Cheolsu,malecharacter,영희,Yeonghee,female',
    '```csv\ntype,raw_name,translated_name,gender,character,철수,Cheolsu,male,character,영희,Yeonghee,female\n```',
    'Here is the glossary:\n```csv\ncharacter,철수,Cheolsu,male,character,영희,Yeonghee,female\n```\nDone.',
    'character\x1f철수\x1fCheolsu\x1fmalecharacter\x1f영희\x1fYeonghee\x1ffemale',
    '"character","철수","Cheolsu","male","character","영희","Yeonghee","female"',
])
def test_zero_or_one_entry_is_retried_as_joined_rows(parser, text):
    assert len(parse(parser, text, _recover_rows=False)) <= 1
    rows = parse(parser, text)
    assert [(r['raw_name'], r['translated_name'], r['gender']) for r in rows] == [
        ('철수', 'Cheolsu', 'male'), ('영희', 'Yeonghee', 'female')]


@pytest.mark.parametrize('parser', ['balanced/full', 'minimal'])
def test_custom_types_and_translated_name_touching_next_type(parser):
    rows = parse(parser, 'term,서울,Seoulfaction,청룡회,Azure Dragon Societyterm,부산,Busan')
    assert [(r['type'], r['raw_name'], r['translated_name']) for r in rows] == [
        ('term', '서울', 'Seoul'), ('faction', '청룡회', 'Azure Dragon Society'), ('term', '부산', 'Busan')]


@pytest.mark.parametrize('parser', ['balanced/full', 'minimal'])
def test_quoted_commas_and_type_words_are_not_row_boundaries(parser, monkeypatch):
    monkeypatch.setenv('GLOSSARY_CUSTOM_FIELDS', '["description"]')
    monkeypatch.setenv('GLOSSARY_INCLUDE_DESCRIPTION', '1')
    text = ('type,raw_name,translated_name,gender,description\n'
            'character,철수,Cheolsu,male,"Says ""hello"",term,not,a row",'
            'character,영희,Yeonghee,female,"A scholar, and friend"')
    rows = parse(parser, text)
    assert len(rows) == 2
    assert rows[0]['description'] == 'Says "hello",term,not,a row'
    assert rows[1]['description'] == 'A scholar, and friend'


@pytest.mark.parametrize('parser', ['balanced/full', 'minimal'])
def test_two_valid_entries_never_invoke_recovery(parser, monkeypatch):
    module = full if parser == 'balanced/full' else minimal
    def unexpected(*args):
        raise AssertionError('recovery ran after two valid entries')
    monkeypatch.setattr(module, 'recover_glossary_row_breaks', unexpected)
    rows = parse(parser, 'term,서울,Seoul\nterm,부산,Busan,term,대구,Daegu')
    assert len(rows) == 2


@pytest.mark.parametrize('parser', ['balanced/full', 'minimal'])
@pytest.mark.parametrize('text', [
    'type,raw_name,translated_name\nterm,서울,Seoul',
    'type,raw_name,translated_name,description\nterm,서울,Seoul,"term,fake,Fake,character,also,Fake"',
    'term,서울,Seoul,term,부산',
    'term,서울,"Seoul,term,부산,Busan',
])
def test_ambiguous_or_incomplete_text_keeps_normal_result(parser, text):
    assert parse(parser, text) == parse(parser, text, _recover_rows=False)


@pytest.mark.parametrize('parser', ['balanced/full', 'minimal'])
def test_type_word_in_unquoted_description_does_not_split_before_header_width(parser):
    text = 'type,raw_name,translated_name,gender,description\ncharacter,철수,Cheolsu,male,term,foo,bar'
    assert parse(parser, text) == parse(parser, text, _recover_rows=False)


def test_recovered_entries_still_pass_normal_validation(monkeypatch):
    text = 'character,철수,Cheolsu,male,character,(),Invalid,female'
    assert full.parse_api_response(text) == full.parse_api_response(text, _recover_rows=False)


def test_balanced_threshold_counts_valid_entries_not_empty_rows(monkeypatch):
    monkeypatch.setenv('GLOSSARY_ENTRY_TYPE_FILTER_MODE', 'none')
    text = ',,\nterm,서울,Seoul,term,부산,Busan'
    assert len(full.parse_api_response(text, _recover_rows=False)) == 2
    assert [e['translated_name'] for e in full.parse_api_response(text)] == ['Seoul', 'Busan']


@pytest.mark.parametrize('fast', ['0', '1'])
def test_minimal_fast_and_normal_paths_recover(fast, monkeypatch):
    monkeypatch.setenv('GLOSSARY_SKIP_ALL_VALIDATION', fast)
    assert len(parse('minimal', 'character,철수,Cheolsu,male,character,영희,Yeonghee,female')) == 2
