# -*- coding: utf-8 -*-
"""Editable placeholder texts (owner 2026-11-13): "let users update the available placeholder prompts".

The fixed texts that ``{relations_instruction}``, ``{subject_tracking_instruction}``, ``{description_example}``
and the other editable placeholders expand to can be replaced from Glossary Manager (right-click the
"Available placeholders" line > "Edit placeholder text…"). The texts are stored in the config key
``glossary_placeholder_texts`` and reach the extraction through ``GLOSSARY_PLACEHOLDER_TEXTS`` (JSON).

With nothing stored every prompt is byte-for-byte what it was before the feature: the digests in
``BASELINE_DIGESTS`` were recorded from origin/main aae300f4 (before the change) with ``capture_prompts``.
Two later fixes change the default prompts on purpose, and only them (``as_before`` undoes exactly
those two and must give the baseline back): Minimal now expands ``{gender_instruction}`` (it sent the
literal placeholder), and the built-in Balanced/Full prompt gets ``{language}`` (it sent ``{language}``).

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/test_glossary_placeholder_texts.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ENV = "GLOSSARY_PLACEHOLDER_TEXTS"
CONFIG_KEY = "glossary_placeholder_texts"

_PROMPT_ENV_KEYS = (
    "GLOSSARY_SYSTEM_PROMPT", "GLOSSARY_CUSTOM_FIELDS", "GLOSSARY_CUSTOM_ENTRY_TYPES",
    "GLOSSARY_TARGET_LANGUAGE", "GLOSSARY_INCLUDE_GENDER_CONTEXT", "GLOSSARY_INCLUDE_DESCRIPTION",
    "GLOSSARY_ENABLE_RELATIONS_TRACKER", "GLOSSARY_RELATIONS_FIELD", "GLOSSARY_RELATIONS_HISTORY_MODE",
    "GLOSSARY_INCLUDE_BOOK_TITLE", "GLOSSARY_USE_SMART_FILTER", "GLOSSARY_FORCE_DISABLE_SMART_FILTER",
    "GLOSSARY_MINIMAL_INCLUDE_RELATIONS", ENV,
)

#: Entry types without any ``has_gender``: ``{gender_instruction}`` expands to nothing.
NO_GENDER_TYPES = {"character": {"enabled": True, "has_gender": False},
                   "term": {"enabled": True, "has_gender": False}}


# =============================================================================================
# prompt capture (the real Balanced/Full build_prompt and the real Minimal request path)
# =============================================================================================

def _set_env(values):
    """Set the prompt env to exactly ``values`` (others removed); returns a restore callable."""
    saved = {key: os.environ.get(key) for key in set(_PROMPT_ENV_KEYS) | set(values)}
    for key in _PROMPT_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update({key: str(value) for key, value in values.items()})

    def restore():
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return restore


def balanced_prompt(env):
    import extract_glossary_from_epub as eg

    restore = _set_env(env)
    try:
        system, _user = eg.build_prompt("서윤은 도현을 보았다.")
    finally:
        restore()
    return system


def minimal_prompt(env, prompt=None, response=None, output_dir=".", cells_out=None):
    """The system message the Minimal path sends (GlossaryManager._extract_with_custom_prompt).

    ``cells_out``: a list that receives the relations cells the run detached from the rows."""
    import GlossaryManager as GM
    import extract_glossary_from_epub as eg
    import unified_api_client as uac

    sent = []

    class FakeClient:
        def __init__(self, *a, **k):
            self.context = None

    def send(messages, **kwargs):
        sent.append(messages[0]["content"])
        return SimpleNamespace(content=response or "type,raw_name,translated_name\ncharacter,서윤,Seoyun"), "stop", None

    extra = {"MODEL": "local/Qwen3.5-0.8B-UD-Q4_K_XL", "API_KEY": "test-key", "GLOSSARY_DEFER_SAVE": "1",
             "_CHUNK_ALREADY_FILTERED": "1", "THREAD_SUBMISSION_DELAY_SECONDS": "0",
             "GLOSSARY_MAX_OUTPUT_TOKENS": "4096"}
    saved_extra = {key: os.environ.get(key) for key in extra}
    os.environ.update(extra)
    saved_attrs = (uac.UnifiedClient, GM.send_with_interrupt, GM._stop_requested, GM._get_stop_file_path)
    uac.UnifiedClient, GM.send_with_interrupt = FakeClient, send
    GM._stop_requested, GM._get_stop_file_path = False, (lambda: None)
    restore = _set_env(env)
    key = GM._minimal_relations_key(output_dir)
    GM._minimal_relations_cells.pop(key, None)
    try:
        GM._extract_with_custom_prompt(prompt or eg.DEFAULT_AUTO_GLOSSARY_PROMPT,
                                       "서윤은 도현을 보았다.\n=== CONTEXT 1 END ===", "korean",
                                       1, 50, 50, None, output_dir, raise_on_failure=True)
        cells = GM._minimal_relations_cells.pop(key, [])
        if cells_out is not None:
            cells_out.extend(cells)
    finally:
        restore()
        uac.UnifiedClient, GM.send_with_interrupt, GM._stop_requested, GM._get_stop_file_path = saved_attrs
        for key, value in saved_extra.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert len(sent) == 1
    return sent[0]


def scenarios():
    """name -> (kind, env): description on/off, gender context on/off, Relations Tracker on/off."""
    out = {}
    for desc in (0, 1):
        for gender in (0, 1):
            for relations in (0, 1):
                tag = f"desc{desc}_gender{gender}_rel{relations}"
                out["balanced_" + tag] = ("balanced", {
                    "GLOSSARY_TARGET_LANGUAGE": "English",
                    "GLOSSARY_INCLUDE_BOOK_TITLE": "0",
                    "GLOSSARY_CUSTOM_FIELDS": json.dumps(["description"] if desc else []),
                    # gender off: no entry type has gender, so {gender_instruction} is empty
                    "GLOSSARY_CUSTOM_ENTRY_TYPES": json.dumps({} if gender else NO_GENDER_TYPES),
                    "GLOSSARY_INCLUDE_GENDER_CONTEXT": str(gender),
                    "GLOSSARY_ENABLE_RELATIONS_TRACKER": str(relations),
                })
                out["minimal_" + tag] = ("minimal", {
                    "GLOSSARY_TARGET_LANGUAGE": "English",
                    "GLOSSARY_INCLUDE_BOOK_TITLE": "0",
                    "GLOSSARY_INCLUDE_DESCRIPTION": str(desc),
                    "GLOSSARY_INCLUDE_GENDER_CONTEXT": str(gender),
                    "GLOSSARY_ENABLE_RELATIONS_TRACKER": str(relations),
                    # Minimal asks for relations only with its own toggle on as well
                    "GLOSSARY_MINIMAL_INCLUDE_RELATIONS": str(relations),
                })
    return out


def capture_prompts(extra_env=None):
    prompts = {}
    for name, (kind, env) in scenarios().items():
        env = dict(env, **(extra_env or {}))
        prompts[name] = balanced_prompt(env) if kind == "balanced" else minimal_prompt(env)
    return prompts


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: sha256 of every scenario prompt on origin/main aae300f4, before editable placeholder texts existed.
#: A deliberate change of a default prompt or default placeholder text must update these.
BASELINE_DIGESTS = {
    'balanced_desc0_gender0_rel0': '8ed32e943e7da11540627a982cb052591e6306b06902c6b6acff757d253e4d36',
    'minimal_desc0_gender0_rel0': 'a229226ab9748b5db444ef1a8a564038c4f3a37dfd0b824de2c41aeafcbb2467',
    'balanced_desc0_gender0_rel1': '78bf05669c32456d83657462901534a36412add23ba6c90fb5db0fbe4b737fee',
    'minimal_desc0_gender0_rel1': 'cb3972220439b055ce335d7c9cca17e4fe822ed3f68b6400b53f14386ae9c6a9',
    'balanced_desc0_gender1_rel0': 'ace88a61fb379bc862e261e5e918f0d2b42e4cebbe2f7d3ca2ac425619a88fc4',
    'minimal_desc0_gender1_rel0': '6b2df2c634c9927cd16a0b5e19814047d853844cace328a405bf45c95d530526',
    'balanced_desc0_gender1_rel1': '64b63e89ff06f4a97942aebe30aa2d46fa960abd8453490050f3958c5c059bec',
    'minimal_desc0_gender1_rel1': '8c6cca561c3290bbd270c28b6d8a37fa07b377ed173e4cec17e43e8474af6508',
    'balanced_desc1_gender0_rel0': '5f22426aa7993ac3188d1f420de5d26ea766fed06782e46e87c4a5e2ac38776a',
    'minimal_desc1_gender0_rel0': '304e8160b556070b21d425e50d659cdf4c4c291d72953c4ba540cfa2abc23291',
    'balanced_desc1_gender0_rel1': '059f1b029846756a69ef2d209a39c9c59a5326728218ca63be98682874ae3ede',
    'minimal_desc1_gender0_rel1': '9788369fdf96cf40203d3f6d40e8ab1163c5c7e3102a5ea81c38ab57b6376ec6',
    'balanced_desc1_gender1_rel0': 'd6383bc77291767d7b2a8c2ba78d8d835b2d528ca24363eedfecff7fa15934fe',
    'minimal_desc1_gender1_rel0': '96289a686645912f070833bdcbe559c097a8b60d06fd1d8c89cc330393fcf171',
    'balanced_desc1_gender1_rel1': 'f56a3425c75b158f6b56f20f4821d1b3401ef7d6f75b11b6095ba508ec2a6cb0',
    'minimal_desc1_gender1_rel1': '3b31011afe5cbea04f46f130386bf125719758834aeeec7c090160895ae3514b',
}

#: sha256 of the same prompts with the two fixes ({gender_instruction} in Minimal, {language} in the
#: built-in Balanced/Full prompt); every scenario contained one of the two literals. The *_rel1 ones also
#: have the long relations text with {gender_types} (2026-11-14; see relations_as_before).
CURRENT_DIGESTS = {
    'balanced_desc0_gender0_rel0': 'f936e7c05efd38dce48746aa86f86306c5476e97d84749a9dfe59ac79051e597',
    'minimal_desc0_gender0_rel0': '05a635f37a44a8256649bd3f1ef16d18cb186643173a3bef2ee3da5d064540bb',
    'balanced_desc0_gender0_rel1': 'ed4e3cac8a49548b20360faee62092f4e1e1e1ca7166b57df4d2225a34d24c00',
    'minimal_desc0_gender0_rel1': '36891d51a43e98e55311e81a496c1e4f0ca0889162b60f535b2f3e5e99ed9a86',
    'balanced_desc0_gender1_rel0': 'cf81dd24ca606a71e62e623d526d14a0d5443397bb3a43c197873ba50aa1c139',
    'minimal_desc0_gender1_rel0': 'cde38f84294d833efc7d7d6b64ced3a1397dcc182c2a469e728d4a61fbb4d0fc',
    'balanced_desc0_gender1_rel1': 'c88f3264ce85f0b2a67795f3deceae5bc1e5564df3572a686969e2628baf218e',
    'minimal_desc0_gender1_rel1': 'aae5abdd45d425f08c0e0737b3626dfa987ca60c4f5f7e5f59514b9614028fd1',
    'balanced_desc1_gender0_rel0': 'd0d8bf230f9f5d4a432a090ad0ca012302ed9bc378a41a4dcb6a3ec054b3507a',
    'minimal_desc1_gender0_rel0': '2c16930db10f5ddd35ef5b398fcda808ba0fd9d6144023b482a7c93fb574ddd9',
    'balanced_desc1_gender0_rel1': 'aa34b330c2a427382ea7aa4e2ab946d7fa22931c15abc2c50963394fed410fe5',
    'minimal_desc1_gender0_rel1': '1ea04df6e6fe8f062e15691ea7d78dbf686f3706fcf33cebc14bd57c7447fe19',
    'balanced_desc1_gender1_rel0': '1c05621d24c05a2115bb5aea03d65b671cc9d6ef177e3e721fe40ca39f7c66c1',
    'minimal_desc1_gender1_rel0': '48b4ea7fdb3d1c7b2de65d158a854e05c74be28efcd11d904a12725ac58862aa',
    'balanced_desc1_gender1_rel1': 'c291d8b0a061472cff7ccd0e9064aec31ca0fb0416f2a53bafb92880af50361a',
    'minimal_desc1_gender1_rel1': 'd30914c011f96f3e3d1f632ee45b4eacb3f7d97f2d4164fa1fb95fc5726f49c4',
}

#: {gender_instruction} with the built-in text and the default entry types (character, titles, nicknames)
GENDER_EXPANDED = ("The gender column applies only to character, titles, nicknames entries — determine gender "
                   "from context clues, leave empty if insufficient.\nFor all other entry types, leave gender empty.")
LANGUAGE_BEFORE = "must be in {language}, While the raw name"
LANGUAGE_NOW = "must be in English, While the raw name"


#: {relations_instruction} is the long form again, without "Fiancé" (owner, 2026-11-14): a line that says
#: which column the relations go in, and full descriptions in its two example rows.
RELATIONS_LABELS_BEFORE = "Older Sister, Wife, Fiancé, Best Friend,"
RELATIONS_LABELS_NOW = "Older Sister, Wife, Best Friend,"
RELATIONS_COLUMN_RULE = re.compile(r"`relations` is its own column[^\n]*: never write relationships in another "
                                   r"column\.\n")
RELATIONS_EXAMPLE_LINE = re.compile(r"^Example rows that mirror each other: [^\n]*$", re.MULTILINE)
RELATIONS_DESCRIPTIONS = (
    ('"The enigmatic guild leader of the Shadow Lotus who operates from the concealed backrooms of the capital"',
     '"Guild leader"'),
    ('"A master swordsman from the Northern Sect known for his icy demeanor and unparalleled skill with the Frost '
     'Blade technique"', '"Swordsman"'),
)


#: {gender_types} in the relations text (owner, 2026-11-14): "for <the entry types with gender> entries only"
#: and "Leave it empty for all other entry types," instead of "character" / "non-character entries,".
RELATIONS_TYPES_NOW = re.compile(r"The `relations` column: for [^\n]*? entries only, list")
RELATIONS_TYPES_BEFORE = "The `relations` column: for character entries only, list"
RELATIONS_OTHERS_NOW = "Leave it empty for all other entry types, surname-only"
RELATIONS_OTHERS_BEFORE = "Leave it empty for non-character entries, surname-only"


def relations_as_before(text):
    """The prompt with the short {relations_instruction} of aae300f4 (no change without the instruction)."""
    if "The `relations` column:" not in text:
        assert "is its own column" not in text
        return text
    assert len(RELATIONS_TYPES_NOW.findall(text)) == 1 and text.count(RELATIONS_OTHERS_NOW) == 1
    text = RELATIONS_TYPES_NOW.sub(RELATIONS_TYPES_BEFORE, text).replace(RELATIONS_OTHERS_NOW, RELATIONS_OTHERS_BEFORE)
    assert text.count(RELATIONS_LABELS_NOW) == 1 and len(RELATIONS_COLUMN_RULE.findall(text)) == 1
    assert len(RELATIONS_EXAMPLE_LINE.findall(text)) == 1
    text = RELATIONS_COLUMN_RULE.sub("", text.replace(RELATIONS_LABELS_NOW, RELATIONS_LABELS_BEFORE))

    def short(match):
        line = match.group(0)
        for long_text, short_text in RELATIONS_DESCRIPTIONS:
            line = line.replace(long_text, short_text)
        return line

    return RELATIONS_EXAMPLE_LINE.sub(short, text)


def as_before(name, text):
    """The prompt with the intended changes undone (Minimal: {gender_instruction}; Balanced/Full: {language};
    both: the long {relations_instruction})."""
    text = relations_as_before(text)
    if name.startswith("minimal"):
        assert "{gender_instruction}" not in text and text.count(GENDER_EXPANDED) == 1
        return text.replace(GENDER_EXPANDED, "{gender_instruction}")
    assert "{language}" not in text and text.count(LANGUAGE_NOW) == 1
    return text.replace(LANGUAGE_NOW, LANGUAGE_BEFORE)


EDITABLE = [
    "description_mandatory", "description_detailed", "description_in_language", "description_excluded_note",
    "description_example", "example", "description_name_split_example", "name_split_example",
    "gender_instruction", "subject_tracking_instruction", "relations_instruction",
]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for key, sub in (("GLOSSARION_APP_DIR", "app"), ("GLOSSARION_LIBRARY_DIR", "lib"),
                     ("OUTPUT_DIRECTORY", "out")):
        path = tmp_path / sub
        path.mkdir()
        monkeypatch.setenv(key, str(path))
    for key in _PROMPT_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture(scope="module")
def eg():
    import extract_glossary_from_epub
    return extract_glossary_from_epub


# =============================================================================================
# nothing stored: byte-for-byte the prompts of before
# =============================================================================================

@pytest.mark.parametrize("extra_env", [None, {ENV: "{}"}, {ENV: "not json"}],
                         ids=["no-env", "empty-dict", "bad-json"])
def test_default_prompts_are_byte_identical_to_before(extra_env):
    """Only the two intended fixes differ from the baseline; undone, every prompt is the old one."""
    prompts = capture_prompts(extra_env)
    assert set(prompts) == set(BASELINE_DIGESTS) == set(CURRENT_DIGESTS)
    changed = [name for name, text in prompts.items() if digest(as_before(name, text)) != BASELINE_DIGESTS[name]]
    assert not changed
    assert {name: digest(text) for name, text in prompts.items()} == CURRENT_DIGESTS


def test_minimal_gets_the_gender_instruction_and_balanced_the_language():
    prompts = capture_prompts()
    for name, text in prompts.items():
        if name.startswith("minimal"):
            assert "{gender_instruction}" not in text
            assert "Columns in this exact order: type,raw_name,translated_name,gender,description\n" + GENDER_EXPANDED in text
        else:
            assert "{language}" not in text
            assert "Critical Requirement: The translated name" in text and LANGUAGE_NOW in text


def test_texts_equal_to_the_defaults_change_nothing(eg):
    stored = {item["name"]: item["default"] for item in eg.editable_placeholders()}
    prompts = capture_prompts({ENV: json.dumps(stored, ensure_ascii=False)})
    assert {name: digest(text) for name, text in prompts.items()} == CURRENT_DIGESTS


def test_the_scenarios_cover_every_switch():
    prompts = capture_prompts()
    assert len(set(prompts.values())) == len(prompts)  # every switch changes the prompt
    on = prompts["balanced_desc1_gender1_rel1"]
    assert "The `relations` column:" in on and "The gender column applies only to" in on
    off = prompts["balanced_desc0_gender0_rel0"]
    assert "`relations`" not in off and "The gender column applies only to" not in off
    assert "Subject Tracking" in prompts["minimal_desc0_gender1_rel0"]
    assert "Subject Tracking" not in prompts["minimal_desc0_gender0_rel0"]


# =============================================================================================
# the lookup
# =============================================================================================

def test_editable_placeholders_list(eg):
    items = eg.editable_placeholders()
    assert [item["name"] for item in items] == EDITABLE
    assert all(item["note"] for item in items)
    defaults = {item["name"]: item["default"] for item in items}
    assert defaults["relations_instruction"].endswith("`{example_row_1}` and `{example_row_2}`")
    assert defaults["gender_instruction"] == (
        "The gender column applies only to {gender_types} entries — determine gender from context clues, "
        "leave empty if insufficient.\nFor all other entry types, leave gender empty.")
    for dynamic in ("fields", "fields1", "language", "entries", "min_frequency", "max_names", "max_titles",
                    "marker"):
        assert dynamic not in defaults


def test_overrides_parse_tolerantly(eg, monkeypatch):
    monkeypatch.setenv(ENV, "{broken")
    assert eg.placeholder_text_overrides() == {}
    assert eg.placeholder_text("example", "DEFAULT") == "DEFAULT"
    monkeypatch.setenv(ENV, json.dumps(["not", "a", "dict"]))
    assert eg.placeholder_text_overrides() == {}
    monkeypatch.setenv(ENV, json.dumps({"example": "Mine", "bad": 3, "empty": ""}))
    assert eg.placeholder_text_overrides() == {"example": "Mine", "empty": ""}
    assert eg.placeholder_text("example", "DEFAULT") == "Mine"
    assert eg.placeholder_text("empty", "DEFAULT") == ""
    monkeypatch.delenv(ENV)
    assert eg.placeholder_text("example", "DEFAULT") == "DEFAULT"


# =============================================================================================
# each override changes its own expansion, and only that
# =============================================================================================

def _scenario_env(name):
    return scenarios()[name][1]


#: placeholder -> a scenario in which the default prompt expands it, and the default text it expands to
_ACTIVE_IN = {
    "description_mandatory": "balanced_desc1_gender0_rel0",
    "description_detailed": "balanced_desc1_gender0_rel0",
    "description_in_language": "balanced_desc1_gender0_rel0",
    "description_excluded_note": "balanced_desc1_gender0_rel0",
    "description_example": "minimal_desc1_gender0_rel0",
    "example": "minimal_desc0_gender0_rel0",
    "description_name_split_example": "minimal_desc1_gender0_rel0",
    "name_split_example": "minimal_desc0_gender0_rel0",
    "gender_instruction": "balanced_desc0_gender1_rel0",
    "subject_tracking_instruction": "minimal_desc0_gender1_rel0",
    "relations_instruction": "minimal_desc0_gender0_rel1",
}


def _prompt(name, stored=None):
    kind, env = scenarios()[name]
    if stored is not None:
        env = dict(env, **{ENV: json.dumps(stored, ensure_ascii=False)})
    return balanced_prompt(env) if kind == "balanced" else minimal_prompt(env)


@pytest.mark.parametrize("placeholder", EDITABLE)
def test_an_override_replaces_only_its_own_text(eg, placeholder):
    scenario = _ACTIVE_IN[placeholder]
    default = {item["name"]: item["default"] for item in eg.editable_placeholders()}[placeholder]
    before = _prompt(scenario)
    marker = f"MY-OWN-{placeholder.upper()}-TEXT"
    after = _prompt(scenario, {placeholder: marker})
    assert marker in after and marker not in before
    if placeholder == "gender_instruction":
        expected_text = default.replace("{gender_types}", "character, titles, nicknames")
    elif placeholder == "relations_instruction":
        start = before.index("Add a `relations` column")
        end = before.index("\n", before.index("Example rows that mirror each other"))
        expected_text = before[start:end]
        marker = "Add a `relations` column as one extra LAST column of every row. " + marker
    else:
        expected_text = default
    assert expected_text in before
    assert after == before.replace(expected_text, marker, 1)


@pytest.mark.parametrize("placeholder", [p for p in EDITABLE if p != "gender_instruction"])
def test_an_empty_override_expands_to_nothing(eg, placeholder):
    scenario = _ACTIVE_IN[placeholder]
    default = {item["name"]: item["default"] for item in eg.editable_placeholders()}[placeholder]
    after = _prompt(scenario, {placeholder: ""})
    if placeholder == "description_in_language":
        assert "Critical Requirement: The translated name column must be in" in after
        return
    if placeholder == "description_excluded_note":
        assert '"Still", "But"\n' in after
        return
    probe = default.strip().splitlines()[0][:40] if placeholder != "relations_instruction" else "The `relations` column"
    assert probe not in after
    # a placeholder that owns its line drops the line: no orphan "- " bullet is left behind
    assert "\n- \n" not in after and not after.endswith("\n- ")
    if placeholder == "relations_instruction":
        # Minimal declares the extra column in front of the text, so that sentence stays
        assert "Add a `relations` column as one extra LAST column of every row. " in after


def test_empty_relations_text_drops_the_line_when_nothing_is_declared(eg):
    after = _prompt("balanced_desc0_gender0_rel1", {"relations_instruction": ""})
    assert "relations_instruction" not in after
    assert "\n- \n" not in after
    assert "The `relations` column:" not in after


def test_gender_override_uses_gender_types_and_stays_empty_without_gender(eg):
    stored = {"gender_instruction": "Gender only for: {gender_types}."}
    on = _prompt("balanced_desc0_gender1_rel0", stored)
    assert "Gender only for: character, titles, nicknames." in on
    off = _prompt("balanced_desc0_gender0_rel0", stored)
    assert "Gender only for" not in off
    assert digest(off) == CURRENT_DIGESTS["balanced_desc0_gender0_rel0"]


def test_relations_override_with_and_without_tokens(eg, monkeypatch):
    import relations_tracking as rt

    with_tokens = "Column {field} in {language}: `{example_row_1}` / `{example_row_2}`"
    text = rt.instruction_text("bonds", "German", template=with_tokens)
    assert text == ("Column bonds in German: `character,이히리ᐐ 나애,Dihirit Ade,female,뢤사난: Older Sister` / "
                    "`character,뢤사난,Kim Sang-hyu,male,이히리ᐐ 나애: Younger Brother`")
    assert rt.instruction_text("bonds", "German", template="No tokens here.") == "No tokens here."
    assert rt.instruction_text("bonds", "German", declare_column=True, template="Plain.") == (
        "Add a `bonds` column as one extra LAST column of every row. Plain.")
    assert rt.instruction_text("bonds", "German", template=None) == rt.instruction_text("bonds", "German")

    after = _prompt("balanced_desc0_gender0_rel1", {"relations_instruction": with_tokens})
    assert "- Column relations in English: `character,이히리ᐐ 나애,Dihirit Ade,female,뢤사난: Older Sister`" in after
    assert "{example_row_1}" not in after and "{field}" not in after


def test_relations_template_uses_the_new_token_names():
    import relations_tracking as rt

    assert "{example_row_1}" in rt._INSTRUCTION_TEMPLATE and "{example_row_2}" in rt._INSTRUCTION_TEMPLATE
    assert "{row_a}" not in rt._INSTRUCTION_TEMPLATE and "{row_b}" not in rt._INSTRUCTION_TEMPLATE
    assert "{example_row" not in rt.instruction_text("relations", "English")


def test_relations_tracking_does_not_import_the_extractor():
    source = (SRC / "relations_tracking.py").read_text(encoding="utf-8")
    assert "extract_glossary_from_epub" not in source


# =============================================================================================
# {gender_types} in the relations text (owner, 2026-11-14)
# =============================================================================================

SECT_TYPES = {"character": {"enabled": True, "has_gender": False}, "sect": {"enabled": True, "has_gender": True},
              "titles": {"enabled": False, "has_gender": True}, "term": {"enabled": True, "has_gender": False}}


def test_relations_text_lists_the_entry_types_with_gender(eg):
    on = {"GLOSSARY_ENABLE_RELATIONS_TRACKER": "1", "GLOSSARY_MINIMAL_INCLUDE_RELATIONS": "1"}
    cases = (({}, "character, titles, nicknames"),                  # the default entry types
             ({"GLOSSARY_CUSTOM_ENTRY_TYPES": json.dumps(SECT_TYPES)}, "sect"),  # a custom type; disabled ones left out
             ({"GLOSSARY_CUSTOM_ENTRY_TYPES": json.dumps(NO_GENDER_TYPES)}, "character"))  # none: "character"
    for extra, listed in cases:
        env = dict(_minimal_env(**on), **extra)
        for system in (balanced_prompt(env), minimal_prompt(env)):
            assert f"The `relations` column: for {listed} entries only, list this character's" in system
            assert "Leave it empty for all other entry types, surname-only entries" in system
            assert "{gender_types}" not in system
            assert ("The gender column applies only to" in system) is (listed != "character")
    assert eg.gender_entry_type_names(SECT_TYPES) == ["sect"]
    assert eg.gender_instruction_text(SECT_TYPES).startswith("The gender column applies only to sect entries")
    assert eg.gender_instruction_text(NO_GENDER_TYPES) == ""


def test_a_stored_relations_text_without_the_token_is_kept(eg):
    stored = {"relations_instruction": "My relations rule for characters."}
    system = _prompt("balanced_desc0_gender1_rel1", stored)
    assert "\n- My relations rule for characters.\n" in system and "{gender_types}" not in system
    stored = {"relations_instruction": "Relations for {gender_types} rows."}
    assert "\n- Relations for character, titles, nicknames rows.\n" in _prompt("balanced_desc0_gender1_rel1", stored)


# =============================================================================================
# placeholder_preview: what a placeholder says right now (the hover text)
# =============================================================================================

PROBE_TEMPLATE = "Return CSV rows.\n{fields}\n- {relations_instruction}\nAAAA\n%s\nZZZZ"
PREVIEW_SCENARIOS = ("desc1_gender1_rel1", "desc0_gender0_rel0", "desc0_gender1_rel1", "desc1_gender0_rel1")
#: Minimal's {min_frequency}/{max_names}/{max_titles}: minimal_prompt passes 1/50/50 (the run reads them from env)
MINIMAL_LIMITS = {"GLOSSARY_MIN_FREQUENCY": "1", "GLOSSARY_MAX_NAMES": "50", "GLOSSARY_MAX_TITLES": "50"}


def _between(system):
    start = system.index("AAAA\n") + len("AAAA\n")
    inner = system[start:system.index("ZZZZ", start)]
    return inner[:-1] if inner.endswith("\n") else inner


def _real_insert(mode, env, name, stored=None):
    """What the real Balanced/Full build_prompt or Minimal request path writes for {name}."""
    if name in PREVIEW_TOKENS:
        parent = "relations_instruction" if (env.get("GLOSSARY_ENABLE_RELATIONS_TRACKER") == "1"
                                             or name != "gender_types") else "gender_instruction"
        if parent == "gender_instruction":
            stored = {parent: "{" + name + "}"}
            template = PROBE_TEMPLATE % "{gender_instruction}"
        else:  # the relations line comes before the (empty) AAAA/ZZZZ block of the template
            stored = {parent: "AAAA\n{" + name + "}\nZZZZ"}
            template = PROBE_TEMPLATE % ""
    else:
        template = PROBE_TEMPLATE % ("{" + name + "}")
    if stored is not None:
        env = dict(env, **{ENV: json.dumps(stored, ensure_ascii=False)})
    if mode == "balanced":
        return _between(balanced_prompt(dict(env, GLOSSARY_SYSTEM_PROMPT=template))), template
    return _between(minimal_prompt(env, prompt=template)), template


PREVIEW_TOKENS = ("gender_types", "field", "column_rule", "example_row_1", "example_row_2")


@pytest.mark.parametrize("scenario", PREVIEW_SCENARIOS)
@pytest.mark.parametrize("mode", ["balanced", "minimal"])
def test_preview_is_what_the_real_prompt_build_inserts(eg, mode, scenario):
    env = dict(scenarios()[f"{mode}_{scenario}"][1], **MINIMAL_LIMITS)
    relations_on = env["GLOSSARY_ENABLE_RELATIONS_TRACKER"] == "1"
    checked = 0
    for name in eg.known_placeholder_names(mode):
        if name == "marker":
            preview = eg.placeholder_preview(name, mode=mode, settings=env)
            assert preview["per_request"] and preview["text"] and preview["active"]
            continue
        if name in ("chapter_text", "text"):
            # always replaced with nothing in the system prompt (the chapter is the user message)
            assert _real_insert(mode, env, name)[0] == ""
            preview = eg.placeholder_preview(name, mode=mode, settings=env)
            assert not preview["per_request"] and preview["active"] is False and preview["text"] == ""
            assert preview["inactive"].startswith("Replaced with nothing in the system prompt")
            continue
        if name in PREVIEW_TOKENS and not relations_on and name != "gender_types":
            preview = eg.placeholder_preview(name, mode=mode, prompt=PROBE_TEMPLATE % "", settings=env)
            assert preview["active"] is False and preview["text"] == ""
            continue
        real, template = _real_insert(mode, env, name)
        # the settings come from the argument: nothing of the scenario is in os.environ here
        assert "GLOSSARY_ENABLE_RELATIONS_TRACKER" not in os.environ
        preview = eg.placeholder_preview(name, mode=mode, prompt=template, settings=env)
        assert not preview["per_request"] and preview["known"]
        assert preview["text"] == real, name
        assert eg.placeholder_expansion(name, mode=mode, prompt=template, settings=env) == real, name
        assert eg.placeholder_is_active(name, mode=mode, prompt=template, settings=env) is preview["active"]
        if not preview["active"]:
            assert real == "", name
        checked += 1
    assert checked >= 15


def test_preview_flags_names_and_live_settings(eg):
    assert eg.known_placeholder_names("balanced", tokens=False) == (
        "fields", "fields1", "language", "entries", "description_mandatory", "description_detailed",
        "description_in_language", "description_excluded_note", "description_example", "example",
        "description_name_split_example", "name_split_example", "gender_instruction",
        "subject_tracking_instruction", "relations_instruction", "chapter_text", "text")
    assert eg.known_placeholder_names("minimal", tokens=False) == (
        "fields", "fields1", "language", "entries", "min_frequency", "max_names", "max_titles", "marker",
        "description_mandatory", "description_detailed", "description_in_language", "description_excluded_note",
        "description_example", "example", "description_name_split_example", "name_split_example",
        "gender_instruction", "subject_tracking_instruction", "relations_instruction")
    assert eg.known_placeholder_names("minimal")[-5:] == PREVIEW_TOKENS

    off = eg.placeholder_preview("{relations_instruction}", settings={"GLOSSARY_ENABLE_RELATIONS_TRACKER": False})
    assert off["active"] is False and off["text"] == "" and off["note"].startswith("Active when the Relations Tracker")
    on = eg.placeholder_preview("relations_instruction", settings={"GLOSSARY_ENABLE_RELATIONS_TRACKER": True},
                                custom_types=SECT_TYPES, language="German")
    assert on["active"] and on["text"].startswith("The `relations` column: for sect entries only")
    assert "in German, 1–3 words" in on["text"]
    # every change shows at once: the same call with other live values says something else
    assert eg.placeholder_expansion("gender_types", custom_types=NO_GENDER_TYPES,
                                    settings={"GLOSSARY_ENABLE_RELATIONS_TRACKER": "1"}) == "character"
    assert eg.placeholder_preview("gender_types", custom_types=NO_GENDER_TYPES)["active"] is False
    assert eg.placeholder_expansion("gender_types", custom_types=SECT_TYPES) == "sect"
    assert eg.placeholder_expansion("language", language="French") == "French"
    assert eg.placeholder_expansion("entries", custom_types=SECT_TYPES) == "Character, Sect, & Term entries"
    assert eg.placeholder_expansion("fields", fields=["description", "notes"], custom_types=SECT_TYPES) == (
        "Columns:\ntype, raw_name, translated_name, gender, description, notes\nEntry Types:\ncharacter, sect, term")
    assert eg.placeholder_expansion("description_mandatory", fields=[]) == ""
    assert eg.placeholder_is_active("description_mandatory", fields=[]) is False
    assert eg.placeholder_expansion("description_mandatory", fields=["description"]) == (
        "The description column is mandatory and must be detailed")
    assert eg.placeholder_expansion("max_names", mode="minimal", settings={"GLOSSARY_MAX_NAMES": 7}) == "7"
    marker = eg.placeholder_preview("marker", mode="minimal")
    assert marker["per_request"] and "context window marker count" in marker["text"]
    unknown = eg.placeholder_preview("max_names", mode="balanced")
    assert unknown["known"] is False and unknown["active"] is False and unknown["text"] == ""
    # a stored text shows, with its own tokens filled in
    stored = json.dumps({"subject_tracking_instruction": "Track pronouns."})
    assert eg.placeholder_expansion("subject_tracking_instruction", mode="minimal", settings={
        "GLOSSARY_INCLUDE_GENDER_CONTEXT": "1", ENV: stored}) == "Track pronouns."
    # the preview leaves nothing behind for the extraction
    assert eg._prompt_settings() is None and ENV not in os.environ


# =============================================================================================
# Glossary Manager: right-click menu and the Placeholder Text dialog (offscreen Qt)
# =============================================================================================

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def gui(qapp, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
    from GlossaryManager_GUI import GlossaryManagerMixin
    import review_dialog

    answers = {"value": True, "asked": []}

    def _confirm(parent, title, text):
        answers["asked"].append(text)
        return answers["value"]

    monkeypatch.setattr(review_dialog, "confirm_prompt_profile_action", _confirm)

    class Harness(GlossaryManagerMixin, QMainWindow):
        def __init__(self):
            super().__init__()
            self.config = {}
            self.saves = 0
            self._screen = QApplication.primaryScreen().geometry()
            self.dialog = self

        def save_config(self, show_message=False):
            self.saves += 1
            return True

        def append_log(self, text):
            pass

    window = Harness()
    window.pages = {}
    for name in ("_setup_manual_glossary_tab", "_setup_auto_glossary_tab"):
        page = QWidget()
        getattr(window, name)(page)
        window.pages[name] = page
    window.answers = answers
    yield window
    dialog = getattr(window, "_placeholder_text_dialog", None)
    if dialog is not None:
        try:
            dialog.hide()
            dialog.deleteLater()
        except RuntimeError:
            pass
    window.deleteLater()


def _placeholder_line(page):
    from PySide6.QtWidgets import QLineEdit
    lines = [w for w in page.findChildren(QLineEdit) if w.text().startswith("Available placeholders:")
             and "{relations_instruction}" in w.text()]
    assert len(lines) == 1
    return lines[0]


@pytest.mark.parametrize("tab", ["_setup_manual_glossary_tab", "_setup_auto_glossary_tab"])
def test_both_placeholder_lines_have_the_new_action(gui, tab):
    from PySide6.QtCore import Qt

    line = _placeholder_line(gui.pages[tab])
    assert line.contextMenuPolicy() == Qt.CustomContextMenu
    menu = gui._placeholder_line_menu(line)
    texts = [a.text().replace("&", "") for a in menu.actions() if not a.isSeparator()]
    assert any(t.startswith("Copy") for t in texts)
    assert any(t.startswith("Select All") for t in texts)
    assert texts[-1] == "Edit placeholder text…"
    # The line's blue text colour is the menu highlight's colour: the menu sets its own text colour.
    assert "#5a9fd4" in line.styleSheet() and "QMenu::item { color: white; }" in menu.styleSheet()
    menu.actions()[-1].trigger()
    dialog = gui._placeholder_text_dialog
    assert dialog.windowTitle() == "Placeholder Text"
    assert not dialog.isModal()
    menu.deleteLater()


def test_dialog_lists_exactly_the_editable_placeholders(gui, eg):
    dialog = gui._open_placeholder_text_dialog()
    combo = dialog.placeholder_combo
    assert [combo.itemData(i) for i in range(combo.count())] == EDITABLE
    assert [combo.itemText(i) for i in range(combo.count())] == ["{" + n + "}" for n in EDITABLE]
    assert not combo.itemIcon(0).isNull()  # Halgakos.ico, like the profile dropdown
    assert dialog.placeholder_editor.toPlainText() == eg.editable_placeholders()[0]["default"]
    assert dialog.placeholder_status.text() == "Default text"
    assert dialog.placeholder_note.text()


def test_save_and_reset_write_config_and_env(gui, eg, monkeypatch):
    import qt_widget_helpers

    beeps = []
    monkeypatch.setattr(qt_widget_helpers, "saved_beep", lambda: beeps.append(1))
    dialog = gui._open_placeholder_text_dialog()
    combo, editor = dialog.placeholder_combo, dialog.placeholder_editor
    buttons = dialog.placeholder_buttons
    combo.setCurrentIndex(EDITABLE.index("relations_instruction"))
    default = editor.toPlainText()
    assert default == eg.editable_placeholders()[-1]["default"]

    editor.setPlainText("Relations in {language}.")
    assert dialog.placeholder_status.text() == "Default text (unsaved changes)"
    saves = gui.saves
    buttons["save"].click()
    assert gui.config[CONFIG_KEY] == {"relations_instruction": "Relations in {language}."}
    assert json.loads(os.environ[ENV]) == {"relations_instruction": "Relations in {language}."}
    assert gui.saves == saves + 1
    assert dialog.placeholder_status.text() == "Edited"
    # the Multi API Key Manager's saved confirmation: green "Saved!" and the Windows sound
    assert buttons["save"].text() == "💾 Saved!" and beeps == [1]
    assert qt_widget_helpers.SAVED_FLASH_COLOR in buttons["save"].styleSheet()
    assert eg.placeholder_text("relations_instruction", "x") == "Relations in {language}."

    # empty text is stored (the placeholder expands to nothing)
    combo.setCurrentIndex(EDITABLE.index("example"))
    editor.setPlainText("")
    buttons["save"].click()
    assert gui.config[CONFIG_KEY] == {"relations_instruction": "Relations in {language}.", "example": ""}

    # a text equal to the default is not stored
    editor.setPlainText(eg.editable_placeholders()[EDITABLE.index("example")]["default"])
    buttons["save"].click()
    assert gui.config[CONFIG_KEY] == {"relations_instruction": "Relations in {language}."}

    combo.setCurrentIndex(EDITABLE.index("relations_instruction"))
    assert editor.toPlainText() == "Relations in {language}."
    buttons["reset"].click()
    assert gui.answers["asked"]  # Reset of a stored text asks first
    assert gui.config[CONFIG_KEY] == {}
    assert json.loads(os.environ[ENV]) == {}
    assert editor.toPlainText() == default
    assert dialog.placeholder_status.text() == "Default text"
    assert buttons["reset"].text() == "✅ Reset!"  # the reset is confirmed on the button, with the sound


def test_reset_always_asks_and_confirms_on_the_button(gui, monkeypatch):
    """Owner: Reset to Default gave no feedback (it looked enabled while it was disabled) and never asked."""
    import qt_widget_helpers

    beeps = []
    monkeypatch.setattr(qt_widget_helpers, "saved_beep", lambda: beeps.append(1))
    dialog = gui._open_placeholder_text_dialog()
    editor, buttons = dialog.placeholder_editor, dialog.placeholder_buttons
    reset = buttons["reset"]
    default = editor.toPlainText()
    assert dialog.placeholder_status.text() == "Default text"
    assert reset.isEnabled() and buttons["reset_all"].isEnabled()  # never a dead button

    # nothing edited: it still asks, and still answers
    saves = gui.saves
    reset.click()
    assert len(gui.answers["asked"]) == 1 and "{" + EDITABLE[0] + "}" in gui.answers["asked"][0]
    assert reset.text() == "✅ Reset!" and beeps == [1]
    assert qt_widget_helpers.SAVED_FLASH_COLOR in reset.styleSheet()
    assert gui.saves == saves and CONFIG_KEY not in gui.config  # nothing was stored: nothing is written

    # an unsaved change: No keeps it, Yes drops it
    editor.setPlainText("draft text")
    gui.answers["value"] = False
    reset.click()
    assert editor.toPlainText() == "draft text" and beeps == [1] and len(gui.answers["asked"]) == 2
    gui.answers["value"] = True
    reset.click()
    assert editor.toPlainText() == default and beeps == [1, 1]
    assert dialog.placeholder_status.text() == "Default text"
    assert gui.saves == saves


def test_reset_all_clears_every_stored_text_and_unsaved_change(gui, eg, monkeypatch):
    import qt_widget_helpers

    beeps = []
    monkeypatch.setattr(qt_widget_helpers, "saved_beep", lambda: beeps.append(1))
    dialog = gui._open_placeholder_text_dialog()
    combo, editor, buttons = dialog.placeholder_combo, dialog.placeholder_editor, dialog.placeholder_buttons
    row = dialog.layout().itemAt(dialog.layout().count() - 1).layout()
    assert [row.itemAt(i).widget() for i in range(row.count()) if row.itemAt(i).widget() is not None] == [
        buttons["save"], buttons["reset"], buttons["reset_all"], buttons["close"]]
    assert buttons["reset_all"].text() == "Reset All to Defaults"
    defaults = {item["name"]: item["default"] for item in eg.editable_placeholders()}

    for name in ("relations_instruction", "example"):
        combo.setCurrentIndex(EDITABLE.index(name))
        editor.setPlainText(f"my {name}")
        buttons["save"].click()
    combo.setCurrentIndex(EDITABLE.index("gender_instruction"))
    editor.setPlainText("an unsaved change")
    combo.setCurrentIndex(EDITABLE.index("example"))
    assert gui.config[CONFIG_KEY] == {"relations_instruction": "my relations_instruction", "example": "my example"}
    beeps.clear()

    gui.answers["value"] = False  # No: nothing changes
    buttons["reset_all"].click()
    assert f"all {len(EDITABLE)} placeholder texts" in gui.answers["asked"][-1]
    assert len(gui.config[CONFIG_KEY]) == 2 and editor.toPlainText() == "my example" and beeps == []
    assert buttons["reset_all"].text() == "Reset All to Defaults"

    gui.answers["value"] = True
    saves = gui.saves
    buttons["reset_all"].click()
    assert gui.config[CONFIG_KEY] == {} and json.loads(os.environ[ENV]) == {}
    assert gui.saves == saves + 1
    assert buttons["reset_all"].text() == "✅ All Reset!" and beeps == [1]
    assert editor.toPlainText() == defaults["example"] and dialog.placeholder_status.text() == "Default text"
    for name in EDITABLE:  # every placeholder shows its default again, the unsaved one too
        combo.setCurrentIndex(EDITABLE.index(name))
        assert editor.toPlainText() == defaults[name], name
        assert dialog.placeholder_status.text() == "Default text"
    assert eg.placeholder_text("relations_instruction", "x") == "x"

    # nothing left to reset: it still asks and answers, and writes nothing
    saves = gui.saves
    buttons["reset_all"].click()
    assert gui.saves == saves and beeps == [1, 1]
    gui.answers["value"] = True
    dialog.placeholder_buttons["close"].click()  # no unsaved change is left: Close does not ask
    assert not dialog.isVisible()


def test_switching_keeps_unsaved_drafts_and_close_asks(gui):
    dialog = gui._open_placeholder_text_dialog()
    combo, editor = dialog.placeholder_combo, dialog.placeholder_editor
    combo.setCurrentIndex(EDITABLE.index("example"))
    editor.setPlainText("draft text")
    combo.setCurrentIndex(EDITABLE.index("gender_instruction"))
    combo.setCurrentIndex(EDITABLE.index("example"))
    assert editor.toPlainText() == "draft text"
    assert dialog.placeholder_status.text() == "Default text (unsaved changes)"
    assert CONFIG_KEY not in gui.config

    gui.answers["value"] = False  # "Discard the unsaved changes?" -> No: the dialog stays open
    dialog.placeholder_buttons["close"].click()
    assert dialog.isVisible()
    assert editor.toPlainText() == "draft text"
    gui.answers["value"] = True
    dialog.placeholder_buttons["close"].click()
    assert not dialog.isVisible()
    assert CONFIG_KEY not in gui.config

    again = gui._open_placeholder_text_dialog()
    assert again is dialog
    assert editor.toPlainText() != "draft text"  # drafts end at Close


# =============================================================================================
# Minimal: {gender_instruction}, and its own "Include Relations Column" toggle
# =============================================================================================

def _minimal_env(**extra):
    env = {"GLOSSARY_TARGET_LANGUAGE": "English", "GLOSSARY_INCLUDE_BOOK_TITLE": "0",
           "GLOSSARY_INCLUDE_DESCRIPTION": "1"}
    env.update({key: str(value) for key, value in extra.items()})
    return env


def test_minimal_gender_instruction_follows_the_entry_types_and_the_stored_text():
    assert GENDER_EXPANDED in minimal_prompt(_minimal_env())
    none = minimal_prompt(_minimal_env(GLOSSARY_CUSTOM_ENTRY_TYPES=json.dumps(NO_GENDER_TYPES)))
    assert "{gender_instruction}" not in none and "The gender column applies only to" not in none
    stored = json.dumps({"gender_instruction": "Gender for {gender_types} only."})
    own = minimal_prompt(_minimal_env(**{ENV: stored}))
    assert "description\nGender for character, titles, nicknames only.\n" in own
    assert "The gender column applies only to" not in own


def test_balanced_language_is_filled_in_the_built_in_prompt_and_custom_prompts_keep_working():
    built_in = balanced_prompt(dict(scenarios()["balanced_desc0_gender0_rel0"][1], GLOSSARY_TARGET_LANGUAGE="German"))
    assert "{language}" not in built_in and "must be in German, While the raw name" in built_in
    custom = balanced_prompt(dict(scenarios()["balanced_desc0_gender0_rel0"][1], GLOSSARY_TARGET_LANGUAGE="German",
                                  GLOSSARY_SYSTEM_PROMPT="Translate into {language}. {fields}"))
    assert custom.startswith("Translate into German.")


RELATIONS_RESPONSE = ("type,raw_name,translated_name,gender,description\n"
                      "character,서윤,Seo-yun,female,Heir,도현: Older Brother\n"
                      "character,도현,Do-hyeon,male,Captain,서윤: Younger Sister\n")


@pytest.mark.parametrize("tracker, toggle", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_minimal_asks_and_detaches_relations_only_with_tracker_and_toggle(tracker, toggle, tmp_path):
    cells = []
    system = minimal_prompt(_minimal_env(GLOSSARY_ENABLE_RELATIONS_TRACKER=tracker,
                                         GLOSSARY_MINIMAL_INCLUDE_RELATIONS=toggle),
                            response=RELATIONS_RESPONSE, output_dir=str(tmp_path), cells_out=cells)
    both = bool(tracker and toggle)
    assert ("The `relations` column:" in system) is both
    assert ("Add a `relations` column as one extra LAST column of every row." in system) is both
    assert "{relations_instruction}" not in system
    if not both:
        # the line is removed exactly as with the tracker off
        off = minimal_prompt(_minimal_env(GLOSSARY_ENABLE_RELATIONS_TRACKER=0))
        assert system == off
    assert bool(cells) is both


def test_minimal_records_relations_only_with_the_toggle_on(tmp_path, monkeypatch):
    import GlossaryManager as GM

    chapters = [{"body": "<p>서윤은 도현을 보았다.</p>", "num": 1}]
    for toggle, expected in (("0", False), ("1", True)):
        out = tmp_path / f"out{toggle}"
        out.mkdir()
        monkeypatch.setenv("GLOSSARY_ENABLE_RELATIONS_TRACKER", "1")
        monkeypatch.setenv("GLOSSARY_MINIMAL_INCLUDE_RELATIONS", toggle)
        GM._minimal_relations_cells[GM._minimal_relations_key(str(out))] = [("서윤", "Seo-yun", "도현: Older Brother")]
        recorded = GM._record_minimal_relations(str(out), chapters)
        assert bool(recorded) is expected


if __name__ == "__main__":  # print the digests (used to record BASELINE_DIGESTS before the change)
    for _name, _text in capture_prompts().items():
        print(f"    {_name!r}: {digest(_text)!r},")
