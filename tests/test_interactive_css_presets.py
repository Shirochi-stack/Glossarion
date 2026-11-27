"""Interactive styles: the 12 built-in presets are scoped under .glx-, CSS-only and degrade without
animation; scope_css rewrites and cleans; user presets; the preview document (GUI-free)."""
import re
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import interactive_css_presets as presets  # noqa: E402

EXPECTED = ('Classic Book', 'Dark Fantasy', 'Cyberpunk Neon', 'LitRPG Status Window', 'Romance Pastel', 'Wuxia Ink',
            'Gothic Horror', 'Sci-Fi HUD', 'Minimal Modern', 'Royal Court', 'Cozy Slice-of-Life', 'Retro Terminal')


def _selectors(css):
    """Every selector of the stylesheet (keyframe frame selectors excluded)."""
    text = re.sub(r'/\*.*?\*/', '', css, flags=re.S)
    out = []
    for prelude, body in presets._parse_blocks(text):
        if prelude.startswith('@keyframes'):
            continue
        if prelude.startswith('@media'):
            out.extend(_selectors(body))
            continue
        out.extend(s.strip() for s in presets._split_top(prelude, ','))
    return out


def test_twelve_builtin_presets():
    assert presets.BUILTIN_PRESET_NAMES == EXPECTED
    assert set(presets.BUILTIN_PRESETS) == set(EXPECTED)


@pytest.mark.parametrize("name", EXPECTED)
def test_builtin_preset_is_scoped_and_css_only(name):
    css = presets.BUILTIN_PRESETS[name]
    assert presets.css_problems(css) == []                       # scope_css has nothing to change
    scoped, warnings = presets.scope_css(css)
    assert warnings == [] and presets.scope_css(scoped) == (scoped, [])
    for selector in _selectors(css):
        assert selector.startswith('.glx-'), selector
    for keyframes in re.findall(r'@keyframes\s+([\w-]+)', css):
        assert keyframes.startswith('glx-'), keyframes
    lowered = css.lower()
    for banned in ('url(http', '@import', 'javascript:', 'expression(', '<script'):
        assert banned not in lowered
    # animations only inside the reduced-motion query; the static look is complete without them
    outside = re.sub(r'@media \(prefers-reduced-motion: no-preference\) \{.*\}\s*$', '', css, flags=re.S)
    assert 'animation' not in outside
    assert '@media (prefers-reduced-motion: no-preference)' in css
    for cls in ('glx-status-window', 'glx-system-message', 'glx-dialogue', 'glx-figure', 'glx-illustration',
                'glx-reveal'):
        assert cls in presets.defined_classes(css)
    assert '.glx-status-window tr:hover td' in css and 'nth-child(even)' in css
    assert 'glx-book' not in presets.assignable_classes(css) and 'glx-dialogue' in presets.assignable_classes(css)


def test_scope_css_rewrites_unscoped_rules_and_strips_unsafe_parts():
    css, warnings = presets.scope_css(
        '@import url(evil.css); body { color: red; background: url(http://x/y.png); } '
        'p, .glx-a > b { color: blue } html body h1 { x: y } div { behavior: url(x.htc); width: expression(1) } '
        'a { color: javascript:alert(1) } @keyframes spin { from { opacity: 0 } to { opacity: 1 } } '
        '.x { animation: spin 1s } @font-face { font-family: z; src: url(z.ttf) } '
        '@media (max-width: 600px) { p { margin: 0 } }')
    assert '@import' not in css and 'http' not in css and 'expression' not in css and 'behavior' not in css
    assert 'javascript' not in css and '@font-face' not in css
    assert '.glx-book {\n  color: red;\n}' in css
    assert '.glx-book p, .glx-a > b {' in css
    assert '.glx-book h1 {' in css
    assert '@keyframes glx-spin' in css and 'animation: glx-spin 1s' in css
    assert '@media (max-width: 600px) {\n  .glx-book p {' in css
    assert presets.css_problems(css) == []
    assert any('glx-spin' in w for w in warnings) and any('@import' in w for w in warnings)


def test_extract_css_block():
    assert presets.extract_css_block("Sure!\n```css\n.glx-a { x: y }\n```\nmore") == ".glx-a { x: y }"
    assert presets.extract_css_block("```\n.glx-b {}\n```") == ".glx-b {}"
    assert presets.extract_css_block(".glx-c {}") == ".glx-c {}"


def test_user_presets_are_stored_scoped_and_listed_after_builtins():
    config = {}
    name, warnings = presets.save_user_preset(config, 'Pirate', 'p { color: brown }')
    assert name == 'Pirate' and warnings
    assert presets.user_presets(config)['Pirate'].startswith('.glx-book p {')
    assert presets.preset_names(config) == list(EXPECTED) + ['Pirate']
    assert presets.unique_preset_name(config, 'Pirate') == 'Pirate 2'
    with pytest.raises(ValueError):
        presets.save_user_preset(config, 'Classic Book', '.glx-a {}')
    config[presets.PRESET_KEY] = 'Pirate'
    assert presets.active_preset_name(config) == 'Pirate'
    assert presets.delete_user_preset(config, 'Pirate')
    assert presets.active_preset_name(config) == presets.DEFAULT_PRESET


def test_book_stylesheet_is_its_own_file(tmp_path):
    path, warnings = presets.write_book_stylesheet(str(tmp_path), presets.BUILTIN_PRESETS['Sci-Fi HUD'])
    assert path == str(tmp_path / 'Interactive_Images' / 'glossarion_interactive.css') and warnings == []
    assert presets.read_book_stylesheet(str(tmp_path)).startswith('/*') is False  # comments are not kept
    assert '.glx-status-window' in presets.read_book_stylesheet(str(tmp_path))
    assert presets.remove_book_stylesheet(str(tmp_path)) and not Path(path).exists()


def test_preview_document_has_every_sample_part_and_no_script():
    html = presets.preview_document(presets.BUILTIN_PRESETS['LitRPG Status Window'])
    assert '<script' not in html.lower()
    for part in ('<h1>', '<h2>', 'class="glx-dialogue"', 'class="glx-status-window"', '<ul>', '<blockquote>',
                 'class="glx-reveal"', '<details><summary>', 'class="glx-scene-break"', 'class="glx-letter"',
                 'class="glx-figure"', 'data:image/svg+xml;base64,', 'class="glx-system-message"'):
        assert part in html, part
    assert html.count('<tr>') == 7  # header + 6 rows


def test_preset_names_for_new_and_duplicate():
    config = {}
    assert presets.preset_name_for_request(config, "  cozy   rainy night in the city ") == "Cozy Rainy Night In"
    assert presets.preset_name_for_request(config, "") == "My Style"
    assert presets.duplicate_name(config, "Classic Book") == "Classic Book copy"
    presets.save_user_preset(config, "Classic Book copy", ".glx-book p { color: red; }")
    assert presets.duplicate_name(config, "Classic Book") == "Classic Book copy 2"