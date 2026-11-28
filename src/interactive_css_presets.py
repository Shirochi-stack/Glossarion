"""Interactive styles for translated chapters: built-in CSS presets, scoping, user presets, preview.

GUI-free (desktop and mobile). Every rule of a book's interactive stylesheet is scoped under a
``.glx-`` class, so it can only style what the Interactive Images step marked (``glx-book`` on the
chapter ``<body>``, ``glx-*`` classes on paragraphs, the ``glx-figure`` images). CSS only: EPUB
readers strip scripts. Animations are keyframes inside ``@media (prefers-reduced-motion:
no-preference)`` and the static look is complete without them.

The stylesheet lives at ``<output>/Interactive_Images/glossarion_interactive.css``, a separate file
from the Attach CSS override (``epub_css_override_path``), so clearing one never touches the other.
The EPUB builder links it after every other stylesheet of a chapter.
"""

import base64
import os
import re
import tempfile

INTERACTIVE_FOLDER = 'Interactive_Images'
STYLESHEET_FILENAME = 'glossarion_interactive.css'

#: Classes the presets style and the planner may assign (``{css_classes}`` lists those a stylesheet defines).
CLASS_VOCABULARY = (
    'glx-book',
    'glx-figure',
    'glx-illustration',
    'glx-status-window',
    'glx-system-message',
    'glx-dialogue',
    'glx-thought',
    'glx-scene-break',
    'glx-letter',
    'glx-flashback',
    'glx-sfx',
    'glx-reveal',
    'glx-glow',
)

#: Classes Glossarion places itself (never offered to the planner).
STRUCTURAL_CLASSES = frozenset({'glx-book', 'glx-figure', 'glx-illustration'})

USER_PRESETS_KEY = 'interactive_css_user_presets'
PRESET_KEY = 'interactive_css_preset'
ENABLED_KEY = 'interactive_css_enabled'
DEFAULT_PRESET = 'Classic Book'


# ---------------------------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------------------------

def _preset(p):
    """One preset's stylesheet from its palette ``p`` (dict); ``p['extra']`` is appended as is and
    ``p['motion']`` goes inside the reduced-motion media query."""
    css = f"""/* Glossarion interactive style: {p['name']} */
.glx-book {{
  font-family: {p['body_font']};
  color: {p['text']};
  background-color: {p['page']};
  line-height: {p.get('line_height', '1.6')};
}}
.glx-book h1, .glx-book h2, .glx-book h3 {{
  font-family: {p['heading_font']};
  color: {p['heading']};
  line-height: 1.25;
}}
.glx-book h1 {{
  text-align: center;
  {p.get('h1', '')}
}}
.glx-book p {{
  margin: 0 0 0.7em 0;
  {p.get('p', '')}
}}
.glx-book blockquote {{
  margin: 1em 1.5em;
  padding: 0.4em 1em;
  border-left: 3px solid {p['accent']};
  color: {p['muted']};
  {p.get('blockquote', '')}
}}
.glx-book ul, .glx-book ol {{
  margin: 0.6em 0 0.9em 1.6em;
  padding: 0;
}}
.glx-book li {{
  margin: 0.2em 0;
}}
.glx-book li::marker {{
  color: {p['accent']};
}}
.glx-book details {{
  margin: 0.9em 0;
  padding: 0.5em 0.9em;
  border: 1px solid {p['rule']};
  border-radius: {p.get('radius', '6px')};
  background-color: {p['panel']};
}}
.glx-book summary {{
  cursor: pointer;
  font-weight: bold;
  color: {p['heading']};
}}
.glx-figure {{
  margin: 1.2em auto;
  text-align: center;
  page-break-inside: avoid;
  break-inside: avoid;
}}
.glx-illustration {{
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
  border-radius: {p.get('radius', '6px')};
  {p.get('illustration', '')}
}}
.glx-status-window {{
  width: 100%;
  margin: 1em 0;
  border-collapse: collapse;
  font-family: {p.get('status_font', p['body_font'])};
  font-size: 0.92em;
  color: {p.get('status_text', p['text'])};
  background-color: {p['panel']};
  border: {p.get('status_border', '1px solid ' + p['rule'])};
  {p.get('status_window', '')}
}}
.glx-status-window th, .glx-status-window td {{
  padding: 0.35em 0.7em;
  text-align: left;
  border-bottom: 1px solid {p['rule']};
}}
.glx-status-window th {{
  color: {p['heading']};
  background-color: {p.get('status_head', p['panel'])};
}}
.glx-status-window tr:nth-child(even) td {{
  background-color: {p['stripe']};
}}
.glx-status-window tr:hover td {{
  background-color: {p['hover']};
}}
.glx-system-message {{
  margin: 0.9em 0;
  padding: 0.45em 0.8em;
  font-family: {p.get('status_font', p['body_font'])};
  color: {p['system']};
  background-color: {p['panel']};
  border-left: 3px solid {p['system']};
  {p.get('system_message', '')}
}}
.glx-dialogue {{
  {p.get('dialogue', 'padding-left: 0.6em; border-left: 2px solid ' + p['rule'] + ';')}
}}
.glx-thought {{
  font-style: italic;
  color: {p['muted']};
}}
.glx-scene-break {{
  text-align: center;
  letter-spacing: 0.5em;
  color: {p['accent']};
  margin: 1.4em 0;
  {p.get('scene_break', '')}
}}
.glx-letter {{
  margin: 1em 1.2em;
  padding: 0.8em 1em;
  background-color: {p['panel']};
  border: 1px solid {p['rule']};
  {p.get('letter', 'font-style: italic;')}
}}
.glx-flashback {{
  color: {p['muted']};
  {p.get('flashback', 'font-style: italic;')}
}}
.glx-sfx {{
  font-weight: bold;
  letter-spacing: 0.08em;
  color: {p['accent']};
  {p.get('sfx', '')}
}}
.glx-reveal {{
  color: {p['heading']};
}}
.glx-glow {{
  color: {p['system']};
  text-shadow: 0 0 6px {p['glow']};
}}
"""
    if p.get('extra'):
        css += p['extra'].strip() + "\n"
    css += """@keyframes glx-shimmer {
  0% { opacity: 0.75; }
  50% { opacity: 1; }
  100% { opacity: 0.75; }
}
@keyframes glx-glow-pulse {
  0% { text-shadow: 0 0 2px @GLOW@; }
  50% { text-shadow: 0 0 12px @GLOW@; }
  100% { text-shadow: 0 0 2px @GLOW@; }
}
@keyframes glx-reveal {
  from { clip-path: inset(0 100% 0 0); }
  to { clip-path: inset(0 0 0 0); }
}
@media (prefers-reduced-motion: no-preference) {
  .glx-reveal {
    animation: glx-reveal 2.4s steps(32, end) 1 both;
  }
  .glx-glow {
    animation: glx-glow-pulse 2.6s ease-in-out infinite;
  }
@MOTION@}
""".replace('@GLOW@', p['glow']).replace('@MOTION@', p.get('motion', ''))
    return css


_PRESET_SPECS = (
    {
        'name': 'Classic Book',
        'body_font': "Georgia, 'Times New Roman', serif", 'heading_font': "Georgia, 'Times New Roman', serif",
        'text': '#2b2620', 'page': '#fbf8f1', 'heading': '#3b2f22', 'accent': '#8a6d3b', 'muted': '#5f574c',
        'panel': '#f3ede0', 'rule': '#d9ccb2', 'stripe': '#ece4d2', 'hover': '#e4d7bd', 'system': '#6b4f2a',
        'glow': '#c9a86a',
        'h1': 'font-variant: small-caps; letter-spacing: 0.06em; font-weight: normal;',
        'p': 'text-align: justify; text-indent: 1.4em;',
        'scene_break': 'font-size: 0.9em;',
        'extra': ".glx-scene-break::before { content: '\\2766  '; }\n"
                 ".glx-scene-break::after { content: '  \\2766'; }",
    },
    {
        'name': 'Dark Fantasy',
        'body_font': "Georgia, 'Palatino Linotype', serif", 'heading_font': "'Palatino Linotype', Georgia, serif",
        'text': '#ddd6c8', 'page': '#1c1b1e', 'heading': '#d8b45c', 'accent': '#b8863b', 'muted': '#a79f92',
        'panel': '#29272c', 'rule': '#5a4a32', 'stripe': '#302d33', 'hover': '#3d3528', 'system': '#f0a15a',
        'glow': '#ff8c3a',
        'h1': 'letter-spacing: 0.08em; border-bottom: 1px solid #5a4a32; padding-bottom: 0.3em;',
        'status_border': '2px double #b8863b',
        'status_head': '#33291c',
        'system_message': 'text-shadow: 0 0 4px #ff8c3a;',
        'motion': "  .glx-system-message {\n    animation: glx-glow-pulse 3s ease-in-out infinite;\n  }\n",
    },
    {
        'name': 'Cyberpunk Neon',
        'body_font': "'Segoe UI', Roboto, Arial, sans-serif", 'heading_font': "'Segoe UI', Roboto, Arial, sans-serif",
        'status_font': "Consolas, 'Courier New', monospace",
        'text': '#d9e6f2', 'page': '#0b0b12', 'heading': '#ff4fd8', 'accent': '#18e4ff', 'muted': '#9aa7c7',
        'panel': '#141425', 'rule': '#2d2d55', 'stripe': '#191932', 'hover': '#26264d', 'system': '#18e4ff',
        'glow': '#ff4fd8', 'radius': '2px',
        'h1': 'text-transform: uppercase; letter-spacing: 0.12em; text-shadow: 0 0 8px #ff4fd8;',
        'status_border': '1px solid #18e4ff',
        'status_window': 'box-shadow: 0 0 10px rgba(24, 228, 255, 0.35);',
        'illustration': 'border: 1px solid #ff4fd8; box-shadow: 0 0 12px rgba(255, 79, 216, 0.4);',
        'motion': "  .glx-book h1, .glx-book h2 {\n    animation: glx-shimmer 1.8s steps(3, end) infinite;\n  }\n",
    },
    {
        'name': 'LitRPG Status Window',
        'body_font': "'Segoe UI', Roboto, Arial, sans-serif", 'heading_font': "'Segoe UI', Roboto, Arial, sans-serif",
        'status_font': "'Segoe UI', Roboto, Arial, sans-serif",
        'text': '#1f2a37', 'page': '#f7f9fc', 'heading': '#1d4f91', 'accent': '#2f7de1', 'muted': '#4b5a6b',
        'panel': 'rgba(32, 92, 190, 0.08)', 'rule': '#9cc0ee', 'stripe': 'rgba(47, 125, 225, 0.10)',
        'hover': 'rgba(47, 125, 225, 0.22)', 'system': '#1f6fd6', 'glow': '#58a6ff',
        'status_text': '#10305c', 'status_head': 'rgba(47, 125, 225, 0.18)',
        'status_border': '1px solid #5b9be8',
        'status_window': 'border-radius: 8px; box-shadow: 0 2px 10px rgba(47, 125, 225, 0.25);',
        'system_message': 'font-weight: bold; border-radius: 6px;',
        'extra': ".glx-system-message::before { content: '\\25C6  '; color: #2f7de1; }\n"
                 ".glx-status-window td + td { font-weight: bold; color: #1d4f91; }",
        'motion': "  .glx-system-message {\n    animation: glx-glow-pulse 2.4s ease-in-out infinite;\n  }\n",
    },
    {
        'name': 'Romance Pastel',
        'body_font': "Georgia, 'Segoe UI', serif", 'heading_font': "'Brush Script MT', 'Segoe Script', cursive",
        'text': '#4a3a44', 'page': '#fff7fa', 'heading': '#b4588a', 'accent': '#c98bb9', 'muted': '#7b6672',
        'panel': '#f7ecf6', 'rule': '#e8c9de', 'stripe': '#f3e3f0', 'hover': '#ecd3e8', 'system': '#9a6fbf',
        'glow': '#e9a8d3', 'radius': '14px',
        'h1': 'font-weight: normal; font-size: 2.1em;',
        'dialogue': 'padding: 0.35em 0.8em; background-color: #fdeef5; border-radius: 12px;',
        'motion': "  .glx-figure {\n    animation: glx-shimmer 4s ease-in-out 1;\n  }\n",
    },
    {
        'name': 'Wuxia Ink',
        'body_font': "'Palatino Linotype', Georgia, serif", 'heading_font': "'Palatino Linotype', Georgia, serif",
        'text': '#1f1b16', 'page': '#f3ead7', 'heading': '#111111', 'accent': '#b3261e', 'muted': '#4e463b',
        'panel': '#ebe0c8', 'rule': '#c9b892', 'stripe': '#e5d8bb', 'hover': '#dccba6', 'system': '#7a1c16',
        'glow': '#d4574e',
        'h1': 'font-weight: bold; letter-spacing: 0.2em; border-bottom: 2px solid #111111;',
        'blockquote': 'border-left: 4px double #111111;',
        'scene_break': 'font-size: 1.1em;',
        'extra': ".glx-scene-break::before { content: '\\25A0  '; color: #b3261e; }\n"
                 ".glx-scene-break::after { content: '  \\25A0'; color: #b3261e; }",
    },
    {
        'name': 'Gothic Horror',
        'body_font': "Georgia, 'Times New Roman', serif", 'heading_font': "'Times New Roman', Georgia, serif",
        'text': '#d6cfcf', 'page': '#0e0c0c', 'heading': '#b3122a', 'accent': '#8f0f22', 'muted': '#9c9090',
        'panel': '#1a1515', 'rule': '#4a1c22', 'stripe': '#201919', 'hover': '#341a1e', 'system': '#d6354f',
        'glow': '#ff2040',
        'h1': 'letter-spacing: 0.1em; text-shadow: 1px 1px 0 #000000, 0 0 6px #5a0010;',
        'letter': 'font-style: italic; background-color: #e8dcc4; color: #2a2118; border: 1px solid #8b7a5a;',
        'sfx': 'text-transform: uppercase;',
        'extra': "@keyframes glx-flicker {\n  0% { opacity: 1; }\n  45% { opacity: 0.55; }\n  50% { opacity: 1; }\n"
                 "  80% { opacity: 0.8; }\n  100% { opacity: 1; }\n}",
        'motion': "  .glx-sfx {\n    animation: glx-flicker 1.6s steps(4, end) infinite;\n  }\n",
    },
    {
        'name': 'Sci-Fi HUD',
        'body_font': "'Segoe UI', Roboto, Arial, sans-serif", 'heading_font': "'Segoe UI', Roboto, Arial, sans-serif",
        'status_font': "Consolas, 'Courier New', monospace",
        'text': '#cfe3e8', 'page': '#121a20', 'heading': '#4fd1c5', 'accent': '#38b2ac', 'muted': '#8aa4ab',
        'panel': '#16232b', 'rule': '#24434d', 'stripe': '#1a2a33', 'hover': '#20404a', 'system': '#4fd1c5',
        'glow': '#4fd1c5', 'radius': '0',
        'h1': 'text-transform: uppercase; letter-spacing: 0.18em;',
        'status_border': '1px solid #38b2ac',
        'status_window': 'outline: 1px solid #24434d; outline-offset: 3px;',
        'system_message': 'text-transform: uppercase; letter-spacing: 0.05em;',
        'motion': "  .glx-system-message {\n    animation: glx-reveal 2s steps(28, end) 1 both;\n  }\n",
    },
    {
        'name': 'Minimal Modern',
        'body_font': "'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
        'heading_font': "'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
        'text': '#222222', 'page': '#ffffff', 'heading': '#111111', 'accent': '#3d7ea6', 'muted': '#666666',
        'panel': '#f6f7f8', 'rule': '#e2e4e7', 'stripe': '#f2f3f5', 'hover': '#e8eef3', 'system': '#3d7ea6',
        'glow': '#9cc3db', 'line_height': '1.75', 'radius': '4px',
        'h1': 'font-weight: 600; letter-spacing: 0.01em;',
        'p': 'margin-bottom: 1em;',
    },
    {
        'name': 'Royal Court',
        'body_font': "Garamond, Georgia, serif", 'heading_font': "Garamond, 'Palatino Linotype', serif",
        'text': '#2a2233', 'page': '#fbf7ee', 'heading': '#4b2367', 'accent': '#b8912f', 'muted': '#5d5366',
        'panel': '#f4eddc', 'rule': '#cdb06a', 'stripe': '#efe5cd', 'hover': '#e6d7b0', 'system': '#5b2c80',
        'glow': '#d9b45a',
        'h1': 'border-top: 3px double #b8912f; border-bottom: 3px double #b8912f; padding: 0.3em 0;',
        'status_border': '3px double #b8912f',
        'letter': 'font-style: normal; border: 3px double #b8912f; text-align: center;',
        'extra': ".glx-book h1 + p::first-letter { float: left; font-size: 3.1em; line-height: 0.9;"
                 " padding-right: 0.08em; color: #4b2367; }",
    },
    {
        'name': 'Cozy Slice-of-Life',
        'body_font': "'Segoe UI', Verdana, sans-serif", 'heading_font': "'Comic Sans MS', 'Segoe Print', cursive",
        'text': '#3d3a33', 'page': '#fffaf0', 'heading': '#c0703a', 'accent': '#e2a35d', 'muted': '#7a7366',
        'panel': '#fff1dc', 'rule': '#f0d2a8', 'stripe': '#fde9cc', 'hover': '#f8dcb3', 'system': '#b5652c',
        'glow': '#f4b76b', 'radius': '12px',
        'status_window': 'border-radius: 12px; box-shadow: 0 3px 8px rgba(160, 110, 50, 0.18);',
        'letter': 'font-style: normal; border-radius: 12px; box-shadow: 0 2px 6px rgba(160, 110, 50, 0.2);',
        'scene_break': 'letter-spacing: 0.9em;',
        'extra': ".glx-scene-break::before { content: '~ '; }\n.glx-scene-break::after { content: ' ~'; }",
    },
    {
        'name': 'Retro Terminal',
        'body_font': "Consolas, 'Courier New', monospace", 'heading_font': "Consolas, 'Courier New', monospace",
        'text': '#7dff8a', 'page': '#050805', 'heading': '#a6ffb0', 'accent': '#3ddc4f', 'muted': '#5bbf66',
        'panel': '#0b140c', 'rule': '#1f4a24', 'stripe': '#0e1c10', 'hover': '#16331a', 'system': '#b8ffc1',
        'glow': '#3ddc4f', 'radius': '0',
        'h1': 'text-transform: uppercase;',
        'status_window': 'background-image: repeating-linear-gradient(0deg, rgba(61, 220, 79, 0.06) 0px, '
                         'rgba(61, 220, 79, 0.06) 1px, transparent 1px, transparent 3px);',
        'extra': ".glx-system-message::after { content: ' \\258C'; }\n"
                 "@keyframes glx-blink {\n  0% { opacity: 1; }\n  50% { opacity: 0; }\n  100% { opacity: 1; }\n}",
        'motion': "  .glx-system-message::after {\n    animation: glx-blink 1s steps(1, end) infinite;\n  }\n",
    },
    # ---- fantasy ----------------------------------------------------------------------------------
    {
        'name': 'Arcane Runes',
        'body_font': "Georgia, 'Palatino Linotype', serif", 'heading_font': "'Palatino Linotype', Georgia, serif",
        'status_font': "'Segoe UI', Roboto, Arial, sans-serif",
        'text': '#e3dcf5', 'page': '#150f26', 'heading': '#c9a8ff', 'accent': '#8f6bff', 'muted': '#a99cc9',
        'panel': '#211a3a', 'rule': '#4a3b7a', 'stripe': '#271f46', 'hover': '#34285e', 'system': '#7fe3ff',
        'glow': '#8f6bff', 'radius': '8px',
        'h1': 'letter-spacing: 0.1em; text-shadow: 0 0 10px #8f6bff;',
        'status_border': '1px solid #8f6bff',
        'status_window': 'box-shadow: 0 0 14px rgba(143, 107, 255, 0.45), inset 0 0 18px rgba(143, 107, 255, 0.12);',
        'system_message': 'border-radius: 6px; box-shadow: 0 0 8px rgba(127, 227, 255, 0.35);',
        'illustration': 'border: 1px solid #4a3b7a; box-shadow: 0 0 16px rgba(143, 107, 255, 0.4);',
        'extra': ".glx-book h1::before { content: '\\16B1  '; color: #7fe3ff; }\n"
                 ".glx-book h1::after { content: '  \\16B1'; color: #7fe3ff; }\n"
                 ".glx-scene-break::before { content: '\\16A0  \\16A2  \\16A6'; }\n"
                 "@keyframes glx-rune-pulse {\n  0% { opacity: 0.55; letter-spacing: 0.3em; }\n"
                 "  50% { opacity: 1; letter-spacing: 0.5em; }\n  100% { opacity: 0.55; letter-spacing: 0.3em; }\n}",
        'motion': "  .glx-scene-break {\n    animation: glx-rune-pulse 4.5s ease-in-out infinite;\n  }\n"
                  "  .glx-system-message {\n    animation: glx-glow-pulse 3s ease-in-out infinite;\n  }\n",
    },
    {
        'name': "Dragon's Hoard",
        'body_font': "Georgia, 'Times New Roman', serif", 'heading_font': "'Palatino Linotype', Georgia, serif",
        'text': '#eadfc8', 'page': '#1d130d', 'heading': '#f2c14e', 'accent': '#d9922b', 'muted': '#b8a888',
        'panel': '#2b1d12', 'rule': '#6b4a1c', 'stripe': '#332315', 'hover': '#46301a', 'system': '#ffb347',
        'glow': '#ffcf5c',
        'h1': 'letter-spacing: 0.07em; text-shadow: 0 1px 0 #6b4a1c, 0 0 10px #d9922b;',
        'status_border': '2px solid #d9922b',
        'status_head': '#3a2812',
        'status_window': 'box-shadow: 0 0 12px rgba(217, 146, 43, 0.4);',
        'system_message': 'border-left-width: 5px; text-shadow: 0 0 5px #d9922b;',
        'illustration': 'border: 2px solid #d9922b; box-shadow: 0 0 14px rgba(255, 207, 92, 0.35);',
        'blockquote': 'border-left: 4px solid #d9922b;',
        'extra': ".glx-scene-break::before { content: '\\25C6  '; }\n.glx-scene-break::after { content: '  \\25C6'; }\n"
                 "@keyframes glx-gold-gleam {\n  0% { text-shadow: 0 0 3px #d9922b; }\n"
                 "  50% { text-shadow: 0 0 14px #ffcf5c, 0 0 3px #fff2c0; }\n  100% { text-shadow: 0 0 3px #d9922b; }\n}",
        'motion': "  .glx-book h1 {\n    animation: glx-gold-gleam 3.5s ease-in-out infinite;\n  }\n"
                  "  .glx-system-message {\n    animation: glx-gold-gleam 3s ease-in-out infinite;\n  }\n",
    },
    {
        'name': 'Elven Grove',
        'body_font': "Garamond, Georgia, serif", 'heading_font': "Garamond, 'Palatino Linotype', serif",
        'text': '#24322a', 'page': '#f1f6ea', 'heading': '#2f6b45', 'accent': '#5f9e6e', 'muted': '#566a5c',
        'panel': '#e3eedb', 'rule': '#b7d0b0', 'stripe': '#dce9d3', 'hover': '#cfe2c6', 'system': '#2f7a55',
        'glow': '#9fd8a8', 'radius': '16px', 'line_height': '1.7',
        'h1': 'font-weight: normal; letter-spacing: 0.05em; font-style: italic;',
        'dialogue': 'padding: 0.3em 0.9em; background-color: #e9f2e2; border-radius: 14px;',
        'status_window': 'border-radius: 14px;',
        'system_message': 'border-radius: 12px;',
        'illustration': 'border: 3px solid #b7d0b0;',
        'extra': ".glx-scene-break::before { content: '\\2767  '; }\n.glx-scene-break::after { content: '  \\2767'; }\n"
                 "@keyframes glx-sway {\n  0% { transform: rotate(-0.6deg); }\n  50% { transform: rotate(0.6deg); }\n"
                 "  100% { transform: rotate(-0.6deg); }\n}",
        'motion': "  .glx-figure {\n    animation: glx-sway 7s ease-in-out infinite;\n  }\n"
                  "  .glx-scene-break {\n    animation: glx-shimmer 4s ease-in-out infinite;\n  }\n",
    },
    {
        'name': "Necromancer's Grimoire",
        'body_font': "Georgia, 'Times New Roman', serif", 'heading_font': "'Palatino Linotype', Georgia, serif",
        'text': '#d3d8cc', 'page': '#0b0f0c', 'heading': '#7dff9a', 'accent': '#4fd47a', 'muted': '#8fa393',
        'panel': '#131a15', 'rule': '#2c4a35', 'stripe': '#18211b', 'hover': '#233a2a', 'system': '#9dffb4',
        'glow': '#3dff7d',
        'h1': 'letter-spacing: 0.12em; text-shadow: 0 0 8px #2fbf62;',
        'status_border': '1px solid #4fd47a',
        'status_window': 'box-shadow: 0 0 12px rgba(61, 255, 125, 0.3), inset 0 0 20px rgba(0, 0, 0, 0.6);',
        'letter': 'font-style: italic; background-color: #d9d0b4; color: #2a2518; border: 1px solid #6e6544;',
        'system_message': 'text-shadow: 0 0 5px #3dff7d;',
        'sfx': 'text-transform: uppercase;',
        'extra': ".glx-scene-break::before { content: '\\2620  '; }\n.glx-scene-break::after { content: '  \\2620'; }\n"
                 "@keyframes glx-wither {\n  0% { opacity: 1; }\n  40% { opacity: 0.45; }\n  55% { opacity: 1; }\n"
                 "  75% { opacity: 0.7; }\n  100% { opacity: 1; }\n}",
        'motion': "  .glx-sfx {\n    animation: glx-wither 2s steps(5, end) infinite;\n  }\n"
                  "  .glx-system-message {\n    animation: glx-glow-pulse 3.2s ease-in-out infinite;\n  }\n",
    },
    {
        'name': 'Celestial Starfall',
        'body_font': "Georgia, 'Palatino Linotype', serif", 'heading_font': "'Palatino Linotype', Georgia, serif",
        'text': '#e6ecfa', 'page': '#0a1030', 'heading': '#ffe9a8', 'accent': '#8fb4ff', 'muted': '#a7b4d6',
        'panel': '#131c4a', 'rule': '#34457f', 'stripe': '#18235a', 'hover': '#233070', 'system': '#bcd2ff',
        'glow': '#ffe9a8', 'radius': '10px',
        'h1': 'letter-spacing: 0.1em; text-shadow: 0 0 10px #8fb4ff;',
        'status_border': '1px solid #8fb4ff',
        'status_window': 'box-shadow: 0 0 14px rgba(143, 180, 255, 0.4);',
        'illustration': 'border: 1px solid #34457f; box-shadow: 0 0 18px rgba(255, 233, 168, 0.35);',
        'extra': ".glx-book { background-image: radial-gradient(circle at 12% 18%, rgba(255, 255, 255, 0.55) 0, "
                 "rgba(255, 255, 255, 0) 1.5px), radial-gradient(circle at 78% 42%, rgba(255, 255, 255, 0.45) 0, "
                 "rgba(255, 255, 255, 0) 1.5px), radial-gradient(circle at 44% 80%, rgba(255, 233, 168, 0.5) 0, "
                 "rgba(255, 233, 168, 0) 2px); background-size: 260px 260px; }\n"
                 ".glx-scene-break::before { content: '\\2726  '; }\n.glx-scene-break::after { content: '  \\2726'; }\n"
                 "@keyframes glx-twinkle {\n  0% { opacity: 0.6; text-shadow: 0 0 2px #ffe9a8; }\n"
                 "  50% { opacity: 1; text-shadow: 0 0 14px #ffe9a8; }\n  100% { opacity: 0.6; text-shadow: 0 0 2px #ffe9a8; }\n}\n"
                 "@keyframes glx-drift {\n  0% { background-position: 0 0; }\n  100% { background-position: 0 260px; }\n}",
        'motion': "  .glx-book {\n    animation: glx-drift 90s linear infinite;\n  }\n"
                  "  .glx-scene-break {\n    animation: glx-twinkle 3s ease-in-out infinite;\n  }\n",
    },
    # ---- leveling system ----------------------------------------------------------------------------
    {
        'name': 'Level Up Burst',
        'body_font': "'Segoe UI', Roboto, Arial, sans-serif", 'heading_font': "'Segoe UI', Roboto, Arial, sans-serif",
        'text': '#1f2430', 'page': '#fffdf4', 'heading': '#c2410c', 'accent': '#f59e0b', 'muted': '#555b6b',
        'panel': '#fff4d6', 'rule': '#f3c969', 'stripe': '#fff0c2', 'hover': '#ffe39a', 'system': '#b45309',
        'glow': '#ffb300', 'radius': '8px',
        'h1': 'text-transform: uppercase; letter-spacing: 0.1em;',
        'status_border': '2px solid #f59e0b',
        'status_window': 'box-shadow: 0 2px 12px rgba(245, 158, 11, 0.35);',
        'system_message': 'font-weight: bold; border-left-width: 6px; border-radius: 6px;',
        'extra': ".glx-system-message::before { content: '\\2605  '; color: #f59e0b; }\n"
                 ".glx-glow { font-weight: bold; text-transform: uppercase; letter-spacing: 0.08em; }\n"
                 "@keyframes glx-burst {\n  0% { transform: scale(1); text-shadow: 0 0 2px #ffb300; }\n"
                 "  40% { transform: scale(1.12); text-shadow: 0 0 16px #ffb300, 0 0 30px #ff7a00; }\n"
                 "  100% { transform: scale(1); text-shadow: 0 0 2px #ffb300; }\n}\n"
                 "@keyframes glx-slide-in {\n  from { opacity: 0; transform: translateX(-24px); }\n"
                 "  to { opacity: 1; transform: translateX(0); }\n}",
        'motion': "  .glx-glow {\n    display: inline-block;\n    animation: glx-burst 1.8s ease-out infinite;\n  }\n"
                  "  .glx-system-message {\n    animation: glx-slide-in 0.8s ease-out 1 both;\n  }\n",
    },
    {
        'name': 'Quest Log',
        'body_font': "Georgia, 'Palatino Linotype', serif", 'heading_font': "'Palatino Linotype', Georgia, serif",
        'status_font': "'Segoe UI', Roboto, Arial, sans-serif",
        'text': '#3a2f22', 'page': '#f4e8cf', 'heading': '#5a3a14', 'accent': '#a8741a', 'muted': '#6b5a40',
        'panel': '#ecdcb8', 'rule': '#c9aa6c', 'stripe': '#e6d4ab', 'hover': '#dcc690', 'system': '#7a4a0f',
        'glow': '#e0a93a', 'radius': '4px',
        'h1': 'border-bottom: 2px solid #a8741a; padding-bottom: 0.2em;',
        'status_border': '2px solid #a8741a',
        'system_message': 'font-weight: bold; border-left-width: 6px;',
        'illustration': 'border: 4px solid #ecdcb8; box-shadow: 0 2px 8px rgba(90, 58, 20, 0.35);',
        'extra': ".glx-book li { list-style-type: none; }\n"
                 ".glx-book li::before { content: '\\2610  '; color: #a8741a; }\n"
                 ".glx-system-message::before { content: '\\2757  '; color: #a8741a; }\n"
                 ".glx-scene-break::before { content: '\\2014  '; }\n.glx-scene-break::after { content: '  \\2014'; }\n"
                 "@keyframes glx-slide-in {\n  from { opacity: 0; transform: translateX(-24px); }\n"
                 "  to { opacity: 1; transform: translateX(0); }\n}",
        'motion': "  .glx-system-message {\n    animation: glx-slide-in 0.7s ease-out 1 both;\n  }\n",
    },
    {
        'name': 'Dungeon System',
        'body_font': "'Segoe UI', Roboto, Arial, sans-serif", 'heading_font': "'Segoe UI', Roboto, Arial, sans-serif",
        'status_font': "Consolas, 'Courier New', monospace",
        'text': '#c7d6ea', 'page': '#070b14', 'heading': '#5db7ff', 'accent': '#2e93ff', 'muted': '#8296b3',
        'panel': '#0d1626', 'rule': '#1d3a63', 'stripe': '#101c32', 'hover': '#17305a', 'system': '#6cc4ff',
        'glow': '#3aa0ff', 'radius': '2px',
        'h1': 'text-transform: uppercase; letter-spacing: 0.16em; text-shadow: 0 0 10px #2e93ff;',
        'status_border': '1px solid #3aa0ff',
        'status_head': '#10264a',
        'status_text': '#9fd0ff',
        'status_window': 'box-shadow: 0 0 18px rgba(58, 160, 255, 0.5), inset 0 0 22px rgba(58, 160, 255, 0.15);',
        'system_message': 'text-transform: uppercase; letter-spacing: 0.05em; border: 1px solid #1d3a63; '
                          'border-left: 4px solid #3aa0ff;',
        'extra': ".glx-system-message::before { content: '[ '; }\n.glx-system-message::after { content: ' ]'; }\n"
                 ".glx-status-window td + td { color: #5db7ff; font-weight: bold; }\n"
                 "@keyframes glx-boot {\n  0% { opacity: 0; transform: scaleY(0.92); }\n"
                 "  60% { opacity: 1; transform: scaleY(1.02); }\n  100% { opacity: 1; transform: scaleY(1); }\n}",
        'motion': "  .glx-status-window {\n    animation: glx-boot 1s ease-out 1 both;\n  }\n"
                  "  .glx-system-message {\n    animation: glx-reveal 1.8s steps(30, end) 1 both;\n  }\n",
    },
    {
        'name': 'Skill Tree Glow',
        'body_font': "'Segoe UI', Roboto, Arial, sans-serif", 'heading_font': "'Segoe UI', Roboto, Arial, sans-serif",
        'status_font': "'Segoe UI', Roboto, Arial, sans-serif",
        'text': '#d6e2ee', 'page': '#10161f', 'heading': '#7be0c3', 'accent': '#3fcfa5', 'muted': '#8ba0b3',
        'panel': '#17202c', 'rule': '#27445a', 'stripe': '#1b2735', 'hover': '#233a4d', 'system': '#7be0c3',
        'glow': '#3fcfa5', 'radius': '10px',
        'h1': 'letter-spacing: 0.06em;',
        'status_border': '1px solid #3fcfa5',
        'status_window': 'box-shadow: 0 0 12px rgba(63, 207, 165, 0.3);',
        'system_message': 'border-radius: 8px;',
        'extra': ".glx-book ul, .glx-book ol { border-left: 2px solid #27445a; padding-left: 0.4em; }\n"
                 ".glx-book li::marker { content: '\\25C6  '; color: #3fcfa5; }\n"
                 ".glx-glow { padding: 0 0.5em; border: 1px solid #3fcfa5; border-radius: 999px; }\n"
                 "@keyframes glx-ring {\n  0% { box-shadow: 0 0 2px rgba(63, 207, 165, 0.6); }\n"
                 "  50% { box-shadow: 0 0 14px rgba(63, 207, 165, 0.95); }\n  100% { box-shadow: 0 0 2px rgba(63, 207, 165, 0.6); }\n}",
        'motion': "  .glx-glow {\n    animation: glx-ring 2.4s ease-in-out infinite;\n  }\n"
                  "  .glx-status-window {\n    animation: glx-shimmer 4s ease-in-out infinite;\n  }\n",
    },
    {
        'name': 'Guild Card',
        'body_font': "'Segoe UI', Roboto, Arial, sans-serif", 'heading_font': "Georgia, 'Times New Roman', serif",
        'status_font': "'Segoe UI', Roboto, Arial, sans-serif",
        'text': '#262a31', 'page': '#eef0f3', 'heading': '#34495e', 'accent': '#7b8794', 'muted': '#5b6572',
        'panel': '#dfe3e8', 'rule': '#b6bcc6', 'stripe': '#e6e9ed', 'hover': '#d3d9e1', 'system': '#34495e',
        'glow': '#c7a24b', 'radius': '10px',
        'h1': 'letter-spacing: 0.08em; text-transform: uppercase;',
        'status_border': '2px solid #8a94a3',
        'status_head': '#c9d0d9',
        'status_window': 'border-radius: 10px; background-image: linear-gradient(115deg, rgba(255, 255, 255, 0) 40%, '
                         'rgba(255, 255, 255, 0.55) 50%, rgba(255, 255, 255, 0) 60%); background-size: 260% 100%; '
                         'background-position: 120% 0; background-repeat: no-repeat; '
                         'box-shadow: 0 3px 10px rgba(52, 73, 94, 0.25);',
        'system_message': 'border-radius: 8px; border-left-color: #c7a24b;',
        'illustration': 'border: 3px solid #b6bcc6;',
        'extra': ".glx-scene-break::before { content: '\\2694  '; }\n.glx-scene-break::after { content: '  \\2694'; }\n"
                 "@keyframes glx-sheen {\n  0% { background-position: 120% 0; }\n  100% { background-position: -20% 0; }\n}",
        'motion': "  .glx-status-window {\n    animation: glx-sheen 5s ease-in-out infinite;\n  }\n",
    },
)

#: name -> stylesheet (read-only; "Duplicate" makes an editable user copy)
BUILTIN_PRESETS = {spec['name']: _preset(spec) for spec in _PRESET_SPECS}
BUILTIN_PRESET_NAMES = tuple(spec['name'] for spec in _PRESET_SPECS)


# ---------------------------------------------------------------------------------------------
# Scoping: every selector under .glx-, keyframes glx-, nothing that loads or runs anything
# ---------------------------------------------------------------------------------------------

_COMMENT_RE = re.compile(r'/\*.*?\*/', re.S)
_SCOPED_SELECTOR_RE = re.compile(r'^\.glx-[A-Za-z0-9_-]')
_UNSAFE_VALUE_RE = re.compile(
    r'url\(\s*[\'"]?\s*(?:https?:|//|javascript:|data:text)|expression\s*\(|javascript:|-moz-binding', re.I)
_UNSAFE_PROPERTY_RE = re.compile(r'^\s*(?:behavior|-moz-binding)\s*$', re.I)


def _split_top(text, sep):
    """Split on ``sep`` outside parentheses / brackets / quotes."""
    parts, depth, quote, start = [], 0, None, 0
    for i, ch in enumerate(text):
        if quote:
            if ch == quote and text[i - 1] != '\\':
                quote = None
        elif ch in '"\'':
            quote = ch
        elif ch in '([':
            depth += 1
        elif ch in ')]':
            depth = max(0, depth - 1)
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _parse_blocks(css):
    """``[(prelude, body)]`` of the top level: ``body`` is None for a statement (``@import x;``)."""
    items, i, n = [], 0, len(css)
    while i < n:
        while i < n and css[i] in ' \t\r\n;':
            i += 1
        if i >= n:
            break
        j, quote = i, None
        while j < n:
            ch = css[j]
            if quote:
                if ch == quote:
                    quote = None
            elif ch in '"\'':
                quote = ch
            elif ch in '{;':
                break
            j += 1
        prelude = css[i:j].strip()
        if j >= n:
            if prelude:
                items.append((prelude, None))
            break
        if css[j] == ';':
            items.append((prelude, None))
            i = j + 1
            continue
        depth, k = 1, j + 1
        while k < n and depth:
            if css[k] == '{':
                depth += 1
            elif css[k] == '}':
                depth -= 1
            k += 1
        items.append((prelude, css[j + 1:k - 1]))
        i = k
    return items


def _clean_declarations(body, warnings):
    out = []
    for decl in _split_top(body, ';'):
        decl = decl.strip()
        if not decl:
            continue
        prop = decl.split(':', 1)[0]
        if _UNSAFE_PROPERTY_RE.match(prop) or _UNSAFE_VALUE_RE.search(decl):
            warnings.append(f"Removed unsafe declaration: {decl[:60]}")
            continue
        out.append(decl)
    return out


def _scope_selector(selector, warnings):
    sel = ' '.join(selector.split())
    if not sel:
        return ''
    if _SCOPED_SELECTOR_RE.match(sel):
        return sel
    low = sel.lower()
    for root in ('html body', 'body', 'html', ':root'):
        if low == root:
            warnings.append(f"Scoped '{sel}' as '.glx-book'")
            return '.glx-book'
        if low.startswith(root + ' ') or low.startswith(root + '>'):
            rest = sel[len(root):].lstrip(' >')
            warnings.append(f"Scoped '{sel}' under .glx-book")
            return '.glx-book ' + rest
    warnings.append(f"Scoped '{sel}' under .glx-book")
    return '.glx-book ' + sel


def _keyframe_names(css):
    return {m.group(1) for m in re.finditer(r'@(?:-webkit-)?keyframes\s+([A-Za-z0-9_-]+)', css)}


def _rename_animation_refs(decls, renames):
    if not renames:
        return decls
    out = []
    for decl in decls:
        prop, _, value = decl.partition(':')
        if prop.strip().lower() in ('animation', 'animation-name', '-webkit-animation', '-webkit-animation-name'):
            for old, new in renames.items():
                value = re.sub(r'(?<![A-Za-z0-9_-])' + re.escape(old) + r'(?![A-Za-z0-9_-])', new, value)
            decl = f"{prop}:{value}"
        out.append(decl)
    return out


def _format_rule(selectors, decls, indent=''):
    inner = ''.join(f"{indent}  {d};\n" for d in decls)
    return f"{indent}{', '.join(selectors)} {{\n{inner}{indent}}}\n"


def _scope_items(items, warnings, renames, indent=''):
    out = []
    for prelude, body in items:
        low = prelude.lower()
        if low.startswith('@'):
            word = low.split(None, 1)[0].rstrip('{')
            if body is None:
                warnings.append(f"Removed '{prelude[:60]}'")
                continue
            if word in ('@keyframes', '@-webkit-keyframes'):
                name = prelude.split(None, 1)[1].strip() if ' ' in prelude else ''
                new_name = renames.get(name, name)
                frames = []
                for frame_sel, frame_body in _parse_blocks(body):
                    if frame_body is None:
                        continue
                    decls = _clean_declarations(frame_body, warnings)
                    frames.append(_format_rule([' '.join(frame_sel.split())], decls, indent + '  '))
                out.append(f"{indent}@keyframes {new_name} {{\n{''.join(frames)}{indent}}}\n")
            elif word in ('@media', '@supports'):
                inner = _scope_items(_parse_blocks(body), warnings, renames, indent + '  ')
                out.append(f"{indent}{' '.join(prelude.split())} {{\n{inner}{indent}}}\n")
            else:
                warnings.append(f"Removed '{word}' block")
            continue
        if body is None:
            continue
        selectors = [s for s in (_scope_selector(part, warnings) for part in _split_top(prelude, ',')) if s]
        decls = _rename_animation_refs(_clean_declarations(body, warnings), renames)
        if selectors and decls:
            out.append(_format_rule(selectors, decls, indent))
    return ''.join(out)


def scope_css(css):
    """``(css, warnings)``: ``css`` with every selector under ``.glx-book`` / a ``.glx-`` class, keyframe names
    starting ``glx-``, and no ``@import``, remote ``url(...)``, ``expression(``, ``behavior:`` or ``javascript:``.
    A stylesheet that already follows the rules comes back with no warnings (only re-indented)."""
    warnings = []
    text = _COMMENT_RE.sub('', str(css or ''))
    renames = {}
    for name in _keyframe_names(text):
        if not name.startswith('glx-'):
            renames[name] = 'glx-' + name
            warnings.append(f"Renamed keyframes '{name}' to 'glx-{name}'")
    return _scope_items(_parse_blocks(text), warnings, renames), warnings


def css_problems(css):
    """Why ``css`` is not a valid interactive stylesheet (``[]`` when it is): what ``scope_css`` would change."""
    return scope_css(css)[1]


def defined_classes(css):
    """The ``glx-`` classes a stylesheet styles, in first-use order."""
    seen = []
    for name in re.findall(r'\.(glx-[A-Za-z0-9_-]+)', _COMMENT_RE.sub('', str(css or ''))):
        if name not in seen:
            seen.append(name)
    return seen


def assignable_classes(css):
    """Classes the planner may put on paragraphs: those ``css`` defines, minus the structural ones."""
    return [name for name in defined_classes(css) if name not in STRUCTURAL_CLASSES]


def extract_css_block(reply):
    """The first ```css (or bare ```) code block of a model reply, else the whole reply."""
    text = str(reply or '')
    match = re.search(r'```\s*css\s*\n(.*?)```', text, re.S | re.I) or re.search(r'```\s*\n(.*?)```', text, re.S)
    return (match.group(1) if match else text).strip()


# ---------------------------------------------------------------------------------------------
# Presets: built-in + the user's own (config ``interactive_css_user_presets``)
# ---------------------------------------------------------------------------------------------

def user_presets(config):
    raw = (config or {}).get(USER_PRESETS_KEY) or {}
    if not isinstance(raw, dict):
        return {}
    return {str(name): str(css) for name, css in raw.items()
            if isinstance(name, str) and name.strip() and isinstance(css, str)}


def preset_names(config=None):
    """Built-ins first, then the user's presets (sorted)."""
    users = [name for name in sorted(user_presets(config), key=str.casefold) if name not in BUILTIN_PRESETS]
    return list(BUILTIN_PRESET_NAMES) + users


def is_builtin_preset(name):
    return name in BUILTIN_PRESETS


def preset_css(config, name):
    """The stylesheet of preset ``name`` (None when unknown)."""
    if name in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[name]
    return user_presets(config).get(name)


def active_preset_name(config):
    name = str((config or {}).get(PRESET_KEY) or DEFAULT_PRESET)
    return name if preset_css(config, name) is not None else DEFAULT_PRESET


def unique_preset_name(config, base):
    names = {n.casefold() for n in preset_names(config)}
    base = str(base or 'My style').strip() or 'My style'
    if base.casefold() not in names:
        return base
    k = 2
    while f"{base} {k}".casefold() in names:
        k += 1
    return f"{base} {k}"


def preset_name_for_request(config, request):
    """A new user preset's name from the words the user typed for "New from prompt" (first four words)."""
    words = ' '.join(str(request or '').split()[:4]) or 'My style'
    return unique_preset_name(config, words.title())


def duplicate_name(config, name):
    return unique_preset_name(config, f"{name} copy")


def save_user_preset(config, name, css):
    """Store a user preset (scoped first). Returns ``(name, warnings)``; built-in names are refused."""
    name = str(name or '').strip()
    if not name:
        raise ValueError("A style needs a name")
    if name in BUILTIN_PRESETS:
        raise ValueError(f"'{name}' is a built-in style; duplicate it to edit a copy")
    scoped, warnings = scope_css(css)
    presets = user_presets(config)
    presets[name] = scoped
    config[USER_PRESETS_KEY] = presets
    return name, warnings


def delete_user_preset(config, name):
    presets = user_presets(config)
    if name not in presets:
        return False
    del presets[name]
    config[USER_PRESETS_KEY] = presets
    if config.get(PRESET_KEY) == name:
        config[PRESET_KEY] = DEFAULT_PRESET
    return True


# ---------------------------------------------------------------------------------------------
# The book stylesheet
# ---------------------------------------------------------------------------------------------

def book_stylesheet_path(output_dir):
    return os.path.join(output_dir, INTERACTIVE_FOLDER, STYLESHEET_FILENAME)


def write_book_stylesheet(output_dir, css):
    """Write the scoped ``css`` as the book's interactive stylesheet (atomic). Returns ``(path, warnings)``."""
    scoped, warnings = scope_css(css)
    path = book_stylesheet_path(output_dir)
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.glx-', suffix='.tmp', dir=folder)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(scoped)
        os.replace(temp, path)
        temp = None
    finally:
        if temp and os.path.exists(temp):
            try:
                os.remove(temp)
            except OSError:
                pass
    return path, warnings


def remove_book_stylesheet(output_dir):
    path = book_stylesheet_path(output_dir)
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def read_book_stylesheet(output_dir):
    try:
        with open(book_stylesheet_path(output_dir), 'r', encoding='utf-8') as handle:
            return handle.read()
    except OSError:
        return ''


# ---------------------------------------------------------------------------------------------
# Preview: a sample chapter that uses every class
# ---------------------------------------------------------------------------------------------

_PLACEHOLDER_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="480" height="270" viewBox="0 0 480 270">'
    '<rect width="480" height="270" fill="#8795a8"/>'
    '<path d="M0 270 L150 120 L250 210 L330 150 L480 270 Z" fill="#5f6d82"/>'
    '<circle cx="380" cy="70" r="30" fill="#c9d3df"/>'
    '<text x="240" y="150" font-family="Arial" font-size="22" fill="#ffffff" text-anchor="middle">'
    'Illustration</text></svg>'
)
PLACEHOLDER_IMAGE_URI = 'data:image/svg+xml;base64,' + base64.b64encode(_PLACEHOLDER_SVG.encode('utf-8')).decode('ascii')

#: Original placeholder text (no real-book quotes).
SAMPLE_CHAPTER_HTML = f"""<body class="glx-book">
<h1>Chapter 7: The Lantern Gate</h1>
<p>Rain had followed Mira all the way from the river market, and it showed no sign of stopping when she reached the old gate.</p>
<p class="glx-dialogue">"You're late," said Tomas, without looking up from the map spread across the stone bench.</p>
<p class="glx-dialogue">"The bridge was closed. I had to go around." She shook the water from her hood.</p>
<p class="glx-thought">He's worried, she thought. He only studies maps that closely when he's worried.</p>
<p class="glx-system-message">[System] A new quest has been unlocked: The Lantern Gate.</p>
<p class="glx-system-message glx-glow">[Level Up!] Mira has reached level 13. All stats increased.</p>
<table class="glx-status-window">
<tr><th>Status</th><th>Mira</th></tr>
<tr><td>Level</td><td>12</td></tr>
<tr><td>Class</td><td>Lantern Keeper</td></tr>
<tr><td>Health</td><td>340 / 340</td></tr>
<tr><td>Mana</td><td>95 / 120</td></tr>
<tr><td>Strength</td><td>18</td></tr>
<tr><td>Skill</td><td>Steady Flame (Lv. 3)</td></tr>
</table>
<h2>What they carried</h2>
<ul>
<li>A brass lantern with a cracked glass panel</li>
<li>Two days of bread and dried apples</li>
<li>A letter neither of them had opened</li>
</ul>
<blockquote>Every gate remembers the last hand that opened it.</blockquote>
<p class="glx-reveal">The lantern flickered once, then burned a steady, impossible blue.</p>
<p class="glx-sfx">Clang!</p>
<figure class="glx-figure"><img class="glx-illustration" src="{PLACEHOLDER_IMAGE_URI}" alt=""/></figure>
<details><summary>Notes on the gate</summary><p>The carvings on the left pillar are older than the town itself.</p></details>
<p class="glx-scene-break">* * *</p>
<p class="glx-flashback">Years ago, their grandmother had stood in the same place and told them never to knock.</p>
<p class="glx-letter">Dear Mira, if you are reading this, the gate has already chosen you. Be kind to it. - G.</p>
<p class="glx-glow">The gate opened.</p>
</body>"""


def preview_document(css):
    """A complete HTML page of the sample chapter styled by ``css`` (no scripts)."""
    scoped, _warnings = scope_css(css)
    body = SAMPLE_CHAPTER_HTML
    return (
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"/>"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
        "<title>Style preview</title>"
        "<style>html, body { margin: 0; padding: 0; } body.glx-book { padding: 18px 22px; }</style>"
        f"<style>\n{scoped}</style></head>\n{body}\n</html>"
    )


__all__ = [
    'BUILTIN_PRESETS', 'BUILTIN_PRESET_NAMES', 'CLASS_VOCABULARY', 'DEFAULT_PRESET', 'ENABLED_KEY',
    'INTERACTIVE_FOLDER', 'PLACEHOLDER_IMAGE_URI', 'PRESET_KEY', 'SAMPLE_CHAPTER_HTML', 'STRUCTURAL_CLASSES',
    'STYLESHEET_FILENAME', 'USER_PRESETS_KEY', 'active_preset_name', 'assignable_classes', 'book_stylesheet_path',
    'css_problems', 'defined_classes', 'delete_user_preset', 'duplicate_name', 'extract_css_block',
    'is_builtin_preset', 'preset_css', 'preset_name_for_request',
    'preset_names', 'preview_document', 'read_book_stylesheet', 'remove_book_stylesheet', 'save_user_preset',
    'scope_css', 'unique_preset_name', 'user_presets', 'write_book_stylesheet',
]
