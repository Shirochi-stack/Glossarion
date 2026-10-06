"""Reader (UI_SPEC §3.11, §5.8): ``/reader/<bid>``.

Pure Python (no Flet; host-tested on 3.10+): ``model`` (settings, scopes, CSS
overrides, positions, labels), ``bridge`` (the page JavaScript bridge and its
events), ``blocks`` (HTML -> native blocks / Markdown), ``live`` (the live
translation feed), ``session`` (one open book over the shared reader cores) and
``document`` (page assembly). The other modules import Flet. This package
``__init__`` imports nothing so the pure modules stay Flet-free.
"""
