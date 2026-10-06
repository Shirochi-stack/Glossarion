"""Output-file naming helpers shared by desktop and mobile.

``_rename_output_files_for_retain`` (and the two Library registry lookups it
uses) moved verbatim from other_settings.py; it was already duck-typed on the
``gui`` owner (``output_dir``, ``config``, ``selected_files``, ``append_log``).
other_settings re-imports every name.

GUI-free; must stay importable on Python 3.10 without Qt.
"""

import json
import os

from epub_package import find_epub_opf_member, find_opf_path


def _library_dir():
    """The Library folder (``library_core.get_library_dir``'s path, not created here).

    U5: honours ``GLOSSARION_LIBRARY_DIR`` like the Library itself (Glossarion Mobile
    keeps its Library in app storage); desktop resolves ``~/Documents/Glossarion/Library``
    exactly as before.
    """
    from library_core import library_root_path
    return library_root_path()


def _library_origins_raw_sources_for_stem(folder_stem):
    """Raw source paths in library_origins.txt whose stem matches *folder_stem*.

    *folder_stem* is expected to be normcased already. Matches against
    both the Library/Raw basename and the recorded original path.
    Returns every hit so callers can detect ambiguity (len > 1 ==
    duplicate origins for one workspace name).
    """
    if not folder_stem:
        return []
    library_dir = _library_dir()
    origins_path = os.path.join(library_dir, 'library_origins.txt')
    try:
        with open(origins_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    raw_map = data.get('raw') or {}
    if not isinstance(raw_map, dict):
        return []
    raw_dir = os.path.join(library_dir, 'Raw')
    matches = []
    for lib_basename, original_path in raw_map.items():
        lb = os.path.basename(str(lib_basename or ''))
        lb_stem = os.path.splitext(lb)[0]
        op = str(original_path or '')
        op_stem = os.path.splitext(os.path.basename(op))[0]
        if (os.path.normcase(lb_stem) != folder_stem
                and os.path.normcase(op_stem) != folder_stem):
            continue
        lib_path = os.path.join(raw_dir, lb)
        if lb and os.path.isfile(lib_path):
            matches.append(os.path.abspath(lib_path))
        elif op and os.path.isfile(op):
            matches.append(os.path.abspath(op))
    return matches


def _library_raw_inputs_for_stem(folder_stem):
    """Paths in library_raw_inputs.txt whose filename stem matches.

    *folder_stem* is expected to be normcased already. Checks the
    registry at the Library root and the legacy copy inside
    ``Library/Raw``. Only existing files are returned.
    """
    if not folder_stem:
        return []
    lib_dir = _library_dir()
    matches = []
    seen = set()
    for reg_path in (os.path.join(lib_dir, 'library_raw_inputs.txt'),
                     os.path.join(lib_dir, 'Raw', 'library_raw_inputs.txt')):
        try:
            with open(reg_path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            stem = os.path.splitext(os.path.basename(ln))[0]
            if os.path.normcase(stem) != folder_stem:
                continue
            key = os.path.normcase(os.path.normpath(ln))
            if key in seen or not os.path.isfile(ln):
                continue
            seen.add(key)
            matches.append(os.path.abspath(ln))
    return matches


def _rename_output_files_for_retain(gui, retain: bool, output_dir: str = None):
    """Rename output files when the 'retain source extension' toggle changes.

    When *retain* is True (toggle ON):
        - Remove the ``response_`` prefix from each file.
        - Restore the extension from the OPF package (strip stacked extensions
          like ``.html.html`` or ``.htm.xhtml`` first).

    When *retain* is False (toggle OFF):
        - Add the ``response_`` prefix if missing.
        - Replace the file extension with ``.html`` (strip stacked
          extensions first).

    Does **nothing** if no OPF package is found in the output directory.
    """
    import xml.etree.ElementTree as _ET

    # Determine the output directory
    if not output_dir or not os.path.isdir(output_dir):
        # Try multiple sources to find the output directory
        candidates = []

        # 1. gui.output_dir (if set)
        _od = getattr(gui, 'output_dir', None)
        if _od:
            candidates.append(_od)

        # 2. Derive from selected EPUB + output override
        override = os.environ.get('OUTPUT_DIRECTORY') or ''
        if not override:
            _cfg = getattr(gui, 'config', None)
            if isinstance(_cfg, dict):
                override = _cfg.get('output_directory', '') or ''

        epub_path = os.environ.get('EPUB_PATH', '')
        if not epub_path:
            _sf = getattr(gui, 'selected_files', None)
            if _sf:
                for _f in _sf:
                    if str(_f).lower().endswith('.epub'):
                        epub_path = str(_f)
                        break

        if epub_path:
            base_name = os.path.splitext(os.path.basename(epub_path))[0]
            if override:
                candidates.append(os.path.join(override, base_name))
            candidates.append(base_name)  # relative to CWD

        # 3. OUTPUT_DIR env (legacy fallback)
        _envod = os.environ.get('OUTPUT_DIR', '')
        if _envod:
            candidates.append(_envod)

        # Pick the first candidate that is an existing directory
        output_dir = None
        for c in candidates:
            if c and os.path.isdir(c):
                output_dir = c
                break

    if not output_dir or not os.path.isdir(output_dir):
        return ('no_opf',)

    opf_path = find_opf_path(output_dir)
    if not opf_path:
        # Fallback: extract the package document from the source EPUB directly.
        # This handles the case where the output dir doesn't have it yet
        # (first run, cleaned output, etc.).
        #
        # Resolution order matters: prefer the ACTUAL input file paths
        # (EPUB_PATH / the selected input files) whose filename stem
        # matches this output folder — the source_epub.txt sidecar is
        # only a last-resort fallback because it can be stale or point
        # at the wrong book. The stem guard is mandatory: EPUB_PATH /
        # selected files are process-global and extracting the wrong
        # book's content.opf into this folder would permanently poison
        # its chapter mapping.
        _folder_stem = os.path.normcase(
            os.path.basename(os.path.normpath(output_dir)))

        def _stem_matches(_p):
            try:
                return os.path.normcase(
                    os.path.splitext(os.path.basename(str(_p)))[0]
                ) == _folder_stem
            except Exception:
                return False

        _epub = ''
        # 1. EPUB_PATH env var — when it matches this workspace
        _env_epub = os.environ.get('EPUB_PATH', '')
        if _env_epub and _stem_matches(_env_epub):
            _epub = _env_epub
        # 2. Selected input files — the one matching this workspace
        if not _epub:
            _sf = getattr(gui, 'selected_files', None)
            if _sf:
                for _f in _sf:
                    if str(_f).lower().endswith('.epub') and _stem_matches(_f):
                        _epub = str(_f)
                        break
        # 3. Library origins registry — a UNIQUE stem match resolves
        #    the raw source directly; duplicates mean ambiguity and
        #    drop through to the next source.
        if not _epub:
            _origin_matches = _library_origins_raw_sources_for_stem(_folder_stem)
            if len(_origin_matches) == 1 and _origin_matches[0].lower().endswith('.epub'):
                _epub = _origin_matches[0]
        # 4. Raw-inputs registry (library_raw_inputs.txt) — every raw
        #    the user ever loaded; first stem match wins.
        if not _epub:
            for _f in _library_raw_inputs_for_stem(_folder_stem):
                if _f.lower().endswith('.epub'):
                    _epub = _f
                    break
        # 5. source_epub.txt sidecar — last resort, only reached when
        #    no registry resolves this workspace; the sidecar can be
        #    stale/overwritten by multi-EPUB runs.
        if not _epub:
            _sidecar = os.path.join(output_dir, 'source_epub.txt')
            if os.path.exists(_sidecar):
                try:
                    with open(_sidecar, 'r', encoding='utf-8') as _sf:
                        _epub = _sf.read().strip()
                except Exception:
                    pass

        if _epub and os.path.isfile(_epub) and _epub.lower().endswith('.epub'):
            import zipfile as _zf
            try:
                with _zf.ZipFile(_epub, 'r') as zf:
                    opf_member = find_epub_opf_member(zf)
                    if opf_member:
                        # Extract to output dir so subsequent calls find it immediately
                        os.makedirs(output_dir, exist_ok=True)
                        _opf_data = zf.read(opf_member)
                        _opf_basename = os.path.basename(opf_member.replace('\\', '/'))
                        opf_path = os.path.join(output_dir, _opf_basename)
                        with open(opf_path, 'wb') as _wf:
                            _wf.write(_opf_data)
            except Exception:
                pass

        if not opf_path or not os.path.exists(opf_path):
            return ('no_opf',)

    # Parse the OPF package to build original basenames + extensions.
    try:
        tree = _ET.parse(opf_path)
        root = tree.getroot()
        ns_uri = ''
        if root.tag.startswith('{'):
            ns_uri = root.tag[1:root.tag.index('}')]
        ns = {'opf': ns_uri} if ns_uri else {}

        # manifest: collect HTML/XHTML items  →  {basename_without_ext: original_ext}
        opf_names = {}  # core_name  → original extension (e.g. '.xhtml')
        manifest_xpath = './/opf:manifest/opf:item' if ns else './/{http://www.idpf.org/2007/opf}manifest/{http://www.idpf.org/2007/opf}item'
        for item in root.findall(manifest_xpath, ns if ns else None):
            href = item.get('href', '')
            media = item.get('media-type', '')
            if not href:
                continue
            if 'html' not in media.lower() and not href.lower().endswith(('.html', '.xhtml', '.htm')):
                continue
            basename = os.path.basename(href)
            # Split into core name and single extension
            name, ext = os.path.splitext(basename)
            if ext:
                opf_names[name] = ext  # e.g. 'chapter001' → '.xhtml'
    except Exception:
        return ('no_opf',)

    if not opf_names:
        return ('no_opf',)

    # Known HTML-like extensions to strip when peeling stacked extensions
    _HTML_EXTS = {'.html', '.xhtml', '.htm', '.xml'}

    def _strip_all_html_exts(filename: str):
        """Strip all trailing HTML-like extensions from *filename*.

        Returns (core_name, list_of_stripped_exts).
        Example: 'chapter001.html.html' → ('chapter001', ['.html', '.html'])
        """
        parts = []
        while True:
            name, ext = os.path.splitext(filename)
            if ext.lower() in _HTML_EXTS:
                parts.append(ext)
                filename = name
            else:
                break
        parts.reverse()
        return filename, parts

    renamed = 0
    errors = []
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        if not os.path.isfile(fpath):
            continue
        # Only consider HTML-like files
        if not fname.lower().endswith(('.html', '.xhtml', '.htm')):
            continue

        if retain:
            # --- Toggle ON: remove response_ prefix, restore opf extension ---
            working = fname
            if working.startswith('response_'):
                working = working[len('response_'):]

            core, _ = _strip_all_html_exts(working)
            opf_ext = opf_names.get(core)
            if opf_ext is None:
                continue  # not in content.opf → skip
            new_name = core + opf_ext
        else:
            # --- Toggle OFF: add response_ prefix, replace ext with .html ---
            working = fname
            if working.startswith('response_'):
                # already has prefix — just fix extension
                core, _ = _strip_all_html_exts(working[len('response_'):])
            else:
                core, _ = _strip_all_html_exts(working)

            if core not in opf_names:
                continue  # not in content.opf → skip
            new_name = 'response_' + core + '.html'

        if new_name == fname:
            continue  # nothing to change
        new_path = os.path.join(output_dir, new_name)
        if os.path.exists(new_path):
            continue  # target already exists → skip to avoid overwrite
        try:
            os.rename(fpath, new_path)
            renamed += 1
        except Exception as e:
            errors.append(f'{fname}: {e}')

    # Update translation_progress.json so renamed files are still recognised
    if renamed:
        progress_path = os.path.join(output_dir, 'translation_progress.json')
        if os.path.exists(progress_path):
            try:
                import json as _json
                with open(progress_path, 'r', encoding='utf-8') as _pf:
                    prog = _json.load(_pf)

                # Build a normalised→new_name map from the renames we just did
                # (rebuild by scanning the current directory state)
                _HTML_EXTS_L = {'.html', '.xhtml', '.htm', '.xml'}
                def _norm_progress(fname):
                    if not fname:
                        return ''
                    base = os.path.basename(fname)
                    if base.startswith('response_'):
                        base = base[len('response_'):]
                    while True:
                        b, e = os.path.splitext(base)
                        if e.lower() in _HTML_EXTS_L:
                            base = b
                        else:
                            break
                    return base.lower()

                # Collect current files keyed by normalised name
                current_files = {}
                for f in os.listdir(output_dir):
                    if f.lower().endswith(('.html', '.xhtml', '.htm')) and os.path.isfile(os.path.join(output_dir, f)):
                        current_files[_norm_progress(f)] = f

                updated = 0
                for _key, info in prog.get('chapters', {}).items():
                    old_out = info.get('output_file')
                    if not old_out:
                        continue
                    norm = _norm_progress(old_out)
                    new_out = current_files.get(norm)
                    if new_out and new_out != old_out:
                        info['output_file'] = new_out
                        updated += 1

                if updated:
                    tmp = progress_path + '.tmp'
                    with open(tmp, 'w', encoding='utf-8') as _pf:
                        _json.dump(prog, _pf, ensure_ascii=False, indent=2)
                    if os.path.exists(progress_path):
                        os.remove(progress_path)
                    os.rename(tmp, progress_path)
            except Exception:
                pass  # progress update is best-effort

    # Log results
    if hasattr(gui, 'append_log'):
        if renamed:
            mode = 'retain source names' if retain else 'response_ prefix'
            gui.append_log(f'✅ Renamed {renamed} file(s) to {mode}')
        if errors:
            for err in errors[:5]:
                gui.append_log(f'⚠️ Rename error: {err}')

    if renamed:
        return ('renamed', renamed)
    return ('no_files',)
