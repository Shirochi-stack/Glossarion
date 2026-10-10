"""Identity-based EPUB image tracking, independent of a page's filename.

The legacy rename map remains a flat lookup. This sidecar retains historical
parents and page bindings across extraction, including unresolved references.
"""
import hashlib
import json
import os
import posixpath
import re
import tempfile
import threading
from functools import wraps
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup

SIDECAR = 'image_reference_map.json'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.svg', '.bmp', '.webp'}
CSS_URL = re.compile(r'url\(\s*([\'"]?)(.*?)\1\s*\)', re.I)
_LOCK = threading.RLock()


def _locked(function):
    @wraps(function)
    def run(*args, **kwargs):
        with _LOCK:
            return function(*args, **kwargs)
    return run


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _load(folder):
    try:
        data = json.loads((Path(folder) / SIDECAR).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {'version': 1, 'parents': {}, 'pages': {}, 'current': {}, 'pending': False}
    # Do not destroy unknown/corrupt metadata before refreshing resources.
    if not isinstance(data, dict) or data.get('version') != 1 or not all(isinstance(data.get(k), dict) for k in ('parents', 'pages', 'current')):
        raise ValueError('Unsupported image reference map')
    return data


def _save(folder, data):
    for identity, record in data['parents'].items():
        record['referencing_pages'] = sorted(
            page for page, refs in data['pages'].items() if identity in refs.values())
    path = Path(folder) / SIDECAR
    serialized = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') == serialized:
        return
    fd, temporary = tempfile.mkstemp(prefix='.image-reference-', suffix='.tmp', dir=folder)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _renames(folder, terminal=True):
    try:
        raw = json.loads((Path(folder) / 'image_rename_map.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    raw = {str(k): v for k, v in raw.items() if isinstance(v, str)}
    if not terminal:
        return {key.casefold(): value for key, value in raw.items()
                if value and Path(value).name == value and '/' not in value and '\\' not in value}
    result = {}
    for original in raw:
        target, seen = original, set()
        while target in raw and raw[target] != target:
            if target in seen:
                target = None
                break
            seen.add(target)
            target = raw[target]
        if target and Path(target).name == target and '/' not in target and '\\' not in target:
            result[original.casefold()] = target
    return result


def _basename(url):
    parsed = urlsplit(url.replace('\\', '/'))
    if parsed.scheme or parsed.netloc or not parsed.path:
        return ''
    return posixpath.basename(unquote(parsed.path))


def _replace_basename(url, filename):
    # Preserve the original directory, escaping, query string and fragment.
    path = re.split(r'[?#]', url, maxsplit=1)[0]
    split = max(path.rfind('/'), path.rfind('\\')) + 1
    return url[:split] + filename + url[len(path):]


def _references(soup):
    """Yield supported local image URLs and setters, in document order."""
    for tag in soup.find_all(True):
        attrs = {'img': ('src',), 'image': ('xlink:href', 'href', '{http://www.w3.org/1999/xlink}href'),
                 'object': ('data',), 'video': ('poster',)}.get(tag.name, ())
        for attr in attrs:
            url = tag.get(attr)
            if isinstance(url, str) and _basename(url):
                yield url, lambda value, tag=tag, attr=attr: tag.__setitem__(attr, value)
        style = tag.get('style', '')
        if isinstance(style, str):
            for match in CSS_URL.finditer(style):
                url = match.group(2)
                if _basename(url) and Path(_basename(url)).suffix.lower() in IMAGE_EXTENSIONS:
                    def set_style(value, tag=tag, url=url):
                        tag['style'] = CSS_URL.sub(
                            lambda m: m.group(0).replace(m.group(2), value, 1)
                            if m.group(2) == url else m.group(0), tag['style'])
                    yield url, set_style


def apply_filename_map(soup, rename_map):
    """Apply extraction aliases without losing URL suffixes or changing remotes."""
    lookup = {name.casefold(): target for name, target in rename_map.items()}
    modified = False
    for url, setter in _references(soup):
        filename = lookup.get(_basename(url).casefold())
        if filename:
            replacement = _replace_basename(url, filename)
            if replacement != url:
                setter(replacement)
                modified = True
    return modified


def _html_files(folder):
    # Retained source extensions can be unusual; honor OPF HTML media types.
    import xml.etree.ElementTree as ET
    from epub_package import find_opf_path
    cores = set()
    opf = find_opf_path(folder)
    if opf:
        try:
            for item in ET.parse(opf).getroot().iter():
                if item.tag.rsplit('}', 1)[-1] == 'item' and 'html' in item.get('media-type', '').lower():
                    cores.add(Path(unquote(item.get('href', ''))).stem.casefold())
        except (OSError, ET.ParseError):
            pass
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.name not in {SIDECAR, 'image_rename_map.json'}
                  and (p.suffix.lower() in {'.html', '.xhtml', '.htm'}
                       or p.stem.casefold().removeprefix('response_') in cores))


def _lookup(data, name, renames):
    folded = {k.casefold(): v for k, v in data['current'].items()}
    # A canonical filename that actually exists takes precedence over an alias.
    return folded.get(name.casefold()) or folded.get(renames.get(name.casefold(), '').casefold())


def _binding(prior, data, name, renames):
    folded = {key.casefold(): value for key, value in prior.items()}
    return folded[name.casefold()] if name.casefold() in folded else _lookup(data, name, renames)


def _parent(data, source_path, sha256, filename):
    identity = _digest((source_path + '\0' + sha256).encode('utf-8'))
    record = data['parents'].setdefault(identity, {})
    record.update(source_path=source_path, sha256=sha256, filename=filename)
    return identity


def _capture_pages(folder, data):
    renames = _renames(folder)
    for path in _html_files(folder):
        soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
        prior = data['pages'].get(path.name, {})
        refs = {}
        for url, _ in _references(soup):
            name = _basename(url)
            # Preserve old/unresolved bindings if the same ref is still present.
            refs[name] = _binding(prior, data, name, renames)
        data['pages'][path.name] = refs


def _recover_pages(folder, data):
    for name, update in list(data.get('repairs', {}).items()):
        digest = _digest((Path(folder) / name).read_bytes())
        if digest == update['after_sha256']:
            data['pages'][name] = update['after_refs']
        elif digest == update['before_sha256']:
            data['pages'][name] = update['before_refs']
        else:
            raise ValueError(f'Image repair interrupted and {name} was subsequently edited')
        del data['repairs'][name]


def historical_pages(folder):
    path = Path(folder) / SIDECAR
    return set(_load(folder)['pages']) if path.exists() else set()


@_locked
def snapshot_before_refresh(folder):
    """Durably bind translated refs before any old image/map is removed.

    A pending snapshot is never rebuilt from partially extracted new resources.
    """
    if not os.path.isdir(folder):
        return  # a first extraction: no output folder yet, nothing translated to bind
    data = _load(folder)
    _recover_pages(folder, data)
    if data['pending']:
        return
    images = Path(folder) / 'images'
    renames = _renames(folder)
    if not data['current'] and images.is_dir():
        originals = {}
        for old, new in renames.items():
            originals.setdefault(new.casefold(), old)
        for path in sorted(images.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith('_temp_rename_'):
                source = originals.get(path.name.casefold(), path.name)
                identity = _parent(data, source, _digest(path.read_bytes()), path.name)
                data['current'][path.name] = identity
    _capture_pages(folder, data)
    data['pending'] = True
    _save(folder, data)


@_locked
def publish_extracted_images(folder, zf, categorize_resource):
    """Publish the new inventory only after physical naming has completed."""
    data = _load(folder)
    renames = _renames(folder)
    direct_renames = _renames(folder, terminal=False)
    current = {}
    paths_by_name = {}
    for member in zf.namelist():
        info = categorize_resource(member, posixpath.basename(member))
        if info and info[0] == 'images':
            paths_by_name.setdefault(info[2], []).append(member)
    for original, members in paths_by_name.items():
        # Flattened source-name collisions cannot establish a unique parent.
        if len(members) != 1:
            continue
        digest = _digest(zf.read(members[0]))
        candidates = dict.fromkeys((direct_renames.get(original.casefold(), original),
                                    renames.get(original.casefold(), original), original))
        for filename in candidates:
            path = Path(folder) / 'images' / filename
            if path.is_file() and _digest(path.read_bytes()) == digest:
                current[filename] = _parent(data, members[0], digest, filename)
                break
    # Include localized remote images and protected resources not in the ZIP.
    images = Path(folder) / 'images'
    if images.is_dir():
        for path in sorted(images.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.name not in current and not path.name.startswith('_temp_rename_'):
                if any(path.name in (name, direct_renames.get(name.casefold()), renames.get(name.casefold())) for name in paths_by_name):
                    continue
                digest = _digest(path.read_bytes())
                previous = data['current'].get(path.name)
                old = data['parents'].get(previous, {})
                if old.get('sha256') == digest:
                    # Protected copies of an image now present in the EPUB
                    # must not make a unique new canonical target ambiguous.
                    view = {'parents': data['parents'], 'current': current}
                    if _resolve_parent(view, previous):
                        continue
                    source_path = old['source_path']
                else:
                    source_path = 'workspace/' + path.name
                identity = _parent(data, source_path, digest, path.name)
                current[path.name] = identity
    data['current'] = current
    data['pending'] = False
    by_source = {data['parents'][identity]['source_path']: identity for identity in current.values()}
    for identity in current.values():
        data['parents'][identity]['source_pages'] = []
    for member in zf.namelist():
        if not member.lower().endswith(('.html', '.xhtml', '.htm')):
            continue
        soup = BeautifulSoup(zf.read(member).decode('utf-8', errors='replace'), 'html.parser')
        for url, _ in _references(soup):
            source = posixpath.normpath(posixpath.join(posixpath.dirname(member), unquote(urlsplit(url).path)))
            identity = by_source.get(source)
            if identity and member not in data['parents'][identity]['source_pages']:
                data['parents'][identity]['source_pages'].append(member)
    _save(folder, data)


def _resolve_parent(data, identity):
    old = data['parents'].get(identity)
    if not old:
        return None
    candidates = [i for i in set(data['current'].values())
                  if data['parents'][i]['sha256'] == old['sha256']]
    same_path = [i for i in candidates if data['parents'][i]['source_path'] == old['source_path']]
    if not same_path and '/' not in old['source_path']:
        # Legacy flat maps only provide source basenames.
        same_path = [i for i in candidates if posixpath.basename(data['parents'][i]['source_path']).casefold() == old['source_path'].casefold()]
    if len(same_path) == 1:
        return same_path[0]
    if len(candidates) == 1:
        return candidates[0]
    return None


@_locked
def record_translated_page(folder, filename):
    """Bind a newly written translation to current parents, replacing old refs."""
    if not (Path(folder) / SIDECAR).exists() or os.getenv('IMAGE_MODE_EPUB_PASSTHROUGH', '0') == '1':
        return
    path = Path(folder) / filename
    if path.parent.resolve() != Path(folder).resolve() or path.suffix.lower() not in {'.html', '.xhtml', '.htm'} or not path.is_file():
        return
    data = _load(folder)
    if data['pending']:
        return
    _recover_pages(folder, data)
    renames = _renames(folder)
    soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
    data['pages'][path.name] = {_basename(url): _lookup(data, _basename(url), renames)
                                for url, _ in _references(soup)}
    _save(folder, data)


@_locked
def repair_image_references(folder, source_dir=None, log=print):
    """Repair by parent identity without renaming physical image resources."""
    source_dir = source_dir or folder
    data = _load(source_dir)
    _recover_pages(folder, data)
    if data['pending']:
        log('Image reference repair deferred: resource refresh is incomplete')
        return 0
    if not data['current'] and not data['parents']:
        snapshot_before_refresh(source_dir)
        data = _load(source_dir)
        data['pending'] = False
    renames = _renames(source_dir)
    changed = 0
    verified = {}
    for path in _html_files(folder):
        content = path.read_text(encoding='utf-8')
        soup = BeautifulSoup(content, 'html.parser')
        prior = data['pages'].get(path.name, {})
        refs, modified = {}, False
        for url, setter in _references(soup):
            name = _basename(url)
            identity = _binding(prior, data, name, renames)
            target = _resolve_parent(data, identity)
            record = data['parents'].get(target, {})
            filename = record.get('filename', '')
            target_path = Path(folder) / 'images' / filename
            if filename and filename not in verified:
                verified[filename] = target_path.is_file() and _digest(target_path.read_bytes()) == record['sha256']
            if not filename or not verified[filename]:
                refs[name] = identity
                log(f'Unresolved image reference in {path.name}: {name}')
                continue
            replacement = _replace_basename(url, filename)
            refs[filename] = target
            if replacement != url:
                setter(replacement)
                modified = True
                log(f'Repaired image reference in {path.name}: {name} → {filename}')
        if modified:
            # A journal disambiguates filename reuse if the process stops
            # between the HTML replacement and its new bindings being saved.
            rendered = str(soup)
            data.setdefault('repairs', {})[path.name] = {
                'before_sha256': _digest(path.read_bytes()),
                'after_sha256': _digest(rendered.encode('utf-8')),
                'before_refs': prior, 'after_refs': refs,
            }
            _save(source_dir, data)
            fd, temporary = tempfile.mkstemp(prefix='.image-html-', dir=folder)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8', newline='') as handle:
                    handle.write(rendered)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            changed += 1
        data['pages'][path.name] = refs
        if modified:
            del data['repairs'][path.name]
            _save(source_dir, data)
    _save(source_dir, data)
    return changed
