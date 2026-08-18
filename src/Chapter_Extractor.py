# Chapter_Extractor.py - Module-level chapter extraction functions
import os
import re
import sys
import json
import io
import threading
import time
import shutil
import hashlib
import warnings
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# Lazy import for PatternManager to speed up ProcessPoolExecutor worker startup on Windows
# The heavy TransateKRtoEN import is deferred until actually needed
_PatternManager = None
_PM = None

def _get_pattern_manager():
    """Lazy initialization of PatternManager to avoid slow imports in worker processes"""
    global _PatternManager, _PM
    if _PatternManager is None:
        pass  # TransateKRtoEN import removed (patterns inlined)
        _PatternManager = PM_Class
        _PM = PM_Class()
    return _PM

# For backward compatibility - property-like access
class _LazyPM:
    def __getattr__(self, name):
        return getattr(_get_pattern_manager(), name)

PM = _LazyPM()

from bs4 import BeautifulSoup
try:
    from bs4 import MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import Counter
from html_duplicate_cleanup import remove_duplicate_heading_paragraph_pairs
from html_tag_entities import fix_stray_p_gt_artifacts, unescape_valid_html_tag_entities
from epub_metadata_utils import (
    extract_dc_metadata,
    restore_truncated_repeatable_metadata,
)
from epub_package import find_epub_opf_member, find_opf_path

_DEFAULT_SPECIAL_KEYWORDS = [
    'cover', 'title', 'toc', 'copyright', 'preface', 'nav', 'message',
    'notice', 'colophon', 'dedication', 'epigraph', 'foreword',
    'acknowledgment', 'author', 'appendix', 'bibliography'
]
_DEFAULT_SPECIAL_EXACT = ['cover', 'index', 'glossary', 'glossary_extension']

_REMOTE_CACHE_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.svg', '.bmp', '.webp'
}

_CHAPTER_EXTRACTION_CACHE_VERSION = 1
_CHAPTER_EXTRACTION_MARKER_NAME = '.chapters_extracted'
_CHAPTERS_FULL_NAME = 'chapters_full.json'
_CHAPTERS_INFO_NAME = 'chapters_info.json'


def _source_epub_image_count_from_zip(zf):
    """Count image resources packaged in the currently open source EPUB."""
    try:
        return sum(
            1
            for info in zf.infolist()
            if (
                not info.is_dir()
                and os.path.splitext(info.filename)[1].lower()
                in _REMOTE_CACHE_IMAGE_EXTENSIONS
            )
        )
    except (AttributeError, OSError, ValueError):
        return None


def _tracked_remote_cache_source_image_count(manifest, images_dir):
    """Return the source-image count recorded by a remote download cache."""
    if not isinstance(manifest, dict):
        return None
    try:
        tracked_count = int(manifest.get('source_epub_image_count'))
        if tracked_count >= 0:
            return tracked_count
    except (TypeError, ValueError):
        pass

    # Version-1 manifests did not store the source count. Infer it from the
    # cached output total, excluding completed remote images that still exist.
    try:
        cached_total = int(manifest.get('cached_image_file_count'))
    except (TypeError, ValueError):
        cached_total = -1
    if cached_total < 0 and os.path.isdir(images_dir):
        try:
            cached_total = sum(
                1
                for name in os.listdir(images_dir)
                if (
                    os.path.isfile(os.path.join(images_dir, name))
                    and os.path.splitext(name)[1].lower()
                    in _REMOTE_CACHE_IMAGE_EXTENSIONS
                )
            )
        except OSError:
            cached_total = -1
    if cached_total < 0:
        return None

    localized_count = 0
    for item in manifest.get('items', []):
        if not isinstance(item, dict) or item.get('status') != 'completed':
            continue
        filename = os.path.basename(str(item.get('filename') or ''))
        if filename and os.path.isfile(os.path.join(images_dir, filename)):
            localized_count += 1
    return max(0, cached_total - localized_count)


def _remote_image_cache_matches_source(output_dir, source_epub_image_count):
    """Validate the preserved cache against the current EPUB image count."""
    if os.getenv('DOWNLOAD_REMOTE_IMAGE_URLS', '0').strip().lower() not in {
        '1', 'true', 'yes', 'on'
    }:
        return False
    try:
        source_epub_image_count = int(source_epub_image_count)
    except (TypeError, ValueError):
        return False
    if source_epub_image_count < 0:
        return False

    images_dir = os.path.join(output_dir, 'images')
    manifest_path = os.path.join(
        images_dir, '.cache', 'remote_image_download_progress.json'
    )
    if not os.path.isfile(manifest_path):
        return False
    try:
        with open(manifest_path, 'r', encoding='utf-8') as handle:
            manifest = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False

    tracked_count = _tracked_remote_cache_source_image_count(
        manifest,
        images_dir,
    )
    return tracked_count == source_epub_image_count


def _special_file_stem(filename):
    base = os.path.basename(str(filename or '')).lower()
    if base.startswith('response_'):
        base = base[len('response_'):]
    while True:
        stem, ext = os.path.splitext(base)
        if ext.lower() not in {'.html', '.xhtml', '.htm', '.xml', '.txt'}:
            break
        base = stem
    return base


def _is_configured_special_file(filename):
    stem = _special_file_stem(filename)
    if not stem:
        return False
    kw_env = os.getenv('SPECIAL_FILE_KEYWORDS', '')
    exact_env = os.getenv('SPECIAL_FILE_EXACT', '')
    keywords = [k.strip().lower() for k in kw_env.split(',') if k.strip()] if kw_env else _DEFAULT_SPECIAL_KEYWORDS
    exact = [k.strip().lower() for k in exact_env.split(',') if k.strip()] if exact_env else _DEFAULT_SPECIAL_EXACT
    return stem in exact or any(keyword in stem for keyword in keywords)

# ---------------------------------------------------------------------------
# Inlined pattern constants for chapter extraction.
# These MUST NOT trigger TransateKRtoEN import (which takes ~18s).
# PPE worker functions use these instead of PM.* to avoid that import.
# ---------------------------------------------------------------------------
_CHAPTER_PATTERNS = [
    (r'chapter[\s_-]*(\d+)', re.IGNORECASE, 'english_chapter'),
    (r'\bch\.?\s*(\d+)\b', re.IGNORECASE, 'english_ch'),
    (r'part[\s_-]*(\d+)', re.IGNORECASE, 'english_part'),
    (r'episode[\s_-]*(\d+)', re.IGNORECASE, 'english_episode'),
    (r'第\s*(\d+)\s*[章节話话回]', 0, 'chinese_chapter'),
    (r'第\s*([一二三四五六七八九十百千万]+)\s*[章节話话回]', 0, 'chinese_chapter_cn'),
    (r'(\d+)[章节話话回]', 0, 'chinese_short'),
    (r'第\s*(\d+)\s*話', 0, 'japanese_wa'),
    (r'第\s*(\d+)\s*章', 0, 'japanese_chapter'),
    (r'その\s*(\d+)', 0, 'japanese_sono'),
    (r'(\d+)話目', 0, 'japanese_wame'),
    (r'제\s*(\d+)\s*[장화권부편]', 0, 'korean_chapter'),
    (r'(\d+)\s*[장화권부편]', 0, 'korean_short'),
    (r'에피소드\s*(\d+)', 0, 'korean_episode'),
    (r'^\s*(\d+)\s*[-–—.\:]', re.MULTILINE, 'generic_numbered'),
    (r'_(\d+)\.x?html?$', re.IGNORECASE, 'filename_number'),
    (r'/(\d+)\.x?html?$', re.IGNORECASE, 'path_number'),
    (r'(\d+)', 0, 'any_number'),
]

_CHINESE_NUMS = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
    '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
    '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24, '二十五': 25,
    '三十': 30, '四十': 40, '五十': 50, '六十': 60, '七十': 70, '八十': 80, '九十': 90, '百': 100,
    '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5, '陆': 6, '柒': 7, '捌': 8, '玖': 9, '拾': 10,
    '佰': 100, '仟': 1000, '萬': 10000, '万': 10000,
    '第一': 1, '第二': 2, '第三': 3, '第四': 4, '第五': 5,
    '首': 1, '次': 2, '初': 1, '末': -1,
}

_FILENAME_EXTRACT_PATTERNS = [
    r'^\d{3}(\d)_(\d{2})_\.x?html?$',
    r'^\d{4}_(\d+)\.x?html?$',
    r'^\d+_(\d+)[_\.]',
    r'^(\d+)[_\.]',
    r'response_(\d+)_',
    r'response_(\d+)\.',
    r'(\d{3,5})[_\.]',
    r'[Cc]hapter[_\s]*(\d+)',
    r'[Cc]h[_\s]*(\d+)',
    r'No(\d+)Chapter',
    r'No(\d+)Section',
    r'No(\d+)(?=\.|_|$)',
    r'第(\d+)[章话回]',
    r'_(\d+)(?:_|\.|$)',
    r'^(\d+)(?:_|\.|$)',
    r'(\d+)',
]

# Stop request function (can be overridden)
def is_stop_requested():
    """Check if stop has been requested - default implementation"""
    return False

# Progress bar for terminal output
class ProgressBar:
    """Simple in-place progress bar for terminal output"""
    _last_line_length = 0
    
    @classmethod
    def update(cls, current, total, prefix="Progress", bar_length=30):
        if total == 0:
            return
        percent = min(100, int(100 * current / total))
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        line = f"\r{prefix}: [{bar}] {current}/{total} ({percent}%)"
        if len(line) < cls._last_line_length:
            line += ' ' * (cls._last_line_length - len(line))
        cls._last_line_length = len(line)
        print(line, end='', flush=True)
    
    @classmethod
    def finish(cls):
        print()
        cls._last_line_length = 0

# Helper for resource filename sanitization
def sanitize_resource_filename(filename):
    """Sanitize resource filenames to be filesystem-safe"""
    import unicodedata
    # Normalize unicode - use NFC to preserve Korean/CJK characters
    # NFKD decomposes Korean Hangul into jamo components, corrupting them
    filename = unicodedata.normalize('NFC', filename)
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return filename

def _collect_image_srcs(soup):
    """Collect all image source URLs from a BeautifulSoup document.
    
    Supports: <img src>, SVG <image href>, <object data>, <video poster>,
    and CSS background-image: url(...) in inline styles.
    """
    image_srcs = []
    
    # <img src>
    for tag in soup.find_all('img'):
        src = tag.get('src', '')
        if src:
            image_srcs.append(src)
    
    # SVG <image> (xlink:href / href) — one per tag
    for tag in soup.find_all('image'):
        href = (tag.get('xlink:href') or 
                tag.get('href') or 
                tag.get('{http://www.w3.org/1999/xlink}href') or '')
        if href:
            image_srcs.append(href)
    
    # <object data> (embedded images/SVGs)
    for tag in soup.find_all('object'):
        data = tag.get('data', '')
        if data:
            image_srcs.append(data)
    
    # <video poster>
    for tag in soup.find_all('video'):
        poster = tag.get('poster', '')
        if poster:
            image_srcs.append(poster)
    
    # CSS background-image: url(...) in inline styles
    for tag in soup.find_all(style=True):
        style = tag.get('style', '')
        if 'url(' in style:
            for m in re.finditer(r'url\(["\']?([^"\')\s]+)["\']?\)', style):
                url = m.group(1)
                if url:
                    image_srcs.append(url)
    
    return image_srcs


def _is_remote_image_url(value):
    """Return True when *value* is an absolute HTTP(S) image reference."""
    if not isinstance(value, str):
        return False
    try:
        return urllib.parse.urlsplit(value.strip()).scheme.lower() in {'http', 'https'}
    except (TypeError, ValueError):
        return False


def _convert_remote_image_to_png(image_bytes):
    """Convert downloaded raster (or SVG, when supported) bytes to real PNG data."""
    if not image_bytes:
        raise ValueError("Remote image response was empty")

    stripped = image_bytes.lstrip()
    looks_like_svg = (
        stripped.startswith(b'<svg')
        or (stripped.startswith(b'<?xml') and b'<svg' in stripped[:4096].lower())
    )
    if looks_like_svg:
        try:
            from cairosvg import svg2png
        except ImportError as exc:
            raise ValueError("Remote SVG conversion requires CairoSVG") from exc
        return svg2png(bytestring=image_bytes)

    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(image_bytes)) as source_image:
        source_image.seek(0)
        source_image.load()
        image = ImageOps.exif_transpose(source_image)
        has_transparency = (
            'A' in image.getbands()
            or (image.mode == 'P' and 'transparency' in image.info)
        )
        if has_transparency:
            image = image.convert('RGBA')
        elif image.mode not in {'RGB', 'L'}:
            image = image.convert('RGB')

        output = io.BytesIO()
        image.save(output, format='PNG')
        return output.getvalue()


def _download_remote_image_as_png(remote_url):
    """Download one remote image and return it converted to PNG bytes."""
    parsed = urllib.parse.urlsplit(remote_url)
    request_url = urllib.parse.urlunsplit(parsed._replace(fragment=''))
    origin = f"{parsed.scheme}://{parsed.netloc}/"
    request = urllib.request.Request(
        request_url,
        headers={
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/126.0 Safari/537.36'
            ),
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Referer': origin,
        },
    )
    try:
        timeout = max(1.0, float(os.getenv('REMOTE_IMAGE_DOWNLOAD_TIMEOUT', '60')))
    except (TypeError, ValueError):
        timeout = 60.0

    response = urllib.request.urlopen(request, timeout=timeout)
    try:
        image_bytes = response.read()
    finally:
        close = getattr(response, 'close', None)
        if callable(close):
            close()
    return _convert_remote_image_to_png(image_bytes)


class _RemoteImageStartThrottle:
    """Serialize remote-image request starts with a configurable interval."""

    def __init__(self, interval_seconds, monotonic=None, sleeper=None):
        try:
            interval_seconds = float(interval_seconds)
        except (TypeError, ValueError):
            interval_seconds = 0.0
        self.interval_seconds = max(0.0, interval_seconds)
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleeper or time.sleep
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self):
        """Wait until this request may start, then reserve the next slot."""
        if self.interval_seconds <= 0:
            return 0.0
        with self._lock:
            now = self._monotonic()
            delay = max(0.0, self._next_start - now)
            if delay > 0:
                self._sleep(delay)
                now = self._monotonic()
            self._next_start = max(now, self._next_start) + self.interval_seconds
            return delay


def _replace_remote_image_refs_in_soup(soup, replacements):
    """Rewrite successfully downloaded remote image references in one document."""
    modified = False

    def _replace_attr(tag, attr):
        nonlocal modified
        value = tag.get(attr, '')
        if not isinstance(value, str):
            return
        replacement = replacements.get(value.strip())
        if replacement:
            tag[attr] = replacement
            modified = True

    for tag in soup.find_all('img'):
        _replace_attr(tag, 'src')

    for tag in soup.find_all('image'):
        for attr in ['xlink:href', 'href', '{http://www.w3.org/1999/xlink}href']:
            _replace_attr(tag, attr)

    for tag in soup.find_all('object'):
        _replace_attr(tag, 'data')

    for tag in soup.find_all('video'):
        _replace_attr(tag, 'poster')

    for tag in soup.find_all(style=True):
        style = tag.get('style', '')
        if not isinstance(style, str) or 'url(' not in style:
            continue
        new_style = style
        for match in re.finditer(r'url\(["\']?([^"\')\s]+)["\']?\)', style):
            remote_url = match.group(1)
            replacement = replacements.get(remote_url.strip())
            if replacement:
                new_style = new_style.replace(remote_url, replacement)
        if new_style != style:
            tag['style'] = new_style
            modified = True

    return modified


def _localize_remote_images(
    chapters,
    output_dir,
    progress_callback=None,
    source_epub_image_count=None,
):
    """Download remote chapter image references and rewrite them to local PNGs.

    This deliberately runs before ``_rename_images_to_chapter_format`` so the
    normal image ownership and rename-map logic treats downloaded files exactly
    like images that were packaged in the source EPUB.
    """
    markup_keys = ('body', 'original_html', 'source_html', 'raw_html')
    remote_urls = []
    seen_urls = set()

    def _collect_markup(markup):
        if not isinstance(markup, str) or not markup:
            return
        try:
            soup = BeautifulSoup(markup, 'html.parser')
            candidates = _collect_image_srcs(soup)
        except Exception:
            return
        for candidate in candidates:
            if not _is_remote_image_url(candidate):
                continue
            remote_url = candidate.strip()
            if remote_url not in seen_urls:
                seen_urls.add(remote_url)
                remote_urls.append(remote_url)

    for chapter in chapters:
        for key in markup_keys:
            _collect_markup(chapter.get(key))

    disk_html_paths = []
    if os.path.isdir(output_dir):
        try:
            for root, directories, files in os.walk(output_dir):
                directories[:] = [
                    name for name in directories
                    if name.casefold() != '.cache'
                ]
                for filename in files:
                    if not filename.lower().endswith(('.html', '.xhtml', '.htm')):
                        continue
                    path = os.path.join(root, filename)
                    disk_html_paths.append(path)
                    try:
                        with open(path, 'r', encoding='utf-8') as handle:
                            _collect_markup(handle.read())
                    except Exception as exc:
                        print(f"   Warning: could not scan remote images in {path}: {exc}")
        except Exception as exc:
            print(f"   Warning: could not scan output HTML files for remote images: {exc}")

    if not remote_urls:
        message = "Remote image download enabled: no HTTP/HTTPS image URLs found"
        if progress_callback:
            progress_callback(message)
        else:
            print(message)
        return chapters

    images_dir = os.path.join(output_dir, 'images')
    cache_dir = os.path.join(images_dir, '.cache')
    progress_path = os.path.join(
        cache_dir,
        'remote_image_download_progress.json',
    )
    os.makedirs(cache_dir, exist_ok=True)
    replacements = {}

    def _timestamp():
        return datetime.now(timezone.utc).isoformat(timespec='seconds')

    def _write_progress_manifest(manifest):
        """Atomically persist progress so interrupted runs remain readable."""
        manifest['updated_at'] = _timestamp()
        temporary = (
            f"{progress_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with open(temporary, 'w', encoding='utf-8') as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, progress_path)
        finally:
            if os.path.exists(temporary):
                try:
                    os.remove(temporary)
                except OSError:
                    pass

    previous_items = {}
    try:
        with open(progress_path, 'r', encoding='utf-8') as handle:
            previous_manifest = json.load(handle)
        if isinstance(previous_manifest, dict):
            previous_items = {
                item.get('url'): item
                for item in previous_manifest.get('items', [])
                if isinstance(item, dict) and item.get('url')
            }
    except (OSError, ValueError, TypeError):
        previous_items = {}

    # The normal chapter-image pass renames ``remote_<url hash>.png`` to a
    # chapter-owned filename. A run interrupted after its progress manifest is
    # reset to ``pending`` can therefore have a perfectly valid cached PNG that
    # is only discoverable through the previous image rename map. Treat the map
    # as the authoritative filename bridge instead of redownloading the URL.
    rename_map = {}
    rename_map_path = os.path.join(output_dir, 'image_rename_map.json')
    try:
        with open(rename_map_path, 'r', encoding='utf-8') as handle:
            loaded_rename_map = json.load(handle)
        if isinstance(loaded_rename_map, dict):
            rename_map = {
                os.path.basename(str(old_name).replace('\\', '/')):
                os.path.basename(str(new_name).replace('\\', '/'))
                for old_name, new_name in loaded_rename_map.items()
                if old_name and new_name
            }
    except (OSError, ValueError, TypeError):
        rename_map = {}
    rename_map_casefold = {
        old_name.casefold(): new_name
        for old_name, new_name in rename_map.items()
    }

    def _cache_filename_candidates(*recorded_names):
        """Yield mapped cache names, preferring the terminal rename target."""
        emitted = set()
        for recorded_name in recorded_names:
            current = os.path.basename(str(recorded_name or '').replace('\\', '/'))
            if not current:
                continue
            chain = []
            seen = set()
            while current and current.casefold() not in seen:
                seen.add(current.casefold())
                chain.append(current)
                current = (
                    rename_map.get(current)
                    or rename_map_casefold.get(current.casefold())
                    or ''
                )
            for candidate in reversed(chain):
                key = candidate.casefold()
                if key not in emitted:
                    emitted.add(key)
                    yield candidate

    try:
        configured_workers = int(os.getenv('REMOTE_IMAGE_DOWNLOAD_WORKERS', '4'))
    except (TypeError, ValueError):
        configured_workers = 4
    worker_count = min(len(remote_urls), max(1, configured_workers))
    try:
        download_interval = float(os.getenv(
            'REMOTE_IMAGE_DOWNLOAD_INTERVAL', '0.5'
        ))
    except (TypeError, ValueError):
        download_interval = 0.5
    download_interval = max(0.0, min(60.0, download_interval))
    request_throttle = _RemoteImageStartThrottle(download_interval)

    total_remote_urls = len(remote_urls)
    download_started_at = time.monotonic()
    started_at = _timestamp()
    item_by_url = {}
    pending_urls = []
    completed = 0
    successful = 0
    failed = 0
    resumed = 0
    downloaded_bytes = 0

    for remote_url in remote_urls:
        download_filename = (
            'remote_'
            f"{hashlib.sha256(remote_url.encode('utf-8')).hexdigest()[:20]}"
            '.png'
        )
        previous_item = previous_items.get(remote_url, {})
        cached_filename = download_filename
        cached_path = os.path.join(images_dir, cached_filename)
        cached_is_png = False
        for candidate_filename in _cache_filename_candidates(
            previous_item.get('filename'),
            previous_item.get('download_filename'),
            download_filename,
        ):
            candidate_path = os.path.join(images_dir, candidate_filename)
            if not os.path.isfile(candidate_path):
                continue
            try:
                with open(candidate_path, 'rb') as handle:
                    candidate_is_png = handle.read(8) == b'\x89PNG\r\n\x1a\n'
            except OSError:
                candidate_is_png = False
            if candidate_is_png:
                cached_filename = candidate_filename
                cached_path = candidate_path
                cached_is_png = True
                break

        item = {
            'url': remote_url,
            'status': 'pending',
            'download_filename': download_filename,
            'filename': download_filename,
            'local_reference': f'images/{download_filename}',
            'bytes': 0,
            'error': None,
            'updated_at': started_at,
        }
        if cached_is_png:
            cached_size = os.path.getsize(cached_path)
            item.update({
                'status': 'completed',
                'download_filename': str(
                    previous_item.get('download_filename') or download_filename
                ),
                'filename': cached_filename,
                'local_reference': f'images/{cached_filename}',
                'bytes': cached_size,
                'completed_at': previous_item.get('completed_at'),
                'updated_at': previous_item.get('updated_at') or started_at,
            })
            replacements[remote_url] = item['local_reference']
            completed += 1
            successful += 1
            resumed += 1
            downloaded_bytes += cached_size
        else:
            pending_urls.append(remote_url)
        item_by_url[remote_url] = item

    try:
        source_epub_image_count = max(0, int(source_epub_image_count))
    except (TypeError, ValueError):
        source_epub_image_count = None

    progress_manifest = {
        'version': 2,
        'status': 'downloading',
        'output_format': 'png',
        'source_epub_image_count': source_epub_image_count,
        'total': total_remote_urls,
        'completed': completed,
        'successful': successful,
        'failed': failed,
        'resumed': resumed,
        'downloaded_bytes': downloaded_bytes,
        'progress_percent': int((completed * 100) / total_remote_urls),
        'workers': worker_count,
        'request_start_interval_seconds': download_interval,
        'started_at': started_at,
        'updated_at': started_at,
        'items': [item_by_url[url] for url in remote_urls],
    }

    def _persist_progress(status=None, completed_at=False):
        if status is not None:
            progress_manifest['status'] = status
        progress_manifest.update({
            'completed': completed,
            'successful': successful,
            'failed': failed,
            'resumed': resumed,
            'downloaded_bytes': downloaded_bytes,
            'progress_percent': int(
                (completed * 100) / total_remote_urls
            ),
        })
        if completed_at:
            progress_manifest['completed_at'] = _timestamp()
        _write_progress_manifest(progress_manifest)

    def _progress_message(completed_count, success_count, failure_count, byte_count):
        elapsed = max(0.001, time.monotonic() - download_started_at)
        percent = int((completed_count * 100) / total_remote_urls)
        rate = completed_count / elapsed if completed_count else 0.0
        remaining = total_remote_urls - completed_count
        eta_seconds = int(remaining / rate) if rate > 0 else 0
        if completed_count == 0:
            eta_label = "calculating"
        elif eta_seconds >= 60:
            eta_label = f"{eta_seconds // 60}m {eta_seconds % 60}s"
        else:
            eta_label = f"{eta_seconds}s"
        size_mib = byte_count / (1024 * 1024)
        return (
            f"Downloading remote images: {completed_count}/{total_remote_urls} "
            f"({percent}%) | {success_count} saved, {failure_count} failed | "
            f"{size_mib:.1f} MiB | {rate:.1f} images/s | ETA {eta_label}"
        )

    _persist_progress()
    initial_progress = _progress_message(
        completed,
        successful,
        failed,
        downloaded_bytes,
    )
    if progress_callback:
        progress_callback(initial_progress)
    else:
        print(initial_progress)
    cache_message = (
        f"Remote image progress cache: {progress_path} | "
        f"{resumed} cached PNG(s) restored out of {total_remote_urls}"
    )
    if progress_callback:
        progress_callback(cache_message)
    else:
        print(cache_message)
    pacing_message = (
        f"⏱️ Remote image download pacing: {worker_count} thread(s), "
        f"{download_interval:g}s between request starts"
    )
    if progress_callback:
        progress_callback(pacing_message)
    else:
        print(pacing_message)

    def _download_and_store(remote_url):
        filename = item_by_url[remote_url]['download_filename']
        destination = os.path.join(images_dir, filename)
        request_throttle.wait()
        png_bytes = _download_remote_image_as_png(remote_url)
        temporary = f"{destination}.{threading.get_ident()}.tmp"
        try:
            with open(temporary, 'wb') as handle:
                handle.write(png_bytes)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                try:
                    os.remove(temporary)
                except OSError:
                    pass
        return remote_url, f"images/{filename}", len(png_bytes)

    failure_details = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_url = {
            executor.submit(_download_and_store, remote_url): remote_url
            for remote_url in pending_urls
        }
        for future in as_completed(future_to_url):
            remote_url = future_to_url[future]
            completed += 1
            try:
                downloaded_url, local_reference, image_size = future.result()
                replacements[downloaded_url] = local_reference
                successful += 1
                downloaded_bytes += image_size
                item_by_url[remote_url].update({
                    'status': 'completed',
                    'filename': os.path.basename(local_reference),
                    'local_reference': local_reference,
                    'bytes': image_size,
                    'error': None,
                    'completed_at': _timestamp(),
                    'updated_at': _timestamp(),
                })
            except Exception as exc:
                failed += 1
                failure_details.append((remote_url, str(exc)))
                item_by_url[remote_url].update({
                    'status': 'failed',
                    'bytes': 0,
                    'error': str(exc),
                    'updated_at': _timestamp(),
                })

            _persist_progress()

            progress_message = _progress_message(
                completed, successful, failed, downloaded_bytes
            )
            if progress_callback:
                progress_callback(progress_message)
            else:
                ProgressBar.update(
                    completed,
                    total_remote_urls,
                    prefix=(
                        f"Remote PNGs | {successful} saved, {failed} failed | "
                        f"{downloaded_bytes / (1024 * 1024):.1f} MiB"
                    ),
                )

    if not progress_callback:
        ProgressBar.finish()

    for remote_url, error in failure_details[:10]:
        warning = (
            "Warning: remote image download failed; keeping original URL "
            f"{remote_url}: {error}"
        )
        if progress_callback:
            progress_callback(warning)
        else:
            print(warning)
    if len(failure_details) > 10:
        warning = (
            f"Warning: {len(failure_details) - 10} additional remote image "
            "download failure(s) omitted from the log"
        )
        if progress_callback:
            progress_callback(warning)
        else:
            print(warning)

    if not replacements:
        progress_manifest['chapters_updated'] = 0
        progress_manifest['html_files_updated'] = 0
        _persist_progress(status='failed', completed_at=True)
        message = (
            f"Remote image localization complete: 0/{total_remote_urls} saved, "
            f"{failed} failed; no HTML references changed"
        )
        if progress_callback:
            progress_callback(message)
        else:
            print(message)
        return chapters

    updated_chapters = 0
    for chapter in chapters:
        chapter_modified = False
        for key in markup_keys:
            markup = chapter.get(key)
            if not isinstance(markup, str) or not markup:
                continue
            try:
                soup = BeautifulSoup(markup, 'html.parser')
                if _replace_remote_image_refs_in_soup(soup, replacements):
                    chapter[key] = str(soup)
                    chapter_modified = True
            except Exception as exc:
                print(f"   Warning: could not rewrite remote images in chapter {chapter.get('num', '?')}: {exc}")
        if chapter_modified:
            updated_chapters += 1

    disk_updated = 0
    for path in disk_html_paths:
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                content = handle.read()
            soup = BeautifulSoup(content, 'html.parser')
            if _replace_remote_image_refs_in_soup(soup, replacements):
                with open(path, 'w', encoding='utf-8') as handle:
                    handle.write(str(soup))
                disk_updated += 1
        except Exception as exc:
            print(f"   Warning: could not rewrite remote images in {path}: {exc}")

    print(
        f"🖼️ Remote image download complete: {len(replacements)}/{len(remote_urls)} localized "
        f"as PNG ({updated_chapters} chapter(s), {disk_updated} saved HTML file(s) updated)"
    )
    progress_manifest['chapters_updated'] = updated_chapters
    progress_manifest['html_files_updated'] = disk_updated
    try:
        progress_manifest['cached_image_file_count'] = sum(
            1
            for name in os.listdir(images_dir)
            if (
                os.path.isfile(os.path.join(images_dir, name))
                and os.path.splitext(name)[1].lower()
                in {'.jpg', '.jpeg', '.png', '.gif', '.svg', '.bmp', '.webp'}
            )
        )
    except OSError:
        pass
    final_status = 'completed' if failed == 0 else 'completed_with_errors'
    _persist_progress(status=final_status, completed_at=True)
    if progress_callback:
        progress_callback(
            f"Remote image localization complete: {successful}/{total_remote_urls} "
            f"saved, {failed} failed; {updated_chapters} chapter(s) updated"
        )
    return chapters


def _record_remote_image_renames(output_dir, successful_renames):
    """Keep the persistent remote-download manifest aligned after renaming."""
    if not successful_renames:
        return
    progress_path = os.path.join(
        output_dir,
        'images',
        '.cache',
        'remote_image_download_progress.json',
    )
    try:
        with open(progress_path, 'r', encoding='utf-8') as handle:
            manifest = json.load(handle)
    except (OSError, ValueError, TypeError):
        return
    if not isinstance(manifest, dict):
        return

    changed = False
    renamed_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    for item in manifest.get('items', []):
        if not isinstance(item, dict) or item.get('status') != 'completed':
            continue
        current_name = os.path.basename(str(item.get('filename') or ''))
        final_name = successful_renames.get(current_name)
        if not final_name:
            continue
        item['filename'] = final_name
        item['local_reference'] = f'images/{final_name}'
        item['renamed_at'] = renamed_at
        item['updated_at'] = renamed_at
        changed = True

    if not changed:
        return
    manifest['updated_at'] = renamed_at
    temporary = f"{progress_path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, progress_path)
    finally:
        if os.path.exists(temporary):
            try:
                os.remove(temporary)
            except OSError:
                pass


def _update_image_refs_in_soup(soup, rename_map):
    """Update all image references in a BeautifulSoup document using the rename map.
    
    Returns True if any references were modified.
    Supports: <img src>, SVG <image href>, <object data>, <video poster>,
    and CSS background-image: url(...) in inline styles.
    """
    modified = False
    
    def _update_attr(tag, attr):
        nonlocal modified
        val = tag.get(attr, '')
        if not val or val.startswith('data:'):
            return
        clean = val.split('?')[0]
        basename = os.path.basename(clean)
        if basename in rename_map:
            new_name = rename_map[basename]
            dir_part = os.path.dirname(clean)
            tag[attr] = f"{dir_part}/{new_name}" if dir_part else new_name
            modified = True
    
    # <img src>
    for tag in soup.find_all('img'):
        _update_attr(tag, 'src')
    
    # SVG <image> (all href variants)
    for tag in soup.find_all('image'):
        for attr in ['xlink:href', 'href', '{http://www.w3.org/1999/xlink}href']:
            _update_attr(tag, attr)
    
    # <object data>
    for tag in soup.find_all('object'):
        _update_attr(tag, 'data')
    
    # <video poster>
    for tag in soup.find_all('video'):
        _update_attr(tag, 'poster')
    
    # CSS background-image: url(...) in inline styles
    for tag in soup.find_all(style=True):
        style = tag.get('style', '')
        if 'url(' not in style:
            continue
        new_style = style
        for m in re.finditer(r'url\(["\']?([^"\')\s]+)["\']?\)', style):
            url = m.group(1)
            if not url or url.startswith('data:'):
                continue
            clean = url.split('?')[0]
            basename = os.path.basename(clean)
            if basename in rename_map:
                new_name = rename_map[basename]
                dir_part = os.path.dirname(clean)
                new_url = f"{dir_part}/{new_name}" if dir_part else new_name
                new_style = new_style.replace(url, new_url)
                modified = True
        if new_style != style:
            tag['style'] = new_style
    
    return modified


def _prepare_single_chapter_image_renames(
    chapters,
    output_dir,
    progress_callback=None,
    status_context='Single-chapter mode',
):
    """Prepare safe canonical image names during a targeted translation.

    Library / Reader single-chapter runs deliberately cannot call
    :func:`_rename_images_to_chapter_format` with their one-chapter subset: doing
    so would assign every other image in the book to the selected chapter.  A
    previous full extraction normally recorded the authoritative assignments in
    ``image_rename_map.json``, so targeted retries replay that map and update the
    freshly extracted chapter markup.  On a first-ever Reader translation, only
    images referenced by the selected chapter are assigned; unrelated book
    images remain untouched.

    Replaying the physical renames also repairs workspaces damaged by an older
    targeted run that restored the source EPUB filenames in ``images/``.
    """
    rename_map_path = os.path.join(output_dir, 'image_rename_map.json')
    images_dir = os.path.join(output_dir, 'images')

    map_was_missing = False
    try:
        with open(rename_map_path, 'r', encoding='utf-8') as f:
            loaded_map = json.load(f)
    except FileNotFoundError:
        loaded_map = {}
        map_was_missing = True
    except Exception as e:
        print(f"⚠️ Could not load existing image rename map: {e}")
        loaded_map = {}

    if not isinstance(loaded_map, dict):
        loaded_map = {}

    # Rename maps contain basenames, not paths.  Enforce that invariant before
    # touching the filesystem so malformed/stale metadata cannot escape images/.
    rename_map = {}
    for original, renamed in loaded_map.items():
        original_name = os.path.basename(str(original or '').replace('\\', '/'))
        renamed_name = os.path.basename(str(renamed or '').replace('\\', '/'))
        if original_name and renamed_name:
            rename_map[original_name] = renamed_name

    # Repair passes can extend the map (original -> prior canonical -> current
    # canonical).  Collapse those chains so a restored source file goes directly
    # to the final name and fresh source markup receives that same final ref.
    collapsed_map = {}
    for original_name in rename_map:
        renamed_name = rename_map[original_name]
        seen = {original_name}
        while renamed_name in rename_map and renamed_name not in seen:
            seen.add(renamed_name)
            next_name = rename_map[renamed_name]
            if next_name == renamed_name:
                break
            renamed_name = next_name
        collapsed_map[original_name] = renamed_name
    rename_map = collapsed_map

    try:
        existing_images = {
            name for name in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, name))
        }
    except OSError:
        existing_images = set()

    # A reader can start a targeted translation before the book has ever had a
    # full translation run.  In that case there is no persisted map yet.  Claim
    # only images actually referenced by the selected chapter; unlike the full
    # pass, never rename unrelated files as covers.
    new_mapping_keys = set()
    reserved_names = set(existing_images) | set(rename_map.values())
    for chapter in chapters:
        chapter_basename = chapter.get('original_basename', '')
        if not chapter_basename:
            chapter_filename = chapter.get('filename', '')
            chapter_basename = (
                os.path.splitext(os.path.basename(chapter_filename))[0]
                if chapter_filename else ''
            )
        if not chapter_basename:
            chapter_basename = f"chapter{int(chapter.get('num', 0)):03d}"
        chapter_stem = os.path.splitext(chapter_basename)[0]

        image_srcs = []
        seen_srcs = set()
        for html_key in ('body', 'original_html', 'source_html', 'raw_html'):
            markup = chapter.get(html_key)
            if not markup:
                continue
            try:
                markup_srcs = _collect_image_srcs(
                    BeautifulSoup(markup, 'html.parser')
                )
            except Exception:
                continue
            for src in markup_srcs:
                if src not in seen_srcs:
                    seen_srcs.add(src)
                    image_srcs.append(src)

        image_counter = 1
        for src in image_srcs:
            if not src or src.startswith('data:'):
                continue
            basename = os.path.basename(src.split('?', 1)[0])
            if not basename or basename in rename_map:
                continue
            if basename not in existing_images:
                basename = next(
                    (name for name in existing_images
                     if name.lower() == basename.lower()),
                    basename,
                )
            if basename not in existing_images:
                continue
            if re.match(r'^.+_img_\d+(?:\.[^.]+)?$', basename, re.IGNORECASE):
                continue

            extension = os.path.splitext(basename)[1]
            renamed_name = f"{chapter_stem}_img_{image_counter}{extension}"
            while renamed_name in reserved_names:
                image_counter += 1
                renamed_name = f"{chapter_stem}_img_{image_counter}{extension}"
            rename_map[basename] = renamed_name
            new_mapping_keys.add(basename)
            reserved_names.add(renamed_name)
            image_counter += 1

    restored_count = 0
    if os.path.isdir(images_dir):
        # Use temporary names just like the full rename pass.  This keeps replay
        # safe when one map destination happens to be another map source.
        temp_renames = []
        for index, (original_name, renamed_name) in enumerate(
            list(rename_map.items())
        ):
            if original_name == renamed_name:
                continue
            original_path = os.path.join(images_dir, original_name)
            renamed_path = os.path.join(images_dir, renamed_name)
            if not os.path.isfile(original_path) or os.path.exists(renamed_path):
                continue
            temp_name = f"_temp_restore_{index}_{original_name}"
            temp_path = os.path.join(images_dir, temp_name)
            try:
                os.rename(original_path, temp_path)
                temp_renames.append(
                    (original_name, temp_path, original_path, renamed_path)
                )
            except OSError as e:
                print(f"   ⚠️ Could not stage image restore {original_name}: {e}")
                if original_name in new_mapping_keys:
                    rename_map.pop(original_name, None)
                    new_mapping_keys.discard(original_name)

        for original_name, temp_path, original_path, renamed_path in temp_renames:
            try:
                if os.path.exists(renamed_path):
                    os.rename(temp_path, original_path)
                    continue
                os.rename(temp_path, renamed_path)
                restored_count += 1
            except OSError as e:
                print(
                    f"   ⚠️ Could not restore image name "
                    f"{os.path.basename(original_path)} → "
                    f"{os.path.basename(renamed_path)}: {e}"
                )
                try:
                    if os.path.exists(temp_path) and not os.path.exists(original_path):
                        os.rename(temp_path, original_path)
                except OSError:
                    pass
                if original_name in new_mapping_keys:
                    rename_map.pop(original_name, None)
                    new_mapping_keys.discard(original_name)

    try:
        existing_images = {
            name for name in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, name))
        }
    except OSError:
        existing_images = set()

    # Only rewrite a reference when its canonical destination is present.  A
    # stale map must never turn a currently valid source reference into a broken
    # one.
    available_map = {
        original: renamed
        for original, renamed in rename_map.items()
        if renamed in existing_images
    }

    if rename_map and (map_was_missing or new_mapping_keys or rename_map != loaded_map):
        try:
            with open(rename_map_path, 'w', encoding='utf-8') as f:
                json.dump(rename_map, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"⚠️ Could not save targeted image rename map: {e}")
    updated_chapters = 0
    if available_map:
        for chapter in chapters:
            chapter_modified = False
            for html_key in ('body', 'original_html', 'source_html', 'raw_html'):
                markup = chapter.get(html_key)
                if not markup:
                    continue
                try:
                    soup = BeautifulSoup(markup, 'html.parser')
                    if _update_image_refs_in_soup(soup, available_map):
                        chapter[html_key] = str(soup)
                        chapter_modified = True
                except Exception:
                    continue
            if chapter_modified:
                updated_chapters += 1

    if restored_count or updated_chapters or new_mapping_keys:
        message = (
            f"{status_context} image state restored: {restored_count} file(s) "
            f"renamed, {updated_chapters} chapter(s) updated"
        )
        print(f"🖼️ {message}")
        if progress_callback:
            progress_callback(message)
    else:
        print(f"🎯 {status_context}: existing image names already preserved")

    return chapters


def prepare_epub_image_assets(epub_path, output_dir, progress_callback=None):
    """Extract and canonically rename only EPUB images needed by review HTML.

    This is the resource-only counterpart of :func:`extract_chapters`: it does
    not run text extraction, create chapter payloads, or start translation.
    Existing complete image state is left untouched.
    """
    result = {
        "ready": False,
        "prepared": False,
        "source_images": 0,
        "extracted": 0,
        "renamed": 0,
        "error": "",
    }

    def _report(message):
        if not callable(progress_callback):
            return
        try:
            progress_callback(str(message))
        except Exception:
            # Status reporting must never prevent image preparation.
            pass

    epub_path = os.path.abspath(str(epub_path or ""))
    output_dir = os.path.abspath(str(output_dir or ""))
    if not os.path.isfile(epub_path) or not epub_path.lower().endswith(".epub"):
        result["error"] = "Source EPUB was not found"
        return result
    if not os.path.isdir(output_dir):
        result["error"] = "Output folder was not found"
        return result

    _report("🖼️ Preparing EPUB image assets for SDLXLIFF review...")

    images_dir = os.path.join(output_dir, "images")
    rename_map_path = os.path.join(output_dir, "image_rename_map.json")
    loaded_map = {}
    try:
        with open(rename_map_path, "r", encoding="utf-8") as handle:
            candidate_map = json.load(handle)
        if isinstance(candidate_map, dict):
            loaded_map = {
                os.path.basename(str(original).replace("\\", "/")):
                os.path.basename(str(renamed).replace("\\", "/"))
                for original, renamed in candidate_map.items()
                if original and renamed
            }
    except (OSError, ValueError, TypeError):
        pass

    if loaded_map and all(
        os.path.isfile(os.path.join(images_dir, renamed))
        for renamed in loaded_map.values()
    ):
        result.update({
            "ready": True,
            "renamed": len(loaded_map),
        })
        _report(
            f"✅ EPUB image assets are already prepared ({len(loaded_map)} file(s))"
        )
        return result

    try:
        import contextlib
        import zipfile
        from glossary_usage import read_epub_spine_chapters

        with zipfile.ZipFile(epub_path, "r") as source_zip:
            names = [name for name in source_zip.namelist() if name and not name.endswith("/")]
            image_members = [
                name for name in names
                if os.path.splitext(name)[1].lower() in _REMOTE_CACHE_IMAGE_EXTENSIONS
            ]
            result["source_images"] = len(image_members)
            if not image_members:
                result["ready"] = True
                _report("ℹ️ Source EPUB contains no image assets to prepare")
                return result

            os.makedirs(images_dir, exist_ok=True)
            existing_before = {
                name for name in os.listdir(images_dir)
                if os.path.isfile(os.path.join(images_dir, name))
            }
            loaded_lookup = {
                str(original).casefold(): renamed
                for original, renamed in loaded_map.items()
            }
            for member in image_members:
                safe_name = sanitize_resource_filename(os.path.basename(member))
                if not safe_name:
                    continue
                canonical_name = loaded_lookup.get(safe_name.casefold(), "")
                if canonical_name and os.path.isfile(os.path.join(images_dir, canonical_name)):
                    continue
                destination = os.path.join(images_dir, safe_name)
                if os.path.isfile(destination):
                    continue
                with source_zip.open(member, "r") as source, open(destination, "wb") as target:
                    shutil.copyfileobj(source, target)
                result["extracted"] += 1

            if result["extracted"]:
                _report(
                    f"📥 Extracted {result['extracted']} of {result['source_images']} "
                    "EPUB image asset(s)"
                )

            spine_members = []
            try:
                spine_members = [
                    str(entry.get("member_path") or "")
                    for entry in read_epub_spine_chapters(
                        epub_path, translate_special=True, include_text=False
                    )
                    if entry.get("member_path")
                ]
            except Exception:
                spine_members = []
            html_members = [
                name for name in names
                if name.lower().endswith((".xhtml", ".html", ".htm"))
            ]
            ordered_html = list(dict.fromkeys(
                spine_members + sorted(
                    (name for name in html_members if name not in set(spine_members)),
                    key=str.casefold,
                )
            ))
            chapters = []
            for chapter_number, member in enumerate(ordered_html, 1):
                try:
                    markup = source_zip.read(member).decode("utf-8", errors="ignore")
                except Exception:
                    continue
                chapters.append({
                    "num": chapter_number,
                    "filename": member,
                    "original_basename": os.path.basename(member),
                    "body": markup,
                })

        # A pristine workspace can use the full canonical pass. If files or a
        # map already exist, replay/extend that state without reclassifying
        # unrelated existing files as covers.
        # These legacy routines print emoji and source-language filenames.
        # The standalone SDLXLIFF viewer may inherit a narrow Windows console
        # encoding, where logging itself would otherwise abort preparation.
        _report("🔄 Applying chapter image rename map for SDLXLIFF review...")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if not loaded_map and not existing_before:
                _rename_images_to_chapter_format(chapters, output_dir, progress_callback)
            else:
                _prepare_single_chapter_image_renames(
                    chapters, output_dir, progress_callback
                )

        final_map = {}
        try:
            with open(rename_map_path, "r", encoding="utf-8") as handle:
                candidate_map = json.load(handle)
            if isinstance(candidate_map, dict):
                final_map = candidate_map
        except (OSError, ValueError, TypeError):
            pass
        result["renamed"] = sum(
            1 for renamed in final_map.values()
            if os.path.isfile(os.path.join(images_dir, os.path.basename(str(renamed))))
        )
        result["ready"] = bool(final_map) and result["renamed"] == len(final_map)
        result["prepared"] = bool(result["extracted"] or final_map != loaded_map)
        if result["ready"]:
            _report(
                f"✅ Prepared {result['renamed']} EPUB image asset(s) for SDLXLIFF review"
            )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        _report(f"❌ EPUB image asset preparation failed: {result['error']}")
    return result


def _rename_images_to_chapter_format(chapters, output_dir, progress_callback=None):
    """Rename image files to chapter-based format and update all references.
    
    Format: chapter{NNN}_img_{M}.{ext}
    - NNN: 3-digit zero-padded chapter number
    - M: sequential image number within that chapter (1-based)
    - ext: original file extension preserved
    
    Also saves image_rename_map.json for use by TransateKRtoEN.py and epub_converter.py.
    This operation is idempotent — re-running produces the same result.
    """
    images_dir = os.path.join(output_dir, 'images')
    rename_map_path = os.path.join(output_dir, 'image_rename_map.json')
    
    if not os.path.isdir(images_dir):
        print("📸 No images directory found — skipping image rename")
        return chapters
    
    # Collect existing image files
    existing_images = set()
    try:
        existing_images = {f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))}
    except Exception as e:
        print(f"⚠️ Could not list images directory: {e}")
        return chapters
    
    if not existing_images:
        print("📸 No image files found — skipping image rename")
        return chapters

    # A valid resource fingerprint reuses the already-renamed image files.
    # Freshly parsed chapter HTML still contains the source EPUB names, so a
    # second full rename pass would otherwise classify every canonical file as
    # unclaimed, rename it as a cover, and overwrite the authoritative map.
    # Replay a complete map instead when its terminal targets exactly match the
    # current image directory.
    existing_rename_map, _ = _load_image_rename_targets(output_dir)
    current_image_names = {name.casefold() for name in existing_images}
    mapped_target_names = {
        name.casefold() for name in existing_rename_map.values()
    }
    if existing_rename_map and mapped_target_names == current_image_names:
        print(
            f"♻️ Reusing complete image rename map for "
            f"{len(existing_images)} existing image file(s)"
        )
        return _prepare_single_chapter_image_renames(
            chapters,
            output_dir,
            progress_callback,
            status_context='Full extraction',
        )
    
    print(f"\n🖼️ Renaming {len(existing_images)} images to chapter-based format...")
    
    # Build mapping: scan each chapter body for <img> references
    # Track which images belong to which chapter (first reference wins)
    rename_map = {}  # original_name -> new_name
    claimed_images = set()  # images that have been assigned to a chapter
    
    for chapter in chapters:
        body = chapter.get('body', '')
        if not body:
            continue
        
        # Get the actual chapter filename stem for naming
        chapter_basename = chapter.get('original_basename', '')
        if not chapter_basename:
            # Fallback: derive from filename
            chapter_filename = chapter.get('filename', '')
            chapter_basename = os.path.splitext(os.path.basename(chapter_filename))[0] if chapter_filename else ''
        if not chapter_basename:
            # Last resort: use chapter number
            chapter_basename = f"chapter{int(chapter.get('num', 0)):03d}"
        
        # Remove extension from basename if it still has one
        chapter_stem = os.path.splitext(chapter_basename)[0]
        
        try:
            soup = BeautifulSoup(body, 'html.parser')
        except Exception:
            continue
        
        # Collect all image references from all supported formats
        image_srcs = _collect_image_srcs(soup)
        if not image_srcs:
            original_markup = chapter.get('original_html') or chapter.get('source_html') or chapter.get('raw_html') or ''
            if original_markup:
                try:
                    image_srcs = _collect_image_srcs(BeautifulSoup(original_markup, 'html.parser'))
                except Exception:
                    image_srcs = []
        
        img_counter = 1
        for src in image_srcs:
            # Skip data URIs
            if src.startswith('data:'):
                continue
            
            # Extract basename from various path formats
            # Handle: ../images/foo.jpg, images/foo.jpg, foo.jpg, etc.
            clean_src = src.split('?')[0]  # Remove query params
            basename = os.path.basename(clean_src)
            
            if not basename or basename in claimed_images:
                continue
            
            # Check if file actually exists
            if basename not in existing_images:
                # Try case-insensitive match
                matched = None
                for existing in existing_images:
                    if existing.lower() == basename.lower():
                        matched = existing
                        break
                if matched:
                    basename = matched
                else:
                    continue
            
            # Generate new name using actual chapter filename
            ext = os.path.splitext(basename)[1]  # Preserve original extension
            new_name = f"{chapter_stem}_img_{img_counter}{ext}"
            
            # Handle collision
            while new_name in rename_map.values():
                img_counter += 1
                new_name = f"{chapter_stem}_img_{img_counter}{ext}"
            
            rename_map[basename] = new_name
            claimed_images.add(basename)
            img_counter += 1
    
    # Handle unclaimed images (not referenced by any chapter) — name as Cover
    unclaimed = existing_images - claimed_images
    unclaimed_to_rename = sorted(unclaimed)
    if unclaimed_to_rename:
        for idx, img_name in enumerate(unclaimed_to_rename):
            ext = os.path.splitext(img_name)[1]
            new_name = f"{idx}_Cover{ext}"
            while new_name in rename_map.values():
                idx += 1
                new_name = f"{idx}_Cover{ext}"
            rename_map[img_name] = new_name
        print(f"   📎 {len(unclaimed_to_rename)} unclaimed images named as Cover")
    
    if not rename_map:
        print("📸 No image references found in chapters — skipping rename")
        return chapters
    
    # Phase 1: Physically rename image files (use temp names to avoid collisions)
    print(f"   📁 Renaming {len(rename_map)} image files...")
    temp_renames = {}  # temp_name -> final_name
    successful_renames = {}  # original_name -> new_name (only successful ones)
    
    # First pass: rename to temporary names
    for original, new_name in rename_map.items():
        original_path = os.path.join(images_dir, original)
        if not os.path.exists(original_path):
            continue
        temp_name = f"_temp_rename_{original}"
        temp_path = os.path.join(images_dir, temp_name)
        try:
            os.rename(original_path, temp_path)
            temp_renames[temp_name] = new_name
            successful_renames[original] = new_name
        except Exception as e:
            print(f"   ⚠️ Could not rename {original}: {e}")
    
    # Second pass: rename from temp to final names
    for temp_name, final_name in temp_renames.items():
        temp_path = os.path.join(images_dir, temp_name)
        final_path = os.path.join(images_dir, final_name)
        try:
            os.rename(temp_path, final_path)
        except Exception as e:
            print(f"   ⚠️ Could not finalize rename {temp_name} -> {final_name}: {e}")
            # Try to restore original name
            original_name = temp_name.replace('_temp_rename_', '', 1)
            try:
                os.rename(temp_path, os.path.join(images_dir, original_name))
                if original_name in successful_renames:
                    del successful_renames[original_name]
            except Exception:
                pass
    
    print(f"   ✅ Successfully renamed {len(successful_renames)} image files")
    
    # Phase 2: Update all image references in chapter bodies
    print(f"   📝 Updating image references in {len(chapters)} chapters...")
    updated_chapters = 0
    for chapter in chapters:
        body = chapter.get('body', '')
        if not body:
            continue
        
        try:
            soup = BeautifulSoup(body, 'html.parser')
            modified = _update_image_refs_in_soup(soup, successful_renames)
            
            if modified:
                chapter['body'] = str(soup)
                updated_chapters += 1

            for html_key in ('original_html', 'source_html', 'raw_html'):
                original_markup = chapter.get(html_key)
                if not original_markup:
                    continue
                try:
                    original_soup = BeautifulSoup(original_markup, 'html.parser')
                    if _update_image_refs_in_soup(original_soup, successful_renames):
                        chapter[html_key] = str(original_soup)
                except Exception:
                    pass
        except Exception as e:
            print(f"   ⚠️ Error updating chapter {chapter.get('num', '?')}: {e}")
    
    print(f"   ✅ Updated image references in {updated_chapters} chapters")
    
    # Phase 3: Save rename map for other scripts
    try:
        with open(rename_map_path, 'w', encoding='utf-8') as f:
            json.dump(successful_renames, f, ensure_ascii=False, indent=2)
        print(f"   💾 Saved image rename map to: image_rename_map.json")
    except Exception as e:
        print(f"   ⚠️ Could not save rename map: {e}")
    
    # Keep persistent remote-download records aligned with the chapter-based
    # filenames produced by this pass.
    _record_remote_image_renames(output_dir, successful_renames)

    # Phase 4: Update on-disk HTML files that were already saved during extraction
    # These files were written before the rename, so they still reference old image names
    print(f"   📄 Updating on-disk HTML files in output directory...")
    disk_updated = 0
    try:
        for fname in os.listdir(output_dir):
            fpath = os.path.join(output_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if not fname.lower().endswith(('.html', '.xhtml', '.htm')):
                continue
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                file_modified = _update_image_refs_in_soup(soup, successful_renames)
                
                if file_modified:
                    with open(fpath, 'w', encoding='utf-8') as f:
                        f.write(str(soup))
                    disk_updated += 1
            except Exception as e:
                print(f"   ⚠️ Error updating {fname}: {e}")
    except Exception as e:
        print(f"   ⚠️ Error scanning output directory: {e}")
    
    if disk_updated > 0:
        print(f"   ✅ Updated {disk_updated} on-disk HTML files")
    
    print(f"🖼️ Image renaming complete: {len(successful_renames)} images renamed")
    return chapters

def _get_best_parser():
    """Determine the best parser available, preferring lxml for CJK text"""
    try:
        import lxml
        return 'lxml'
    except ImportError:
        return 'html.parser'

def _sort_by_opf_spine(chapters, opf_path):
    """Sort chapters according to OPF spine order"""
    try:
        import xml.etree.ElementTree as ET
        
        # Read OPF file
        with open(opf_path, 'r', encoding='utf-8') as f:
            opf_content = f.read()
        
        # Parse OPF
        root = ET.fromstring(opf_content)
        
        # Find namespaces
        ns = {'opf': 'http://www.idpf.org/2007/opf'}
        if root.tag.startswith('{'):
            default_ns = root.tag[1:root.tag.index('}')]
            ns = {'opf': default_ns}
        
        # Build manifest map (id -> href)
        manifest = {}
        for item in root.findall('.//opf:manifest/opf:item', ns):
            item_id = item.get('id')
            href = item.get('href')
            if item_id and href:
                manifest[item_id] = href
        
        # Get spine order
        spine_order = []
        spine = root.find('.//opf:spine', ns)
        if spine is not None:
            for itemref in spine.findall('opf:itemref', ns):
                idref = itemref.get('idref')
                if idref and idref in manifest:
                    href = manifest[idref]
                    spine_order.append(href)
        
        if not spine_order:
            print("⚠️ No spine order found in OPF, keeping original order")
            return chapters
        
        # Create a mapping of filenames to spine position
        spine_map = {}
        for idx, href in enumerate(spine_order):
            # Try different matching strategies
            basename = os.path.basename(href)
            spine_map[basename] = idx
            spine_map[href] = idx
            # Also store without extension for flexible matching
            name_no_ext = os.path.splitext(basename)[0]
            spine_map[name_no_ext] = idx
        
        print(f"📋 OPF spine contains {len(spine_order)} items")
        
        # Sort chapters based on spine order
        def get_spine_position(chapter):
            # Try to match chapter to spine
            filename = chapter.get('filename', '')
            basename = chapter.get('original_basename', '')
            
            # Try exact filename match
            if filename in spine_map:
                return spine_map[filename]
            
            # Try basename match
            if basename in spine_map:
                return spine_map[basename]
            
            # Try basename of filename
            if filename:
                fname_base = os.path.basename(filename)
                if fname_base in spine_map:
                    return spine_map[fname_base]
            
            # Try without extension
            if basename:
                if basename + '.html' in spine_map:
                    return spine_map[basename + '.html']
                if basename + '.xhtml' in spine_map:
                    return spine_map[basename + '.xhtml']
            
            # Fallback to chapter number * 1000 (to sort after spine items)
            return 1000000 + chapter.get('num', 0)
        
        # Sort chapters
        sorted_chapters = sorted(chapters, key=get_spine_position)
        
        # Store the raw OPF position so translation code can build offset maps.
        # Also store a list-based sequential position as spine_order (used
        # by other code paths that don't need offset mapping).
        for idx, chapter in enumerate(sorted_chapters, 1):
            opf_pos = get_spine_position(chapter)
            chapter['opf_spine_index'] = opf_pos if opf_pos < 1000000 else None
            chapter['spine_order'] = idx
        
        # Log reordering info
        reordered_count = 0
        for idx, chapter in enumerate(sorted_chapters):
            original_idx = chapters.index(chapter)
            if original_idx != idx:
                reordered_count += 1
        
        if reordered_count > 0:
            print(f"🔄 Reordered {reordered_count} chapters to match OPF spine")
        else:
            print(f"✅ Chapter order already matches OPF spine")
        
        return sorted_chapters
        
    except Exception as e:
        print(f"⚠️ Could not sort by OPF spine: {e}")
        import traceback
        traceback.print_exc()
        return chapters


def protect_angle_brackets_with_korean(text: str) -> str:
    """Protect CJK text in angle brackets from HTML parsing"""
    if text is None:
        return ""
    
    import re
    # Extended pattern to include Korean, Chinese, and Japanese characters
    cjk_pattern = r'[가-힣ㄱ-ㅎㅏ-ㅣ一-龿ぁ-ゟァ-ヿ]'
    bracket_pattern = rf'<([^<>]*{cjk_pattern}[^<>]*)>'
    
    def replace_brackets(match):
        content = match.group(1)
        return f'&#60;{content}&#62;'
    
    return re.sub(bracket_pattern, replace_brackets, text)

def ensure_all_opf_chapters_extracted(zf, chapters, out):
    """Ensure ALL chapters from OPF spine are extracted, not just what ChapterExtractor found"""
    
    # Parse OPF to get ALL chapters in spine
    opf_chapters = []
    
    try:
        # The OPF filename is arbitrary; container.xml identifies the package.
        opf_member = find_epub_opf_member(zf)
        opf_content = zf.read(opf_member) if opf_member else None
        
        if not opf_content:
            return chapters  # No OPF, return original
        
        import xml.etree.ElementTree as ET
        root = ET.fromstring(opf_content)
        
        # Handle namespaces
        ns = {'opf': 'http://www.idpf.org/2007/opf'}
        if root.tag.startswith('{'):
            default_ns = root.tag[1:root.tag.index('}')]
            ns = {'opf': default_ns}
        
        # Get manifest
        manifest = {}
        for item in root.findall('.//opf:manifest/opf:item', ns):
            item_id = item.get('id')
            href = item.get('href')
            media_type = item.get('media-type', '')
            
            if item_id and href and ('html' in media_type.lower() or href.endswith(('.html', '.xhtml', '.htm'))):
                manifest[item_id] = href
        
        # Get spine order
        spine = root.find('.//opf:spine', ns)
        if spine:
            for itemref in spine.findall('opf:itemref', ns):
                idref = itemref.get('idref')
                if idref and idref in manifest:
                    href = manifest[idref]
                    filename = os.path.basename(href)
                    
                    # Skip configured special files only when they have no
                    # numbers. Numbered special files can still be translated
                    # later, but they keep chapter number 0.
                    import re
                    has_numbers = bool(re.search(r'\d', filename))
                    translate_special = os.getenv('TRANSLATE_SPECIAL_FILES', '0') == '1'
                    if not translate_special and not has_numbers and _is_configured_special_file(filename):
                        continue
                    
                    opf_chapters.append(href)
        
        print(f"📚 OPF spine contains {len(opf_chapters)} chapters")
        
        # Check which OPF chapters are missing from extraction
        extracted_files = set()
        for c in chapters:
            if 'filename' in c:
                extracted_files.add(c['filename'])
            if 'original_basename' in c:
                extracted_files.add(c['original_basename'])
        
        missing_chapters = []
        for opf_chapter in opf_chapters:
            basename = os.path.basename(opf_chapter)
            if basename not in extracted_files and opf_chapter not in extracted_files:
                missing_chapters.append(opf_chapter)
        
        if missing_chapters:
            print(f"⚠️ {len(missing_chapters)} chapters in OPF but not extracted!")
            print(f"   Missing: {missing_chapters[:5]}{'...' if len(missing_chapters) > 5 else ''}")
            
            # Extract the missing chapters
            for href in missing_chapters:
                try:
                    # Read the chapter content
                    content = zf.read(href).decode('utf-8')
                    
                    # Extract chapter number
                    import re
                    basename = os.path.basename(href)
                    matches = re.findall(r'(\d+)', basename)
                    if _is_configured_special_file(basename):
                        chapter_num = 0
                    elif matches:
                        chapter_num = int(matches[-1])
                    else:
                        chapter_num = len(chapters) + 1
                    
                    # Create chapter entry
                    from bs4 import BeautifulSoup
                    parser = 'lxml' if 'lxml' in sys.modules else 'html.parser'
                    soup = BeautifulSoup(content, parser)
                    
                    # Get title
                    title = "Chapter " + str(chapter_num)
                    title_tag = soup.find('title')
                    if title_tag:
                        title = title_tag.get_text().strip() or title
                    else:
                        for tag in ['h1', 'h2', 'h3']:
                            header = soup.find(tag)
                            if header:
                                title = header.get_text().strip() or title
                                break
                    
                    # Save the chapter file
                    output_filename = f"chapter_{chapter_num:04d}_{basename}"
                    output_path = os.path.join(out, output_filename)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    image_srcs = _collect_image_srcs(soup)

                    # Add to chapters list
                    new_chapter = {
                        'num': chapter_num,
                        'title': title,
                        'body': content,
                        'filename': href,
                        'original_basename': basename,
                        'file_size': len(content),
                        'has_images': bool(image_srcs),
                        'image_count': len(image_srcs),
                        'detection_method': 'opf_recovery',
                        'content_hash': None  # Will be calculated later
                    }
                    
                    chapters.append(new_chapter)
                    print(f"   ✅ Recovered chapter {chapter_num}: {basename}")
                    
                except Exception as e:
                    print(f"   ❌ Failed to extract {href}: {e}")
            
            # Re-sort chapters by number
            chapters.sort(key=lambda x: x['num'])
            print(f"✅ Total chapters after OPF recovery: {len(chapters)}")
        
    except Exception as e:
        print(f"⚠️ Error checking OPF chapters: {e}")
        import traceback
        traceback.print_exc()
    
    return chapters
    
def extract_chapters(zf, output_dir, parser=None, progress_callback=None, pattern_manager=None):
    """Extract chapters and all resources from EPUB using ThreadPoolExecutor
    
    Args:
        zf: ZipFile object of the EPUB
        output_dir: Output directory for extracted files
        parser: BeautifulSoup parser to use ('lxml' or 'html.parser')
        progress_callback: Optional callback for progress updates
        pattern_manager: Optional PatternManager instance for chapter detection
    """
    import time
    
    # Initialize defaults if not provided
    if parser is None:
        parser = _get_best_parser()
    # pattern_manager is no longer used - kept for API compatibility
    
    # Check stop at the very beginning
    if is_stop_requested():
        print("❌ Extraction stopped by user")
        return []
        
    print("🚀 Starting EPUB extraction with ThreadPoolExecutor...")
    print(f"📄 Using parser: {parser} {'(optimized for CJK)' if parser == 'lxml' else '(standard)'}")
    
    # Initial progress
    if progress_callback:
        progress_callback("Starting EPUB extraction...")
    
    # First, save the authoritative OPF for reference under its source name.
    opf_member = find_epub_opf_member(zf)
    if opf_member:
        try:
            opf_content = zf.read(opf_member).decode('utf-8', errors='ignore')
            opf_basename = os.path.basename(opf_member.replace('\\', '/'))
            opf_output_path = os.path.join(output_dir, opf_basename)
            with open(opf_output_path, 'w', encoding='utf-8') as f:
                f.write(opf_content)
            print(f"📋 Saved OPF package: {opf_member} → {opf_basename}")
        except Exception as e:
            print(f"⚠️ Could not save OPF package: {e}")
    
    # Get extraction mode from environment
    extraction_mode = os.getenv("EXTRACTION_MODE", "smart").lower()
    print(f"✅ Using {extraction_mode.capitalize()} extraction mode")
    
    # Get number of workers from environment or use default
    max_workers = int(os.getenv("EXTRACTION_WORKERS", "2"))
    source_epub_image_count = _source_epub_image_count_from_zip(zf)
    preserve_remote_images = _remote_image_cache_matches_source(
        output_dir,
        source_epub_image_count,
    )
    if preserve_remote_images:
        print(
            "♻️ Remote image cache matches the source EPUB image count; "
            "preserving the images directory"
        )
    print(f"🔧 Using {max_workers} workers for parallel processing")
    
    # Single-chapter mode (SINGLE_CHAPTER_FILTER): reuse a prior extraction's
    # resources.  If the images directory is absent/empty (including a
    # workspace damaged by the old targeted-run cleanup), restore packaged
    # resources once; only the selected HTML file is still parsed below.
    _single_mode = bool((os.getenv("SINGLE_CHAPTER_FILTER", "") or "").strip())
    images_dir = os.path.join(output_dir, 'images')
    try:
        existing_packaged_images = [
            name for name in os.listdir(images_dir)
            if (
                os.path.isfile(os.path.join(images_dir, name))
                and os.path.splitext(name)[1].lower()
                in {'.jpg', '.jpeg', '.png', '.gif', '.svg', '.bmp', '.webp'}
            )
        ]
    except OSError:
        existing_packaged_images = []
    _single_resource_bootstrap = (
        _single_mode
        and source_epub_image_count > 0
        and not existing_packaged_images
    )
    if _single_mode and not _single_resource_bootstrap:
        print("🎯 Single-chapter mode: skipping full resource extraction (css/fonts/images)")
        extracted_resources = {'css': [], 'fonts': [], 'images': [],
                               'epub_structure': [], 'other': []}
    else:
        if _single_resource_bootstrap:
            print(
                "🎯 Single-chapter mode: restoring missing EPUB image "
                "resources before the targeted retry"
            )
            try:
                os.remove(os.path.join(output_dir, '.resources_extracted'))
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"⚠️ Could not reset the resource marker: {e}")
        extracted_resources = _extract_all_resources(
            zf,
            output_dir,
            progress_callback,
            preserve_images=(preserve_remote_images or _single_resource_bootstrap),
        )

    # Check stop after resource extraction
    if is_stop_requested():
        print("❌ Extraction stopped by user")
        return []

    chapter_cache_signature = None
    if not _single_mode and pattern_manager is None:
        chapter_cache_signature = _chapter_extraction_cache_signature(
            extraction_mode,
            parser,
        )
        cached_chapters, cache_reason = _load_chapter_extraction_cache(
            output_dir,
            chapter_cache_signature,
        )
        if cached_chapters is not None:
            message = (
                f"Chapter cache matches {chapter_cache_signature['engine']} "
                f"settings; loaded {len(cached_chapters)} chapters without "
                "scanning or processing the EPUB"
            )
            print(f"📚 {message}")
            if progress_callback:
                progress_callback(f"📚 {message}")
            return cached_chapters
        if cache_reason != 'chapter cache marker is missing':
            print(f"♻️ Chapter cache invalid ({cache_reason}); rebuilding")
            _remove_chapter_extraction_marker(output_dir)
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        print("📋 Loading existing metadata...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        source_metadata = _extract_epub_metadata(zf)
        restored_fields = restore_truncated_repeatable_metadata(
            metadata, source_metadata
        )
        if restored_fields:
            print(
                "📋 Restored truncated repeatable metadata fields: "
                + ", ".join(sorted(restored_fields))
            )
    else:
        print("📋 Extracting fresh metadata...")
        metadata = _extract_epub_metadata(zf)
        print(f"📋 Extracted metadata: {list(metadata.keys())}")
    
    chapters, detected_language = _extract_chapters_universal(zf, extraction_mode, parser, progress_callback, pattern_manager)
    
    # Sort chapters according to OPF spine order if available
    opf_path = find_opf_path(output_dir)
    if opf_path and chapters:
        print("📋 Sorting chapters according to OPF spine order...")
        chapters = _sort_by_opf_spine(chapters, opf_path)
        print(f"✅ Chapters sorted according to OPF reading order")
    
    # Check stop after chapter extraction
    if is_stop_requested():
        print("❌ Extraction stopped by user")
        return []
    
    if not chapters:
        print("❌ No chapters could be extracted!")
        return []
    
    # Remote URLs must be localized before the regular image ownership,
    # chapter rename, and image_rename_map.json pass.
    if os.getenv('DOWNLOAD_REMOTE_IMAGE_URLS', '0').strip().lower() in {
        '1', 'true', 'yes', 'on'
    }:
        chapters = _localize_remote_images(
            chapters,
            output_dir,
            progress_callback,
            source_epub_image_count=source_epub_image_count,
        )

    # Rename images to chapter-based format (chapter001_img_1.jpg, etc.).
    # Image output mode also needs this map; its passthrough HTML copy applies
    # image_rename_map.json before writing response files.
    # Skipped in single-chapter mode — renaming with a one-chapter list would
    # mis-claim images that belong to chapters not present in this run.
    if not _single_mode:
        chapters = _rename_images_to_chapter_format(chapters, output_dir, progress_callback)
    else:
        print("🎯 Single-chapter mode: preparing canonical image names")
        chapters = _prepare_single_chapter_image_renames(
            chapters,
            output_dir,
            progress_callback,
        )
    
    chapters_info_path = os.path.join(output_dir, 'chapters_info.json')
    chapters_info = []
    chapters_info_lock = threading.Lock()
    
    def process_chapter(chapter):
        """Process a single chapter"""
        # Check stop in worker
        if is_stop_requested():
            return None
            
        info = {
            'num': chapter['num'],
            'title': chapter['title'],
            'original_filename': chapter.get('filename', ''),
            'has_images': chapter.get('has_images', False),
            'image_count': chapter.get('image_count', 0),
            'text_length': chapter.get('file_size', len(chapter.get('body', ''))),
            'detection_method': chapter.get('detection_method', 'unknown'),
            'content_hash': chapter.get('content_hash', '')
        }
        
        if chapter.get('has_images'):
            try:
                soup = BeautifulSoup(chapter.get('body', ''), parser)
                info['images'] = _collect_image_srcs(soup)
            except:
                info['images'] = []
        
        return info
    
    # Process chapters in parallel
    print(f"🔄 Processing {len(chapters)} chapters in parallel...")
    
    if progress_callback:
        progress_callback(f"Processing {len(chapters)} chapters...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chapter = {
            executor.submit(process_chapter, chapter): chapter 
            for chapter in chapters
        }
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_chapter):
            if is_stop_requested():
                print("❌ Extraction stopped by user")
                # Cancel remaining futures
                for f in future_to_chapter:
                    f.cancel()
                return []
            
            try:
                result = future.result()
                if result:
                    with chapters_info_lock:
                        chapters_info.append(result)
                    completed += 1
                    
                    # Yield to GUI periodically (can be disabled for max speed)
                    if completed % 5 == 0 and os.getenv("ENABLE_GUI_YIELD", "1") == "1":
                        time.sleep(0.001)
                    
                    # Progress updates
                    if completed % 10 == 0 or completed == len(chapters):
                        if progress_callback:
                            progress_msg = f"Processed {completed}/{len(chapters)} chapters"
                            progress_callback(progress_msg)
                        else:
                            # Show progress bar in terminal
                            ProgressBar.update(completed, len(chapters), prefix="📊 Processing metadata")
            except Exception as e:
                chapter = future_to_chapter[future]
                print(f"   ❌ Error processing chapter {chapter['num']}: {e}")
    
    # Finish progress bar
    if not progress_callback:
        ProgressBar.finish()
    
    # Sort chapters_info by chapter number to maintain order
    chapters_info.sort(key=lambda x: x['num'])
    
    print(f"✅ Successfully processed {len(chapters_info)} chapters")

    if _single_mode and os.path.exists(chapters_info_path):
        # Merge into any existing chapters_info.json so a single-chapter run
        # doesn't clobber the bookkeeping written by a previous full run.
        try:
            with open(chapters_info_path, 'r', encoding='utf-8') as f:
                existing_info = json.load(f)
            if isinstance(existing_info, list):
                new_nums = {c.get('num') for c in chapters_info}
                merged_info = [c for c in existing_info
                               if isinstance(c, dict) and c.get('num') not in new_nums]
                merged_info.extend(chapters_info)
                merged_info.sort(key=lambda x: x.get('num', 0))
                chapters_info = merged_info
        except Exception as e:
            print(f"⚠️ Could not merge existing chapters_info.json: {e}")

    chapters_info_saved = _write_json_atomic(
        chapters_info_path,
        chapters_info,
        indent=2,
    )
    if not chapters_info_saved:
        print("⚠️ Could not save chapters_info.json atomically")

    print(f"💾 Saved detailed chapter info to: chapters_info.json")

    if _single_mode:
        # Merge instead of overwrite — keep counts/titles from the previous
        # full extraction (``metadata`` was preloaded from metadata.json
        # above when it exists) and only fold in this chapter's title.
        titles = metadata.get('chapter_titles') or {}
        titles.update({str(c['num']): c['title'] for c in chapters})
        metadata['chapter_titles'] = titles
        metadata.setdefault('chapter_count', len(titles))
        metadata.setdefault('detected_language', detected_language)
        metadata.setdefault('extraction_mode', extraction_mode)
    else:
        metadata.update({
            'chapter_count': len(chapters),
            'chapter_payloads_ready': sum(
                1 for chapter in chapters
                if isinstance(chapter.get('body'), str)
            ),
            'detected_language': detected_language,
            'extracted_resources': extracted_resources,
            'extraction_mode': extraction_mode,
            'extraction_summary': {
                'total_chapters': len(chapters),
                'chapter_range': f"{chapters[0]['num']}-{chapters[-1]['num']}",
                'resources_extracted': sum(len(files) for files in extracted_resources.values())
            }
        })

        metadata['chapter_titles'] = {
            str(c['num']): c['title'] for c in chapters
        }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved comprehensive metadata to: {metadata_path}")

    if not _single_mode:
        _create_extraction_report(output_dir, metadata, chapters, extracted_resources)
    _log_extraction_summary(chapters, extracted_resources, detected_language)

    chapter_cache_saved = False
    if (
        not _single_mode
        and chapter_cache_signature is not None
        and chapters_info_saved
    ):
        chapter_cache_saved = _write_chapter_extraction_cache(
            output_dir,
            chapters,
            chapter_cache_signature,
            detected_language,
        )
        if chapter_cache_saved:
            print(
                f"💾 Saved validated {chapter_cache_signature['engine']} "
                "chapter cache"
            )
        else:
            print("⚠️ Could not save validated chapter cache")
    if not _single_mode and not chapter_cache_saved:
        # The worker/main process still requires this artifact even when cache
        # metadata could not be committed.
        _write_json_atomic(
            os.path.join(output_dir, _CHAPTERS_FULL_NAME),
            chapters,
        )
    
    print(f"🔍 VERIFICATION: {extraction_mode.capitalize()} chapter extraction completed successfully")
    print(f"⚡ Used {max_workers} workers for parallel processing")
    
    return chapters

_RESOURCE_EXTRACTION_MARKER_VERSION = 3
_RESOURCE_FINGERPRINT_CHUNK_SIZE = 8 * 1024 * 1024
_RESOURCE_MARKER_DIRECTORIES = {
    'css': 'css',
    'fonts': 'fonts',
    'epub_structure': '',
    'other': '',
}
_RESOURCE_MARKER_TYPES = tuple(_RESOURCE_MARKER_DIRECTORIES)


def _source_epub_content_fingerprint(zf):
    """Return a SHA-256 fingerprint of every byte in the source EPUB.

    Hashing the complete archive, rather than mtimes or ZIP member metadata,
    makes even a one-byte edit outside a member payload invalidate the marker.
    ``ZipFile`` instances backed by an in-memory stream are supported for
    tests and non-path callers by hashing their underlying file object while
    restoring its original position afterwards.
    """
    source_path = getattr(zf, 'filename', None)
    try:
        source_path = os.fspath(source_path) if source_path is not None else ''
    except TypeError:
        source_path = ''

    hasher = hashlib.sha256()
    total_size = 0

    def _consume(stream):
        nonlocal total_size
        while True:
            chunk = stream.read(_RESOURCE_FINGERPRINT_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
            total_size += len(chunk)

    try:
        if source_path and os.path.isfile(source_path):
            with open(source_path, 'rb') as source:
                _consume(source)
        else:
            source = getattr(zf, 'fp', None)
            if source is None or not hasattr(source, 'seek'):
                return None
            original_position = source.tell()
            try:
                source.seek(0)
                _consume(source)
            finally:
                source.seek(original_position)
    except (OSError, ValueError, AttributeError):
        return None

    return {
        'algorithm': 'sha256',
        'sha256': hasher.hexdigest(),
        'size': total_size,
    }


def _env_flag(name, default='0'):
    return os.getenv(name, default).strip().lower() in {
        '1', 'true', 'yes', 'on'
    }


def _chapter_extraction_cache_signature(extraction_mode, parser):
    """Return settings that can change extracted chapter payloads.

    The selected engine is explicit so an html2text run can never consume a
    BeautifulSoup artifact merely because both came from the same EPUB.
    Runtime-only settings such as worker count and progress throttling are
    intentionally excluded.
    """
    mode = str(extraction_mode or 'smart').strip().lower()
    signature = {
        'engine': 'html2text' if mode == 'enhanced' else 'beautifulsoup',
        'extraction_mode': mode,
        'parser': str(parser or ''),
        'disable_chapter_merging': _env_flag('DISABLE_CHAPTER_MERGING'),
        'download_remote_image_urls': _env_flag('DOWNLOAD_REMOTE_IMAGE_URLS'),
        'translate_special_files': _env_flag('TRANSLATE_SPECIAL_FILES'),
        'special_file_keywords': os.getenv('SPECIAL_FILE_KEYWORDS', '').strip(),
        'special_file_exact': os.getenv('SPECIAL_FILE_EXACT', '').strip(),
        'batch_translate_headers': _env_flag('BATCH_TRANSLATE_HEADERS'),
        'use_title': _env_flag('USE_TITLE'),
        'ignore_header': _env_flag('IGNORE_HEADER'),
        'remove_duplicate_h1_p': _env_flag('REMOVE_DUPLICATE_H1_P'),
    }
    if mode == 'enhanced':
        enhanced_filtering = os.getenv('ENHANCED_FILTERING', 'smart').strip().lower()
        if enhanced_filtering == 'full':
            enhanced_filtering = 'comprehensive'
        model_name = os.getenv('MODEL', '').strip().lower()
        traditional_model = (
            model_name in {'deepl', 'google-translate', 'google-translate-free'}
            or model_name.startswith('deepl/')
            or model_name.startswith('google-translate/')
        )
        signature.update({
            'enhanced_filtering': enhanced_filtering,
            'enhanced_preserve_structure': _env_flag(
                'ENHANCED_PRESERVE_STRUCTURE', '1'
            ),
            'enhanced_single_line_break': _env_flag(
                'ENHANCED_SINGLE_LINE_BREAK'
            ),
            'html2text_escape_snob': _env_flag('HTML2TEXT_ESCAPE_SNOB'),
            'fix_empty_attr_tags_extract': _env_flag(
                'FIX_EMPTY_ATTR_TAGS_EXTRACT'
            ),
            'force_bs_for_traditional': _env_flag(
                'FORCE_BS_FOR_TRADITIONAL'
            ),
            'traditional_model': traditional_model,
        })
    else:
        signature['fix_stray_p_gt_bs'] = _env_flag('FIX_STRAY_P_GT_BS')
    return signature


def _write_json_atomic(path, payload, indent=None):
    """Write JSON completely before replacing the visible destination."""
    temporary_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(temporary_path, 'w', encoding='utf-8') as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=indent,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        return True
    except (OSError, TypeError, ValueError):
        try:
            os.remove(temporary_path)
        except OSError:
            pass
        return False


def _file_content_fingerprint(path):
    """Return a SHA-256 fingerprint for a cache dependency file."""
    hasher = hashlib.sha256()
    size = 0
    try:
        with open(path, 'rb') as handle:
            while True:
                chunk = handle.read(_RESOURCE_FINGERPRINT_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                size += len(chunk)
    except OSError:
        return None
    return {
        'algorithm': 'sha256',
        'sha256': hasher.hexdigest(),
        'size': size,
    }


def _resource_marker_source_fingerprint(output_dir):
    marker_path = os.path.join(output_dir, '.resources_extracted')
    try:
        with open(marker_path, 'r', encoding='utf-8') as handle:
            marker = json.load(handle)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(marker, dict)
        or marker.get('version') != _RESOURCE_EXTRACTION_MARKER_VERSION
        or not isinstance(marker.get('source_epub'), dict)
    ):
        return None
    source = marker['source_epub']
    if (
        source.get('algorithm') != 'sha256'
        or not source.get('sha256')
        or not isinstance(source.get('size'), int)
    ):
        return None
    return dict(source)


def _chapter_extraction_marker_path(output_dir):
    return os.path.join(output_dir, _CHAPTER_EXTRACTION_MARKER_NAME)


def _remove_chapter_extraction_marker(output_dir):
    try:
        os.remove(_chapter_extraction_marker_path(output_dir))
    except OSError:
        pass


def _load_chapter_extraction_cache(output_dir, signature):
    """Return ``(chapters, reason)`` for a validated chapter cache."""
    marker_path = _chapter_extraction_marker_path(output_dir)
    try:
        with open(marker_path, 'r', encoding='utf-8') as handle:
            marker = json.load(handle)
    except FileNotFoundError:
        return None, 'chapter cache marker is missing'
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None, 'chapter cache marker is unreadable'

    if not isinstance(marker, dict):
        return None, 'chapter cache marker is invalid'
    if marker.get('version') != _CHAPTER_EXTRACTION_CACHE_VERSION:
        return None, 'chapter cache version changed'

    source_fingerprint = _resource_marker_source_fingerprint(output_dir)
    if source_fingerprint is None:
        return None, 'validated source EPUB fingerprint is unavailable'
    if marker.get('source_epub') != source_fingerprint:
        return None, 'source EPUB fingerprint changed'
    if marker.get('signature') != signature:
        cached_engine = (
            marker.get('signature', {}).get('engine')
            if isinstance(marker.get('signature'), dict) else None
        )
        if cached_engine and cached_engine != signature.get('engine'):
            return None, (
                f"selected engine is {signature.get('engine')}, "
                f"cached engine is {cached_engine}"
            )
        return None, 'chapter extraction settings changed'

    rename_map_path = os.path.join(output_dir, 'image_rename_map.json')
    current_rename_map = (
        _file_content_fingerprint(rename_map_path)
        if os.path.isfile(rename_map_path) else None
    )
    if marker.get('image_rename_map') != current_rename_map:
        return None, 'image rename map changed'

    artifacts = marker.get('artifacts')
    if not isinstance(artifacts, dict):
        return None, 'chapter artifact inventory is missing'

    chapters_path = os.path.join(output_dir, _CHAPTERS_FULL_NAME)
    info_path = os.path.join(output_dir, _CHAPTERS_INFO_NAME)
    for artifact_name, artifact_path in (
        (_CHAPTERS_FULL_NAME, chapters_path),
        (_CHAPTERS_INFO_NAME, info_path),
    ):
        expected = artifacts.get(artifact_name)
        if not isinstance(expected, dict):
            return None, f'{artifact_name} inventory is missing'
        try:
            current_size = os.path.getsize(artifact_path)
        except OSError:
            return None, f'{artifact_name} is missing'
        if current_size != expected.get('size'):
            return None, f'{artifact_name} size changed'

    try:
        with open(chapters_path, 'r', encoding='utf-8') as handle:
            chapters = json.load(handle)
        with open(info_path, 'r', encoding='utf-8') as handle:
            chapters_info = json.load(handle)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None, 'chapter cache JSON is unreadable'

    if not isinstance(chapters, list) or not all(
        isinstance(chapter, dict)
        and isinstance(chapter.get('body'), str)
        for chapter in chapters
    ):
        return None, 'chapters_full.json payload is invalid'
    if not isinstance(chapters_info, list):
        return None, 'chapters_info.json payload is invalid'
    expected_count = artifacts[_CHAPTERS_FULL_NAME].get('count')
    if len(chapters) != expected_count:
        return None, 'chapter count changed'
    if len(chapters_info) != artifacts[_CHAPTERS_INFO_NAME].get('count'):
        return None, 'chapter info count changed'
    return chapters, ''


def _write_chapter_extraction_cache(
    output_dir,
    chapters,
    signature,
    detected_language,
):
    """Atomically commit chapter artifacts, then their validation marker."""
    chapters_path = os.path.join(output_dir, _CHAPTERS_FULL_NAME)
    info_path = os.path.join(output_dir, _CHAPTERS_INFO_NAME)
    if not _write_json_atomic(chapters_path, chapters):
        return False
    source_fingerprint = _resource_marker_source_fingerprint(output_dir)
    if source_fingerprint is None:
        return False
    try:
        with open(info_path, 'r', encoding='utf-8') as handle:
            chapters_info = json.load(handle)
        if not isinstance(chapters_info, list):
            return False
        chapters_size = os.path.getsize(chapters_path)
        info_size = os.path.getsize(info_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False

    rename_map_path = os.path.join(output_dir, 'image_rename_map.json')
    rename_map_fingerprint = (
        _file_content_fingerprint(rename_map_path)
        if os.path.isfile(rename_map_path) else None
    )
    marker = {
        'version': _CHAPTER_EXTRACTION_CACHE_VERSION,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_epub': source_fingerprint,
        'signature': dict(signature),
        'image_rename_map': rename_map_fingerprint,
        'detected_language': str(detected_language or 'unknown'),
        'artifacts': {
            _CHAPTERS_FULL_NAME: {
                'size': chapters_size,
                'count': len(chapters),
            },
            _CHAPTERS_INFO_NAME: {
                'size': info_size,
                'count': len(chapters_info),
            },
        },
    }
    return _write_json_atomic(
        _chapter_extraction_marker_path(output_dir),
        marker,
        indent=2,
    )


def _load_image_rename_targets(output_dir):
    """Return original-name -> terminal-name mappings from the rename sidecar."""
    rename_map_path = os.path.join(output_dir, 'image_rename_map.json')
    try:
        with open(rename_map_path, 'r', encoding='utf-8') as handle:
            loaded = json.load(handle)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}, {}
    if not isinstance(loaded, dict):
        return {}, {}

    raw = {}
    for original, renamed in loaded.items():
        original_name = os.path.basename(str(original or '').replace('\\', '/'))
        renamed_name = os.path.basename(str(renamed or '').replace('\\', '/'))
        if not original_name or not renamed_name:
            continue
        raw[original_name] = renamed_name

    # Later repair passes can extend a mapping into a chain, for example
    # ``1.png -> chapter001_img_1.png -> chapter002_img_1.png``. Validation
    # must check the terminal filename rather than the now-missing intermediate.
    raw_folded_keys = {name.casefold(): name for name in raw}
    exact = {}
    folded = {}
    for original_name, first_target in raw.items():
        current_name = first_target
        seen = {original_name.casefold()}
        while current_name.casefold() not in seen:
            seen.add(current_name.casefold())
            next_key = raw_folded_keys.get(current_name.casefold())
            if next_key is None:
                break
            next_name = raw.get(next_key)
            if not next_name or next_name == current_name:
                break
            current_name = next_name
        exact[original_name] = current_name
        folded[original_name.casefold()] = current_name
    return exact, folded


def _validate_resource_extraction_marker(
    marker_path,
    output_dir,
    source_fingerprint,
):
    """Return ``(valid, reason)`` for a versioned resource marker."""
    try:
        with open(marker_path, 'r', encoding='utf-8') as handle:
            marker = json.load(handle)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False, 'legacy or unreadable marker'

    if not isinstance(marker, dict):
        return False, 'invalid marker payload'
    if marker.get('version') != _RESOURCE_EXTRACTION_MARKER_VERSION:
        return False, 'marker version changed'
    if not isinstance(source_fingerprint, dict):
        return False, 'source EPUB could not be fingerprinted'

    recorded_source = marker.get('source_epub')
    if not isinstance(recorded_source, dict):
        return False, 'source EPUB fingerprint is missing'
    if (
        recorded_source.get('algorithm') != 'sha256'
        or recorded_source.get('sha256') != source_fingerprint.get('sha256')
        or recorded_source.get('size') != source_fingerprint.get('size')
    ):
        return False, 'source EPUB content changed'

    recorded_images = marker.get('images')
    if not isinstance(recorded_images, dict):
        return False, 'image inventory is missing'
    expected_images = recorded_images.get('source_filenames')
    if not isinstance(expected_images, list):
        return False, 'image inventory is invalid'

    images_dir = os.path.join(output_dir, 'images')
    if not os.path.isdir(images_dir):
        return False, 'images directory is missing'

    rename_exact, rename_folded = _load_image_rename_targets(output_dir)
    missing = []
    for original in expected_images:
        original_name = os.path.basename(str(original or '').replace('\\', '/'))
        if not original_name:
            return False, 'image inventory contains an invalid filename'
        current_name = rename_exact.get(original_name)
        if not current_name:
            current_name = rename_folded.get(original_name.casefold(), original_name)
        if not os.path.isfile(os.path.join(images_dir, current_name)):
            missing.append(current_name)
            if len(missing) >= 3:
                break
    if missing:
        return False, 'missing extracted image file(s): ' + ', '.join(missing)

    recorded_resources = marker.get('resources')
    if not isinstance(recorded_resources, dict):
        return False, 'non-image resource inventory is missing'
    missing_resources = []
    for resource_type in _RESOURCE_MARKER_TYPES:
        expected_names = recorded_resources.get(resource_type)
        if not isinstance(expected_names, list):
            return False, f'{resource_type} resource inventory is invalid'
        relative_dir = _RESOURCE_MARKER_DIRECTORIES[resource_type]
        for recorded_name in expected_names:
            filename = os.path.basename(
                str(recorded_name or '').replace('\\', '/')
            )
            if not filename:
                return False, f'{resource_type} inventory contains an invalid filename'
            candidate = (
                os.path.join(output_dir, relative_dir, filename)
                if relative_dir else os.path.join(output_dir, filename)
            )
            if not os.path.isfile(candidate):
                missing_resources.append(
                    f'{relative_dir}/{filename}' if relative_dir else filename
                )
                if len(missing_resources) >= 3:
                    break
        if len(missing_resources) >= 3:
            break
    if missing_resources:
        return False, (
            'missing extracted resource file(s): '
            + ', '.join(missing_resources)
        )
    return True, ''


def _write_resource_extraction_marker(
    marker_path,
    source_fingerprint,
    expected_image_filenames,
    expected_resource_filenames,
):
    """Atomically write the completed source/image resource fingerprint."""
    if not isinstance(source_fingerprint, dict):
        return False
    marker = {
        'version': _RESOURCE_EXTRACTION_MARKER_VERSION,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_epub': dict(source_fingerprint),
        'images': {
            # These are the packaged basenames before the chapter rename pass.
            # Validation resolves them through image_rename_map.json.
            'source_filenames': sorted(set(expected_image_filenames)),
        },
        'resources': {
            resource_type: sorted(set(
                expected_resource_filenames.get(resource_type, [])
            ))
            for resource_type in _RESOURCE_MARKER_TYPES
        },
    }
    temporary_path = (
        f"{marker_path}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with open(temporary_path, 'w', encoding='utf-8') as handle:
            json.dump(marker, handle, ensure_ascii=False, indent=2)
        os.replace(temporary_path, marker_path)
        return True
    except OSError:
        try:
            os.remove(temporary_path)
        except OSError:
            pass
        return False


def _extract_all_resources(
    zf,
    output_dir,
    progress_callback=None,
    preserve_images=False,
):
    """Extract all resources with parallel processing"""
    import time
    
    extracted_resources = {
        'css': [],
        'fonts': [],
        'images': [],
        'epub_structure': [],
        'other': []
    }
    
    # Check if already extracted. Legacy timestamp-only markers are never
    # trusted: the source EPUB and every expected packaged image must validate.
    extraction_marker = os.path.join(output_dir, '.resources_extracted')
    remote_download_enabled = os.getenv(
        'DOWNLOAD_REMOTE_IMAGE_URLS', '0'
    ).strip().lower() in {'1', 'true', 'yes', 'on'}
    source_fingerprint = None
    marker_exists = os.path.isfile(extraction_marker)
    marker_valid = False
    marker_reason = ''
    if marker_exists:
        print("🔐 Validating source EPUB and extracted image fingerprint...")
        source_fingerprint = _source_epub_content_fingerprint(zf)
        marker_valid, marker_reason = _validate_resource_extraction_marker(
            extraction_marker,
            output_dir,
            source_fingerprint,
        )
        if marker_valid and (
            not remote_download_enabled or preserve_images
        ):
            print("📦 Resource fingerprint matches, skipping extraction...")
            return _count_existing_resources(output_dir, extracted_resources)

        if marker_valid:
            print(
                "♻️ Remote image cache is missing or belongs to a different "
                "source image count; refreshing EPUB image resources"
            )
        else:
            print(f"♻️ Resource fingerprint invalid ({marker_reason}); re-extracting")
            # A changed source or missing packaged image must rebuild the image
            # directory, even if the remote-image cache would normally preserve it.
            preserve_images = False
            # The old map describes the invalidated image set. Keeping it would
            # make the newly extracted original filenames resolve to stale
            # renamed targets until the later rename pass replaces the sidecar.
            try:
                os.remove(os.path.join(output_dir, 'image_rename_map.json'))
            except OSError:
                pass
        try:
            os.remove(extraction_marker)
        except OSError:
            pass
    
    _cleanup_old_resources(output_dir, preserve_images=preserve_images)
    
    # Create directories
    for resource_type in ['css', 'fonts', 'images']:
        os.makedirs(os.path.join(output_dir, resource_type), exist_ok=True)
    
    # Only print if no callback (avoid duplicates in subprocess)
    if not progress_callback:
        print(f"📦 Extracting resources in parallel...")
    
    # Get list of files to process
    file_list = [f for f in zf.namelist() if not f.endswith('/') and os.path.basename(f)]
    expected_image_filenames = []
    expected_resource_filenames = {
        resource_type: [] for resource_type in _RESOURCE_MARKER_TYPES
    }
    for file_path in file_list:
        resource_info = _categorize_resource(
            file_path,
            os.path.basename(file_path),
        )
        if resource_info and resource_info[0] == 'images':
            expected_image_filenames.append(resource_info[2])
        elif resource_info and resource_info[0] in expected_resource_filenames:
            expected_resource_filenames[resource_info[0]].append(
                resource_info[2]
            )
    
    # Thread-safe lock for extracted_resources
    resource_lock = threading.Lock()
    
    def extract_single_resource(file_path):
        if is_stop_requested():
            return None
            
        try:
            file_data = zf.read(file_path)
            resource_info = _categorize_resource(file_path, os.path.basename(file_path))
            
            if resource_info:
                resource_type, target_dir, safe_filename = resource_info
                target_path = os.path.join(output_dir, target_dir, safe_filename) if target_dir else os.path.join(output_dir, safe_filename)
                
                with open(target_path, 'wb') as f:
                    f.write(file_data)
                
                # Thread-safe update
                with resource_lock:
                    extracted_resources[resource_type].append(safe_filename)
                
                return (resource_type, safe_filename)
        except Exception as e:
            print(f"[WARNING] Failed to extract {file_path}: {e}")
            return None
    
    # Process files in parallel
    total_resources = len(file_list)
    extracted_count = 0
    
    # Use same worker count as chapter processing
    resource_workers = int(os.getenv("EXTRACTION_WORKERS", "2"))
    
    with ThreadPoolExecutor(max_workers=resource_workers) as executor:
        futures = {executor.submit(extract_single_resource, file_path): file_path 
                  for file_path in file_list}
        
        for future in as_completed(futures):
            if is_stop_requested():
                executor.shutdown(wait=False)
                break
            
            extracted_count += 1
            
            # Progress update every 20 files
            if extracted_count % 20 == 0:
                if progress_callback:
                    progress_callback(f"Extracting resources: {extracted_count}/{total_resources}")
                else:
                    # Print progress bar in terminal
                    ProgressBar.update(extracted_count, total_resources, prefix="📦 Extracting resources")
            
            # Yield to GUI periodically (can be disabled for max speed)
            if extracted_count % 10 == 0 and os.getenv("ENABLE_GUI_YIELD", "1") == "1":
                time.sleep(0.001)
                
            result = future.result()
            if result:
                resource_type, filename = result
                # Only print for important resources
                if extracted_count < 10 or resource_type in ['css', 'fonts']:
                    print(f"   📄 Extracted {resource_type}: {filename}")
    
    # Show 100% completion
    if progress_callback:
        progress_callback(f"Extracting resources: {total_resources}/{total_resources}")
    else:
        ProgressBar.update(total_resources, total_resources, prefix="📦 Extracting resources")
        ProgressBar.finish()
    
    # Mark as complete only when every expected resource exists. The later
    # chapter rename pass may change image filenames; validation follows
    # image_rename_map.json for those files.
    images_dir = os.path.join(output_dir, 'images')
    missing_after_extract = [
        name for name in set(expected_image_filenames)
        if not os.path.isfile(os.path.join(images_dir, name))
    ]
    missing_resource_after_extract = []
    for resource_type, expected_names in expected_resource_filenames.items():
        relative_dir = _RESOURCE_MARKER_DIRECTORIES[resource_type]
        for name in set(expected_names):
            candidate = (
                os.path.join(output_dir, relative_dir, name)
                if relative_dir else os.path.join(output_dir, name)
            )
            if not os.path.isfile(candidate):
                missing_resource_after_extract.append(
                    f'{relative_dir}/{name}' if relative_dir else name
                )

    if (
        is_stop_requested()
        or missing_after_extract
        or missing_resource_after_extract
    ):
        try:
            os.remove(extraction_marker)
        except OSError:
            pass
        if missing_after_extract:
            print(
                "[WARNING] Resource marker not written; missing extracted "
                f"image file(s): {', '.join(sorted(missing_after_extract)[:3])}"
            )
        if missing_resource_after_extract:
            print(
                "[WARNING] Resource marker not written; missing extracted "
                "resource file(s): "
                + ', '.join(sorted(missing_resource_after_extract)[:3])
            )
    else:
        if source_fingerprint is None:
            source_fingerprint = _source_epub_content_fingerprint(zf)
        if not _write_resource_extraction_marker(
            extraction_marker,
            source_fingerprint,
            expected_image_filenames,
            expected_resource_filenames,
        ):
            print("[WARNING] Could not write resource extraction fingerprint")
    
    _validate_critical_files(output_dir, extracted_resources)
    return extracted_resources

def _extract_chapters_universal(zf, extraction_mode="smart", parser=None, progress_callback=None, pattern_manager=None):
    """Universal chapter extraction with four modes: smart, comprehensive, full, enhanced
    
    All modes now properly merge Section/Chapter pairs
    Enhanced mode uses html2text for superior text processing
    Now with parallel processing for improved performance
    """
    # Initialize defaults if not provided
    if parser is None:
        parser = _get_best_parser()
    # pattern_manager is no longer used - kept for API compatibility
    
    # Check stop at the beginning
    if is_stop_requested():
        print("❌ Chapter extraction stopped by user")
        return [], 'unknown'
    
    # Import time for yielding
    import time
    
    # Initialize enhanced extractor if using enhanced mode
    enhanced_extractor = None
    enhanced_filtering = extraction_mode  # Default fallback
    preserve_structure = True
    
    # Special file filtering (cover, nav, title, etc.) is handled downstream
    # by the translation pipeline via TRANSLATE_SPECIAL_FILES toggle.
    # The extraction stage collects ALL HTML files unconditionally.
    
    if extraction_mode == "enhanced":
        print("🚀 Initializing Enhanced extraction mode with html2text...")
        
        # Get enhanced mode configuration from environment
        enhanced_filtering = os.getenv("ENHANCED_FILTERING", "smart")
        # Avoid 'full' with html2text to prevent XML declaration artifacts; use 'comprehensive' instead
        if str(enhanced_filtering).lower() == 'full':
            enhanced_filtering = 'comprehensive'
        preserve_structure = os.getenv("ENHANCED_PRESERVE_STRUCTURE", "1") == "1"
        
        print(f"  • Enhanced filtering level: {enhanced_filtering}")
        print(f"  • Preserve structure: {preserve_structure}")
        
        # Try to initialize enhanced extractor
        try:
            # Import our enhanced extractor (assume it's in the same directory or importable)
            from enhanced_text_extractor import EnhancedTextExtractor
            enhanced_extractor = EnhancedTextExtractor(
                filtering_mode=enhanced_filtering,
                preserve_structure=preserve_structure
            )
            print("✅ Enhanced text extractor initialized successfully")
                
        except ImportError as e:
            print(f"❌ Enhanced text extractor module not found: {e}")
            print(f"❌ Cannot use enhanced extraction mode. Please install enhanced_text_extractor or select a different extraction mode.")
            raise e
        except Exception as e:
            print(f"❌ Enhanced extractor initialization failed: {e}")
            print(f"❌ Cannot use enhanced extraction mode. Please select a different extraction mode.")
            raise e
    
    chapters = []
    sample_texts = []
    
    # First phase: Collect HTML files
    html_files = []
    file_list = zf.namelist()
    total_files = len(file_list)
    
    # Update progress for file collection
    if progress_callback and total_files > 100:
        progress_callback(f"Scanning {total_files} files in EPUB...")
    elif total_files > 100 and not progress_callback:
        # Print initial message for progress bar (only if no callback)
        print(f"📂 Scanning {total_files} files in EPUB...")
    
    for idx, name in enumerate(file_list):
        # Check stop while collecting files
        if is_stop_requested():
            print("❌ Chapter extraction stopped by user")
            return [], 'unknown'
        
        # Yield to GUI every 50 files (can be disabled for max speed)
        if idx % 50 == 0 and idx > 0:
            if os.getenv("ENABLE_GUI_YIELD", "1") == "1":
                time.sleep(0.001)  # Brief yield to GUI
            if total_files > 100:
                if progress_callback:
                    progress_callback(f"Scanning files: {idx}/{total_files}")
                else:
                    # Print progress bar in terminal
                    ProgressBar.update(idx, total_files, prefix="📂 Scanning files")
            
        if name.lower().endswith(('.xhtml', '.html', '.htm')):
            html_files.append(name)
    
    # Print final 100% progress update before finishing
    if total_files > 100:
        if progress_callback:
            progress_callback(f"Scanning files: {total_files}/{total_files}")
        else:
            # Show 100% completion
            ProgressBar.update(total_files, total_files, prefix="📂 Scanning files")
    
    # Finish progress bar if we were using it
    if total_files > 100 and not progress_callback:
        ProgressBar.finish()
    
    # Update mode description to include enhanced mode
    mode_description = {
        "smart": "potential content files",
        "comprehensive": "HTML files", 
        "full": "ALL HTML/XHTML files (no filtering)",
        "enhanced": f"files (enhanced with {enhanced_filtering} filtering)"
    }
    print(f"📚 Found {len(html_files)} {mode_description.get(extraction_mode, 'files')} in EPUB")
    
    # Sort files to ensure proper order
    html_files.sort()
    
    # Check if merging is disabled via environment variable
    disable_merging = os.getenv("DISABLE_CHAPTER_MERGING", "0") == "1"
    
    processed_files = set()
    merge_candidates = {}  # Store potential merges without reading files yet
    
    if disable_merging:
        print("📌 Chapter merging is DISABLED - processing all files independently")
    else:
        print("📌 Chapter merging is ENABLED")
        
        # Only do merging logic if not disabled
        file_groups = {}
        
        # Group files by their base number to detect Section/Chapter pairs
        for file_path in html_files:
            filename = os.path.basename(file_path)
            
            # Try different patterns to extract base number
            base_num = None
            
            # Pattern 1: "No00014" from "No00014Section.xhtml"
            match = re.match(r'(No\d+)', filename)
            if match:
                base_num = match.group(1)
            else:
                # Pattern 2: "0014" from "0014_section.html" or "0014_chapter.html"
                match = re.match(r'^(\d+)[_\-]', filename)
                if match:
                    base_num = match.group(1)
                else:
                    # Pattern 3: Just numbers at the start
                    match = re.match(r'^(\d+)', filename)
                    if match:
                        base_num = match.group(1)
            
            if base_num:
                if base_num not in file_groups:
                    file_groups[base_num] = []
                file_groups[base_num].append(file_path)
        
        # Identify merge candidates WITHOUT reading files yet
        for base_num, group_files in sorted(file_groups.items()):
            if len(group_files) == 2:
                # Check if we have a Section/Chapter pair based on filenames only
                section_file = None
                chapter_file = None
                
                for file_path in group_files:
                    basename = os.path.basename(file_path)
                    # More strict detection - must have 'section' or 'chapter' in the filename
                    if 'section' in basename.lower() and 'chapter' not in basename.lower():
                        section_file = file_path
                    elif 'chapter' in basename.lower() and 'section' not in basename.lower():
                        chapter_file = file_path
                
                if section_file and chapter_file:
                    # Store as potential merge candidate
                    merge_candidates[chapter_file] = section_file
                    processed_files.add(section_file)
                    print(f"[DEBUG] Potential merge candidate: {base_num}")
                    print(f"  Section: {os.path.basename(section_file)}")
                    print(f"  Chapter: {os.path.basename(chapter_file)}")
    
    # Filter out section files that were marked for merging
    files_to_process = []
    for file_path in html_files:
        if not disable_merging and file_path in processed_files:
            print(f"[DEBUG] Skipping section file: {file_path}")
            continue
        files_to_process.append(file_path)
    
    print(f"📚 Processing {len(files_to_process)} files after merge analysis")
    if progress_callback:
        progress_callback(f"Preparing to process {len(files_to_process)} chapters...")

    # ── Single-chapter mode ─────────────────────────────────────────────
    # When SINGLE_CHAPTER_FILTER is set (Library / Reader "Translate" on a
    # single chapter entry), only the matching HTML file is actually parsed
    # and extracted. The FULL ``files_to_process`` list is still used as the
    # numbering reference (via ``_file_index_map`` below) so the single
    # chapter receives exactly the same chapter number it would get during
    # a full extraction run.
    files_to_run = files_to_process
    _single_target = (os.getenv("SINGLE_CHAPTER_FILTER", "") or "").strip()
    if _single_target:
        _tgt_norm = _single_target.replace("\\", "/").lower().lstrip("/")
        _tgt_base = os.path.basename(_tgt_norm)
        _matches = [
            f for f in files_to_process
            if f.replace("\\", "/").lower().lstrip("/") == _tgt_norm
            or os.path.basename(f).lower() == _tgt_base
        ]
        if _matches:
            files_to_run = _matches[:1]
            print(f"🎯 Single-chapter extraction: {files_to_run[0]} "
                  f"(skipping the other {len(files_to_process) - 1} files; numbering preserved)")
            if progress_callback:
                progress_callback(
                    f"Single-chapter extraction: {os.path.basename(files_to_run[0])}")
        else:
            print(f"⚠️ SINGLE_CHAPTER_FILTER '{_single_target}' did not match any "
                  f"HTML file — falling back to full extraction")

    # Map each file to its position in the FULL processing list so chapter
    # numbering stays stable even when only a subset is actually processed.
    _file_index_map = {f: i for i, f in enumerate(files_to_process)}

    # Initialize collections for aggregating results
    file_size_groups = {}
    h1_count = 0
    h2_count = 0
    skipped_files = []

    # Progress tracking
    total_files = len(files_to_run)

    # Prepare arguments for parallel processing
    zip_file_path = zf.filename

    # Process files in parallel or sequentially based on file count
    # Only print if no callback (avoid duplicates)
    if not progress_callback:
        print(f"🚀 Processing {len(files_to_run)} HTML files...")

    # Initial progress - no message needed, progress bar will show

    candidate_chapters = []  # For smart mode
    chapters_direct = []      # For other modes

    # Decide whether to use parallel processing
    use_parallel = len(files_to_run) > 10
    
    if use_parallel:
        # Get worker count from environment variable
        max_workers = int(os.getenv("EXTRACTION_WORKERS", "2"))
        print(f"📦 Using parallel processing with {max_workers} workers...")
        if progress_callback:
            progress_callback(f"Starting {max_workers} extraction workers...")
        
        # --- Heartbeat: show elapsed time while workers start up ----
        _heartbeat_stop = threading.Event()
        _startup_start = time.time()

        def _heartbeat():
            elapsed = 0
            while not _heartbeat_stop.is_set():
                _heartbeat_stop.wait(3.0)       # tick every 3 s
                if _heartbeat_stop.is_set():
                    break
                elapsed = time.time() - _startup_start
                msg = f"⏱️ Spawning workers... elapsed {elapsed:.0f}s"
                if progress_callback:
                    progress_callback(msg)
                else:
                    print(msg, flush=True)

        _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        _hb_thread.start()
        # -----------------------------------------------------------

        # Use ProcessPoolExecutor for true multi-process parallelism
        # Now that all functions are at module level and picklable, we can use processes
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(
                    _process_single_html_file,
                    file_path=file_path,
                    file_index=_file_index_map.get(file_path, idx),
                    zip_file_path=zip_file_path,
                    parser=parser,
                    merge_candidates=merge_candidates,
                    disable_merging=disable_merging,
                    enhanced_extractor=enhanced_extractor,
                    extraction_mode=extraction_mode,
                    enhanced_filtering=enhanced_filtering,
                    preserve_structure=preserve_structure,
                    protect_angle_brackets_func=protect_angle_brackets_with_korean,
                    pattern_manager=pattern_manager,
                    files_to_process=files_to_process,
                    is_stop_requested=is_stop_requested
                ): (file_path, idx)
                for idx, file_path in enumerate(files_to_run)
            }
            # Collect results as they complete with progress tracking
            processed_count = 0
            for future in as_completed(future_to_file):
                # Stop heartbeat on first result (workers are now active)
                if not _heartbeat_stop.is_set():
                    _heartbeat_stop.set()
                    _startup_elapsed = time.time() - _startup_start
                    if _startup_elapsed >= 2.0:
                        msg = f"✅ Workers ready ({_startup_elapsed:.1f}s)"
                        if progress_callback:
                            progress_callback(msg)
                        else:
                            print(msg, flush=True)
                if is_stop_requested():
                    print("❌ Chapter processing stopped by user")
                    executor.shutdown(wait=False)
                    return [], 'unknown'
                
                try:
                    # Unpack result from _process_single_html_file
                    result = future.result()
                    chapter_info, h1_found, h2_found, file_size, sample_text, skipped_info = result
                    
                    # Update progress
                    processed_count += 1
                    if processed_count % 5 == 0:
                        if progress_callback:
                            progress_msg = f"Processing chapters: {processed_count}/{total_files} ({processed_count*100//total_files}%)"
                            progress_callback(progress_msg)
                        else:
                            # Print progress bar in terminal
                            ProgressBar.update(processed_count, total_files, prefix="📚 Processing chapters")
                    
                    # Aggregate header counts
                    if h1_found:
                        h1_count += 1
                    if h2_found:
                        h2_count += 1
                    
                    # Collect file size groups and sample texts
                    if chapter_info:
                        effective_mode = enhanced_filtering if extraction_mode == "enhanced" else extraction_mode
                        if effective_mode == "smart" and file_size > 0:
                            if file_size not in file_size_groups:
                                file_size_groups[file_size] = []
                            file_path, _ = future_to_file[future]
                            file_size_groups[file_size].append(file_path)
                            
                            # Collect sample texts
                            if sample_text and len(sample_texts) < 5:
                                sample_texts.append(sample_text)
                        
                        # For smart mode when merging is enabled, collect candidates
                        # Otherwise, add directly to chapters
                        if effective_mode == "smart" and not disable_merging:
                            candidate_chapters.append(chapter_info)
                        else:
                            chapters_direct.append(chapter_info)
                    
                    # Collect skipped info
                    if skipped_info:
                        skipped_files.append(skipped_info)
                        
                except Exception as e:
                    file_path, idx = future_to_file[future]
                    print(f"[ERROR] Process error processing {file_path}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Show 100% completion
        if progress_callback:
            progress_callback(f"Processing chapters: {total_files}/{total_files} (100%)")
        else:
            ProgressBar.update(total_files, total_files, prefix="📚 Processing chapters")
    else:
        print("📦 Using sequential processing (small file count)...")
        
        # Process files sequentially for small EPUBs
        for idx, file_path in enumerate(files_to_run):
            if is_stop_requested():
                print("❌ Chapter processing stopped by user")
                return [], 'unknown'

            # Call the module-level function directly
            result = _process_single_html_file(
                file_path=file_path,
                file_index=_file_index_map.get(file_path, idx),
                zip_file_path=zip_file_path,
                parser=parser,
                merge_candidates=merge_candidates,
                disable_merging=disable_merging,
                enhanced_extractor=enhanced_extractor,
                extraction_mode=extraction_mode,
                enhanced_filtering=enhanced_filtering,
                preserve_structure=preserve_structure,
                protect_angle_brackets_func=protect_angle_brackets_with_korean,
                pattern_manager=pattern_manager,
                files_to_process=files_to_process,
                is_stop_requested=is_stop_requested
            )
            
            # Unpack result
            chapter_info, h1_found, h2_found, file_size, sample_text, skipped_info = result
            
            # Update progress
            if (idx + 1) % 5 == 0:
                if progress_callback:
                    progress_msg = f"Processing chapters: {idx+1}/{total_files} ({(idx+1)*100//total_files}%)"
                    progress_callback(progress_msg)
                else:
                    # Print progress bar in terminal
                    ProgressBar.update(idx+1, total_files, prefix="📚 Processing chapters")
            
            # Aggregate header counts
            if h1_found:
                h1_count += 1
            if h2_found:
                h2_count += 1
            
            # Collect file size groups and sample texts
            if chapter_info:
                effective_mode = enhanced_filtering if extraction_mode == "enhanced" else extraction_mode
                if effective_mode == "smart" and file_size > 0:
                    if file_size not in file_size_groups:
                        file_size_groups[file_size] = []
                    file_size_groups[file_size].append(file_path)
                    
                    # Collect sample texts
                    if sample_text and len(sample_texts) < 5:
                        sample_texts.append(sample_text)
                
                # For smart mode when merging is enabled, collect candidates
                # Otherwise, add directly to chapters
                if effective_mode == "smart" and not disable_merging:
                    candidate_chapters.append(chapter_info)
                else:
                    chapters_direct.append(chapter_info)
            
            # Collect skipped info
            if skipped_info:
                skipped_files.append(skipped_info)
        
        # Show 100% completion for sequential mode
        if progress_callback:
            progress_callback(f"Processing chapters: {total_files}/{total_files} (100%)")
        else:
            ProgressBar.update(total_files, total_files, prefix="📚 Processing chapters")
    
    # Final progress update and cleanup progress bar
    if not progress_callback:
        ProgressBar.finish()
    else:
        progress_callback(f"Chapter processing complete: {len(candidate_chapters) + len(chapters_direct)} chapters")
    
    import time as _post_time
    _post_start = _post_time.time()
    
    # Print skip summary if any files were skipped
    if skipped_files:
        print(f"\n📊 Skipped {len(skipped_files)} files during processing:")
        empty_count = sum(1 for _, reason, _ in skipped_files if reason == 'empty')
        if empty_count > 0:
            print(f"   • {empty_count} nearly empty files")
        # Show first 3 examples if debug enabled
        if os.getenv('DEBUG_SKIP_MESSAGES', '0') == '1' and skipped_files:
            print("   Examples:")
            for path, reason, size in skipped_files[:3]:
                print(f"     - {os.path.basename(path)} ({size} chars)")
    
    # Sort direct chapters by file index to maintain order
    chapters_direct.sort(key=lambda x: x["file_index"])
    
    _sort_elapsed = _post_time.time() - _post_start
    if _sort_elapsed > 1.0:
        msg = f"  ⏱️ Post-processing: sorting ({_sort_elapsed:.1f}s)"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
    
    # Post-process smart mode candidates (only when merging is enabled)
    effective_mode = enhanced_filtering if extraction_mode == "enhanced" else extraction_mode
    if effective_mode == "smart" and candidate_chapters and not disable_merging:
        # Check stop before post-processing
        if is_stop_requested():
            print("❌ Chapter post-processing stopped by user")
            return chapters, 'unknown'
            
        print(f"\n[SMART MODE] Processing {len(candidate_chapters)} candidate files...")
        
        # Sort candidates by file index to maintain order
        candidate_chapters.sort(key=lambda x: x["file_index"])
        
        # Debug: Show what files we have
        section_files = [c for c in candidate_chapters if 'section' in c['original_basename'].lower()]
        chapter_files = [c for c in candidate_chapters if 'chapter' in c['original_basename'].lower() and 'section' not in c['original_basename'].lower()]
        other_files = [c for c in candidate_chapters if c not in section_files and c not in chapter_files]
        
        print(f"  📊 File breakdown:")
        print(f"    • Section files: {len(section_files)}")
        print(f"    • Chapter files: {len(chapter_files)}")
        print(f"    • Other files: {len(other_files)}")
        
        # Original smart mode logic when merging is enabled
        # First, separate files with detected chapter numbers from those without
        numbered_chapters = []
        unnumbered_chapters = []
        
        for idx, chapter in enumerate(candidate_chapters):
            # Yield periodically during categorization (can be disabled for max speed)
            if idx % 10 == 0 and idx > 0 and os.getenv("ENABLE_GUI_YIELD", "1") == "1":
                time.sleep(0.001)
                
            if chapter["num"] is not None:
                numbered_chapters.append(chapter)
            else:
                unnumbered_chapters.append(chapter)
        
        print(f"  • Files with chapter numbers: {len(numbered_chapters)}")
        print(f"  • Files without chapter numbers: {len(unnumbered_chapters)}")
        
        # Check if we have hash-based filenames (no numbered chapters found)
        if not numbered_chapters and unnumbered_chapters:
            print("  ⚠️ No chapter numbers found - likely hash-based filenames")
            print("  → Using file order as chapter sequence")
            
            # Sort by file index to maintain order
            unnumbered_chapters.sort(key=lambda x: x["file_index"])
            
            # Assign sequential numbers
            for i, chapter in enumerate(unnumbered_chapters, 1):
                chapter["num"] = i
                chapter["detection_method"] = f"{extraction_mode}_hash_filename_sequential" if extraction_mode == "enhanced" else "hash_filename_sequential"
                if not chapter["title"] or chapter["title"] == chapter["original_basename"]:
                    chapter["title"] = f"Chapter {i}"
            
            chapters = unnumbered_chapters
        else:
            # We have some numbered chapters
            chapters = numbered_chapters
            
            # For unnumbered files, check if they might be duplicates or appendices
            if unnumbered_chapters:
                print(f"  → Analyzing {len(unnumbered_chapters)} unnumbered files...")
                
                # Get the max chapter number
                max_num = max(c["num"] for c in numbered_chapters)
                
                # Check each unnumbered file
                for chapter in unnumbered_chapters:
                    # Check stop in post-processing loop
                    if is_stop_requested():
                        print("❌ Chapter post-processing stopped by user")
                        return chapters, 'unknown'
                        
                    # Check if it's very small (might be a separator or note)
                    if chapter["file_size"] < 200:
                        # Collect for summary instead of printing
                        # Note: _smart_mode_skips defined in outer scope
                        _smart_mode_skips.append(('small', chapter['filename'], chapter['file_size']))
                        continue
                    
                    # Check if it has similar size to existing chapters (might be duplicate)
                    size = chapter["file_size"]
                    similar_chapters = [c for c in numbered_chapters 
                                      if abs(c["file_size"] - size) < 50]
                    
                    if similar_chapters:
                        # Might be a duplicate, skip it (collect for summary)
                        _smart_mode_skips.append(('duplicate', chapter['filename'], len(similar_chapters)))
                        continue
                    
                    # Otherwise, add as appendix
                    max_num += 1
                    chapter["num"] = max_num
                    chapter["detection_method"] = f"{extraction_mode}_appendix_sequential" if extraction_mode == "enhanced" else "appendix_sequential"
                    if not chapter["title"] or chapter["title"] == chapter["original_basename"]:
                        chapter["title"] = f"Appendix {max_num}"
                    chapters.append(chapter)
                    print(f"    [ADD] Added as chapter {max_num}: {chapter['filename']}")
    else:
        # For other modes or smart mode with merging disabled
        chapters = chapters_direct
    
    # Print smart mode skip summary if any
    if '_smart_mode_skips' in locals() and _smart_mode_skips:
        print(f"\n📊 Smart mode filtering summary:")
        small_count = sum(1 for reason, _, _ in _smart_mode_skips if reason == 'small')
        dup_count = sum(1 for reason, _, _ in _smart_mode_skips if reason == 'duplicate')
        if small_count > 0:
            print(f"   • Skipped {small_count} very small files")
        if dup_count > 0:
            print(f"   • Skipped {dup_count} possible duplicates")
        # Show examples if debug enabled
        if os.getenv('DEBUG_SKIP_MESSAGES', '0') == '1':
            print("   Examples:")
            for reason, filename, detail in _smart_mode_skips[:3]:
                if reason == 'small':
                    print(f"     - {filename} ({detail} chars)")
                else:
                    print(f"     - {filename} (similar to {detail} chapters)")
        # Clear the list
        _smart_mode_skips = []
    
    # Sort chapters by number
    chapters.sort(key=lambda x: x["num"])
    
    # Ensure chapter numbers are integers
    # When merging is disabled, all chapters should have integer numbers anyway
    for chapter in chapters:
        if isinstance(chapter["num"], float):
            chapter["num"] = int(chapter["num"])
    
    # Final validation
    if chapters:
        print(f"\n✅ Final chapter count: {len(chapters)}")
        print(f"   • Chapter range: {chapters[0]['num']} - {chapters[-1]['num']}")
        
        # Enhanced mode summary
        if extraction_mode == "enhanced":
            enhanced_count = sum(1 for c in chapters if c.get('enhanced_extraction', False))
            total_chars = sum(len(c.get('body', '')) for c in chapters if c.get('enhanced_extraction', False))
            avg_chars = total_chars // enhanced_count if enhanced_count > 0 else 0
            print(f"   🚀 Enhanced extraction: {enhanced_count}/{len(chapters)} chapters, {total_chars:,} total chars (avg: {avg_chars:,})")
        
        # Check for gaps (informational only — non-contiguous numbering is
        # normal in EPUBs where spine items like images/TOC create gaps)
        chapter_nums = set(c["num"] for c in chapters)
        full_range = set(range(min(chapter_nums), max(chapter_nums) + 1))
        gaps = sorted(full_range - chapter_nums)
        if gaps:
            # Collapse into ranges for compact display
            ranges, i = [], 0
            while i < len(gaps):
                start = gaps[i]
                while i + 1 < len(gaps) and gaps[i + 1] == gaps[i] + 1:
                    i += 1
                end = gaps[i]
                ranges.append(str(start) if start == end else f"{start}–{end}")
                i += 1
            print(f"   ℹ️ Gaps in chapter numbering ({len(gaps)}): {', '.join(ranges)}")
    
    # Language detection
    combined_sample = ' '.join(sample_texts) if effective_mode == "smart" else ''
    detected_language = _detect_content_language(combined_sample) if combined_sample else 'unknown'
    
    _total_post = _post_time.time() - _post_start
    if _total_post > 1.0:
        msg = f"  ⏱️ Post-processing complete ({_total_post:.1f}s)"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
    
    if chapters:
        _print_extraction_summary(chapters, detected_language, extraction_mode, 
                                     h1_count if effective_mode == "smart" else 0, 
                                     h2_count if effective_mode == "smart" else 0,
                                     file_size_groups if effective_mode == "smart" else {})
    
    return chapters, detected_language

def _extract_chapter_info(soup, file_path, content_text, html_content, pattern_manager):
    """Extract chapter number and title from various sources with parallel pattern matching"""
    chapter_num = None
    chapter_title = None
    detection_method = None
    
    # SPECIAL HANDLING: When we have Section/Chapter pairs, differentiate them
    filename = os.path.basename(file_path)
    
    # Handle different naming patterns for Section/Chapter files
    if ('section' in filename.lower() or '_section' in filename.lower()) and 'chapter' not in filename.lower():
        # For Section files, add 0.1 to the base number
        # Try different patterns
        match = re.search(r'No(\d+)', filename)
        if not match:
            match = re.search(r'^(\d+)[_\-]', filename)
        if not match:
            match = re.search(r'^(\d+)', filename)
            
        if match:
            base_num = int(match.group(1))
            chapter_num = base_num + 0.1  # Section gets .1
            detection_method = "filename_section_special"
            
    elif ('chapter' in filename.lower() or '_chapter' in filename.lower()) and 'section' not in filename.lower():
        # For Chapter files, use the base number
        # Try different patterns
        match = re.search(r'No(\d+)', filename)
        if not match:
            match = re.search(r'^(\d+)[_\-]', filename)
        if not match:
            match = re.search(r'^(\d+)', filename)
            
        if match:
            chapter_num = int(match.group(1))
            detection_method = "filename_chapter_special"
    
    # If not handled by special logic, continue with normal extraction
    if not chapter_num:
        # Try filename first - use parallel pattern matching for better performance
        chapter_patterns = [(pattern, flags, method) for pattern, flags, method in _CHAPTER_PATTERNS 
                          if method.endswith('_number')]
        
        if len(chapter_patterns) > 3:  # Only parallelize if we have enough patterns
            # Parallel pattern matching for filename
            with ThreadPoolExecutor(max_workers=min(4, len(chapter_patterns))) as executor:
                def try_pattern(pattern_info):
                    pattern, flags, method = pattern_info
                    match = re.search(pattern, file_path, flags)
                    if match:
                        try:
                            num_str = match.group(1)
                            if num_str.isdigit():
                                return int(num_str), f"filename_{method}"
                            elif method == 'chinese_chapter_cn':
                                pass  # TransateKRtoEN import removed (patterns inlined)
                                pm = None  # No longer needed
                                converted = _convert_chinese_number(num_str, pm)
                                if converted:
                                    return converted, f"filename_{method}"
                        except (ValueError, IndexError):
                            pass
                    return None, None
                
                # Submit all patterns
                futures = [executor.submit(try_pattern, pattern_info) for pattern_info in chapter_patterns]
                
                # Check results as they complete
                for future in as_completed(futures):
                    try:
                        num, method = future.result()
                        if num:
                            chapter_num = num
                            detection_method = method
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            break
                    except Exception:
                        continue
        else:
            # Sequential processing for small pattern sets
            for pattern, flags, method in chapter_patterns:
                match = re.search(pattern, file_path, flags)
                if match:
                    try:
                        num_str = match.group(1)
                        if num_str.isdigit():
                            chapter_num = int(num_str)
                            detection_method = f"filename_{method}"
                            break
                        elif method == 'chinese_chapter_cn':
                            pass  # TransateKRtoEN import removed (patterns inlined)
                            pm = None  # No longer needed
                            converted = _convert_chinese_number(num_str, pm)
                            if converted:
                                chapter_num = converted
                                detection_method = f"filename_{method}"
                                break
                    except (ValueError, IndexError):
                        continue
    
    # Try content if not found in filename
    if not chapter_num:
        # Check ignore settings for batch translation
        batch_translate_active = os.getenv('BATCH_TRANSLATE_HEADERS', '0') == '1'
        use_title_tag = os.getenv('USE_TITLE', '0') == '1' or not batch_translate_active
        ignore_header_tags = os.getenv('IGNORE_HEADER', '0') == '1' and batch_translate_active
        
        # Prepare all text sources to check in parallel
        text_sources = []
        
        # Add title tag if using titles
        if use_title_tag and soup.title and soup.title.string:
            title_text = soup.title.string.strip()
            text_sources.append(("title", title_text, True))  # True means this can be chapter_title
        
        # Add headers if not ignored
        if not ignore_header_tags:
            for header_tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                headers = soup.find_all(header_tag)
                for header in headers[:3]:  # Limit to first 3 of each type
                    header_text = header.get_text(strip=True)
                    if header_text:
                        text_sources.append((f"header_{header_tag}", header_text, True))
        
        # Add first paragraphs
        first_elements = soup.find_all(['p', 'div'])[:5]
        for elem in first_elements:
            elem_text = elem.get_text(strip=True)
            if elem_text:
                text_sources.append(("content", elem_text, False))  # False means don't use as chapter_title
        
        # Process text sources in parallel if we have many
        if len(text_sources) > 5:
            with ThreadPoolExecutor(max_workers=min(6, len(text_sources))) as executor:
                def extract_from_source(source_info):
                    source_type, text, can_be_title = source_info
                    num, method = _extract_from_text(text, source_type, pattern_manager)
                    return num, method, text if (num and can_be_title) else None
                
                # Submit all text sources
                future_to_source = {executor.submit(extract_from_source, source): source 
                                  for source in text_sources}
                
                # Process results as they complete
                for future in as_completed(future_to_source):
                    try:
                        num, method, title = future.result()
                        if num:
                            chapter_num = num
                            detection_method = method
                            if title and not chapter_title:
                                chapter_title = title
                            # Cancel remaining futures
                            for f in future_to_source:
                                f.cancel()
                            break
                    except Exception:
                        continue
        else:
            # Sequential processing for small text sets
            for source_type, text, can_be_title in text_sources:
                num, method = _extract_from_text(text, source_type, pattern_manager)
                if num:
                    chapter_num = num
                    detection_method = method
                    if can_be_title and not chapter_title:
                        chapter_title = text
                    break
        
        # Final fallback to filename patterns
        if not chapter_num:
            filename_base = os.path.basename(file_path)
            # Parallel pattern matching for filename extraction
            if len(_FILENAME_EXTRACT_PATTERNS) > 3:
                with ThreadPoolExecutor(max_workers=min(4, len(_FILENAME_EXTRACT_PATTERNS))) as executor:
                    def try_filename_pattern(pattern):
                        match = re.search(pattern, filename_base, re.IGNORECASE)
                        if match:
                            try:
                                return int(match.group(1))
                            except (ValueError, IndexError):
                                pass
                        return None
                    
                    futures = [executor.submit(try_filename_pattern, pattern) 
                             for pattern in _FILENAME_EXTRACT_PATTERNS]
                    
                    for future in as_completed(futures):
                        try:
                            num = future.result()
                            if num:
                                chapter_num = num
                                detection_method = "filename_number"
                                for f in futures:
                                    f.cancel()
                                break
                        except Exception:
                            continue
            else:
                # Sequential for small pattern sets
                for pattern in _FILENAME_EXTRACT_PATTERNS:
                    match = re.search(pattern, filename_base, re.IGNORECASE)
                    if match:
                        chapter_num = int(match.group(1))
                        detection_method = "filename_number"
                        break
    
    # Extract title if not already found (with ignore settings support)
    if not chapter_title:
        # Check settings for batch translation
        batch_translate_active = os.getenv('BATCH_TRANSLATE_HEADERS', '0') == '1'
        use_title_tag = os.getenv('USE_TITLE', '0') == '1' or not batch_translate_active
        ignore_header_tags = os.getenv('IGNORE_HEADER', '0') == '1' and batch_translate_active
        
        # Try title tag if using titles
        if use_title_tag and soup.title and soup.title.string:
            chapter_title = soup.title.string.strip()
        
        # Try header tags if not ignored and no title found
        if not chapter_title and not ignore_header_tags:
            for header_tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                header = soup.find(header_tag)
                if header:
                    chapter_title = header.get_text(strip=True)
                    break
        
        # Final fallback
        if not chapter_title:
            chapter_title = f"Chapter {chapter_num}" if chapter_num else None
    
    chapter_title = re.sub(r'\s+', ' ', chapter_title).strip() if chapter_title else None
    
    return chapter_num, chapter_title, detection_method


def _extract_from_text(text, source_type, pattern_manager):
    """Extract chapter number from text using patterns with parallel matching for large pattern sets"""
    # Get patterns that don't end with '_number'
    text_patterns = [(pattern, flags, method) for pattern, flags, method in _CHAPTER_PATTERNS
                    if not method.endswith('_number')]
    
    # Only use parallel processing if we have many patterns
    if len(text_patterns) > 5:
        with ThreadPoolExecutor(max_workers=min(4, len(text_patterns))) as executor:
            def try_text_pattern(pattern_info):
                pattern, flags, method = pattern_info
                match = re.search(pattern, text, flags)
                if match:
                    try:
                        num_str = match.group(1)
                        if num_str.isdigit():
                            return int(num_str), f"{source_type}_{method}"
                        elif method == 'chinese_chapter_cn':
                            pass  # TransateKRtoEN import removed (patterns inlined)
                            pm = None  # No longer needed
                            converted = _convert_chinese_number(num_str, pm)
                            if converted:
                                return converted, f"{source_type}_{method}"
                    except (ValueError, IndexError):
                        pass
                return None, None
            
            # Submit all patterns
            futures = [executor.submit(try_text_pattern, pattern_info) for pattern_info in text_patterns]
            
            # Check results as they complete
            for future in as_completed(futures):
                try:
                    num, method = future.result()
                    if num:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        return num, method
                except Exception:
                    continue
    else:
        # Sequential processing for small pattern sets
        for pattern, flags, method in text_patterns:
            match = re.search(pattern, text, flags)
            if match:
                try:
                    num_str = match.group(1)
                    if num_str.isdigit():
                        return int(num_str), f"{source_type}_{method}"
                    elif method == 'chinese_chapter_cn':
                        pass  # TransateKRtoEN import removed (patterns inlined)
                        pm = None  # No longer needed
                        converted = _convert_chinese_number(num_str, pm)
                        if converted:
                            return converted, f"{source_type}_{method}"
                except (ValueError, IndexError):
                    continue
    
    return None, None

def _convert_chinese_number(cn_num, pattern_manager):
    """Convert Chinese number to integer"""
    if cn_num in _CHINESE_NUMS:
        return _CHINESE_NUMS[cn_num]
    
    if '十' in cn_num:
        parts = cn_num.split('十')
        if len(parts) == 2:
            tens = _CHINESE_NUMS.get(parts[0], 1) if parts[0] else 1
            ones = _CHINESE_NUMS.get(parts[1], 0) if parts[1] else 0
            return tens * 10 + ones
    
    return None

def _detect_content_language( text_sample):
    """Detect the primary language of content with parallel processing for large texts"""
    
    # For very short texts, use sequential processing
    if len(text_sample) < 1000:
        scripts = {
            'korean': 0,
            'japanese_hiragana': 0,
            'japanese_katakana': 0,
            'chinese': 0,
            'latin': 0
        }
        
        for char in text_sample:
            code = ord(char)
            if 0xAC00 <= code <= 0xD7AF:
                scripts['korean'] += 1
            elif 0x3040 <= code <= 0x309F:
                scripts['japanese_hiragana'] += 1
            elif 0x30A0 <= code <= 0x30FF:
                scripts['japanese_katakana'] += 1
            elif 0x4E00 <= code <= 0x9FFF:
                scripts['chinese'] += 1
            elif 0x0020 <= code <= 0x007F:
                scripts['latin'] += 1
    else:
        # For longer texts, use parallel processing
        # Split text into chunks for parallel processing
        chunk_size = max(500, len(text_sample) // (os.cpu_count() or 4))
        chunks = [text_sample[i:i + chunk_size] for i in range(0, len(text_sample), chunk_size)]
        
        # Thread-safe accumulator
        scripts_lock = threading.Lock()
        scripts = {
            'korean': 0,
            'japanese_hiragana': 0,
            'japanese_katakana': 0,
            'chinese': 0,
            'latin': 0
        }
        
        def process_chunk(text_chunk):
            """Process a chunk of text and return script counts"""
            local_scripts = {
                'korean': 0,
                'japanese_hiragana': 0,
                'japanese_katakana': 0,
                'chinese': 0,
                'latin': 0
            }
            
            for char in text_chunk:
                code = ord(char)
                if 0xAC00 <= code <= 0xD7AF:
                    local_scripts['korean'] += 1
                elif 0x3040 <= code <= 0x309F:
                    local_scripts['japanese_hiragana'] += 1
                elif 0x30A0 <= code <= 0x30FF:
                    local_scripts['japanese_katakana'] += 1
                elif 0x4E00 <= code <= 0x9FFF:
                    local_scripts['chinese'] += 1
                elif 0x0020 <= code <= 0x007F:
                    local_scripts['latin'] += 1
            
            return local_scripts
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, len(chunks))) as executor:
            # Submit all chunks
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            
            # Collect results
            for future in as_completed(futures):
                try:
                    chunk_scripts = future.result()
                    # Thread-safe accumulation
                    with scripts_lock:
                        for script, count in chunk_scripts.items():
                            scripts[script] += count
                except Exception as e:
                    print(f"[WARNING] Error processing chunk in language detection: {e}")
    
    # Language determination logic (same as original)
    total_cjk = scripts['korean'] + scripts['japanese_hiragana'] + scripts['japanese_katakana'] + scripts['chinese']
    
    if scripts['korean'] > total_cjk * 0.3:
        return 'korean'
    elif scripts['japanese_hiragana'] + scripts['japanese_katakana'] > total_cjk * 0.2:
        return 'japanese'
    elif scripts['chinese'] > total_cjk * 0.3:
        return 'chinese'
    elif scripts['latin'] > len(text_sample) * 0.7:
        return 'english'
    else:
        return 'unknown'

# Global flag to track if language has been printed
_language_printed = False

def _print_extraction_summary( chapters, detected_language, extraction_mode, h1_count, h2_count, file_size_groups):
    """Print extraction summary"""
    global _language_printed
    
    print(f"\n📊 Chapter Extraction Summary ({extraction_mode.capitalize()} Mode):")
    print(f"   • Total chapters extracted: {len(chapters)}")
    
    # Format chapter range handling both int and float
    first_num = chapters[0]['num']
    last_num = chapters[-1]['num']
    
    print(f"   • Chapter range: {first_num} to {last_num}")
    
    # Only print detected language once per session
    if not _language_printed and detected_language and detected_language != 'unknown':
        print(f"   🌐 Detected language: {detected_language}")
        _language_printed = True
    
    if extraction_mode == "smart":
        print(f"   • Primary header type: {'<h2>' if h2_count > h1_count else '<h1>'}")
    
    image_only_count = sum(1 for c in chapters if c.get('is_image_only', False))
    text_only_count = sum(1 for c in chapters if not c.get('has_images', False) and c.get('file_size', 0) >= 500)
    mixed_count = sum(1 for c in chapters if c.get('has_images', False) and c.get('file_size', 0) >= 500)
    empty_count = sum(1 for c in chapters if c.get('file_size', 0) < 50)
    
    print(f"   • Text-only chapters: {text_only_count}")
    print(f"   • Image-only chapters: {image_only_count}")
    print(f"   • Mixed content chapters: {mixed_count}")
    print(f"   • Empty/minimal content: {empty_count}")
    
    # Check for merged chapters
    merged_count = sum(1 for c in chapters if c.get('was_merged', False))
    if merged_count > 0:
        print(f"   • Merged chapters: {merged_count}")
    
    if extraction_mode == "smart":
        method_stats = Counter(c['detection_method'] for c in chapters)
        print(f"   📈 Detection methods used:")
        for method, count in method_stats.most_common():
            print(f"      • {method}: {count} chapters")
        
        large_groups = [size for size, files in file_size_groups.items() if len(files) > 1]
        if large_groups:
            print(f"   ⚠️ Found {len(large_groups)} file size groups with potential duplicates")
    else:
        print(f"   • Empty/placeholder: {empty_count}")
        
    if extraction_mode == "full":
        print(f"   🔍 Full extraction preserved all HTML structure and tags")

def _extract_epub_metadata(zf):
    """Extract comprehensive metadata from EPUB file including all custom fields"""
    meta = {}
    # Use lxml for XML if available
    try:
        import lxml
        xml_parser = 'lxml-xml'
    except ImportError:
        xml_parser = 'xml'
    try:
        if (opf_member := find_epub_opf_member(zf)):
                opf_content = zf.read(opf_member)
                soup = BeautifulSoup(opf_content, xml_parser)
                
                # Preserve every value for repeatable Dublin Core fields such
                # as dc:subject instead of silently keeping only the first.
                meta.update(extract_dc_metadata(soup))
                
                # Extract ALL meta tags (not just series)
                meta_tags = soup.find_all('meta')
                for meta_tag in meta_tags:
                    # Try different attribute names for the metadata name
                    name = meta_tag.get('name') or meta_tag.get('property', '')
                    content = meta_tag.get('content', '')
                    
                    if name and content:
                        # Store original name for debugging
                        original_name = name
                        
                        # Clean up common prefixes
                        if name.startswith('calibre:'):
                            name = name[8:]  # Remove 'calibre:' prefix
                        elif name.startswith('dc:'):
                            name = name[3:]  # Remove 'dc:' prefix
                        elif name.startswith('opf:'):
                            name = name[4:]  # Remove 'opf:' prefix
                        
                        # Normalize the field name - replace hyphens with underscores
                        name = name.replace('-', '_')
                        
                        # Don't overwrite if already exists (prefer direct tags over meta tags)
                        if name not in meta:
                            meta[name] = content
                            
                            # Debug output for custom fields
                            if original_name != name:
                                print(f"   • Found custom field: {original_name} → {name}")
                
                # Special handling for series information (maintain compatibility)
                if 'series' not in meta:
                    series_tags = soup.find_all('meta', attrs={'name': lambda x: x and 'series' in x.lower()})
                    for series_tag in series_tags:
                        series_name = series_tag.get('content', '')
                        if series_name:
                            meta['series'] = series_name
                            break
                
                # Extract refines metadata (used by some EPUB creators)
                refines_metas = soup.find_all('meta', attrs={'refines': True})
                for refine in refines_metas:
                    property_name = refine.get('property', '')
                    content = refine.get_text(strip=True) or refine.get('content', '')
                    
                    if property_name and content:
                        # Clean property name
                        if ':' in property_name:
                            property_name = property_name.split(':')[-1]
                        property_name = property_name.replace('-', '_')
                        
                        if property_name not in meta:
                            meta[property_name] = content
                
                # Log extraction summary
                print(f"📋 Extracted {len(meta)} metadata fields")
                
                # Show standard vs custom fields
                standard_keys = {'title', 'creator', 'language', 'subject', 'description', 
                               'publisher', 'date', 'identifier', 'source', 'rights', 
                               'contributor', 'type', 'format', 'relation', 'coverage'}
                custom_keys = set(meta.keys()) - standard_keys
                
                if custom_keys:
                    print(f"📋 Standard fields: {len(standard_keys & set(meta.keys()))}")
                    print(f"📋 Custom fields found: {sorted(custom_keys)}")
                    
                    # Show sample values for custom fields (truncated)
                    for key in sorted(custom_keys)[:5]:  # Show first 5 custom fields
                        value = str(meta[key])
                        if len(value) > 50:
                            value = value[:47] + "..."
                        print(f"   • {key}: {value}")
                    
                    if len(custom_keys) > 5:
                        print(f"   • ... and {len(custom_keys) - 5} more custom fields")
                
    except Exception as e:
        print(f"[WARNING] Failed to extract metadata: {e}")
        import traceback
        traceback.print_exc()
    
    return meta

def _categorize_resource( file_path, file_name):
    """Categorize a file and return (resource_type, target_dir, safe_filename)"""
    file_path_lower = file_path.lower()
    file_name_lower = file_name.lower()
    
    if file_path_lower.endswith('.css'):
        return 'css', 'css', sanitize_resource_filename(file_name)
    elif file_path_lower.endswith(('.ttf', '.otf', '.woff', '.woff2', '.eot')):
        return 'fonts', 'fonts', sanitize_resource_filename(file_name)
    elif file_path_lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg', '.bmp', '.webp')):
        return 'images', 'images', sanitize_resource_filename(file_name)
    elif (file_path_lower.endswith(('.opf', '.ncx')) or 
          file_name_lower == 'container.xml' or
          'container.xml' in file_path_lower):
        if 'container.xml' in file_path_lower:
            safe_filename = 'container.xml'
        else:
            safe_filename = file_name
        return 'epub_structure', None, safe_filename
    elif file_path_lower.endswith(('.js', '.xml', '.txt')):
        return 'other', None, sanitize_resource_filename(file_name)
    
    return None

def _cleanup_old_resources(output_dir, preserve_images=False):
    """Clean up old resource directories and EPUB structure files"""
    print("🧹 Cleaning up any existing resource directories...")
    
    cleanup_success = True
    preserve_remote_images = bool(preserve_images)
    
    for resource_type in ['css', 'fonts', 'images']:
        resource_dir = os.path.join(output_dir, resource_type)
        if resource_type == 'images' and preserve_remote_images:
            if os.path.isdir(resource_dir):
                print(
                    "   ♻️ Preserving images directory and remote download cache"
                )
            continue
        if os.path.exists(resource_dir):
            try:
                shutil.rmtree(resource_dir)
                print(f"   🗑️ Removed old {resource_type} directory")
            except PermissionError as e:
                print(f"   ⚠️ Cannot remove {resource_type} directory (permission denied) - will merge with existing files")
                cleanup_success = False
            except Exception as e:
                print(f"   ⚠️ Error removing {resource_type} directory: {e} - will merge with existing files")
                cleanup_success = False
    
    epub_structure_files = ['container.xml', 'content.opf', 'toc.ncx']
    for epub_file in epub_structure_files:
        input_path = os.path.join(output_dir, epub_file)
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
                print(f"   🗑️ Removed old {epub_file}")
            except PermissionError:
                print(f"   ⚠️ Cannot remove {epub_file} (permission denied) - will use existing file")
            except Exception as e:
                print(f"   ⚠️ Error removing {epub_file}: {e}")
    
    try:
        for file in os.listdir(output_dir):
            if file.lower().endswith(('.opf', '.ncx')):
                file_path = os.path.join(output_dir, file)
                try:
                    os.remove(file_path)
                    print(f"   🗑️ Removed old EPUB file: {file}")
                except PermissionError:
                    print(f"   ⚠️ Cannot remove {file} (permission denied)")
                except Exception as e:
                    print(f"   ⚠️ Error removing {file}: {e}")
    except Exception as e:
        print(f"⚠️ Error scanning for EPUB files: {e}")
    
    if not cleanup_success:
        print("⚠️ Some cleanup operations failed due to file permissions")
        print("   The program will continue and merge with existing files")
    
    return cleanup_success

def _count_existing_resources( output_dir, extracted_resources):
    """Count existing resources when skipping extraction"""
    for resource_type in ['css', 'fonts', 'images', 'epub_structure']:
        if resource_type == 'epub_structure':
            epub_files = []
            for file in ['container.xml', 'content.opf', 'toc.ncx']:
                if os.path.exists(os.path.join(output_dir, file)):
                    epub_files.append(file)
            try:
                for file in os.listdir(output_dir):
                    if file.lower().endswith(('.opf', '.ncx')) and file not in epub_files:
                        epub_files.append(file)
            except:
                pass
            extracted_resources[resource_type] = epub_files
        else:
            resource_dir = os.path.join(output_dir, resource_type)
            if os.path.exists(resource_dir):
                try:
                    files = [f for f in os.listdir(resource_dir) if os.path.isfile(os.path.join(resource_dir, f))]
                    extracted_resources[resource_type] = files
                except:
                    extracted_resources[resource_type] = []
    
    total_existing = sum(len(files) for files in extracted_resources.values())
    print(f"✅ Found {total_existing} existing resource files")
    return extracted_resources

def _validate_critical_files( output_dir, extracted_resources):
    """Validate that critical EPUB files were extracted"""
    total_extracted = sum(len(files) for files in extracted_resources.values())
    print(f"✅ Extracted {total_extracted} resource files:")
    
    for resource_type, files in extracted_resources.items():
        if files:
            if resource_type == 'epub_structure':
                print(f"   • EPUB Structure: {len(files)} files")
                for file in files:
                    print(f"     - {file}")
            else:
                print(f"   • {resource_type.title()}: {len(files)} files")
    
    critical_files = ['container.xml']
    missing_critical = [f for f in critical_files if not os.path.exists(os.path.join(output_dir, f))]
    
    if missing_critical:
        print(f"⚠️ WARNING: Missing critical EPUB files: {missing_critical}")
        print("   This may prevent proper EPUB reconstruction!")
    else:
        print("✅ All critical EPUB structure files extracted successfully")
    
    opf_files = [f for f in extracted_resources['epub_structure'] if f.lower().endswith('.opf')]
    if not opf_files:
        print("⚠️ WARNING: No OPF file found! This will prevent EPUB reconstruction.")
    else:
        print(f"✅ Found OPF file(s): {opf_files}")

def _is_expected_cover_chapter(chapter):
    """Return True for EPUB cover/title-page files expected to be image-only."""
    if not isinstance(chapter, dict):
        return False
    if chapter.get('is_cover') is True:
        return True

    import re
    for key in (
        'title', 'filename', 'original_filename', 'original_basename',
        'original_html_file',
    ):
        value = str(chapter.get(key, '') or '').strip().lower()
        if not value:
            continue
        basename = os.path.basename(value.replace('\\', '/'))
        stem = os.path.splitext(basename)[0]
        compact = re.sub(r'[^a-z0-9]+', '', stem)
        normalized = re.sub(r'[^a-z0-9]+', '_', stem).strip('_')
        if (
            normalized in {'cover', 'cover_page', 'title_page'}
            or compact in {'cover', 'coverpage', 'titlepage'}
        ):
            return True
    return False


def _create_extraction_report( output_dir, metadata, chapters, extracted_resources):
    """Create comprehensive extraction report with HTML file tracking"""
    report_path = os.path.join(output_dir, 'extraction_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("EPUB Extraction Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"EXTRACTION MODE: {metadata.get('extraction_mode', 'unknown').upper()}\n\n")
        
        f.write("METADATA:\n")
        for key, value in metadata.items():
            if key not in ['chapter_titles', 'extracted_resources', 'extraction_mode']:
                f.write(f"  {key}: {value}\n")
        
        f.write(f"\nCHAPTERS ({len(chapters)}):\n")
        
        text_chapters = []
        image_only_chapters = []
        mixed_chapters = []
        
        for chapter in chapters:
            if chapter.get('has_images') and chapter.get('file_size', 0) < 500:
                image_only_chapters.append(chapter)
            elif chapter.get('has_images') and chapter.get('file_size', 0) >= 500:
                mixed_chapters.append(chapter)
            else:
                text_chapters.append(chapter)
        
        if text_chapters:
            f.write(f"\n  TEXT CHAPTERS ({len(text_chapters)}):\n")
            for c in text_chapters:
                f.write(f"    {c['num']:3d}. {c['title']} ({c['detection_method']})\n")
                if c.get('original_html_file'):
                    f.write(f"         → {c['original_html_file']}\n")
        
        if image_only_chapters:
            f.write(f"\n  IMAGE-ONLY CHAPTERS ({len(image_only_chapters)}):\n")
            for c in image_only_chapters:
                f.write(f"    {c['num']:3d}. {c['title']} (images: {c.get('image_count', 0)})\n")
                if c.get('original_html_file'):
                    f.write(f"         → {c['original_html_file']}\n")
                if 'body' in c:
                    try:
                        soup = BeautifulSoup(c['body'], 'html.parser')
                        images = soup.find_all('img')
                        for img in images[:3]:
                            src = img.get('src', 'unknown')
                            f.write(f"         • Image: {src}\n")
                        if len(images) > 3:
                            f.write(f"         • ... and {len(images) - 3} more images\n")
                    except:
                        pass
        
        if mixed_chapters:
            f.write(f"\n  MIXED CONTENT CHAPTERS ({len(mixed_chapters)}):\n")
            for c in mixed_chapters:
                f.write(f"    {c['num']:3d}. {c['title']} (text: {c.get('file_size', 0)} chars, images: {c.get('image_count', 0)})\n")
                if c.get('original_html_file'):
                    f.write(f"         → {c['original_html_file']}\n")
        
        f.write(f"\nRESOURCES EXTRACTED:\n")
        for resource_type, files in extracted_resources.items():
            if files:
                if resource_type == 'epub_structure':
                    f.write(f"  EPUB Structure: {len(files)} files\n")
                    for file in files:
                        f.write(f"    - {file}\n")
                else:
                    f.write(f"  {resource_type.title()}: {len(files)} files\n")
                    for file in files[:5]:
                        f.write(f"    - {file}\n")
                    if len(files) > 5:
                        f.write(f"    ... and {len(files) - 5} more\n")
        
        f.write(f"\nCHAPTER PAYLOADS PREPARED:\n")
        chapter_payloads_ready = sum(
            1 for chapter in chapters
            if isinstance(chapter.get('body'), str)
        )
        f.write(
            f"  Total: {chapter_payloads_ready}/{len(chapters)} chapters\n"
        )
        f.write(
            "  Storage: in-memory chapter bodies; the async worker writes "
            "chapters_full.json after extraction\n"
        )
        
        f.write(f"\nPOTENTIAL ISSUES:\n")
        issues = []
        
        actionable_image_only_chapters = [
            chapter for chapter in image_only_chapters
            if not _is_expected_cover_chapter(chapter)
        ]
        if actionable_image_only_chapters:
            image_only_count = len(actionable_image_only_chapters)
            chapter_word = "chapter" if image_only_count == 1 else "chapters"
            verb = "contains" if image_only_count == 1 else "contain"
            issues.append(
                f"  • {image_only_count} {chapter_word} {verb} only images "
                "(may need OCR)"
            )
        
        missing_payloads = sum(
            1 for chapter in chapters
            if not isinstance(chapter.get('body'), str)
        )
        if missing_payloads > 0:
            issues.append(
                f"  • {missing_payloads} chapters are missing extracted content"
            )
        
        if not extracted_resources.get('epub_structure'):
            issues.append("  • No EPUB structure files found (may affect reconstruction)")
        
        if not issues:
            f.write("  None detected - extraction appears successful!\n")
        else:
            for issue in issues:
                f.write(issue + "\n")
    
    print(f"📄 Saved extraction report to: {report_path}")

def _log_extraction_summary(chapters, extracted_resources, detected_language):
    """Log readiness of the chapter payload consumed by translation."""
    extraction_mode = chapters[0].get('extraction_mode', 'unknown') if chapters else 'unknown'
    chapter_payloads_ready = sum(
        1 for chapter in chapters
        if isinstance(chapter.get('body'), str)
    )
    chapter_data_ready = bool(
        chapters and chapter_payloads_ready == len(chapters)
    )
    
    print(f"\n✅ {extraction_mode.capitalize()} extraction complete!")
    print(f"   📚 Chapters: {len(chapters)}")
    print(
        f"   📄 Chapter payloads ready: "
        f"{chapter_payloads_ready}/{len(chapters)}"
    )
    print(f"   🎨 Resources: {sum(len(files) for files in extracted_resources.values())}")
    print(f"   🌍 Language: {detected_language}")
    
    image_only_count = sum(1 for c in chapters if c.get('has_images') and c.get('file_size', 0) < 500)
    if image_only_count > 0:
        print(f"   📸 Image-only chapters: {image_only_count}")
    
    epub_files = extracted_resources.get('epub_structure', [])
    if epub_files:
        print(f"   📋 EPUB Structure: {len(epub_files)} files ({', '.join(epub_files)})")
    else:
        print(f"   ⚠️ No EPUB structure files extracted!")
    
    print(f"\n🔍 Pre-flight check readiness:")
    print(
        f"   ✅ Chapter data: "
        f"{'READY' if chapter_data_ready else 'NOT READY'}"
    )
    print(f"   ✅ Metadata: READY")
    print(f"   ✅ Resources: READY")
    
def _process_single_html_file(
    file_path,
    file_index,
    zip_file_path,
    parser,
    merge_candidates,
    disable_merging,
    enhanced_extractor,
    extraction_mode,
    enhanced_filtering,
    preserve_structure,
    protect_angle_brackets_func,
    pattern_manager,
    files_to_process,
    is_stop_requested
):
    """Process a single HTML file from an EPUB - standalone function for multiprocessing.
    
    This function is at module level to be picklable for ProcessPoolExecutor.
    All needed data must be passed as parameters.
    
    Returns:
        tuple: (chapter_info, h1_found, h2_found, file_size, sample_text, skipped_info)
        - chapter_info: dict with chapter data, or None if skipped/error
        - h1_found: bool indicating if h1 tags were found
        - h2_found: bool indicating if h2 tags were found  
        - file_size: int size of content text
        - sample_text: str text sample for language detection
        - skipped_info: tuple (file_path, reason, detail) if skipped, else None
    """
    from bs4 import BeautifulSoup
    import os
    import zipfile
    
    # Check stop
    if is_stop_requested():
        return None, False, False, 0, '', None
    
    try:
        # Open our own ZipFile instance for thread safety
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            # Read file data
            file_data = zf.read(file_path)
        
        # Decode the file data
        html_content = None
        detected_encoding = None
        for encoding in ['utf-8', 'utf-16', 'gb18030', 'shift_jis', 'euc-kr', 'gbk', 'big5']:
            try:
                html_content = file_data.decode(encoding)
                detected_encoding = encoding
                break
            except UnicodeDecodeError:
                continue
        
        if not html_content:
            print(f"[WARNING] Could not decode {file_path}")
            return None, False, False, 0, '', None
        
        # Check if this file needs merging
        if not disable_merging and file_path in merge_candidates:
            section_file = merge_candidates[file_path]
            print(f"[DEBUG] Processing merge for: {file_path}")
            
            try:
                # Read section file with our own ZipFile
                with zipfile.ZipFile(zip_file_path, 'r') as zf:
                    section_data = zf.read(section_file)
                section_html = None
                for encoding in ['utf-8', 'utf-16', 'gb18030', 'shift_jis', 'euc-kr', 'gbk', 'big5']:
                    try:
                        section_html = section_data.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if section_html:
                    # Quick check if section is small enough to merge
                    section_soup = BeautifulSoup(section_html, parser)
                    section_text = section_soup.get_text(strip=True)
                    
                    if len(section_text) < 200:  # Merge if section is small
                        # Extract body content
                        chapter_soup = BeautifulSoup(html_content, parser)
                        
                        if section_soup.body:
                            section_body_content = ''.join(str(child) for child in section_soup.body.children)
                        else:
                            section_body_content = section_html
                        
                        if chapter_soup.body:
                            chapter_body_content = ''.join(str(child) for child in chapter_soup.body.children)
                        else:
                            chapter_body_content = html_content
                        
                        # Merge content
                        html_content = section_body_content + "\n<hr/>\n" + chapter_body_content
                        print(f"  → MERGED: Section ({len(section_text)} chars) + Chapter")
                    else:
                        print(f"  → NOT MERGED: Section too large ({len(section_text)} chars)")
                
            except Exception as e:
                print(f"[WARNING] Failed to merge {file_path}: {e}")
        
        # === ENHANCED EXTRACTION POINT ===
        content_html = None
        content_text = None
        chapter_title = None
        enhanced_extraction_used = False
        
        # Determine whether to use enhanced extractor
        use_enhanced = enhanced_extractor and extraction_mode == "enhanced"
        force_bs_traditional = False
        try:
            force_bs = os.getenv('FORCE_BS_FOR_TRADITIONAL', '0') == '1'
            model_env = os.getenv('MODEL', '')
            # Check for traditional translation API (inline to avoid circular imports)
            is_traditional_api = model_env in ['deepl', 'google-translate', 'google-translate-free'] or model_env.startswith('deepl/') or model_env.startswith('google-translate/')
            if force_bs and is_traditional_api:
                use_enhanced = False
                force_bs_traditional = True
        except Exception:
            pass
        
        # Use enhanced extractor if available and allowed
        if use_enhanced:
            clean_content, _, chapter_title = enhanced_extractor.extract_chapter_content(
                html_content, enhanced_filtering
            )
            enhanced_extraction_used = True
            
            content_html = clean_content
            content_text = clean_content
        
        # BeautifulSoup method (only for non-enhanced modes)
        if not enhanced_extraction_used:
            if extraction_mode == "enhanced" and not force_bs_traditional:
                print(f"❌ Skipping {file_path} - enhanced extraction required but not available")
                return None, False, False, 0, '', None
            
            # Parse the (possibly merged) content
            protected_html = protect_angle_brackets_func(html_content)
            soup = BeautifulSoup(protected_html, parser)
            
            # Get effective mode for filtering
            effective_filtering = enhanced_filtering if extraction_mode == "enhanced" else extraction_mode
            
            # In full mode, keep the entire HTML structure
            if effective_filtering == "full":
                content_html = html_content
                content_text = soup.get_text(strip=True)
            else:
                # Smart and comprehensive modes extract body content
                if soup.body:
                    content_html = str(soup.body)
                    content_text = soup.body.get_text(strip=True)
                else:
                    content_html = html_content
                    content_text = soup.get_text(strip=True)
            content_html = unescape_valid_html_tag_entities(content_html)
            if os.getenv('FIX_STRAY_P_GT_BS', '0') == '1':
                content_html = fix_stray_p_gt_artifacts(content_html)
            
            # Extract title (with ignore settings support)
            chapter_title = None
            
            # Check settings for batch translation
            batch_translate_active = os.getenv('BATCH_TRANSLATE_HEADERS', '0') == '1'
            use_title_tag = os.getenv('USE_TITLE', '0') == '1' or not batch_translate_active
            ignore_header_tags = os.getenv('IGNORE_HEADER', '0') == '1' and batch_translate_active
            
            # Extract from title tag if using titles
            if use_title_tag and soup.title and soup.title.string:
                chapter_title = soup.title.string.strip()
            
            # Extract from header tags if not ignored and no title found
            if not chapter_title and not ignore_header_tags:
                for header_tag in ['h1', 'h2', 'h3']:
                    header = soup.find(header_tag)
                    if header:
                        chapter_title = header.get_text(strip=True)
                        break
            
            # Fallback to filename if nothing found
            if not chapter_title:
                chapter_title = os.path.splitext(os.path.basename(file_path))[0]
        
        # Get the effective extraction mode for processing logic
        effective_mode = enhanced_filtering if extraction_mode == "enhanced" else extraction_mode
        
        # Skip truly empty files in smart mode
        if effective_mode == "smart" and not disable_merging and len(content_text.strip()) < 10:
            empty_soup = BeautifulSoup(html_content, parser)
            if not empty_soup.find('img'):
                skipped_info = (file_path, 'empty', len(content_text))
                return None, False, False, 0, '', skipped_info
        
        # Get actual chapter number based on original position
        actual_chapter_num = files_to_process.index(file_path) + 1
        
        # Mode-specific logic
        detection_method = None
        h1_found = False
        h2_found = False
        
        if effective_mode == "comprehensive" or effective_mode == "full":
            # For comprehensive/full mode, use sequential numbering
            chapter_num = actual_chapter_num
            
            if not chapter_title:
                chapter_title = os.path.splitext(os.path.basename(file_path))[0]
            
            detection_method = f"{extraction_mode}_sequential" if extraction_mode == "enhanced" else f"{effective_mode}_sequential"
            
        elif effective_mode == "smart":
            # For smart mode, when merging is disabled, use sequential numbering
            if disable_merging:
                chapter_num = actual_chapter_num
                
                if not chapter_title:
                    chapter_title = os.path.splitext(os.path.basename(file_path))[0]
                
                detection_method = f"{extraction_mode}_sequential_no_merge" if extraction_mode == "enhanced" else "sequential_no_merge"
            else:
                # When merging is enabled, try to extract chapter info
                protected_html = protect_angle_brackets_func(html_content)
                soup = BeautifulSoup(protected_html, parser)
                
                # Count headers
                h1_tags = soup.find_all('h1')
                h2_tags = soup.find_all('h2')
                h1_found = len(h1_tags) > 0
                h2_found = len(h2_tags) > 0
                
                # Extract chapter number and title
                chapter_num, extracted_title, detection_method = _extract_chapter_info(
                    soup, file_path, content_text, html_content, pattern_manager
                )
                
                # Use extracted title if we don't have one
                if extracted_title and not chapter_title:
                    chapter_title = extracted_title
                
                # For hash-based filenames, chapter_num might be None
                if chapter_num is None:
                    chapter_num = actual_chapter_num
                    detection_method = f"{extraction_mode}_sequential_fallback" if extraction_mode == "enhanced" else "sequential_fallback"
                    print(f"[DEBUG] No chapter number found in {file_path}, assigning: {chapter_num}")
        
        # Filter content_html for title/header settings (before processing)
        batch_translate_active = os.getenv('BATCH_TRANSLATE_HEADERS', '0') == '1'
        use_title_tag = os.getenv('USE_TITLE', '0') == '1' or not batch_translate_active
        ignore_header_tags = os.getenv('IGNORE_HEADER', '0') == '1' and batch_translate_active
        remove_duplicate_h1_p = os.getenv('REMOVE_DUPLICATE_H1_P', '0') == '1'
        
        if (not use_title_tag or ignore_header_tags or remove_duplicate_h1_p) and content_html and not enhanced_extraction_used:
            # Parse the content HTML to remove unused tags
            content_soup = BeautifulSoup(content_html, parser)
            
            # Remove title tags if not using titles
            if not use_title_tag:
                for title_tag in content_soup.find_all('title'):
                    title_tag.decompose()
            
            # Remove header tags if ignored
            if ignore_header_tags:
                for header_tag in content_soup.find_all(['h1', 'h2', 'h3']):
                    header_tag.decompose()
            
            # Remove duplicate heading+P pairs, ignoring empty tags in between.
            if remove_duplicate_h1_p:
                remove_duplicate_heading_paragraph_pairs(content_soup, check_previous=False)
            
            # Update content_html with filtered version
            content_html = str(content_soup)
        
        # Process images and metadata
        protected_html = protect_angle_brackets_func(html_content)
        soup = BeautifulSoup(protected_html, parser)
        images = soup.find_all('img')
        image_srcs = _collect_image_srcs(soup)
        has_images = len(image_srcs) > 0
        is_image_only_chapter = has_images and len(content_text.strip()) < 500
        
        if is_image_only_chapter:
            print(f"[DEBUG] Image-only chapter detected: {file_path} ({len(images)} images, {len(content_text)} chars)")
        
        # Calculate content hash (inline to avoid circular imports)
        import hashlib
        content_hash = hashlib.sha256(content_html.encode('utf-8', errors='ignore')).hexdigest()
        
        file_size = len(content_text)
        sample_text = content_text[:500] if effective_mode == "smart" else ''
        
        # Ensure chapter_num is always an integer
        if isinstance(chapter_num, float):
            chapter_num = int(chapter_num)
        if _is_configured_special_file(file_path):
            chapter_num = 0
            detection_method = "configured_special_file"
        
        # Create chapter info
        chapter_info = {
            "num": chapter_num,
            "title": chapter_title or f"Chapter {chapter_num}",
            "body": content_html,
            "filename": file_path,
            # IMPORTANT: For PDFs, we must preserve the original filename including extension
            # so that chapter_splitter.py can detect it as PDF content.
            # But we also want to preserve the basename for display/logging.
            "source_file": os.path.basename(zip_file_path) if zip_file_path else file_path,
            "original_filename": os.path.basename(file_path),
            "original_basename": os.path.splitext(os.path.basename(file_path))[0],
            "content_hash": content_hash,
            "detection_method": detection_method if detection_method else "pending",
            "file_size": file_size,
            "has_images": has_images,
            "image_count": len(image_srcs),
            "is_empty": len(content_text.strip()) == 0,
            "is_image_only": is_image_only_chapter,
            "extraction_mode": extraction_mode,
            "file_index": file_index
        }
        
        # Add enhanced extraction info if used
        if enhanced_extraction_used:
            chapter_info["enhanced_extraction"] = True
            chapter_info["enhanced_filtering"] = enhanced_filtering
            chapter_info["preserve_structure"] = preserve_structure
            chapter_info["markdown_provenance"] = getattr(enhanced_extractor, "last_markdown_provenance", {})
            chapter_info["html2text_blocks"] = getattr(enhanced_extractor, "last_html2text_blocks", []) or []
            chapter_info["html2text_blocks_source_hash"] = content_hash

        # Store original HTML for image-only chapters so text-mode copy-as-is
        # preserves document metadata such as <title> and stylesheet links.
        if enhanced_extraction_used or is_image_only_chapter:
            chapter_info["original_html"] = html_content
        
        # Add merge info if applicable
        if not disable_merging and file_path in merge_candidates:
            chapter_info["was_merged"] = True
            chapter_info["merged_with"] = merge_candidates[file_path]
        
        if effective_mode == "smart":
            chapter_info["language_sample"] = content_text[:500]
            # Debug for section files
            if 'section' in chapter_info['original_basename'].lower():
                print(f"[DEBUG] Added section file to candidates: {chapter_info['original_basename']} (size: {chapter_info['file_size']})")
        
        return chapter_info, h1_found, h2_found, file_size, sample_text, None
                    
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, False, False, 0, '', None
