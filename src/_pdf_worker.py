#!/usr/bin/env python3
"""
PDF Generation Worker - Runs PDF generation in a separate process to prevent GUI freezing.

Protocol:
  - Receives a JSON config file path as argv[1]
  - Outputs [PROGRESS], [INFO], [ERROR], [RESULT] lines to stdout
  - The config file contains all parameters needed for PDF generation
"""

import sys
import os
import io

# Force UTF-8 encoding for stdout/stderr on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import time
import traceback
import mimetypes
import base64
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def log(msg):
    """Output a progress message to stdout for the manager to read."""
    print(f"[PROGRESS] {msg}", flush=True)


def info(msg):
    """Output an info message."""
    print(f"[INFO] {msg}", flush=True)


def _read_image_as_pdf_compatible(fpath, ctype):
    """Read image bytes, converting webp to JPEG/PNG since WeasyPrint does not support webp."""
    if ctype == 'image/webp':
        try:
            from PIL import Image
            from io import BytesIO
            _compression_on = os.environ.get('ENABLE_IMAGE_COMPRESSION', '0') == '1'
            with Image.open(fpath) as img:
                buf = BytesIO()
                if not _compression_on:
                    if img.mode not in ('RGB', 'RGBA', 'L', 'LA'):
                        img = img.convert('RGBA')
                    img.save(buf, format='PNG')
                    return buf.getvalue(), 'image/png'
                pdf_fmt = os.environ.get('PDF_IMAGE_FORMAT', 'jpeg').lower()
                if pdf_fmt == 'png':
                    optimize = os.environ.get('PDF_PNG_OPTIMIZE', '1') == '1'
                    compress_level = int(os.environ.get('PDF_PNG_COMPRESS_LEVEL', '6'))
                    if img.mode not in ('RGB', 'RGBA', 'L', 'LA'):
                        img = img.convert('RGBA')
                    img.save(buf, format='PNG', optimize=optimize, compress_level=compress_level)
                    return buf.getvalue(), 'image/png'
                else:
                    quality = int(os.environ.get('IMAGE_COMPRESSION_QUALITY', '80'))
                    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode != 'RGBA':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[3])
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(buf, format='JPEG', quality=quality, optimize=True)
                    return buf.getvalue(), 'image/jpeg'
        except Exception:
            pass
    with open(fpath, 'rb') as f:
        return f.read(), ctype


def _build_pdf_toc_html(chapter_titles_info, settings, chapter_page_map):
    """Build HTML for PDF table of contents with clickable links and page numbers"""
    import html as _html

    toc_html = (
        '<html><head><style>'
        'body { font-family: serif; margin: 40px; }'
        'h1 { text-align: center; margin-bottom: 30px; }'
        'ul { list-style-type: none; padding: 0; margin: 0; }'
        'li { margin-bottom: 6px; padding: 4px 0; border-bottom: 1px dotted #ccc; }'
        'a { text-decoration: none; color: #333; display: flex; justify-content: space-between; }'
        'a:hover { color: #0066cc; text-decoration: underline; }'
        '.title { flex: 1; }'
        '.page-num { font-weight: bold; min-width: 40px; text-align: right; margin-left: 10px; }'
        '</style></head><body>'
        '<h1>Table of Contents</h1><ul>'
    )

    _dedup_on = os.environ.get('DEDUPLICATE_TOC', '0') == '1'
    _seen_titles = set()

    for chap_num in sorted(chapter_titles_info.keys()):
        title, confidence, source = chapter_titles_info[chap_num]
        if title.strip().lower() in ('untitled chapter', 'untitled'):
            continue
        if _dedup_on:
            if title in _seen_titles:
                continue
            _seen_titles.add(title)
        safe_title = _html.escape(title)

        page_num = ""
        if settings.get('toc_numbers') and chapter_page_map:
            page_num = chapter_page_map.get(chap_num, "")
            if not page_num and source:
                page_num = chapter_page_map.get(source, "")
            page_num = str(page_num) if page_num else ""

        page_span = f'<span class="page-num">{page_num}</span>' if page_num else ''
        toc_html += f'<li><a href="#chapter-{chap_num}"><span class="title">{safe_title}</span>{page_span}</a></li>'

    toc_html += '</ul></body></html>'
    return toc_html


def run_pdf_generation(config_path):
    """Main PDF generation logic, extracted from epub_converter._generate_pdf."""
    import html as html_module

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Restore environment variables from config
    for key, val in config.get('env_vars', {}).items():
        os.environ[key] = str(val)

    output_dir = config['output_dir']
    images_dir = config['images_dir']
    css_dir = config['css_dir']
    html_files = config['html_files']
    # chapter_titles_info keys must be integers
    chapter_titles_info = {int(k): tuple(v) for k, v in config['chapter_titles_info'].items()}
    processed_images = config.get('processed_images', {})
    cover_file = config.get('cover_file')
    metadata = config.get('metadata', {})

    # Ensure fontconfig is available on Windows for WeasyPrint
    import tempfile as _tempfile
    if not os.environ.get("FONTCONFIG_FILE"):
        _fc_dir = _tempfile.mkdtemp(prefix="fontconfig_")
        _fc_path = os.path.join(_fc_dir, "fonts.conf")
        if not os.path.exists(_fc_path):
            with open(_fc_path, "w", encoding="utf-8") as _f:
                _f.write('<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n'
                         '<fontconfig><dir>WINDOWSFONTDIR</dir>'
                         '<cachedir>~/.cache/fontconfig</cachedir></fontconfig>\n')
        os.environ["FONTCONFIG_FILE"] = _fc_path
        os.environ["FONTCONFIG_PATH"] = _fc_dir
        os.environ["FC_CONFIG_FILE"] = _fc_path

    try:
        from weasyprint import HTML as WeasyHTML
    except ImportError:
        log("⚠️ WeasyPrint not installed - PDF generation disabled.")
        print(f'[RESULT] {json.dumps({"success": False, "error": "WeasyPrint not installed"})}', flush=True)
        return

    log("📄 Generating PDF...")
    start_time = time.time()

    settings = {
        'page_numbers': os.environ.get('PDF_PAGE_NUMBERS', '1') == '1',
        'page_number_alignment': os.environ.get('PDF_PAGE_NUMBER_ALIGNMENT', 'center'),
        'toc': os.environ.get('PDF_GENERATE_TOC', '1') == '1',
        'toc_numbers': os.environ.get('PDF_TOC_PAGE_NUMBERS', '1') == '1',
    }

    # Determine PDF output path
    # Import FileUtils for sanitize_filename
    try:
        from epub_converter import FileUtils
        book_title = metadata.get('title', os.path.basename(output_dir))
        safe_title = FileUtils.sanitize_filename(book_title, allow_unicode=True)
    except Exception:
        book_title = metadata.get('title', os.path.basename(output_dir))
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', book_title)
    pdf_path = os.path.join(output_dir, f"{safe_title}.pdf")

    log(f"  PDF output: {pdf_path}")
    log(f"  Settings: page_numbers={settings['page_numbers']}, toc={settings['toc']}, "
        f"toc_numbers={settings['toc_numbers']}, alignment={settings['page_number_alignment']}")
    compression_enabled = os.environ.get('ENABLE_IMAGE_COMPRESSION', '0') == '1'
    if compression_enabled:
        pdf_img_fmt = os.environ.get('PDF_IMAGE_FORMAT', 'jpeg').upper()
        if pdf_img_fmt == 'PNG':
            img_detail = "format=PNG"
        else:
            img_quality = os.environ.get('IMAGE_COMPRESSION_QUALITY', '80')
            img_detail = f"format=JPEG, quality={img_quality}"
        log(f"  Image: compression=enabled, {img_detail}")
    else:
        log("  Image: compression=disabled")

    # --- Build image lookup ---
    _mime_fallback = {
        '.webp': 'image/webp', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png', '.gif': 'image/gif', '.bmp': 'image/bmp',
        '.svg': 'image/svg+xml',
    }
    images_by_name = {}
    if os.path.isdir(images_dir):
        for fname in os.listdir(images_dir):
            fpath = os.path.join(images_dir, fname)
            if os.path.isfile(fpath):
                ctype, _ = mimetypes.guess_type(fpath)
                if not ctype:
                    ctype = _mime_fallback.get(os.path.splitext(fname)[1].lower())
                if ctype and ctype.startswith('image/'):
                    images_by_name[fname] = (fpath, ctype)
                    base = os.path.splitext(fname)[0]
                    images_by_name[base] = (fpath, ctype)

    # --- Pre-build ALL data URIs in parallel ---
    _data_uri_cache = {}
    _unique_images = {}
    for key, (fpath, ctype) in images_by_name.items():
        if fpath not in _unique_images:
            _unique_images[fpath] = (ctype, [])
        _unique_images[fpath][1].append(key)

    num_images = len(_unique_images)
    if num_images > 0:
        num_workers = int(os.environ.get('EXTRACTION_WORKERS', '4'))
        num_workers = max(1, min(num_workers, num_images))
        log(f"  🖼️ Pre-converting {num_images} images for PDF with {num_workers} threads...")

        _img_done = [0]
        _img_lock = threading.Lock()

        def _convert_one_image(fpath, ctype, keys):
            try:
                img_bytes, final_ctype = _read_image_as_pdf_compatible(fpath, ctype)
                b64 = base64.b64encode(img_bytes).decode('utf-8')
                data_uri = f"data:{final_ctype};base64,{b64}"
                for k in keys:
                    _data_uri_cache[k] = data_uri
            except Exception:
                pass
            with _img_lock:
                _img_done[0] += 1
                done = _img_done[0]
            if done % 500 == 0 or done == num_images:
                log(f"  🖼️ Pre-converted {done}/{num_images} images")

        with ThreadPoolExecutor(max_workers=num_workers) as img_executor:
            futures = []
            for fpath, (ctype, keys) in _unique_images.items():
                futures.append(img_executor.submit(_convert_one_image, fpath, ctype, keys))
            for f in futures:
                f.result()

        log(f"  ✅ Image pre-conversion complete ({len(_data_uri_cache)} cache entries)")

    def _data_uri_for_src(src_value):
        if not src_value or src_value.startswith('data:'):
            return None
        from urllib.parse import unquote
        raw = unquote(src_value).replace('\\', '/')
        filename = os.path.basename(raw)
        for key in [raw, filename, os.path.splitext(filename)[0]]:
            if key in _data_uri_cache:
                return _data_uri_cache[key]
        return None

    def _embed_images_in_html(content):
        def replace_attr(match):
            attr = match.group(1)
            quote = match.group(2)
            src_value = match.group(3)
            data_uri = _data_uri_for_src(src_value)
            if data_uri:
                return f'{attr}={quote}{data_uri}{quote}'
            return match.group(0)
        content = re.sub(
            r'(\b(?:src|href|xlink:href)\b)\s*=\s*([\'"])([^\'"]+)\2',
            replace_attr, content, flags=re.IGNORECASE
        )
        def replace_css_url(match):
            quote = match.group(1) or ''
            url_value = match.group(2)
            if url_value.startswith('data:'):
                return match.group(0)
            data_uri = _data_uri_for_src(url_value)
            if data_uri:
                return f'url("{data_uri}")'
            return match.group(0)
        content = re.sub(
            r'url\(\s*([\'"]?)([^\'")\s]+)\1\s*\)',
            replace_css_url, content, flags=re.IGNORECASE
        )
        return content

    def _extract_body_content(html_content):
        body_match = re.search(r'<body[^>]*>(.*)</body>', html_content, re.DOTALL | re.IGNORECASE)
        if body_match:
            return body_match.group(1)
        return html_content

    def _rewrite_epub_hrefs(content, epub_file_map):
        def _replace(m):
            val = m.group(1)
            low = val.lower()
            if low.startswith(('#', 'data:', 'http://', 'https://', 'mailto:', 'tel:')):
                return m.group(0)
            if not any(c in val for c in ('/', '.xhtml', '.html', '.htm')):
                return m.group(0)
            file_part = val.split('#')[0].split('?')[0]
            basename = os.path.basename(file_part)
            anchor = epub_file_map.get(basename)
            if anchor:
                return f'href="{anchor}"'
            if '#' in val:
                frag = val.split('#', 1)[1]
                if frag:
                    return f'href="#{frag}"'
            return m.group(0)
        return re.sub(r'href="([^"]*)"', _replace, content, flags=re.IGNORECASE)

    # --- Collect CSS ---
    styles = ""
    if os.path.isdir(css_dir):
        for css_file in sorted(os.listdir(css_dir)):
            if css_file.endswith('.css'):
                try:
                    with open(os.path.join(css_dir, css_file), 'r', encoding='utf-8') as f:
                        styles += f.read() + "\n"
                except Exception:
                    pass

    styles += " @page { margin: 15mm; } "
    styles += " img { max-width: 100%; height: auto; display: block; margin: 10px auto; } "
    styles += " h1, h2, h3, h4, h5, h6 { bookmark-level: none; } "
    styles += ' h1.pdf-bm { bookmark-level: 1; font-size: 0 !important; line-height: 0 !important; margin: 0 !important; padding: 0 !important; height: 0 !important; overflow: hidden !important; } '

    alignment = settings['page_number_alignment']
    page_position = f'@bottom-{alignment}' if alignment != 'center' else '@bottom-center'
    if settings['page_numbers']:
        styles += f" @page {{ {page_position} {{ content: counter(page); color: rgba(0,0,0,0.4); font-size: 10pt; }} }} "

    _dedup_enabled = os.environ.get('DEDUPLICATE_TOC', '0') == '1'
    _dedup_use_translated = os.environ.get('DEDUPLICATE_TOC_USE_TRANSLATED', '0') == '1'
    _seen_bm_titles = set()

    styles = _embed_images_in_html(styles)

    # --- Process chapters ---
    documents = []
    chapter_page_map = {}
    current_page = 0

    # Create mapping from source filename to chapter number
    source_to_chapter = {}
    for chap_num, (title, conf, source) in chapter_titles_info.items():
        if source:
            source_to_chapter[source] = chap_num

    # --- Cover page ---
    if cover_file:
        try:
            cover_path = None
            for fname in os.listdir(images_dir):
                fpath = os.path.join(images_dir, fname)
                if os.path.isfile(fpath):
                    if fname == cover_file or os.path.splitext(fname)[0] == os.path.splitext(cover_file)[0]:
                        cover_path = fpath
                        break
            if cover_path:
                ctype, _ = mimetypes.guess_type(cover_path)
                if not ctype:
                    ctype = _mime_fallback.get(os.path.splitext(cover_path)[1].lower())
                if ctype and ctype.startswith('image/'):
                    cover_bytes, cover_ctype = _read_image_as_pdf_compatible(cover_path, ctype)
                    b64 = base64.b64encode(cover_bytes).decode('utf-8')
                    cover_data_uri = f"data:{cover_ctype};base64,{b64}"
                    cover_html = (
                        f'<html><head><style>'
                        f'@page {{ margin: 0; }}'
                        f'@page {{ @bottom-left {{ content: none; }} @bottom-center {{ content: none; }} @bottom-right {{ content: none; }} }}'
                        f'body {{ margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }}'
                        f'img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}'
                        f'</style></head><body>'
                        f'<img src="{cover_data_uri}" alt="Cover" />'
                        f'</body></html>'
                    )
                    cover_doc = WeasyHTML(string=cover_html, base_url=output_dir).render()
                    documents.append(cover_doc)
                    current_page += len(cover_doc.pages)
                    log("  ✅ Added cover page to PDF")
        except Exception as e:
            log(f"  ⚠️ Failed to add cover to PDF: {e}")

    # --- TOC dry run ---
    toc_page_count = 0
    toc_insert_index = len(documents)
    if settings['toc']:
        log("  Calculating TOC size...")
        dummy_toc_html = _build_pdf_toc_html(chapter_titles_info, settings, None)
        toc_css = "<style>@page { @bottom-left { content: none; } @bottom-center { content: none; } @bottom-right { content: none; } }</style>"
        dummy_toc_html = dummy_toc_html.replace("</head>", f"{toc_css}</head>")
        try:
            _toc_stop = threading.Event()
            _toc_start = time.time()
            def _toc_heartbeat():
                while not _toc_stop.is_set():
                    if _toc_stop.wait(3.0):
                        break
                    elapsed = time.time() - _toc_start
                    log(f"  ⏳ Calculating TOC... ({elapsed:.0f}s elapsed)")
            _toc_hb = threading.Thread(target=_toc_heartbeat, daemon=True)
            _toc_hb.start()
            toc_doc = WeasyHTML(string=dummy_toc_html, base_url=output_dir).render()
            _toc_stop.set()
            _toc_hb.join(timeout=1)
            toc_page_count = len(toc_doc.pages)
        except Exception as e:
            log(f"  ⚠️ TOC size estimation failed: {e}")
            toc_page_count = 1
        current_page += toc_page_count
        _toc_elapsed = time.time() - _toc_start
        log(f"  Estimated TOC: {toc_page_count} pages ({_toc_elapsed:.1f}s)")

    # --- Build EPUB href map ---
    epub_file_map = {}
    _epub_path = os.environ.get('EPUB_PATH', '')
    if _epub_path and os.path.exists(_epub_path):
        try:
            import zipfile as _zf_mod
            with _zf_mod.ZipFile(_epub_path) as _zf:
                _container = _zf.read('META-INF/container.xml').decode('utf-8', errors='replace')
                _opf_m = re.search(r'full-path="([^"]+)"', _container)
                if _opf_m:
                    _opf_txt = _zf.read(_opf_m.group(1)).decode('utf-8', errors='replace')
                    _manifest = {m.group(1): m.group(2)
                                 for m in re.finditer(r'<item\b[^>]*\bid="([^"]+)"[^>]*\bhref="([^"]+)"', _opf_txt)}
                    _spine_hrefs = []
                    for _iid in re.findall(r'<itemref\b[^>]*\bidref="([^"]+)"', _opf_txt):
                        _href = _manifest.get(_iid, '')
                        if not _href.lower().endswith(('.xhtml', '.html', '.htm')):
                            continue
                        _pm = re.search(rf'<item\b[^>]*\bid="{re.escape(_iid)}"[^>]*\bproperties="([^"]*)"', _opf_txt)
                        if _pm and 'nav' in _pm.group(1):
                            continue
                        _spine_hrefs.append(_href)
                    for _si, _sh in enumerate(_spine_hrefs):
                        if _si < len(html_files):
                            _hf = html_files[_si]
                            _cn = source_to_chapter.get(_hf, _si)
                            epub_file_map[os.path.basename(_sh)] = f'#chapter-{_cn}'
        except Exception as _em:
            log(f"  ⚠️ Could not build EPUB href map: {_em}")

    # --- Build combined chapter HTML ---
    log(f"  Building combined chapter document ({len(html_files)} chapters)...")
    all_chapters_html = ""
    chapters_order = []

    for i, html_file in enumerate(html_files):
        file_path = os.path.join(output_dir, html_file)
        if not os.path.exists(file_path):
            continue

        chap_num = source_to_chapter.get(html_file, i)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            content = _embed_images_in_html(content)
            body_content = _extract_body_content(content)
            body_content = _rewrite_epub_hrefs(body_content, epub_file_map)

            _bm_title = chapter_titles_info.get(chap_num, ('', 0, ''))[0]
            if _bm_title and _bm_title.strip().lower() in ('untitled chapter', 'untitled'):
                _bm_title = ''
            if _bm_title and _dedup_enabled:
                _bm_key = _bm_title if _dedup_use_translated else chapter_titles_info.get(chap_num, ('', 0, ''))[0]
                if _bm_key in _seen_bm_titles:
                    _bm_title = ''
                else:
                    _seen_bm_titles.add(_bm_key)
            _bm_h1 = (f'<h1 class="pdf-bm">{html_module.escape(str(_bm_title))}</h1>'
                       if _bm_title else '')
            if i > 0:
                body_content = f'<div style="page-break-before: always;" id="chapter-{chap_num}">{_bm_h1}{body_content}</div>'
            else:
                body_content = f'<div id="chapter-{chap_num}">{_bm_h1}{body_content}</div>'

            all_chapters_html += body_content
            chapters_order.append((html_file, chap_num))

            if (i + 1) % 10 == 0 or (i + 1) == len(html_files):
                log(f"  [{i+1}/{len(html_files)}] Added to document")

        except Exception as e:
            log(f"  ⚠️ Failed to process {html_file}: {e}")

    # --- Render ---
    if all_chapters_html:
        log("  Rendering combined chapter document...")
        combined_html = f"<html><head><style>{styles}</style></head><body>{all_chapters_html}</body></html>"
        try:
            _render_stop = threading.Event()
            _render_start = time.time()
            def _render_heartbeat():
                while not _render_stop.is_set():
                    if _render_stop.wait(3.0):
                        break
                    elapsed = time.time() - _render_start
                    log(f"  ⏳ Rendering... ({elapsed:.0f}s elapsed)")
            _render_hb = threading.Thread(target=_render_heartbeat, daemon=True)
            _render_hb.start()
            chapters_doc = WeasyHTML(string=combined_html, base_url=output_dir).render()
            _render_stop.set()
            _render_hb.join(timeout=1)
            _render_elapsed = time.time() - _render_start
            log(f"  ✅ Rendering complete ({_render_elapsed:.1f}s)")
            documents.append(chapters_doc)

            # Resolve page numbers
            _anchor_to_page = {}
            for _pidx, _pg in enumerate(chapters_doc.pages):
                for _aid in _pg.anchors:
                    if _aid not in _anchor_to_page:
                        _anchor_to_page[_aid] = _pidx + 1
            _fallback_page = 1
            for _html_file, _chap_num in chapters_order:
                _pg_num = _anchor_to_page.get(f'chapter-{_chap_num}')
                if _pg_num is not None:
                    chapter_page_map[_html_file] = _pg_num
                    chapter_page_map[_chap_num] = _pg_num
                    _fallback_page = _pg_num + 1
                else:
                    chapter_page_map[_html_file] = _fallback_page
                    chapter_page_map[_chap_num] = _fallback_page
                    _fallback_page += 1

            current_page += len(chapters_doc.pages)
            log(f"  Combined document: {len(chapters_doc.pages)} pages")
        except Exception as e:
            log(f"  ⚠️ Failed to render combined document: {e}")

    if not documents:
        log("⚠️ No chapters rendered for PDF")
        print(f'[RESULT] {json.dumps({"success": False, "error": "No chapters rendered"})}', flush=True)
        return

    # --- Final TOC ---
    if settings['toc'] and chapter_titles_info:
        log("  Generating TOC with page numbers...")
        real_toc_html = _build_pdf_toc_html(chapter_titles_info, settings, chapter_page_map)
        toc_css = "<style>@page { @bottom-left { content: none; } @bottom-center { content: none; } @bottom-right { content: none; } }</style>"
        real_toc_html = real_toc_html.replace("</head>", f"{toc_css}</head>")
        try:
            toc_doc = WeasyHTML(string=real_toc_html, base_url=output_dir).render()
            documents.insert(toc_insert_index, toc_doc)
        except Exception as e:
            log(f"  ⚠️ TOC generation failed: {e}")

    # --- Merge and write ---
    log("  Merging all pages...")
    all_pages = [page for doc in documents for page in doc.pages]
    log(f"  Total pages: {len(all_pages)}")
    log("  Writing PDF to disk...")
    try:
        _pdf_write_kwargs = {}
        if compression_enabled:
            _pdf_quality = int(os.environ.get('IMAGE_COMPRESSION_QUALITY', '80'))
            _pdf_write_kwargs['image_quality'] = _pdf_quality
            log(f"  PDF image quality: {_pdf_quality}%")
        else:
            _pdf_write_kwargs['uncompressed_pdf'] = True
            log("  PDF images: uncompressed (preserving original quality)")
        _write_stop = threading.Event()
        _write_start = time.time()
        def _write_heartbeat():
            while not _write_stop.is_set():
                if _write_stop.wait(3.0):
                    break
                elapsed = time.time() - _write_start
                log(f"  ⏳ Writing PDF... ({elapsed:.0f}s elapsed)")
        _write_hb = threading.Thread(target=_write_heartbeat, daemon=True)
        _write_hb.start()
        documents[0].copy(all_pages).write_pdf(pdf_path, **_pdf_write_kwargs)
        _write_stop.set()
        _write_hb.join(timeout=1)
        _write_elapsed = time.time() - _write_start
        log(f"  ✅ PDF written ({_write_elapsed:.1f}s)")
    except BaseException as e:
        log(f"  ❌ write_pdf failed: {type(e).__name__}: {e}")
        log(f"  [DEBUG] {traceback.format_exc()}")
        print(f'[RESULT] {json.dumps({"success": False, "error": str(e)})}', flush=True)
        return

    elapsed = time.time() - start_time
    if os.path.exists(pdf_path):
        file_size = os.path.getsize(pdf_path)
        log(f"✅ PDF created: {pdf_path}")
        log(f"📊 PDF size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        log(f"⏱️ PDF generation took {elapsed:.1f}s")
        print(f'[RESULT] {json.dumps({"success": True, "pdf_path": pdf_path, "file_size": file_size, "elapsed": elapsed})}', flush=True)
    else:
        log("❌ PDF file was not created")
        print(f'[RESULT] {json.dumps({"success": False, "error": "PDF file was not created"})}', flush=True)


def main():
    """Main entry point for worker process."""
    if len(sys.argv) < 2:
        print("[ERROR] Usage: _pdf_worker.py <config_json_path>", flush=True)
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}", flush=True)
        sys.exit(1)

    try:
        run_pdf_generation(config_path)
    except Exception as e:
        print(f"[ERROR] PDF generation failed: {e}", flush=True)
        print(f'[RESULT] {json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()})}', flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
