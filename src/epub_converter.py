import os
import sys
import io
import json
import mimetypes
import re
from ebooklib import epub
from bs4 import BeautifulSoup
import unicodedata

try:
    # Python 3.7+ lets us just reconfigure the existing stream
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
except AttributeError:
    if sys.stdout is None:
        devnull = open(os.devnull, "wb")
        sys.stdout = io.TextIOWrapper(devnull, encoding='utf-8', errors='ignore')
    elif hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
        except:
            pass

def sanitize_filename(filename):
    """
    Sanitize filename to be safe for EPUB and filesystem.
    Removes or replaces problematic characters.
    """
    # Normalize unicode characters
    filename = unicodedata.normalize('NFKD', filename)
    
    # Replace problematic characters with safe alternatives
    replacements = {
        '/': '_',
        '\\': '_',
        ':': '_',
        '*': '_',
        '?': '_',
        '"': '_',
        '<': '_',
        '>': '_',
        '|': '_',
        '\n': '_',
        '\r': '_',
        '\t': '_'
    }
    
    for old, new in replacements.items():
        filename = filename.replace(old, new)
    
    # Remove any remaining control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Limit length to avoid filesystem issues
    name, ext = os.path.splitext(filename)
    if len(name) > 200:
        name = name[:200]
    
    return name + ext

def verify_css_in_chapter(chapter_html, css_files):
    """Verify that CSS links are present in chapter HTML"""
    soup = BeautifulSoup(chapter_html, 'html.parser')
    found_css = []
    
    for link in soup.find_all('link', rel='stylesheet'):
        href = link.get('href', '')
        for css_file in css_files:
            if css_file in href:
                found_css.append(css_file)
    
    return found_css

def rewrite_img_paths(html):
    soup = BeautifulSoup(html, 'html.parser')
    for img in soup.find_all('img'):
        src = img.get('src', '')
        if src.startswith("http"):
            filename = os.path.basename(src.split("?")[0])
            img['src'] = f"images/{sanitize_filename(filename)}"
        elif os.path.isfile(src):
            img['src'] = f"images/{sanitize_filename(os.path.basename(src))}"
    return str(soup)

def fallback_compile_epub(base_dir, log_callback=None):
    """Modified to include CSS and maintain structure"""
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    try:
        OUTPUT_DIR = base_dir
        IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
        CSS_DIR = os.path.join(OUTPUT_DIR, "css")
        FONTS_DIR = os.path.join(OUTPUT_DIR, "fonts")
        METADATA = os.path.join(OUTPUT_DIR, "metadata.json")

        # Debug: Check what files exist
        log(f"[DEBUG] Scanning directory: {OUTPUT_DIR}")
        all_files = os.listdir(OUTPUT_DIR)
        html_files = [f for f in all_files if f.startswith("response_") and f.endswith(".html")]
        log(f"[DEBUG] Found {len(html_files)} translated HTML files")
        
        if not html_files:
            log("❌ No translated HTML files found (files starting with 'response_' and ending with '.html')")
            log(f"[DEBUG] Files in directory: {[f for f in all_files if f.endswith('.html')][:5]}...")
            raise Exception("No translated chapters found to compile into EPUB")

        # Load metadata
        if os.path.exists(METADATA):
            try:
                with open(METADATA, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                log(f"[WARNING] Failed to load metadata.json: {e}")
                meta = {}
        else:
            log("[WARNING] metadata.json not found, using defaults.")
            meta = {}

        # Initialize book with proper error handling
        book = epub.EpubBook()
        
        # Set book metadata with fallbacks
        book.set_identifier(meta.get("identifier", f"translated-{os.path.basename(base_dir)}"))
        book.set_title(meta.get("title", os.path.basename(base_dir)))
        book.set_language(meta.get("language", "en"))
        
        if meta.get("creator"):
            book.add_author(meta["creator"])

        spine = []
        toc = []
        processed_images = {}  # Track processed images to avoid duplicates
        chapters_added = 0  # Track how many chapters we actually add

        # Add CSS files to EPUB
        css_items = []
        if os.path.isdir(CSS_DIR):
            css_files = [f for f in sorted(os.listdir(CSS_DIR)) if f.endswith('.css')]
            log(f"[DEBUG] Found {len(css_files)} CSS files to add")
            
            for css_file in css_files:
                css_path = os.path.join(CSS_DIR, css_file)
                try:
                    with open(css_path, 'r', encoding='utf-8') as f:
                        css_content = f.read()
                    
                    # Log CSS file size for debugging
                    log(f"[DEBUG] Reading CSS: {css_file} ({len(css_content)} bytes)")
                    
                    css_item = epub.EpubItem(
                        uid=f"css_{css_file}",
                        file_name=f"css/{css_file}",
                        media_type="text/css",
                        content=css_content
                    )
                    book.add_item(css_item)
                    css_items.append(css_item)
                    log(f"✅ Added CSS: {css_file}")
                except Exception as e:
                    log(f"[WARNING] Failed to add CSS {css_file}: {e}")
        
        # Add fonts if they exist
        if os.path.isdir(FONTS_DIR):
            for font_file in os.listdir(FONTS_DIR):
                font_path = os.path.join(FONTS_DIR, font_file)
                if os.path.isfile(font_path):
                    try:
                        mime_type = 'application/font-woff'
                        if font_file.endswith('.ttf'):
                            mime_type = 'application/x-font-ttf'
                        elif font_file.endswith('.otf'):
                            mime_type = 'application/x-font-opentype'
                        
                        with open(font_path, 'rb') as f:
                            book.add_item(epub.EpubItem(
                                uid=f"font_{font_file}",
                                file_name=f"fonts/{font_file}",
                                media_type=mime_type,
                                content=f.read()
                            ))
                        log(f"✅ Added font: {font_file}")
                    except Exception as e:
                        log(f"[WARNING] Failed to add font {font_file}: {e}")

        # Embed images with better error handling
        image_files = []
        if os.path.isdir(IMAGES_DIR):
            try:
                for img in sorted(os.listdir(IMAGES_DIR)):
                    path = os.path.join(IMAGES_DIR, img)
                    if os.path.isfile(path):
                        ctype, _ = mimetypes.guess_type(path)
                        if ctype and ctype.startswith("image"):
                            # Sanitize the image filename
                            safe_name = sanitize_filename(img)
                            image_files.append(safe_name)
                            processed_images[img] = safe_name
                            log(f"[DEBUG] Found image: {img} -> {safe_name}")
            except OSError as e:
                log(f"[WARNING] Error reading images directory: {e}")

        # Determine cover file
        cover_file = None
        cover_patterns = ['cover', 'Cover', 'COVER', 'front', 'Front']
        
        # First, look for files with common cover names
        for pattern in cover_patterns:
            for ext in ["jpg", "jpeg", "png"]:
                candidate = f"{pattern}.{ext}"
                safe_candidate = sanitize_filename(candidate)
                
                # Check if this file exists in processed images
                if safe_candidate in image_files or candidate in processed_images:
                    cover_file = processed_images.get(candidate, safe_candidate)
                    log(f"[DEBUG] Found cover by name match: {cover_file}")
                    break
            if cover_file:
                break
        
        # If no cover found by name, use the first image
        if not cover_file and image_files:
            cover_file = image_files[0]
            log(f"[DEBUG] Using first image as cover: {cover_file}")

        # Embed non-cover images
        for original_name, safe_name in processed_images.items():
            if safe_name == cover_file:
                continue
            
            img_path = os.path.join(IMAGES_DIR, original_name)
            try:
                ctype, _ = mimetypes.guess_type(img_path)
                with open(img_path, 'rb') as fp:
                    data = fp.read()
                
                book.add_item(epub.EpubItem(
                    uid=safe_name,
                    file_name=f"images/{safe_name}",
                    media_type=ctype or "image/jpeg",
                    content=data
                ))
            except (IOError, OSError) as e:
                log(f"[WARNING] Failed to embed image {original_name}: {e}")

        # Embed cover image and page
        if cover_file:
            # Find the original filename for the cover
            original_cover = None
            for orig, safe in processed_images.items():
                if safe == cover_file:
                    original_cover = orig
                    break
            
            if original_cover:
                cover_path = os.path.join(IMAGES_DIR, original_cover)
                try:
                    with open(cover_path, 'rb') as fp:
                        cover_data = fp.read()
                    
                    # Create the cover image item
                    cover_img = epub.EpubItem(
                        uid="cover-image",
                        file_name=f"images/{cover_file}",
                        media_type=mimetypes.guess_type(cover_path)[0] or "image/jpeg",
                        content=cover_data
                    )
                    book.add_item(cover_img)
                    
                    # IMPORTANT: Set this image as the book's cover
                    # This adds proper metadata so readers recognize it
                    book.set_cover("cover.jpg", cover_data)
                    
                    # Also create a cover page for readers that need it
                    
                    # Also add cover page for readers that need it
                    cover_page = epub.EpubHtml(
                        title="Cover",
                        file_name="cover.xhtml",
                        lang=meta.get("language", "en")
                    )
                    
                    # Build cover page HTML with CSS links
                    cover_html_parts = [
                        '<!DOCTYPE html>',
                        '<html xmlns="http://www.w3.org/1999/xhtml">',
                        '<head>',
                        '<title>Cover</title>'
                    ]
                    
                    # Add CSS links to cover page
                    if css_items:
                        for css_item in css_items:
                            css_filename = css_item.file_name.split('/')[-1]
                            cover_html_parts.append(f'<link rel="stylesheet" type="text/css" href="css/{css_filename}"/>')
                    
                    cover_html_parts.extend([
                        '</head>',
                        '<body style="text-align:center;padding:0;margin:0;">',
                        f'<img src="images/{cover_file}" alt="Cover" style="max-width:100%;height:auto;"/>',
                        '</body></html>'
                    ])
                    
                    cover_page.content = '\n'.join(cover_html_parts)
                    book.add_item(cover_page)
                    
                    # Add to spine at the beginning
                    spine.insert(0, cover_page)
                    # Note: Not adding cover to TOC as most readers handle it separately
                    chapters_added += 1
                    
                    log(f"✅ Set cover image: {cover_file}")
                    log("   • Cover will appear before Table of Contents")
                    log("   • Cover is not listed in Table of Contents (standard practice)")
                    
                except (IOError, OSError) as e:
                    log(f"[WARNING] Failed to add cover image: {e}")

        # Collect translated HTML files and parse chapter numbers
        chapter_tuples = []  # list of (chapter_number, filename)
        chapter_seen = set()  # Track seen chapter numbers to avoid true duplicates
        
        try:
            for fn in sorted(html_files):  # Use the html_files we already found
                log(f"[DEBUG] Processing file: {fn}")
                m = re.match(r"response_(\d+)_", fn)
                if m:
                    num = int(m.group(1))
                    # Only add if we haven't seen this chapter number
                    if num not in chapter_seen:
                        chapter_tuples.append((num, fn))
                        chapter_seen.add(num)
                    else:
                        log(f"[WARNING] Skipping duplicate chapter {num}: {fn}")
                else:
                    log(f"[WARNING] File doesn't match expected pattern: {fn}")
        except OSError as e:
            log(f"[ERROR] Failed to read output directory: {e}")
            raise

        # Sort chapters by actual number (handles missing chapters correctly)
        chapter_tuples.sort(key=lambda x: x[0])
        log(f"[DEBUG] Found {len(chapter_tuples)} unique chapters to process")

        # Add chapters to book
        for num, fn in chapter_tuples:
            path = os.path.join(OUTPUT_DIR, fn)
            try:
                log(f"[DEBUG] Reading chapter {num} from {fn}")
                with open(path, 'r', encoding='utf-8') as f:
                    raw = f.read()
                
                if not raw.strip():
                    log(f"[WARNING] Chapter {num} is empty, skipping")
                    continue
                
                soup = BeautifulSoup(raw, 'html.parser')
                
                # Add CSS links to chapter if not present
                if css_items:
                    # Check if chapter has a head tag, create one if not
                    if not soup.head:
                        head_tag = soup.new_tag('head')
                        # Add a placeholder title tag
                        title_tag = soup.new_tag('title')
                        title_tag.string = f"Chapter {num}"  # Use chapter number as placeholder
                        head_tag.append(title_tag)
                        
                        if soup.html:
                            soup.html.insert(0, head_tag)
                        else:
                            # Create html tag if missing
                            html_tag = soup.new_tag('html')
                            html_tag.append(head_tag)
                            if soup.body:
                                html_tag.append(soup.body.extract())
                            else:
                                body_tag = soup.new_tag('body')
                                # Move all content to body
                                for child in list(soup.children):
                                    body_tag.append(child.extract())
                                html_tag.append(body_tag)
                            soup.append(html_tag)
                    
                    # Add CSS links - check for existing ones first
                    existing_css_hrefs = set()
                    for existing_link in soup.find_all('link', rel='stylesheet'):
                        href = existing_link.get('href', '')
                        existing_css_hrefs.add(os.path.basename(href))
                    
                    # Add only missing CSS links
                    for css_item in css_items:
                        css_filename = css_item.file_name.split('/')[-1]
                        if css_filename not in existing_css_hrefs:
                            link_tag = soup.new_tag('link')
                            link_tag['rel'] = 'stylesheet'
                            link_tag['type'] = 'text/css'
                            link_tag['href'] = f"css/{css_filename}"  # CSS files are in css/ subdirectory
                            soup.head.append(link_tag)
                
                # Fix image paths with sanitized names
                for img in soup.find_all('img'):
                    src = img.get('src', '')
                    basename = os.path.basename(src.split('?')[0])
                    # Use the sanitized version if we have it
                    safe_name = processed_images.get(basename, sanitize_filename(basename))
                    img['src'] = f"images/{safe_name}"

                # Use metadata title mapping or default
                title = meta.get("titles", {}).get(str(num), f"Chapter {num}")
                
                # Update the title tag if it exists
                if soup.head and soup.head.title:
                    soup.head.title.string = title
                elif soup.head and not soup.head.title:
                    # Add title tag if head exists but no title
                    title_tag = soup.new_tag('title')
                    title_tag.string = title
                    soup.head.insert(0, title_tag)
                
                chap = epub.EpubHtml(
                    title=title,
                    file_name=fn,
                    lang=meta.get("language", "en")
                )
                chap.content = str(soup)
                
                # Link CSS files to this chapter (don't use add_item)
                # The CSS files are already added to the book, we just need to reference them
                
                book.add_item(chap)
                spine.append(chap)
                toc.append(chap)
                chapters_added += 1
                
                # Verify CSS links were added
                if css_items:
                    css_files = [item.file_name.split('/')[-1] for item in css_items]
                    found_css = verify_css_in_chapter(str(soup), css_files)
                    if found_css:
                        log(f"✅ Added chapter {num}: {title} (CSS: {', '.join(found_css)})")
                    else:
                        log(f"⚠️ Added chapter {num}: {title} (WARNING: No CSS links found!)")
                else:
                    log(f"✅ Added chapter {num}: {title}")
                
            except (IOError, OSError) as e:
                log(f"[WARNING] Failed to add chapter {num} from {fn}: {e}")
                continue

        # Check if we have any content
        if chapters_added == 0:
            log("❌ No chapters were successfully added to the EPUB")
            raise Exception("No chapters could be added to the EPUB - all files were empty or unreadable")

        log(f"[DEBUG] Total chapters added: {chapters_added}")

        # Optional gallery for extra images
        gallery_images = [img for img in image_files if img != cover_file]
        if gallery_images:
            gallery_page = epub.EpubHtml(
                title="Gallery",
                file_name="gallery.xhtml",
                lang=meta.get("language", "en")
            )
            html_parts = [
                '<!DOCTYPE html>',
                '<html xmlns="http://www.w3.org/1999/xhtml">',
                '<head><title>Gallery</title>'
            ]
            
            # Add CSS links to gallery
            if css_items:
                for css_item in css_items:
                    css_filename = css_item.file_name.split('/')[-1]
                    html_parts.append(f'<link rel="stylesheet" type="text/css" href="css/{css_filename}"/>')
            
            html_parts.extend([
                '</head>',
                '<body>',
                '<h1>Image Gallery</h1>'
            ])
            
            for img in gallery_images:
                html_parts.append(
                    f'<div style="text-align:center;margin:20px;">'
                    f'<img src="images/{img}" alt="{img}"/>'
                    f'</div>'
                )
            html_parts.extend(['</body>', '</html>'])
            
            gallery_page.content = '\n'.join(html_parts)
            
            book.add_item(gallery_page)
            spine.append(gallery_page)
            toc.append(gallery_page)

        # Finalize TOC and spine
        book.toc = toc
        
        # Set spine with cover first (if it exists), then nav, then chapters
        if spine and spine[0].title == "Cover":
            # Cover is first in spine, so put it before nav
            book.spine = [spine[0], 'nav'] + spine[1:]
            log("📖 Reading order: Cover → Table of Contents → Chapters")
            log("   Note: Some readers may still show ToC first based on their settings")
        else:
            # No cover, nav comes first
            book.spine = ['nav'] + spine
            log("📖 Reading order: Table of Contents → Chapters")
        
        log("\n📱 Cover Display by Reader:")
        log("   • Calibre: Shows cover → ToC → Chapters")
        log("   • Kindle: Usually shows cover first")
        log("   • Apple Books: Shows cover in library, may start at ToC")
        log("   • Adobe Digital Editions: Shows cover → ToC → Chapters")

        # Add navigation files
        book.add_item(epub.EpubNav())
        book.add_item(epub.EpubNcx())
        
        # Add guide for cover (helps some readers)
        if spine and spine[0].title == "Cover":
            book.guide = [
                {"type": "cover", "title": "Cover", "href": spine[0].file_name}
            ]

        # Write out final EPUB with error handling
        base_name = os.path.basename(OUTPUT_DIR)
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}.epub")
        try:
            log(f"[DEBUG] Writing EPUB to: {out_path}")
            log(f"[DEBUG] Total CSS files included: {len(css_items)}")
            epub.write_epub(out_path, book)
            log(f"✅ EPUB created at: {out_path}")
            
            # Verify EPUB contents
            if css_items:
                log(f"✅ Successfully embedded {len(css_items)} CSS files")
                for css_item in css_items:
                    log(f"   • {css_item.file_name}")
                log("\n💡 To verify CSS is working:")
                log("   1. Open the EPUB in a reader like Calibre")
                log("   2. Right-click and 'View' or 'Inspect' a chapter")
                log("   3. Check the <head> section for CSS links")
                log("   4. Verify styles are applied to the content")
                log("\n⚠️  Note: Some EPUB readers may ignore or override CSS")
                log("   - Kindle devices often ignore most CSS")
                log("   - Some readers only support basic CSS")
                log("   - Try Calibre or Adobe Digital Editions for best CSS support")
        except Exception as e:
            log(f"❌ Failed to write EPUB: {e}")
            raise

    except Exception as e:
        log(f"❌ EPUB compilation failed with error: {e}")
        raise
