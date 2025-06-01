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
    """Modified to accept a logging callback"""
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    try:
        OUTPUT_DIR = base_dir
        IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
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
        for ext in ("jpg", "jpeg", "png"):  
            candidate = f"cover.{ext}"
            safe_candidate = sanitize_filename(candidate)
            if safe_candidate in image_files or candidate in processed_images:
                cover_file = processed_images.get(candidate, safe_candidate)
                break
        
        if not cover_file and image_files:
            cover_file = image_files[0]

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
                    book.set_cover("cover.jpg", cover_data)
                    
                    # Also add cover page for readers that need it
                    cover_page = epub.EpubHtml(
                        title="Cover",
                        file_name="cover.xhtml",
                        lang=meta.get("language", "en")
                    )
                    cover_page.content = (
                        '<!DOCTYPE html>'
                        '<html xmlns="http://www.w3.org/1999/xhtml">'
                        '<head><title>Cover</title></head>'
                        '<body style="text-align:center;padding:0;margin:0;">'
                        f'<img src="images/{cover_file}" alt="Cover" style="max-width:100%;height:auto;"/>'
                        '</body></html>'
                    )
                    book.add_item(cover_page)
                    
                    # Add to spine and toc ONLY ONCE at the beginning
                    spine.insert(0, cover_page)
                    toc.insert(0, cover_page)
                    chapters_added += 1
                    
                    log(f"✅ Set cover image: {cover_file}")
                    
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
                
                # Fix image paths with sanitized names
                for img in soup.find_all('img'):
                    src = img.get('src', '')
                    basename = os.path.basename(src.split('?')[0])
                    # Use the sanitized version if we have it
                    safe_name = processed_images.get(basename, sanitize_filename(basename))
                    img['src'] = f"images/{safe_name}"

                # Use metadata title mapping or default
                title = meta.get("titles", {}).get(str(num), f"Chapter {num}")
                chap = epub.EpubHtml(
                    title=title,
                    file_name=fn,
                    lang=meta.get("language", "en")
                )
                chap.content = str(soup)
                book.add_item(chap)
                spine.append(chap)
                toc.append(chap)
                chapters_added += 1
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
            html_parts = ["<html><body><h1>Image Gallery</h1>"]
            for img in gallery_images:
                html_parts.append(
                    "<div style='text-align:center;margin:20px;'>"
                    f"<img src='images/{img}' alt='{img}'/>"
                    "</div>"
                )
            html_parts.append("</body></html>")
            gallery_page.content = "".join(html_parts)
            book.add_item(gallery_page)
            spine.append(gallery_page)
            toc.append(gallery_page)

        # Finalize TOC and spine
        book.toc = toc
        book.spine = ['nav'] + spine

        # Add navigation files
        book.add_item(epub.EpubNav())
        book.add_item(epub.EpubNcx())

        # Write out final EPUB with error handling
        base_name = os.path.basename(OUTPUT_DIR)
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}.epub")
        try:
            log(f"[DEBUG] Writing EPUB to: {out_path}")
            epub.write_epub(out_path, book)
            log(f"✅ EPUB created at: {out_path}")
        except Exception as e:
            log(f"❌ Failed to write EPUB: {e}")
            raise

    except Exception as e:
        log(f"❌ EPUB compilation failed with error: {e}")
        raise
