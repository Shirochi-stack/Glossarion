import os
import sys
import io
import json
import mimetypes
import re
from ebooklib import epub
from bs4 import BeautifulSoup

try:
    # Python 3.7+ lets us just reconfigure the existing stream
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
except AttributeError:
    if sys.stdout is None:
        devnull = open(os.devnull, "wb")
        sys.stdout = io.TextIOWrapper(devnull, encoding='utf-8', errors='ignore')
    elif hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

# rewrite_img_paths retains image path logic if needed elsewhere

def rewrite_img_paths(html):
    soup = BeautifulSoup(html, 'html.parser')
    for img in soup.find_all('img'):
        src = img.get('src', '')
        if src.startswith("http"):
            filename = os.path.basename(src.split("?")[0])
            img['src'] = f"images/{filename}"
        elif os.path.isfile(src):
            img['src'] = f"images/{os.path.basename(src)}"
    return str(soup)


def fallback_compile_epub(base_dir):
    """
    Compile an EPUB using translated HTML files in base_dir,
    preserving actual chapter numbers from filenames.
    """
    OUTPUT_DIR = base_dir
    IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
    METADATA = os.path.join(OUTPUT_DIR, "metadata.json")

    # Load metadata
    if os.path.exists(METADATA):
        with open(METADATA, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    else:
        print("[FALLBACK] Warning: metadata.json not found, using defaults.")
        meta = {}

    # Initialize book
    book = epub.EpubBook()
    book.set_identifier(meta.get("identifier", "translated-fallback"))
    book.set_title(meta.get("title", "Translated Fallback"))
    book.set_language(meta.get("language", "en"))
    if meta.get("creator"):
        book.add_author(meta["creator"])

    spine = []
    toc = []

    # Embed images
    image_files = []
    if os.path.isdir(IMAGES_DIR):
        for img in sorted(os.listdir(IMAGES_DIR)):
            path = os.path.join(IMAGES_DIR, img)
            ctype, _ = mimetypes.guess_type(path)
            if ctype and ctype.startswith("image"):
                image_files.append(img)

    # Determine cover file
    cover_file = None
    for ext in ("jpg", "jpeg", "png"):  
        candidate = f"cover.{ext}"
        if candidate in image_files:
            cover_file = candidate
            break
    if not cover_file and image_files:
        cover_file = image_files[0]

    # Embed non-cover images
    for img in image_files:
        if img == cover_file:
            continue
        img_path = os.path.join(IMAGES_DIR, img)
        ctype, _ = mimetypes.guess_type(img_path)
        with open(img_path, 'rb') as fp:
            data = fp.read()
        book.add_item(epub.EpubItem(
            uid=img,
            file_name=f"images/{img}",
            media_type=ctype or "image/jpeg",
            content=data
        ))

    # Embed cover image and page
    if cover_file:
        cover_path = os.path.join(IMAGES_DIR, cover_file)
        with open(cover_path, 'rb') as fp:
            cover_data = fp.read()
        book.add_item(epub.EpubItem(
            uid="cover-image",
            file_name=f"images/{cover_file}",
            media_type=mimetypes.guess_type(cover_path)[0] or "image/jpeg",
            content=cover_data
        ))
        cover_page = epub.EpubHtml(
            title="Cover",
            file_name="cover.xhtml",
            lang=meta.get("language", "en")
        )
        cover_page.content = (
            "<html><body style='text-align:center;padding-top:50px;'>"
            f"<img src='images/{cover_file}' alt='Cover'/>"
            "</body></html>"
        )
        book.add_item(cover_page)
        spine.append(cover_page)
        toc.append(cover_page)

    # Collect translated HTML files and parse chapter numbers
    chapter_tuples = []  # list of (chapter_number, filename)
    for fn in os.listdir(OUTPUT_DIR):
        if fn.startswith("response_") and fn.endswith(".html"):
            m = re.match(r"response_(\d+)_", fn)
            if m:
                num = int(m.group(1))
                chapter_tuples.append((num, fn))

    # Sort chapters by actual number (handles missing chapters correctly)
    chapter_tuples.sort(key=lambda x: x[0])

    # Add chapters to book
    for num, fn in chapter_tuples:
        path = os.path.join(OUTPUT_DIR, fn)
        raw = open(path, 'r', encoding='utf-8').read()
        soup = BeautifulSoup(raw, 'html.parser')
        for img in soup.find_all('img'):
            src = img.get('src', '')
            cleaned = os.path.basename(src.split('?')[0])
            img['src'] = f"images/{cleaned}"

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
    if cover_file:
        chapter_spine = [item for item in spine if item is not cover_page]
        book.spine = [cover_page, 'nav'] + chapter_spine
    else:
        book.spine = ['nav'] + spine

    # Add navigation files
    book.add_item(epub.EpubNav())
    book.add_item(epub.EpubNcx())

    # Write out final EPUB
    out_path = os.path.join(OUTPUT_DIR, "translated_fallback.epub")
    epub.write_epub(out_path, book)
    print(f"✅ Fallback EPUB created at: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python epub_converter.py [output_folder]")
    else:
        fallback_compile_epub(sys.argv[1])
