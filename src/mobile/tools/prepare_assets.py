#!/usr/bin/env python3
"""Generate the mobile app's build-time assets (gitignored; CI runs this in ``prepare``).

1. ``app/assets/tiktoken/``: the tiktoken BPE files for ``cl100k_base`` and
   ``o200k_base``, named exactly as ``tiktoken.load.read_file_cached`` looks them up
   (``sha1(blob_url)``), plus ``MANIFEST.toml``. Each file is checked against the
   sha256 tiktoken pins. ``runtime_bootstrap.seed_tiktoken_cache`` copies them into
   ``TIKTOKEN_CACHE_DIR`` so the first offline run works.

   Files come from an existing local tiktoken cache when one has a verified copy
   (``$TIKTOKEN_CACHE_DIR``, ``$DATA_GYM_CACHE_DIR``, ``<tmp>/data-gym-cache``).
   Otherwise they are downloaded from openaipublic (``--offline`` forbids that).

2. ``app/assets/selftest/selftest_ko_12ch.epub``: a small, byte-reproducible
   12-chapter Korean EPUB built with ebooklib, plus ``MANIFEST.toml``. It has more than
   10 chapters so ``Chapter_Extractor`` takes its worker-pool path. It includes a nav,
   an NCX, a stylesheet and one image, and recurring character/place names so a
   glossary pass has something to find. The device self-test and the host E2E tests
   use it.

Usage::

    python tools/prepare_assets.py                 # both
    python tools/prepare_assets.py --only tiktoken --offline
    python tools/prepare_assets.py --check         # verify existing assets, write nothing

Needs ``ebooklib`` (and lxml) for the EPUB; everything else is stdlib.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import os
import re
import struct
import sys
import tempfile
import urllib.request
import zipfile
import zlib
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

MOBILE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ASSETS = MOBILE_DIR / "app" / "assets"
PYPROJECT = MOBILE_DIR / "pyproject.toml"

# name -> (blob url, sha256). Same constants as tiktoken_ext/openai_public.py (0.12/0.13/0.14).
TIKTOKEN_ENCODINGS = {
    "cl100k_base": ("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
                    "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"),
    "o200k_base": ("https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
                   "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"),
}
MANIFEST_NAME = "MANIFEST.toml"
SELFTEST_EPUB = "selftest_ko_12ch.epub"
FIXED_TIME = _dt.datetime(2026, 1, 1, 0, 0, 0, tzinfo=_dt.timezone.utc)
ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cache_key(url: str) -> str:
    """File name tiktoken uses inside its cache dir (tiktoken/load.py: sha1(blobpath))."""
    return hashlib.sha1(url.encode()).hexdigest()


def _write_if_changed(path: Path, data: bytes) -> bool:
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def pinned_version(dist: str) -> str:
    if tomllib is None or not PYPROJECT.exists():
        return ""
    with open(PYPROJECT, "rb") as f:
        deps = tomllib.load(f).get("project", {}).get("dependencies", [])
    for d in deps:
        m = re.match(rf"^\s*{re.escape(dist)}\s*==\s*([^\s;,]+)", d, re.I)
        if m:
            return m.group(1)
    return ""


def installed_tiktoken_constants() -> dict:
    """URL/hash pairs from an installed tiktoken_ext (if any), to catch upstream changes."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("tiktoken_ext.openai_public")
    except (ImportError, ValueError):
        return {}
    if not spec or not spec.origin or not os.path.exists(spec.origin):
        return {}
    text = Path(spec.origin).read_text(encoding="utf-8", errors="replace")
    found = {}
    for name in TIKTOKEN_ENCODINGS:
        m = re.search(rf'"(https://[^"]+/{name}\.tiktoken)",\s*expected_hash="([0-9a-f]{{64}})"', text)
        if m:
            found[name] = (m.group(1), m.group(2))
    return found


# --------------------------------------------------------------------------- tiktoken
def _local_cache_dirs() -> list[Path]:
    dirs = []
    for env in ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"):
        if os.environ.get(env):
            dirs.append(Path(os.environ[env]))
    dirs.append(Path(tempfile.gettempdir()) / "data-gym-cache")
    return dirs


def _download(url: str, timeout: float = 120.0) -> bytes:
    last = None
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "glossarion-prepare-assets/1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except OSError as e:  # URLError is an OSError
            last = e
    raise RuntimeError(f"download failed for {url}: {last}")


def prepare_tiktoken(out_dir: Path, offline: bool = False, check: bool = False, log=print) -> dict:
    out_dir = Path(out_dir)
    upstream = installed_tiktoken_constants()
    for name, pair in upstream.items():
        if pair != TIKTOKEN_ENCODINGS[name]:
            raise RuntimeError(f"installed tiktoken_ext pins {name} as {pair}, this tool has "
                               f"{TIKTOKEN_ENCODINGS[name]}; update TIKTOKEN_ENCODINGS")
    entries = {}
    for name, (url, digest) in TIKTOKEN_ENCODINGS.items():
        key = cache_key(url)
        target = out_dir / key
        if target.exists() and sha256(target.read_bytes()) == digest:
            data, origin = target.read_bytes(), "existing"
        elif check:
            raise RuntimeError(f"{target} is missing or has the wrong sha256 (run prepare_assets.py)")
        else:
            data, origin = None, ""
            for d in _local_cache_dirs():
                p = d / key
                if p.is_file() and sha256(p.read_bytes()) == digest:
                    data, origin = p.read_bytes(), f"local cache {d}"
                    break
            if data is None:
                if offline:
                    raise RuntimeError(f"{name}: not in a local tiktoken cache and --offline was given")
                data, origin = _download(url), "download"
                if sha256(data) != digest:
                    raise RuntimeError(f"{name}: downloaded file has sha256 {sha256(data)}, expected {digest}")
            _write_if_changed(target, data)
        entries[name] = {"cache_key": key, "url": url, "sha256": digest, "bytes": len(data)}
        log(f"tiktoken {name}: {key} ({len(data):,} bytes, {origin})")
    lines = ["# Generated by src/mobile/tools/prepare_assets.py. Do not edit.",
             "# tiktoken BPE cache seed: runtime_bootstrap.seed_tiktoken_cache copies each file into",
             "# TIKTOKEN_CACHE_DIR. tiktoken looks files up by sha1(url) and checks sha256 (tiktoken/load.py).",
             "format = 1",
             f"tiktoken = {_toml_str(pinned_version('tiktoken'))}", ""]
    for name, e in entries.items():
        lines += [f"[encodings.{name}]",
                  f"cache_key = {_toml_str(e['cache_key'])}",
                  f"file = {_toml_str(e['cache_key'])}",
                  f"url = {_toml_str(e['url'])}",
                  f"sha256 = {_toml_str(e['sha256'])}",
                  f"bytes = {e['bytes']}", ""]
    manifest = "\n".join(lines).encode("utf-8")
    if check:
        mp = out_dir / MANIFEST_NAME
        if not mp.exists() or mp.read_bytes() != manifest:
            raise RuntimeError(f"{mp} is missing or stale (run prepare_assets.py)")
    else:
        _write_if_changed(out_dir / MANIFEST_NAME, manifest)
        known = {e["cache_key"] for e in entries.values()} | {MANIFEST_NAME}
        for p in out_dir.iterdir():
            if p.is_file() and p.name not in known:
                p.unlink()
    return entries


# --------------------------------------------------------------------------- self-test EPUB
CHARACTERS = ["이서연", "강민호", "한지우", "백도윤"]
PLACES = ["아르덴 왕국", "은빛 숲", "검은 탑", "루미나 성"]
TERMS = ["마나석", "성검 엘리시온", "푸른 기사단"]

CHAPTERS = [
    ("은빛 숲의 소녀", [
        "새벽 안개가 은빛 숲을 덮고 있었다. 이서연은 낡은 검을 허리에 차고 숲길을 걸었다.",
        "“오늘은 반드시 마나석을 찾을 거야.” 이서연이 작게 중얼거렸다.",
        "나무 사이로 푸른 빛이 번쩍였다. 그녀는 숨을 죽이고 빛을 향해 다가갔다.",
    ]),
    ("아르덴 왕국의 기사", [
        "아르덴 왕국의 성문 앞에는 푸른 기사단의 깃발이 펄럭이고 있었다.",
        "강민호는 성벽 위에서 이서연을 내려다보며 손을 흔들었다. “늦었잖아, 서연아.”",
        "이서연은 웃으며 대답했다. “숲에서 이상한 빛을 봤어. 마나석일지도 몰라.”",
    ]),
    ("마나석의 비밀", [
        "강민호는 마나석을 손바닥 위에 올려놓고 오랫동안 바라보았다.",
        "“이건 평범한 마나석이 아니야. 검은 탑의 문장이 새겨져 있어.”",
        "이서연은 등골이 서늘해지는 것을 느꼈다. 검은 탑은 백도윤이 다스리는 곳이었다.",
    ]),
    ("검은 탑으로", [
        "두 사람은 아르덴 왕국을 떠나 검은 탑으로 향했다. 길은 험하고 바람은 차가웠다.",
        "“백도윤이 무엇을 꾸미는지 알아내야 해.” 강민호가 지팡이를 고쳐 쥐었다.",
        "멀리 검은 탑의 그림자가 하늘을 찌를 듯 솟아 있었다.",
    ]),
    ("강민호의 마법", [
        "탑의 첫 번째 문은 마법으로 잠겨 있었다. 강민호는 주문을 외우기 시작했다.",
        "푸른 불꽃이 문을 감싸더니 이내 자물쇠가 녹아내렸다.",
        "“역시 강민호야.” 이서연이 감탄하자 그는 어깨를 으쓱했다.",
    ]),
    ("한지우 공주", [
        "탑 안의 감옥에는 아르덴 왕국의 공주 한지우가 갇혀 있었다.",
        "“이서연 기사님, 와 주셨군요.” 한지우의 목소리가 떨렸다.",
        "한지우의 목걸이에는 푸른 기사단의 문장이 새겨져 있었다.",
    ]),
    ("백도윤의 음모", [
        "백도윤은 마나석을 모아 성검 엘리시온을 깨우려 하고 있었다.",
        "“성검 엘리시온이 깨어나면 아르덴 왕국은 끝이다.” 한지우가 말했다.",
        "이서연은 검을 뽑았다. 이제 물러설 곳은 없었다.",
    ]),
    ("성검 엘리시온", [
        "탑의 꼭대기에서 성검 엘리시온이 희미한 빛을 내뿜고 있었다.",
        "백도윤이 천천히 돌아섰다. “늦었군, 이서연.”",
        "강민호가 한지우를 감싸며 뒤로 물러섰다.",
    ]),
    ("루미나 성의 밤", [
        "세 사람은 가까스로 탈출해 루미나 성에 몸을 숨겼다.",
        "한지우는 창밖의 달을 바라보며 은빛 숲에서 보낸 어린 시절을 떠올렸다.",
        "“내일은 푸른 기사단이 도착할 거예요.” 그녀가 조용히 말했다.",
    ]),
    ("배신", [
        "그러나 새벽에 성문을 연 것은 푸른 기사단이 아니라 백도윤의 병사들이었다.",
        "강민호는 이를 악물었다. “누군가 우리를 팔아넘겼어.”",
        "이서연은 한지우의 손을 잡고 비밀 통로로 달렸다.",
    ]),
    ("최후의 결전", [
        "검은 탑 앞에서 이서연과 백도윤이 마주 섰다.",
        "강민호의 마법이 하늘을 가르고, 마나석이 하나둘 부서졌다.",
        "마침내 성검 엘리시온이 이서연의 손에 쥐어졌다.",
    ]),
    ("새로운 아침", [
        "아르덴 왕국에 다시 아침이 찾아왔다. 은빛 숲의 안개도 걷혔다.",
        "한지우는 왕좌에 올라 푸른 기사단을 새로 세웠다.",
        "이서연과 강민호는 루미나 성의 언덕에서 떠오르는 해를 바라보았다.",
    ]),
]

CSS = (b"body { font-family: serif; line-height: 1.7; margin: 0 5%; }\n"
       b"h1 { font-size: 1.4em; text-align: center; margin: 1.5em 0 1em; }\n"
       b"p { text-indent: 1em; margin: 0 0 0.6em; }\n"
       b".emblem { display: block; margin: 1em auto; width: 4em; }\n")


def tiny_png(width: int = 16, height: int = 16) -> bytes:
    """A small deterministic RGB PNG (blue emblem), built with zlib only."""
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            inside = (x - width / 2 + 0.5) ** 2 + (y - height / 2 + 0.5) ** 2 <= (width / 2.5) ** 2
            row += bytes((40, 90, 200) if inside else (245, 245, 250))
        rows.append(bytes(row))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + \
        chunk(b"IEND", b"")


def build_selftest_epub(path: Path) -> bytes:
    """Write the 12-chapter Korean fixture EPUB to ``path`` and return its bytes (reproducible)."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("urn:glossarion:selftest-ko-12ch")
    book.set_title("글로사리온 자가진단: 은빛 숲의 기사")
    book.set_language("ko")
    book.add_author("글로사리온 테스트")
    book.add_metadata("DC", "description", "Glossarion mobile self-test fixture (12 chapters, Korean).")
    book.add_metadata("DC", "date", "2026-01-01")

    style = epub.EpubItem(uid="style_main", file_name="style/main.css", media_type="text/css", content=CSS)
    book.add_item(style)
    emblem = epub.EpubImage(uid="img_emblem", file_name="images/emblem.png", media_type="image/png",
                            content=tiny_png())
    book.add_item(emblem)

    chapters = []
    for i, (title, paragraphs) in enumerate(CHAPTERS, start=1):
        heading = f"제{i}화 {title}"
        body = [f"<h1>{heading}</h1>"]
        if i == 6:
            body.append('<img class="emblem" src="../images/emblem.png" alt="푸른 기사단의 문장"/>')
        body += [f"<p>{p}</p>" for p in paragraphs]
        ch = epub.EpubHtml(uid=f"chapter{i:04d}", title=heading, file_name=f"text/chapter{i:04d}.xhtml", lang="ko")
        ch.content = ("<html><head><title>" + heading + "</title></head><body>" + "".join(body) +
                      "</body></html>")
        ch.add_item(style)
        book.add_item(ch)
        chapters.append(ch)

    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + chapters

    with tempfile.TemporaryDirectory() as td:
        raw_path = Path(td) / "raw.epub"
        epub.write_epub(str(raw_path), book, {"mtime": FIXED_TIME})
        raw = raw_path.read_bytes()
    data = normalize_zip(raw)
    _write_if_changed(Path(path), data)
    return data


def normalize_zip(raw: bytes) -> bytes:
    """Rewrite a zip with fixed timestamps (mimetype first, stored) so the bytes are reproducible."""
    src = zipfile.ZipFile(io.BytesIO(raw))
    names = src.namelist()
    order = (["mimetype"] if "mimetype" in names else []) + [n for n in names if n != "mimetype"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as dst:
        for name in order:
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            dst.writestr(info, src.read(name))
    return buf.getvalue()


def prepare_selftest(out_dir: Path, check: bool = False, log=print) -> dict:
    out_dir = Path(out_dir)
    epub_path = out_dir / SELFTEST_EPUB
    if check:
        if not epub_path.exists():
            raise RuntimeError(f"{epub_path} is missing (run prepare_assets.py)")
        data = epub_path.read_bytes()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        data = build_selftest_epub(epub_path)
    info = {"file": SELFTEST_EPUB, "sha256": sha256(data), "bytes": len(data), "chapters": len(CHAPTERS)}
    lines = ["# Generated by src/mobile/tools/prepare_assets.py. Do not edit.",
             "# Self-test fixture: a 12-chapter Korean EPUB (>10 chapters, so Chapter_Extractor uses its pool path).",
             "format = 1", "",
             "[epub]",
             f"file = {_toml_str(SELFTEST_EPUB)}",
             f"sha256 = {_toml_str(info['sha256'])}",
             f"bytes = {info['bytes']}",
             'title = "글로사리온 자가진단: 은빛 숲의 기사"',
             'language = "ko"',
             f"chapters = {len(CHAPTERS)}",
             'chapter_files = [' + ", ".join(_toml_str(f"text/chapter{i:04d}.xhtml")
                                             for i in range(1, len(CHAPTERS) + 1)) + "]",
             'images = ["images/emblem.png"]',
             "",
             "[glossary]",
             "# Names that recur across chapters; a glossary pass over the book should find them.",
             "characters = [" + ", ".join(_toml_str(c) for c in CHARACTERS) + "]",
             "places = [" + ", ".join(_toml_str(p) for p in PLACES) + "]",
             "terms = [" + ", ".join(_toml_str(t) for t in TERMS) + "]", ""]
    manifest = "\n".join(lines).encode("utf-8")
    if check:
        mp = out_dir / MANIFEST_NAME
        if not mp.exists() or mp.read_bytes() != manifest:
            raise RuntimeError(f"{mp} is missing or stale (run prepare_assets.py)")
    else:
        _write_if_changed(out_dir / MANIFEST_NAME, manifest)
    log(f"selftest EPUB: {epub_path.name} ({len(data):,} bytes, {len(CHAPTERS)} chapters, sha256 {info['sha256'][:16]}...)")
    return info


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate app/assets/tiktoken and app/assets/selftest.")
    ap.add_argument("--assets", default=str(DEFAULT_ASSETS), help="assets dir (default: src/mobile/app/assets)")
    ap.add_argument("--only", choices=["tiktoken", "selftest"], help="generate just one asset group")
    ap.add_argument("--offline", action="store_true", help="never download (use local tiktoken caches only)")
    ap.add_argument("--check", action="store_true", help="verify existing assets instead of writing")
    args = ap.parse_args(argv)
    assets = Path(args.assets)
    try:
        if args.only in (None, "tiktoken"):
            prepare_tiktoken(assets / "tiktoken", offline=args.offline, check=args.check)
        if args.only in (None, "selftest"):
            prepare_selftest(assets / "selftest", check=args.check)
    except ImportError as e:
        print(f"prepare_assets: missing dependency: {e} (install the mobile host env: uv sync)", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"prepare_assets: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
