"""Stub - ebooklib not available on Android."""
ITEM_DOCUMENT = 9
ITEM_STYLE = 3
ITEM_IMAGE = 6
ITEM_NAVIGATION = 1
ITEM_COVER = 2
class EpubBook: pass
class EpubHtml: pass
class EpubItem: pass
def read_epub(*a, **kw): raise NotImplementedError("ebooklib unavailable on Android")
def write_epub(*a, **kw): raise NotImplementedError("ebooklib unavailable on Android")
# Sub-module alias so `from ebooklib import epub` works
import sys as _sys
class _EpubModule:
    ITEM_DOCUMENT = 9
    ITEM_STYLE = 3
    ITEM_IMAGE = 6
    ITEM_NAVIGATION = 1
    ITEM_COVER = 2
    EpubBook = EpubBook
    EpubHtml = EpubHtml
    EpubItem = EpubItem
    read_epub = staticmethod(read_epub)
    write_epub = staticmethod(write_epub)
epub = _EpubModule()
