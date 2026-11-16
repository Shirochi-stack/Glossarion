"""Alias of the shared ``share_hosting_gofile`` in src/ (moved there for the desktop Library).

Importing this module gives the shared module itself, so attributes patched here are the ones the
code uses.
"""

import sys

import share_hosting_gofile as _shared

sys.modules[__name__] = _shared
