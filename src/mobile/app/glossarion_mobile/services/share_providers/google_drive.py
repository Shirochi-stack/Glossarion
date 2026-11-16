"""Alias of the shared ``google_drive_core`` in src/ (moved there for the desktop Library).

Importing this module gives the shared module itself, so attributes patched here are the ones the
code uses.
"""

import sys

import google_drive_core as _shared

sys.modules[__name__] = _shared
