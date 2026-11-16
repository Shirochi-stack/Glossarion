"""Alias of the shared ``share_hosting_transferit`` in src/ (moved there for the desktop Library).

Importing this module gives the shared module itself, so attributes patched here are the ones the
code uses.
"""

import sys

import share_hosting_transferit as _shared

sys.modules[__name__] = _shared
