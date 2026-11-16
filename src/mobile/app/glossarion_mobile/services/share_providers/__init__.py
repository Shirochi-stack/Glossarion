"""Moved to src/: the share-link providers are shared with the desktop Library.

The code lives in ``share_hosting`` (this package's former contents: errors, cancel token, HTTP
client, provider descriptions, consent text, ``SecretBox``), ``share_hosting_gofile``,
``share_hosting_pixeldrain``, ``share_hosting_send``, ``share_hosting_transferit`` and
``google_drive_core``. The app imports those directly; this package and its submodules stay as
aliases of them, so older imports (and the host tests) keep working on the very same objects.
"""

import share_hosting as _shared

globals().update({name: value for name, value in vars(_shared).items()
                  if not (name.startswith("__") and name.endswith("__")) or name == "__all__"})
