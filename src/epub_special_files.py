"""Apply special-file rules to an EPUB's ordered document list."""

import re
from pathlib import PurePosixPath


def special_file_flags(filenames, predicate, *, protect_interior=False):
    """Optionally protect interior documents and numbered trailing files.

    The original first and last non-special documents are the boundaries.
    Leading files keep their classification, even when their names have digits.
    """
    filenames = list(filenames)
    flags = [bool(predicate(filename)) for filename in filenames]
    if protect_interior:
        first = next((index for index, flag in enumerate(flags) if not flag), None)
        if first is not None:
            last = next(index for index in range(len(flags) - 1, first - 1, -1) if not flags[index])
            flags[first:last + 1] = [False] * (last - first + 1)
            # Only the original trailing block gets this numbered-file override.
            # Do not turn exempted files into new anchors that protect other tail files.
            for index in range(last + 1, len(flags)):
                stem = PurePosixPath(str(filenames[index]).replace("\\", "/")).stem
                if re.search(r"\d", stem):
                    flags[index] = False
    return flags
