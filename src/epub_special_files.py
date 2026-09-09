"""Apply special-file rules to an EPUB's ordered document list."""


def special_file_flags(filenames, predicate, *, protect_interior=False):
    """Optionally limit special files to the leading and trailing runs.

    The first and last non-special documents are the boundaries. Document
    positions come from reading order, not numbers embedded in filenames.
    """
    flags = [bool(predicate(filename)) for filename in filenames]
    if protect_interior:
        first = next((index for index, flag in enumerate(flags) if not flag), None)
        if first is not None:
            last = next(index for index in range(len(flags) - 1, first - 1, -1) if not flags[index])
            flags[first:last + 1] = [False] * (last - first + 1)
    return flags
