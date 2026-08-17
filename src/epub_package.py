"""Shared discovery helpers for EPUB OPF package documents.

``content.opf`` is a common filename, not an EPUB requirement.  The package
document selected by ``META-INF/container.xml`` is authoritative; filename
and extension searches below are deliberately recovery fallbacks.
"""

from __future__ import annotations

import os
import posixpath
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import unquote


def _local_name(tag: object) -> str:
    return str(tag or "").rsplit("}", 1)[-1]


def _normalized_member_name(name: object) -> str:
    return unquote(str(name or "")).replace("\\", "/").lstrip("/")


def _declared_rootfile(container_data: bytes | str) -> Optional[str]:
    try:
        root = ET.fromstring(container_data)
    except (ET.ParseError, TypeError, ValueError):
        return None
    for element in root.iter():
        if _local_name(element.tag).lower() != "rootfile":
            continue
        full_path = _normalized_member_name(element.attrib.get("full-path"))
        if full_path:
            return full_path
    return None


def find_epub_opf_member(zf) -> Optional[str]:
    """Return the authoritative OPF member name from an open EPUB ZipFile.

    Resolution order is:

    1. The rootfile declared by ``META-INF/container.xml``.
    2. A package named ``content.opf`` (legacy/common convention).
    3. Any ``.opf`` member, selected deterministically.
    """
    try:
        names = [name for name in zf.namelist() if not str(name).endswith("/")]
    except (AttributeError, OSError, ValueError):
        return None

    by_normalized_name = {
        _normalized_member_name(name).casefold(): name for name in names
    }
    container_member = by_normalized_name.get("meta-inf/container.xml")
    if container_member:
        try:
            declared = _declared_rootfile(zf.read(container_member))
        except (KeyError, OSError, ValueError):
            declared = None
        if declared:
            selected = by_normalized_name.get(declared.casefold())
            if selected:
                return selected

    opf_members = [
        name
        for name in names
        if _normalized_member_name(name).casefold().endswith(".opf")
    ]
    if not opf_members:
        return None

    content_members = [
        name
        for name in opf_members
        if posixpath.basename(_normalized_member_name(name)).casefold()
        == "content.opf"
    ]
    candidates = content_members or opf_members
    return min(
        candidates,
        key=lambda name: (
            _normalized_member_name(name).count("/"),
            _normalized_member_name(name).casefold(),
        ),
    )


def find_opf_path(root_dir: str, recursive: bool = True) -> Optional[str]:
    """Return an extracted workspace's authoritative OPF package path.

    Both intact EPUB directory layouts and Glossarion's flattened resource
    layout are supported.  In the latter, ``container.xml`` may be at the
    workspace root and the OPF may retain only its source basename.
    """
    if not root_dir or not os.path.isdir(root_dir):
        return None
    root_dir = os.path.abspath(root_dir)

    container_candidates = (
        os.path.join(root_dir, "META-INF", "container.xml"),
        os.path.join(root_dir, "container.xml"),
    )
    for container_path in container_candidates:
        if not os.path.isfile(container_path):
            continue
        try:
            with open(container_path, "rb") as stream:
                declared = _declared_rootfile(stream.read())
        except OSError:
            declared = None
        if not declared:
            continue
        declared_parts = [part for part in declared.split("/") if part]
        declared_candidate = os.path.abspath(
            os.path.join(root_dir, *declared_parts)
        )
        try:
            declared_is_local = (
                os.path.commonpath((root_dir, declared_candidate)) == root_dir
            )
        except ValueError:
            declared_is_local = False
        candidates = [declared_candidate] if declared_is_local else []
        # Glossarion extracts package resources into a flat workspace.
        candidates.append(os.path.join(root_dir, posixpath.basename(declared)))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

    conventional = (
        os.path.join(root_dir, "content.opf"),
        os.path.join(root_dir, "OEBPS", "content.opf"),
        os.path.join(root_dir, "EPUB", "content.opf"),
    )
    for candidate in conventional:
        if os.path.isfile(candidate):
            return candidate

    try:
        root_opfs = sorted(
            (
                os.path.join(root_dir, name)
                for name in os.listdir(root_dir)
                if name.casefold().endswith(".opf")
                and os.path.isfile(os.path.join(root_dir, name))
            ),
            key=lambda path: os.path.basename(path).casefold(),
        )
    except OSError:
        return None
    if root_opfs:
        return root_opfs[0]
    if not recursive:
        return None

    deep_opfs = []
    try:
        for dirpath, _dirs, files in os.walk(root_dir):
            for name in files:
                if name.casefold().endswith(".opf"):
                    deep_opfs.append(os.path.join(dirpath, name))
    except OSError:
        return None
    return min(
        deep_opfs,
        key=lambda path: (
            os.path.relpath(path, root_dir).count(os.sep),
            os.path.relpath(path, root_dir).casefold(),
        ),
        default=None,
    )
