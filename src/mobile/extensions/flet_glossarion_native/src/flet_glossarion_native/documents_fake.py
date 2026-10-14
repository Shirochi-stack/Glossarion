"""Test double for the document-destination API (no Flet import, nothing touches the disk).

``FakeDocumentsNative`` has the same async methods and answers as ``GlossarionNative``'s
document API on Android (``DocumentDestinations.kt``): the write-mode chain, the "missing only when
the folder root answers" rule, create retries that adopt a late-appearing file, persisted grants,
late picker answers. ``FakeCloudProvider`` is the cloud app behind it, with switches for the
provider behaviours the U10 research found on real phones:

* ``accepted_modes``: modes the provider opens (Drive once refused ``wt``);
* ``truncates_w``: whether ``w`` truncates (OneDrive's does not -> stale tail);
* ``seekable``: descriptors are real files (``Os.ftruncate`` works) or pipes;
* ``offline``: opens throw FileNotFoundException("Error downloading file") (Nextcloud);
* ``fail_next_creates`` / ``create_appears_late``: a create that throws (and maybe exists anyway);
* ``listing_lag``: new documents stay out of listings for N listings (eventual consistency);
* ``supports_rename``: ``DocumentsContract.renameDocument`` works (FLAG_SUPPORTS_RENAME);
* ``no_space``, ``revoke()``, ``move_out()``, ``delete_document()``, ``uninstall()``.

Use it in host tests::

    provider = FakeCloudProvider(label="Drive")
    native = FakeDocumentsNative(provider)
    provider.next_pick("folder", provider.add_folder("Glossarion"))
    target = (await native.pick_folder())["target"]
"""

from __future__ import annotations

import asyncio
import itertools
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from flet_glossarion_native.documents import (
    DEFAULT_MODE_CHAIN,
    TRUNCATING_MODES,
    DocumentError,
    DocumentScope,
    error_result,
    normalize_modes,
    stable_id,
)

__all__ = ["FakeCloudProvider", "FakeDocumentsNative", "FakeNode", "ProviderFault"]

_TREE_PREFIX = "content://{authority}/tree/"


class ProviderFault(Exception):
    """An exception the fake provider throws, named like its Java counterpart."""

    def __init__(self, kind: str, message: str = "") -> None:
        super().__init__(message or kind)
        self.kind = kind  # "security" | "not_found" | "illegal_argument" | "no_space" | "io" | "unsupported"


@dataclass
class FakeNode:
    doc_id: str
    name: str
    is_dir: bool
    parent: Optional["FakeNode"] = None
    data: bytearray = field(default_factory=bytearray)
    children: list = field(default_factory=list)
    deleted: bool = False
    writable: bool = True
    versions: int = 0
    listed_after: int = 0  # visible in listings once the provider's listing counter passes it

    @property
    def size(self) -> int:
        return len(self.data)


class FakeCloudProvider:
    """An in-memory DocumentsProvider plus the system picker that offers it."""

    def __init__(
        self,
        *,
        authority: str = "com.example.cloud.documents",
        label: str = "Cloud",
        accepted_modes: Sequence[str] = ("wt", "rwt", "w", "rw", "r"),
        truncates_w: bool = True,
        seekable: bool = True,
        supports_tree: bool = True,
        grant_write: bool = True,
        supports_rename: bool = True,
    ) -> None:
        self.authority = authority
        self.label = label
        self.accepted_modes = set(accepted_modes)
        self.truncates_w = truncates_w
        self.seekable = seekable
        self.supports_tree = supports_tree
        self.grant_write = grant_write
        self.supports_rename = supports_rename
        self.offline = False
        self.no_space = False
        self.installed = True
        self.fail_next_creates = 0
        self.create_appears_late = False
        self.listing_lag = 0
        self.fail_after_bytes: Optional[int] = None
        self._listings = 0
        self._ids = itertools.count(1)
        self.root = FakeNode(doc_id="root", name=label, is_dir=True)
        self.nodes: dict[str, FakeNode] = {"root": self.root}
        self.grants: dict[str, dict] = {}  # uri -> {"read", "write", "persisted_time"}
        self.moved_out: set[str] = set()
        self._picks: list = []
        self.opened_modes: list[str] = []

    # ---- building the cloud ----------------------------------------------------------------

    def _new_id(self) -> str:
        return f"doc{next(self._ids)}"

    def add_folder(self, name: str, parent: Optional[FakeNode] = None) -> FakeNode:
        return self._add(name, True, parent or self.root)

    def add_file(self, name: str, data: bytes = b"", parent: Optional[FakeNode] = None) -> FakeNode:
        node = self._add(name, False, parent or self.root)
        node.data = bytearray(data)
        return node

    def _add(self, name: str, is_dir: bool, parent: FakeNode) -> FakeNode:
        node = FakeNode(doc_id=self._new_id(), name=name, is_dir=is_dir, parent=parent,
                        listed_after=self._listings + self.listing_lag)
        parent.children.append(node)
        self.nodes[node.doc_id] = node
        return node

    def files_named(self, name: str, parent: Optional[FakeNode] = None) -> list[FakeNode]:
        parent = parent or self.root
        return [c for c in parent.children if not c.deleted and c.name == name]

    def all_files(self) -> list[FakeNode]:
        return [n for n in self.nodes.values() if not n.deleted and not n.is_dir]

    # ---- URIs -------------------------------------------------------------------------------

    def tree_uri(self, node: FakeNode) -> str:
        return f"content://{self.authority}/tree/{node.doc_id}"

    def doc_uri(self, node: FakeNode, tree: Optional[str] = None) -> str:
        if tree:
            return f"{tree}/document/{node.doc_id}"
        return f"content://{self.authority}/document/{node.doc_id}"

    @staticmethod
    def doc_id_of(uri: str) -> str:
        return uri.rstrip("/").rsplit("/", 1)[-1]

    def tree_root_id(self, tree: str) -> str:
        return tree.split("/tree/", 1)[1].split("/", 1)[0]

    # ---- the user / the system --------------------------------------------------------------

    def next_pick(self, kind: str, node: Optional[FakeNode] = None, *, name: Optional[str] = None) -> None:
        """Queue the user's answer to the next picker: a node, None (cancel) or a new file name."""
        self._picks.append((kind, node, name))

    def take_pick(self) -> tuple:
        return self._picks.pop(0) if self._picks else (None, None, None)

    def revoke(self, uri: Optional[str] = None) -> None:
        """Drop a persisted grant (or all of them: app data cleared / reinstalled)."""
        if uri is None:
            self.grants.clear()
        else:
            self.grants.pop(uri, None)

    def move_out(self, node: FakeNode) -> None:
        """The user moved the document out of the picked folder (SecurityException for it)."""
        self.moved_out.add(node.doc_id)

    def delete_document(self, node: FakeNode) -> None:
        node.deleted = True

    def uninstall(self) -> None:
        self.installed = False

    # ---- provider calls (raise ProviderFault like the Java side throws) ---------------------

    def _check_access(self, uri: str) -> FakeNode:
        if not self.installed:
            raise ProviderFault("illegal_argument", "Unknown authority")
        tree = uri.split("/document/")[0] if "/tree/" in uri else None
        grant_uri = tree if tree else uri
        if grant_uri not in self.grants:
            raise ProviderFault("security", "Permission Denial")
        node = self.nodes.get(self.doc_id_of(uri))
        if node is None:
            return None  # type: ignore[return-value]
        if tree:
            root_id = self.tree_root_id(tree)
            if node.doc_id in self.moved_out or not self._is_under(node, root_id):
                raise ProviderFault("security", "Document is not a descendant of the tree")
        return node

    def _is_under(self, node: FakeNode, root_id: str) -> bool:
        current: Optional[FakeNode] = node
        while current is not None:
            if current.doc_id == root_id:
                return True
            current = current.parent
        return False

    def query(self, uri: str) -> Optional[list[dict]]:
        """Rows for a document URI (None = no cursor, like DocumentsProvider.query on FNFE)."""
        node = self._check_access(uri)
        if node is None or node.deleted:
            return None
        return [self._row(node)]

    def query_children(self, tree: str, parent_uri: str) -> Optional[list[dict]]:
        node = self._check_access(parent_uri)
        if node is None or node.deleted:
            return None
        self._listings += 1
        return [self._row(c) for c in node.children if not c.deleted and c.listed_after < self._listings]

    def _row(self, node: FakeNode) -> dict:
        flags = 0
        if node.writable:
            flags |= 0x2  # FLAG_SUPPORTS_WRITE
        flags |= 0x4  # FLAG_SUPPORTS_DELETE
        if node.is_dir:
            flags |= 0x8  # FLAG_DIR_SUPPORTS_CREATE
        if self.supports_rename:
            flags |= 0x40  # FLAG_SUPPORTS_RENAME
        return {
            "document_id": node.doc_id, "name": node.name,
            "mime": "vnd.android.document/directory" if node.is_dir else "application/octet-stream",
            "size": None if node.is_dir else node.size, "modified": node.versions, "flags": flags,
        }

    def create(self, parent_uri: str, name: str, is_dir: bool) -> str:
        node = self._check_access(parent_uri)
        if node is None or node.deleted:
            raise ProviderFault("not_found", "Parent not found")
        if self.no_space:
            raise ProviderFault("no_space", "ENOSPC (No space left on device)")
        if self.fail_next_creates > 0:
            self.fail_next_creates -= 1
            if self.create_appears_late:
                self._add(name, is_dir, node)
            raise ProviderFault("io", "Drive: create failed, try again")
        taken = {c.name for c in node.children if not c.deleted}
        final = name
        if name in taken and self.authority == "com.android.externalstorage.documents":
            stem, ext = os.path.splitext(name)
            n = 1
            while f"{stem} ({n}){ext}" in taken:
                n += 1
            final = f"{stem} ({n}){ext}"
        child = self._add(final, is_dir, node)
        tree = parent_uri.split("/document/")[0]
        return self.doc_uri(child, tree)

    def open(self, uri: str, mode: str) -> "FakeDescriptor":
        node = self._check_access(uri)
        if mode not in self.accepted_modes:
            raise ProviderFault("not_found", f"Unsupported mode: {mode}")
        if self.offline:
            raise ProviderFault("not_found", "Error downloading file")
        if node is None or node.deleted:
            raise ProviderFault("not_found", "Missing file")
        self.opened_modes.append(mode)
        return FakeDescriptor(self, node, mode)

    def delete(self, uri: str) -> bool:
        node = self._check_access(uri)
        if node is None or node.deleted:
            raise ProviderFault("not_found", "Missing file")
        node.deleted = True
        return True

    def rename(self, uri: str, name: str) -> str:
        """``DocumentsContract.renameDocument``: the document keeps its id (and URI) here."""
        node = self._check_access(uri)
        if node is None or node.deleted:
            raise ProviderFault("not_found", "Missing file")
        if not self.supports_rename:
            raise ProviderFault("unsupported", "Rename not supported")
        if node.parent is not None and any(c is not node and not c.deleted and c.name == name
                                           for c in node.parent.children):
            raise ProviderFault("illegal_state", "Already exists")
        node.name = name
        return uri


class FakeDescriptor:
    def __init__(self, provider: FakeCloudProvider, node: FakeNode, mode: str) -> None:
        self.provider = provider
        self.node = node
        self.mode = mode
        self.buffer = bytearray() if (mode in TRUNCATING_MODES or (mode == "w" and provider.truncates_w)) \
            else bytearray(node.data)
        self.position = 0
        self.closed = False

    @property
    def stat_size(self) -> int:
        return len(self.buffer) if self.provider.seekable else -1

    def write(self, chunk: bytes) -> None:
        limit = self.provider.fail_after_bytes
        if limit is not None and self.position + len(chunk) > limit:
            raise ProviderFault("io", "Connection reset")
        if self.provider.no_space:
            raise ProviderFault("no_space", "ENOSPC (No space left on device)")
        end = self.position + len(chunk)
        if end > len(self.buffer):
            self.buffer.extend(b"\0" * (end - len(self.buffer)))
        self.buffer[self.position:end] = chunk
        self.position = end
        # In-place providers expose the bytes as they are written.
        self.node.data = bytearray(self.buffer)

    def truncate(self, size: int) -> None:
        if not self.provider.seekable:
            raise ProviderFault("io", "EINVAL")
        del self.buffer[size:]
        self.node.data = bytearray(self.buffer)

    def close(self) -> None:
        self.closed = True
        self.node.data = bytearray(self.buffer)
        self.node.versions += 1

    def close_with_error(self) -> None:
        self.closed = True


def _classify(fault: ProviderFault, not_found: str) -> str:
    if fault.kind == "no_space":
        return DocumentError.NO_SPACE.value
    if fault.kind == "security":
        return DocumentError.PERMISSION_LOST.value
    if fault.kind in ("not_found", "illegal_argument") and "mode" in str(fault).lower():
        return DocumentError.UNSUPPORTED_MODE.value
    if fault.kind == "not_found":
        return not_found
    return DocumentError.PROVIDER_ERROR.value


def _ok(**values: Any) -> dict:
    out = {"ok": True, "error": None, "message": None, "retryable": False}
    out.update(values)
    return out


class FakeDocumentsNative:
    """``GlossarionNative``'s document methods answered by a :class:`FakeCloudProvider`."""

    platform = "android"

    def __init__(self, provider: FakeCloudProvider, *, chunk_bytes: int = 1 << 20,
                 on_progress: Optional[Callable[[dict], Any]] = None) -> None:
        self.provider = provider
        self.chunk_bytes = chunk_bytes
        self.on_progress = on_progress
        self.calls: list[tuple[str, dict]] = []
        self.cancelled: set[str] = set()
        self.late_results: list[dict] = []
        self.picker_open = False

    # ---- refs ---------------------------------------------------------------------------------

    def _ref(self, uri: Optional[str], document: str, row: Optional[dict], ident: str) -> dict:
        flags = (row or {}).get("flags") or 0
        is_dir = bool(row) and row.get("mime") == "vnd.android.document/directory"
        return {
            "platform": "android", "kind": "folder" if is_dir else "file", "id": ident,
            "uri": uri, "document": document, "bookmark": None, "root": None, "path": None,
            "name": (row or {}).get("name"), "mime": (row or {}).get("mime"),
            "size": (row or {}).get("size"), "mtime": (row or {}).get("modified"), "flags": flags,
            "provider": self.provider.authority, "can_write": bool(flags & 0x2),
            "can_create": is_dir and bool(flags & 0x8), "can_delete": bool(flags & 0x4), "virtual": False,
        }

    @staticmethod
    def _parse(raw: Any) -> Optional[tuple[str, Optional[str], str]]:
        if isinstance(raw, str):
            raw = {"kind": "folder", "uri": raw} if "/tree/" in raw and "/document/" not in raw \
                else {"kind": "file", "document": raw}
        if not isinstance(raw, Mapping):
            return None
        tree = raw.get("uri") or None
        document = raw.get("document") or None
        if document is None and tree:
            document = f"{tree}/document/{tree.split('/tree/', 1)[1]}"
        if document is None:
            return None
        kind = "folder" if raw.get("kind") == "folder" or not raw.get("document") else "file"
        return kind, tree, document

    def _root_of(self, tree: str) -> str:
        return f"{tree}/document/{self.provider.tree_root_id(tree)}"

    def _id_for(self, tree: Optional[str], document: str) -> str:
        if tree and document == self._root_of(tree):
            return stable_id(f"android:{tree}")
        return stable_id(f"android:{document}")

    def _diagnose(self, tree: Optional[str], document: str) -> tuple[str, str]:
        p = self.provider
        if tree:
            if tree not in p.grants:
                return DocumentError.PERMISSION_LOST.value, DocumentScope.TARGET.value
            root = self._root_of(tree)
            try:
                rows = p.query(root)
            except ProviderFault as fault:
                code = _classify(fault, DocumentError.PROVIDER_ERROR.value)
                return code if code == DocumentError.PERMISSION_LOST.value else DocumentError.PROVIDER_ERROR.value, \
                    DocumentScope.TARGET.value
            if not rows:
                return DocumentError.PROVIDER_ERROR.value, DocumentScope.TARGET.value
            if document == root:
                return DocumentError.PROVIDER_ERROR.value, DocumentScope.TARGET.value
            try:
                rows = p.query(document)
            except ProviderFault as fault:
                if fault.kind == "security":
                    return DocumentError.MISSING.value, DocumentScope.DOCUMENT.value
                return DocumentError.PROVIDER_ERROR.value, DocumentScope.DOCUMENT.value
            return (DocumentError.MISSING.value if not rows else DocumentError.PROVIDER_ERROR.value,
                    DocumentScope.DOCUMENT.value)
        if document not in p.grants:
            return DocumentError.PERMISSION_LOST.value, DocumentScope.DOCUMENT.value
        try:
            rows = p.query(document)
        except ProviderFault as fault:
            code = DocumentError.PERMISSION_LOST.value if fault.kind == "security" else DocumentError.PROVIDER_ERROR.value
            return code, DocumentScope.DOCUMENT.value
        return (DocumentError.MISSING.value if not rows else DocumentError.PROVIDER_ERROR.value,
                DocumentScope.DOCUMENT.value)

    def _diagnosed(self, tree: Optional[str], document: str, message: str, **extra: Any) -> dict:
        code, scope = self._diagnose(tree, document)
        out = error_result(code, message, scope=scope, **extra)
        if code == DocumentError.MISSING.value:
            out["proven"] = tree is not None
        return out

    # ---- pickers --------------------------------------------------------------------------------

    async def _pick(self, kind: str, op_id: Optional[str], **args: Any) -> dict:
        self.calls.append(("pick_" + kind, dict(args)))
        if self.picker_open:
            return error_result(DocumentError.BUSY, "Another picker is already open")
        answer_kind, node, name = self.provider.take_pick()
        if answer_kind is None or (node is None and name is None):
            return error_result(DocumentError.CANCELLED, "No location was chosen")
        p = self.provider
        if kind == "folder":
            if not p.supports_tree:
                return error_result(DocumentError.CANCELLED, "The cloud app is not offered as a folder")
            tree = p.tree_uri(node)
            p.grants[tree] = {"read": True, "write": p.grant_write, "persisted_time": len(p.grants)}
            root = self._root_of(tree)
            row = p.query(root)[0]
            target = self._ref(tree, root, row, stable_id(f"android:{tree}"))
            target.update(kind="folder", provider_label=p.label, persisted=True, own_folder=False,
                          can_write=p.grant_write, can_create=p.grant_write)
            return _ok(target=target, persisted=True, persist_error=None)
        if node is None:
            node = p.add_file(name or args.get("name") or "file")
        uri = p.doc_uri(node)
        p.grants[uri] = {"read": True, "write": p.grant_write, "persisted_time": len(p.grants)}
        row = p.query(uri)[0]
        document = self._ref(None, uri, row, stable_id(f"android:{uri}"))
        document.update(kind="file", provider_label=p.label, persisted=True, own_folder=False)
        out = _ok(document=document, persisted=True, persist_error=None, write=None)
        source = args.get("source_path")
        if kind == "save_location" and source and os.path.isfile(source):
            out["write"] = self._write_chain(None, uri, source, normalize_modes(args.get("mode_chain")),
                                             op_id or "pick", True)
        return out

    async def pick_folder(self, *, initial: Any = None, op_id: Optional[str] = None) -> dict:
        return await self._pick("folder", op_id, initial=initial)

    async def pick_save_location(self, name: str, mime_type: str, source_path: Optional[str] = None, *,
                                 initial: Any = None, mode_chain: Sequence[str] = DEFAULT_MODE_CHAIN,
                                 op_id: Optional[str] = None) -> dict:
        return await self._pick("save_location", op_id, name=name, mime_type=mime_type,
                                source_path=source_path, mode_chain=mode_chain)

    async def pick_document(self, mime_types: Optional[Sequence[str]] = None, *, initial: Any = None,
                            op_id: Optional[str] = None) -> dict:
        return await self._pick("document", op_id, mime_types=mime_types)

    async def take_document_results(self) -> list[dict]:
        out, self.late_results = self.late_results, []
        return out

    # ---- folders --------------------------------------------------------------------------------

    def _children_named(self, tree: str, document: str, name: str) -> Optional[list[dict]]:
        rows = self.provider.query_children(tree, document)
        if rows is None:
            return None
        out = []
        for row in rows:
            if row["name"] != name:
                continue
            uri = f"{tree}/document/{row['document_id']}"
            out.append(self._ref(tree, uri, row, stable_id(f"android:{uri}")))
        return out

    async def list_children(self, folder: Any, *, names: Optional[Sequence[str]] = None) -> dict:
        self.calls.append(("list_children", {"folder": folder, "names": names}))
        parsed = self._parse(folder)
        if parsed is None or not parsed[1]:
            return error_result(DocumentError.BAD_ARGS, "folder is required")
        _, tree, document = parsed
        try:
            rows = self.provider.query_children(tree, document)
        except ProviderFault as fault:
            return self._diagnosed(tree, document, str(fault))
        if rows is None:
            return self._diagnosed(tree, document, "The cloud app did not list the folder")
        wanted = set(names) if names is not None else None
        children = []
        for row in rows:
            if wanted is not None and row["name"] not in wanted:
                continue
            uri = f"{tree}/document/{row['document_id']}"
            children.append(self._ref(tree, uri, row, stable_id(f"android:{uri}")))
        return _ok(children=children, complete=True)

    def _create_child(self, tree: Optional[str], document: str, name: str, is_dir: bool, on_exists: str) -> dict:
        if not tree:
            return error_result(DocumentError.BAD_ARGS, "Files can only be created inside a picked folder")
        try:
            before = self._children_named(tree, document, name)
        except ProviderFault as fault:
            if fault.kind == "security":
                return self._diagnosed(tree, document, str(fault))
            before = None
        existing = before[0] if before else None
        if existing is not None and on_exists != "rename":
            if on_exists == "fail":
                return error_result(DocumentError.EXISTS, f'"{name}" already exists',
                                    scope=DocumentScope.DOCUMENT, document=existing)
            return _ok(document=existing, created=False, adopted=True)
        before_ids = {c["document"] for c in before} if before is not None else None
        last: Optional[ProviderFault] = None
        for attempt in range(2):
            try:
                uri = self.provider.create(document, name, is_dir)
                row = (self.provider.query(uri) or [None])[0]
                return _ok(document=self._ref(tree, uri, row, stable_id(f"android:{uri}")), created=True, adopted=False)
            except ProviderFault as fault:
                if fault.kind == "security":
                    return self._diagnosed(tree, document, str(fault))
                if fault.kind == "no_space":
                    return error_result(DocumentError.NO_SPACE, str(fault), scope=DocumentScope.TARGET)
                last = fault
            if attempt == 0 and before_ids is not None:
                try:
                    appeared = [c for c in (self._children_named(tree, document, name) or [])
                                if c["document"] not in before_ids]
                except ProviderFault:
                    appeared = []
                if len(appeared) == 1:
                    return _ok(document=appeared[0], created=True, adopted=False, late=True)
        if last is not None and last.kind == "not_found":
            return self._diagnosed(tree, document, str(last))
        if last is not None and last.kind == "unsupported":
            # DocumentsProvider.createDocument's default UnsupportedOperationException("Create not supported"):
            # this folder takes no new items of that kind (a provider that refuses sub-folders) - not a failure
            # worth retrying (``DocumentDestinations.createChild``)
            return error_result(DocumentError.READ_ONLY, str(last), scope=DocumentScope.DOCUMENT,
                                create_unsupported=True)
        return error_result(_classify(last, DocumentError.PROVIDER_ERROR.value) if last else DocumentError.PROVIDER_ERROR,
                            str(last), scope=DocumentScope.TARGET)

    async def create_file(self, folder: Any, name: str, mime_type: str, *, on_exists: str = "rename") -> dict:
        self.calls.append(("create_file", {"folder": folder, "name": name, "on_exists": on_exists}))
        parsed = self._parse(folder)
        if parsed is None:
            return error_result(DocumentError.BAD_ARGS, "folder is required")
        return self._create_child(parsed[1], parsed[2], name, False, on_exists)

    async def create_folder(self, parent: Any, name: str, *, on_exists: str = "adopt") -> dict:
        self.calls.append(("create_folder", {"folder": parent, "name": name, "on_exists": on_exists}))
        parsed = self._parse(parent)
        if parsed is None:
            return error_result(DocumentError.BAD_ARGS, "folder is required")
        made = self._create_child(parsed[1], parsed[2], name, True, on_exists)
        if made.get("ok"):
            made["folder"] = made.pop("document")
            made["folder"]["kind"] = "folder"
        return made

    # ---- writing --------------------------------------------------------------------------------

    def _remote_size(self, document: str) -> Optional[int]:
        try:
            fd = self.provider.open(document, "r")
        except ProviderFault:
            fd = None
        if fd is not None and fd.stat_size >= 0:
            return fd.stat_size
        try:
            rows = self.provider.query(document)
        except ProviderFault:
            return None
        return rows[0]["size"] if rows else None

    def _write_chain(self, tree: Optional[str], document: str, source: str, modes: Sequence[str],
                     op_id: str, verify: bool) -> dict:
        with open(source, "rb") as handle:
            data = handle.read()
        total = len(data)
        attempts: list[dict] = []
        remote_before: Optional[int] = None
        known = False
        permission = False
        for mode in modes:
            if op_id in self.cancelled:
                return error_result(DocumentError.CANCELLED, "Cancelled", scope=DocumentScope.DOCUMENT,
                                    attempts=attempts, written=0)
            if mode in ("w", "rw"):
                if not known:
                    remote_before, known = self._remote_size(document), True
                if remote_before is None or total < remote_before:
                    attempts.append({"mode": mode, "error": "skipped", "message": "shorter or unknown"})
                    continue
            try:
                fd = self.provider.open(document, mode)
            except ProviderFault as fault:
                code = _classify(fault, "not_found")
                attempts.append({"mode": mode, "error": code, "message": str(fault)})
                if code == DocumentError.NO_SPACE.value:
                    return error_result(code, str(fault), scope=DocumentScope.DOCUMENT, attempts=attempts, written=0)
                if code == DocumentError.PERMISSION_LOST.value:
                    permission = True
                    break
                continue
            return self._stream(fd, mode, tree, document, data, total, op_id, verify, attempts, remote_before)
        only_modes = not permission and attempts and all(
            a["error"] in (DocumentError.UNSUPPORTED_MODE.value, "skipped") for a in attempts)
        if only_modes:
            return error_result(DocumentError.UNSUPPORTED_MODE, "The cloud app accepted none of the write modes",
                                scope=DocumentScope.DOCUMENT, attempts=attempts, needs_replace=True,
                                remote_size=remote_before, written=0)
        return self._diagnosed(tree, document, "; ".join(f"{a['mode']}: {a['error']}" for a in attempts),
                               attempts=attempts, written=0)

    def _stream(self, fd: FakeDescriptor, mode: str, tree: Optional[str], document: str, data: bytes, total: int,
                op_id: str, verify: bool, attempts: list, remote_before: Optional[int]) -> dict:
        truncating = mode in TRUNCATING_MODES
        written = 0
        try:
            for start in range(0, total, self.chunk_bytes):
                if op_id in self.cancelled:
                    raise ProviderFault("cancelled", "cancelled")
                chunk = data[start:start + self.chunk_bytes]
                fd.write(chunk)
                written += len(chunk)
                if self.on_progress is not None:
                    self.on_progress({"type": "progress", "op_id": op_id, "written": written, "total": total})
            if not truncating and fd.stat_size >= 0:
                fd.truncate(written)
            fd.close()
        except ProviderFault as fault:
            code = DocumentError.CANCELLED.value if fault.kind == "cancelled" else _classify(
                fault, DocumentError.PROVIDER_ERROR.value)
            fd.close_with_error()
            attempts.append({"mode": mode, "error": code, "message": str(fault)})
            extra = {"attempts": attempts, "mode": mode, "written": written,
                     "remote_damaged": truncating or written > 0}
            if code == DocumentError.PERMISSION_LOST.value:
                return self._diagnosed(tree, document, str(fault), **extra)
            return error_result(code, str(fault), scope=DocumentScope.DOCUMENT, **extra)
        attempts.append({"mode": mode, "error": "ok", "message": None})
        verified = None
        if verify:
            try:
                check = self.provider.open(document, "r")
                verified = check.stat_size if check.stat_size >= 0 else None
            except ProviderFault:
                verified = None
        row = (self.provider.query(document) or [None])[0]
        ref = self._ref(tree, document, row, self._id_for(tree, document))
        if verified is not None and verified != written:
            stale = verified > written
            return error_result(DocumentError.SIZE_MISMATCH, "length differs", scope=DocumentScope.DOCUMENT,
                                attempts=attempts, mode=mode, written=written, verified_size=verified,
                                stale_tail=stale, needs_replace=stale, remote_damaged=True, document=ref)
        return _ok(document=ref, mode=mode, written=written, total=total, verified_size=verified,
                   verified=verified is not None, reported_size=(row or {}).get("size"),
                   remote_size_before=remote_before, attempts=attempts)

    async def write_file(self, target_or_doc: Any, source_path: str, *, name: Optional[str] = None,
                         mime_type: Optional[str] = None, on_exists: str = "rename",
                         mode_chain: Sequence[str] = DEFAULT_MODE_CHAIN, verify: bool = True,
                         op_id: Optional[str] = None, timeout: Optional[float] = None) -> dict:
        op = op_id or f"op{len(self.calls)}"
        self.calls.append(("write_file", {"ref": target_or_doc, "source_path": source_path, "name": name,
                                          "mode_chain": tuple(mode_chain), "op_id": op}))
        await asyncio.sleep(0)
        modes = normalize_modes(mode_chain)
        parsed = self._parse(target_or_doc)
        if parsed is None or not modes:
            return error_result(DocumentError.BAD_ARGS, "ref and mode_chain are required", op_id=op)
        if not source_path or not os.path.isfile(source_path):
            return error_result(DocumentError.SOURCE_MISSING, "The file to upload is missing",
                                scope=DocumentScope.SOURCE, op_id=op)
        kind, tree, document = parsed
        created = adopted = False
        try:
            if kind == "folder":
                made = self._create_child(tree, document, name or os.path.basename(source_path), False, on_exists)
                if not made.get("ok"):
                    made["op_id"] = op
                    return made
                created, adopted = bool(made.get("created")), bool(made.get("adopted"))
                document = made["document"]["document"]
            out = self._write_chain(tree, document, source_path, modes, op, verify)
        finally:
            self.cancelled.discard(op)
        out.update(created=created, adopted=adopted, op_id=op)
        return out

    # ---- the rest -------------------------------------------------------------------------------

    async def stat(self, document: Any) -> dict:
        parsed = self._parse(document)
        if parsed is None:
            return error_result(DocumentError.BAD_ARGS, "document is required")
        _, tree, uri = parsed
        try:
            rows = self.provider.query(uri)
        except ProviderFault as fault:
            return self._diagnosed(tree, uri, str(fault))
        if not rows:
            return self._diagnosed(tree, uri, "The cloud app did not answer for this document")
        return _ok(document=self._ref(tree, uri, rows[0], self._id_for(tree, uri)))

    async def delete(self, document: Any) -> dict:
        parsed = self._parse(document)
        if parsed is None:
            return error_result(DocumentError.BAD_ARGS, "document is required")
        _, tree, uri = parsed
        try:
            self.provider.delete(uri)
        except ProviderFault as fault:
            return self._diagnosed(tree, uri, str(fault))
        return _ok(deleted=True)

    async def rename_document(self, document: Any, name: str) -> dict:
        self.calls.append(("rename_document", {"name": name}))
        parsed = self._parse(document)
        if parsed is None or not name:
            return error_result(DocumentError.BAD_ARGS, "document and name are required")
        _, tree, uri = parsed
        if tree and uri == self._root_of(tree):
            return error_result(DocumentError.BAD_ARGS, "The picked folder itself is not renamed")
        try:
            renamed = self.provider.rename(uri, name)
        except ProviderFault as fault:
            if fault.kind == "unsupported":
                return error_result(DocumentError.UNAVAILABLE, str(fault), scope=DocumentScope.DOCUMENT)
            if fault.kind == "illegal_state":
                return error_result(DocumentError.EXISTS, str(fault), scope=DocumentScope.DOCUMENT)
            return self._diagnosed(tree, uri, str(fault))
        rows = self.provider.query(renamed)
        return _ok(document=self._ref(tree, renamed, rows[0] if rows else None, stable_id(f"android:{renamed}")),
                   renamed=True)

    async def query_root(self, target: Any) -> dict:
        parsed = self._parse(target)
        if parsed is None:
            return error_result(DocumentError.BAD_ARGS, "target is required")
        _, tree, uri = parsed
        p = self.provider
        if not p.installed:
            return error_result(DocumentError.PERMISSION_LOST, "The cloud app is no longer installed",
                                scope=DocumentScope.TARGET, provider_missing=True)
        grant_uri = tree or uri
        grant = p.grants.get(grant_uri)
        if grant is None:
            return error_result(DocumentError.PERMISSION_LOST, "Glossarion no longer has access",
                                scope=DocumentScope.TARGET)
        root = self._root_of(tree) if tree else uri
        try:
            rows = p.query(root)
        except ProviderFault as fault:
            code = DocumentError.PERMISSION_LOST if fault.kind == "security" else DocumentError.PROVIDER_ERROR
            return error_result(code, str(fault), scope=DocumentScope.TARGET)
        if not rows:
            return error_result(DocumentError.PROVIDER_ERROR if tree else DocumentError.MISSING,
                                "The cloud app did not answer", scope=DocumentScope.TARGET)
        ident = stable_id(f"android:{tree}") if tree else stable_id(f"android:{uri}")
        ref = self._ref(tree, root, rows[0], ident)
        ref.update(kind="folder" if tree else "file", provider_label=p.label, persisted=True,
                   own_folder=False, can_write=bool(grant.get("write")))
        return _ok(target=ref)

    async def release(self, target: Any) -> bool:
        if isinstance(target, str):
            uri = target
        else:
            parsed = self._parse(target)
            if parsed is None:
                return False
            kind, tree, document = parsed
            uri = tree if tree and (document == self._root_of(tree) or kind == "folder" and not
                                    (isinstance(target, Mapping) and target.get("document"))) else document
        return self.provider.grants.pop(uri, None) is not None

    async def list_grants(self) -> list[dict]:
        return [{"uri": uri, "read": g["read"], "write": g["write"], "persisted_time": g["persisted_time"],
                 "tree": "/tree/" in uri and "/document/" not in uri} for uri, g in self.provider.grants.items()]

    async def cancel_document_op(self, op_id: str) -> bool:
        if not op_id:
            return False
        self.cancelled.add(str(op_id))
        return True
