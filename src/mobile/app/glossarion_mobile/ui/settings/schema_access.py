"""Tolerant access to the shared ``settings_schema`` module (no Flet import).

The schema (``src/settings_schema.py`` + generated ``settings_schema_data.py``)
is GUI-free and shared with desktop. Its contract (plan §2, U2):

    SettingSpec(key, type, default, save_default, converter, var_names, widget_sources,
                env, section, label, tooltip, choices, minimum, maximum, visible_if,
                locked_if, platforms, discrepancies)
    all_specs(), spec(key), sections() -> [Section(id, title, keys, group)],
    effective_default(key), coerce(key, value), search(query) -> [SettingSpec],
    is_available(key, platform='mobile') -> (bool, reason)

``SchemaAccess`` imports it lazily (after the bootstrap put the backend on
``sys.path``), normalises sections, and turns every failure into a value the
UI can render (an unavailable schema shows a notice instead of crashing; a
schema without ``coerce`` stores values as typed). ``visible_if``/``locked_if``
rule ids are evaluated by ``settings_schema.evaluate_rule`` or
``settings_rules.evaluate`` when one exists; rules are optional data until the
curated overlay grows.
"""

from __future__ import annotations

import copy
import importlib
import logging
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from glossarion_mobile.ui.settings.model import (
    config_path,
    env_names,
    group_title,
    label_for,
    ordered_groups,
    plain_text,
    spec_attr,
)

__all__ = ["SchemaAccess", "SearchHit", "SectionInfo", "UNAVAILABLE_REASON"]

log = logging.getLogger("glossarion.settings")

UNAVAILABLE_REASON = "Not available on mobile"
_LOCK_REASON = "Locked by the current settings"
_HIDDEN_REASON = "Not used with the current settings"
SEARCH_LIMIT = 50


@dataclass(frozen=True)
class SectionInfo:
    id: str
    title: str
    keys: tuple[str, ...]
    group: str = "Settings"


@dataclass(frozen=True)
class SearchHit:
    key: str
    spec: Any
    section_id: str
    section_title: str
    group: str

    @property
    def breadcrumb(self) -> str:
        return f"{self.group} › {self.section_title}"


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


class SchemaAccess:
    """Wraps the schema module; ``module=None`` imports ``settings_schema`` on first use."""

    def __init__(
        self,
        module: Any = None,
        *,
        platform: str = "mobile",
        module_name: str = "settings_schema",
        rules_module_name: str = "settings_rules",
    ) -> None:
        self._module = module
        self._tried = module is not None
        self._module_name = module_name
        self._rules_module_name = rules_module_name
        self._rules: Any = None
        self._rules_tried = False
        self.platform = platform
        self.error: Optional[str] = None
        self._sections: Optional[list[SectionInfo]] = None
        self._section_by_id: dict[str, SectionInfo] = {}
        self._section_by_key: dict[str, SectionInfo] = {}
        self._defaults: dict[str, Any] = {}
        self._defaults_lock = threading.Lock()
        self.pending_defaults: set[str] = set()

    # ---- module ------------------------------------------------------------------------

    @property
    def module(self) -> Any:
        if not self._tried:
            self._tried = True
            try:
                self._module = importlib.import_module(self._module_name)
            except Exception as exc:  # missing in this build, or broken
                self.error = f"{type(exc).__name__}: {exc}"
                log.warning("settings schema unavailable: %s", self.error)
                self._module = None
        return self._module

    @property
    def available(self) -> bool:
        return self.module is not None

    def _rules_module(self) -> Any:
        if not self._rules_tried:
            self._rules_tried = True
            try:
                self._rules = importlib.import_module(self._rules_module_name)
            except Exception:
                self._rules = None
        return self._rules

    # ---- sections ------------------------------------------------------------------------

    def sections(self) -> list[SectionInfo]:
        if self._sections is not None:
            return list(self._sections)
        out: list[SectionInfo] = []
        module = self.module
        raw: Any = []
        if module is not None:
            try:
                raw = list(module.sections())
            except Exception as exc:
                self.error = f"sections() failed: {type(exc).__name__}: {exc}"
                log.exception("settings_schema.sections() failed")
                raw = []
        for item in raw:
            section_id = str(_field(item, "id", "") or "").strip()
            if not section_id:
                continue
            keys = tuple(str(k) for k in (_field(item, "keys", ()) or ()))
            out.append(
                SectionInfo(
                    id=section_id,
                    title=str(_field(item, "title", "") or section_id),
                    keys=keys,
                    group=group_title(section_id, _field(item, "group", "")),
                )
            )
        self._sections = out
        self._section_by_id = {s.id: s for s in out}
        self._section_by_key = {}
        for section in out:
            for key in section.keys:
                self._section_by_key.setdefault(key, section)
        return list(out)

    def groups(self) -> list[tuple[str, list[SectionInfo]]]:
        sections = self.sections()
        order = ordered_groups([s.group for s in sections])
        return [(group, [s for s in sections if s.group == group]) for group in order]

    def section(self, section_id: str) -> Optional[SectionInfo]:
        self.sections()
        return self._section_by_id.get(section_id)

    def section_for_key(self, key: str) -> Optional[SectionInfo]:
        self.sections()
        found = self._section_by_key.get(key)
        if found is not None:
            return found
        spec = self.spec(key)
        section_id = str(spec_attr(spec, "section", "") or "") if spec is not None else ""
        return self._section_by_id.get(section_id)

    # ---- specs ---------------------------------------------------------------------------

    def spec(self, key: str) -> Any:
        module = self.module
        if module is None:
            return None
        try:
            return module.spec(key)
        except Exception:
            return None

    def all_specs(self) -> list[Any]:
        module = self.module
        if module is None:
            return []
        try:
            return list(module.all_specs())
        except Exception:
            log.exception("settings_schema.all_specs() failed")
            return []

    def path_of(self, key: str) -> tuple:
        """config.json path of ``key`` (nested settings are dotted paths into their parent)."""
        spec = self.spec(key)
        return config_path(spec) if spec is not None else (key,)

    def specs_for(self, section: SectionInfo) -> list[Any]:
        out = []
        for key in section.keys:
            spec = self.spec(key)
            if spec is not None:
                out.append(spec)
        return out

    @staticmethod
    def _marker(spec: Any) -> tuple[Optional[str], Optional[str]]:
        """(``"$ref"``/``"$expr"``, target) when the schema default is a lazy marker."""
        default = getattr(spec, "default", None)
        if isinstance(default, dict) and len(default) == 1:
            tag, target = next(iter(default.items()))
            if tag in ("$ref", "$expr", "$attr"):
                return str(tag), str(target)
        return None, None

    def effective_default(self, key: str, *, allow_import: bool = False) -> Any:
        """Schema display default (cached). A ``$ref`` default whose module is not imported yet
        returns None on the UI loop (``pending_defaults``); ``warm_defaults`` resolves them on a
        worker thread, because importing e.g. extract_glossary_from_epub takes seconds."""
        name = ".".join(key) if isinstance(key, tuple) else str(key)
        with self._defaults_lock:
            if name in self._defaults:
                return copy.deepcopy(self._defaults[name])
        module = self.module
        if module is None:
            return None
        spec = self.spec(name)
        tag, target = self._marker(spec)
        if tag == "$ref" and target and ":" in target and not allow_import:
            if target.split(":", 1)[0] not in sys.modules:
                with self._defaults_lock:
                    self.pending_defaults.add(name)
                return None
        func = getattr(module, "effective_default", None)
        try:
            if func is not None:
                value = func(name)
            else:
                value = spec_attr(spec, "default", None) if spec is not None and tag is None else None
        except Exception:
            value = None
        with self._defaults_lock:
            self._defaults[name] = copy.deepcopy(value)
            self.pending_defaults.discard(name)
        return value

    def warm_defaults(self) -> int:
        """Resolve every display default, importing ``$ref`` modules (blocking; worker thread only)."""
        count = 0
        for spec in self.all_specs():
            key = str(spec_attr(spec, "key", ""))
            if key:
                self.effective_default(key, allow_import=True)
                count += 1
        return count

    def default_note(self, key: str) -> Optional[str]:
        """Why no default can be shown: computed at run time, or not resolved yet."""
        spec = self.spec(key)
        tag, _target = self._marker(spec)
        if tag in ("$expr", "$attr"):
            return "computed when a run starts"
        with self._defaults_lock:
            if key in self.pending_defaults:
                return "loading default…"
        return None

    def help_text(self, spec: Any) -> str:
        return plain_text(spec_attr(spec, "tooltip", ""))

    def coerce(self, key: str, value: Any) -> Any:
        """Schema conversion of an edited value; raises ``ValueError`` with a message for invalid input."""
        module = self.module
        func = getattr(module, "coerce", None) if module is not None else None
        if func is None:
            return value
        try:
            return func(key, value)
        except Exception as exc:  # desktop converter lambdas raise whatever they raise
            raise ValueError(str(exc) or f"Invalid value for {key}") from exc

    # ---- availability, rules ------------------------------------------------------------

    def availability(self, key: str) -> tuple[bool, Optional[str]]:
        module = self.module
        if module is None:
            return True, None
        func = getattr(module, "is_available", None)
        if func is not None:
            try:
                result = func(key, self.platform)
            except TypeError:
                result = func(key)
            except Exception:
                return True, None
            if isinstance(result, tuple):
                ok = bool(result[0])
                reason = result[1] if len(result) > 1 else None
                return ok, (str(reason) if reason else (None if ok else UNAVAILABLE_REASON))
            return bool(result), (None if result else UNAVAILABLE_REASON)
        spec = self.spec(key)
        platforms = spec_attr(spec, "platforms", None) if spec is not None else None
        if platforms and self.platform not in platforms:
            return False, UNAVAILABLE_REASON
        return True, None

    def _evaluate(self, rule: Any, config: Mapping[str, Any], default_reason: str) -> Optional[str]:
        if not rule:
            return None
        rules = rule if isinstance(rule, (tuple, list)) else (rule,)
        for item in rules:
            result: Any = None
            try:
                if callable(item):
                    result = item(config)
                else:
                    evaluator = self._evaluator()
                    if evaluator is None:
                        continue
                    result = evaluator(str(item), config)
            except Exception:
                log.debug("rule %r failed", item, exc_info=True)
                continue
            if isinstance(result, str) and result.strip():
                return result.strip()
            if result:
                return default_reason
        return None

    def _evaluator(self) -> Optional[Callable[[str, Mapping[str, Any]], Any]]:
        for module in (self.module, self._rules_module()):
            if module is None:
                continue
            for name in ("evaluate_rule", "evaluate"):
                func = getattr(module, name, None)
                if callable(func):
                    return func
        return None

    def lock_reason(self, spec: Any, config: Mapping[str, Any]) -> Optional[str]:
        return self._evaluate(spec_attr(spec, "locked_if", None), config, _LOCK_REASON)

    def hidden_reason(self, spec: Any, config: Mapping[str, Any]) -> Optional[str]:
        """Reason when ``visible_if`` says the field is not used (it stays visible, disabled)."""
        rule = spec_attr(spec, "visible_if", None)
        if not rule:
            return None
        rules = rule if isinstance(rule, (tuple, list)) else (rule,)
        evaluator = self._evaluator()
        for item in rules:
            try:
                if callable(item):
                    result = item(config)
                elif evaluator is not None:
                    result = evaluator(str(item), config)
                else:
                    continue
            except Exception:
                continue
            if result is False or result == "" or result == 0:
                return _HIDDEN_REASON
        return None

    # ---- search --------------------------------------------------------------------------

    def _hit(self, spec: Any) -> Optional[SearchHit]:
        key = str(spec_attr(spec, "key", "") or "")
        if not key:
            return None
        section = self.section_for_key(key)
        if section is None:
            return None
        return SearchHit(key=key, spec=spec, section_id=section.id, section_title=section.title, group=section.group)

    def search(self, query: str, limit: int = SEARCH_LIMIT) -> list[SearchHit]:
        """Schema search (label, help, key, env names, section path), grouped by section, at most ``limit``."""
        text = (query or "").strip()
        if not text or self.module is None:
            return []
        specs: list[Any] = []
        func = getattr(self.module, "search", None)
        if func is not None:
            try:
                specs = list(func(text))
            except Exception:
                log.exception("settings_schema.search() failed")
                specs = []
        if not specs:
            specs = self._fallback_search(text)
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for spec in specs:
            hit = self._hit(spec)
            if hit is None or hit.key in seen:
                continue
            seen.add(hit.key)
            hits.append(hit)
            if len(hits) >= limit:
                break
        return hits

    def _fallback_search(self, text: str) -> list[Any]:
        needle = text.casefold()
        scored: list[tuple[int, int, Any]] = []
        for index, spec in enumerate(self.all_specs()):
            key = str(spec_attr(spec, "key", ""))
            label = label_for(spec)
            section = self.section_for_key(key)
            fields = [
                (0, key.casefold()),
                (1, label.casefold()),
                (2, " ".join(env_names(spec)).casefold()),
                (3, str(spec_attr(spec, "tooltip", "")).casefold()),
                (4, f"{section.group} {section.title}".casefold() if section is not None else ""),
            ]
            best = None
            for rank, value in fields:
                if value == needle:
                    best = rank * 3
                    break
                if value.startswith(needle):
                    best = rank * 3 + 1 if best is None else min(best, rank * 3 + 1)
                elif needle in value:
                    best = rank * 3 + 2 if best is None else min(best, rank * 3 + 2)
            if best is not None:
                scored.append((best, index, spec))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [spec for _score, _index, spec in scored]
