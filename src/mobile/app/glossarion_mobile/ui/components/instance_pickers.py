"""The multi-instance assignment rows (design §6, ``instance_plan``): the desktop assignment dialog's per-file
"Instance N" combo plus the jobs line ("2 parallel jobs · Instance 1: 1 book · Instance 2: 1 book").

Shared by the Library TranslateSheet and the chat BatchPlanCard: both get ``(instances, assignment)`` from
``LibraryService.instance_assignment`` (``instance_plan.is_multi_instance`` / ``default_assignment``) and Start
queues one job per instance of :meth:`InstancePickers.groups` (``instance_plan.plan``).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import flet as ft

__all__ = ["InstancePickers", "jobs_line"]


def jobs_line(plan: Mapping[int, Sequence[str]]) -> str:
    """"2 parallel jobs · Instance 1: 1 book · Instance 2: 1 book" for ``{instance: [files]}``; "" for {}."""
    if not plan:
        return ""
    parts = [f"Instance {n}: {len(files)} book{'s' if len(files) != 1 else ''}" for n, files in plan.items()]
    lead = f"{len(plan)} parallel jobs" if len(plan) > 1 else f"1 job on Instance {next(iter(plan))}'s keys"
    return " · ".join([lead, *parts])


class InstancePickers:
    """``instances`` / ``assignment`` ({source: instance}) of a multi-instance run over ``sources``, one dropdown per
    source (:meth:`picker`) and the jobs line (:attr:`text`). An empty ``assignment`` is a single-instance run: no
    pickers, the line hidden. ``push(*controls)`` shows a change; ``on_change(assignment)`` follows each pick."""

    def __init__(self, instances: Sequence[int], assignment: Mapping[str, int], sources: Sequence[str], *,
                 picker_key: str, text_key: str, push: Optional[Callable[..., Any]] = None,
                 on_change: Optional[Callable[[dict], Any]] = None) -> None:
        self.instances: list = list(instances)
        self.assignment: dict = dict(assignment)
        self.sources: list = [s for s in sources if s]
        self.pickers: dict = {}  # source -> Dropdown
        self.picker_key = picker_key
        self.push = push
        self.on_change = on_change
        self.text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                            visible=bool(self.assignment), key=text_key)
        self.refresh(push=False)

    def picker(self, index: int, source: str) -> ft.Dropdown:
        """A file row's instance dropdown (the desktop assignment dialog's per-file combo)."""
        picker = ft.Dropdown(value=str(self.assignment[source]), dense=True, width=150,
                             options=[ft.DropdownOption(key=str(n), text=f"Instance {n}") for n in self.instances],
                             on_select=lambda e, s=source: self.set(s, e.control.value),
                             key=f"{self.picker_key}-{index}")
        self.pickers[source] = picker
        return picker

    def groups(self) -> dict:
        """{instance: [sources]} of the current assignment (``instance_plan.plan``); {} for a single-instance run."""
        if not self.assignment:
            return {}
        import instance_plan

        return instance_plan.plan(self.sources, self.assignment)

    def set(self, source: str, value: Any) -> int:
        """A picker changed: ``source`` runs in instance ``value`` (an instance with keys; others are refused)."""
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = 0
        if number not in self.instances:
            number = self.assignment.get(source, self.instances[0] if self.instances else 1)
        self.assignment[source] = number
        picker = self.pickers.get(source)
        if picker is not None:
            picker.value = str(number)
        self.refresh()
        if self.on_change is not None:
            self.on_change(dict(self.assignment))
        return number

    def refresh(self, *, push: bool = True) -> None:
        """The jobs line over the files (one line on a phone, so Start stays in view; the key editor explains
        instances)."""
        line = jobs_line(self.groups())
        if line:
            self.text.value = line
        if push and self.push is not None:
            self.push(self.text, *self.pickers.values())
