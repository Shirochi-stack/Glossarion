"""One model's thinking effort as a single "Off -> max" list (hotfix 5, the composer's mini model picker).

Owner: "Include the thinking efforts associated with that model. Synced to the model picker settings menu.
With drop down going from off to whatever that model supports. Fallback to openai compatible effort
dropdown for unknown thinking routes and add the toggle ... called 'pass thinking to unknown routes'. Keep
this compact. Effort dropdown + pass thinking to unknown routes toggle".

``effort_control(model, effective)`` reads the global keys the model sheet's "Thinking & effort" tiles and
Settings › Thinking & reasoning show (``model_catalog.THINKING_FIELDS``) and folds a family's enable switch
and its effort into one list; a choice's ``writes`` are what a pick stores (the caller writes them through
``setting_writes.write_settings``, so the full sheet and Settings show the same values). What a model
supports follows the backend's own rules. Every list starts with Off: the family's thinking switch turned
off (v9.15.7 hotfix 6, owner: "should've included off, which just maps to the thinking toggle state being off"),
also for the models that cannot send a "none" effort.

* GPT family (OpenAI, ChatGPT, OpenRouter, NIM, OpenCode): Off · Low · Medium · High · Extra high · Max
  (``enable_gpt_thinking`` + ``gpt_effort``); a stored ``none`` shows as Low on models that cannot send it
  (GPT-6: ``reasoning_compatibility.NO_NONE_SUPPORT``, the backend sends ``low``).
* Gemini 3 and later (``gemini_policy.uses_gemini_thinking_level``): Off · Minimal · Low · Medium · High
  (``thinking_level``; Pro has no Minimal). Gemini 2.x / Gemma: Off · Dynamic · token budgets
  (``thinking_budget``).
* Claude: adaptive models (``temperature_compatibility.claude_uses_adaptive_thinking``, or Force adaptive
  on) Off · Low ... Max (``anthropic_effort``); older ones Off · token budgets
  (``anthropic_thinking_budget``); both behind ``enable_anthropic_thinking``.
* DeepSeek / Chutes: Off · Low · High · Max (``enable_deepseek_thinking`` + ``deepseek_effort``).
* Any other LLM route ("unknown"): the GPT keys, which the backend sends to OpenAI-compatible routes only
  while ``pass_thinking_all_openai`` ("Pass thinking to unknown routes") is on, so that toggle comes with it.
* Translation services without a model to think (DeepL, Google Translate) and routes that cannot run on
  mobile have no effort.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from glossarion_mobile.services import model_catalog as mc

__all__ = [
    "EffortChoice",
    "EffortControl",
    "PASS_UNKNOWN_KEY",
    "PASS_UNKNOWN_LABEL",
    "effort_control",
]

PASS_UNKNOWN_KEY = "pass_thinking_all_openai"
PASS_UNKNOWN_LABEL = "Pass thinking to unknown routes"
NO_THINKING_PROVIDERS = frozenset({"deepl", "google-translate", "google-translate-free"})

LEVEL_LABELS = {"minimal": "Minimal", "low": "Low", "medium": "Medium", "high": "High", "xhigh": "Extra high",
                "max": "Max"}
GPT_LEVELS = ("low", "medium", "high", "xhigh", "max")
GEMINI_LEVELS = ("minimal", "low", "medium", "high")
ANTHROPIC_LEVELS = ("low", "medium", "high", "xhigh", "max")
DEEPSEEK_LEVELS = ("low", "high", "max")
#: token budgets offered for budget-based models (a stored budget that is none of them is listed too)
GEMINI_BUDGETS = ((1024, "Low"), (8192, "Medium"), (24576, "High"))
ANTHROPIC_BUDGETS = ((4096, "Low"), (10000, "Medium"), (32000, "High"))
#: reasoning_compatibility.NO_NONE_SUPPORT when the backend is not importable (host tests without src/)
_NO_NONE_FALLBACK = (r"(?:^|/)gpt-?6(?:[.-]|$)",)


@dataclass(frozen=True)
class EffortChoice:
    value: str  # "off", a level ("low" ...), "dynamic" or "budget:<tokens>"
    label: str
    writes: tuple = ()  # ((key, value), ...) a pick stores


@dataclass(frozen=True)
class EffortControl:
    family: str  # "gpt" / "gemini" / "anthropic" / "deepseek" (the keys it writes)
    choices: tuple
    value: str  # the current choice
    unknown: bool = False  # an unknown route: the GPT keys plus the "Pass thinking to unknown routes" toggle
    pass_unknown: bool = False  # pass_thinking_all_openai (meaningful for unknown routes)

    def choice(self, value: Any) -> Optional[EffortChoice]:
        for choice in self.choices:
            if choice.value == value:
                return choice
        return None

    @property
    def title(self) -> str:
        return "Effort"


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bare(model: str) -> str:
    """The model name without route prefixes (``authgem/gemini-3-pro`` -> ``gemini-3-pro``)."""
    return str(model or "").strip().lower().rsplit("/", 1)[-1]


def _no_none(model: str) -> bool:
    try:
        from reasoning_compatibility import NO_NONE_SUPPORT  # shared backend
    except Exception:
        patterns: tuple = _NO_NONE_FALLBACK
    else:
        patterns = tuple(NO_NONE_SUPPORT)
    name = str(model or "").strip().lower()
    return any(re.search(pattern, name) for pattern in patterns)


def _uses_level(model: str) -> bool:
    try:
        from gemini_policy import uses_gemini_thinking_level
    except Exception:
        match = re.search(r"gemini-(\d+)", model)
        return bool(match and int(match.group(1)) >= 3)
    return bool(uses_gemini_thinking_level(model))


def _adaptive(model: str) -> bool:
    try:
        from temperature_compatibility import claude_uses_adaptive_thinking
    except Exception:
        return False
    return bool(claude_uses_adaptive_thinking(model))


def _off(key: str) -> EffortChoice:
    return EffortChoice("off", "Off", ((key, False),))


def _levels(enable_key: str, effort_key: str, levels: tuple) -> list:
    return [EffortChoice(level, LEVEL_LABELS.get(level, level.title()), ((enable_key, True), (effort_key, level)))
            for level in levels]


def _budgets(enable_key: str, budget_key: str, presets: tuple, current: Optional[int]) -> list:
    choices = [EffortChoice(f"budget:{tokens}", f"{label} · {tokens:,} tokens", ((enable_key, True), (budget_key, tokens)))
               for tokens, label in presets]
    if current is not None and current > 0 and all(tokens != current for tokens, _label in presets):
        choices.append(EffortChoice(f"budget:{current}", f"{current:,} tokens", ((enable_key, True), (budget_key, current))))
        choices.sort(key=lambda c: int(c.value.split(":", 1)[1]))
    return choices


def _gpt(model: str, effective: Callable[[str], Any], *, unknown: bool) -> EffortControl:
    enabled = _bool(effective("enable_gpt_thinking"))
    effort = str(effective("gpt_effort") or "medium").strip().lower()
    choices = [_off("enable_gpt_thinking")] + _levels("enable_gpt_thinking", "gpt_effort", GPT_LEVELS)
    if not enabled:
        value = "off"
    elif effort in GPT_LEVELS:
        value = effort
    else:  # "none": what the backend sends
        value = "low" if _no_none(model) else "off"
    return EffortControl("gpt", tuple(choices), value, unknown=unknown,
                         pass_unknown=_bool(effective(PASS_UNKNOWN_KEY)) if unknown else False)


def _gemini(model: str, effective: Callable[[str], Any]) -> EffortControl:
    name = _bare(model)
    enabled = _bool(effective("enable_gemini_thinking"))
    if _uses_level(name):
        pro = "pro" in name and "flash" not in name
        levels = GEMINI_LEVELS[1:] if pro else GEMINI_LEVELS
        level = str(effective("thinking_level") or "high").strip().lower()
        if not enabled:
            value = "off"
        else:
            value = level if level in levels else ("low" if level == "minimal" else "high")
        return EffortControl("gemini", tuple([_off("enable_gemini_thinking")]
                                             + _levels("enable_gemini_thinking", "thinking_level", levels)), value)
    budget = _int(effective("thinking_budget"), -1)
    choices: list = [_off("enable_gemini_thinking")]
    choices.append(EffortChoice("dynamic", "Dynamic", (("enable_gemini_thinking", True), ("thinking_budget", -1))))
    custom = budget if enabled and budget > 0 else None
    choices.extend(_budgets("enable_gemini_thinking", "thinking_budget", GEMINI_BUDGETS, custom))
    if not enabled or budget == 0:
        value = "off"
    elif budget < 0:
        value = "dynamic"
    else:
        value = f"budget:{budget}"
    return EffortControl("gemini", tuple(choices), value)


def _anthropic(model: str, effective: Callable[[str], Any]) -> EffortControl:
    enabled = _bool(effective("enable_anthropic_thinking"))
    choices: list = [_off("enable_anthropic_thinking")]
    if _adaptive(_bare(model)) or _bool(effective("anthropic_force_adaptive")):
        choices.extend(_levels("enable_anthropic_thinking", "anthropic_effort", ANTHROPIC_LEVELS))
        effort = str(effective("anthropic_effort") or "medium").strip().lower()
        value = effort if enabled and effort in ANTHROPIC_LEVELS else ("off" if not enabled else "medium")
        return EffortControl("anthropic", tuple(choices), value)
    budget = _int(effective("anthropic_thinking_budget"), 10000)
    choices.extend(_budgets("enable_anthropic_thinking", "anthropic_thinking_budget", ANTHROPIC_BUDGETS,
                            budget if budget > 0 else None))
    value = f"budget:{budget}" if enabled and budget > 0 else "off"
    return EffortControl("anthropic", tuple(choices), value)


def _deepseek(effective: Callable[[str], Any]) -> EffortControl:
    enabled = _bool(effective("enable_deepseek_thinking"))
    effort = str(effective("deepseek_effort") or "high").strip().lower()
    choices = [_off("enable_deepseek_thinking")] + _levels("enable_deepseek_thinking", "deepseek_effort",
                                                           DEEPSEEK_LEVELS)
    value = effort if enabled and effort in DEEPSEEK_LEVELS else "off"
    return EffortControl("deepseek", tuple(choices), value)


def effort_control(model: Optional[str], effective: Callable[[str], Any], *, custom_routes: Any = None
                   ) -> Optional[EffortControl]:
    """The Effort list for ``model`` (``effective(key)``: the stored value, else its default, i.e.
    ``MobileConfigStore.effective``); None when the route has nothing to set."""
    model = str(model or "").strip()
    if not model or mc.excluded_route(model):
        return None
    provider = mc.provider_of(model, custom_routes)
    if provider in NO_THINKING_PROVIDERS:
        return None
    family = mc.family_of(model, provider)
    if family == "gemini":
        return _gemini(model, effective)
    if family == "anthropic":
        return _anthropic(model, effective)
    if family == "deepseek":
        return _deepseek(effective)
    return _gpt(model, effective, unknown=family != "gpt")
