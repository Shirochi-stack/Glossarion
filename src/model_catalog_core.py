"""Model catalog edits shared by the desktop Model Manager and the mobile app.

Glossarion mobile rewrite, milestone U4 (shared-core design section 3.6 "model_catalog_core").
The data side of TranslatorGUI's model catalog handlers, moved out verbatim; the desktop
methods call these functions and keep their widget work (combo / completer refresh, list
widget icons, poll border, message boxes):

* **Tombstones** (``model_manager_removed_models``): ``restore_removed_models`` is the body of
  ``_restore_removed_model_choices`` (an explicit re-add or a user-triggered successful poll
  clears exact deletion tombstones, saves, and rolls back when the save fails);
  ``model_order_removed_keys`` / ``apply_model_order`` / ``save_model_order`` are the data
  part of ``_save_model_order`` (the Model Manager remembers models taken out of the list so
  catalog merges never resurrect them).
* **Poll markers** (seven-day "✓ polled" confirmations): ``polled_models_by_provider`` /
  ``polled_model_keys`` (``_ensure_polled_model_marker_state``), ``model_poll_marker``
  (``_apply_polled_model_icons``), ``merge_polled_provider_models`` (a successful provider
  catalog replaces that provider's confirmations; failures keep unexpired ones).
* **Provider refresh** (``_apply_provider_model_catalog_refresh``): ``confirmed_catalog_models``,
  ``catalog_display_models``, ``online_catalog_summary``, ``catalog_skip_counts``,
  ``manager_poll_models``, ``poll_status_text``, ``auto_poll_message``; ``apply_provider_refresh``
  composes them in the desktop order for a config-only caller (mobile).
* **Custom prefixes** (``_collect_custom_prefix_routes_from_table``): ``custom_prefix_route_from_row``
  per table row and ``validate_custom_prefix_routes`` for already-read rows.

The catalog itself (built-in list, cache, polls, merge) stays in ``model_options``.

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "POLLED_MODEL_TOOLTIP", "EXPLICIT_READD_NOT_SAVED", "EMPTY_MODEL_LIST", "MODEL_LIST_NOT_SAVED",
    "MODEL_LIST_NOT_SAVED_LOG", "MODEL_LIST_SAVED_LOG", "MODEL_ORDER_KEYS", "NO_ONLINE_CATALOG",
    "ConfigSnapshot", "ProviderRefresh",
    "restore_removed_models", "restore_removed_model_choices",
    "model_order_removed_keys", "apply_model_order", "save_model_order",
    "polled_models_by_provider", "polled_model_keys", "polled_key_set", "casefold_model_keys",
    "model_poll_marker", "merge_polled_provider_models",
    "confirmed_catalog_models", "catalog_display_models", "online_catalog_summary", "catalog_skip_counts",
    "known_catalog_keys", "manager_poll_models", "poll_status_text", "auto_poll_message",
    "apply_provider_refresh",
    "custom_prefix_route_from_row", "validate_custom_prefix_routes",
]

POLLED_MODEL_TOOLTIP = "✓ Confirmed by a successful provider poll within the past 7 days"
EXPLICIT_READD_NOT_SAVED = "❌ Explicit model re-add was not saved; config.json write failed"
EMPTY_MODEL_LIST = ("Empty List", "The model list cannot be empty. Add at least one model.")
MODEL_LIST_NOT_SAVED = (
    "Model List Not Saved",
    "Glossarion could not write the model list to config.json. "
    "The manager will remain open so you can retry.",
)
MODEL_LIST_NOT_SAVED_LOG = "❌ Model list was not saved; config.json write failed"
MODEL_LIST_SAVED_LOG = "✓ Model list updated"
NO_ONLINE_CATALOG = "No online catalog responded; using static fallbacks."

#: Config keys a Model Manager save writes (and rolls back when config.json cannot be written).
MODEL_ORDER_KEYS = ('custom_model_list', 'model_manager_removed_models', 'model_mousewheel_locked')


class _Unset:
    def __repr__(self):
        return '<unset>'


_UNSET: Any = _Unset()


# =============================================================================================
# Tombstones
# =============================================================================================

def restore_removed_models(config, model_values, *, add_to_saved=False, save=None, log=None):
    """Clear exact deletion tombstones after an explicit user re-add.

    Passive cache/startup merges still honor ``model_manager_removed_models``. Only a manual
    entry or a user-triggered successful poll calls this path. With *add_to_saved* the
    re-added models are appended to ``custom_model_list`` too. *save* (zero-argument
    callable; desktop: ``save_config(show_message=False)``) persists the change: when it
    returns False or raises, the config is rolled back and *log* gets
    ``EXPLICIT_READD_NOT_SAVED``.

    Returns None when nothing was restored (no model value, *config* not a dict, no tombstone
    matched), False when the save failed, else the remaining tombstone list.
    Moved from TranslatorGUI._restore_removed_model_choices (the desktop method refreshes the
    model pickers afterwards).
    """
    values = []
    seen_values = set()
    for model in model_values or ():
        value = str(model or '').strip()
        key = value.casefold()
        if not value or key in seen_values:
            continue
        values.append(value)
        seen_values.add(key)
    if not values:
        return None

    if not isinstance(config, dict):
        return None
    old_removed = config.get('model_manager_removed_models')
    removed = [
        str(model).strip()
        for model in (old_removed if isinstance(old_removed, list) else [])
        if str(model).strip()
    ]
    restored_keys = {
        model.casefold() for model in removed
        if model.casefold() in seen_values
    }
    if not restored_keys:
        return None

    remaining_removed = [
        model for model in removed if model.casefold() not in restored_keys
    ]
    had_custom = 'custom_model_list' in config
    old_custom = config.get('custom_model_list')
    custom_models = list(old_custom) if isinstance(old_custom, list) else []
    if add_to_saved:
        custom_keys = {
            str(model).strip().casefold()
            for model in custom_models
            if str(model).strip()
        }
        for value in values:
            if (
                value.casefold() in restored_keys
                and value.casefold() not in custom_keys
            ):
                custom_models.append(value)
                custom_keys.add(value.casefold())

    config['model_manager_removed_models'] = remaining_removed
    if add_to_saved:
        config['custom_model_list'] = custom_models

    try:
        saved = save() if callable(save) else True
    except Exception:
        saved = False
    if saved is False:
        if old_removed is None:
            config.pop('model_manager_removed_models', None)
        else:
            config['model_manager_removed_models'] = old_removed
        if add_to_saved:
            if had_custom:
                config['custom_model_list'] = old_custom
            else:
                config.pop('custom_model_list', None)
        if callable(log):
            log(EXPLICIT_READD_NOT_SAVED)
        return False
    return remaining_removed


#: Name the desktop method carries (callers probing for the desktop name find the same function).
restore_removed_model_choices = restore_removed_models


class ConfigSnapshot:
    """Values of some config keys (present or absent) to restore after a failed save."""

    def __init__(self, config, keys):
        self._saved = [(key, key in config, config.get(key)) for key in keys]

    def restore(self, config):
        for key, had, old in self._saved:
            if had:
                config[key] = old
            else:
                config.pop(key, None)


def model_order_removed_keys(new_order, previous_models, removed_models):
    """Tombstones after a Model Manager save: the saved ones plus every model taken out.

    Remember explicit removals so startup/default/online catalog merges can add genuinely
    new entries without resurrecting deleted ones; models in *new_order* are never tombstoned.
    From TranslatorGUI._save_model_order.
    """
    new_keys = {str(model).strip().casefold() for model in new_order}
    removed_keys = {
        str(model).strip().casefold()
        for model in removed_models
        if str(model).strip()
    }
    removed_keys.update(
        str(model).strip().casefold()
        for model in previous_models
        if str(model).strip() and str(model).strip().casefold() not in new_keys
    )
    removed_keys.difference_update(new_keys)
    return removed_keys


def apply_model_order(config, new_order, removed_keys, *, wheel_locked=_UNSET):
    """Write a Model Manager save into *config*; returns the ConfigSnapshot to roll back with.

    ``custom_model_list`` = *new_order*, ``model_manager_removed_models`` = sorted tombstones,
    ``model_mousewheel_locked`` = *wheel_locked* when given (the desktop "Lock mouse wheel").
    From TranslatorGUI._save_model_order.
    """
    snapshot = ConfigSnapshot(config, MODEL_ORDER_KEYS)
    if wheel_locked is not _UNSET:
        config['model_mousewheel_locked'] = wheel_locked
    config['custom_model_list'] = new_order
    config['model_manager_removed_models'] = sorted(removed_keys)
    return snapshot


def save_model_order(config, new_order, previous_models, *, wheel_locked=_UNSET, save=None):
    """A whole Model Manager save on *config*: tombstones, write, save, roll back on failure.

    Returns False for an empty list (``EMPTY_MODEL_LIST``) or a failed *save* (config rolled
    back; ``MODEL_LIST_NOT_SAVED``), else True. The desktop ``_save_model_order`` runs the
    same steps with its widget refreshes between the write and the save.
    """
    new_order = list(new_order)
    if not new_order:
        return False
    removed_keys = model_order_removed_keys(
        new_order, list(previous_models or ()), config.get('model_manager_removed_models', []))
    snapshot = apply_model_order(config, new_order, removed_keys, wheel_locked=wheel_locked)
    saved = save() if callable(save) else True
    if saved is False:
        snapshot.restore(config)
        return False
    return True


# =============================================================================================
# Poll markers
# =============================================================================================

def polled_models_by_provider(catalogs):
    """``{provider: {casefolded model}}`` from model_options.get_current_polled_provider_models()."""
    return {
        str(provider): {
            str(model).casefold() for model in (models or [])
        }
        for provider, models in catalogs.items()
    }


def polled_model_keys(by_provider):
    """model_options.PolledModelKeys over every provider's confirmed models."""
    from model_options import PolledModelKeys

    return PolledModelKeys(
        model
        for models in by_provider.values()
        for model in models
    )


def polled_key_set(by_provider):
    """Plain set of every provider's confirmed (casefolded) models."""
    return {
        model
        for models in by_provider.values()
        for model in models
    }


def casefold_model_keys(polled_model_keys):
    return {
        str(model).casefold() for model in (polled_model_keys or set())
    }


def model_poll_marker(model, polled_model_keys, hide_unpolled=False):
    """``(is_polled, hidden, tooltip)`` of one Model Manager row (exact casefold match).

    From TranslatorGUI._apply_polled_model_icons (*polled_model_keys* already casefolded).
    """
    is_polled = model.casefold() in polled_model_keys
    return (
        is_polled,
        hide_unpolled and not is_polled,
        POLLED_MODEL_TOOLTIP if is_polled else "",
    )


def merge_polled_provider_models(current, provider_models, online):
    """Poll markers after a refresh: a successful catalog fully replaces that provider's
    confirmations (omitted model IDs lose their marker); failures retain unexpired ones."""
    polled_by_provider = dict(current)
    # Failures retain unexpired confirmations. A successful catalog fully
    # replaces that provider's confirmations, removing omitted model IDs.
    for name in online:
        polled_by_provider[name] = {
            str(model).casefold()
            for model in (provider_models.get(name, []) or [])
        }
    return polled_by_provider


# =============================================================================================
# Provider refresh (TranslatorGUI._apply_provider_model_catalog_refresh)
# =============================================================================================

def confirmed_catalog_models(provider_models):
    """``(models, casefolded keys)`` every provider catalog of a refresh returned."""
    confirmed_models = [
        model
        for models in provider_models.values()
        for model in (models or [])
    ]
    confirmed_model_keys = {
        str(model).strip().casefold()
        for model in confirmed_models
        if str(model).strip()
    }
    return confirmed_models, confirmed_model_keys


def catalog_display_models(config, online_models):
    """The model picker list: saved order + new catalog entries, minus tombstones."""
    from model_options import merge_saved_model_options

    custom_models = config.get('custom_model_list')
    removed_models = config.get('model_manager_removed_models', [])
    display_models = merge_saved_model_options(
        custom_models if isinstance(custom_models, list) else None,
        online_models,
        removed_models,
    )
    return display_models


def online_catalog_summary(statuses, provider_models):
    """``(online providers, number of models they returned)``."""
    online = [name for name, status in statuses.items() if str(status).startswith('online')]
    online_model_count = sum(
        len(provider_models.get(name, []) or []) for name in online
    )
    return online, online_model_count


def catalog_skip_counts(statuses):
    """``(providers skipped for missing credentials, providers unavailable)``."""
    credential_skip_count = sum(
        1 for status in statuses.values()
        if 'no provider credential' in str(status)
    )
    unavailable_count = sum(
        1 for status in statuses.values()
        if str(status).startswith('static fallback')
        and 'no provider credential' not in str(status)
    )
    return credential_skip_count, unavailable_count


def known_catalog_keys(online_models):
    """What the Model Manager remembers as catalog (not custom) models after a poll."""
    return {
        str(model).casefold() for model in online_models
    }


def manager_poll_models(previous_models, known_models, removed_models, online_models, *,
                        explicit_poll=False, confirmed_model_keys=()):
    """The Model Manager list after "Poll Providers": the polled catalog minus tombstones,
    then the manager's genuinely custom entries.

    Retains draft deletions the poll did not confirm; confirmed models are explicit re-adds
    when the poll was manual. *known_models*: the casefolded catalog the manager opened with.
    """
    from model_options import merge_saved_model_options

    custom_models = [
        model for model in previous_models
        if str(model).casefold() not in known_models
    ]
    previous_keys = {
        str(model).casefold() for model in previous_models
    }
    removed_keys = {
        str(model).casefold()
        for model in removed_models
    }
    # Retain draft deletions that the current poll did not confirm.
    # Confirmed models are explicit re-adds when this is a manual poll.
    removed_keys.update(known_models - previous_keys)
    if explicit_poll:
        removed_keys.difference_update(confirmed_model_keys)
    refreshed_models = merge_saved_model_options(
        None,
        online_models,
        removed_keys,
    )
    refreshed_keys = {str(model).casefold() for model in refreshed_models}
    refreshed_models.extend(
        model for model in custom_models
        if (
            str(model).casefold() not in refreshed_keys
            and str(model).casefold() not in removed_keys
        )
    )
    return refreshed_models


def poll_status_text(online, online_model_count, total_models, credential_skip_count, unavailable_count):
    """The Model Manager poll status line (empty *online*: ``NO_ONLINE_CATALOG``)."""
    if online:
        return (
            f"✓ {online_model_count} online · {', '.join(sorted(online))}\n"
            f"{total_models} total incl. fallbacks\n"
            f"{credential_skip_count} need credentials · {unavailable_count} unavailable"
        )
    return NO_ONLINE_CATALOG


def auto_poll_message(requested_provider, statuses, provider_models, previous_model_keys):
    """Log line of a provider-scoped (auto) poll: models found and how many are new."""
    status = str(statuses.get(requested_provider, "static fallback (no result)"))
    if status.startswith('online'):
        refreshed_provider_models = list(
            provider_models.get(requested_provider, []) or []
        )
        new_model_keys = set()
        for model in refreshed_provider_models:
            key = str(model).casefold()
            if key not in previous_model_keys:
                new_model_keys.add(key)
        new_model_count = len(new_model_keys)
        new_model_label = (
            "1 new model found"
            if new_model_count == 1
            else f"{new_model_count} new models found"
        )
        return (
            f"✅ Auto-poll complete: {requested_provider} — "
            f"{len(refreshed_provider_models)} models · {new_model_label}"
        )
    return f"⚠️ Auto-poll failed: {requested_provider} — {status}"


@dataclass
class ProviderRefresh:
    """What a completed catalog refresh changes (``apply_provider_refresh``)."""

    applied: bool                                 # False: the result carried no models
    explicit_poll: bool = False
    display_models: List[str] = field(default_factory=list)
    confirmed_models: List[str] = field(default_factory=list)
    confirmed_model_keys: set = field(default_factory=set)
    online: List[str] = field(default_factory=list)
    online_model_count: int = 0
    credential_skip_count: int = 0
    unavailable_count: int = 0
    polled_by_provider: Dict[str, set] = field(default_factory=dict)
    polled_model_keys: frozenset = frozenset()
    requested_provider: Optional[str] = None
    auto_poll_message: Optional[str] = None
    status_text: str = ''
    restored: Any = None                          # restore_removed_models' result (explicit polls)


def apply_provider_refresh(config, result, *, explicit=False, polled_by_provider=None,
                           previous_models=(), save=None, log=None):
    """Apply a model_options catalog refresh result to *config*, in the desktop order.

    An explicit (user-triggered) poll first clears the tombstones of every model a provider
    confirmed (``restore_removed_models``, saved through *save*). Then the picker list is
    rebuilt (saved order + new entries - tombstones) and the poll markers merged with
    *polled_by_provider* (default: the persisted seven-day markers). *previous_models* are the
    models shown before (for the "N new models found" count of a provider-scoped poll).
    """
    explicit_poll = bool(getattr(result, 'restore_removed_models', False)) or bool(explicit)
    online_models = list(getattr(result, 'models', []) or [])
    if not online_models:
        return ProviderRefresh(applied=False, explicit_poll=explicit_poll)

    previous_model_keys = {str(model).casefold() for model in (previous_models or ())}
    provider_models = dict(getattr(result, 'provider_models', {}) or {})
    confirmed_models, confirmed_model_keys = confirmed_catalog_models(provider_models)
    restored = None
    if explicit_poll and confirmed_models:
        restored = restore_removed_models(config, confirmed_models, add_to_saved=False, save=save, log=log)

    display_models = catalog_display_models(config, online_models)
    statuses = dict(getattr(result, 'statuses', {}) or {})
    online, online_model_count = online_catalog_summary(statuses, provider_models)
    requested_provider = getattr(result, 'requested_provider', None)
    if polled_by_provider is None:
        try:
            from model_options import get_current_polled_provider_models

            catalogs = get_current_polled_provider_models()
        except Exception:
            catalogs = {}
        polled_by_provider = polled_models_by_provider(catalogs)
    merged = merge_polled_provider_models(polled_by_provider, provider_models, online)
    credential_skip_count, unavailable_count = catalog_skip_counts(statuses)
    return ProviderRefresh(
        applied=True,
        explicit_poll=explicit_poll,
        display_models=display_models,
        confirmed_models=confirmed_models,
        confirmed_model_keys=confirmed_model_keys,
        online=online,
        online_model_count=online_model_count,
        credential_skip_count=credential_skip_count,
        unavailable_count=unavailable_count,
        polled_by_provider=merged,
        polled_model_keys=polled_model_keys(merged),
        requested_provider=requested_provider,
        auto_poll_message=(auto_poll_message(requested_provider, statuses, provider_models, previous_model_keys)
                           if requested_provider else None),
        status_text=poll_status_text(online, online_model_count, len(online_models),
                                     credential_skip_count, unavailable_count),
        restored=restored,
    )


# =============================================================================================
# Custom prefixes (TranslatorGUI._collect_custom_prefix_routes_from_table)
# =============================================================================================

def _endpoint_type_helpers(is_valid, normalize):
    if is_valid is None or normalize is None:
        from run_env import RunEnvMixin

        is_valid = is_valid or RunEnvMixin._is_valid_custom_prefix_endpoint_type
        normalize = normalize or RunEnvMixin._normalize_custom_prefix_endpoint_type
    return is_valid, normalize


def custom_prefix_route_from_row(row_number, prefix, endpoint_type, routing, seen, *,
                                 is_valid=None, normalize=None):
    """Validate one Custom prefixes row: ``(route, None)``, ``(None, None)`` for a blank row, or
    ``(None, (title, message))`` for the warning the desktop shows.

    *routing* may be a zero-argument callable (the desktop reads the Base URL cell only after
    the endpoint type passed); *seen* collects the lower-cased prefixes already listed.
    """
    is_valid, normalize = _endpoint_type_helpers(is_valid, normalize)
    if not is_valid(endpoint_type):
        return None, (
            "Invalid Endpoint Type",
            f"Endpoint Type on row {row_number} must be an absolute path like "
            "/chat/completions, /v1/ocr, or /v1/custom.",
        )
    endpoint_type = normalize(endpoint_type)
    routing = routing() if callable(routing) else routing

    if not prefix and not routing:
        return None, None
    if not prefix or not routing:
        return None, ("Incomplete Prefix Route",
                      f"Row {row_number} needs both a prefix and Base URL.")

    prefix = prefix.replace('\\', '/').lstrip('/')
    if any(ch.isspace() for ch in prefix):
        return None, ("Invalid Prefix",
                      f"Prefix on row {row_number} cannot contain spaces.")
    if not prefix.endswith('/'):
        prefix = f"{prefix}/"

    routing = routing.rstrip('/')
    if not routing.lower().startswith(('http://', 'https://')):
        return None, ("Invalid Base URL",
                      f"Base URL on row {row_number} must start with http:// or https://.")

    key = prefix.lower()
    if key in seen:
        return None, ("Duplicate Prefix",
                      f"'{prefix}' is already listed.")
    seen.add(key)
    return {
        'prefix': prefix,
        'routing': routing,
        'endpoint_type': endpoint_type,
    }, None


def validate_custom_prefix_routes(rows):
    """Validate edited Custom prefixes rows (``prefix``, ``routing`` or ``base_url``, ``endpoint_type``).

    Returns ``(routes, None)`` or ``(None, (title, message))`` for the first invalid row,
    with the desktop Model Manager's checks and messages.
    """
    routes = []
    seen = set()
    for index, row in enumerate(rows):
        prefix = str(row.get('prefix', '') or '').strip()
        endpoint_type = str(row.get('endpoint_type', '/chat/completions') or '').strip()
        routing = str(row.get('routing', row.get('base_url', '')) or '').strip()
        route, error = custom_prefix_route_from_row(index + 1, prefix, endpoint_type, routing, seen)
        if error is not None:
            return None, error
        if route is not None:
            routes.append(route)
    return routes, None
