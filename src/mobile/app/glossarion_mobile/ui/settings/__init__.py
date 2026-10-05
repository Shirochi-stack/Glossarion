"""Schema-driven Settings (UI_SPEC §4.15, §5.7): home, section pages, search, tiles, editors.

``integration.SettingsFeature`` wires ``MobileConfigStore``/``Prefs`` and the
screens into ``GlossarionApp``; everything else here is built from
``settings_schema`` (shared, GUI-free) through ``schema_access.SchemaAccess``.
"""
