"""A bounded, safe-area drawer; only its conversation list scrolls."""

import flet as ft


@ft.control("GlossarionNavigationDrawer")
class NavigationDrawer(ft.NavigationDrawer):
    """Uses the Scaffold's constraints instead of a Python-sized outer ListView."""
