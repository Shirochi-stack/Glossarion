"""Glossarion mobile app package (Flet 1.0.3).

Importing this package is cheap and side-effect free: it never imports Flet or
any backend module. ``runtime_bootstrap.bootstrap()`` must run before anything
imports the backend (see ``app/main.py``).
"""

APP_ID = "com.glossarion.app"
APP_NAME = "Glossarion"

DEEP_LINK_SCHEME = "glossarion"
DEEP_LINK_HOST = "app"
OAUTH_RETURN_URL = f"{DEEP_LINK_SCHEME}://{DEEP_LINK_HOST}/oauth/return"
SELFTEST_ROUTE = "/__selftest__"

# Prefix for iOS BGContinuedProcessingTask identifiers
# (Info.plist BGTaskSchedulerPermittedIdentifiers = ["com.glossarion.app.job.*"]).
JOB_TASK_ID_PREFIX = f"{APP_ID}.job."

__all__ = [
    "APP_ID",
    "APP_NAME",
    "DEEP_LINK_SCHEME",
    "DEEP_LINK_HOST",
    "OAUTH_RETURN_URL",
    "SELFTEST_ROUTE",
    "JOB_TASK_ID_PREFIX",
]
