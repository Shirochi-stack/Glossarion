# Central application version metadata.
#
# Bump APP_VERSION here; GUI titles, startup logs, splash text, and PyInstaller
# spec APP_NAME values read from this module.

APP_VERSION = "8.7.0"
VERSION_TAG = f"v{APP_VERSION}"
APP_DISPLAY_NAME = f"Glossarion {VERSION_TAG}"
APP_STARTUP_MESSAGE = f"Starting {APP_DISPLAY_NAME}..."
APP_READY_MESSAGE = f"{APP_DISPLAY_NAME} - Ready to use!"
APP_USER_MODEL_ID = f"Glossarion.Translator.{APP_VERSION}"

SPEC_APP_NAMES = {
    "translator.spec": f"Glossarion {VERSION_TAG}",
    "translator_Heavy.spec": f"H_Glossarion Heavy {VERSION_TAG}",
    "translator_lite.spec": f"L_Glossarion_Lite {VERSION_TAG}",
    "translator_lite_linux.spec": f"L_Glossarion_Lite_{VERSION_TAG}_Linux",
    "translator_lite_mac.spec": f"L_Glossarion_Lite_{VERSION_TAG}_MAC",
    "translator_lite_mac_intel.spec": f"L_Glossarion_Lite_{VERSION_TAG}_MAC_Intel",
    "translator_NoCuda.spec": f"N_Glossarion_NoCuda {VERSION_TAG}",
    "translator_TurboLite.spec": f"L_Glossarion_TurboLite {VERSION_TAG}",
}


def get_spec_app_name(spec_path):
    """Return the APP_NAME for a PyInstaller spec path."""
    import os

    return SPEC_APP_NAMES.get(os.path.basename(str(spec_path)), APP_DISPLAY_NAME)
