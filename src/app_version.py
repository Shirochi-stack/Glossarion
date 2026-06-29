# Central application version metadata.
#
# Bump APP_VERSION here; GUI titles, startup logs, splash text, and PyInstaller
# spec APP_NAME values read from this module.

APP_VERSION = "9.4.0"
VERSION_TAG = f"v{APP_VERSION}"


def get_runtime_package_label(executable_name=None):
    """Return the package/build label visible to users for this runtime."""
    import os
    import sys

    if executable_name is None:
        if not getattr(sys, "frozen", False):
            return ""
        executable_name = os.path.basename(sys.executable)

    exe_name = os.path.splitext(os.path.basename(str(executable_name)))[0].lower()
    normalized = exe_name.replace("-", "_").replace(" ", "_")

    if "omegalite" in normalized:
        return "OmegaLite"
    if "superlite" in normalized:
        return "SuperLite"
    if "turbolite" in normalized:
        return "TurboLite"
    if "nocuda" in normalized or "no_cuda" in normalized or normalized.startswith("n_"):
        return "NoCuda"
    if "heavy" in normalized or normalized.startswith("h_"):
        return "Heavy"
    if "lite" in normalized or normalized.startswith("l_"):
        return "Lite"
    return ""


def get_runtime_app_display_name(executable_name=None):
    """Return the app title, including the package label for bundled builds."""
    package_label = get_runtime_package_label(executable_name)
    if package_label:
        return f"Glossarion {package_label} {VERSION_TAG}"
    return f"Glossarion {VERSION_TAG}"


APP_DISPLAY_NAME = get_runtime_app_display_name()
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
    "translator_lite_mac_NoCuda.spec": f"N_Glossarion_NoCuda_{VERSION_TAG}_MAC",
    "translator_lite_mac_intel_NoCuda.spec": f"N_Glossarion_NoCuda_{VERSION_TAG}_MAC_Intel",
    "translator_NoCuda.spec": f"N_Glossarion_NoCuda {VERSION_TAG}",
    "translator_TurboLite.spec": f"L_Glossarion_TurboLite {VERSION_TAG}",
}


def get_spec_app_name(spec_path):
    """Return the APP_NAME for a PyInstaller spec path."""
    import os

    return SPEC_APP_NAMES.get(os.path.basename(str(spec_path)), APP_DISPLAY_NAME)
