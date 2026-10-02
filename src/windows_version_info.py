"""Windows executable metadata shared by the PyInstaller build specs.

These file details do not replace Authenticode signing or establish a trusted
publisher for Smart App Control. Sign the finished executable separately.
"""

import sys

from app_version import APP_VERSION


APP_PUBLISHER = "shirochi-stack"


def get_windows_version_info(app_name):
    """Return the Windows version resource, or None on other platforms."""
    if sys.platform != "win32":
        return None

    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo,
        StringFileInfo,
        StringStruct,
        StringTable,
        VarFileInfo,
        VarStruct,
        VSVersionInfo,
    )

    parts = tuple(int(part) for part in APP_VERSION.split("."))
    version = (parts + (0, 0, 0, 0))[:4]
    return VSVersionInfo(
        ffi=FixedFileInfo(
            filevers=version,
            prodvers=version,
            mask=0x3F,
            flags=0,
            OS=0x40004,
            fileType=0x1,
            subtype=0,
            date=(0, 0),
        ),
        kids=[
            StringFileInfo([
                StringTable("040904B0", [
                    StringStruct("CompanyName", APP_PUBLISHER),
                    StringStruct("FileDescription", app_name),
                    StringStruct("FileVersion", APP_VERSION),
                    StringStruct("InternalName", app_name),
                    StringStruct("OriginalFilename", app_name + ".exe"),
                    StringStruct("ProductName", "Glossarion"),
                    StringStruct("ProductVersion", APP_VERSION),
                ]),
            ]),
            VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
        ],
    )
