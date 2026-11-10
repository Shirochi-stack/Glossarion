"""Size filters shared by the Windows PyInstaller specs.

Build-time helper only: the specs import it from SPEC_DIR (like
windows_version_info), and nothing in the app imports it, so it is not in any
spec's app_files / app_modules. The rules and the measurements behind them are
in docs/nocuda_bloat_report.md and docs/lite_builds_bloat_report.md.
"""

import glob
import os
import sys


def exclude_windows_system_icu(analysis):
    """Keep Qt's Windows ICU imports from resolving to a copy found on the build PATH."""
    if sys.platform != 'win32':
        return
    system_icu = {'icu.dll', 'icuin.dll', 'icuuc.dll'}
    for field in ('binaries', 'datas'):
        entries = getattr(analysis, field)
        setattr(analysis, field, [entry for entry in entries if slim_norm(entry[0]) not in system_icu])


def msys2_weasyprint_dlls(bin_dir):
    """The MSYS2 DLLs WeasyPrint can load: the PE import closure of the names its
    text/ffi.py dlopen()s, plus libcairo-2 (cairocffi/cairosvg, if present).

    The whole mingw64/bin folder also holds GTK3, librsvg, Tcl/Tk, gcc's isl,
    libpython3.14 and so on (about 23 MB in the exe) that nothing in this
    build loads. Returns None when the closure cannot be computed, so the
    caller falls back to the full glob.
    """
    try:
        from PyInstaller.depend import bindepend
    except Exception:
        return None
    roots = [
        'libgobject-2.0-0.dll', 'libpango-1.0-0.dll', 'libharfbuzz-0.dll',
        'libharfbuzz-subset-0.dll', 'libfontconfig-1.dll', 'libpangoft2-1.0-0.dll',
        'libcairo-2.dll', 'libpixman-1-0.dll',
    ]
    present = {os.path.basename(p).lower(): p for p in glob.glob(os.path.join(bin_dir, '*.dll'))}
    if 'libpango-1.0-0.dll' not in present or 'libgobject-2.0-0.dll' not in present:
        return None
    need = set()
    stack = [r.lower() for r in roots if r.lower() in present]
    try:
        while stack:
            name = stack.pop()
            if name in need:
                continue
            need.add(name)
            for imp_name, _resolved in bindepend.get_imports(present[name]):
                base = os.path.basename(imp_name).lower()
                if base in present and base not in need:
                    stack.append(base)
    except Exception as exc:
        print(f"  WARNING: MSYS2 closure failed ({exc}); shipping the whole folder")
        return None
    return [present[n] for n in sorted(need)]


def slim_norm(dest):
    return str(dest).replace('\\', '/').lower()


def slim_unused(dest):
    n = slim_norm(dest)
    base = n.rsplit('/', 1)[-1]
    # cv2's FFmpeg video I/O plugins: no VideoCapture/VideoWriter anywhere in src
    if n.startswith('cv2/') and base.startswith('opencv_videoio_ffmpeg'):
        return True
    # Left by an onnxruntime-directml wheel; the CPU onnxruntime.dll never loads it
    if n == 'onnxruntime/capi/directml.dll':
        return True
    # A second python3xx.dll inside a package folder (torchvision); the root one is used
    if '/' in n and base.startswith('python3') and base.endswith('.dll'):
        return True
    # Build / test tooling and Application Verifier leftovers
    if n.startswith('torch/bin/protoc') or n in ('vfcompat.dll', 'appverifui.dll'):
        return True
    if n.startswith('pyside6/'):
        rest = n[len('pyside6/'):]
        # WebEngine DevTools inspector frontend and the debug-build resources.
        # CDP over QTWEBENGINE_REMOTE_DEBUGGING (autharena/) is served by the core DLL.
        if rest.startswith('resources/') and ('devtools_resources' in base or '.debug.' in base):
            return True
        # Chromium falls back to en-US for its context-menu / error-page strings
        if rest.startswith('translations/qtwebengine_locales/'):
            return base != 'en-us.pak'
        # Qt's own .qm translations: no QTranslator in the app
        if rest.startswith('translations/') and rest.count('/') == 1 and base.endswith('.qm'):
            return True
        # QML import tree: the app has no QML (WebEngine widgets do not import QML modules)
        if rest.startswith('qml/'):
            return True
    return False


def pyside6_unreachable_qt_dlls(entries):
    """Root PySide6/Qt6*.dll files that no shipped PySide6 module, plugin or
    QtWebEngineProcess.exe links (directly or indirectly). After the qml/ tree is
    gone these are the Quick3D/Charts/Graphs/Location/QuickControls2... DLLs only
    QML plugins used. Returns an empty set if the PE scan fails."""
    try:
        from PyInstaller.depend import bindepend
    except Exception:
        return set()
    root_dlls = {}
    roots = []
    for dest, src, _typ in entries:
        n = slim_norm(dest)
        if not n.startswith('pyside6/') or not n.endswith(('.dll', '.pyd', '.exe')):
            continue
        rest = n[len('pyside6/'):]
        if '/' not in rest and rest.endswith('.dll'):
            root_dlls[rest] = src
        else:
            roots.append(src)
    need = set()
    stack = list(roots)
    seen = set()
    try:
        while stack:
            path = stack.pop()
            key = os.path.normcase(str(path))
            if key in seen or not os.path.exists(path):
                continue
            seen.add(key)
            for imp_name, _resolved in bindepend.get_imports(path):
                base = os.path.basename(imp_name).lower()
                if base in root_dlls and base not in need:
                    need.add(base)
                    stack.append(root_dlls[base])
    except Exception as exc:
        print(f"  WARNING: PySide6 closure failed ({exc}); keeping all Qt DLLs")
        return set()
    return {name for name in root_dlls if name.startswith('qt6') and name not in need}


def pyside6_unused_bindings(names):
    """A predicate for PySide6 files a build never loads, from bundle-relative
    names such as 'QtOpenGL.pyd' or 'plugins/qmltooling/' (a trailing '/'
    matches the whole folder). Use it as the `extra` of slim_analysis."""
    wanted = {name.lower() for name in names}
    folders = tuple(name for name in wanted if name.endswith('/'))

    def unused(dest):
        n = slim_norm(dest)
        if not n.startswith('pyside6/'):
            return False
        rest = n[len('pyside6/'):]
        return rest in wanted or (bool(folders) and rest.startswith(folders))

    return unused


def slim_analysis(a, label, extra=None):
    """Apply slim_unused (plus an optional per-spec `extra` predicate on the
    bundle-relative name) to a.binaries / a.datas, then drop the root Qt6 DLLs
    that nothing left links."""
    print("\n" + "=" * 60)
    print(f"SLIM: REMOVING FILES NO {label} FEATURE LOADS...")
    print("=" * 60)
    before = (len(a.binaries), len(a.datas))

    def drop(dest):
        return slim_unused(dest) or (extra is not None and extra(dest))

    a.binaries = [e for e in a.binaries if not drop(e[0])]
    a.datas = [e for e in a.datas if not drop(e[0])]
    qt_unreachable = pyside6_unreachable_qt_dlls(list(a.binaries) + list(a.datas))
    if qt_unreachable:
        print(f"  Qt DLLs only QML plugins linked ({len(qt_unreachable)}): {sorted(qt_unreachable)}")
        a.binaries = [e for e in a.binaries if slim_norm(e[0])[len('pyside6/'):] not in qt_unreachable
                      or not slim_norm(e[0]).startswith('pyside6/')]
    print(f"  Binaries: {before[0]} -> {len(a.binaries)}, data files: {before[1]} -> {len(a.datas)}")
