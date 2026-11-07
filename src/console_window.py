#console_window.py
"""Console window of the compiled Windows builds, opened at runtime only when it is wanted.

Every Windows spec builds the exe without a console (``console=False``). The GUI process opens
one itself with AllocConsole (owner request, 2026-11-07):

* only when Other Settings > Debug Controls > "Show CMD" (``show_cmd``) is on, from startup or as soon
  as it is switched on. Manga runs no longer open it (owner, 2026-11-08: "the user doesn't really need
  to see the CMD, if they want to, they can enable the CMD toggle in other settings").

Once opened the console is kept and only hidden / shown again, because worker processes started
meanwhile may be attached to it. Python's ``print`` and the C-level output of native libraries
(ONNX Runtime, CUDA) both reach it: ``startup`` puts a stand-in stream on ``sys.stdout`` /
``sys.stderr`` (windowed builds start with them ``None``) that writes nowhere until a console is
opened and to the console after, so streams captured by reference before that (logging handlers,
redirect/tee wrappers) follow it; the std handles, CRT file descriptors 1/2 and the C ``stdout`` /
``stderr`` streams are pointed at CONOUT$ too. Ctrl+C and Ctrl+Break in the console are ignored.
Closing a console window ends every process attached to it, so the console's Close button is
removed (``_disable_close_button``); "Show CMD" off hides it instead.

Only frozen Windows builds are affected: running from source keeps the terminal it was started
from, and "Show CMD" does nothing there. Helper processes (the same exe started with a ``--run-...``
or another helper flag) never open, hide or show a console. The exe is a windowed program, so its
multiprocessing children (the manga inpainting worker, process pools) do not inherit the GUI's
console: ``attach_worker`` attaches such a child to its parent's console when the parent has one,
so their output shows there too, as it did in the old console builds.
"""

import io
import json
import os
import sys
import threading

SW_HIDE = 0
SW_SHOWNOACTIVATE = 4
GW_OWNER = 4

CTRL_C_EVENT = 0
CTRL_BREAK_EVENT = 1
CTRL_CLOSE_EVENT = 2
SC_CLOSE = 0xF060
MF_BYCOMMAND = 0

CONFIG_KEY = "show_cmd"
CONSOLE_TITLE = "Glossarion console"

# argv flags that start the exe as a helper process instead of the GUI (translator_gui.py,
# multiprocessing). Flags starting with --run- are helpers too.
_HELPER_FLAGS = (
    "--autharena-qt-browser",
    "--authza-captcha",
    "--authnd-mint-token",
    "--gemini-free-search",
    "--multiprocessing-fork",
)

_lock = threading.RLock()
_state = {"started": False, "show_cmd": False, "allocated": False}
_streams = []          # the stand-in streams startup put on sys.stdout / sys.stderr
_ctrl_handler = None   # keeps the ctypes callback alive
_libs = {}


def _win(name):
    """A private ctypes view of kernel32 / user32, so the argtypes set here never change the
    shared ``ctypes.windll`` functions other modules call."""
    lib = _libs.get(name)
    if lib is None:
        import ctypes
        lib = _libs[name] = ctypes.WinDLL(name + ".dll", handle=getattr(ctypes.windll, name)._handle)
    return lib


class _ConsoleStream(io.TextIOBase):
    """``sys.stdout`` / ``sys.stderr`` stand-in: drops text until a console is opened, then
    writes it there. Keeping one object lets references captured earlier follow the console."""

    def __init__(self, name):
        super().__init__()
        self._name = name
        self.target = None

    @property
    def name(self):
        return self._name

    @property
    def encoding(self):
        return getattr(self.target, "encoding", None) or "utf-8"

    @property
    def errors(self):
        return getattr(self.target, "errors", None) or "replace"

    @property
    def buffer(self):
        target = self.target
        if target is not None and hasattr(target, "buffer"):
            return target.buffer
        return io.BytesIO()

    def writable(self):
        return True

    def write(self, text):
        target = self.target
        if target is not None:
            try:
                target.write(text)
            except Exception:
                pass
        return len(text)

    def flush(self):
        target = self.target
        if target is not None:
            try:
                target.flush()
            except Exception:
                pass

    def isatty(self):
        target = self.target
        try:
            return bool(target is not None and target.isatty())
        except Exception:
            return False

    def fileno(self):
        target = self.target
        if target is None:
            raise io.UnsupportedOperation("no console")
        return target.fileno()

    def reconfigure(self, **kwargs):
        # the console stream is UTF-8 already (the app asks for UTF-8 at startup)
        return None


def _console_hwnd():
    """The window showing this process's console, or 0 when it has none (or it isn't Windows).

    Under Windows Terminal (the Windows 11 default) GetConsoleWindow returns an invisible
    "PseudoConsoleWindow" that ignores ShowWindow; the visible window is its owner, the
    Terminal window (checked on Windows build 26200, Terminal 1.24).
    """
    if sys.platform != "win32":
        return 0
    try:
        import ctypes
        from ctypes import wintypes
        kernel32, user32 = _win("kernel32"), _win("user32")
        kernel32.GetConsoleWindow.restype = wintypes.HWND
        user32.GetWindow.restype = wintypes.HWND
        user32.GetWindow.argtypes = (wintypes.HWND, wintypes.UINT)
        user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
        hwnd = kernel32.GetConsoleWindow()
        if not hwnd:
            return 0
        name = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, name, 64)
        if name.value == "PseudoConsoleWindow":
            owner = user32.GetWindow(hwnd, GW_OWNER)
            return int(owner or 0)
        return int(hwnd)
    except Exception:
        return 0


def _show(cmd):
    hwnd = _console_hwnd()
    if not hwnd:
        return False
    try:
        from ctypes import wintypes
        _win("user32").ShowWindow(wintypes.HWND(hwnd), cmd)
        return True
    except Exception:
        return False


def _on_console_event(event):
    """Console control handler: Ctrl+C / Ctrl+Break typed in the console are ignored (they would
    otherwise end the app). A close event is left to the system, which ends every process
    attached to the console whatever a handler answers (tested 2026-11-07, also after
    FreeConsole in the handler); ``_disable_close_button`` keeps the user from sending one."""
    return event in (CTRL_C_EVENT, CTRL_BREAK_EVENT)


def _disable_close_button(hwnd):
    """Remove Close from a classic console window's system menu: that greys out its X button
    and Alt+F4. A Windows Terminal tab cannot be protected like this."""
    if not hwnd:
        return False
    try:
        import ctypes
        from ctypes import wintypes
        user32 = _win("user32")
        user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
        name = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(wintypes.HWND(hwnd), name, 64)
        if name.value != "ConsoleWindowClass":
            return False
        user32.GetSystemMenu.restype = wintypes.HMENU
        user32.GetSystemMenu.argtypes = (wintypes.HWND, wintypes.BOOL)
        user32.DeleteMenu.argtypes = (wintypes.HMENU, wintypes.UINT, wintypes.UINT)
        menu = user32.GetSystemMenu(wintypes.HWND(hwnd), False)
        return bool(menu and user32.DeleteMenu(menu, SC_CLOSE, MF_BYCOMMAND))
    except Exception:
        return False


def _install_ctrl_handler(kernel32):
    global _ctrl_handler
    if _ctrl_handler is not None:
        return
    import ctypes
    from ctypes import wintypes
    handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    _ctrl_handler = handler_type(lambda event: bool(_on_console_event(int(event))))
    kernel32.SetConsoleCtrlHandler(_ctrl_handler, True)


def _loaded_ucrt(kernel32):
    """The UCRT this process already uses (by module handle: a frozen build's ctypes looks a bare
    "ucrtbase" up in the bundle folder and fails)."""
    import ctypes
    from ctypes import wintypes
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
    handle = kernel32.GetModuleHandleW("ucrtbase.dll")
    if not handle:
        raise OSError("ucrtbase.dll is not loaded")
    return ctypes.CDLL("ucrtbase.dll", handle=handle)


def _redirect_c_stdio(kernel32):
    """Point the std handles, CRT fds 1/2 and C stdout/stderr at the new console."""
    import ctypes
    import msvcrt
    from ctypes import wintypes
    GENERIC_READ, GENERIC_WRITE = 0x80000000, 0x40000000
    FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING = 1, 2, 3
    STD_OUTPUT_HANDLE, STD_ERROR_HANDLE = -11, -12
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    kernel32.SetStdHandle.argtypes = (wintypes.DWORD, wintypes.HANDLE)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    invalid = wintypes.HANDLE(-1).value
    # The C stdout/stderr streams of the shared UCRT start unattached in a windowed process.
    # Reopened first: freopen closes the stream's old descriptor, which can be fd 1 / 2 itself.
    try:
        ucrt = _loaded_ucrt(kernel32)
        ucrt.__acrt_iob_func.restype = ctypes.c_void_p
        ucrt.__acrt_iob_func.argtypes = (ctypes.c_uint,)
        ucrt._wfreopen.restype = ctypes.c_void_p
        ucrt._wfreopen.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_void_p)
        ucrt.setvbuf.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_size_t)
        for index in (1, 2):
            stream = ucrt.__acrt_iob_func(index)
            if ucrt._wfreopen("CONOUT$", "w", stream):
                ucrt.setvbuf(stream, None, 4, 0)   # _IONBF, like console stdio
    except Exception:
        pass
    for std, fd in ((STD_OUTPUT_HANDLE, 1), (STD_ERROR_HANDLE, 2)):
        handle = kernel32.CreateFileW("CONOUT$", GENERIC_READ | GENERIC_WRITE,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None)
        if not handle or handle == invalid:
            continue
        try:
            new_fd = msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_TEXT)
        except OSError:
            kernel32.CloseHandle(handle)
            continue
        try:
            os.dup2(new_fd, fd)
        except OSError:
            kernel32.SetStdHandle(std & 0xFFFFFFFF, handle)   # new_fd stays open, owning it
            continue
        os.close(new_fd)   # closes `handle`; fd 1/2 hold their own duplicate
        kernel32.SetStdHandle(std & 0xFFFFFFFF, msvcrt.get_osfhandle(fd))


def _allocate():
    """Open a console for this process and send stdout / stderr to it. True when it has one."""
    if _state["allocated"]:
        return True
    try:
        kernel32 = _win("kernel32")
        if not kernel32.AllocConsole():
            return False
        _state["allocated"] = True
        try:
            kernel32.SetConsoleTitleW(CONSOLE_TITLE)
        except Exception:
            pass
        _disable_close_button(_console_hwnd())
        _connect(kernel32)
        return True
    except Exception:
        return False


def _connect(kernel32):
    """This process just got a console (opened or attached): send stdout / stderr to it."""
    _install_ctrl_handler(kernel32)
    _redirect_c_stdio(kernel32)
    for stream in _streams:
        try:
            stream.target = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
        except OSError:
            stream.target = None


def _apply():
    """Open / show the console when it is wanted, hide it when it is not."""
    with _lock:
        if not _state["started"]:
            return False
        if _state["show_cmd"]:
            if not _allocate():
                return False
            return _show(SW_SHOWNOACTIVATE)
        if _state["allocated"]:
            return _show(SW_HIDE)
        return False


def managed(argv=None):
    """True when this process should manage the console: a frozen Windows build's GUI process."""
    if not getattr(sys, "frozen", False) or sys.platform != "win32":
        return False
    args = list(sys.argv if argv is None else argv)[1:]
    return not any(a.startswith("--run-") or a in _HELPER_FLAGS for a in args)


def read_show_cmd(config_path=None):
    """The saved "Show CMD" setting (False when the config is missing or unreadable)."""
    try:
        if config_path is None:
            from app_paths import config_file_path
            config_path = config_file_path()
        with open(config_path, "r", encoding="utf-8") as f:
            return bool(json.load(f).get(CONFIG_KEY, False))
    except Exception:
        return False


def _install_streams():
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            stream = _ConsoleStream("<console %s>" % name)
            _streams.append(stream)
            setattr(sys, name, stream)
            if getattr(sys, "__%s__" % name, None) is None:
                setattr(sys, "__%s__" % name, stream)


def startup(config_path=None, argv=None):
    """Called once when the compiled GUI starts (no-op from source and in helper processes):
    installs the stand-in streams and opens the console right away when "Show CMD" is on."""
    with _lock:
        if _state["started"] or not managed(argv):
            return False
        _state["started"] = True
        _install_streams()
        _state["show_cmd"] = read_show_cmd(config_path)
        return _apply()


def _std_output_is_redirected(kernel32):
    """True when the std output handle is a pipe or a file a parent passed on purpose."""
    from ctypes import wintypes
    FILE_TYPE_DISK, FILE_TYPE_PIPE = 1, 3
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.GetStdHandle.argtypes = (wintypes.DWORD,)
    kernel32.GetFileType.argtypes = (wintypes.HANDLE,)
    for std in (-11, -12):
        handle = kernel32.GetStdHandle(std & 0xFFFFFFFF)
        if handle and handle != wintypes.HANDLE(-1).value:
            if kernel32.GetFileType(handle) in (FILE_TYPE_DISK, FILE_TYPE_PIPE):
                return True
    return False


def attach_worker(argv=None):
    """A multiprocessing child of the compiled GUI (``--multiprocessing-fork parent_pid=N``, before
    ``freeze_support`` runs it): attach to the parent's console when the parent has opened one, so
    the child's print and native output show there. Never opens, hides or shows a console, and
    leaves a child whose output a parent redirected to a pipe or file alone."""
    if not getattr(sys, "frozen", False) or sys.platform != "win32":
        return False
    args = list(sys.argv if argv is None else argv)[1:]
    if "--multiprocessing-fork" not in args:
        return False
    parent = 0
    for arg in args:
        if arg.startswith("parent_pid="):
            try:
                parent = int(arg.split("=", 1)[1])
            except ValueError:
                parent = 0
    if parent <= 0:
        return False
    with _lock:
        if _state["allocated"]:
            return True
        try:
            kernel32 = _win("kernel32")
            if kernel32.GetConsoleWindow() or _std_output_is_redirected(kernel32):
                return False
            from ctypes import wintypes
            kernel32.AttachConsole.argtypes = (wintypes.DWORD,)
            if not kernel32.AttachConsole(parent):
                return False
            _state["allocated"] = True
            _install_streams()
            _connect(kernel32)
            return True
        except Exception:
            return False


def set_show_cmd(enabled):
    """"Show CMD" was toggled: show (opening it if needed) or hide the console right away."""
    with _lock:
        _state["show_cmd"] = bool(enabled)
        return _apply()

