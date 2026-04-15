# android_oauth.py
"""
Android-specific OAuth flow for AuthGPT (ChatGPT Plus/Pro subscription).

On Android, the standard localhost:1455 callback approach does NOT work
because Android's system browser runs in a separate process and cannot
reach a localhost HTTP server running inside the app's Python process.

Two strategies are used instead:

Strategy A (Android): Manual paste-back flow
  - Opens the auth URL in the browser
  - Shows a dialog with a text field where the user can paste the
    auth code from the callback URL
  - No localhost server needed

Strategy B (Desktop fallback): Original localhost:1455 flow

Requires `pyjnius` for Android Intent access.
"""

import os
import sys
import threading
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Reuse the core auth logic from the desktop module
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


def _open_browser_android(url: str) -> bool:
    """Open a URL in the Android system browser via Intent."""
    try:
        from kivy.utils import platform
        if platform != 'android':
            # Fallback to webbrowser on non-Android
            import webbrowser
            webbrowser.open(url)
            return True

        from jnius import autoclass
        Intent = autoclass('android.content.Intent')
        Uri = autoclass('android.net.Uri')
        PythonActivity = autoclass('org.kivy.android.PythonActivity')

        intent = Intent(Intent.ACTION_VIEW)
        intent.setData(Uri.parse(url))
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)

        activity = PythonActivity.mActivity
        activity.startActivity(intent)
        return True
    except Exception as e:
        logger.error("Failed to open browser on Android: %s", e)
        return False


def _is_android():
    """Check if running on Android."""
    try:
        from kivy.utils import platform
        return platform == 'android'
    except ImportError:
        return False


def start_oauth_flow(
    on_success=None,
    on_error=None,
    timeout: int = 300,
    account_id: int = 0,
    system: str = "authgpt",
):
    """
    Start the AuthGPT or AuthGem OAuth flow on Android.

    On Android, this uses a manual paste-back flow because the localhost
    callback server is unreachable from the system browser:

    1. Opens the auth URL in the browser
    2. Shows a dialog with instructions and a text field
    3. After the user logs in, the browser shows a "callback" page
       (which will fail to load since localhost isn't available)
    4. The user copies the full URL from the browser address bar
    5. Pastes it into the dialog text field
    6. The app extracts the auth code and exchanges it for tokens

    Parameters
    ----------
    on_success : callable(tokens_dict)
        Called on the main thread when authentication succeeds.
    on_error : callable(error_message_str)
        Called on the main thread when authentication fails.
    timeout : int
        Maximum seconds to wait for the user to complete login.
    account_id : int
        The multi-account ID slot (e.g., 2 for authgpt2).
    system : str
        "authgpt" or "authgem".
    """
    if _is_android():
        _start_oauth_flow_android(on_success, on_error, timeout, account_id, system)
    else:
        _start_oauth_flow_desktop(on_success, on_error, timeout, account_id, system)


def _start_oauth_flow_android(on_success, on_error, timeout, account_id, system):
    """Android OAuth flow using manual paste-back."""
    try:
        if system == "authgem":
            _android_authgem_flow(on_success, on_error, timeout, account_id)
        else:
            _android_authgpt_flow(on_success, on_error, timeout, account_id)
    except Exception as e:
        import traceback
        msg = f"OAuth setup failed: {e}\n{traceback.format_exc()}"
        logger.error(msg)
        _dispatch_error(on_error, str(e))


def _android_authgpt_flow(on_success, on_error, timeout, account_id):
    """AuthGPT OAuth flow for Android — manual URL paste-back."""
    from authgpt_auth import (
        generate_pkce, build_auth_url, exchange_code_for_tokens,
        get_store, CALLBACK_HOST, CALLBACK_PORT, CALLBACK_PATH,
    )
    import secrets
    from urllib.parse import urlparse, parse_qs

    code_verifier, code_challenge = generate_pkce()
    state = secrets.token_urlsafe(32)
    redirect_uri = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
    auth_url = build_auth_url(code_challenge, state, redirect_uri)

    logger.info(f"Opening browser for ChatGPT Account {account_id} login...")

    # Open the browser
    if not _open_browser_android(auth_url):
        _dispatch_error(on_error,
            "Could not open browser. Please install a web browser and try again.")
        return

    # Show the paste-back dialog on the main thread
    from kivy.clock import Clock

    def _show_dialog(dt):
        _show_paste_dialog(
            system_name=f"ChatGPT{'' if account_id == 0 else f' {account_id}'}",
            on_url_pasted=lambda url: _handle_authgpt_paste(
                url, state, code_verifier, redirect_uri,
                account_id, on_success, on_error,
            ),
            on_cancel=lambda: _dispatch_error(on_error, "Login cancelled by user."),
        )

    Clock.schedule_once(_show_dialog, 0)


def _handle_authgpt_paste(pasted_url, expected_state, code_verifier,
                           redirect_uri, account_id, on_success, on_error):
    """Process the pasted callback URL for AuthGPT."""
    def _worker():
        try:
            from urllib.parse import urlparse, parse_qs
            from authgpt_auth import exchange_code_for_tokens, get_store

            parsed = urlparse(pasted_url.strip())
            qs = parse_qs(parsed.query)

            auth_code = qs.get("code", [None])[0]
            returned_state = qs.get("state", [None])[0]
            error = qs.get("error", [None])[0]

            if error:
                _dispatch_error(on_error, f"OAuth error from server: {error}")
                return
            if not auth_code:
                _dispatch_error(on_error,
                    "Could not find auth code in the pasted URL.\n"
                    "Make sure you copy the FULL URL from the browser address bar "
                    "after logging in (it should contain '?code=...').")
                return
            if returned_state != expected_state:
                _dispatch_error(on_error,
                    "OAuth state mismatch — the URL may be from a different login attempt. "
                    "Please try logging in again.")
                return

            # Exchange code for tokens
            logger.info("Exchanging auth code for tokens...")
            tokens = exchange_code_for_tokens(auth_code, code_verifier, redirect_uri)

            # Save tokens
            store = get_store(account_id)
            store.save_tokens(tokens)
            logger.info("AuthGPT tokens saved successfully.")

            if on_success:
                try:
                    from kivy.clock import Clock
                    Clock.schedule_once(lambda dt: on_success(tokens), 0)
                except ImportError:
                    on_success(tokens)

        except Exception as e:
            import traceback
            msg = f"Token exchange failed: {e}"
            logger.error(f"{msg}\n{traceback.format_exc()}")
            _dispatch_error(on_error, msg)

    threading.Thread(target=_worker, daemon=True).start()


def _android_authgem_flow(on_success, on_error, timeout, account_id):
    """AuthGem OAuth flow for Android — same paste-back approach."""
    try:
        import authgem_auth

        # AuthGem's run_oauth_flow also opens a browser + localhost callback.
        # We need to intercept the browser open and do paste-back instead.
        auth_url = None

        # Monkey-patch webbrowser.open to capture the URL
        import webbrowser
        original_open = webbrowser.open

        def _capture_url(url, *args, **kwargs):
            nonlocal auth_url
            auth_url = url
            logger.info(f"Captured AuthGem auth URL: {url}")
            _open_browser_android(url)
            return True

        webbrowser.open = _capture_url

        # Try to get the auth URL from AuthGem by starting the flow
        # but we can't use localhost, so we just open the browser
        # and do the paste-back approach
        try:
            # AuthGem uses a similar PKCE flow — get the auth URL
            if hasattr(authgem_auth, 'build_auth_url') and hasattr(authgem_auth, 'generate_pkce'):
                code_verifier, code_challenge = authgem_auth.generate_pkce()
                import secrets
                state = secrets.token_urlsafe(32)
                redirect_uri = authgem_auth.REDIRECT_URI if hasattr(authgem_auth, 'REDIRECT_URI') else "http://localhost:1455/auth/callback"
                auth_url = authgem_auth.build_auth_url(code_challenge, state, redirect_uri)
            else:
                # Start the flow to capture the URL via our monkey-patched webbrowser.open
                # Run in timeout-limited thread
                flow_thread = threading.Thread(
                    target=lambda: authgem_auth.run_oauth_flow(timeout=5),
                    daemon=True
                )
                flow_thread.start()
                flow_thread.join(timeout=3)  # Give it time to open browser

        finally:
            webbrowser.open = original_open

        if not auth_url:
            _dispatch_error(on_error, "Could not determine AuthGem login URL.")
            return

        from kivy.clock import Clock

        def _show_dialog(dt):
            _show_paste_dialog(
                system_name=f"AuthGem{'' if account_id == 0 else f' {account_id}'}",
                on_url_pasted=lambda url: _handle_authgem_paste(
                    url, account_id, on_success, on_error,
                ),
                on_cancel=lambda: _dispatch_error(on_error, "Login cancelled by user."),
            )

        Clock.schedule_once(_show_dialog, 0)

    except Exception as e:
        import traceback
        msg = f"AuthGem login setup failed: {e}"
        logger.error(f"{msg}\n{traceback.format_exc()}")
        _dispatch_error(on_error, msg)


def _handle_authgem_paste(pasted_url, account_id, on_success, on_error):
    """Process the pasted callback URL for AuthGem."""
    def _worker():
        try:
            import authgem_auth
            from urllib.parse import urlparse, parse_qs

            parsed = urlparse(pasted_url.strip())
            qs = parse_qs(parsed.query)
            auth_code = qs.get("code", [None])[0]

            if not auth_code:
                _dispatch_error(on_error,
                    "Could not find auth code in the pasted URL.\n"
                    "Make sure you copy the FULL URL from the browser address bar.")
                return

            # Try to exchange using AuthGem's exchange function
            if hasattr(authgem_auth, 'exchange_code_for_tokens'):
                tokens = authgem_auth.exchange_code_for_tokens(auth_code)
            else:
                _dispatch_error(on_error,
                    "AuthGem module does not support code exchange. "
                    "Please update your AuthGem module.")
                return

            store = authgem_auth.get_store(account_id)
            store.save_tokens(tokens)
            logger.info("AuthGem tokens saved successfully.")

            if on_success:
                try:
                    from kivy.clock import Clock
                    Clock.schedule_once(lambda dt: on_success(tokens), 0)
                except ImportError:
                    on_success(tokens)

        except Exception as e:
            import traceback
            msg = f"AuthGem token exchange failed: {e}"
            logger.error(f"{msg}\n{traceback.format_exc()}")
            _dispatch_error(on_error, msg)

    threading.Thread(target=_worker, daemon=True).start()


def _show_paste_dialog(system_name, on_url_pasted, on_cancel):
    """Show a KivyMD dialog with instructions and a text field for pasting the callback URL.
    
    The dialog explains to the user what to do after the browser opens:
    1. Log in normally in the browser
    2. After login, the browser will try to navigate to localhost (which will fail)
    3. Copy the full URL from the browser address bar
    4. Paste it into the text field and tap "Submit"
    """
    from kivymd.uix.dialog import MDDialog
    from kivymd.uix.button import MDFlatButton, MDRaisedButton
    from kivymd.uix.textfield import MDTextField
    from kivymd.uix.boxlayout import MDBoxLayout
    from kivy.uix.boxlayout import BoxLayout
    from kivy.metrics import dp

    # Content layout
    content = BoxLayout(
        orientation='vertical',
        spacing=dp(12),
        padding=[dp(4), dp(8)],
        size_hint_y=None,
        height=dp(220),
    )

    from kivymd.uix.label import MDLabel
    instructions = MDLabel(
        text=(
            f"[b]{system_name} Login[/b]\n\n"
            "1. Log in normally in the browser that just opened\n"
            "2. After login, the page will show an error (this is expected)\n"
            "3. [b]Copy the FULL URL[/b] from the browser address bar\n"
            "    It will look like: http://localhost:1455/auth/callback?code=...\n"
            "4. Paste it below and tap Submit"
        ),
        markup=True,
        font_style="Body2",
        size_hint_y=None,
        height=dp(150),
    )
    content.add_widget(instructions)

    url_field = MDTextField(
        hint_text="Paste the callback URL here",
        multiline=False,
        size_hint_y=None,
        height=dp(48),
    )
    content.add_widget(url_field)

    dialog = None

    def _on_submit(*args):
        nonlocal dialog
        url = url_field.text.strip()
        if dialog:
            dialog.dismiss()
        if url:
            on_url_pasted(url)
        else:
            on_cancel()

    def _on_cancel_btn(*args):
        nonlocal dialog
        if dialog:
            dialog.dismiss()
        on_cancel()

    dialog = MDDialog(
        title=f"{system_name} Login",
        type="custom",
        content_cls=content,
        buttons=[
            MDFlatButton(
                text="CANCEL",
                on_release=_on_cancel_btn,
            ),
            MDRaisedButton(
                text="SUBMIT",
                on_release=_on_submit,
            ),
        ],
    )
    dialog.open()


def _start_oauth_flow_desktop(on_success, on_error, timeout, account_id, system):
    """Desktop OAuth flow — original localhost:1455 callback approach."""
    def _worker():
        try:
            if system == "authgem":
                # AuthGem Flow
                import authgem_auth
                import webbrowser
                original_open = webbrowser.open
                
                def _browser_mock(url, *args, **kwargs):
                    logger.info(f"Opening browser for AuthGem Account {account_id} login...")
                    if not _open_browser_android(url):
                        _dispatch_error(on_error, "Could not open browser for AuthGem login.")
                    return True
                
                webbrowser.open = _browser_mock
                try:
                    tokens = authgem_auth.run_oauth_flow(timeout=timeout)
                finally:
                    webbrowser.open = original_open
                    
                store = authgem_auth.get_store(account_id)
                store.save_tokens(tokens)
                logger.info("AuthGem tokens saved successfully.")
            else:
                # AuthGPT Flow
                from authgpt_auth import (
                    generate_pkce, build_auth_url, exchange_code_for_tokens,
                    get_store, CALLBACK_HOST, CALLBACK_PORT, CALLBACK_PATH,
                    _OAuthCallbackHandler,
                )
                from http.server import HTTPServer
                import secrets

                # Verify port is available
                import socket
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind((CALLBACK_HOST, CALLBACK_PORT))
                except OSError:
                    _dispatch_error(
                        on_error,
                        f"Port {CALLBACK_PORT} is busy. Close other apps and retry."
                    )
                    return

                # Generate PKCE values
                code_verifier, code_challenge = generate_pkce()
                state = secrets.token_urlsafe(32)
                redirect_uri = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
                auth_url = build_auth_url(code_challenge, state, redirect_uri)

                # Start callback server
                server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), _OAuthCallbackHandler)
                server._auth_code = None
                server._returned_state = None
                server._error = None
                server.timeout = timeout

                # Open browser
                logger.info(f"Opening browser for ChatGPT Account {account_id} login...")
                import webbrowser
                webbrowser.open(auth_url)

                # Serve until callback
                server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                server_thread.start()
                server_thread.join(timeout=timeout)
                server.shutdown()

                # Check result
                if server._error:
                    _dispatch_error(on_error, f"OAuth error: {server._error}")
                    return
                if not server._auth_code:
                    _dispatch_error(on_error, "Login timed out — no callback received.")
                    return
                if server._returned_state != state:
                    _dispatch_error(on_error, "OAuth state mismatch — possible CSRF attack.")
                    return

                # Exchange code for tokens
                logger.info("Exchanging auth code for tokens...")
                tokens = exchange_code_for_tokens(
                    server._auth_code, code_verifier, redirect_uri
                )

                # Save tokens
                store = get_store(account_id)
                store.save_tokens(tokens)
                logger.info("AuthGPT tokens saved successfully.")

            # Success callback
            if on_success:
                try:
                    from kivy.clock import Clock
                    Clock.schedule_once(lambda dt: on_success(tokens), 0)
                except ImportError:
                    on_success(tokens)

        except Exception as e:
            import traceback
            msg = f"AuthGPT login failed: {e}\n{traceback.format_exc()}"
            logger.error(msg)
            _dispatch_error(on_error, str(e))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def _dispatch_error(on_error, message):
    """Dispatch error callback on the main thread."""
    if on_error:
        try:
            from kivy.clock import Clock
            Clock.schedule_once(lambda dt: on_error(message), 0)
        except ImportError:
            on_error(message)


def has_valid_token(account_id: int = 0, system: str = "authgpt") -> bool:
    """Check if we have a valid (non-expired) AuthGPT or AuthGem token."""
    try:
        if system == "authgem":
            import authgem_auth
            store = authgem_auth.get_store(account_id)
            return store.has_tokens
        else:
            from authgpt_auth import get_store
            store = get_store(account_id)
            tokens = store.load_tokens()
            if not tokens or not tokens.get("access_token"):
                return False
            import time
            expires_at = tokens.get("expires_at", 0)
            return time.time() < (expires_at - 300)  # 5 min margin
    except Exception:
        return False


def get_account_email(account_id: int = 0, system: str = "authgpt") -> str:
    """Return the logged-in email, or empty string."""
    try:
        if system == "authgem":
            import authgem_auth
            info = authgem_auth.get_store(account_id).account_info
            return info.get("email", "")
        else:
            from authgpt_auth import get_store
            info = get_store(account_id).account_info
            return info.get("email", "")
    except Exception:
        return ""


def logout(account_id: int = 0, system: str = "authgpt"):
    """Clear stored OAuth tokens."""
    try:
        if system == "authgem":
            import authgem_auth
            authgem_auth.get_store(account_id).clear_tokens()
        else:
            from authgpt_auth import get_store
            get_store(account_id).clear_tokens()
    except Exception:
        pass
