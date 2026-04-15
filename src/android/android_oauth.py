# android_oauth.py
"""
Android-specific OAuth flow for AuthGPT (ChatGPT Plus/Pro subscription).

Uses the same localhost:1455 callback approach as the desktop flow, but
opens the browser via Android Intent instead of Python's webbrowser module.

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


def start_oauth_flow(
    on_success=None,
    on_error=None,
    timeout: int = 300,
    account_id: int = 0,
    system: str = "authgpt",
):
    """
    Start the AuthGPT or AuthGem OAuth flow on Android.

    1. Starts a localhost HTTP callback server on port 1455
    2. Opens the auth URL in the system browser via Android Intent
    3. Waits for the OAuth callback
    4. Exchanges the code for tokens
    5. Saves tokens to the AuthGPTTokenStore

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
    """
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
                if not _open_browser_android(auth_url):
                    _dispatch_error(on_error, "Could not open browser for login.")
                    server.server_close()
                    return

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
