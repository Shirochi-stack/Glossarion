"""Key-pool cooldowns from a route's own wait (a DeepSeek mute rests the key for the session) and the
Indefinite Rate Limit Retry default (off; authds/ always retries its temporary mute)."""
import math
import time

import multi_api_key_manager as mk


def pool_with(*models):
    pool = mk.APIKeyPool()
    for model in models:
        pool.add_key(mk.APIKeyEntry("", model, cooldown=60))
    return pool


def test_rate_limit_uses_the_routes_wait_then_the_keys_own_cooldown():
    pool = pool_with("authds/flash", "authds/pro")
    pool.mark_key_error(0, 429, cooldown=5)
    key = pool.keys[0]
    assert not key.is_available()
    key.last_error_time = time.time() - 6
    assert key.is_available()  # the route's 5 s, not the key's 60 s
    pool.mark_key_error(0, 429)
    key.last_error_time = time.time() - 6
    assert not key.is_available()  # no route wait: the key's own cooldown


def test_a_muted_account_stays_out_of_the_pool_for_the_session():
    pool = pool_with("authds/flash", "gemini-3.5-flash")
    pool.mark_key_error(0, 429, cooldown=math.inf)
    pool.keys[0].last_error_time = time.time() - 10 ** 7
    assert not pool.keys[0].is_available()
    assert pool.keys[1].is_available()  # the other keys keep serving


def test_indefinite_retry_follows_the_toggle_only(monkeypatch):
    from unified_api_client import UnifiedClient
    client = object.__new__(UnifiedClient)
    monkeypatch.delenv("INDEFINITE_RATE_LIMIT_RETRY", raising=False)
    assert client._indefinite_rate_limit_retry() is False  # off unless the run exports it
    monkeypatch.setenv("INDEFINITE_RATE_LIMIT_RETRY", "0")
    assert client._indefinite_rate_limit_retry() is False
    monkeypatch.setenv("INDEFINITE_RATE_LIMIT_RETRY", "1")
    assert client._indefinite_rate_limit_retry() is True


def test_a_session_block_is_never_retried():
    from unified_api_client import UnifiedClient, UnifiedClientError
    blocked = UnifiedClientError("muted", error_type="rate_limit", details={"session_blocked": True})
    assert UnifiedClient._blocked_for_session(blocked)
    assert not UnifiedClient._blocked_for_session(UnifiedClientError("429", error_type="rate_limit"))
