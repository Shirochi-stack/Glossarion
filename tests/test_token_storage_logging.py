import ast
import json
import logging
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest
import token_encryption as storage


@pytest.fixture(autouse=True)
def flush_storage_logs():
    storage._flush_storage_logs()
    yield
    storage._flush_storage_logs()


def test_encryption_and_decryption_logs_exclude_credentials(tmp_path, capsys):
    path = tmp_path / "test_tokens.json"
    secret = {"access_token": "private-token-value"}
    storage.save_encrypted_tokens(secret, str(path))
    assert storage.load_encrypted_tokens(str(path)) == secret
    storage._flush_storage_logs()
    output = capsys.readouterr()
    assert not output.out
    for message in ("🔒 1 account encrypted", "🔓 1 account decrypted"):
        assert message in output.err
    assert "private-token-value" not in output.err


@pytest.mark.parametrize("combined", [False, True])
def test_thirty_accounts_log_one_summary_per_operation(tmp_path, capsys, combined, monkeypatch):
    class DeferredTimer:
        def __init__(self, *args):
            pass
        def start(self):
            pass
        def cancel(self):
            pass
    monkeypatch.setattr(storage.threading, "Timer", DeferredTimer)
    accounts = {str(i): {"email": f"account{i}@example.com", "access_token": "secret"} for i in range(30)}
    records = {"accounts.enc": accounts} if combined else {
        f"authgpt_tokens_{i}.json": account for i, account in accounts.items()
    }
    for filename, value in records.items():
        path = str(tmp_path / filename)
        storage.save_encrypted_tokens(value, path)
        storage.load_encrypted_tokens(path)
        storage.load_encrypted_tokens(path)  # A repeat read is not another account.
    storage._flush_storage_logs()
    lines = capsys.readouterr().err.splitlines()
    assert lines == ["🔒 30 accounts encrypted — first: account0@example.com",
                     "🔓 30 accounts decrypted — first: account0@example.com"]


@pytest.mark.parametrize("operation", ["encrypt", "decrypt"])
def test_crypto_failure_has_no_success_log(tmp_path, monkeypatch, capsys, operation):
    path = tmp_path / "test_tokens.json"
    path.write_bytes(storage._ENCRYPTED_HEADER + b"invalid")
    def fail(*args):
        raise RuntimeError("private-token-value")
    monkeypatch.setattr(storage, operation + "_tokens", fail)
    with pytest.raises(RuntimeError):
        if operation == "encrypt":
            storage.save_encrypted_tokens({}, str(path))
        else:
            storage.load_encrypted_tokens(str(path))
    output = capsys.readouterr().err
    assert "❌ Credential " + operation + "ion failed" in output
    assert "successfully" not in output and "encrypted and saved" not in output
    assert "private-token-value" not in output


@pytest.mark.parametrize("provider", ["authgpt", "authgem", "authgrok", "authcd"])
@pytest.mark.parametrize("missing_module", [False, True])
def test_plaintext_fallback_is_explicitly_logged(tmp_path, monkeypatch, caplog, provider, missing_module):
    source = Path(__file__).resolve().parents[1] / "src" / (provider + "_auth.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "save_tokens")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    namespace = {"logger": logging.getLogger(provider), "json": json}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    if missing_module:
        monkeypatch.setitem(sys.modules, "token_encryption", None)
    else:
        def fail(*args):
            raise RuntimeError("test encryption unavailable")
        monkeypatch.setattr(storage, "save_encrypted_tokens", fail)
    path = tmp_path / "tokens.json"
    store = SimpleNamespace(_lock=threading.RLock(), _token_file=str(path),
                            _ensure_dir=lambda: None, _fire_change_callbacks=lambda: None)
    with caplog.at_level(logging.WARNING):
        namespace["save_tokens"](store, {"access_token": "private-token-value"})
    assert json.loads(path.read_text())["access_token"] == "private-token-value"
    assert "🔓" in caplog.text and "WITHOUT ENCRYPTION" in caplog.text
    assert "private-token-value" not in caplog.text
