import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import proxy_token_storage as vault
import token_encryption as tokens


@pytest.fixture
def adapter(tmp_path):
    path = tmp_path / "runtime with spaces"
    path.mkdir()
    vault._write_adapter(path)
    return path / "glossarion-token-storage.mjs"


def run_js(adapter, body, payload, *, env=None, unix=False):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable")
    script = ("Object.defineProperty(process,'platform',{value:'linux'});\n" if unix else "")
    script += "import {readFileSync} from 'node:fs';\n"
    script += f"const vault=await import({json.dumps(adapter.as_uri())});\n"
    script += "const input=readFileSync(0,'utf8');\n" + body
    return subprocess.run([node, "--input-type=module", "-e", script], input=payload,
                          text=True, capture_output=True, env=env, timeout=45)


def test_python_javascript_encrypted_roundtrip(adapter):
    value = {"accounts": [{"email": "test@example.com", "refreshToken": "synthetic-secret"}]}
    python_cipher = tokens.encrypt_tokens(value).decode()
    result = run_js(adapter, "process.stdout.write(vault.decryptSerialized(input));", python_cipher)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == value
    result = run_js(adapter, "process.stdout.write(vault.encryptSerialized(input));", json.dumps(value))
    assert result.returncode == 0, result.stderr
    assert "synthetic-secret" not in result.stdout
    assert tokens.decrypt_tokens(result.stdout.encode()) == value


def test_unix_fernet_matches_python_format(adapter, tmp_path):
    key = os.urandom(32)
    (tmp_path / ".glossarion").mkdir()
    (tmp_path / ".glossarion" / ".token_key").write_text(base64.b64encode(key).decode())
    env = dict(os.environ, HOME=str(tmp_path), USERPROFILE=str(tmp_path))
    plaintext = json.dumps({"refreshToken": "synthetic-secret"}).encode()
    cipher = "GLSE1:" + base64.b64encode(tokens._fernet_encrypt(plaintext, key)).decode()
    decoded = run_js(adapter, "process.stdout.write(vault.decryptSerialized(input));", cipher, env=env, unix=True)
    assert decoded.returncode == 0, decoded.stderr
    assert decoded.stdout.encode() == plaintext
    encoded = run_js(adapter, "process.stdout.write(vault.encryptSerialized(input));", plaintext.decode(), env=env, unix=True)
    assert encoded.returncode == 0, encoded.stderr
    assert tokens._fernet_decrypt(base64.b64decode(encoded.stdout[6:]), key) == plaintext


def test_corrupt_cipher_fails_without_disclosing_data(adapter):
    result = run_js(adapter, "process.stdout.write(vault.decryptSerialized(input));", "GLSE1:broken")
    assert result.returncode != 0
    assert not result.stdout
    assert "Credential encryption/decryption failed" in result.stderr


def test_atomic_migration_and_failed_write_preserve_credentials(tmp_path, monkeypatch):
    path = tmp_path / "accounts.json"
    value = {"accounts": [{"refreshToken": "synthetic-secret"}]}
    path.write_text(json.dumps(value))
    assert vault.load_accounts(path, migrate=True) == value
    assert tokens.is_encrypted(str(path))
    before = path.read_bytes()
    def fail(*args):
        raise RuntimeError("encryption unavailable")
    monkeypatch.setattr(tokens, "encrypt_tokens", fail)
    with pytest.raises(RuntimeError):
        vault.save_accounts(path, {"accounts": []})
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_opencode_auth_migration_preserves_other_providers(tmp_path):
    path = tmp_path / "auth.json"
    value = {"google": {"type": "oauth", "refresh": "synthetic-refresh|project", "access": "old-access", "expires": 123},
             "other": {"type": "api", "key": "other-provider-key"}}
    path.write_text(json.dumps(value))
    vault.migrate_opencode_oauth(path)
    result = json.loads(path.read_text())
    assert result["other"] == value["other"]
    assert tokens.decrypt_tokens(result["google"]["refresh"].encode()) == value["google"]["refresh"]
    assert result["google"]["access"] == "" and result["google"]["expires"] == 0
    assert "synthetic-refresh" not in path.read_text()
    before = path.read_bytes()
    vault.migrate_opencode_oauth(path)
    assert path.read_bytes() == before


def test_runtime_patch_incompatibility_leaves_source_unchanged(tmp_path):
    path = tmp_path / "src" / "auth" / "storage.ts"
    path.parent.mkdir(parents=True)
    path.write_text("upstream changed")
    with pytest.raises(RuntimeError, match="incompatible"):
        vault.patch_antigravity(tmp_path)
    assert path.read_text() == "upstream changed"


def test_specs_include_proxy_credential_adapter():
    for path in (Path(__file__).resolve().parents[1] / "src").glob("translator*.spec"):
        source = path.read_text(encoding="utf-8")
        assert "('proxy_token_storage.py', '.')" in source, path
        assert "'proxy_token_storage'," in source, path


def test_installed_antigravity_storage_roundtrip(tmp_path):
    import antigravity_proxy as proxy
    bun = shutil.which("bun")
    roots = list((Path(proxy._get_proxy_data_dir()) / "runtime").glob("*/src/auth/storage.ts"))
    if not bun or not roots:
        pytest.skip("Installed Antigravity/Bun fixture unavailable")
    root = tmp_path / "runtime with spaces"
    destination = root / "src" / "auth"
    destination.mkdir(parents=True)
    shutil.copy2(roots[-1], destination / "storage.ts")
    vault.patch_antigravity(root)
    vault.patch_antigravity(root)
    account_path = tmp_path / "antigravity-accounts.json"
    value = {"accounts": [{"refreshToken": "synthetic-secret", "email": "test@example.com"}], "strategy": "round-robin"}
    script = f"import {{saveConfig,loadConfig}} from {json.dumps((destination / 'storage.ts').as_posix())}; import {{readFileSync}} from 'node:fs'; await saveConfig(JSON.parse(readFileSync(0,'utf8'))); console.log(JSON.stringify(await loadConfig()));"
    result = subprocess.run([bun, "-e", script], input=json.dumps(value), text=True,
                            capture_output=True, env=dict(os.environ, ACCOUNTS_FILE=str(account_path)), timeout=45)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == value
    assert vault.load_accounts(account_path) == value
    assert "synthetic-secret" not in account_path.read_text()


def test_installed_ocagy_storage_and_refresh_roundtrip(tmp_path):
    import ocagy_cli
    bun = shutil.which("bun")
    roots = [root for root in ocagy_cli._plugin_install_candidates() if (root / "dist/src/plugin/storage.js").is_file()]
    if not bun or not roots:
        pytest.skip("Installed OcAgy/Bun fixture unavailable")
    original = roots[0]
    root = tmp_path / "plugin"
    shutil.copytree(original / "dist", root / "dist")
    shutil.copy2(original / "package.json", root / "package.json")
    vault.patch_ocagy(root)
    vault.patch_ocagy(root)
    oauth = (root / "dist/src/antigravity/oauth.js").read_text()
    assert "access: tokenPayload.access_token," not in oauth
    assert 'access: "",' in oauth
    config = tmp_path / "config"
    config.mkdir()
    storage_path = config / "antigravity-accounts.json"
    value = {"version": 4, "accounts": [{"refreshToken": "synthetic-secret", "email": "test@example.com", "addedAt": 1, "lastUsed": 2}], "activeIndex": 0}
    script = f"import {{saveAccounts,loadAccounts}} from {json.dumps((root / 'dist/src/plugin/storage.js').as_posix())}; import {{formatRefreshParts,parseRefreshParts}} from {json.dumps((root / 'dist/src/plugin/auth.js').as_posix())}; import {{readFileSync}} from 'node:fs'; await saveAccounts(JSON.parse(readFileSync(0,'utf8'))); const packed=formatRefreshParts({{refreshToken:'synthetic-secret',projectId:'test-project'}}); if(!packed.startsWith('GLSE1:') || parseRefreshParts(packed).refreshToken!=='synthetic-secret') throw Error('refresh roundtrip'); console.log(JSON.stringify(await loadAccounts()));"
    result = subprocess.run([bun, "-e", script], input=json.dumps(value), text=True,
                            capture_output=True, env=dict(os.environ, OPENCODE_CONFIG_DIR=str(config), NODE_PATH=str(original.parent)), timeout=45)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["accounts"] == value["accounts"]
    assert vault.load_accounts(storage_path)["accounts"] == value["accounts"]
    assert "synthetic-secret" not in storage_path.read_text()
    storage_path.write_text("GLSE1:broken")
    rejected = subprocess.run([bun, "-e", script], input=json.dumps(value), text=True,
                              capture_output=True, env=dict(os.environ, OPENCODE_CONFIG_DIR=str(config), NODE_PATH=str(original.parent)), timeout=45)
    assert rejected.returncode != 0
    assert storage_path.read_text() == "GLSE1:broken"
