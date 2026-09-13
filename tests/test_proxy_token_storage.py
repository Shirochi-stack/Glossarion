import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

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


ANTIGRAVITY_STORAGE_SOURCE = '''import { join } from "path";
const ACCOUNTS_FILE = process.env.ACCOUNTS_FILE || join(process.cwd(), "antigravity-accounts.json");
interface StorageFormat { accounts: any[]; strategy?: string; }
export async function loadConfig(): Promise<StorageFormat> {
  try {
    const file = Bun.file(ACCOUNTS_FILE);
    if (await file.exists()) {
      const data = await file.json();
      return Array.isArray(data) ? { accounts: data, strategy: "hybrid" } : data;
    }
  } catch (e) {
    console.error("Failed to load accounts:", e);
  }
  return { accounts: [] };
}
export async function saveConfig(config: StorageFormat): Promise<void> {
  try {
    await Bun.write(ACCOUNTS_FILE, JSON.stringify(config, null, 2));
  } catch (e) {
    console.error("Failed to save accounts:", e);
  }
}
'''


def make_antigravity_storage(tmp_path, *, legacy_encrypted=False):
    root = tmp_path / "synthetic runtime with spaces"
    path = root / "src" / "auth" / "storage.ts"
    path.parent.mkdir(parents=True)
    source = ANTIGRAVITY_STORAGE_SOURCE
    if legacy_encrypted:
        # Installed v1 runtimes already encrypt each write, but allow concurrent
        # replacements of the same file. They also need the new queue patch.
        source = source.replace(
            "const data = await file.json();",
            "const data = JSON.parse(decryptSerialized(await file.text()));",
        ).replace(
            "await Bun.write(ACCOUNTS_FILE, JSON.stringify(config, null, 2));",
            "const temporary = ACCOUNTS_FILE + '.' + randomUUID() + '.tmp';\n"
            "    try {\n"
            "      await writeFile(temporary, encryptSerialized(JSON.stringify(config, null, 2)), {mode:0o600});\n"
            "      await rename(temporary, ACCOUNTS_FILE);\n"
            "    } finally { await unlink(temporary).catch(() => {}); }",
        ).replace('console.error("Failed to load accounts:", e);', "throw e;").replace(
            'console.error("Failed to save accounts:", e);', "throw e;"
        )
        source = (
            'import {encryptSerialized, decryptSerialized} from "./glossarion-token-storage.mjs";\n'
            'import {writeFile, rename, unlink} from "node:fs/promises";\n'
            'import {randomUUID} from "node:crypto";\n' + source
        )
    path.write_text(source, encoding="utf-8")
    vault.patch_antigravity(root)
    patched = path.read_bytes()
    vault.patch_antigravity(root)
    assert path.read_bytes() == patched
    return path


@pytest.mark.parametrize("legacy_encrypted", [False, True])
def test_antigravity_concurrent_saves_preserve_encrypted_snapshots(tmp_path, legacy_encrypted):
    bun = shutil.which("bun")
    if not bun:
        pytest.skip("Bun is unavailable")
    storage = make_antigravity_storage(tmp_path, legacy_encrypted=legacy_encrypted)
    account_path = tmp_path / "antigravity-accounts.json"
    value = {"accounts": [{"email": "test@example.com", "refreshToken": "synthetic-secret",
                            "padding": "x" * 51200, "revision": 0}], "strategy": "round-robin"}
    script = (
        f"import {{saveConfig,loadConfig}} from {json.dumps(storage.as_posix())};\n"
        "import {readFileSync} from 'node:fs';\n"
        "const data=JSON.parse(readFileSync(0,'utf8'));\n"
        "await saveConfig(data);\n"
        "await loadConfig();\n"
        "const saves=[];\n"
        "for(let revision=0;revision<6;revision++){\n"
        "  data.accounts[0].revision=revision;\n"
        "  saves.push(saveConfig(data));\n"
        "}\n"
        "data.accounts[0].revision=999;\n"
        "await Promise.all(saves);\n"
        "console.log(JSON.stringify(await loadConfig()));"
    )
    result = subprocess.run([bun, "-e", script], input=json.dumps(value), text=True,
                            capture_output=True, env=dict(os.environ, ACCOUNTS_FILE=str(account_path)), timeout=60)
    assert result.returncode == 0, result.stderr
    value["accounts"][0]["revision"] = 5
    assert json.loads(result.stdout) == value
    assert vault.load_accounts(account_path) == value
    assert tokens.is_encrypted(str(account_path))
    assert "synthetic-secret" not in account_path.read_text()
    assert list(tmp_path.glob("*.tmp")) == []


def test_antigravity_failed_save_preserves_credentials_and_queue_recovers(tmp_path):
    bun = shutil.which("bun")
    if not bun:
        pytest.skip("Bun is unavailable")
    storage = make_antigravity_storage(tmp_path)
    account_dir = tmp_path / "accounts"
    account_dir.mkdir()
    account_path = account_dir / "antigravity-accounts.json"
    script = (
        f"import {{saveConfig,loadConfig}} from {json.dumps(storage.as_posix())};\n"
        "import assert from 'node:assert/strict';\n"
        "import {readFile,rename} from 'node:fs/promises';\n"
        "import {dirname} from 'node:path';\n"
        "const path=process.env.ACCOUNTS_FILE, directory=dirname(path), moved=directory+'.moved';\n"
        "await saveConfig({accounts:[{refreshToken:'original-synthetic-secret'}]});\n"
        "const before=await readFile(path);\n"
        "await rename(directory,moved);\n"
        "try {\n"
        "  await assert.rejects(saveConfig({accounts:[{refreshToken:'failed-synthetic-secret'}]}),\n"
        "                       error=>error.code==='ENOENT');\n"
        "} finally { await rename(moved,directory); }\n"
        "assert.deepEqual(await readFile(path),before);\n"
        "await saveConfig({accounts:[{refreshToken:'next-synthetic-secret'}]});\n"
        "console.log(JSON.stringify(await loadConfig()));"
    )
    result = subprocess.run([bun, "-e", script], text=True, capture_output=True,
                            env=dict(os.environ, ACCOUNTS_FILE=str(account_path)), timeout=60)
    assert result.returncode == 0, result.stderr
    expected = {"accounts": [{"refreshToken": "next-synthetic-secret"}]}
    assert json.loads(result.stdout) == expected
    assert vault.load_accounts(account_path) == expected
    assert tokens.is_encrypted(str(account_path))
    assert "synthetic-secret" not in account_path.read_text()
    assert list(account_dir.glob("*.tmp")) == []


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


@pytest.fixture
def ocagy_runtime(tmp_path):
    import ocagy_cli
    bun = shutil.which("bun")
    roots = [root for root in ocagy_cli._plugin_install_candidates()
             if (root / "dist/src/plugin/storage.js").is_file()]
    if not bun or not roots:
        pytest.skip("Installed OcAgy/Bun fixture unavailable")
    original = roots[0]
    root = tmp_path / "synthetic ocagy plugin"
    shutil.copytree(original / "dist", root / "dist")
    shutil.copy2(original / "package.json", root / "package.json")
    vault.patch_ocagy(root)
    storage = root / "dist/src/plugin/storage.js"
    patched = storage.read_bytes()
    vault.patch_ocagy(root)
    assert storage.read_bytes() == patched
    config = tmp_path / "synthetic ocagy config"
    config.mkdir()
    accounts = config / "antigravity-accounts.json"
    vault.save_accounts(accounts, {"version": 4, "accounts": [], "activeIndex": 0})
    env = dict(os.environ, OPENCODE_CONFIG_DIR=str(config), NODE_PATH=str(original.parent))
    hidden = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    return bun, storage, accounts, env, hidden


@pytest.mark.parametrize("separate_processes", [False, True])
def test_ocagy_concurrent_encrypted_saves_merge_all_accounts(ocagy_runtime, separate_processes):
    bun, storage, accounts, env, hidden = ocagy_runtime
    script = (
        f"import {{saveAccounts}} from {json.dumps(storage.as_posix())};\n"
        "const save=index=>saveAccounts({version:4,activeIndex:0,accounts:[{\n"
        "  email:`account${index}@example.com`,refreshToken:`synthetic-${index}`,\n"
        "  addedAt:index,lastUsed:index,padding:'x'.repeat(51200)\n"
        "}]});\n"
    )
    if separate_processes:
        script += "await save(Number(process.env.TEST_ACCOUNT_INDEX)); console.log('saved');"
        processes = []
        outputs = []
        deadline = time.monotonic() + 60
        try:
            for index in range(6):
                processes.append(subprocess.Popen(
                    [bun, "-e", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    env=dict(env, TEST_ACCOUNT_INDEX=str(index)), **hidden,
                ))
            for process in processes:
                outputs.append(process.communicate(timeout=max(0.1, deadline - time.monotonic())))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)
        for process, (stdout, stderr) in zip(processes, outputs):
            assert process.returncode == 0, stderr
            assert stdout.strip() == "saved"
    else:
        script += "await Promise.all(Array.from({length:6},(_,index)=>save(index))); console.log('saved');"
        result = subprocess.run([bun, "-e", script], text=True, capture_output=True,
                                env=env, timeout=60, **hidden)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "saved"
    result = vault.load_accounts(accounts)
    assert len(result["accounts"]) == 6
    assert sorted(account["refreshToken"] for account in result["accounts"]) == [
        f"synthetic-{index}" for index in range(6)
    ]
    assert all(account["padding"] == "x" * 51200 for account in result["accounts"])
    assert tokens.is_encrypted(str(accounts))
    assert "synthetic-" not in accounts.read_text()
    assert list(accounts.parent.glob("*.tmp")) == []


def test_ocagy_replace_failure_preserves_credentials_and_queued_snapshot(ocagy_runtime):
    bun, storage, accounts, env, hidden = ocagy_runtime
    adapter = storage.parent / "glossarion-token-storage.mjs"
    (storage.parent / "test-real-token-storage.mjs").write_bytes(adapter.read_bytes())
    adapter.write_text(
        'import {encryptSerialized as encrypt,decryptSerialized} from "./test-real-token-storage.mjs";\n'
        'export {decryptSerialized};\n'
        'export function encryptSerialized(text) {\n'
        '  if (JSON.parse(text).accounts?.some(account=>account.failSave)) {\n'
        '    throw Object.assign(new Error("Synthetic encryption failure"), '
        '{code:"GLOSSARION_CREDENTIAL_CRYPTO"});\n'
        '  }\n'
        '  return encrypt(text);\n'
        '}\n', encoding="utf-8",
    )
    script = (
        f"import {{saveAccountsReplace,loadAccounts}} from {json.dumps(storage.as_posix())};\n"
        "import assert from 'node:assert/strict';\n"
        "import {readFile} from 'node:fs/promises';\n"
        f"const path={json.dumps(accounts.as_posix())};\n"
        "await saveAccountsReplace({version:4,activeIndex:0,accounts:[{refreshToken:'original-synthetic-secret'}]});\n"
        "const before=await readFile(path);\n"
        "await assert.rejects(saveAccountsReplace({version:4,activeIndex:0,accounts:[{\n"
        "  refreshToken:'failed-synthetic-secret',failSave:true\n"
        "}]}), error=>error.code==='GLOSSARION_CREDENTIAL_CRYPTO');\n"
        "assert.deepEqual(await readFile(path),before);\n"
        "const data={version:4,activeIndex:0,accounts:[{refreshToken:'first-synthetic-secret'}]};\n"
        "const first=saveAccountsReplace(data);\n"
        "data.accounts[0].refreshToken='last-synthetic-secret';\n"
        "const last=saveAccountsReplace(data);\n"
        "data.accounts[0].refreshToken='mutated-after-save';\n"
        "await Promise.all([first,last]);\n"
        "assert.equal((await loadAccounts()).accounts[0].refreshToken,'last-synthetic-secret');\n"
        "console.log('saved');"
    )
    result = subprocess.run([bun, "-e", script], text=True, capture_output=True,
                            env=env, timeout=60, **hidden)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "saved"
    assert [account["refreshToken"] for account in vault.load_accounts(accounts)["accounts"]] == [
        "last-synthetic-secret"
    ]
    assert tokens.is_encrypted(str(accounts))
    assert "synthetic-secret" not in accounts.read_text()
    assert list(accounts.parent.glob("*.tmp")) == []
