"""Encrypted account-file adapters for the JavaScript OAuth runtimes.

Uses token_encryption's GLSE1 envelope, Windows DPAPI, and Unix Fernet key.
No decrypted credential files, command-line secrets, or plaintext fallback.
"""
import json
import os
from pathlib import Path
import sys
import tempfile

import token_encryption


def prepare_key():
    if sys.platform != "win32":
        # Never use token_encryption's emergency obfuscation fallback here.
        from cryptography.fernet import Fernet  # noqa: F401
        token_encryption._get_symmetric_key()


def save_accounts(path, value):
    prepare_key()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        token_encryption.save_encrypted_tokens(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_accounts(path, *, migrate=False):
    path = Path(path)
    if not path.is_file():
        return None
    raw = path.read_bytes()
    if raw.startswith(token_encryption._ENCRYPTED_HEADER):
        return token_encryption.load_encrypted_tokens(str(path))
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, (dict, list)):
        raise ValueError("Invalid proxy account store")
    if migrate:
        save_accounts(path, value)
    return value


JS_ADAPTER = r'''// Glossarion token_encryption GLSE1 adapter v1
import {spawnSync} from 'node:child_process';
import {createCipheriv, createDecipheriv, createHmac, randomBytes, timingSafeEqual} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {homedir} from 'node:os';
import {join} from 'node:path';

const fail = () => { const e = new Error('Credential encryption/decryption failed; account file was not replaced'); e.code = 'GLOSSARION_CREDENTIAL_CRYPTO'; throw e; };
function dpapi(input, decrypt) {
  // Only static code is passed as an argument. Secret bytes travel over stdin.
  const method = decrypt ? 'Unprotect' : 'Protect';
  const script = `Add-Type -AssemblyName System.Security; $d=[Convert]::FromBase64String([Console]::In.ReadToEnd()); $r=[Security.Cryptography.ProtectedData]::${method}($d,$null,[Security.Cryptography.DataProtectionScope]::CurrentUser); [Console]::Out.Write([Convert]::ToBase64String($r))`;
  const systemRoot=process.env.SystemRoot || process.env.SYSTEMROOT || 'C:\\Windows';
  const executable=join(systemRoot,'System32','WindowsPowerShell','v1.0','powershell.exe');
  const env=Object.fromEntries(Object.entries(process.env).filter(([name])=>['systemroot','windir','temp','tmp','userprofile','localappdata','appdata','homedrive','homepath'].includes(name.toLowerCase())));
  const result = spawnSync(executable, ['-NoProfile','-NonInteractive','-EncodedCommand',Buffer.from(script,'utf16le').toString('base64')], {
    input: input.toString('base64'), encoding:'utf8', windowsHide:true, env, timeout:30000, maxBuffer:32*1024*1024
  });
  if (result.error || result.status !== 0 || !result.stdout.trim()) fail();
  return Buffer.from(result.stdout.trim(),'base64');
}
function symmetricKey() {
  let key;
  if (process.platform === 'darwin') {
    const r = spawnSync('security',['find-generic-password','-a','glossarion','-s','glossarion-token-encryption','-w'], {encoding:'utf8',timeout:10000});
    if (r.error || r.status !== 0) fail();
    key = Buffer.from(r.stdout.trim(),'base64');
  } else {
    key = Buffer.from(readFileSync(join(homedir(),'.glossarion','.token_key'),'utf8').trim(),'base64');
  }
  if (key.length !== 32) fail();
  return key;
}
function encryptUnix(data) {
  const key=symmetricKey(), iv=randomBytes(16), header=Buffer.alloc(9);
  header[0]=0x80; header.writeBigUInt64BE(BigInt(Math.floor(Date.now()/1000)),1);
  const cipher=createCipheriv('aes-128-cbc',key.subarray(16),iv);
  const signed=Buffer.concat([header,iv,cipher.update(data),cipher.final()]);
  const mac=createHmac('sha256',key.subarray(0,16)).update(signed).digest();
  const encoded=Buffer.concat([signed,mac]).toString('base64').replaceAll('+','-').replaceAll('/','_');
  return Buffer.from('FRN:'+encoded);
}
function decryptUnix(data) {
  if (data.subarray(0,4).toString() !== 'FRN:') fail();
  const key=symmetricKey(), token=Buffer.from(data.subarray(4).toString(),'base64url');
  if (token.length<73 || token[0]!==0x80) fail();
  const signed=token.subarray(0,-32), mac=token.subarray(-32);
  if (!timingSafeEqual(mac,createHmac('sha256',key.subarray(0,16)).update(signed).digest())) fail();
  const cipher=createDecipheriv('aes-128-cbc',key.subarray(16),token.subarray(9,25));
  return Buffer.concat([cipher.update(token.subarray(25,-32)),cipher.final()]);
}
export function encryptSerialized(text) {
  try { const raw=Buffer.from(text,'utf8'); return 'GLSE1:'+(process.platform==='win32'?dpapi(raw,false):encryptUnix(raw)).toString('base64'); }
  catch { return fail(); }
}
export function decryptSerialized(text) {
  if (!text.startsWith('GLSE1:')) return text; // Legacy JSON migration is done before launch.
  try { const raw=Buffer.from(text.slice(6),'base64'); return (process.platform==='win32'?dpapi(raw,true):decryptUnix(raw)).toString('utf8'); }
  catch { return fail(); }
}
'''


def _write_adapter(directory):
    prepare_key()
    target = Path(directory) / "glossarion-token-storage.mjs"
    target.write_text(JS_ADAPTER, encoding="utf-8")


def patch_antigravity(runtime_dir):
    path = Path(runtime_dir) / "src" / "auth" / "storage.ts"
    source = path.read_text(encoding="utf-8")
    if "glossarion-token-storage.mjs" in source:
        _write_adapter(path.parent)
        return
    replacements = {
        "const data = await file.json();": "const data = JSON.parse(decryptSerialized(await file.text()));",
        "await Bun.write(ACCOUNTS_FILE, JSON.stringify(config, null, 2));":
            "const temporary = ACCOUNTS_FILE + '.' + randomUUID() + '.tmp';\n    try {\n      await writeFile(temporary, encryptSerialized(JSON.stringify(config, null, 2)), {mode:0o600});\n      await rename(temporary, ACCOUNTS_FILE);\n    } finally { await unlink(temporary).catch(() => {}); }",
        'console.error("Failed to load accounts:", e);': 'throw e;',
        'console.error("Failed to save accounts:", e);': 'throw e;',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError("Antigravity credential adapter is incompatible; refusing plaintext storage")
        source = source.replace(old, new)
    source = 'import {encryptSerialized, decryptSerialized} from "./glossarion-token-storage.mjs";\nimport {writeFile, rename, unlink} from "node:fs/promises";\nimport {randomUUID} from "node:crypto";\n' + source
    _write_adapter(path.parent)
    path.write_text(source, encoding="utf-8")


def patch_ocagy(plugin_root):
    path = Path(plugin_root) / "dist" / "src" / "plugin" / "storage.js"
    source = path.read_text(encoding="utf-8")
    if "glossarion-token-storage.mjs" in source:
        _write_adapter(path.parent)
        _patch_ocagy_oauth(plugin_root)
        return
    replacements = {
        'const content = await fs.readFile(path, "utf-8");': ('const content = decryptSerialized(await fs.readFile(path, "utf-8"));', 2),
        'const content = JSON.stringify(merged, null, 2);': ('const content = encryptSerialized(JSON.stringify(merged, null, 2));', 1),
        'const content = JSON.stringify(storage, null, 2);': ('const content = encryptSerialized(JSON.stringify(storage, null, 2));', 1),
        'await fs.writeFile(path, JSON.stringify({ version: 4, accounts: [], activeIndex: 0 }, null, 2),': ('await fs.writeFile(path, encryptSerialized(JSON.stringify({ version: 4, accounts: [], activeIndex: 0 }, null, 2)),', 1),
    }
    for old, (new, count) in replacements.items():
        if source.count(old) != count:
            raise RuntimeError("OcAgy credential adapter is incompatible; refusing plaintext storage")
        source = source.replace(old, new)
    # Do not turn unreadable credentials into an empty store that may overwrite them.
    source = source.replace('catch (error) {', 'catch (error) {\n        if (error.code === "GLOSSARION_CREDENTIAL_CRYPTO") throw error;')
    source = 'import {encryptSerialized, decryptSerialized} from "./glossarion-token-storage.mjs";\n' + source
    _write_adapter(path.parent)
    path.write_text(source, encoding="utf-8")
    _patch_ocagy_oauth(plugin_root)


def _patch_ocagy_oauth(plugin_root):
    root = Path(plugin_root) / "dist" / "src"
    paths = [root / "plugin" / "auth.js", root / "antigravity" / "oauth.js"]
    staged = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        if path.name == "oauth.js":
            # OpenCode persists the OAuth callback's access field too. Follow
            # the plugin's stored-account login path: refresh into memory on use.
            source = source.replace("access: tokenPayload.access_token,", 'access: "",')
            source = source.replace("expires: calculateTokenExpiry(startTime, tokenPayload.expires_in),", "expires: 0,")
        if "glossarion-token-storage.mjs" in source:
            staged.append((path, source))
            continue
        if path.name == "auth.js":
            old = 'return parts.managedProjectId ? `${base}|${parts.managedProjectId}` : base;'
            if source.count(old) != 1 or source.count('export function parseRefreshParts(refresh) {') != 1:
                raise RuntimeError("OcAgy OAuth credential adapter is incompatible")
            source = source.replace(old, 'return encryptSerialized(JSON.stringify(parts.managedProjectId ? `${base}|${parts.managedProjectId}` : base));')
            source = source.replace('export function parseRefreshParts(refresh) {', 'export function parseRefreshParts(refresh) {\n    if (refresh?.startsWith("GLSE1:")) refresh = JSON.parse(decryptSerialized(refresh));')
            source = 'import {encryptSerialized, decryptSerialized} from "./glossarion-token-storage.mjs";\n' + source
        else:
            if source.count('refresh: storedRefresh,') != 1:
                raise RuntimeError("OcAgy login credential adapter is incompatible")
            source = source.replace('refresh: storedRefresh,', 'refresh: encryptSerialized(JSON.stringify(storedRefresh)),')
            source = 'import {encryptSerialized} from "../plugin/glossarion-token-storage.mjs";\n' + source
        staged.append((path, source))
    for path, source in staged:
        path.write_text(source, encoding="utf-8")


def migrate_opencode_oauth(path):
    """Encrypt only Google's OAuth refresh value; leave other providers untouched."""
    path = Path(path)
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    google = value.get("google", {})
    if google.get("type") != "oauth":
        return
    refresh = google.get("refresh")
    changed = False
    if refresh and not refresh.startswith("GLSE1:"):
        prepare_key()
        google["refresh"] = token_encryption.encrypt_tokens(refresh).decode("ascii")
        changed = True
    if google.get("access"):
        google["access"], google["expires"] = "", 0
        changed = True
    if changed:
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
