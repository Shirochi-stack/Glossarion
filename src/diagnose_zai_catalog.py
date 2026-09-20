"""Diagnose Z.AI model-catalog 400s.

Runs the billing/balance probe against the live gateway with several request
shapes and prints the status + response body for each, so the exact missing
parameter or header is visible instead of a bare "parameter error".

Usage:  python diagnose_zai_catalog.py [account_id]

Prints no credentials: the JWT stays inside the Bun credential store, exactly
as it does for the real probe.
"""
import json
import os
import subprocess
import sys
import tempfile

import glm_proxy

PROBE = r'''
import os from "node:os";
import { loadCredential } from "./src/auth/store.ts";

const credential = await loadCredential();
if (!credential?.jwt) throw new Error("no JWT in the credential store; sign in again");

const appVersion = process.env.ZCODE_APP_VERSION || "3.14.0";
const platform = `${process.env.ZCODE_IDENTITY_PLATFORM || process.platform}-${process.env.ZCODE_IDENTITY_ARCH || os.arch()}`;
const base = "https://zcode.z.ai/api/v1/zcode-plan/billing/balance";

let identityHeaders = null;
try {
  const [{ buildIdentityHeaders }, { loadConfig }] = await Promise.all([
    import("./src/proxy/identity.ts"),
    import("./src/config/loader.ts"),
  ]);
  const config = loadConfig(process.env.ZCODE_PROXY_CONFIG);
  identityHeaders = buildIdentityHeaders(config.identity);
  delete identityHeaders["X-ZCode-Agent"];
  console.log("identity headers: " + JSON.stringify(Object.keys(identityHeaders)));
  console.log("deviceMid present: " + Boolean(config.identity?.deviceMid));
} catch (error) {
  console.log("identity headers UNAVAILABLE: " + String(error).slice(0, 300));
}

const minimal = {
  "HTTP-Referer": "https://zcode.z.ai",
  "User-Agent": `ZCode/${appVersion}`,
  "X-Title": "Z Code@glossarion",
  "X-ZCode-App-Version": appVersion,
};

const urls = [
  ["app_version only", `${base}?app_version=${appVersion}`],
  ["app_version+platform", `${base}?app_version=${appVersion}&platform=${platform}`],
  ["no params", base],
];
const headerSets = [["minimal", minimal]];
if (identityHeaders) headerSets.push(["identity", identityHeaders]);

for (const [urlLabel, url] of urls) {
  for (const [headerLabel, base_headers] of headerSets) {
    const headers = {
      ...base_headers,
      Authorization: `Bearer ${credential.jwt}`,
      Accept: "application/json",
    };
    try {
      const r = await fetch(url, { headers, signal: AbortSignal.timeout(20000) });
      const t = await r.text();
      console.log(`[${urlLabel} | ${headerLabel}] ${r.status} ${t.slice(0, 200)}`);
    } catch (e) {
      console.log(`[${urlLabel} | ${headerLabel}] THREW ${String(e).slice(0, 160)}`);
    }
  }
}
'''


def main() -> int:
    account = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    runtime_dir = glm_proxy._latest_existing_runtime()
    if not runtime_dir:
        print("No zcode-api runtime is installed yet; start Glossarion once first.")
        return 1
    meta_path = os.path.join(runtime_dir, ".glossarion-runtime.json")
    try:
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        print(f"runtime: {runtime_dir}  (zcode-api {meta.get('version')})")
    except OSError:
        print(f"runtime: {runtime_dir}")

    env = glm_proxy._runtime_env(account)
    env["ZCODE_APP_VERSION"] = glm_proxy.ZCODE_APP_VERSION
    creds = env.get("ZCODE_PROXY_CREDENTIALS_PATH", "")
    print(f"credentials: {creds}  exists={os.path.isfile(creds)}")
    print(f"config:      {env.get('ZCODE_PROXY_CONFIG')}")

    bun = glm_proxy._bun_command()
    if not bun:
        print("bun is not installed; start Glossarion once first.")
        return 1

    with tempfile.NamedTemporaryFile(
        "w", suffix=".ts", dir=runtime_dir, delete=False, encoding="utf-8"
    ) as handle:
        handle.write(PROBE)
        script = handle.name
    try:
        result = subprocess.run(
            [*bun, "run", script],
            cwd=runtime_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
        print(result.stdout)
        if result.stderr.strip():
            print("--- stderr ---")
            print(result.stderr[-3000:])
        return result.returncode
    finally:
        os.unlink(script)


if __name__ == "__main__":
    raise SystemExit(main())
