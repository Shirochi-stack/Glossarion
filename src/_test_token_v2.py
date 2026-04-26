"""Test claude.ai token endpoint with browser-like headers to bypass Cloudflare."""
import requests, json, sys, hashlib, base64, secrets, webbrowser
from urllib.parse import urlencode, urlparse, parse_qs

CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTH_URL = "https://claude.ai/oauth/authorize"
SCOPES = "user:profile user:inference org:create_api_key"

# Try multiple redirect URIs
REDIRECT_URIS = [
    "https://platform.claude.com/oauth/code/callback",
    "https://console.anthropic.com/oauth/code/callback",
]
AUTH_REDIRECT = REDIRECT_URIS[0]

# Generate PKCE
raw = secrets.token_bytes(32)
code_verifier = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
state = secrets.token_urlsafe(32)

params = {
    "client_id": CLIENT_ID, "response_type": "code",
    "redirect_uri": AUTH_REDIRECT, "scope": SCOPES,
    "state": state, "code_challenge": code_challenge,
    "code_challenge_method": "S256",
}
auth_url = f"{AUTH_URL}?{urlencode(params)}"
print(f"Opening browser...\n")
webbrowser.open(auth_url)

print("Paste the callback URL or copied code:\n")
raw_input = input("> ").strip()

if raw_input.startswith("http"):
    parsed = urlparse(raw_input)
    code = parse_qs(parsed.query).get("code", [None])[0]
    if not code:
        code = parse_qs(parsed.fragment).get("code", [None])[0]
elif "#" in raw_input:
    code = raw_input.split("#")[0]
else:
    code = raw_input

print(f"\nAuth code: {code}")

# Browser-like headers to bypass Cloudflare
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://claude.ai",
    "Referer": "https://claude.ai/",
}

TOKEN_URLS = [
    "https://claude.ai/api/auth/oauth/token",
    "https://console.anthropic.com/v1/oauth/token",
    "https://platform.claude.com/v1/oauth/token",
]

for url in TOKEN_URLS:
    for ruri in REDIRECT_URIS:
        payload = {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "redirect_uri": ruri,
            "code_verifier": code_verifier,
        }
        for enc_name, kw in [("data", {"data": payload}), ("json", {"json": payload})]:
            print(f"\n{'='*70}")
            print(f"  {enc_name:4s}  {url}")
            print(f"  redir: {ruri}")
            try:
                resp = requests.post(url, headers=BROWSER_HEADERS, timeout=15, **kw)
                print(f"  Status: {resp.status_code}")
                ct = resp.headers.get("Content-Type", "")
                if "json" in ct:
                    body = resp.json()
                    print(f"  Body: {json.dumps(body)[:300]}")
                else:
                    print(f"  Body: {resp.text[:200]}")
                if resp.status_code < 400:
                    print("\n  *** SUCCESS! ***")
                    print(f"  URL:      {url}")
                    print(f"  encoding: {enc_name}")
                    print(f"  redir:    {ruri}")
                    sys.exit(0)
            except Exception as e:
                print(f"  Error: {e}")

print("\n\nAll failed.")
