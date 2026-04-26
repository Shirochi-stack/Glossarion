"""Minimal test: use console.anthropic.com for BOTH auth + redirect_uri."""
import requests, json, sys, hashlib, base64, secrets, webbrowser
from urllib.parse import urlencode, urlparse, parse_qs

CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
# Use the OLD redirect_uri in the auth request itself
REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
AUTH_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
SCOPES = "user:profile user:inference org:create_api_key"

raw = secrets.token_bytes(32)
code_verifier = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
state = secrets.token_urlsafe(32)

params = {
    "client_id": CLIENT_ID, "response_type": "code",
    "redirect_uri": REDIRECT_URI, "scope": SCOPES,
    "state": state, "code_challenge": code_challenge,
    "code_challenge_method": "S256",
}
auth_url = f"{AUTH_URL}?{urlencode(params)}"
print(f"redirect_uri in auth request: {REDIRECT_URI}")
print(f"Token endpoint: {TOKEN_URL}")
print(f"\nOpening browser...\n")
webbrowser.open(auth_url)

print("IMPORTANT: Check the ACTUAL URL your browser lands on after login!")
print("Paste the callback URL or the copied code:\n")
raw_input = input("> ").strip()

if raw_input.startswith("http"):
    # Print what domain the redirect actually went to
    actual_host = urlparse(raw_input).netloc
    print(f"\n*** Actual redirect went to: {actual_host} ***")
    parsed = urlparse(raw_input)
    code = parse_qs(parsed.query).get("code", [None])[0]
    if not code:
        code = parse_qs(parsed.fragment).get("code", [None])[0]
elif "#" in raw_input:
    code = raw_input.split("#")[0]
else:
    code = raw_input

print(f"Auth code: {code}")

# Try with the redirect_uri we actually sent in the auth request
payload = {
    "grant_type": "authorization_code",
    "client_id": CLIENT_ID,
    "code": code,
    "redirect_uri": REDIRECT_URI,
    "code_verifier": code_verifier,
}

print(f"\n--- Test 1: data= (form-encoded) ---")
resp = requests.post(TOKEN_URL, data=payload, timeout=15)
print(f"Status: {resp.status_code}")
try:
    print(f"Body: {json.dumps(resp.json())[:400]}")
except:
    print(f"Body: {resp.text[:400]}")

if resp.status_code >= 400:
    print(f"\n--- Test 2: json= ---")
    resp2 = requests.post(TOKEN_URL, json=payload, timeout=15)
    print(f"Status: {resp2.status_code}")
    try:
        print(f"Body: {json.dumps(resp2.json())[:400]}")
    except:
        print(f"Body: {resp2.text[:400]}")

    # Also try with the platform redirect_uri in token exchange
    print(f"\n--- Test 3: data= with platform.claude.com redirect_uri ---")
    payload2 = {**payload, "redirect_uri": "https://platform.claude.com/oauth/code/callback"}
    resp3 = requests.post(TOKEN_URL, data=payload2, timeout=15)
    print(f"Status: {resp3.status_code}")
    try:
        print(f"Body: {json.dumps(resp3.json())[:400]}")
    except:
        print(f"Body: {resp3.text[:400]}")

    # Also try platform token URL with platform redirect
    print(f"\n--- Test 4: platform URL + platform redirect + data= ---")
    resp4 = requests.post("https://platform.claude.com/v1/oauth/token", data=payload2, timeout=15)
    print(f"Status: {resp4.status_code}")
    try:
        print(f"Body: {json.dumps(resp4.json())[:400]}")
    except:
        print(f"Body: {resp4.text[:400]}")
else:
    print("\n*** SUCCESS! ***")
    print(json.dumps(resp.json(), indent=2)[:500])
