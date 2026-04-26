"""Quick diagnostic: try all URL + encoding combos for the token exchange."""
import requests, json, sys

CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

# The auth code to test (paste from the callback URL)
if len(sys.argv) < 3:
    print("Usage: python _test_token_exchange.py <AUTH_CODE> <CODE_VERIFIER>")
    print("\nTo get these values, add these prints in authcd_auth.py start_oauth_flow():")
    print('  print(f"CODE_VERIFIER={code_verifier}")')
    print("Then run the login, copy the code + verifier, and run this script.")
    sys.exit(1)

auth_code = sys.argv[1]
code_verifier = sys.argv[2]

URLS = [
    "https://platform.claude.com/v1/oauth/token",
    "https://console.anthropic.com/v1/oauth/token",
]

REDIRECT_URIS = [
    "https://platform.claude.com/oauth/code/callback",
    "https://console.anthropic.com/oauth/code/callback",
]

payload_base = {
    "grant_type": "authorization_code",
    "client_id": CLIENT_ID,
    "code": auth_code,
    "code_verifier": code_verifier,
}

for url in URLS:
    for ruri in REDIRECT_URIS:
        for encoding in ["json", "data"]:
            payload = {**payload_base, "redirect_uri": ruri}
            print(f"\n{'='*60}")
            print(f"URL:          {url}")
            print(f"redirect_uri: {ruri}")
            print(f"encoding:     {encoding}")
            try:
                if encoding == "json":
                    resp = requests.post(url, json=payload, timeout=15)
                else:
                    resp = requests.post(url, data=payload, timeout=15)
                print(f"Status:       {resp.status_code}")
                try:
                    print(f"Response:     {resp.json()}")
                except:
                    print(f"Response:     {resp.text[:200]}")
                if resp.status_code < 400:
                    print("\n*** SUCCESS! This combination works! ***")
            except Exception as e:
                print(f"Error:        {e}")
