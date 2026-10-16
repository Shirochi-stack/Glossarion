"""Create the Android upload keystore for Build Mobile and (optionally) store it as repo secrets.

Every Glossarion APK built after this is signed with the same key, so a new build installs over the
previous one (no uninstall, data kept). Without these secrets the workflow signs with a throw-away
debug key that changes on every run, and Android refuses the update ("package conflicts").

Run it yourself (the password is typed into a hidden prompt and never printed or logged)::

    py -3.12 src/mobile/tools/make_android_keystore.py --set-secrets

It writes ``glossarion-upload.jks`` (PKCS12, alias ``upload``, RSA 4096, 30 years) to
``%USERPROFILE%\\Documents\\Glossarion-signing`` (or ``--out``), refuses to overwrite an existing
keystore, and with ``--set-secrets`` runs ``gh secret set`` for ANDROID_KEYSTORE_BASE64 and
ANDROID_KEYSTORE_PASSWORD (values passed on stdin). Keep the keystore and its password backed up:
losing them means installed copies cannot be updated again (only uninstall + reinstall).

For GitLab, store the SAME keystore as the project's masked, protected CI/CD variables (the GitLab
Android build signs with them, see docs/gitlab_ci.md)::

    py -3.12 src/mobile/tools/make_android_keystore.py --existing --gitlab-variables
"""

from __future__ import annotations

import argparse
import base64
import datetime
import getpass
import os
import subprocess
import sys
from pathlib import Path

REPO = "Shirochi-stack/Glossarion"
GITLAB_REPO = "shirochi-stack/Glossarion"
ALIAS = "upload"


def default_out() -> Path:
    return Path(os.path.expanduser("~")) / "Documents" / "Glossarion-signing" / "glossarion-upload.jks"


def build_keystore(password: str, *, common_name: str = "Glossarion") -> bytes:
    """A PKCS12 keystore holding one RSA 4096 key and its self-signed certificate under ``ALIAS``."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import pkcs12
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365 * 30))
        .sign(key, hashes.SHA256())
    )
    return pkcs12.serialize_key_and_certificates(
        ALIAS.encode(), key, cert, None, serialization.BestAvailableEncryption(password.encode())
    )


def fingerprint(data: bytes, password: str) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.serialization import pkcs12

    _key, cert, _extra = pkcs12.load_key_and_certificates(data, password.encode())
    return cert.fingerprint(hashes.SHA256()).hex(":").upper()


def ask_password() -> str:
    while True:
        first = getpass.getpass("Keystore password (8+ characters, hidden): ")
        if len(first) < 8:
            print("Too short; use at least 8 characters.")
            continue
        if getpass.getpass("Repeat the password: ") != first:
            print("The passwords differ; try again.")
            continue
        return first


def set_secret(name: str, value: str) -> None:
    subprocess.run(["gh", "secret", "set", name, "--repo", REPO], input=value.encode(), check=True)


def set_gitlab_variable(name: str, value: str) -> None:
    subprocess.run(["glab", "variable", "set", name, "--masked", "--protected", "--raw", "-R", GITLAB_REPO],
                   input=value.encode(), check=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=default_out(), help="where to write the keystore")
    parser.add_argument("--set-secrets", action="store_true", help="store it as the repo's Android signing secrets")
    parser.add_argument("--existing", action="store_true", help="use the keystore already at --out (never create one)")
    parser.add_argument("--gitlab-variables", action="store_true", help="store it as the GitLab project's Android signing variables")
    parser.add_argument("--password-env", help=argparse.SUPPRESS)  # tests only: read the password from this env var
    args = parser.parse_args(argv)

    out: Path = args.out
    if args.existing:
        if not out.is_file():
            print(f"{out} does not exist; pass --out with the keystore your installs use.")
            return 1
        data = out.read_bytes()
        password = os.environ[args.password_env] if args.password_env else getpass.getpass("Keystore password (hidden): ")
        try:
            print(f"Certificate SHA-256: {fingerprint(data, password)}")
        except ValueError:
            print("The password does not open this keystore.")
            return 1
        if args.set_secrets:
            set_secret("ANDROID_KEYSTORE_BASE64", base64.b64encode(data).decode())
            set_secret("ANDROID_KEYSTORE_PASSWORD", password)
            print(f"Secrets ANDROID_KEYSTORE_BASE64 and ANDROID_KEYSTORE_PASSWORD set on {REPO}.")
        if args.gitlab_variables:
            set_gitlab_variable("ANDROID_KEYSTORE_BASE64", base64.b64encode(data).decode())
            set_gitlab_variable("ANDROID_KEYSTORE_PASSWORD", password)
            print(f"Masked, protected variables ANDROID_KEYSTORE_BASE64 and ANDROID_KEYSTORE_PASSWORD set on {GITLAB_REPO}.")
        return 0
    if out.exists():
        print(f"{out} already exists; not overwriting it (it may be the key your users' installs depend on).")
        return 1
    password = os.environ[args.password_env] if args.password_env else ask_password()
    data = build_keystore(password)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"Keystore written: {out}")
    print(f"Alias: {ALIAS}")
    print(f"Certificate SHA-256: {fingerprint(data, password)}")
    if args.set_secrets:
        set_secret("ANDROID_KEYSTORE_BASE64", base64.b64encode(data).decode())
        set_secret("ANDROID_KEYSTORE_PASSWORD", password)
        print(f"Secrets ANDROID_KEYSTORE_BASE64 and ANDROID_KEYSTORE_PASSWORD set on {REPO}.")
    if args.gitlab_variables:
        set_gitlab_variable("ANDROID_KEYSTORE_BASE64", base64.b64encode(data).decode())
        set_gitlab_variable("ANDROID_KEYSTORE_PASSWORD", password)
        print(f"Masked, protected variables ANDROID_KEYSTORE_BASE64 and ANDROID_KEYSTORE_PASSWORD set on {GITLAB_REPO}.")
    print("Back up the keystore file and its password somewhere safe (not in the repo).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
