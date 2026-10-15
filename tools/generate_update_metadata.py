"""Create per-file SHA-256 consumer metadata after final packaging/signing.

No upload, signing, credentials, or external dependencies. Each variant has a
unique sidecar name so parallel release jobs never overwrite a shared manifest.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import runpy


def metadata_bytes(path: Path, tag: str, expected_sha256: str | None = None) -> bytes:
    path = Path(path)
    if not re.fullmatch(r'v[0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9_.+-]*)', tag):
        raise ValueError('A version tag such as v9.14.4 is required.')
    if path.is_symlink() or not path.is_file():
        raise ValueError('Update artifact must be a regular file, not a symlink.')
    if re.search(r'[<>:"/\\|?*%!\x00-\x1f]', path.name) or path.name.endswith((' ', '.')):
        raise ValueError('Unsafe artifact filename.')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    value = digest.hexdigest()
    if expected_sha256 is not None and value != expected_sha256:
        raise ValueError('Artifact changed before checksum publication.')
    return (json.dumps({'schema_version': 1, 'tag': tag, 'name': path.name,
                        'size': path.stat().st_size, 'sha256': value}, indent=2) + '\n').encode('utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('patterns', nargs='+', help='Artifact paths/globs, quoted on every shell.')
    parser.add_argument('--tag', help='Default: GitHub tag, otherwise source VERSION_TAG.')
    args = parser.parse_args(argv)
    source_tag = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'src/app_version.py'))['VERSION_TAG']
    tag = args.tag or (os.environ.get('GITHUB_REF_NAME') if os.environ.get('GITHUB_REF_TYPE') == 'tag' else source_tag)
    if tag != source_tag:
        raise ValueError('Release tag differs from src/app_version.py; bump the packaged version first.')
    paths = sorted({Path(p) for pattern in args.patterns for p in glob.glob(pattern)})
    if not paths:
        raise ValueError('No artifacts matched; refusing an empty checksum output.')
    for path in paths:
        output = Path(str(path) + '.sha256.json')
        output.write_bytes(metadata_bytes(path, tag))
        print('Generated SHA-256 metadata: ' + output.name)


if __name__ == '__main__':
    main()
