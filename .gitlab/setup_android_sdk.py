"""Install Google's command-line SDK in this disposable CI checkout, checking its digest."""
from pathlib import Path
import hashlib
import os
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

base = 'https://dl.google.com/android/repository/'
root = Path(os.environ['ANDROID_HOME'])
root.mkdir(parents=True, exist_ok=True)
document = ET.fromstring(urllib.request.urlopen(base + 'repository2-3.xml', timeout=120).read())
package = next(p for p in document.iter('remotePackage') if p.attrib['path'] == 'cmdline-tools;latest')
archive = next(p for p in package.findall('./archives/archive') if p.findtext('host-os') == 'linux')
complete = archive.find('complete')
name = complete.findtext('url')
assert name and '/' not in name and name.endswith('.zip'), 'Unexpected SDK archive location'
download = root / name
with urllib.request.urlopen(base + name, timeout=120) as response, download.open('wb') as stream:
    shutil.copyfileobj(response, stream)
checksum = complete.find('checksum')
algorithm = checksum.attrib.get('type', 'sha1')
assert algorithm in ('sha1', 'sha256'), 'Unsupported SDK checksum'
assert hashlib.file_digest(download.open('rb'), algorithm).hexdigest() == checksum.text.strip(), 'SDK digest mismatch'
with zipfile.ZipFile(download) as sdk:
    assert all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in sdk.namelist())
    sdk.extractall(root / 'unpack')
target = root / 'cmdline-tools/latest'
target.parent.mkdir(exist_ok=True)
shutil.move(str(root / 'unpack/cmdline-tools'), target)
for executable in (target / 'bin').iterdir():
    executable.chmod(0o755)
download.unlink()
print('Verified Android SDK command-line tools:', name)
