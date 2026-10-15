# GUI-free release checks and SHA-256 downloads shared by desktop and mobile.
"""GitHub-primary release discovery with a stable-tag GitLab fallback.

The desktop consumes normalized GitHub-shaped releases and verified downloads.
Mobile reuses discovery, cache and skipped-version config, and opens download
links in the system browser. SHA-256 verifies integrity against HTTPS metadata;
it does not authenticate a publisher or replace code signing.
"""
import os
import sys
import time
import re
import hashlib
import json
from urllib.parse import quote, urlparse, urljoin
from datetime import datetime, timezone
from typing import Any, Optional, Dict, Tuple, List

import requests
from packaging import version

__all__ = ["UpdateCoreMixin", "HeadlessUpdateChecker", "UpdateVerificationError", "UpdateCancelled"]


class UpdateVerificationError(ValueError):
    """Untrusted/missing checksum metadata or downloaded bytes; never retry another provider."""


class UpdateCancelled(Exception):
    """Cancellation must never start a mirror download or installation."""


class UpdateCoreMixin:
    """Release checks without a GUI (see the module docstring for the host contract)."""

    MIN_UPDATE_SIZE = 60_000_000  # 60 MB (decimal bytes)

    @classmethod
    def _eligible_update_asset(cls, asset):
        try:
            return (cls._safe_asset_name(asset.get("name", ""))
                    and "glossarion" in asset.get("name", "").casefold()
                    and int(asset.get("size", 0)) >= cls.MIN_UPDATE_SIZE)
        except (TypeError, ValueError):
            return False

    @classmethod
    def _validate_update_file(cls, file_path):
        if "glossarion" not in os.path.basename(file_path).casefold():
            raise ValueError("Update filename must contain Glossarion.")
        if not os.path.isfile(file_path):
            raise ValueError("Update file was not saved.")
        size = os.path.getsize(file_path)
        if size < cls.MIN_UPDATE_SIZE:
            raise ValueError("Update file must be at least 60 MB (60,000,000 bytes).")
        return size

    GITHUB_API_URL = "https://api.github.com/repos/Shirochi-stack/Glossarion/releases"
    GITHUB_LATEST_URL = "https://api.github.com/repos/Shirochi-stack/Glossarion/releases/latest"

    def _detect_build_variant(self) -> str:
        """Detect which build variant is currently running from the exe filename.
        
        Returns one of: 'OmegaLite', 'SuperLite', 'TurboLite', 'Lite', 'NoCuda', 'Standard'
        """
        try:
            if getattr(sys, 'frozen', False):
                exe_name = os.path.basename(sys.executable).lower()
            else:
                # Dev mode: check APP_NAME from the running spec if available
                exe_name = ''
            
            if 'omegalite' in exe_name:
                return 'OmegaLite'
            elif 'superlite' in exe_name:
                return 'SuperLite'
            elif 'turbolite' in exe_name:
                return 'TurboLite'
            elif 'nocuda' in exe_name or 'no_cuda' in exe_name or exe_name.startswith('n_'):
                return 'NoCuda'
            elif 'lite' in exe_name:
                return 'Lite'
            else:
                return 'Standard'
        except Exception:
            return 'Standard'  # Safe default

    @staticmethod
    def _detect_arch() -> str:
        """Detect current CPU architecture for choosing native macOS assets."""
        try:
            import platform
            machine = (platform.machine() or '').lower()
            if machine in ('arm64', 'aarch64'):
                return 'arm64'
            if machine in ('x86_64', 'amd64', 'i386', 'i686'):
                return 'x64'
        except Exception:
            pass
        return 'unknown'

    @staticmethod
    def _detect_platform() -> str:
        """Detect the current operating system.
        
        Returns one of: 'windows', 'macos', 'linux'
        """
        if sys.platform == 'win32':
            return 'windows'
        elif sys.platform == 'darwin':
            return 'macos'
        else:
            return 'linux'

    @staticmethod
    def _asset_platform(asset_name: str) -> str:
        """Determine which platform a release asset targets from its filename.
        
        Returns one of: 'windows', 'macos', 'linux', 'unknown'
        """
        name_lower = asset_name.lower()
        if name_lower.endswith('.exe'):
            return 'windows'
        elif name_lower.endswith('.dmg'):
            return 'macos'
        elif name_lower.endswith(('.appimage', '.deb', '.rpm', '.tar.gz')):
            return 'linux'
        # Check for platform keywords in filename
        if 'macos' in name_lower or 'darwin' in name_lower or 'osx' in name_lower:
            return 'macos'
        if 'linux' in name_lower:
            return 'linux'
        if 'windows' in name_lower or 'win' in name_lower:
            return 'windows'
        return 'unknown'

    @staticmethod
    def _asset_arch(asset_name: str) -> str:
        """Determine architecture targeted by a release asset filename."""
        name_lower = asset_name.lower()
        if any(token in name_lower for token in ('intel', 'x86_64', 'x64', 'amd64')):
            return 'x64'
        if any(token in name_lower for token in ('arm64', 'aarch64', 'apple_silicon', 'apple-silicon')):
            return 'arm64'
        if name_lower.endswith('.dmg') and 'mac' in name_lower:
            # Current mac naming uses *_MAC.dmg for Apple Silicon and *_MAC_Intel.dmg for Intel.
            return 'arm64'
        return 'unknown'

    GITLAB_API_URL = "https://gitlab.com/api/v4/projects/87623444"
    INSTALLABLE_SUFFIXES = ('.exe', '.dmg', '.appimage', '.deb', '.tar.gz')

    @staticmethod
    def _release_version(tag):
        return version.Version(str(tag).removeprefix('v'))

    @staticmethod
    def _safe_asset_name(name):
        return (isinstance(name, str) and name not in ('', '.', '..')
                and not name.endswith((' ', '.'))
                and not re.search(r'[<>:"/\\|?*%!\x00-\x1f]', name))

    @staticmethod
    def _https_url(url):
        parts = urlparse(str(url))
        if parts.scheme != 'https' or not parts.hostname or parts.username or parts.password:
            raise UpdateVerificationError('Update URLs must use HTTPS without embedded credentials.')
        return str(url)

    @classmethod
    def _asset_url(cls, url, provider):
        cls._https_url(url)
        parts = urlparse(url)
        hosts = ('github.com', 'api.github.com') if provider == 'github' else ('gitlab.com',)
        if parts.hostname not in hosts or parts.port not in (None, 443):
            raise UpdateVerificationError('Update asset is outside the configured release provider.')
        return url

    def _check_abort(self):
        if getattr(self, '_cancel_update_check', lambda: False)():
            raise UpdateCancelled()
        deadline = getattr(self, '_update_check_deadline', None)
        if deadline is not None and time.monotonic() >= deadline:
            raise requests.Timeout('Update check exceeded its time budget. Please try again later.')

    def _get(self, url, accept='application/octet-stream', **kwargs):
        """No credentials; TLS verification stays enabled, including frozen builds."""
        for _ in range(6):
            self._check_abort()
            deadline = getattr(self, '_update_check_deadline', None)
            remaining = max(0.1, deadline - time.monotonic()) if deadline is not None else None
            timeout = (min(3, remaining), min(5, remaining)) if remaining is not None else (10, 30)
            self._https_url(url)
            response = requests.get(url, headers={'User-Agent': 'Glossarion-Updater',
                                                 'Accept': accept},
                                    timeout=timeout, verify=True, allow_redirects=False, **kwargs)
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get('Location')
                response.close()
                if not location:
                    raise ValueError('Update redirect has no destination.')
                url = self._https_url(urljoin(url, location))
                continue
            try:
                response.raise_for_status()
            except Exception:
                response.close()
                raise
            return response
        raise requests.TooManyRedirects('Too many update download redirects.')

    def _json(self, url, limit=2_000_000):
        # Bound decoded metadata before parsing, including chunked responses.
        with self._get(url, accept='application/json', stream=True) as response:
            chunks, length = [], 0
            for chunk in response.iter_content(65536):
                self._check_abort()
                length += len(chunk)
                if length > limit:
                    raise UpdateVerificationError('Update metadata exceeds the allowed size.')
                chunks.append(chunk)
        try:
            return json.loads(b''.join(chunks))
        except (ValueError, UnicodeError) as exc:
            raise ValueError('Invalid update metadata JSON.') from exc

    @staticmethod
    def _can_fallback(exc):
        # A certificate failure or bad checksum is a hard stop, never an outage workaround.
        if isinstance(exc, (UpdateVerificationError, requests.exceptions.SSLError)):
            return False
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(exc, requests.HTTPError):
            status = getattr(exc.response, 'status_code', 0) or 0
            return status in (403, 404, 429) or 500 <= status < 600
        return isinstance(exc, (ValueError, KeyError, TypeError))

    def _normalise_release(self, raw, provider, hydrate=True):
        if not isinstance(raw, dict):
            raise ValueError('Invalid release response.')
        tag = raw.get('tag_name')
        if not isinstance(tag, str) or not tag:
            raise ValueError('Release has no valid version tag.')
        parsed = self._release_version(tag)
        if parsed.is_prerelease or parsed.is_devrelease or raw.get('draft') or raw.get('prerelease'):
            raise ValueError('Release is not in the stable update channel.')
        if provider == 'gitlab' and raw.get('released_at'):
            if not isinstance(raw['released_at'], str):
                raise ValueError('Invalid release publication date.')
            published = datetime.fromisoformat(raw['released_at'].replace('Z', '+00:00'))
            if published.replace(tzinfo=published.tzinfo or timezone.utc).timestamp() > time.time():
                raise ValueError('Scheduled release is not published yet.')
        result = dict(raw)
        if provider == 'gitlab':
            if not isinstance(raw.get('assets', {}), dict) or not isinstance(raw.get('_links', {}), dict):
                raise ValueError('Invalid GitLab release structure.')
            result.update(body=raw.get('description', ''), published_at=raw.get('released_at', ''),
                          html_url=raw.get('_links', {}).get('self') or
                          'https://gitlab.com/shirochi-stack/Glossarion/-/releases/' + quote(tag, safe=''))
            raw_assets = raw.get('assets', {}).get('links', [])
        else:
            raw_assets = raw.get('assets', [])
        if not isinstance(raw_assets, list) or len(raw_assets) > 200:
            raise ValueError('Invalid release asset list.')
        assets = []
        for item in raw_assets:
            if not isinstance(item, dict) or not self._safe_asset_name(item.get('name')):
                continue
            asset = dict(item)
            asset['browser_download_url'] = item.get('browser_download_url') if provider == 'github' else item.get('direct_asset_url') or item.get('url')
            asset['_provider'] = provider
            asset['_release_tag'] = tag
            assets.append(asset)
        result.update(assets=assets, _provider=provider)
        if not isinstance(result.get('body') or '', str):
            raise ValueError('Invalid release notes.')
        result['body'] = re.sub(r'\n{3,}', '\n\n', result.get('body') or '')
        # GitLab links have no size. Read the matching per-file consumer metadata.
        if provider == 'gitlab' and hydrate:
            for asset in assets:
                if asset['name'].lower().endswith(self.INSTALLABLE_SUFFIXES):
                    try:
                        prepared = self._prepare_update_asset(asset, result)
                        asset.update(prepared)
                    except (requests.RequestException, ValueError) as exc:
                        asset['_verification_error'] = str(exc)
                        asset['size'] = 0
        return result

    def _gitlab_release_for_tag(self, tag, hydrate=True):
        raw = self._json(self.GITLAB_API_URL + '/releases/' + quote(tag, safe=''))
        if not isinstance(raw, dict) or raw.get('tag_name') != tag:
            raise UpdateVerificationError('GitLab returned a different release tag than requested.')
        return self._normalise_release(raw, 'gitlab', hydrate)

    def _gitlab_releases(self, count=10):
        releases, candidates = [], {}
        # Tag ordering is numeric, not lexical/date order. Follow pages; fail explicitly
        # if an unexpectedly large tag set exceeds the bounded discovery window.
        for page in range(1, 21):
            self._check_abort()
            tags = self._json(self.GITLAB_API_URL + '/repository/tags?order_by=version&sort=desc&search=%5Ev&per_page=100&page=' + str(page))
            if not isinstance(tags, list):
                raise ValueError('Invalid GitLab tag list.')
            for tag in tags:
                try:
                    parsed = self._release_version(tag['name'])
                except (ValueError, KeyError, TypeError):
                    continue
                if parsed.is_prerelease or parsed.is_devrelease or not tag.get('release'):
                    continue
                candidates[tag['name']] = parsed
            if len(tags) < 100:
                break
        else:
            raise ValueError('GitLab tag discovery exceeded 2000 tags; retry via the release page.')
        for tag in sorted(candidates, key=candidates.get, reverse=True):
            self._check_abort()
            try:
                releases.append(self._gitlab_release_for_tag(tag, hydrate=not releases))
            except requests.HTTPError as exc:
                if getattr(exc.response, 'status_code', 0) != 404:
                    raise
            except UpdateVerificationError:
                raise
            except ValueError:
                continue
            if len(releases) >= count:
                break
        return releases

    def fetch_multiple_releases(self, count=10):
        try:
            if getattr(self, '_release_provider', 'github') == 'gitlab':
                return self._gitlab_releases(count)
            raw = self._json(self.GITHUB_API_URL + '?per_page=' + str(count))
            if not isinstance(raw, list):
                raise ValueError('Invalid GitHub release list.')
            releases = []
            for item in raw:
                try:
                    releases.append(self._normalise_release(item, 'github'))
                except ValueError:
                    continue
            return sorted(releases, key=lambda item: self._release_version(item['tag_name']), reverse=True)
        except (requests.RequestException, ValueError, TypeError):
            return []

    def _check_for_updates_core(self, force_show=False):
        self._update_check_performed = False
        if not force_show and time.time() - self._last_check_time < self._check_cache_duration:
            return False, None
        self._update_check_performed = True
        self.update_available = False
        self._update_check_deadline = time.monotonic() + getattr(self, '_update_check_timeout', 30)
        history = None
        try:
            try:
                release = self._normalise_release(self._json(self.GITHUB_LATEST_URL), 'github')
                self._release_provider = 'github'
                if (getattr(self, '_build_variant', '') != 'Mobile'
                        and self._release_version(release['tag_name']) > self._release_version(self.CURRENT_VERSION)
                        and (force_show or release['tag_name'] not in self.main_gui.config.get('skipped_versions', []))
                        and not any(self._compatible_update_asset(a) for a in release['assets'])):
                    # Missing binaries can use the same tag on GitLab; never substitute
                    # another version merely because GitHub has no build for this OS.
                    try:
                        mirror = self._gitlab_release_for_tag(release['tag_name'])
                        if any(self._eligible_update_asset(a) and self._compatible_update_asset(a)
                               for a in mirror['assets']):
                            release, self._release_provider = mirror, 'gitlab'
                            history = [mirror]
                    except (requests.RequestException, ValueError) as exc:
                        if not self._can_fallback(exc):
                            raise
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                if not self._can_fallback(exc):
                    raise
                releases = self._gitlab_releases()
                if not releases:
                    raise ValueError('GitHub is unavailable and GitLab has no stable tagged releases.') from exc
                release = releases[0]
                self._release_provider = 'gitlab'
                history = releases
            self.latest_release = release
            history = history if history is not None else self.fetch_multiple_releases()
            # The downloader must use the release we checked, not history's first row.
            self.all_releases = [release] + [r for r in history if r['tag_name'] != release['tag_name']]
            if not force_show and release['tag_name'] in self.main_gui.config.get('skipped_versions', []):
                return False, None
            self.update_available = self._release_version(release['tag_name']) > self._release_version(self.CURRENT_VERSION)
            return self.update_available, release if self.update_available else None
        finally:
            self._save_last_check_time()
            self._update_check_deadline = None

    def _compatible_update_asset(self, asset):
        name = asset.get('name', '')
        arch = self._asset_arch(name)
        current_arch = self._detect_arch()
        return (self._safe_asset_name(name) and name.lower().endswith(self.INSTALLABLE_SUFFIXES)
                and self._asset_platform(name) == self._detect_platform()
                and (arch == 'unknown' or current_arch == 'unknown' or arch == current_arch))

    def _prepare_update_asset(self, asset, release):
        """Resolve SHA-256 before downloading. Missing metadata is never size-only success."""
        prepared = dict(asset)
        name, tag = asset.get('name', ''), release.get('tag_name', '')
        if not self._safe_asset_name(name) or 'glossarion' not in name.casefold():
            raise UpdateVerificationError('Unsafe update filename.')
        provider = release.get('_provider', 'github')
        self._asset_url(asset.get('browser_download_url', ''), provider)
        native_digest = asset.get('digest')
        digest = None
        if native_digest is not None:
            if not isinstance(native_digest, str) or not re.fullmatch(r'sha256:[0-9a-fA-F]{64}', native_digest):
                raise UpdateVerificationError('Invalid release asset SHA-256 digest.')
            digest = native_digest.split(':', 1)[1].lower()
        metadata = [a for a in release.get('assets', []) if a.get('name') == name + '.sha256.json']
        if len(metadata) > 1:
            raise UpdateVerificationError('Duplicate update checksum metadata.')
        expected_size = asset.get('size', 0)
        if metadata:
            url = self._asset_url(metadata[0].get('browser_download_url', ''), provider)
            data = self._json(url, limit=65536)
            if (not isinstance(data, dict) or data.get('schema_version') != 1 or data.get('tag') != tag
                    or data.get('name') != name or not isinstance(data.get('size'), int)
                    or isinstance(data.get('size'), bool) or data['size'] <= 0
                    or not isinstance(data.get('sha256'), str)
                    or not re.fullmatch(r'[0-9a-fA-F]{64}', data['sha256'])):
                raise UpdateVerificationError('Checksum metadata does not match this release asset.')
            if expected_size and int(expected_size) != data['size']:
                raise UpdateVerificationError('Checksum metadata size conflicts with the release asset.')
            if digest and digest != data['sha256'].lower():
                raise UpdateVerificationError('Checksum metadata conflicts with the provider SHA-256 digest.')
            digest, expected_size = data['sha256'].lower(), data['size']
        if not digest:
            raise UpdateVerificationError('This release has no published SHA-256 for this file. Automatic download and installation are blocked; see the release page.')
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < self.MIN_UPDATE_SIZE:
            raise UpdateVerificationError('Update must be at least 60 MB with an exact published size.')
        prepared.update(size=expected_size, _sha256=digest, _release_tag=tag, _provider=provider)
        return prepared

    def _mirror_update_asset(self, prepared):
        release = self._gitlab_release_for_tag(prepared['_release_tag'])
        matches = [a for a in release['assets'] if a['name'] == prepared['name']]
        if len(matches) != 1:
            raise UpdateVerificationError('GitLab has no unambiguous mirror of this exact release file.')
        mirror = self._prepare_update_asset(matches[0], release)
        if mirror['_sha256'] != prepared['_sha256'] or mirror['size'] != prepared['size']:
            raise UpdateVerificationError('GitLab mirror differs from the selected GitHub release file.')
        return mirror

    @classmethod
    def _verify_downloaded_update(cls, file_path, asset):
        size = cls._validate_update_file(file_path)
        if not asset or not re.fullmatch(r'[0-9a-f]{64}', asset.get('_sha256', '')):
            raise UpdateVerificationError('No SHA-256 verification metadata for this download.')
        if size != asset['size']:
            raise UpdateVerificationError('Update size does not match the published size.')
        digest = hashlib.sha256()
        with open(file_path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != asset['_sha256']:
            raise UpdateVerificationError('SHA-256 mismatch. Update rejected; installation is blocked.')
        return size

    def _download_verified_update(self, asset, release, destination, progress, cancelled):
        prepared = self._prepare_update_asset(asset, release)
        if not self._compatible_update_asset(prepared):
            raise UpdateVerificationError('Update does not match this platform/architecture.')
        part = destination + '.part'
        try:
            for attempt in range(2):
                digest, downloaded = hashlib.sha256(), 0
                try:
                    if cancelled():
                        raise UpdateCancelled()
                    with self._get(prepared['browser_download_url'], stream=True) as response:
                        with open(part, 'wb') as stream:
                            for chunk in response.iter_content(65536):
                                if cancelled():
                                    raise UpdateCancelled()
                                downloaded += len(chunk)
                                if downloaded > prepared['size']:
                                    raise UpdateVerificationError('Update exceeds the published size.')
                                stream.write(chunk)
                                digest.update(chunk)
                                progress(min(100, downloaded * 100 // prepared['size']), downloaded / 1048576, prepared['size'] / 1048576)
                    if cancelled():
                        raise UpdateCancelled()
                    if downloaded != prepared['size'] or digest.hexdigest() != prepared['_sha256']:
                        raise UpdateVerificationError('SHA-256 or exact size mismatch. Installation is blocked.')
                    os.replace(part, destination)
                    return prepared
                except requests.RequestException as exc:
                    if cancelled():
                        raise UpdateCancelled() from exc
                    if attempt or prepared['_provider'] != 'github' or not self._can_fallback(exc):
                        raise
                    prepared = self._mirror_update_asset(prepared)
        finally:
            if os.path.isfile(part):
                os.remove(part)


    def _save_last_check_time(self):
        """Save the last update check time to config"""
        try:
            current_time = time.time()
            self._last_check_time = current_time
            self.main_gui.config['last_update_check_time'] = current_time
            # Save config without showing message
            self.main_gui.save_config(show_message=False)
        except Exception as e:
            print(f"[DEBUG] Failed to save last check time: {e}")

    def _skip_latest_version(self):
        """Add the latest release's tag to ``skipped_versions`` and save (the config part of
        UpdateManager.skip_version). Returns the tag."""
        # Get current skipped versions list
        if 'skipped_versions' not in self.main_gui.config:
            self.main_gui.config['skipped_versions'] = []
        
        # Add this version to skipped list
        version_tag = self.latest_release['tag_name']
        if version_tag not in self.main_gui.config['skipped_versions']:
            self.main_gui.config['skipped_versions'].append(version_tag)
        
        # Save config
        self.main_gui.save_config(show_message=False)
        return version_tag


class HeadlessUpdateChecker(UpdateCoreMixin):
    """The ``UpdateCoreMixin`` host without a GUI (the mobile About > Updates screen, tests).

    ``config_host`` is the ``main_gui`` the moved methods use: a ``config`` dict and
    ``save_config(show_message=False)``. The attributes are the ones ``UpdateManager.__init__``
    sets for the moved methods (same defaults: a 30-minute check cache)."""

    def __init__(self, config_host: Any, current_version: str, *, build_variant: Optional[str] = None) -> None:
        self.main_gui = config_host
        self.update_available = False
        self.latest_release = None
        self.all_releases = []
        self._last_check_time = self.main_gui.config.get('last_update_check_time', 0)
        self._check_cache_duration = 1800  # Cache for 30 minutes
        self._build_variant = build_variant if build_variant is not None else self._detect_build_variant()
        self.CURRENT_VERSION = str(current_version or "0.0.0")
