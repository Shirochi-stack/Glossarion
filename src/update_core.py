# update_core.py - GUI-free release checks for Glossarion (desktop UpdateManager + mobile Updates)
"""Release-check logic shared by the desktop ``update_manager.UpdateManager`` (PySide6) and the
mobile app's About > Updates screen.

``UpdateCoreMixin`` holds the members moved verbatim out of ``UpdateManager`` (U9): the
GitHub release URLs, the update-asset size/name checks, the build-variant / platform /
architecture detection and asset classification, ``fetch_multiple_releases``,
``_save_last_check_time`` and two splits:

* ``_check_for_updates_core`` is the try-body of ``UpdateManager._check_for_updates_internal``
  (cache window, skipped versions, ``/releases/latest``, release history, version compare);
  it raises the ``requests`` / ``ValueError`` errors the desktop turns into message boxes.
* ``_skip_latest_version`` is the config part of ``UpdateManager.skip_version``.

A host provides ``main_gui`` (``config`` dict + ``save_config(show_message=False)``),
``CURRENT_VERSION``, ``_last_check_time``, ``_check_cache_duration``, ``latest_release``,
``all_releases`` and ``update_available`` - ``UpdateManager.__init__`` on the desktop,
``HeadlessUpdateChecker`` below everywhere else. Python 3.10+, no Qt.
"""
import os
import sys
import time
import re
from typing import Any, Optional, Dict, Tuple, List

import requests
from packaging import version

__all__ = ["UpdateCoreMixin", "HeadlessUpdateChecker"]


class UpdateCoreMixin:
    """Release checks without a GUI (see the module docstring for the host contract)."""

    MIN_UPDATE_SIZE = 60_000_000  # 60 MB (decimal bytes)

    @classmethod
    def _eligible_update_asset(cls, asset):
        try:
            return ("glossarion" in os.path.basename(asset.get("name", "")).casefold()
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

    def fetch_multiple_releases(self, count=10) -> List[Dict]:
        """Fetch multiple releases from GitHub
        
        Args:
            count: Number of releases to fetch
            
        Returns:
            List of release data dictionaries
        """
        try:
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'Glossarion-Updater'
            }
            
            # Fetch multiple releases with minimal retry logic
            max_retries = 1  # Reduced to prevent hanging
            timeout = 20  # Very short timeout
            
            for attempt in range(max_retries + 1):
                try:
                    response = requests.get(
                        f"{self.GITHUB_API_URL}?per_page={count}", 
                        headers=headers, 
                        timeout=timeout
                    )
                    response.raise_for_status()
                    break  # Success
                except (requests.Timeout, requests.ConnectionError) as e:
                    if attempt == max_retries:
                        raise  # Re-raise after final attempt
                    time.sleep(1)
            
            releases = response.json()
            
            # Process each release's notes
            for release in releases:
                if 'body' in release and release['body']:
                    # Clean up but don't truncate for history viewing
                    body = release['body']
                    # Just clean up excessive newlines
                    body = re.sub(r'\n{3,}', '\n\n', body)
                    release['body'] = body
            
            return releases
            
        except Exception as e:
            print(f"Error fetching releases: {e}")
            return []

    def _check_for_updates_core(self, force_show=False) -> Tuple[bool, Optional[Dict]]:
        """Check GitHub for newer releases (the try-body of UpdateManager._check_for_updates_internal).

        Args:
            force_show: If True, ignore the 30-minute cache and the skipped versions

        Returns:
            Tuple of (update_available, release_info)

        Raises:
            requests.Timeout / requests.ConnectionError / requests.HTTPError / ValueError
        """
        # Check if we need to skip the check due to cache
        current_time = time.time()
        if not force_show and (current_time - self._last_check_time) < self._check_cache_duration:
            return False, None
        
        # Check if this version was previously skipped
        skipped_versions = self.main_gui.config.get('skipped_versions', [])
        
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Glossarion-Updater'
        }
        
        # Try with reasonable timeout and minimal retries to prevent hanging
        max_retries = 0  # No retries to prevent hanging
        timeout = 30  # Reasonable timeout
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(self.GITHUB_LATEST_URL, headers=headers, timeout=timeout)
                response.raise_for_status()
                break  # Success, exit retry loop
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt == max_retries:
                    # Last attempt failed, save check time and re-raise
                    self._save_last_check_time()
                    raise
                print(f"[DEBUG] Network error on attempt {attempt + 1}: {e}")
                time.sleep(1)  # Short delay before retry
        
        release_data = response.json()
        latest_version = release_data['tag_name'].lstrip('v')
        
        # Save successful check time
        self._save_last_check_time()
        
        # Fetch all releases for history regardless (with timeout protection)
        try:
            self.all_releases = self.fetch_multiple_releases(count=10)
        except Exception as e:
            print(f"[DEBUG] Could not fetch release history: {e}")
            self.all_releases = [release_data]  # Use just the latest release
        self.latest_release = release_data
        
        # Check if this version was skipped by user
        if release_data['tag_name'] in skipped_versions and not force_show:
            return False, None
        
        # Compare versions
        if version.parse(latest_version) > version.parse(self.CURRENT_VERSION):
            self.update_available = True
            
            # Update available - will be handled by signal
            print(f"[DEBUG] Update available for version {latest_version}")
                
            return True, release_data
        else:
            # We're up to date
            self.update_available = False
            
            # Dialog will be shown via signal if force_show is True
            return False, None

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
