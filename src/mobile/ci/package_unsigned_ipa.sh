#!/usr/bin/env bash
# Package the .app from an unsigned `flet build ipa` (`flutter build ipa --no-codesign`) into a
# Payload/ IPA that AltStore / SideStore re-sign on install.
#
# Usage: package_unsigned_ipa.sh <flet-ipa-output-dir | *.xcarchive | *.app> <dest.ipa>
#
# Flet copies build/ios/archive/* into its output dir, so the app is at
# <output>/<Name>.xcarchive/Products/Applications/<Name>.app.
set -euo pipefail

SRC="${1:?usage: package_unsigned_ipa.sh <flet-ipa-output-dir|.xcarchive|.app> <dest.ipa>}"
DEST="${2:?usage: package_unsigned_ipa.sh <flet-ipa-output-dir|.xcarchive|.app> <dest.ipa>}"

err() {
  echo "::error title=Unsigned IPA::$*" >&2
  exit 1
}

[ -e "$SRC" ] || err "source not found: $SRC"

app=""
case "$SRC" in
  *.app|*.app/)
    app="${SRC%/}"
    ;;
  *.xcarchive|*.xcarchive/)
    app="$(find "${SRC%/}/Products/Applications" -mindepth 1 -maxdepth 1 -type d -name '*.app' | sort | head -n 1)"
    ;;
  *)
    # Prefer an archived app; fall back to a plain .app anywhere below the output dir.
    app="$(find "$SRC" -type d -path '*.xcarchive/Products/Applications/*.app' -prune | sort | head -n 1)"
    if [ -z "$app" ]; then
      app="$(find "$SRC" -type d -name '*.app' -prune | sort | head -n 1)"
    fi
    ;;
esac
[ -n "$app" ] && [ -d "$app" ] || err "no .app bundle found under $SRC"
[ -f "$app/Info.plist" ] || err "$app has no Info.plist"
echo "App bundle: $app"

count="$(find "$SRC" -type d -name '*.app' -prune | wc -l | tr -d ' ')"
if [ "${count:-0}" -gt 1 ]; then
  echo "::warning title=Unsigned IPA::$count .app bundles under $SRC; packaging $app"
fi

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/Payload"
if command -v ditto > /dev/null 2>&1; then
  ditto "$app" "$stage/Payload/$(basename "$app")"
else
  cp -a "$app" "$stage/Payload/"
fi

mkdir -p "$(dirname "$DEST")"
dest_abs="$(cd "$(dirname "$DEST")" && pwd)/$(basename "$DEST")"
rm -f "$dest_abs"
# -y keeps symlinks as links (framework bundles), -X drops extra file attributes.
(cd "$stage" && zip -q -r -y -X "$dest_abs" Payload)

app_name="$(basename "$app")"
unzip -l "$dest_abs" "Payload/$app_name/Info.plist" > /dev/null 2>&1 \
  || err "Payload/$app_name/Info.plist is missing from $dest_abs"

if command -v shasum > /dev/null 2>&1; then
  sha="$(shasum -a 256 "$dest_abs" | cut -d' ' -f1)"
else
  sha="$(sha256sum "$dest_abs" | cut -d' ' -f1)"
fi
size="$(du -h "$dest_abs" | cut -f1)"
echo "Wrote $dest_abs ($size, sha256 $sha)"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "## Unsigned IPA"
    echo
    echo "- File: \`$(basename "$dest_abs")\` ($size)"
    echo "- App: \`$app_name\`"
    echo "- SHA-256: \`$sha\`"
  } >> "$GITHUB_STEP_SUMMARY"
fi
