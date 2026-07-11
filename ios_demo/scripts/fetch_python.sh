#!/usr/bin/env bash
#
# fetch_python.sh — download a prebuilt CPython-for-iOS (Python-Apple-support)
# and place Python.xcframework where the Xcode project expects it.
#
# This is the ONE binary prerequisite the repo does not vendor (~80 MB, both
# slices). Requires a network connection on your Mac (the *app* stays fully
# offline; this is a build-time download only).
#
# Result:
#   MicroWorldDemo/Frameworks/Python.xcframework
#
# The stdlib itself is NOT staged here. It is embedded at BUILD time by the
# "Embed Python stdlib" script phase (scripts/embed_stdlib.sh), which reads
# straight from Python.xcframework/<slice>/lib/python3.x and picks the slice
# that matches what Xcode is actually building for (device vs simulator).
# Staging it once, ahead of time, into a static resource is what caused a
# device/simulator ABI mismatch bug ("No module named '_struct'") — the two
# slices' compiled extension modules are not interchangeable.
#
# Version is pinned; bump PY_VER / SUPPORT_TAG deliberately.
set -euo pipefail

PY_VER="${PY_VER:-3.13}"
# See https://github.com/beeware/Python-Apple-support/releases for current tags.
SUPPORT_TAG="${SUPPORT_TAG:-3.13-b6}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IOS_DEMO="$(cd "$HERE/.." && pwd)"
PROJ="$IOS_DEMO/MicroWorldDemo"
FRAMEWORKS="$PROJ/Frameworks"

URL="https://github.com/beeware/Python-Apple-support/releases/download/${SUPPORT_TAG}/Python-${PY_VER}-iOS-support.${SUPPORT_TAG#${PY_VER}-}.tar.gz"

echo "Python-Apple-support ${SUPPORT_TAG} (CPython ${PY_VER})"
echo "URL: $URL"
mkdir -p "$FRAMEWORKS"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading…"
curl -fL "$URL" -o "$TMP/support.tar.gz"
echo "Extracting…"
tar -xzf "$TMP/support.tar.gz" -C "$TMP"

if [ -d "$TMP/Python.xcframework" ]; then
  rm -rf "$FRAMEWORKS/Python.xcframework"
  cp -R "$TMP/Python.xcframework" "$FRAMEWORKS/"
else
  echo "!! Python.xcframework not found in archive; inspect $TMP and adjust paths." >&2
  ls -la "$TMP" >&2
  exit 1
fi

# Disable the framework's Clang modulemap. We include <Python.h> TEXTUALLY
# (quoted) with the Headers dir on HEADER_SEARCH_PATHS; the modulemap `exclude`s
# the cpython/* headers, so any attempt to build "module Python" fails with
# "'cpython/pyatomic_gcc.h' file not found". Renaming it guarantees Clang never
# tries. Linking is explicit (the framework is in Link Binary With Libraries),
# so we don't need the modulemap's `link "Python"`.
find "$FRAMEWORKS/Python.xcframework" -name "module.modulemap" -path "*Python.framework/Headers*" \
  -exec sh -c 'mv "$1" "$1.disabled"' _ {} \; 2>/dev/null || true
echo "Disabled Python.framework modulemaps (textual include is used instead)."

for slice in "ios-arm64" "ios-arm64_x86_64-simulator"; do
  if [ ! -d "$FRAMEWORKS/Python.xcframework/$slice/lib/python${PY_VER}/encodings" ]; then
    echo "warning: expected stdlib not found at $slice/lib/python${PY_VER}/encodings — the embed_stdlib.sh build phase will fail for that platform." >&2
  fi
done

echo "Done. Now run stage_bundle.sh (if you haven't) and 'xcodegen generate'."
