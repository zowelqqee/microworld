#!/usr/bin/env bash
#
# embed_stdlib.sh — Xcode build-phase script. Copies the CORRECT-platform
# slice of the CPython stdlib (device vs simulator) into the built product.
#
# Why this has to run at build time, not once via a static folder reference:
# Python.xcframework ships two ABI-incompatible slices — device (`ios-arm64`,
# Mach-O platform IOS) and simulator (`ios-arm64_x86_64-simulator`, Mach-O
# platform IOSSIMULATOR). Xcode's xcframework embedding picks the right slice
# for Python.framework automatically, but the stdlib's compiled extension
# modules (`_struct.so`, `_socket.so`, etc.) are plain bundled resources, not
# part of the xcframework mechanism — so they need the same platform-aware
# selection, done here via $PLATFORM_NAME. Getting this wrong doesn't fail the
# build; it fails at runtime with a confusing "No module named '_struct'"
# (dyld silently refuses to dlopen a device-tagged .so in a simulator process,
# and CPython's import system reports that as "not found").
#
# On a PHYSICAL DEVICE there is a second, separate requirement: every Mach-O
# binary dlopen'd at runtime needs its own code signature (AMFI enforces this;
# the Simulator does not, which is why this only shows up on-device). Xcode's
# automatic signing recursively signs recognized bundle structures (.framework,
# .appex) but does NOT sign arbitrary loose files copied in by a script phase —
# so each lib-dynload/*.so must be explicitly signed with the app's identity here,
# after copying, before Xcode's final whole-app codesign pass runs (build
# script phases run before that final pass).
set -euo pipefail

case "${PLATFORM_NAME:-}" in
  iphoneos) SLICE="ios-arm64" ;;
  iphonesimulator) SLICE="ios-arm64_x86_64-simulator" ;;
  *)
    echo "warning: unrecognised PLATFORM_NAME='${PLATFORM_NAME:-}', defaulting to device slice" >&2
    SLICE="ios-arm64"
    ;;
esac

SRC="${PROJECT_DIR}/Frameworks/Python.xcframework/${SLICE}/lib/python3.13"
# On iOS, resources land at the product/bundle root (no Contents/Resources).
DEST="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/python-stdlib"

if [ ! -d "$SRC" ]; then
  echo "error: expected stdlib at $SRC" >&2
  echo "       Run ios_demo/scripts/fetch_python.sh to download Python.xcframework." >&2
  exit 1
fi

rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
cp -R "$SRC" "$DEST"
find "$DEST" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$DEST/test" "$DEST/tests" 2>/dev/null || true

# Code-sign every compiled extension module we just copied in. On a physical
# device, AMFI requires code loaded with dlopen to carry the app's development
# signing identity. An ad-hoc signature can pass `codesign --verify` on the Mac
# but is rejected by the device with `code signature invalid` when CPython
# imports (for example) `_struct`. The simulator has no provisioning identity,
# so ad-hoc signing remains appropriate there.
#
# Xcode always runs script phases via `/bin/sh -c <path>`; the file's actual
# first line is Xcode's own `#!/bin/sh` (our `#!/usr/bin/env bash` on line 2 is
# inert — shebangs only count as the file's first line). When bash runs AS
# `sh` it disables process substitution (`<(...)`), so this must stay plain
# POSIX — `find -exec`, not a `while read < <(...)` loop.
if [ "${PLATFORM_NAME:-}" = "iphoneos" ]; then
  SIGNING_IDENTITY="${EXPANDED_CODE_SIGN_IDENTITY:-}"
  if [ -z "$SIGNING_IDENTITY" ] || [ "$SIGNING_IDENTITY" = "-" ]; then
    echo "error: no development signing identity was provided for the iPhone build" >&2
    echo "       Select a Team for the MicroWorldDemo target, then clean and rebuild." >&2
    exit 1
  fi
else
  SIGNING_IDENTITY="-"
fi

find "$DEST" -name "*.so" -exec codesign --force --sign "$SIGNING_IDENTITY" --timestamp=none {} \;
find "$DEST" -name "*.so" -exec codesign --verify --strict {} \;
SO_COUNT=$(find "$DEST" -name "*.so" | wc -l | tr -d ' ')

echo "Embedded python-stdlib for PLATFORM_NAME=${PLATFORM_NAME:-?} (slice=$SLICE) -> $DEST"
echo "Code-signed $SO_COUNT extension module(s)"
