#!/usr/bin/env bash
###############################################################################
# build_vitalium.sh — reproducibly build + install Vitalium.vst3 (arm64, no sudo)
#
# WHAT: Vitalium is the GPL-3 fork of the Vital wavetable synth, built by DISTRHO
#   from mtytel/vital with -DNO_AUTH=1. It ships CONTENT-FREE (no proprietary
#   factory presets/wavetables, no account/login) — we drive only its init state
#   with our own parameters, so it is fully ship-clean for the ES2 stem renderer.
#
# WHY BUILD FROM SOURCE: there is no live prebuilt macOS arm64 *VST3* of Vitalium.
#   DISTRHO/PawPaw's macOS pkg ships LV2 only (DawDreamer can't host LV2), and
#   DISTRHO-Ports CI does emit a macOS universal VST3 but its GH-Actions artifacts
#   expire after 90 days and were never attached to a Release. So we compile.
#
# LICENSE: GPL-3.0.
#   - upstream synth:  https://github.com/mtytel/vital        (GPL-3.0)
#   - port we build:   https://github.com/DISTRHO/DISTRHO-Ports   port "vitalium"
#     ports-juce6.0/vitalium/, meson license 'GPLv3', built with NO_AUTH=1.
#
# HOST PROVEN ON: Apple M2 Max, macOS 25.x (arm64), Xcode 26.6, Homebrew (user-owned).
# TOOLCHAIN: full Xcode CLT + `brew install meson` (pulls ninja). No sudo anywhere.
#
# PINNED (from the build that produced the shipped bundle):
#   DISTRHO-Ports  d3b62da2e83c69b0866af5bb2e29ac78dc8014cf
#   libs/juce6.0   24b7e3b2b2c4713d53163b58aed8f79a605218e9  (DISTRHO/JUCE)
# Omit the pin (set PIN=0) to build tip of master.
###############################################################################
set -euo pipefail

PIN="${PIN:-1}"
DISTRHO_PORTS_SHA="d3b62da2e83c69b0866af5bb2e29ac78dc8014cf"

# Work dir: keep it under es2/assets so the recipe only touches allowed paths.
BUILD_ROOT="${BUILD_ROOT:-/Users/jhurliman/.claude/jobs/7eb03476/tmp/es2/assets/vitalium-build}"
VST3_DIR="${VST3_DIR:-$HOME/Library/Audio/Plug-Ins/VST3}"
DEST="$VST3_DIR/Vitalium.vst3"

export PATH="/opt/homebrew/bin:$PATH"
command -v meson >/dev/null || { echo "need meson: brew install meson"; exit 1; }
command -v ninja >/dev/null || { echo "need ninja: brew install ninja"; exit 1; }

echo ">> Vitalium build starting into $BUILD_ROOT"
mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"

# 1. Source. Vitalium's own C++ is vendored directly in the repo (not a submodule);
#    only the JUCE library for the juce6.0 line is a submodule, so init just that.
if [ ! -d DISTRHO-Ports/.git ]; then
  git clone --depth 1 https://github.com/DISTRHO/DISTRHO-Ports.git
fi
cd DISTRHO-Ports
if [ "$PIN" = "1" ]; then
  git fetch --depth 1 origin "$DISTRHO_PORTS_SHA"
  git checkout -q "$DISTRHO_PORTS_SHA"
fi
git submodule update --init --depth 1 libs/juce6.0/source

# 2. Configure: native arm64 (NOT universal — we only need arm64, halves time/disk),
#    juce6.0 only, ONLY the vitalium plugin, ONLY the VST3 format.
rm -rf build
meson setup build \
  --buildtype=release \
  --prefix=/usr \
  -Dbuild-juce60-only=true \
  -Dplugins=vitalium \
  -Dbuild-lv2=false \
  -Dbuild-vst2=false \
  -Dbuild-vst3=true

# 3. Build just the VST3 bundle target (unity build; ~20s warm, a few min cold).
ninja -C build ports-juce6.0/vitalium.vst3

SRC_DYLIB="$BUILD_ROOT/DISTRHO-Ports/build/ports-juce6.0/vitalium.vst3/Contents/MacOS/vitalium.dylib"
file "$SRC_DYLIB" | grep -q arm64 || { echo "!! built binary is not arm64"; exit 1; }
nm -gU "$SRC_DYLIB" | grep -q _GetPluginFactory || { echo "!! missing VST3 factory export"; exit 1; }

# 4. Repackage into a spec-compliant macOS VST3 bundle.
#    The meson target emits Contents/MacOS/vitalium.dylib with NO Info.plist and a
#    .dylib-suffixed binary; macOS CFBundle-based VST3 hosting (JUCE/DawDreamer)
#    needs Info.plist + a CFBundleExecutable-matching binary name. Fix both here.
STAGE="$BUILD_ROOT/Vitalium.vst3"
rm -rf "$STAGE"
mkdir -p "$STAGE/Contents/MacOS"
cp "$SRC_DYLIB" "$STAGE/Contents/MacOS/Vitalium"
chmod +x "$STAGE/Contents/MacOS/Vitalium"
printf 'BNDL????' > "$STAGE/Contents/PkgInfo"
cat > "$STAGE/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>English</string>
	<key>CFBundleExecutable</key>
	<string>Vitalium</string>
	<key>CFBundleGetInfoString</key>
	<string>Vitalium 1.0.6, GPL-3.0, DISTRHO/DISTRHO-Ports port of mtytel/vital</string>
	<key>CFBundleIdentifier</key>
	<string>studio.kx.distrho.vitalium</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleName</key>
	<string>Vitalium</string>
	<key>CFBundlePackageType</key>
	<string>BNDL</string>
	<key>CFBundleShortVersionString</key>
	<string>1.0.6</string>
	<key>CFBundleSignature</key>
	<string>Vita</string>
	<key>CFBundleVersion</key>
	<string>1.0.6</string>
	<key>NSHumanReadableCopyright</key>
	<string>GNU General Public License v3.0</string>
</dict>
</plist>
PLIST
plutil -lint "$STAGE/Contents/Info.plist"
# Ad-hoc sign so the arm64 dylib loads cleanly (no Gatekeeper/quarantine friction).
xattr -cr "$STAGE" 2>/dev/null || true
codesign --force --deep --sign - "$STAGE"

# 5. Install with ditto (no sudo).
mkdir -p "$VST3_DIR"
rm -rf "$DEST"
ditto "$STAGE" "$DEST"

echo ">> installed: $DEST"
lipo -archs "$DEST/Contents/MacOS/Vitalium"
shasum -a 256 "$DEST/Contents/MacOS/Vitalium"

# 6. Verify it hosts + makes sound in DawDreamer (init state, one MIDI note).
VENV_PY="/Users/jhurliman/.claude/jobs/7eb03476/tmp/es2/.venv/bin/python"
"$VENV_PY" - "$DEST" <<'PY'
import sys, dawdreamer as daw, numpy as np
e = daw.RenderEngine(44100, 512)
s = e.make_plugin_processor('v', sys.argv[1])          # benign "invalid URI" on stderr is expected
print('params', len(s.get_parameters_description()))
s.add_midi_note(60, 110, 0.0, 1.0)
e.load_graph([(s, [])]); e.render(2.0)
a = np.asarray(e.get_audio())
print('peak', float(np.abs(a).max()), 'rms', float(np.sqrt((a**2).mean())))
PY

echo ">> done. To reclaim disk: rm -rf $BUILD_ROOT/DISTRHO-Ports"
