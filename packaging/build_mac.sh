#!/bin/bash
# Build "Damage Annotator.app" and DamageAnnotator-<arch>.dmg (run from the project root).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -f artifacts/onnx/prior.onnx ] || $PY -m dmgseg.onnx.export        # the models, once
$PY packaging/make_icon.py
( cd packaging && ../.venv/bin/pyinstaller --noconfirm --distpath dist --workpath build/pyinstaller damage_annotator.spec )
APP="packaging/dist/Damage Annotator.app"
"$APP/Contents/MacOS/Damage Annotator" --selftest                    # the packaged app must work
STAGE=packaging/build/dmg && rm -rf "$STAGE" && mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/" && ln -s /Applications "$STAGE/Applications"
DMG="packaging/dist/DamageAnnotator-$(uname -m).dmg"
rm -f "$DMG"
hdiutil create -volname "Damage Annotator" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
du -sh "$APP" "$DMG"
