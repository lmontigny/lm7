#!/usr/bin/env bash
# Build the LM7 iOS validator.
#
#   ./build.sh project     generate LM7Validator.xcodeproj and stage fixtures
#   ./build.sh ipa         additionally build an unsigned .ipa for AWS Device Farm
#
# Run tools/export_ios_fixture.py first; it writes the artifact and golden
# fixture this app bundles. See docs/ios-device-testing.md.
set -euo pipefail

cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
ARTIFACTS="$ROOT/artifacts/ios"

if [ ! -f "$ARTIFACTS/coreml-mlp.lm7/compiled_model.pte" ]; then
    echo "error: missing $ARTIFACTS/coreml-mlp.lm7/compiled_model.pte" >&2
    echo "run:   .venv-et/bin/python tools/export_ios_fixture.py" >&2
    exit 1
fi

mkdir -p Resources
cp "$ARTIFACTS/coreml-mlp.lm7/compiled_model.pte" Resources/
cp "$ARTIFACTS/golden.json" Resources/

xcodegen generate

if [ "${1:-project}" = "project" ]; then
    echo
    echo "open LM7Validator.xcodeproj  # then pick a simulator or your iPhone"
    exit 0
fi

# Unsigned device build for Device Farm, which re-signs public-fleet apps.
#
# -target rather than -scheme is deliberate: a machine that has Xcode but has
# not downloaded the iOS platform has no build destinations, so -scheme fails
# with "iOS <ver> is not installed" even though the SDK is present. -target
# skips destination resolution. It requires SYMROOT instead of
# -derivedDataPath.
xcodebuild \
    -project LM7Validator.xcodeproj \
    -target LM7Validator \
    -sdk iphoneos \
    -configuration Release \
    SYMROOT="$PWD/build" \
    -clonedSourcePackagesDirPath "$PWD/spm" \
    CODE_SIGNING_ALLOWED=NO \
    CODE_SIGNING_REQUIRED=NO \
    CODE_SIGN_IDENTITY="" \
    build

cd build/Release-iphoneos
rm -rf Payload ../../LM7Validator.ipa
mkdir -p Payload
cp -R LM7Validator.app Payload/
zip -qry ../../LM7Validator.ipa Payload
cd ../..

echo
echo "unsigned ipa: $PWD/LM7Validator.ipa"
