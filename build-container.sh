#!/bin/bash

# ComfyUI Container Build Script
# Usage: 
#   ./build-container.sh [VERSION] [--push]
#   VERSION: ComfyUI version to build (e.g., "git", "0.8.2") - defaults to "git"
#   --push: Push to GHCR after building

set -e

VERSION=""
PUSH_FLAG=""
ZLUDA_FLAG=""
DOCKER_REPO=${DOCKER_REPO:-"ghcr.io/kramins"}

# Parse arguments — flags can appear in any order
for arg in "$@"; do
    case "$arg" in
        --push)   PUSH_FLAG="--push" ;;
        --zluda)  ZLUDA_FLAG="true" ;;
        --*)      echo "Unknown flag: $arg"; exit 1 ;;
        *)        VERSION="$arg" ;;
    esac
done

VERSION="${VERSION:-git}"

if [ -n "$PUSH_FLAG" ]; then
    echo "🚀 Will push to GHCR after building"
fi

if [ -n "$ZLUDA_FLAG" ]; then
    BUILD_TARGET="zluda"
    IMAGE_NAME="comfyui-amd-zluda"
else
    BUILD_TARGET="comfyui-amd"
    IMAGE_NAME="comfyui-amd"
fi

echo "🔨 Building ComfyUI AMD container"
echo "   Version: $VERSION"
echo "   Backend: ${ZLUDA_FLAG:+ZLUDA}${ZLUDA_FLAG:-ROCm}"
echo "   Registry: $DOCKER_REPO/$IMAGE_NAME"
echo ""

# Export environment variables for docker bake
export COMFYUI_VERSIONS="$VERSION"
export DOCKER_REPO
export ADD_LATEST_TAG="false"

# Build (and optionally push) using docker bake
docker buildx bake \
    --file docker-bake.hcl \
    $PUSH_FLAG \
    $BUILD_TARGET

echo ""
echo "✅ Build completed successfully!"
echo "   Image: $DOCKER_REPO/$IMAGE_NAME:$VERSION"

if [ -z "$PUSH_FLAG" ]; then
    echo ""
    echo "💡 To push to GHCR, run: ./build-container.sh $VERSION${ZLUDA_FLAG:+ --zluda} --push"
fi