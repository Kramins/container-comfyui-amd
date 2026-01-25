#!/bin/bash

# ComfyUI Container Build Script
# Usage: 
#   ./build-container.sh [VERSION] [--push]
#   VERSION: ComfyUI version to build (e.g., "git", "0.8.2") - defaults to "git"
#   --push: Push to GHCR after building

set -e

VERSION=${1:-git}
PUSH_FLAG=""
DOCKER_REPO=${DOCKER_REPO:-"ghcr.io/kramins"}

# Check if --push flag is provided
if [[ "$*" == *"--push"* ]]; then
    PUSH_FLAG="--push"
    echo "🚀 Will push to GHCR after building"
fi

echo "🔨 Building ComfyUI AMD container"
echo "   Version: $VERSION"
echo "   Registry: $DOCKER_REPO/comfyui-amd"
echo ""

# Export environment variables for docker bake
export COMFYUI_VERSIONS="$VERSION"
export DOCKER_REPO
export ADD_LATEST_TAG="false"

# Build (and optionally push) using docker bake
docker buildx bake \
    --file docker-bake.hcl \
    $PUSH_FLAG \
    comfyui-amd

echo ""
echo "✅ Build completed successfully!"
echo "   Image: $DOCKER_REPO/comfyui-amd:$VERSION"

if [ -z "$PUSH_FLAG" ]; then
    echo ""
    echo "💡 To push to GHCR, run: ./build-container.sh $VERSION --push"
fi