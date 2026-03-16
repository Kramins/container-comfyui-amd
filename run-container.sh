#!/bin/bash

# ComfyUI Container Run Script
# Usage: ./run-container.sh [VERSION]
# VERSION: ComfyUI version tag (e.g., "latest", "git", "0.8.2") - defaults to "latest"

VERSION="${1:-latest}"
CONTAINER_NAME="comfyui"

# Load .env file if present (values set there override defaults below)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -o allexport
    source "${SCRIPT_DIR}/.env"
    set +o allexport
fi

IMAGE="ghcr.io/kramins/comfyui-amd:$VERSION"
COMFYUI_DATA="${COMFYUI_DATA:-$HOME/comfyui}"

echo "🚀 Starting ComfyUI container"
echo "   Version: $VERSION"
echo "   Image: $IMAGE"
echo "   Data directory: $COMFYUI_DATA"
echo ""

# Create data directories if they don't exist
mkdir -p "$COMFYUI_DATA"/{models,custom_nodes,output,user,input,temp,cache/huggingface}

# Stop and remove the container if it exists
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "⏹️  Stopping existing $CONTAINER_NAME container..."
    docker stop $CONTAINER_NAME
    sleep 2
fi

if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
    echo "🗑️  Removing existing $CONTAINER_NAME container..."
    docker rm $CONTAINER_NAME
    sleep 1
fi

echo "▶️  Starting container..."
docker run --rm -it --privileged \
    -v /dev:/dev \
    -p 8188:8188 \
    -v "$COMFYUI_DATA":/data/ \
    --name $CONTAINER_NAME \
    $IMAGE
