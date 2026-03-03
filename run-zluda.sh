#!/bin/bash

# ComfyUI ZLUDA Container Run Script
# Usage: ./run-zluda.sh [VERSION]
# VERSION: ComfyUI version tag (e.g., "git", "latest") - defaults to "git"

VERSION="${1:-git}"
CONTAINER_NAME="comfyui-zluda"
IMAGE="ghcr.io/kramins/comfyui-amd-zluda:$VERSION"
COMFYUI_DATA="$HOME/comfyui-zluda"

echo "🚀 Starting ComfyUI container (ZLUDA backend)"
echo "   Version: $VERSION"
echo "   Image: $IMAGE"
echo "   Data directory: $COMFYUI_DATA"
echo ""

# Create data directories if they don't exist
mkdir -p "$COMFYUI_DATA"/{models,custom_nodes,output,user,input,temp}

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
    -p 8190:8188 \
    -v "$COMFYUI_DATA":/data/ \
    --name $CONTAINER_NAME \
    $IMAGE
