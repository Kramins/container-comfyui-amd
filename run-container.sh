#!/bin/bash

# Check if the container is running and stop it if it is
if [ "$(docker ps -q -f name=comfyui)" ]; then
    echo "Stopping existing comfyui container..."
    docker stop comfyui
    sleep 5
fi

# Remove the container if it exists
if [ "$(docker ps -a -q -f name=comfyui)" ]; then
    echo "Removing existing comfyui container..."
    docker rm comfyui
    sleep 5
fi

docker run --rm --privileged  -v /dev:/dev -p 8188:8188 \
-v ~/comfyui/models:/app/models \
-v ~/comfyui/custom_nodes:/app/custom_nodes \
-v ~/comfyui/output:/app/output \
-v ~/comfyui/user:/app/user \
--name comfyui kramins/comfyui-amd