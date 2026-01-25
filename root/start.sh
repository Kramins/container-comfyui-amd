#!/bin/bash
ROCM_VERSION="6.4"  # Set your ROCm version here
VENV_PATH="/data/venv"
PIP_BIN="$VENV_PATH/bin/pip"


echo "[INFO] Starting ComfyUI setup..."

# Handle ComfyUI source based on version
if [ "${COMFYUI_VERSION}" = "git" ]; then
    # Git version: clone or update at runtime
    if [ -d /app/.git ]; then
        echo "[INFO] ComfyUI repository already exists. Pulling latest changes..."
        cd /app || exit 1
        git pull origin main
    else
        echo "[INFO] Cloning ComfyUI repository..."
        git clone --depth 1 https://github.com/Comfy-Org/ComfyUI.git /app/
    fi
else
    # Release version: should already be in /app from build
    echo "[INFO] Using ComfyUI release version: ${COMFYUI_VERSION}"
    if [ ! -f /app/main.py ]; then
        echo "[ERROR] ComfyUI release files not found in /app. Build may have failed."
        exit 1
    fi
fi


# Create the virtual environment if it doesn't exist or is incomplete
if [ ! -f "$VENV_PATH/bin/python" ]; then
    echo "[INFO] Creating virtual environment..."
    mkdir -p "$VENV_PATH"
    python3 -m venv "$VENV_PATH"
fi

for folder in models user output input custom_nodes temp; do
    if [ ! -d "/data/$folder" ]; then
        echo "[INFO] Creating /data/$folder..."
        mkdir -p "/data/$folder"
    fi
done

# Install requirements

# Install ROCM Torch Requirements
echo "[INFO] Installing ROCm Torch requirements..."
"$PIP_BIN" install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}"



echo "[INFO] Installing ComfyUI requirements..."
requirements_file="/app/requirements.txt"
if [ ! -f "$requirements_file" ]; then
    echo "[ERROR] Requirements file $requirements_file not found."
    exit 1
fi
"$PIP_BIN" install -r "$requirements_file"

# Clone custom nodes and install requirements if present
CUSTOM_NODES_PATH="/data/custom_nodes"
declare -A node_mapping
node_mapping=(
    ["https://github.com/ltdrdata/ComfyUI-Manager"]="comfyui-manager"
)
for repo in "${!node_mapping[@]}"; do
    directory_name="${node_mapping[$repo]}"
    directory="$CUSTOM_NODES_PATH/$directory_name"
    echo "[INFO] Preparing to clone custom node: $repo -> $directory"
    git clone --depth 1 "$repo" "$directory"
    requirements_file="$directory/requirements.txt"
    if [ -f "$requirements_file" ]; then
        echo "[INFO] Installing requirements for custom node: $directory_name"
        "$PIP_BIN" install -r "$requirements_file"
    fi
done

# Start ComfyUI
echo "[INFO] Starting ComfyUI..."
"$VENV_PATH/bin/python" /app/main.py \
    --listen 0.0.0.0 \
    --base-directory /data \