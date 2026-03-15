#!/bin/bash
ROCM_VERSION="7.1"  # Set your ROCm version here
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

for folder in models user output input custom_nodes temp cache/huggingface; do
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

# Detect GPU architecture for conditional package installs and tuning
GFX_ARCH=""
if command -v rocminfo &>/dev/null; then
    GFX_ARCH=$(rocminfo 2>/dev/null | grep "Name:" | grep -o "gfx[0-9]*" | head -1)
fi
echo "[INFO] Detected GPU arch: ${GFX_ARCH:-unknown}"

# ── Per-arch environment tuning ───────────────────────────────────────────────
# Only set HSA_OVERRIDE_GFX_VERSION if it wasn't already provided externally.
# This lets the .env / docker run -e value always win.
COMFYUI_EXTRA_ARGS=""

case "${GFX_ARCH}" in
    # ── RDNA1 (RX 5000) ──────────────────────────────────────────────────────
    gfx1010|gfx1011|gfx1012)
        echo "[INFO] GPU family: RDNA1"
        export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.1.0}"
        # No hipBLASLt, no xformers wheel support
        ;;

    # ── RDNA2 (RX 6000) ──────────────────────────────────────────────────────
    gfx1030|gfx1031|gfx1032|gfx1034|gfx1035|gfx1036)
        echo "[INFO] GPU family: RDNA2 — using torch.compile for fast attention"
        export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
        # xformers pre-built wheel crashes on RDNA2; torch.compile via --fast covers it
        COMFYUI_EXTRA_ARGS="--use-pytorch-cross-attention"
        ;;

    # ── RDNA3 (RX 7000) ──────────────────────────────────────────────────────
    gfx1100|gfx1101|gfx1102)
        echo "[INFO] GPU family: RDNA3 — enabling hipBLASLt"
        export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
        export TORCH_BLAS_PREFER_HIPBLASLT=1
        COMFYUI_EXTRA_ARGS="--use-pytorch-cross-attention"
        ;;

    # ── RDNA3.5 (RX 8000) ────────────────────────────────────────────────────
    gfx1150|gfx1151)
        echo "[INFO] GPU family: RDNA3.5 — enabling hipBLASLt"
        export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.0}"
        export TORCH_BLAS_PREFER_HIPBLASLT=1
        COMFYUI_EXTRA_ARGS="--use-pytorch-cross-attention"
        ;;

    # ── CDNA (Instinct MI series) ─────────────────────────────────────────────
    gfx908|gfx90a|gfx940|gfx941|gfx942)
        echo "[INFO] GPU family: CDNA (Instinct) — enabling hipBLASLt"
        export TORCH_BLAS_PREFER_HIPBLASLT=1
        ;;

    *)
        echo "[INFO] GPU arch ${GFX_ARCH:-unknown} — using default settings"
        ;;
esac

echo "[INFO] HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-not set}"
echo "[INFO] TORCH_BLAS_PREFER_HIPBLASLT=${TORCH_BLAS_PREFER_HIPBLASLT:-not set}"

# Install AMD performance optimisation packages
echo "[INFO] Installing AMD performance packages..."
# xformers pre-built wheels target gfx908/gfx90a/gfx1100+.
# RDNA1/RDNA2 (gfx1010–gfx1036) crash with the pre-built wheel — skip them.
case "${GFX_ARCH}" in
    gfx101*|gfx103*)
        echo "[INFO] Skipping xformers on ${GFX_ARCH} (RDNA1/RDNA2): pre-built wheel unsupported."
        echo "[INFO]   To use xformers, build from source: PYTORCH_ROCM_ARCH=${GFX_ARCH} pip install xformers --no-binary xformers"
        ;;
    *)
        "$PIP_BIN" install xformers --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}" \
            || echo "[WARN] xformers install failed, continuing without it"
        ;;
esac
"$PIP_BIN" install onnxruntime-rocm bitsandbytes torchao

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
# shellcheck disable=SC2086
"$VENV_PATH/bin/python" /app/main.py \
    --listen 0.0.0.0 \
    --base-directory /data \
    --fast \
    ${COMFYUI_EXTRA_ARGS}