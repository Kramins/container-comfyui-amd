#!/bin/bash
# ZLUDA requires pre-CUDA-12 PyTorch. CUDA 12+ Thrust uses cudaStreamWaitValue32
# which is not implemented in ZLUDA/ROCm on Linux → cudaErrorNotSupported at model load.
# cu118 = CUDA 11.8 is the last pre-CUDA-12 release with good ZLUDA compatibility.
CUDA_INDEX="cu118"
VENV_PATH="/data/venv"
PIP_BIN="$VENV_PATH/bin/pip"

echo "[INFO] Starting ComfyUI setup (ZLUDA backend)..."

# Configure ZLUDA library interception
# Prepending ZLUDA's path ensures its libcuda.so, libcublas.so etc. are found
# before any system CUDA libraries, routing all CUDA calls through ZLUDA -> HIP -> ROCm
export LD_LIBRARY_PATH="${ZLUDA_PATH}:${LD_LIBRARY_PATH}"

echo "[INFO] ZLUDA_PATH: ${ZLUDA_PATH}"
echo "[INFO] LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

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

# Inject ZLUDA PyTorch compatibility shim via sitecustomize.py.
# This runs automatically before any Python code starts, disabling CUDA 12+ code paths
# (flash attention, cuBLAS-LT, cuDNN SDP) that ZLUDA's PTX translator hasn't fully
# implemented yet. DISABLE_ADDMM_CUDA_LT=1 (set via ENV in Dockerfile) covers addmm.
SITE_PACKAGES=$(find "$VENV_PATH/lib" -maxdepth 2 -name "site-packages" -type d | head -1)
if [ -n "$SITE_PACKAGES" ]; then
    cat > "$SITE_PACKAGES/sitecustomize.py" << 'PYEOF'
# ZLUDA compatibility shim — injected by start-zluda.sh at container startup.
# Disables backends not implemented in ZLUDA's PTX translator and patches ops
# known to fail (torch.topk triggers cudaErrorNotSupported on ZLUDA).
import os
if os.environ.get("BACKEND") == "zluda":
    try:
        import torch
        # Disable attention backends that rely on cuDNN / Flash Attention CUDA 12+ code
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)

        # torch.topk triggers cudaErrorNotSupported on ZLUDA.
        # Workaround: execute on CPU and move result back to original device.
        _topk = torch.topk
        def _zluda_topk(tensor, *args, **kwargs):
            device = tensor.device
            values, indices = _topk(tensor.cpu(), *args, **kwargs)
            return torch.return_types.topk((values.to(device), indices.to(device)))
        torch.topk = _zluda_topk
    except ImportError:
        pass
PYEOF
    echo "[INFO] Injected ZLUDA PyTorch compat shim -> $SITE_PACKAGES/sitecustomize.py"
fi

for folder in models user output input custom_nodes temp; do
    if [ ! -d "/data/$folder" ]; then
        echo "[INFO] Creating /data/$folder..."
        mkdir -p "/data/$folder"
    fi
done

# Install CUDA-flavored PyTorch
# ZLUDA intercepts the CUDA calls at runtime via LD_LIBRARY_PATH, so we install
# the stock CUDA build rather than the ROCm build
echo "[INFO] Installing CUDA PyTorch (CUDA calls will be intercepted by ZLUDA at runtime)..."
"$PIP_BIN" install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_INDEX}"

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
echo "[INFO] Starting ComfyUI with ZLUDA backend..."
# CUDA_LAUNCH_BLOCKING=1: makes CUDA errors synchronous so the traceback points to the
# actual failing op rather than a later unrelated call. Remove once stable.
export CUDA_LAUNCH_BLOCKING=1

# --force-fp32: run models in float32 to avoid fp16/bf16 ops that ZLUDA may not translate
# --disable-dynamic-vram: prevents comfy-aimdo from activating (enables_dynamic_vram() returns
#   False), which avoids "VBAR allocation failed". comfy-aimdo uses CUDA VMM APIs
#   (cuMemAddressReserve, cuMemCreate, cuMemMap) that ZLUDA does not implement.
#   comfy-aimdo must remain *installed* because main.py has a hard import of it.
# --disable-async-offload: async weight offloading is enabled by default for Nvidia GPUs.
#   ZLUDA appears as Nvidia, so it would be auto-enabled. It uses multi-stream CUDA APIs
#   (cudaStreamCreate, cudaEventRecord, pinned memory transfers) that ZLUDA may not fully
#   implement, causing "operation not supported" errors during sampling.
"$VENV_PATH/bin/python" /app/main.py \
    --listen 0.0.0.0 \
    --base-directory /data \
    --force-fp32 \
    --disable-dynamic-vram \
    --disable-async-offload
