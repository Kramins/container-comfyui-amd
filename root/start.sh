#!/bin/bash
# =============================================================================
# Configuration
# =============================================================================

ROCM_VERSION="7.1"
VENV_PATH="/data/venv"
PIP_BIN="$VENV_PATH/bin/pip"

# MIOpen kernel database — pre-compiled kernels to avoid SearchImpl at runtime.
# Downloaded once to /data/miopen-kdb and reused across container restarts.
KDB_ROCM_VER="7.1.1"
KDB_PKG_VER="3.5.1.70101-38~24.04"
KDB_DIR="/data/miopen-kdb"

# Custom nodes: map of repo URL → directory name under /data/custom_nodes
declare -A CUSTOM_NODES=(
    ["https://github.com/ltdrdata/ComfyUI-Manager"]="comfyui-manager"
)

# =============================================================================
# MIOpen / ROCm Environment
# =============================================================================

export MIOPEN_BACKEND=MLIR
export MIOPEN_FIND_MODE=2        # check DB + search if missing
export MIOPEN_USER_DB_PATH=/tmp/miopen
export MIOPEN_ENABLE_CACHE=1
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

# =============================================================================
# ComfyUI Source
# =============================================================================

echo "[INFO] Starting ComfyUI setup..."

if [ "${COMFYUI_VERSION}" = "git" ]; then
    if [ -d /app/.git ]; then
        echo "[INFO] Updating ComfyUI repository..."
        cd /app || exit 1
        git pull origin main
    else
        echo "[INFO] Cloning ComfyUI repository..."
        git clone --depth 1 https://github.com/Comfy-Org/ComfyUI.git /app/
    fi
else
    echo "[INFO] Using ComfyUI release version: ${COMFYUI_VERSION}"
    if [ ! -f /app/main.py ]; then
        echo "[ERROR] ComfyUI release files not found in /app. Build may have failed."
        exit 1
    fi
fi

# =============================================================================
# Virtual Environment
# =============================================================================

if [ ! -f "$VENV_PATH/bin/python" ]; then
    echo "[INFO] Creating virtual environment..."
    mkdir -p "$VENV_PATH"
    python3 -m venv "$VENV_PATH"
fi

# =============================================================================
# Data Directories
# =============================================================================

for folder in models user output input custom_nodes temp; do
    if [ ! -d "/data/$folder" ]; then
        echo "[INFO] Creating /data/$folder..."
        mkdir -p "/data/$folder"
    fi
done

# =============================================================================
# GPU Architecture Detection
# =============================================================================

GFX_ARCH=""
if command -v rocminfo &>/dev/null; then
    GFX_ARCH=$(rocminfo 2>/dev/null | grep "Name:" | grep -o "gfx[0-9]*" | head -1)
fi
echo "[INFO] Detected GPU arch: ${GFX_ARCH:-unknown}"

# Per-arch tuning: HSA override version, hipBLASLt, and extra ComfyUI launch args.
# HSA_OVERRIDE_GFX_VERSION respects any value already present in the environment
# (e.g. from .env / docker run -e) and only falls back to the arch default.
COMFYUI_EXTRA_ARGS=""

case "${GFX_ARCH}" in
    gfx1010|gfx1011|gfx1012)
        echo "[INFO] GPU family: RDNA1"
        export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.1.0}"
        ;;
    gfx1030|gfx1031|gfx1032|gfx1034|gfx1035|gfx1036)
        echo "[INFO] GPU family: RDNA2"
        export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
        COMFYUI_EXTRA_ARGS="--use-pytorch-cross-attention"
        ;;
    gfx1100|gfx1101|gfx1102)
        echo "[INFO] GPU family: RDNA3 — enabling hipBLASLt"
        export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
        export TORCH_BLAS_PREFER_HIPBLASLT=1
        COMFYUI_EXTRA_ARGS="--use-pytorch-cross-attention"
        ;;
    gfx1150|gfx1151)
        echo "[INFO] GPU family: RDNA3.5 — enabling hipBLASLt"
        export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.0}"
        export TORCH_BLAS_PREFER_HIPBLASLT=1
        COMFYUI_EXTRA_ARGS="--use-pytorch-cross-attention"
        ;;
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

# =============================================================================
# Package Installation
# =============================================================================

echo "[INFO] Installing ROCm PyTorch..."
"$PIP_BIN" install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}"

echo "[INFO] Installing ComfyUI requirements..."
if [ ! -f /app/requirements.txt ]; then
    echo "[ERROR] /app/requirements.txt not found."
    exit 1
fi
"$PIP_BIN" install -r /app/requirements.txt

echo "[INFO] Installing AMD performance packages..."
# xformers pre-built wheels target gfx908/gfx90a/gfx1100+.
# RDNA1/RDNA2 (gfx1010–gfx1036) crash with the pre-built wheel — skip them.
case "${GFX_ARCH}" in
    gfx101*|gfx103*)
        echo "[INFO] Skipping xformers on ${GFX_ARCH} (RDNA1/RDNA2): pre-built wheel unsupported."
        ;;
    *)
        "$PIP_BIN" install xformers \
            --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}" \
            || echo "[WARN] xformers install failed, continuing without it"
        ;;
esac
"$PIP_BIN" install onnxruntime-rocm bitsandbytes torchao

# =============================================================================
# MIOpen Kernel Database (kdb)
# =============================================================================

if [ -n "${GFX_ARCH}" ]; then
    KDB_FILE="${KDB_DIR}/${GFX_ARCH}.kdb"
    if [ ! -f "${KDB_FILE}" ]; then
        echo "[INFO] Downloading MIOpen kdb for ${GFX_ARCH} (~745 MB, once only)..."
        mkdir -p "${KDB_DIR}"
        KDB_DEB="miopen-hip-${GFX_ARCH}kdb_${KDB_PKG_VER}_amd64.deb"
        KDB_URL="https://repo.radeon.com/rocm/apt/${KDB_ROCM_VER}/pool/main/m/miopen-hip-${GFX_ARCH}kdb/${KDB_DEB}"
        if curl -fL "${KDB_URL}" -o "/tmp/${KDB_DEB}" 2>&1; then
            dpkg-deb -x "/tmp/${KDB_DEB}" /tmp/kdb-extract
            find /tmp/kdb-extract -name "*.kdb" -exec cp {} "${KDB_FILE}" \;
            rm -rf "/tmp/${KDB_DEB}" /tmp/kdb-extract
            echo "[INFO] MIOpen kdb saved to ${KDB_FILE}."
        else
            echo "[WARN] MIOpen kdb not available for ${GFX_ARCH} — SearchImpl fallback active."
        fi
    else
        echo "[INFO] MIOpen kdb already present: ${KDB_FILE}"
    fi
    export MIOPEN_SYSTEM_DB_PATH="${KDB_DIR}"
fi

# =============================================================================
# Custom Nodes
# =============================================================================

for repo in "${!CUSTOM_NODES[@]}"; do
    dir_name="${CUSTOM_NODES[$repo]}"
    dir="/data/custom_nodes/${dir_name}"
    echo "[INFO] Installing custom node: ${dir_name}"
    git clone --depth 1 "${repo}" "${dir}"
    if [ -f "${dir}/requirements.txt" ]; then
        echo "[INFO] Installing requirements for: ${dir_name}"
        "$PIP_BIN" install -r "${dir}/requirements.txt"
    fi
done

# =============================================================================
# Launch ComfyUI
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              ComfyUI AMD — Launch Settings               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  GPU arch            : ${GFX_ARCH:-unknown}"
echo "  ComfyUI version     : ${COMFYUI_VERSION:-unknown}"
echo "  ROCm torch index    : rocm${ROCM_VERSION}"
echo "  HSA_OVERRIDE_GFX    : ${HSA_OVERRIDE_GFX_VERSION:-not set}"
echo "  HIPBLASLT           : ${TORCH_BLAS_PREFER_HIPBLASLT:-not set}"
echo "  HIP_ALLOC_CONF      : ${PYTORCH_HIP_ALLOC_CONF:-not set}"
echo "  MIOPEN_FIND_MODE    : ${MIOPEN_FIND_MODE:-not set}"
echo "  MIOPEN_SYSTEM_DB    : ${MIOPEN_SYSTEM_DB_PATH:-not set}"
echo "  Extra launch args   : ${COMFYUI_EXTRA_ARGS:-(none)}"
echo "  Venv                : ${VENV_PATH}"
echo "  Data dir            : /data"
echo ""
echo "  Launching in 5 seconds... (Ctrl-C to abort)"
echo ""
sleep 5

# shellcheck disable=SC2086
"$VENV_PATH/bin/python" /app/main.py \
    --listen 0.0.0.0 \
    --base-directory /data \
    ${COMFYUI_EXTRA_ARGS}
