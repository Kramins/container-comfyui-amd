#!/bin/bash
# ZLUDA Container Debug Script
# Written for iterative AI-assisted debugging — not intended as a user-facing script.
#
# Modes:
#   (no args)         Interactive bash shell in the container
#   --test-structure  Layer 1: verify ZLUDA/ROCm/Python files landed correctly (no GPU needed)
#   --test-gpu        Layer 2: rocminfo — confirm GPU is visible through /dev passthrough
#   --test-zluda      Layer 3: try loading ZLUDA's libcuda.so via Python ctypes
#   --test-torch      Layer 4: install CUDA torch into venv then check torch.cuda.is_available()
#   --test-all        Run layers 1-4 in sequence, stop on first failure
#   --start           Run start.sh (full boot, same as production)
#
# All modes print clearly delimited output so failures are easy to locate.

set -e

VERSION="${VERSION:-git}"
IMAGE="ghcr.io/kramins/comfyui-amd-zluda:${VERSION}"
CONTAINER_NAME="comfyui-zluda-debug"
COMFYUI_DATA="$HOME/comfyui-zluda"
MODE="${1:---shell}"

# Shared docker run flags — mirrors run-container.sh exactly
DOCKER_FLAGS=(
    --rm
    --privileged
    -v /dev:/dev
    -v "$COMFYUI_DATA":/data/
    -e BACKEND=zluda
    -e ZLUDA_PATH=/opt/zluda
    -e LD_LIBRARY_PATH="/opt/zluda:${LD_LIBRARY_PATH}"
    --name "$CONTAINER_NAME"
)

# Stop any existing debug container before starting a new one
if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
    docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1
fi

mkdir -p "$COMFYUI_DATA"/{models,custom_nodes,output,user,input,temp}

separator() { echo; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $1"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

# ──────────────────────────────────────────────
# Layer 1 — structure check (no GPU required)
# ──────────────────────────────────────────────
CMD_STRUCTURE='
separator() { echo; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $1"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

separator "LAYER 1 — Structure"

echo "[CHECK] ZLUDA_PATH=$ZLUDA_PATH"
echo "[CHECK] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo
echo "--- ZLUDA /opt/zluda ---"
ls -lh /opt/zluda/ 2>&1 || echo "ERROR: /opt/zluda missing or empty"

echo
echo "--- Key ZLUDA libs ---"
for lib in libcuda.so libcublas.so libcufft.so libcurand.so libnvrtc.so; do
    if [ -f "/opt/zluda/$lib" ]; then
        echo "  OK  $lib"
    else
        echo "  MISSING  $lib"
    fi
done

echo
echo "--- ROCm HIP runtime ---"
if command -v rocminfo > /dev/null 2>&1; then
    echo "  OK  rocminfo found at $(which rocminfo)"
else
    echo "  MISSING  rocminfo not in PATH"
fi
ls /opt/rocm/lib/libamdhip64.so 2>/dev/null && echo "  OK  libamdhip64.so" || echo "  MISSING  libamdhip64.so"

echo
echo "--- Python ---"
python3 --version 2>&1 || echo "ERROR: python3 not found"

echo
echo "--- /app (ComfyUI) ---"
if [ -f /app/main.py ]; then
    echo "  OK  /app/main.py exists"
else
    echo "  MISSING  /app/main.py (expected for git version — will be cloned at runtime)"
fi

echo
echo "--- /start.sh ---"
[ -f /start.sh ] && echo "  OK  /start.sh exists" || echo "  MISSING  /start.sh"
'

# ──────────────────────────────────────────────
# Layer 2 — GPU visibility
# ──────────────────────────────────────────────
CMD_GPU='
separator() { echo; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $1"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
separator "LAYER 2 — GPU visibility (rocminfo)"
echo
rocminfo 2>&1 | head -80
'

# ──────────────────────────────────────────────
# Layer 3 — ZLUDA library loading via ctypes
# ──────────────────────────────────────────────
CMD_ZLUDA='
separator() { echo; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $1"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
separator "LAYER 3 — ZLUDA libcuda.so loading"
echo
python3 - <<'"'"'PYEOF'"'"'
import ctypes, os, sys

zluda_path = os.environ.get("ZLUDA_PATH", "/opt/zluda")
lib_path = os.path.join(zluda_path, "libcuda.so")

print(f"Attempting to load: {lib_path}")
try:
    lib = ctypes.CDLL(lib_path)
    print(f"OK  libcuda.so loaded successfully")
    # Probe cuInit — returns 100 (no device) if HIP has no GPU, 0 if GPU found
    cu_init = lib.cuInit
    cu_init.restype = ctypes.c_int
    result = cu_init(ctypes.c_uint(0))
    if result == 0:
        print("OK  cuInit returned 0 — GPU detected and ZLUDA is functional")
    elif result == 100:
        print(f"WARN cuInit returned 100 (CUDA_ERROR_NO_DEVICE) — ZLUDA loaded but no GPU visible")
        print("     Check Layer 2 (rocminfo) — the GPU may not be passed through correctly")
    else:
        print(f"WARN cuInit returned {result} — check ZLUDA/ROCm compatibility")
except OSError as e:
    print(f"FAIL Could not load libcuda.so: {e}")
    sys.exit(1)
PYEOF
'

# ──────────────────────────────────────────────
# Layer 4 — PyTorch CUDA via ZLUDA
# ──────────────────────────────────────────────
CMD_TORCH='
separator() { echo; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $1"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
separator "LAYER 4 — PyTorch CUDA via ZLUDA"

VENV=/data/venv-debug-zluda
PIP=$VENV/bin/pip
PYTHON=$VENV/bin/python

echo "Using isolated venv: $VENV (separate from production venv)"
echo

if [ ! -f "$PYTHON" ]; then
    echo "[INFO] Creating debug venv..."
    python3 -m venv "$VENV"
fi

echo "[INFO] Installing CUDA PyTorch (cu118, pre-CUDA-12 for ZLUDA compatibility)... (this may take a few minutes on first run)"
"$PIP" install --quiet --timeout 300 torch --index-url https://download.pytorch.org/whl/cu118

echo
echo "[TEST] torch.cuda.is_available()"
# Python script encoded as base64 to avoid single-quote/heredoc nesting issues
# inside the CMD_TORCH bash variable.
TORCH_TEST_PY='aW1wb3J0IG9zLCB0b3JjaApkaXNhYmxlX3ZhbCA9IG9zLmVudmlyb24uZ2V0KCJESVNBQkxFX0FERE1NX0NVREFfTFQiLCAibm90IHNldCIpCnByaW50KGYiICB0b3JjaCB2ZXJzaW9uICAgICAgICAgIDoge3RvcmNoLl9fdmVyc2lvbl9ffSIpCnByaW50KGYiICBESVNBQkxFX0FERE1NX0NVREFfTFQgIDoge2Rpc2FibGVfdmFsfSIpCgojIERpc2FibGUgQ1VEQSAxMisgY29kZSBwYXRocyBub3QgeWV0IGltcGxlbWVudGVkIGluIFpMVURBJ3MgUFRYIHRyYW5zbGF0b3IKdG9yY2guYmFja2VuZHMuY3Vkbm4uZW5hYmxlZCA9IEZhbHNlCnRvcmNoLmJhY2tlbmRzLmN1ZGEuZW5hYmxlX2ZsYXNoX3NkcChGYWxzZSkKdG9yY2guYmFja2VuZHMuY3VkYS5lbmFibGVfbWF0aF9zZHAoVHJ1ZSkKdG9yY2guYmFja2VuZHMuY3VkYS5lbmFibGVfbWVtX2VmZmljaWVudF9zZHAoRmFsc2UpCnRvcmNoLmJhY2tlbmRzLmN1ZGEuZW5hYmxlX2N1ZG5uX3NkcChGYWxzZSkKcHJpbnQoIiAgZmxhc2hfc2RwIGRpc2FibGVkLCBtYXRoX3NkcCBlbmFibGVkIChaTFVEQSBjb21wYXQgbW9kZSkiKQoKcHJpbnQoZiIgIGN1ZGEgYXZhaWxhYmxlOiB7dG9yY2guY3VkYS5pc19hdmFpbGFibGUoKX0iKQppZiB0b3JjaC5jdWRhLmlzX2F2YWlsYWJsZSgpOgogICAgcHJpbnQoZiIgIGRldmljZSBjb3VudCAgOiB7dG9yY2guY3VkYS5kZXZpY2VfY291bnQoKX0iKQogICAgcHJpbnQoZiIgIGRldmljZSBuYW1lICAgOiB7dG9yY2guY3VkYS5nZXRfZGV2aWNlX25hbWUoMCl9IikKICAgIHByaW50KCkKICAgIHByaW50KCJTVUNDRVNTIC0gUHlUb3JjaCBDVURBIGlzIGZ1bmN0aW9uYWwgdGhyb3VnaCBaTFVEQSIpCiAgICB4ID0gdG9yY2gub25lcygzLCAzKS5jdWRhKCkKICAgIHkgPSB0b3JjaC5vbmVzKDMsIDMpLmN1ZGEoKQogICAgeiA9IHggKyB5CiAgICBwcmludChmIiAgdGVuc29yIG9wIHRlc3Q6IHt6WzBdWzBdLml0ZW0oKX0gKGV4cGVjdGVkIDIuMCkiKQplbHNlOgogICAgcHJpbnQoKQogICAgcHJpbnQoIkZBSUwgLSB0b3JjaC5jdWRhLmlzX2F2YWlsYWJsZSgpIHJldHVybmVkIEZhbHNlIikKICAgIHByaW50KCIgIENoZWNrIGxheWVycyAyIGFuZCAzIGZvciBHUFUvWkxVREEgaXNzdWVzIikK'
echo "$TORCH_TEST_PY" | base64 -d | "$PYTHON"
'

# ──────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────
echo "Image : $IMAGE"
echo "Data  : $COMFYUI_DATA"
echo "Mode  : $MODE"
echo

case "$MODE" in
    --shell)
        separator "Interactive shell — ZLUDA env pre-configured"
        echo "  ZLUDA_PATH=/opt/zluda is set"
        echo "  LD_LIBRARY_PATH includes /opt/zluda"
        echo "  Tip: run each layer manually or step through /start.sh"
        echo
        docker run -it "${DOCKER_FLAGS[@]}" "$IMAGE" /bin/bash
        ;;

    --test-structure)
        docker run -it "${DOCKER_FLAGS[@]}" "$IMAGE" /bin/bash -c "$CMD_STRUCTURE"
        ;;

    --test-gpu)
        docker run -it "${DOCKER_FLAGS[@]}" "$IMAGE" /bin/bash -c "$CMD_GPU"
        ;;

    --test-zluda)
        docker run -it "${DOCKER_FLAGS[@]}" "$IMAGE" /bin/bash -c "$CMD_ZLUDA"
        ;;

    --test-torch)
        docker run -it "${DOCKER_FLAGS[@]}" "$IMAGE" /bin/bash -c "$CMD_TORCH"
        ;;

    --test-all)
        separator "Running all test layers"
        for layer in --test-structure --test-gpu --test-zluda --test-torch; do
            bash "$0" "$layer" || { echo; echo "STOPPED — layer $layer failed"; exit 1; }
        done
        separator "All layers passed"
        ;;

    --start)
        separator "Running start.sh (full production boot)"
        echo "ComfyUI will be available at http://localhost:8190"
        docker run -it "${DOCKER_FLAGS[@]}" -p 8190:8188 "$IMAGE" /bin/bash /start.sh
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [--shell|--test-structure|--test-gpu|--test-zluda|--test-torch|--test-all|--start]"
        exit 1
        ;;
esac
