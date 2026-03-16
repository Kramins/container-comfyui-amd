#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run-benchmark.sh  –  Run the AMD GPU performance benchmark inside the
#                       ComfyUI container.
#
# Usage:
#   ./run-benchmark.sh [VERSION]
#
# VERSION defaults to "latest".  Use "git" for the git-built image.
#
# Prerequisites:
#   Run the container normally at least once so that /data/venv is set up
#   (torch + ComfyUI requirements installed).  The benchmark mounts the same
#   /data volume and reuses that environment.
#
# Results are written to ./benchmark/results/ on the host.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VERSION="${1:-latest}"
IMAGE="ghcr.io/kramins/comfyui-amd:${VERSION}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env file if present (values set there override defaults below)
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -o allexport
    source "${SCRIPT_DIR}/.env"
    set +o allexport
fi

RESULTS_DIR="${SCRIPT_DIR}/benchmark/results"
COMFYUI_DATA="${COMFYUI_DATA:-${HOME}/comfyui}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_FILE="benchmark_${TIMESTAMP}.json"

mkdir -p "${RESULTS_DIR}"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          ComfyUI AMD GPU Benchmark                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Image:    ${IMAGE}"
echo "  Data dir: ${COMFYUI_DATA}"
echo "  Results:  ${RESULTS_DIR}/${RESULT_FILE}"
echo ""

# ── sanity check: venv must already exist from a prior container run ─────────
if [ ! -f "${COMFYUI_DATA}/venv/bin/python" ]; then
    echo "❌  Virtual environment not found at ${COMFYUI_DATA}/venv"
    echo ""
    echo "    Run the container normally once first so that dependencies are"
    echo "    installed, then re-run this benchmark:"
    echo ""
    echo "      ./run-container.sh ${VERSION}"
    echo ""
    exit 1
fi

# ── install / update optimisation packages into the existing venv ───────────
ROCM_VER="${ROCM_VERSION:-6.4}"
HSA_GFX="${HSA_OVERRIDE_GFX_VERSION:-}"
echo "  Installing optimisation packages into venv (rocm${ROCM_VER})..."
docker run --rm --privileged \
    -v /dev:/dev \
    -v "${COMFYUI_DATA}:/data" \
    -e "ROCM_VER=${ROCM_VER}" \
    --entrypoint /bin/bash \
    "${IMAGE}" \
    -c '
        set -e
        PIP=/data/venv/bin/pip
        echo "[install] onnxruntime-rocm bitsandbytes torchao..."
        $PIP install -q onnxruntime-rocm bitsandbytes torchao
        # xformers: detect GPU arch and skip on gfx1030 (segfaults with pre-built wheel)
        GFX=$(rocminfo 2>/dev/null | grep "Name:" | grep -o "gfx[0-9]*" | head -1)
        if [ "${GFX}" = "gfx1030" ]; then
            echo "[install] Skipping xformers on gfx1030 (pre-built wheel crashes on RDNA2)"
        else
            echo "[install] xformers..."
            $PIP install -q xformers --index-url "https://download.pytorch.org/whl/rocm${ROCM_VER}" || \
                echo "[install] xformers failed, continuing"
        fi
        echo "[install] done."
    '
echo ""

# ── run the benchmark ────────────────────────────────────────────────────────
docker run --rm -it --privileged \
    -v /dev:/dev \
    -v "${COMFYUI_DATA}:/data" \
    -v "${SCRIPT_DIR}/benchmark/benchmark.py:/benchmark/benchmark.py:ro" \
    -v "${RESULTS_DIR}:/data/output" \
    -e "BENCHMARK_OUTPUT=/data/output/${RESULT_FILE}" \
    -e "VENV_PATH=/data/venv" \
    -e "PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512" \
    ${HSA_GFX:+-e "HSA_OVERRIDE_GFX_VERSION=${HSA_GFX}"} \
    --name comfyui-benchmark \
    --entrypoint /bin/bash \
    "${IMAGE}" \
    -c '/data/venv/bin/python /benchmark/benchmark.py 2>&1 | tee /data/output/benchmark_$(date +%Y%m%d_%H%M%S).log'

echo ""
echo "✅  Benchmark complete."
echo "    JSON:  ${RESULTS_DIR}/${RESULT_FILE}"
echo "    Logs:  ${RESULTS_DIR}/"
echo ""

# ── optional: print a quick comparison of all past results ──────────────────
JSON_COUNT=$(find "${RESULTS_DIR}" -name "benchmark_*.json" 2>/dev/null | wc -l)
if [ "${JSON_COUNT}" -gt 1 ]; then
    echo "──────────────────────────────────────────────────────────"
    echo "  Historical results (newest first):"
    echo ""
    printf "  %-24s  %8s  %8s  %8s  %10s  %10s\n" \
        "Timestamp" "FP16 TF" "Conv ms" "SDPA ms" "BW GB/s" "steps/s"
    echo "  ────────────────────────────────────────────────────────"
    find "${RESULTS_DIR}" -name "benchmark_*.json" | sort -r | while read -r f; do
        ts=$(basename "$f" .json | sed 's/benchmark_//')
        fp16=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('matmul_fp16_tflops','?'))" 2>/dev/null)
        conv=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('conv2d_ms','?'))" 2>/dev/null)
        sdpa=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('sdpa_native_ms','?'))" 2>/dev/null)
        bw=$(python3   -c "import json; d=json.load(open('$f')); print(d.get('memory_bandwidth_gbs','?'))" 2>/dev/null)
        sps=$(python3  -c "import json; d=json.load(open('$f')); g=d.get('comfyui_gen',{}); print(g.get('steps_per_sec','—') if isinstance(g,dict) else '—')" 2>/dev/null)
        printf "  %-24s  %8s  %8s  %8s  %10s  %10s\n" "$ts" "$fp16" "$conv" "$sdpa" "$bw" "$sps"
    done
    echo ""
fi
