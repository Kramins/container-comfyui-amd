#!/usr/bin/env python3
"""
ComfyUI AMD GPU Performance Benchmark

Measures GPU compute, attention, and convolution performance so you can
establish a baseline and compare results before/after each optimization.

Usage (inside container, with venv activated):
    python /benchmark/benchmark.py

Output:
    - Human-readable results to stdout
    - JSON summary to $BENCHMARK_OUTPUT (default: /data/output/benchmark_<timestamp>.json)
"""

import sys
import os
import time
import json
import datetime
import subprocess

WARMUP_ITERS = 5
BENCH_ITERS  = 25

# ─────────────────────── helpers ────────────────────────────────────────────

def header(title: str):
    width = 62
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")

def section(title: str):
    print(f"\n  ── {title}")

def row(label: str, value: str, extra: str = ""):
    extra_str = f"  {extra}" if extra else ""
    print(f"    {label:<32} {value}{extra_str}")


def save_results(results: dict, output_path: str):
    """Save results JSON to disk (called after each major section)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def bench_fn(fn, warmup: int = WARMUP_ITERS, iters: int = BENCH_ITERS) -> float:
    """Return mean execution time in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


# ─────────────────────── import torch ───────────────────────────────────────

header("ComfyUI AMD GPU Benchmark")
print(f"  Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    import torch
except ImportError:
    print("\n❌  PyTorch is not installed. Run the ComfyUI container normally first")
    print("    so the virtual environment is set up, then re-run this benchmark.")
    sys.exit(1)

results: dict = {
    "timestamp":    datetime.datetime.now().isoformat(),
    "python":       sys.version,
    "torch":        torch.__version__,
}

# ─────────────────────── GPU detection ──────────────────────────────────────

section("Environment")
row("Python", sys.version.split()[0])
row("PyTorch", torch.__version__)

# ROCm version from environment / file
rocm_ver = os.environ.get("ROCM_VERSION", "")
if not rocm_ver:
    try:
        with open("/opt/rocm/.info/version") as f:
            rocm_ver = f.read().strip()
    except Exception:
        rocm_ver = "unknown"
row("ROCm", rocm_ver)
results["rocm_version"] = rocm_ver

if not torch.cuda.is_available():
    print("\n❌  No GPU detected by PyTorch/HIP. Exiting.")
    sys.exit(1)

device      = torch.device("cuda:0")
gpu_name    = torch.cuda.get_device_name(device)
gpu_props   = torch.cuda.get_device_properties(device)
vram_total  = gpu_props.total_memory / 1024**3

row("GPU", gpu_name)
row("VRAM total", f"{vram_total:.1f} GB")
row("Num GPUs", str(torch.cuda.device_count()))

# Detect gfx architecture via rocminfo
gfx_version = "unknown"
try:
    ri = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=15)
    for line in ri.stdout.splitlines():
        line = line.strip()
        if line.startswith("Name:") and "gfx" in line:
            gfx_version = line.split()[-1]
            break
except Exception:
    pass
row("GPU arch (gfx)", gfx_version)

results["gpu_name"]      = gpu_name
results["vram_total_gb"] = round(vram_total, 1)
results["gfx_version"]   = gfx_version
results["num_gpus"]      = torch.cuda.device_count()

# ─────────────────────── optional package detection ─────────────────────────

section("Installed Optimisation Packages")

packages = {}
def _check(display: str, import_name: str) -> bool:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        row(display, f"✓  {ver}")
        packages[display] = ver
        return True
    except ImportError:
        row(display, "✗  not installed")
        packages[display] = None
        return False

_check("xformers",           "xformers")
_check("flash-attn",         "flash_attn")
_check("pytorch-triton-rocm","triton")
_check("bitsandbytes",       "bitsandbytes")
_check("onnxruntime-rocm",   "onnxruntime")
_check("torchao",            "torchao")

results["packages"] = packages

# ── determine output path early so we can save incrementally ────────────────
ts          = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
default_out = f"/data/output/benchmark_{ts}.json"
output_path = os.environ.get("BENCHMARK_OUTPUT", default_out)
save_results(results, output_path)

# ────────────���────────── warmup ─────────────────────────────────────────────

# Small warmup so the GPU clock ramps before real measurements
_ = torch.matmul(
    torch.randn(512, 512, dtype=torch.float16, device=device),
    torch.randn(512, 512, dtype=torch.float16, device=device),
)
torch.cuda.synchronize()

# ─────────────────────── FP16 matmul (TFLOPS) ───────────────────────────────

section("FP16 Matrix Multiply  (proxy for linear layers)")
M = N = K = 4096
a = torch.randn(M, K, dtype=torch.float16, device=device)
b = torch.randn(K, N, dtype=torch.float16, device=device)

ms = bench_fn(lambda: torch.matmul(a, b))
flops = 2 * M * N * K
tflops = flops / (ms / 1000) / 1e12
row(f"FP16 {M}×{K}×{N}", f"{ms:.2f} ms", f"→ {tflops:.2f} TFLOPS")
results["matmul_fp16_ms"]     = round(ms, 2)
results["matmul_fp16_tflops"] = round(tflops, 2)

# BF16
try:
    ab = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    bb = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    ms_bf = bench_fn(lambda: torch.matmul(ab, bb))
    tflops_bf = flops / (ms_bf / 1000) / 1e12
    row(f"BF16 {M}×{K}×{N}", f"{ms_bf:.2f} ms", f"→ {tflops_bf:.2f} TFLOPS")
    results["matmul_bf16_ms"]     = round(ms_bf, 2)
    results["matmul_bf16_tflops"] = round(tflops_bf, 2)
except Exception as exc:
    row("BF16", f"skipped ({exc})")

del a, b
torch.cuda.empty_cache()

# ─────────────────────── 2-D convolution (UNet proxy) ───────────────────────

section("2-D Convolution  (proxy for UNet residual blocks)")
import torch.nn as nn

conv = nn.Conv2d(320, 320, 3, padding=1).half().to(device)
x_c  = torch.randn(2, 320, 64, 64, dtype=torch.float16, device=device)

with torch.no_grad():
    ms_conv = bench_fn(lambda: conv(x_c))
row("Conv2d 320→320, 2×64×64", f"{ms_conv:.2f} ms")
results["conv2d_ms"] = round(ms_conv, 2)

del conv, x_c
torch.cuda.empty_cache()

# ─────────────────────── attention benchmarks ───────────────────────────────

section("Scaled Dot-Product Attention  (proxy for cross-attention layers)")

# Typical SDXL UNet latent: 64×64 = 4096 tokens, 8 heads, head-dim 64
B, H, S, D = 2, 8, 4096, 64
q = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
k = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
v = torch.randn(B, H, S, D, dtype=torch.float16, device=device)

# ── PyTorch native SDPA (baseline) ──
with torch.no_grad():
    ms_sdpa = bench_fn(
        lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
    )
row("Native SDPA (baseline)", f"{ms_sdpa:.2f} ms", "← baseline")
results["sdpa_native_ms"] = round(ms_sdpa, 2)

# ── xformers ──
try:
    import xformers.ops as xops
    # Quick pre-flight: tiny tensor to confirm this arch supports xformers.
    # Run in a subprocess so a segfault / hard-crash can't kill the benchmark.
    _probe = subprocess.run(
        [sys.executable, "-c",
         "import torch, xformers.ops as x; "
         "q=torch.randn(1,1,16,16,dtype=torch.float16,device='cuda'); "
         "x.memory_efficient_attention(q,q,q); "
         "torch.cuda.synchronize(); print('ok')"],
        capture_output=True, text=True, timeout=30
    )
    if _probe.returncode != 0 or "ok" not in _probe.stdout:
        stderr_tail = _probe.stderr.strip().splitlines()[-3:] if _probe.stderr.strip() else []
        stdout_tail = _probe.stdout.strip().splitlines()[-2:] if _probe.stdout.strip() else []
        detail = " | ".join(stderr_tail + stdout_tail) or "no output"
        raise RuntimeError(f"xformers probe failed: {detail}")
    # xformers expects (B, S, H, D)
    qx = q.permute(0, 2, 1, 3).contiguous()
    kx = k.permute(0, 2, 1, 3).contiguous()
    vx = v.permute(0, 2, 1, 3).contiguous()
    with torch.no_grad():
        ms_xf = bench_fn(
            lambda: xops.memory_efficient_attention(qx, kx, vx)
        )
    speedup = ms_sdpa / ms_xf
    row("xformers mem-efficient attn", f"{ms_xf:.2f} ms", f"{speedup:.2f}× faster")
    results["sdpa_xformers_ms"]  = round(ms_xf, 2)
    results["xformers_speedup"]  = round(speedup, 2)
except ImportError:
    row("xformers", "not installed")
except Exception as exc:
    row("xformers", f"not supported on this arch  ({exc})")
    results["sdpa_xformers_ms"] = "unsupported"

# ── flash-attn ──
try:
    from flash_attn import flash_attn_func  # requires flash-attn v2+
    _probe_fa = subprocess.run(
        [sys.executable, "-c",
         "import torch; from flash_attn import flash_attn_func; "
         "q=torch.randn(1,1,16,16,dtype=torch.float16,device='cuda'); "
         "flash_attn_func(q,q,q); torch.cuda.synchronize(); print('ok')"],
        capture_output=True, text=True, timeout=30
    )
    if _probe_fa.returncode != 0 or "ok" not in _probe_fa.stdout:
        raise RuntimeError(
            _probe_fa.stderr.strip().splitlines()[-1] if _probe_fa.stderr.strip() else "flash-attn probe failed"
        )
    qf = q.permute(0, 2, 1, 3).contiguous()
    kf = k.permute(0, 2, 1, 3).contiguous()
    vf = v.permute(0, 2, 1, 3).contiguous()
    with torch.no_grad():
        ms_fa = bench_fn(lambda: flash_attn_func(qf, kf, vf))
    speedup_fa = ms_sdpa / ms_fa
    row("flash-attn v2", f"{ms_fa:.2f} ms", f"{speedup_fa:.2f}× faster")
    results["sdpa_flash_ms"]      = round(ms_fa, 2)
    results["flash_attn_speedup"] = round(speedup_fa, 2)
except ImportError:
    row("flash-attn", "not installed")
except Exception as exc:
    row("flash-attn", f"not supported on this arch  ({exc})")
    results["sdpa_flash_ms"] = "unsupported"

del q, k, v
torch.cuda.empty_cache()

# ─────────────────────── torch.compile SDPA ─────────────────────────────────

section("torch.compile SDPA  (JIT-fused via triton-rocm)")
try:
    import triton  # noqa: F401 — just confirms triton-rocm is present

    B2, H2, S2, D2 = 2, 8, 4096, 64
    qc = torch.randn(B2, H2, S2, D2, dtype=torch.float16, device=device)
    kc = torch.randn(B2, H2, S2, D2, dtype=torch.float16, device=device)
    vc = torch.randn(B2, H2, S2, D2, dtype=torch.float16, device=device)

    @torch.compile(backend="inductor")
    def compiled_sdpa(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # First call triggers JIT compilation (will be slow — expected)
    row("Compiling (first call)...", "please wait", "")
    with torch.no_grad():
        _ = compiled_sdpa(qc, kc, vc)
        torch.cuda.synchronize()

    # Actual benchmark after warm-up
    with torch.no_grad():
        ms_compile = bench_fn(lambda: compiled_sdpa(qc, kc, vc))

    # Re-measure native for a fair side-by-side comparison
    qn = torch.randn(B2, H2, S2, D2, dtype=torch.float16, device=device)
    kn = torch.randn(B2, H2, S2, D2, dtype=torch.float16, device=device)
    vn = torch.randn(B2, H2, S2, D2, dtype=torch.float16, device=device)
    with torch.no_grad():
        ms_native2 = bench_fn(
            lambda: torch.nn.functional.scaled_dot_product_attention(qn, kn, vn)
        )

    speedup_compile = ms_native2 / ms_compile
    row("Native SDPA (re-measured)", f"{ms_native2:.2f} ms", "← comparison base")
    row("torch.compile SDPA",       f"{ms_compile:.2f} ms", f"{speedup_compile:.2f}× faster")
    results["sdpa_compiled_ms"]       = round(ms_compile, 2)
    results["torch_compile_speedup"]  = round(speedup_compile, 2)

    del qc, kc, vc, qn, kn, vn
    torch.cuda.empty_cache()

except ImportError:
    row("torch.compile SDPA", "triton-rocm not installed")
    results["sdpa_compiled_ms"] = None
except Exception as exc:
    row("torch.compile SDPA", f"error: {exc}")
    results["sdpa_compiled_ms"] = None

save_results(results, output_path)

# ─────────────────────── memory bandwidth ───────────────────────────────────

section("Memory Bandwidth")
# 256M float32 = 1 GB per buffer; copy = 2 GB transferred
EL = 256 * 1024 * 1024
src = torch.randn(EL, dtype=torch.float32, device=device)
dst = torch.empty(EL, dtype=torch.float32, device=device)

ms_bw = bench_fn(lambda: dst.copy_(src))
bytes_transferred = EL * 4 * 2          # read + write
bw_gbs = bytes_transferred / (ms_bw / 1000) / 1e9
row("Copy 2 GB", f"{ms_bw:.2f} ms", f"→ {bw_gbs:.0f} GB/s")
results["memory_bandwidth_gbs"] = round(bw_gbs, 0)

del src, dst
torch.cuda.empty_cache()

# ─────────────────────── VRAM footprint ─────────────────────────────────────

section("VRAM After Benchmarks")
alloc_gb = torch.cuda.memory_allocated(device) / 1024**3
resv_gb  = torch.cuda.memory_reserved(device)  / 1024**3
row("Allocated", f"{alloc_gb:.2f} GB")
row("Reserved",  f"{resv_gb:.2f} GB")
row("Total",     f"{vram_total:.2f} GB")
results["vram_allocated_gb"] = round(alloc_gb, 2)
results["vram_reserved_gb"]  = round(resv_gb, 2)

header("SUMMARY")
row("GPU",                f"{gpu_name}  [{gfx_version}]")
row("FP16 TFLOPS",        f"{results.get('matmul_fp16_tflops', 'n/a')} TFLOPS")
row("Memory bandwidth",   f"{results.get('memory_bandwidth_gbs', 'n/a')} GB/s")
row("Conv2d latency",     f"{results.get('conv2d_ms', 'n/a')} ms")
row("SDPA (native)",      f"{results.get('sdpa_native_ms', 'n/a')} ms  ← baseline")

xf_ms = results.get("sdpa_xformers_ms")
if isinstance(xf_ms, float):
    row("SDPA (xformers)",
        f"{xf_ms} ms",
        f"({results.get('xformers_speedup', '?')}× vs native)")
elif xf_ms == "unsupported":
    row("SDPA (xformers)", "unsupported on this arch")
else:
    row("SDPA (xformers)", "not available")

fa_ms = results.get("sdpa_flash_ms")
if isinstance(fa_ms, float):
    row("SDPA (flash-attn)",
        f"{fa_ms} ms",
        f"({results.get('flash_attn_speedup', '?')}× vs native)")
elif fa_ms == "unsupported":
    row("SDPA (flash-attn)", "unsupported on this arch")
else:
    row("SDPA (flash-attn)", "not available")

tc_ms = results.get("sdpa_compiled_ms")
if isinstance(tc_ms, float):
    row("SDPA (torch.compile)",
        f"{tc_ms} ms",
        f"({results.get('torch_compile_speedup', '?')}× vs native)")
else:
    row("SDPA (torch.compile)", "not available")

# ─────────────────────── save final JSON ────────────────────────────────────
save_results(results, output_path)
print(f"\n  📄 Results saved → {output_path}\n")
