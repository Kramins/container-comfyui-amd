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
VENV_PATH    = os.environ.get("VENV_PATH", "/data/venv")

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

# ─────────────────────── real-world ComfyUI generation ───────────────────────

header("Real-World ComfyUI Generation")

import glob
import signal
import urllib.request
import urllib.error

_BENCH_PORT = 8289   # use a non-default port so it never clashes with a live instance
_STEPS      = 20
_WIDTH      = 512
_HEIGHT     = 512
_PROMPT     = "a scenic mountain landscape at sunset, photorealistic, high quality"
_NEGATIVE   = "blurry, low quality, ugly, watermark"

def _find_checkpoint() -> str | None:
    """Return the first .safetensors or .ckpt found in /data/models/checkpoints/."""
    for pattern in ("*.safetensors", "*.ckpt"):
        hits = sorted(glob.glob(f"/data/models/checkpoints/{pattern}"))
        # Prefer smaller / simpler models for a quick benchmark (skip huge fp8 LTX etc.)
        for h in hits:
            name = os.path.basename(h)
            size_gb = os.path.getsize(h) / 1024**3
            # Use the first model that looks like SD1.5 or SDXL (< 10 GB)
            if size_gb < 10:
                return name
        if hits:
            return os.path.basename(hits[0])
    return None


def _wait_for_comfyui(port: int, proc: subprocess.Popen, timeout: int = 180) -> bool:
    """Poll /system_stats until ready or timeout. Drain+print stdout to see what ComfyUI says."""
    import select
    url      = f"http://127.0.0.1:{port}/system_stats"
    deadline = time.monotonic() + timeout
    log_buf  = []
    while time.monotonic() < deadline:
        # drain stdout so the pipe never blocks the child process
        if proc.stdout:
            while True:
                ready, _, _ = select.select([proc.stdout], [], [], 0)
                if not ready:
                    break
                chunk = os.read(proc.stdout.fileno(), 8192)
                if not chunk:
                    break
                text = chunk.decode(errors="replace")
                log_buf.append(text)
                for line in text.splitlines():
                    print(f"  [ComfyUI] {line}", flush=True)
        # Bail early if server died
        if proc.poll() is not None:
            return False, log_buf
        try:
            urllib.request.urlopen(url, timeout=2)
            return True, log_buf
        except Exception:
            time.sleep(1)
    return False, log_buf


def _build_workflow(ckpt_name: str, steps: int, width: int, height: int,
                    prompt: str, negative: str, seed: int = 42) -> dict:
    return {
        "4":  {"class_type": "CheckpointLoaderSimple",
               "inputs":     {"ckpt_name": ckpt_name}},
        "5":  {"class_type": "EmptyLatentImage",
               "inputs":     {"width": width, "height": height, "batch_size": 1}},
        "6":  {"class_type": "CLIPTextEncode",
               "inputs":     {"text": prompt,   "clip": ["4", 1]}},
        "7":  {"class_type": "CLIPTextEncode",
               "inputs":     {"text": negative, "clip": ["4", 1]}},
        "3":  {"class_type": "KSampler",
               "inputs":     {"seed": seed, "steps": steps, "cfg": 7.0,
                              "sampler_name": "euler", "scheduler": "normal",
                              "denoise": 1.0,
                              "model":         ["4", 0],
                              "positive":      ["6", 0],
                              "negative":      ["7", 0],
                              "latent_image":  ["5", 0]}},
        "8":  {"class_type": "VAEDecode",
               "inputs":     {"samples": ["3", 0], "vae": ["4", 2]}},
        "9":  {"class_type": "SaveImage",
               "inputs":     {"filename_prefix": "benchmark", "images": ["8", 0]}},
    }


ckpt = _find_checkpoint()
if ckpt is None:
    row("Checkpoint", "none found in /data/models/checkpoints — skipping")
    results["comfyui_gen"] = "no_model"
else:
    # Locate ComfyUI source — it's baked into /app for release builds;
    # for git builds it gets cloned at runtime by start.sh.
    # If /app/main.py is missing (benchmark container, git build), clone it.
    _app_path = "/app"
    if not os.path.exists(f"{_app_path}/main.py"):
        _clone_path = "/tmp/comfyui-bench"
        if not os.path.exists(f"{_clone_path}/main.py"):
            row("", "Cloning ComfyUI for generation test...")
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/Comfy-Org/ComfyUI.git", _clone_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        _app_path = _clone_path

    row("Checkpoint", ckpt)
    row("Resolution", f"{_WIDTH}×{_HEIGHT},  {_STEPS} steps")
    row("", "Starting ComfyUI server on port 8289...")

    # For RDNA1/RDNA2 xformers pre-built wheel segfaults — force pytorch attention.
    # Detect from HSA_OVERRIDE_GFX_VERSION or gfx_version (already captured above).
    _gfx = gfx_version or os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
    _rdna12 = any(_gfx.startswith(p) for p in ("gfx101", "gfx103", "10.1", "10.3"))
    _attn_flag = ["--use-pytorch-cross-attention"] if _rdna12 else []

    def _drain_comfyui(proc: subprocess.Popen, label: str = "[ComfyUI]") -> None:
        """Non-blocking drain of ComfyUI stdout; print each line prefixed."""
        import select
        if proc.stdout is None:
            return
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0)
            if not ready:
                break
            chunk = os.read(proc.stdout.fileno(), 8192)
            if not chunk:
                break
            for line in chunk.decode(errors="replace").splitlines():
                print(f"  {label} {line}", flush=True)

    _comfyui_proc = None
    try:
        _env = os.environ.copy()
        _env.setdefault("MIOPEN_USER_DB_PATH",  "/data/miopen")
        _env.setdefault("MIOPEN_DISABLE_CACHE", "0")

        _cmd = (
            [f"{VENV_PATH}/bin/python", f"{_app_path}/main.py",
             "--listen", "127.0.0.1",
             "--port",   str(_BENCH_PORT),
             "--base-directory", "/data",
             "--fast",
             "--disable-auto-launch"]
            + _attn_flag
        )
        row("", f"ComfyUI cmd: {' '.join(_cmd[-4:])}")
        _comfyui_proc = subprocess.Popen(
            _cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_env,
        )

        ready, _log_buf = _wait_for_comfyui(_BENCH_PORT, _comfyui_proc, timeout=180)
        if not ready:
            _drain_comfyui(_comfyui_proc)
            _tail = "".join(_log_buf)[-4000:]
            raise RuntimeError(
                f"ComfyUI did not become ready within 180 s.\nLast output:\n{_tail}"
            )

        row("", "Server ready — submitting workflow...")

        # ── submit prompt ──────────────────────────────────────────────
        workflow  = _build_workflow(ckpt, _STEPS, _WIDTH, _HEIGHT, _PROMPT, _NEGATIVE)
        payload   = json.dumps({"prompt": workflow}).encode()
        req       = urllib.request.Request(
                        f"http://127.0.0.1:{_BENCH_PORT}/prompt",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST")
        resp      = urllib.request.urlopen(req, timeout=30)
        prompt_id = json.loads(resp.read())["prompt_id"]

        t_submit  = time.monotonic()
        row("", f"Queued  prompt_id={prompt_id}")

        # ── poll /history until done ───────────────────────────────────
        deadline = time.monotonic() + 3600   # 1-hour hard limit (MIOpen first-run tuning can take long)
        _poll_n  = 0
        while time.monotonic() < deadline:
            # Stream ComfyUI output so we can see what it's doing
            _drain_comfyui(_comfyui_proc)

            # Detect server death early — don't wait 600 s for nothing
            if _comfyui_proc.poll() is not None:
                _drain_comfyui(_comfyui_proc)  # final drain
                raise RuntimeError(
                    f"ComfyUI server exited unexpectedly (code {_comfyui_proc.returncode})"
                )

            time.sleep(5)
            _poll_n += 1
            hist_url = f"http://127.0.0.1:{_BENCH_PORT}/history/{prompt_id}"
            try:
                hist_resp = urllib.request.urlopen(hist_url, timeout=10)
                hist      = json.loads(hist_resp.read())
            except Exception as _he:
                if _poll_n % 6 == 0:  # print every ~30 s so we know it's alive
                    print(f"  [poll] waiting... ({int(time.monotonic()-t_submit)}s elapsed, /history: {_he})", flush=True)
                continue

            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI generation error: {msgs}")
                if status.get("completed", False):
                    break
        else:
            raise RuntimeError("Generation timed out after 3600 s")

        total_sec      = time.monotonic() - t_submit
        steps_per_sec  = _STEPS / total_sec

        row("Total time",   f"{total_sec:.1f} s")
        row("Steps/sec",    f"{steps_per_sec:.2f}")
        row("ms/step",      f"{total_sec * 1000 / _STEPS:.0f} ms")

        results["comfyui_gen"] = {
            "model":          ckpt,
            "steps":          _STEPS,
            "width":          _WIDTH,
            "height":         _HEIGHT,
            "total_sec":      round(total_sec, 2),
            "steps_per_sec":  round(steps_per_sec, 2),
            "ms_per_step":    round(total_sec * 1000 / _STEPS, 0),
        }
        save_results(results, output_path)

    except Exception as exc:
        # Drain any remaining ComfyUI output before printing the error
        if _comfyui_proc:
            _drain_comfyui(_comfyui_proc)
        row("ComfyUI generation", f"failed: {exc}")
        results["comfyui_gen"] = {"error": str(exc)}
    finally:
        if _comfyui_proc and _comfyui_proc.poll() is None:
            _comfyui_proc.send_signal(signal.SIGTERM)
            try:
                _comfyui_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                _comfyui_proc.kill()

# ─────────────────────── summary ─────────────────────────────────────────────

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

gen = results.get("comfyui_gen")
if isinstance(gen, dict) and "total_sec" in gen:
    row("ComfyUI generation",
        f"{gen['total_sec']:.1f} s",
        f"  {gen.get('steps_per_sec', '?'):.2f} steps/s  "
        f"({gen.get('model', '?')}, {gen.get('steps', '?')} steps, "
        f"{gen.get('width', '?')}×{gen.get('height', '?')})")
elif isinstance(gen, dict) and "error" in gen:
    row("ComfyUI generation", f"failed — {gen['error'][:80]}")
elif gen == "no_model":
    row("ComfyUI generation", "skipped — no checkpoint found in /data/models/checkpoints")
else:
    row("ComfyUI generation", "skipped")

# ─────────────────────── save final JSON ────────────────────────────────────
save_results(results, output_path)
print(f"\n  📄 Results saved → {output_path}\n")
