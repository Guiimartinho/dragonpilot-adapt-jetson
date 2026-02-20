# DragonPilot Jetson AGX Xavier - Optimization Details

Single source of truth for all Jetson optimizations, benchmarks, bugs fixed, and files.

Hardware: 8 cores ARM Carmel 2.26GHz, 512 CUDA cores Volta sm_72, 64 tensor cores, 32GB RAM, 2x NVDEC, NVENC, 2x NVDLA, TensorRT, JetPack 5.1.4

---

## OPTIMIZATION STATUS

| # | Optimization | Status | Result |
|---|-------------|--------|--------|
| 1 | FP16 end-to-end (Tensor Cores) | Done | compile3.py keeps FP16 on GPU |
| 2 | CUDA Graphs (JIT_BATCH_SIZE=16) | Done | Kernels consolidated into CUDA graphs, ~8.8ms inference |
| 3 | Tensor Cores (TC=1) | Done | 64 Volta tensor cores active |
| 4 | VisionBuf mapped memory (cudaHostRegisterMapped) | Done | GPU-direct access via cudaHostGetDevicePointer |
| 5 | TensorRT runner + install script | Done | Auto-fallback to tinygrad |
| 6 | NVDLA runner | Done | 3-tier fallback: DLA0→DLA1→GPU TRT→tinygrad |
| 7 | Camera CSI V4L2 | Ready | For real camera, inactive during replay |
| 8 | DPMS disable | Done | Prevents UI freeze every 10min |
| 9 | VNC optimized | Done | CPU from 37% to ~2% idle |
| 10 | NVDEC hardware decode | Done | h264_nvv4l2dec/hevc_nvv4l2dec |
| 11 | NVENC hardware encode | Done | h264_nvmpi/h264_nvenc, CBR, zero-latency |
| 12 | Native CUDA kernels | Done | transform.cu + loadyuv.cu via nvcc sm_72 |
| 13 | Core affinity (8 cores) | Done | Optimized distribution, reduced core 7 overload |
| 14 | MPC N=32→N=24 | Done | 25% faster lateral planning |
| 15 | Cache slip_factor | Done | _slip_factor_cache with auto-invalidation |
| 16 | Parked mode (10W) | Done | nvpmodel -m 1, disables cores 4-7 |
| 17 | Dynamic Frequency Scaling | Done | CPU/GPU/EMC adaptive |
| 18 | Fan controller PID | Done | Target 65C driving, 70C parked, 5% hysteresis |
| 19 | tmpfs log buffer | Done | /dev/shm staging, flush NVMe 5s |
| 20 | Huge Pages CUDA | Done | 256 x 2MB (512MB) for TLB |
| 21 | CUDA zero-copy preprocessing (driving) | Done | 46ms→13.66ms (3.4x), 0% frame drops |
| 22 | Benchmark script | Done | tools/jetson/benchmark_modeld.py |
| 23 | Ring buffer optimization | Done | Single cudaMemcpy replaces sequential loop |
| 24 | loadyuv kernel truncation fix | Done | Handles last <8 bytes correctly |
| 25 | CUDA pipeline safety | Done | Bounds checks, slot reuse, error validation |
| 26 | TensorRT/DLA Python 3.8+ fix | Done | `from __future__ import annotations` |
| 27 | CUDA zero-copy preprocessing (dmon) | Done | CudaMonitoringModelFrame in cuda_frame_bridge |
| 28 | Batched .tolist() in fill_model_msg | Done | 42 calls→~12, bulk T.tolist() conversions |
| 29 | UI core affinity fix | Done | UI pinned to core 6, no longer collides with locationd |
| 30 | BEAM=2 autotuner | Done | tinygrad kernel optimization for Volta sm_72 |
| 31 | UI window centering | Done | Undecorated window + vertical center on display |
| 32 | VSync enabled by default | Done | Smooth 60 FPS rendering, no tearing |
| 33 | MSAA disabled on Jetson | Done | Saves 4x fragment shader work |
| 34 | UI 60 FPS default all platforms | Done | Was 20 FPS on tici, now 60 everywhere |
| 35 | pandaStates publisher for demo | Done | Enables ignition→started→model rendering |

---

## BENCHMARKS (Single Source of Truth)

> Measured on real system running replay + modeld + dmonitoringmodeld + UI, 60s collection, 1200 frames.

### modeld End-to-End (CUDA Zero-Copy + BEAM=2, 60s, 1200 frames)

| Metric | Value |
|--------|-------|
| **modeld execution (median)** | **13.66ms** |
| modeld execution (mean) | 13.76ms |
| modeld execution (P95) | 14.74ms |
| modeld execution (P99) | 15.64ms |
| modeld execution (min) | 13.13ms |
| modeld execution (max) | 18.74ms |
| modeld stddev | 0.47ms |
| **Frame drops** | **0 (0.00%)** |
| **FPS** | **20.0 (stable)** |
| Errors (>50ms) | 0 |

### dmonitoringmodeld (tinygrad CUDA, 60s, 1201 frames)

| Metric | Value |
|--------|-------|
| **dmon execution (median)** | **20.65ms** |
| dmon execution (mean) | 21.26ms |
| dmon execution (P95) | 24.72ms |
| dmon execution (P99) | 26.10ms |
| dmon execution (min) | 19.22ms |
| dmon execution (max) | 33.47ms |

### OpenCL vs CUDA Zero-Copy

| Metric | OpenCL (POCL) | CUDA Zero-Copy | Improvement |
|--------|--------------|----------------|-------------|
| modeld execution | ~46ms | **13.66ms** | **3.4x faster** |
| Frame drops | frequent | **0%** | eliminated |
| Pipeline | CL→CPU→CUDA roundtrip | GPU direct (zero-copy) | no D2H/H2D |

### System Resources (60s average)

| Metric | Value |
|--------|-------|
| CPU temp (avg / max) | 48.9C / 49.0C |
| GPU temp (avg) | 48.5C |
| GPU usage (avg) | 8.6% |
| RAM usage (avg) | 18.0% (~5.8 GB / 32 GB) |
| cameraOdometry frames | 1200 (100%) |
| UI FPS | ~1-2 fps (limited by replay rate) |
| x11vnc CPU | ~2% idle / ~29% streaming |
| Stress test 12h | 0 restarts, 0 errors |

### Comparison with comma 3 (Qualcomm Snapdragon 845)

| Metric | comma 3 (official) | Jetson AGX Xavier | Status |
|--------|-------------------|-------------------|--------|
| modeld latency | ~12ms | **13.66ms** | Competitive |
| modeld FPS | 20 | **20** | Equal |
| DMS latency | ~15ms | **20.65ms** | Acceptable |
| Errors (>50ms) | 0 | **0** | Equal |
| GPU headroom | ~60% | **91.4%** | Much better |

### Individual Model Inference (tinygrad CUDA Graphs)

| Model | Size | Time |
|-------|------|------|
| driving_vision | 50.3M | ~8.7ms |
| driving_policy | 7.0M | ~3.3ms |
| dmonitoring_model | 9.6M | ~6.5ms |

Compiled with `DEV=CUDA FLOAT16=1 JIT_BATCH_SIZE=16 TC=1 BEAM=2`.

---

## BUGS FIXED

- `USE_TC=1` → `TC=1`: tinygrad reads `ContextVar("TC", 1)`, not `USE_TC`. Tensor cores were not activating.
- `CUDA_OPT=1` removed: not a real tinygrad variable, zero effect.
- False zero-copy in commonmodel.h: `clEnqueueMapBuffer` with POCL always returns different pointer. Simplified to honest `clEnqueueReadBuffer`.
- DPMS X11: 600s standby disabled virtual display, causing UI at 1fps.
- CUDA stream invalidation: `cudaStreamCreate()` before tinygrad init created streams invalidated by CUDA graph compilation. Fix: use default stream (0).
- Ring buffer temporal: wrong implementation caused zeroed output. Fix: 5 slots with correct per-frame shift.
- YUV420 upload size: VisionBuf with padding between Y and UV (`uv_offset > stride*height`). Fix: `uv_offset + stride*height/2`.
- `if (ext_stream)` false for stream=0: in C++, `cudaStream_t(0)` tests as nullptr. Fix: `bool use_ext` flag.
- `gpu_initialized_` race: flag was set before cudaMalloc validation. Fix: set only after all allocations succeed.
- `g_next_id` overflow: never reset after MAX_FRAMES=8 creates. Fix: slot reuse (find first nullptr).
- Missing bounds checks in cuda_frame_bridge: all g_frames[] access now validated.
- loadyuv/copy kernel truncation: last <8 bytes were dropped. Fix: byte-by-byte fallback for remainder.
- `cudaHostUnregister` without error check. Fix: check return and handle `cudaErrorHostMemoryNotRegistered`.
- TensorRT runner Python syntax: `dict[str, type]` requires Python 3.10+. Fix: `from __future__ import annotations`.
- dmonitoringmodeld exception catch: `except ImportError` didn't catch `TypeError`/`RuntimeError` from trt_runtime.so. Fix: `except Exception`.
- UI core 6 collides with locationd core 0: UI used `cores = {0}` on Jetson. Fix: `cores = {6}` and apply after JETSON check.
- fill_model_msg 42x `.tolist()` overhead: each small numpy array called `.tolist()` individually. Fix: batch via `array.T.tolist()` (1 call for 3-6 columns).
- UI fullscreen mode clips content: `FLAG_FULLSCREEN_MODE` on 1920x960 window caused top clipping on 1920x1080 display. Fix: use `FLAG_WINDOW_UNDECORATED` + `set_window_position(0, y_offset)` to center vertically.
- Model lines not rendering in demo: UI requires `pandaStates` with `ignitionLine=True` for `ui_state.ignition=True`, which gates `ui_state.started` and `model_renderer._render()`. Without pandaStates, model visualization was silently skipped.
- MSAA 4x on Jetson wasting GPU: `FLAG_MSAA_4X_HINT` was applied to all platforms. Fix: only enable on PC where GPU headroom exists.

---

## IMPLEMENTATION DETAILS

### Phase 1: Model Inference

**1.1 VisionBuf Mapped Memory** — `msgq_repo/msgq/visionipc/visionbuf_jetson.cc`

Uses `cudaHostRegister(Mapped)` + `cudaHostGetDevicePointer()` for GPU-direct access to shared memory pages. GPU can read VisionBuf data directly without explicit cudaMemcpy.

**1.2 CUDA Graphs** — `selfdrive/modeld/modeld.py`, `SConscript`

`JIT_BATCH_SIZE=32` consolidates kernels into CUDA graphs: Run 0 baseline, Run 1 JIT capture, Run 2 graph capture, Runs 3+ graph replay (~8.8ms).

**1.3 FP16 End-to-End** — `tinygrad_repo/examples/openpilot/compile3.py`

Removes `.cast('float32')` final when `FLOAT16=1` + `DEV=CUDA`, keeping output FP16 on GPU.

**1.4 Tensor Cores** — `TC=1` activates 64 Volta sm_72 tensor cores for FP16 GEMM.

**1.5 TensorRT Backend** — `selfdrive/modeld/runners/tensorrt_runner.py` (317 lines)

Complete runner with ctypes bridge to `trt_runtime.so` (C++ TRT API). ONNX→engine build via trtexec, FP16, engine caching in `/data/trt_engines/`, LayerNorm decomposition for TRT 8.5.

**1.6 NVDLA** — `selfdrive/modeld/runners/dla_runner.py` (128 lines)

Fallback: DLA core 0 → DLA core 1 → GPU TensorRT.

**1.7 Native CUDA Kernels** — `transform.cu`, `loadyuv.cu`, headers

Replaces OpenCL kernels, compiled with nvcc -arch=sm_72. Eliminates POCL interop overhead (~12ms saved).

**1.8 CUDA Zero-Copy Preprocessing** — `commonmodel_cuda.h`, `cuda_frame_bridge.cpp`, `modeld.py`

Pipeline: VisionBuf host SHM → cudaMemcpy H2D → CUDA transform+loadyuv → device ptr → Tensor.from_blob() (zero-copy). Default CUDA stream (0) avoids tinygrad context invalidation. Lazy GPU alloc after tinygrad init. Ring buffer: single cudaMemcpy shift (was sequential loop).

**1.9 CUDA Zero-Copy dmonitoringmodeld** — `cuda_frame_bridge.cpp`, `dmonitoringmodeld.py`

CudaMonitoringModelFrame: VisionBuf → cudaMemcpy H2D → CUDA transform → device ptr → Tensor.from_blob(). Same zero-copy pattern as driving model. Falls back to OpenCL if CUDA bridge not available.

**1.10 Batched .tolist() Optimization** — `fill_model_msg.py`

Reduced 42 individual `.tolist()` calls to ~12 bulk conversions. 2D numpy arrays pre-converted via `array.T.tolist()` (1 call per structure instead of per-column). Applies to plan, lane lines, road edges, leads, meta, and pose data.

**1.11 BEAM=2 Autotuner** — `SConscript`

tinygrad `BEAM=2` searches 2 kernel variants during compilation, selecting the fastest for Volta sm_72. One-time ~30min cost, results cached in .pkl files.

**1.12 Huge Pages** — `system/hardware/jetson/hugepages.py`

256 x 2MB = 512MB pre-allocated for CUDA. THP in madvise mode. 5-10% TLB improvement.

### Phase 2: Video Hardware Acceleration

**2.1 NVDEC** — `tools/replay/nvdec_decoder.cc` (247 lines)

Dual decoder: h264_nvv4l2dec → CUDA hwaccel → software fallback.

**2.2 NVENC** — `system/loggerd/encoder/nvenc_encoder.cc` (253 lines)

Dual codec: h264_nvmpi → h264_nvenc → software fallback. CBR, zero-latency.

### Phase 3: UI

**3.1 DPMS Disable** — `scripts/jetson_replay.sh`

Prevents X11 standby from freezing UI at 1fps.

**3.2 VNC Optimized** — `-wait 50 -defer 30 -noxdamage`, CPU from 37% to ~2% idle.

**3.3 Window Centering** — `system/ui/lib/application.py`

Replaced `FLAG_FULLSCREEN_MODE` with `FLAG_WINDOW_UNDECORATED` + `set_window_position(0, y_offset)`. Centers the 1920x960 window vertically on 1920x1080 display (60px top + 960px content + 60px bottom). Prevents top clipping that occurred with fullscreen mode.

**3.4 VSync + 60 FPS Default** — `system/ui/lib/application.py`

VSync enabled by default (`ENABLE_VSYNC=1`). FPS target set to 60 for all platforms (was 20 on tici/tizi). Eliminates tearing and provides smooth rendering.

**3.5 MSAA Disabled on Jetson** — `system/ui/lib/application.py`

`FLAG_MSAA_4X_HINT` only applied on PC. On Jetson/TICI, MSAA is skipped to save 4x fragment shader work, freeing GPU for model inference.

**3.6 Display Resolution Mapping** — Comma 3 native: 2160x1080 (2:1 ultra-wide). Jetson HDMI: 1920x1080. `BIG=1 SCALE=0.889` maps the UI to 1920x960, filling 89% of the display with correct aspect ratio.

### Phase 4: Controls

**4.1 Core Affinity** (optimized distribution)

| Core | Process | Priority | Role |
|------|---------|----------|------|
| 0 | locationd, calibrationd | 5 | Estimation (moved from core 7) |
| 1 | controlsd | 53 (SCHED_FIFO) | Vehicle control |
| 2 | card | 53 (SCHED_FIFO) | Car interface |
| 3-4 | modeld | 54 (SCHED_FIFO) | CUDA inference |
| 5 | selfdrived | 53 (SCHED_FIFO) | System state |
| 6 | dmonitoringmodeld, UI | 5/51 | CUDA/DLA driver monitoring + UI rendering |
| 7 | plannerd, radard, paramsd, lagd, torqued, dmonitoringd | 5-51 | Planning + low-priority |

Core 7 reduced from 8 to 6 processes. UI pinned to core 6 (was core 0, colliding with locationd).

**4.2 MPC N=24** — `lat_mpc.py:28`, ~25% faster with minimal quality loss.

**4.3 Cache slip_factor** — `vehicle_model.py`, auto-invalidated in `update_params()`.

### Phase 5: System

**5.1 Parked Mode** — `hardware.py` → `set_power_save()`. Parked: 10W, 4 cores. Driving: MAXN 30W, 8 cores.

**5.2 DFS** — `dfs.py`. CPU 1.2-2.27 GHz, GPU 520-1377 MHz, EMC 665-2133 MHz.

**5.3 Fan PID** — `fan_controller.py`. Target 65C driving, 70C parked, 5% hysteresis.

**5.4 tmpfs Logs** — `tmpfs_logger.py`. /dev/shm staging, flush 5s to NVMe, max 512MB.

---

## FILES CREATED

| File | Status | Description |
|------|--------|-------------|
| `msgq_repo/msgq/visionipc/visionbuf_jetson.cc` | Active | VisionBuf with cudaHostRegisterMapped |
| `selfdrive/modeld/runners/tensorrt_runner.py` | Ready | TensorRT runner (activates with install script) |
| `selfdrive/modeld/runners/dla_runner.py` | Ready | NVDLA runner (activates with TensorRT) |
| `selfdrive/modeld/runners/trt_runtime.cpp` | Ready | C++ TRT bridge (compiled by install script) |
| `selfdrive/modeld/transforms/transform.cu` | Active | CUDA warp perspective kernel (sm_72) |
| `selfdrive/modeld/transforms/loadyuv.cu` | Active | CUDA YUV loading kernels |
| `selfdrive/modeld/transforms/transform_cuda.h` | Active | CudaTransform struct |
| `selfdrive/modeld/transforms/loadyuv_cuda.h` | Active | CudaLoadYUVState struct |
| `selfdrive/modeld/models/commonmodel_cuda.h` | Active | CUDA preprocessing pipeline (zero-copy, lazy init) |
| `selfdrive/modeld/models/cuda_frame_bridge.cpp` | Active | C bridge for ctypes (Tensor.from_blob) |
| `tools/jetson/benchmark_modeld.py` | Active | Benchmark: latency, FPS, GPU/CPU/temp |
| `tools/replay/nvdec_decoder.cc` | Active | NVDEC H.264/HEVC decoder |
| `tools/replay/nvdec_decoder.h` | Active | NVDEC header |
| `system/loggerd/encoder/nvenc_encoder.cc` | Active | NVENC H.264 encoder |
| `system/loggerd/encoder/nvenc_encoder.h` | Active | NVENC header |
| `system/hardware/jetson/hardware.py` | Active | Jetson hardware class + DFS + watchdog |
| `system/hardware/jetson/dfs.py` | Active | Dynamic Frequency Scaling |
| `system/hardware/jetson/fan_controller.py` | Active | Fan controller PID |
| `system/hardware/jetson/hugepages.py` | Active | Huge pages 512MB + THP |
| `system/hardware/jetson/tmpfs_logger.py` | Active | tmpfs log staging |
| `system/camerad/cameras/camera_jetson.py` | Ready | Camera CSI V4L2 |
| `system/camerad/jetson_camerad.py` | Ready | Camera daemon |
| `scripts/jetson_replay.sh` | Active | Replay with VNC and DPMS fix |
| `scripts/jetson_stress_test.sh` | Active | 12h stress test with watchdog |
| `scripts/jetson_install_tensorrt.sh` | Active | TensorRT + trt_runtime.so install |

## FILES MODIFIED

| File | Change |
|------|--------|
| `cereal/services.py` | `from __future__ import annotations` (Python 3.8) |
| `tinygrad_repo/examples/openpilot/compile3.py` | FP16 output without cast to FP32 |
| `selfdrive/modeld/modeld.py` | CUDA env vars + zero-copy bridge + core [3,4] + BEAM=2 |
| `selfdrive/modeld/dmonitoringmodeld.py` | CUDA env vars + DLA/TRT fallback + core 6 + CUDA zero-copy |
| `selfdrive/modeld/fill_model_msg.py` | Batched .tolist() (42→~12 calls) |
| `selfdrive/ui/ui.py` | UI core affinity: core 6 on Jetson (was core 0) |
| `system/ui/lib/application.py` | Window centering, VSync default, MSAA off, 60 FPS, undecorated mode |
| `selfdrive/modeld/SConscript` | jarch64 flags + nvcc + cudart + trt_runtime.so |
| `selfdrive/modeld/models/commonmodel.h` | Removed false zero-copy |
| `selfdrive/controls/controlsd.py` | Core 1 on Jetson |
| `selfdrive/car/card.py` | Core 2 on Jetson |
| `selfdrive/selfdrived/selfdrived.py` | Core 5 on Jetson |
| `selfdrive/controls/plannerd.py` | Core 7 on Jetson |
| `selfdrive/controls/radard.py` | Core 7 on Jetson |
| `selfdrive/monitoring/dmonitoringd.py` | Core 7 on Jetson |
| `selfdrive/locationd/locationd.py` | Core 0 on Jetson |
| `selfdrive/locationd/calibrationd.py` | Core 0 on Jetson |
| `selfdrive/locationd/torqued.py` | Core 7 on Jetson |
| `selfdrive/locationd/paramsd.py` | Core 7 on Jetson |
| `selfdrive/locationd/lagd.py` | Core 7 on Jetson |
| `selfdrive/controls/lib/lateral_mpc_lib/lat_mpc.py` | N=32→N=24 |
| `opendbc_repo/opendbc/car/vehicle_model.py` | _slip_factor_cache |
| `system/hardware/hw.py` | tmpfs log root for Jetson |
| `system/loggerd/SConscript` | Link vipc_extra_libs + nvenc_encoder |
| `system/loggerd/encoderd.cc` | `#ifdef __JETSON__` → NvencEncoder |
| `tools/replay/SConscript` | Link vipc_extra_libs + nvdec_decoder |
| `msgq_repo/SConscript` | jarch64 build path + vipc_extra_libs |
| `msgq_repo/msgq/visionipc/visionbuf.h` | Added d_addr for CUDA mapped pointer |
| `msgq_repo/msgq/visionipc/visionbuf_jetson.cc` | cudaHostRegisterMapped + error checks |
| `system/manager/process_config.py` | Registered jetson_camerad |

---

## ROADMAP

### Done (Weeks 1-8)
- [x] FP16, CUDA Graphs, Tensor Cores, VisionBuf pinned
- [x] TensorRT runner, NVDLA runner, Camera CSI V4L2
- [x] NVDEC, NVENC, native CUDA kernels
- [x] Core affinity, MPC N=24, slip_factor cache
- [x] Power management, DFS, fan PID, tmpfs logs, huge pages
- [x] CUDA zero-copy preprocessing driving (13.66ms, 3.4x faster)
- [x] CUDA zero-copy preprocessing dmonitoringmodeld
- [x] Batched .tolist() in fill_model_msg (42→~12 calls)
- [x] UI core affinity fix (core 6, no locationd collision)
- [x] BEAM=2 autotuner for tinygrad kernel optimization
- [x] Benchmark script, stress test 12h
- [x] Pipeline safety: bounds checks, slot reuse, error handling
- [x] Ring buffer single-copy optimization
- [x] loadyuv kernel truncation fix
- [x] TensorRT/DLA Python compatibility fix
- [x] Core affinity redistribution (core 7 overload fix)
- [x] VisionBuf upgraded to cudaHostRegisterMapped
- [x] UI window centering (undecorated + vertical center on 1920x1080)
- [x] VSync enabled by default (smooth 60 FPS, no tearing)
- [x] MSAA disabled on Jetson (saves 4x fragment work)
- [x] 60 FPS default for all platforms (was 20 on tici)
- [x] pandaStates publisher for demo mode (enables model rendering)
- [x] Display resolution mapping (2160x1080 → 1920x960 via SCALE=0.889)
- [x] Model lines rendering verified (path, lane lines, lead indicators)

### Pending
- [ ] Install TensorRT on Jetson (`sudo bash scripts/jetson_install_tensorrt.sh`)
- [ ] Render texture elimination
- [ ] NV12 native format (eliminate conversions)
- [ ] DMA-BUF for VisionBuf camera pipeline
- [ ] Precompiled Python + boot optimization
- [ ] Real camera + vehicle testing
