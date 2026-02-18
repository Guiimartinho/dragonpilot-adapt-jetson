# File Inventory: Jetson Port

## Files MODIFIED

| # | File | Phase | Change Description | Status |
|---|------|-------|--------------------|--------|
| 1 | `SConstruct` | 1 | Add `jarch64` arch detection, CUDA include/lib paths, `-D__JETSON__` flag | FEITO |
| 2 | `system/hardware/__init__.py` | 1 | Add `JETSON` flag, import Jetson class, update `PC` logic | FEITO |
| 3 | `system/hardware/hw.h` | 1 | Add `#elif __JETSON__` -> `HardwareJetson` | FEITO |
| 4 | `system/hardware/hw.py` | 5 | tmpfs log root para Jetson via JETSON import | FEITO |
| 5 | `system/manager/process_config.py` | 1 | Import `JETSON`, adjust `sensord`/`beepd` enabled flags | FEITO |
| 6 | `system/loggerd/encoderd.cc` | 1 | Add `#elif __JETSON__` -> `FfmpegEncoder` | FEITO |
| 7 | `selfdrive/modeld/modeld.py` | 3 | CUDA env vars, JIT warmup tracking, core affinity (cores 3-4) | FEITO |
| 8 | `selfdrive/modeld/dmonitoringmodeld.py` | 3 | CUDA env vars, core affinity (core 6) | FEITO |
| 9 | `selfdrive/modeld/SConscript` | 3 | CUDA compile flags + nvcc kernel compilation + cudart link | FEITO |
| 10 | `system/hardware/hardwared.py` | 6 | Import and instantiate `JetsonFanController`, DFS integration | FEITO |
| 11 | `selfdrive/modeld/models/commonmodel.h` | Opt | Non-blocking OpenCL map + event-based sync for Jetson | FEITO |
| 12 | `tinygrad_repo/examples/openpilot/compile3.py` | Opt | Keep FP16 inputs on CUDA for tensor core acceleration | FEITO |
| 13 | `selfdrive/controls/controlsd.py` | Opt | Core affinity: pin to core 1 (SCHED_FIFO 53) | FEITO |
| 14 | `selfdrive/car/card.py` | Opt | Core affinity: pin to core 2 (SCHED_FIFO 53) | FEITO |
| 15 | `selfdrive/selfdrived/selfdrived.py` | Opt | Core affinity: pin to cores 5-6 (SCHED_FIFO 53) | FEITO |
| 16 | `selfdrive/controls/lib/lateral_mpc_lib/lat_mpc.py` | Opt | Reduce MPC horizon N=32 -> N=24 | FEITO |
| 17 | `tools/replay/nvdec_decoder.cc` | Opt | Safety: null check, buffer overflow, memory leak, odd width | FEITO |
| 18 | `system/loggerd/encoder/nvenc_encoder.cc` | Opt | Pre-alloc buffers, dimension validation | FEITO |
| 19 | `system/loggerd/encoder/nvenc_encoder.h` | Opt | Pre-allocated I420 buffer members | FEITO |
| 20 | `selfdrive/ui/onroad/cameraview.py` | Opt | EGL image cache LRU eviction | FEITO |
| 21 | `system/ui/lib/shader_polygon.py` | Opt | Cached triangulated polygon vertices | FEITO |
| 22 | `system/hardware/jetson/hardware.py` | 5 | Huge pages + DFS initialization in initialize_hardware() | FEITO |
| 23 | `selfdrive/monitoring/dmonitoringd.py` | 5 | Core affinity: pin to core 7 on Jetson | FEITO |
| 24 | `selfdrive/locationd/torqued.py` | 5 | Core affinity: pin to core 7 on Jetson | FEITO |
| 25 | `selfdrive/locationd/paramsd.py` | 5 | Core affinity: pin to core 7 on Jetson | FEITO |
| 26 | `selfdrive/locationd/locationd.py` | 5 | Core affinity: pin to core 7 on Jetson | FEITO |
| 27 | `selfdrive/locationd/calibrationd.py` | 5 | Core affinity: pin to core 7 on Jetson | FEITO |
| 28 | `selfdrive/locationd/lagd.py` | 5 | Core affinity: pin to core 7 on Jetson | FEITO |
| 29 | `opendbc_repo/opendbc/car/vehicle_model.py` | Opt | Cache _slip_factor para evitar recalculo | FEITO |

## Files CREATED

| # | File | Phase | Description | Status |
|---|------|-------|-------------|--------|
| 1 | `system/hardware/jetson/__init__.py` | 1 | Empty Python module init | FEITO |
| 2 | `system/hardware/jetson/hardware.py` | 1 | Jetson hardware class (249 lines) — thermal, fan, DFS, power, hugepages | FEITO |
| 3 | `system/hardware/jetson/hardware.h` | 1 | C++ HardwareJetson class | FEITO |
| 4 | `system/hardware/jetson/fan_controller.py` | 6 | PWM fan control with PID, hysteresis, NaN safety (71 lines) | FEITO |
| 5 | `system/hardware/jetson/dfs.py` | 6 | Dynamic Frequency Scaling with thread lock, EMC control | FEITO |
| 6 | `system/hardware/jetson/hugepages.py` | 5 | Huge pages: 256 x 2MB (512MB) para CUDA TLB | FEITO |
| 7 | `system/hardware/jetson/tmpfs_logger.py` | 5 | tmpfs log staging: /dev/shm → NVMe flush 5s | FEITO |
| 8 | `selfdrive/modeld/transforms/transform.cu` | Opt | CUDA warp perspective kernel (83 lines) | FEITO |
| 9 | `selfdrive/modeld/transforms/loadyuv.cu` | Opt | CUDA YUV loading kernels (113 lines) | FEITO |
| 10 | `selfdrive/modeld/transforms/transform_cuda.h` | Opt | CUDA transform declarations + CudaTransform struct | FEITO |
| 11 | `selfdrive/modeld/transforms/loadyuv_cuda.h` | Opt | CUDA YUV loading declarations + CudaLoadYUVState struct | FEITO |
| 12 | `selfdrive/modeld/models/commonmodel_cuda.h` | Opt | CUDA preprocessing pipeline (CudaModelFrame classes) | FEITO |
| 13 | `tools/replay/nvdec_decoder.cc` | 5 | NVDEC decoder V4L2 + CUDA hwaccel (247 lines) | FEITO |
| 14 | `tools/replay/nvdec_decoder.h` | 5 | NVDEC decoder header | FEITO |
| 15 | `system/loggerd/encoder/nvenc_encoder.cc` | 5 | NVENC encoder h264_nvmpi + fallbacks (253 lines) | FEITO |
| 16 | `system/loggerd/encoder/nvenc_encoder.h` | 5 | NVENC encoder header | FEITO |
| 17 | `selfdrive/modeld/runners/tensorrt_runner.py` | Opt | TensorRT runner com FP16 (310 lines) | FEITO |
| 18 | `selfdrive/modeld/runners/dla_runner.py` | Opt | DLA runner: DLA0→DLA1→GPU fallback (127 lines) | FEITO |
| 19 | `scripts/jetson_install_tensorrt.sh` | Opt | Script completo de instalacao TensorRT | FEITO |
| 20 | `scripts/jetson_replay.sh` | Test | Script automatizado replay + VNC + UI | FEITO |
| 21 | `scripts/jetson_stress_test.sh` | Test | Stress test com watchdog (12h) | FEITO |
| 22 | `system/hardware/jetson/tests/__init__.py` | Test | Test module init | FEITO |
| 23 | `system/hardware/jetson/tests/test_hardware.py` | Test | Tests for Jetson hardware abstraction | FEITO |
| 24 | `system/hardware/jetson/tests/test_fan_controller.py` | Test | Tests for fan controller PID/safety | FEITO |
| 25 | `system/hardware/jetson/tests/test_dfs.py` | Test | Tests for DFS thread safety/inputs | FEITO |

## Symlinks to CREATE

| Source | Target |
|--------|--------|
| `third_party/acados/jarch64` | -> `aarch64` |
| `third_party/libyuv/jarch64` | -> `aarch64` |

## Files that need NO changes (portable)

### Core Control Logic
- `selfdrive/controls/controlsd.py` (core affinity optimized for Jetson)
- `selfdrive/controls/plannerd.py`
- `selfdrive/controls/radard.py`
- `selfdrive/controls/lib/acm.py` (DragonPilot: Adaptive Coasting)
- `selfdrive/controls/lib/aem.py` (DragonPilot: Adaptive Experimental Mode)
- `selfdrive/controls/lib/dtsc.py` (DragonPilot: Dynamic Turn Speed Control)

### Car Interfaces
- `selfdrive/car/` (all car-specific code)
- `opendbc_repo/` (CAN database definitions)

### Messaging
- `cereal/` (Cap'n Proto schemas)
- `msgq_repo/` (message queue library)

### UI
- `selfdrive/ui/` (Raylib-based, cross-platform)

### Panda
- `selfdrive/pandad/` (USB via libusb-1.0)
- `panda/` (firmware - unchanged)

### DragonPilot Features
- `dragonpilot/settings.py`
- `dragonpilot/selfdrive/` (all DP-specific features)

## Reference Files (templates for new code)

| New File | Based On |
|----------|----------|
| `system/hardware/jetson/hardware.py` | `system/hardware/pc/hardware.py` (79 lines) |
| `system/hardware/jetson/hardware.h` | `system/hardware/pc/hardware.h` (~15 lines) |
| `system/hardware/jetson/fan_controller.py` | `system/hardware/fan_controller.py` (37 lines) |
