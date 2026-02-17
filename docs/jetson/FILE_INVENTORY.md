# File Inventory: Jetson Port

## Files to MODIFY

| # | File | Phase | Change Description |
|---|------|-------|--------------------|
| 1 | `SConstruct` | 1 | Add `jarch64` arch detection, CUDA include/lib paths, `-D__JETSON__` flag |
| 2 | `system/hardware/__init__.py` | 1 | Add `JETSON` flag, import Jetson class, update `PC` logic |
| 3 | `system/hardware/hw.h` | 1 | Add `#elif __JETSON__` -> `HardwareJetson` |
| 4 | `system/manager/process_config.py` | 1 | Import `JETSON`, adjust `sensord`/`beepd` enabled flags |
| 5 | `system/loggerd/encoderd.cc` | 1 | Add `#elif __JETSON__` -> `FfmpegEncoder` |
| 6 | `selfdrive/modeld/modeld.py` | 3 | CUDA env vars, JIT warmup tracking, core affinity (cores 3-4) |
| 7 | `selfdrive/modeld/dmonitoringmodeld.py` | 3 | CUDA env vars, core affinity (core 6) |
| 8 | `selfdrive/modeld/SConscript` | 3 | Add `jarch64` entry with CUDA compile flags |
| 9 | `system/hardware/hardwared.py` | 6 | Import and instantiate `JetsonFanController`, DFS integration |
| 10 | `selfdrive/modeld/models/commonmodel.h` | Opt | Non-blocking OpenCL map + event-based sync for Jetson |
| 11 | `tinygrad_repo/examples/openpilot/compile3.py` | Opt | Keep FP16 inputs on CUDA for tensor core acceleration |
| 12 | `selfdrive/controls/controlsd.py` | Opt | Core affinity: pin to core 1 (was [1,2]) |
| 13 | `selfdrive/car/card.py` | Opt | Core affinity: pin to core 2 (was [1,2]) |
| 14 | `selfdrive/selfdrived/selfdrived.py` | Opt | Core affinity: pin to cores 5-6 (was [1,2]) |
| 15 | `selfdrive/controls/lib/lateral_mpc_lib/lat_mpc.py` | Opt | Reduce MPC horizon N=32 -> N=24 |
| 16 | `tools/replay/nvdec_decoder.cc` | Opt | Safety: null check, buffer overflow, memory leak, odd width |
| 17 | `system/loggerd/encoder/nvenc_encoder.cc` | Opt | Pre-alloc buffers, dimension validation |
| 18 | `system/loggerd/encoder/nvenc_encoder.h` | Opt | Pre-allocated I420 buffer members |
| 19 | `selfdrive/ui/onroad/cameraview.py` | Opt | EGL image cache LRU eviction |
| 20 | `system/ui/lib/shader_polygon.py` | Opt | Cached triangulated polygon vertices |

## Files to CREATE

| # | File | Phase | Description |
|---|------|-------|-------------|
| 1 | `system/hardware/jetson/__init__.py` | 1 | Empty Python module init |
| 2 | `system/hardware/jetson/hardware.py` | 1 | Jetson hardware class with DFS, thermal, power |
| 3 | `system/hardware/jetson/hardware.h` | 1 | C++ HardwareJetson class (~30 lines) |
| 4 | `system/hardware/jetson/fan_controller.py` | 6 | PWM fan control with PID, hysteresis, NaN safety |
| 5 | `system/hardware/jetson/dfs.py` | 6 | Dynamic Frequency Scaling with thread lock, EMC control |
| 6 | `selfdrive/modeld/transforms/transform.cu` | Opt | CUDA warp perspective kernel (replaces OpenCL) |
| 7 | `selfdrive/modeld/transforms/loadyuv.cu` | Opt | CUDA YUV loading kernels (replaces OpenCL) |
| 8 | `tools/replay/nvdec_decoder.cc` | 5 | NVDEC decoder (V4L2 + CUDA hwaccel) |
| 9 | `tools/replay/nvdec_decoder.h` | 5 | NVDEC decoder header |
| 10 | `system/loggerd/encoder/nvenc_encoder.cc` | 5 | NVENC encoder (h264_nvmpi + fallbacks) |
| 11 | `system/loggerd/encoder/nvenc_encoder.h` | 5 | NVENC encoder header |
| 12 | `system/hardware/jetson/tests/__init__.py` | Test | Test module init |
| 13 | `system/hardware/jetson/tests/test_hardware.py` | Test | Tests for Jetson hardware abstraction |
| 14 | `system/hardware/jetson/tests/test_fan_controller.py` | Test | Tests for fan controller PID/safety |
| 15 | `system/hardware/jetson/tests/test_dfs.py` | Test | Tests for DFS thread safety/inputs |

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
