# Porting Plan: DragonPilot 0.10.3 -> Jetson AGX Xavier

## Context

The goal is to port DragonPilot 0.10.3 (openpilot fork) to run natively on the NVIDIA Jetson AGX Xavier. DragonPilot was developed for comma hardware (Snapdragon 845 / Qualcomm Adreno 618), and the port requires adaptation across 5 layers: platform detection, build system, GPU compute (QCOM -> CUDA), camera pipeline, and hardware abstraction.

**Target hardware**: Jetson AGX Xavier - 8x ARM Carmel cores, 512 CUDA cores (Volta sm_72), 32GB RAM, 32 TOPS, JetPack 5.1.4 (Ubuntu 20.04, CUDA 11.4), NVMe 434GB free.

**Core challenge**: The codebase is tightly coupled to Qualcomm (Spectra ISP, Adreno GPU via KGSL, ION memory allocator, AGNOS OS). The port replaces these dependencies with NVIDIA/Jetson equivalents.

---

## Phase 0: Jetson Environment Preparation

### 0.1 Install Python 3.11+
`pyproject.toml` requires `>= 3.11, < 3.13`. Jetson ships Python 3.8.

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-dev python3.11-venv python3.11-distutils
python3.11 -m ensurepip
python3.11 -m venv /data/openpilot_venv
source /data/openpilot_venv/bin/activate
```

### 0.2 Install Build Toolchain
```bash
sudo apt install clang clang++ libclang-dev
sudo apt install libzmq3-dev libcapnp-dev capnproto
sudo apt install libavformat-dev libavcodec-dev libavutil-dev libswscale-dev
sudo apt install libusb-1.0-0-dev libi2c-dev portaudio19-dev
sudo apt install ocl-icd-opencl-dev
sudo apt install libsqlite3-dev libffi-dev libssl-dev
pip install scons cython numpy pycapnp cffi
```

### 0.3 Verify CUDA and OpenCL
```bash
nvcc --version         # should show CUDA 11.4
clinfo                 # should show Volta GPU (GV10B)
python3.11 -c "import numpy; print(numpy.__version__)"  # >= 2.0
```

### 0.4 Clone DragonPilot
```bash
mkdir -p /data
cd /data
git clone git@github.com:Guiimartinho/dragonpilot-adapt-jetson.git openpilot
cd openpilot
git submodule update --init --recursive
```

### 0.5 Create Marker File
```bash
sudo touch /JETSON
```

**Milestone**: Environment ready - Python 3.11 + CUDA 11.4 + clang working.

---

## Phase 1: Platform Identity and Build System

### 1.1 SConstruct - Add `jarch64` architecture

**File**: `SConstruct` (lines 28-40, 96-125, 171, 210-211)

Add Jetson detection after line 33:
```python
elif arch == "aarch64" and os.path.isfile('/JETSON'):
  arch = "jarch64"
```

Add `jarch64` to the assert:
```python
assert arch in ["larch64", "jarch64", "aarch64", "x86_64", "Darwin"]
```

Add Jetson-specific flags after line 106:
```python
elif arch == "jarch64":
  env.Append(CPPPATH=[
    "#third_party/opencl/include",
    "/usr/local/cuda/include",
  ])
  env.Append(LIBPATH=[
    "/usr/local/lib",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/local/cuda/lib64",
  ])
  arch_flags = ["-D__JETSON__", "-mcpu=cortex-a57"]
  env.Append(CCFLAGS=arch_flags)
  env.Append(CXXFLAGS=arch_flags)
```

Update cache dir (line 171):
```python
cache_dir = '/data/scons_cache' if arch in ("larch64", "jarch64") else '/tmp/scons_cache'
```

### 1.2 Symlinks for third_party prebuilds

The `third_party/acados/aarch64/` and `third_party/libyuv/aarch64/` directories already exist and are ABI-compatible (both aarch64).

```bash
cd third_party/acados && ln -s aarch64 jarch64
cd third_party/libyuv && ln -s aarch64 jarch64
```

### 1.3 C++ Hardware Abstraction

**File**: `system/hardware/hw.h`

```cpp
#if __TICI__
#include "system/hardware/tici/hardware.h"
#define Hardware HardwareTici
#elif __JETSON__
#include "system/hardware/jetson/hardware.h"
#define Hardware HardwareJetson
#else
#include "system/hardware/pc/hardware.h"
#define Hardware HardwarePC
#endif
```

**New file**: `system/hardware/jetson/hardware.h` (based on `pc/hardware.h`)

### 1.4 Python Hardware Abstraction

**File**: `system/hardware/__init__.py`

```python
TICI = os.path.isfile('/TICI')
JETSON = os.path.isfile('/JETSON')
AGNOS = os.path.isfile('/AGNOS')
PC = not TICI and not JETSON

if TICI:
  HARDWARE = cast(HardwareBase, Tici())
elif JETSON:
  from openpilot.system.hardware.jetson.hardware import Jetson
  HARDWARE = cast(HardwareBase, Jetson())
else:
  HARDWARE = cast(HardwareBase, Pc())
```

**New files**:
- `system/hardware/jetson/__init__.py`
- `system/hardware/jetson/hardware.py` (~100 lines, based on `pc/hardware.py`)

Jetson hardware class methods:

| Method | Jetson Implementation |
|--------|----------------------|
| `get_device_type()` | `return "jetson"` |
| `get_os_version()` | read `/etc/nv_tegra_release` |
| `get_serial()` | read `/sys/module/tegra_fuse/parameters/tegra_chip_uid` |
| `reboot()` | `subprocess.call(["sudo", "reboot"])` |
| `shutdown()` | `subprocess.call(["sudo", "poweroff"])` |
| `get_thermal_config()` | Map zones: CPU-therm, GPU-therm, Tdiode_tegra |
| `get_gpu_usage_percent()` | read `/sys/devices/gpu.0/load` (divide by 10) |
| `get_current_power_draw()` | read INA3221 at `/sys/bus/i2c/drivers/ina3221/` |
| `set_power_save(on)` | `nvpmodel -m 2` (15W) vs `nvpmodel -m 0` (MAXN) |
| `initialize_hardware()` | `jetson_clocks` + configure fan |
| `get_network_type()` | Detect WiFi/Ethernet via `/sys/class/net/` |
| `get_modem_temperatures()` | `return []` (no modem) |

### 1.5 Process Config

**File**: `system/manager/process_config.py`

- Add `JETSON` import
- `sensord`: change to `enabled=TICI`
- `beepd`: change to `enabled=(TICI or JETSON) and LITE`
- `qcomgpsd/ubloxd/pigeond`: keep `enabled=TICI`

### 1.6 Encoderd

**File**: `system/loggerd/encoderd.cc`

```cpp
#ifdef __TICI__
#define Encoder V4LEncoder
#elif __JETSON__
#define Encoder FfmpegEncoder
#else
#define Encoder FfmpegEncoder
#endif
```

**Milestone**: `scons -j8` compiles without errors on the Jetson.

---

## Phase 2: VisionIPC and OpenCL on Jetson

### 2.1 VisionBuf Memory Allocator

The code already has two paths in `msgq_repo/SConscript`:
- `larch64`: uses `visionbuf_ion.cc` (Qualcomm ION - NOT portable)
- Others: uses `visionbuf_cl.cc` (generic OpenCL - PORTABLE)

For `jarch64`, it falls into the generic path automatically. **No changes needed.**

### 2.2 Validate OpenCL

NVIDIA provides OpenCL 1.2 on JetPack. The kernels in `selfdrive/modeld/transforms/transform.cl` and `loadyuv.cl` use basic OpenCL 1.2 and should work without modification.

**Milestone**: OpenCL works, VisionBuf allocates memory, transform kernels compile.

---

## Phase 3: Model Inference with Tinygrad CUDA

### 3.1 Device Selection

**Files**: `selfdrive/modeld/modeld.py` and `dmonitoringmodeld.py`

```python
from openpilot.system.hardware import TICI, JETSON
if TICI:
  os.environ['DEV'] = 'QCOM'
elif JETSON:
  os.environ['DEV'] = 'CUDA'
else:
  os.environ['DEV'] = 'CPU'
```

### 3.2 Compile Model Pickles for CUDA

**File**: `selfdrive/modeld/SConscript`

```python
flags = {
    'larch64': 'DEV=QCOM FLOAT16=1 NOLOCALS=1 IMAGE=2 JIT_BATCH_SIZE=0',
    'jarch64': 'DEV=CUDA FLOAT16=1 JIT_BATCH_SIZE=0',
    'Darwin': f'DEV=CPU HOME={os.path.expanduser("~")}',
}.get(arch, 'DEV=CPU CPU_LLVM=1')
```

### 3.3 OpenCL -> CUDA Bridge

The non-TICI path in `modeld.py` (lines 201-204) already works for Jetson:
```python
else:
    frame_input = self.frames[key].buffer_from_cl(imgs_cl[key]).reshape(...)
    self.vision_inputs[key] = Tensor(frame_input, dtype=dtypes.uint8).realize()
```

With `DEV=CUDA`, the `Tensor()` constructor automatically places data on the CUDA device.

**Fallback**: If `DEV=CUDA` fails, try `DEV=CL` (OpenCL backend).

**Milestone**: Models compile for CUDA. Inference runs on Volta GPU. Target: < 50ms per frame.

---

## Phase 4: Camera Pipeline

### 4.1 Initial: webcamerad (USB camera)

**No code changes needed.** Already supported:

```bash
export USE_WEBCAM=1
export ROAD_CAM=0  # /dev/video0
```

### 4.2 Future: Native MIPI CSI-2 camera (Phase 4b)

For production with MIPI cameras (IMX477, IMX390):
- Create `system/camerad/cameras/camera_jetson.cc` using libargus or V4L2
- Reference: [xnxpilot project](https://github.com/eFiniLan/xnxpilot)

**Milestone**: Camera frames flow through VisionIPC. modeld runs inference in real-time.

---

## Phase 5: Vehicle Communication (Panda)

### 5.1 Panda via USB

**100% portable.** Uses libusb-1.0. No code changes needed.

### 5.2 Sensors (IMU/GPS)

Without sensord initially. Options:
- USB IMU (Bosch BMI088)
- USB GPS (gpsd-compatible)
- Panda's built-in accelerometer

**Milestone**: CAN messages flow from Panda to controlsd.

---

## Phase 6: Performance Optimization

### 6.1 Fan Controller

**New file**: `system/hardware/jetson/fan_controller.py`
- Control fan via `/sys/devices/pwm-fan/target_pwm` (0-255)

### 6.2 Power Mode
```bash
sudo nvpmodel -m 0   # MAXN (30W)
sudo jetson_clocks    # lock max frequencies
```

### 6.3 Future: Native CUDA kernels

Replace `transform.cl` and `loadyuv.cl` with CUDA kernels to eliminate OpenCL -> CPU -> CUDA overhead.

### 6.4 Future: TensorRT

If tinygrad CUDA performance is insufficient, export ONNX -> TensorRT engines.

---

## Phase 7: Integration Testing

### 7.1 Without Hardware
```bash
python -m selfdrive.manager.manager
tools/replay/replay --demo-route
```

### 7.2 With Hardware
1. USB webcam -> verify frames
2. Panda USB -> verify CAN messages
3. Vehicle engagement -> real-world test

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Python 3.11 packages fail on ARM64 | Medium | High | Build from source, conda-forge |
| clang `-mcpu=carmel` unsupported | High | Low | Use `-mcpu=cortex-a57` |
| Tinygrad CUDA fails on sm_72 | Low | High | Fallback `DEV=CL` or TensorRT |
| OpenCL kernels incompatible | Low | Medium | NVIDIA OpenCL well-tested |
| Inference > 50ms | Medium | High | FLOAT16 + CUDA_GRAPH or TensorRT |
| USB camera latency | Medium | Medium | Migrate to MIPI CSI-2 |
