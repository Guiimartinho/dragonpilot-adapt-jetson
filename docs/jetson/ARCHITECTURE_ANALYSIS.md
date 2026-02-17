# Architecture Analysis: DragonPilot for Jetson Port

## Platform Detection (3 Layers)

### Layer 1: Python (`system/hardware/__init__.py`)
```python
TICI = os.path.isfile('/TICI')  # -> Tici() class
PC = not TICI                    # -> Pc() class
# NEW: JETSON = os.path.isfile('/JETSON') -> Jetson() class
```

### Layer 2: C++ (`system/hardware/hw.h`)
```cpp
#if __TICI__    -> HardwareTici
#else           -> HardwarePC
// NEW: #elif __JETSON__ -> HardwareJetson
```

### Layer 3: Build System (`SConstruct`)
```python
aarch64 + /TICI -> "larch64" with -D__TICI__ -mcpu=cortex-a57
# NEW: aarch64 + /JETSON -> "jarch64" with -D__JETSON__ -mcpu=cortex-a57
```

---

## Hardware Abstraction Hierarchy

```
HardwareBase (system/hardware/base.py) - 25+ abstract methods
  |
  +-- Tici (system/hardware/tici/hardware.py) - 430 lines
  |     - Qualcomm thermal zones (cpu-silver-usr, gpu0-usr, etc.)
  |     - Adreno GPU control (/sys/class/kgsl/kgsl-3d0)
  |     - ION memory, AGNOS bootloader, eSIM, internal Panda
  |
  +-- Pc (system/hardware/pc/hardware.py) - 79 lines
  |     - Stub implementations for development
  |
  +-- Jetson (system/hardware/jetson/hardware.py) - NEW ~100 lines
        - Tegra thermal zones (CPU-therm, GPU-therm)
        - NVIDIA GPU via sysfs (/sys/devices/gpu.0/)
        - INA3221 power monitoring
        - nvpmodel / jetson_clocks power management
```

---

## GPU Compute Pipeline

### Current (Qualcomm Tici)
```
Camera Frame (YUV420)
  -> OpenCL kernels (transform.cl, loadyuv.cl) via Adreno GPU
  -> qcom_tensor_from_opencl_address() [zero-copy QCOM path]
  -> tinygrad DEV=QCOM (ops_qcom.py -> /dev/kgsl-3d0)
  -> Model inference on Adreno 618
```

### Target (Jetson Xavier)
```
Camera Frame (YUV420)
  -> OpenCL kernels (same .cl files) via NVIDIA OpenCL 1.2
  -> buffer_from_cl() -> CPU -> Tensor() [CPU-copy bridge]
  -> tinygrad DEV=CUDA (ops_cuda.py -> CUDA 11.4)
  -> Model inference on Volta GPU (sm_72, 512 CUDA cores)
```

### Future Optimization
```
Camera Frame (YUV420)
  -> CUDA kernels (.cu files replacing .cl)
  -> Direct CUDA memory [zero-copy]
  -> tinygrad DEV=CUDA with CUDA_GRAPH=1
  -> Model inference on Volta GPU with Tensor Cores (FP16)
```

---

## Model Inference Architecture

### Models (ONNX -> tinygrad pickle)
| Model | Input | Output | Purpose |
|-------|-------|--------|---------|
| `driving_vision.onnx` | 512x256 YUV (2 frames) | Hidden state, lane lines, leads | Visual perception |
| `driving_policy.onnx` | Hidden state + desire | Driving plan, trajectory | Decision making |
| `dmonitoring_model.onnx` | 1440x960 Y plane | Driver attention, face pose | Driver monitoring |

### Compilation Flags by Platform
| Platform | Flags | Backend |
|----------|-------|---------|
| Tici (larch64) | `DEV=QCOM FLOAT16=1 NOLOCALS=1 IMAGE=2` | Qualcomm Adreno |
| Jetson (jarch64) | `DEV=CUDA FLOAT16=1` | NVIDIA CUDA |
| PC (x86_64) | `DEV=CPU CPU_LLVM=1` | CPU with LLVM |
| macOS (Darwin) | `DEV=CPU` | CPU |

---

## Camera System

### Current: Qualcomm Spectra ISP (NOT portable)
- `system/camerad/cameras/camera_qcom2.cc` - Proprietary ISP
- `system/camerad/cameras/spectra.cc` - CDM commands, IFE/BPS processing
- V4L_EVENT_CAM_REQ_MGR_EVENT (not standard V4L2)
- Sensors: OX03C10, OS04C10, AR0231 via I2C

### Jetson: Two Options
1. **webcamerad** (immediate): `tools/webcam/camerad.py` - USB cameras via OpenCV
2. **Native camera** (future): `camera_jetson.cc` using libargus or V4L2 for MIPI CSI-2

---

## VisionIPC Memory Management

### Current Paths
| Platform | Allocator | File |
|----------|-----------|------|
| Tici (larch64) | ION (Qualcomm) | `visionbuf_ion.cc` |
| Others | OpenCL generic | `visionbuf_cl.cc` |

Jetson uses `visionbuf_cl.cc` (the generic path). No changes needed.

---

## Communication Stack

### Panda (CAN Interface) - 100% Portable
- USB via libusb-1.0 (`selfdrive/pandad/panda.cc`)
- SPI via `/dev/spidev0.0` (optional, Tici internal)
- CAN protocol is hardware-independent

### Cereal Messaging - 100% Portable
- Cap'n Proto serialization
- msgq or ZMQ transport
- SubMaster/PubMaster pattern

### UI (Raylib) - 100% Portable
- Cross-platform graphics via pyray
- EGL/OpenGL ES (Jetson native support)

---

## Process Manager

### Platform-Specific Processes
| Process | Tici | Jetson | PC |
|---------|------|--------|----|
| camerad (native) | Yes | No (use webcamerad) | No |
| webcamerad | Optional | Yes (USE_WEBCAM=1) | Optional |
| sensord | Yes (LSM6DS3 I2C) | No (no IMU) | No |
| qcomgpsd | Yes | No | No |
| ubloxd | Yes | No | No |
| hardwared | Yes | Yes | Yes |
| modeld | Yes (QCOM) | Yes (CUDA) | Yes (CPU) |
| pandad | Yes (SPI/USB) | Yes (USB only) | Yes (USB) |
| ui | Yes | Yes | Yes |

---

## Qualcomm-Specific Code (NOT portable)

| Component | File | Replacement for Jetson |
|-----------|------|----------------------|
| Spectra ISP | `cameras/camera_qcom2.cc`, `spectra.cc` | webcamerad / camera_jetson.cc |
| ION allocator | `visionbuf_ion.cc` | `visionbuf_cl.cc` (already exists) |
| KGSL GPU | `ops_qcom.py`, `/dev/kgsl-3d0` | `ops_cuda.py` (already exists in tinygrad) |
| QCOM OpenCL ext | `CL_CONTEXT_PRIORITY_HINT_QCOM` | Standard OpenCL (no change needed) |
| AGNOS bootloader | `tici/agnos.py` | Standard NVIDIA U-Boot (skip) |
| eSIM/LPA | `tici/esim.py` | Not needed |
| Internal Panda | GPIO reset via `tici/pins.py` | External Panda USB only |

## Portable Code (~80% of codebase)

- All control logic (`selfdrive/controls/`)
- All car interfaces (`selfdrive/car/`)
- DragonPilot features (ALKA, ACM, AEM, DTSC, RED)
- Messaging system (cereal/msgq)
- Logging (loggerd with ffmpeg encoder)
- UI (Raylib-based)
- Panda communication (USB)
- Model inference logic (just change `DEV` env var)
