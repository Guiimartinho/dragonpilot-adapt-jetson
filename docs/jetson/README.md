# DragonPilot Jetson AGX Xavier Port

This directory contains all documentation for porting DragonPilot 0.10.3 to the NVIDIA Jetson AGX Xavier platform.

## Documents

| Document | Description |
|----------|-------------|
| [Porting Plan](PORTING_PLAN.md) | Complete phased plan for the port (Phases 0-7) |
| [Architecture Analysis](ARCHITECTURE_ANALYSIS.md) | Hardware abstraction, build system, GPU pipeline analysis |
| [File Inventory](FILE_INVENTORY.md) | All files to modify/create with details |
| [Hardware Specs](HARDWARE_SPECS.md) | Jetson AGX Xavier specs and comparison with comma Tici |

## Quick Overview

**Goal**: Run DragonPilot 0.10.3 natively on NVIDIA Jetson AGX Xavier

**Approach**: Full modern port (Path B) - not based on old xnxpilot 0.8.9

**Key Changes**:
- Platform identity: `jarch64` with `-D__JETSON__` flag
- GPU compute: Qualcomm QCOM/Adreno -> NVIDIA CUDA (Volta sm_72)
- Camera: Qualcomm Spectra ISP -> USB webcam (Phase 4) / MIPI CSI-2 (Phase 4b)
- Memory: ION allocator -> Standard OpenCL `visionbuf_cl.cc`
- Model inference: tinygrad `DEV=QCOM` -> `DEV=CUDA`

**Phases**:
- Phase 0: Environment preparation (Python 3.11, CUDA, toolchain)
- Phase 1: Platform identity and build system
- Phase 2: VisionIPC and OpenCL validation
- Phase 3: Model inference with tinygrad CUDA
- Phase 4: Camera pipeline
- Phase 5: Vehicle communication (Panda USB)
- Phase 6: Performance optimization
- Phase 7: Integration testing
