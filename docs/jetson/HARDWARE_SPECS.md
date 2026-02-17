# Hardware Specifications

## Jetson AGX Xavier vs comma Tici (Snapdragon 845)

| Spec | Jetson AGX Xavier | comma Tici (Snapdragon 845) |
|------|-------------------|----------------------------|
| **CPU** | 8x ARM Carmel @ 2.27 GHz | 4x Kryo 385 Gold + 4x Silver |
| **GPU** | 512 CUDA cores (Volta sm_72) | Adreno 630 (OpenCL) |
| **AI Performance** | 32 TOPS (DLA + GPU) | ~6 TOPS (Adreno + Hexagon DSP) |
| **RAM** | 32 GB LPDDR4x | 4 GB LPDDR4x |
| **Tensor Cores** | 64 (FP16/INT8) | None |
| **Camera Interface** | MIPI CSI-2 (up to 6 cameras) | MIPI CSI-2 (3 cameras) |
| **Storage** | NVMe SSD (434 GB free) | 64 GB eMMC |
| **OS** | JetPack 5.1.4 (Ubuntu 20.04) | AGNOS (custom Linux) |
| **CUDA** | 11.4 | N/A (Qualcomm OpenCL) |
| **Python** | 3.8 (needs 3.11+) | 3.11 |
| **Power** | 10-30W (configurable) | ~5W |
| **Connectivity** | Ethernet, WiFi, USB | WiFi, LTE, USB |

## Jetson AGX Xavier Details

- **SoC**: NVIDIA Xavier (Tegra194)
- **GPU Architecture**: Volta (compute capability 7.2)
- **CPU Architecture**: NVIDIA Carmel (ARMv8.2-A, compatible with cortex-a57 flags)
- **Memory Bandwidth**: 137 GB/s
- **Video Encode**: 2x 4K60 | 4x 4K30 (NVENC)
- **Video Decode**: 2x 8K30 | 6x 4K60 (NVDEC)
- **DLA**: 2x NVDLA engines (Deep Learning Accelerator)
- **Vision Accelerator**: 7-Way VLIW

## Thermal Zones (Jetson)

```
/sys/devices/virtual/thermal/thermal_zone*/type:
  - CPU-therm
  - GPU-therm
  - AUX-therm
  - AO-therm
  - Tdiode_tegra
  - PMIC-Die
```

## Power Monitoring (Jetson)

INA3221 3-channel current/voltage monitor:
```
/sys/bus/i2c/drivers/ina3221/1-0040/
  - in_power0_input  (GPU rail)
  - in_power1_input  (CPU rail)
  - in_power2_input  (SoC rail)
```

## Power Modes (nvpmodel)

| Mode | Name | Power | CPU Cores | CPU Freq | GPU Freq |
|------|------|-------|-----------|----------|----------|
| 0 | MAXN | 30W | 8 | 2.27 GHz | 1.38 GHz |
| 1 | MODE_10W | 10W | 4 | 1.2 GHz | 520 MHz |
| 2 | MODE_15W | 15W | 4 | 1.2 GHz | 670 MHz |
| 3 | MODE_30W_ALL | 30W | 8 | 1.2 GHz | 900 MHz |

**Recommended for DragonPilot**: Mode 0 (MAXN) with `jetson_clocks` for maximum performance.

## Current Jetson State

```
IP: 192.168.3.152
User: xavier
JetPack: 5.1.4 (L4T R35.6.2)
CUDA: 11.4
eMMC: 28GB (21GB used / 5.4GB free)
NVMe: 458GB (694MB used / 434GB free)
RAM: 32GB (1.1GB used)
```
