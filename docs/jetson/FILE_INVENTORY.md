# File Inventory: Jetson Port

## Files to MODIFY

| # | File | Phase | Change Description |
|---|------|-------|--------------------|
| 1 | `SConstruct` | 1 | Add `jarch64` arch detection, CUDA include/lib paths, `-D__JETSON__` flag |
| 2 | `system/hardware/__init__.py` | 1 | Add `JETSON` flag, import Jetson class, update `PC` logic |
| 3 | `system/hardware/hw.h` | 1 | Add `#elif __JETSON__` -> `HardwareJetson` |
| 4 | `system/manager/process_config.py` | 1 | Import `JETSON`, adjust `sensord`/`beepd` enabled flags |
| 5 | `system/loggerd/encoderd.cc` | 1 | Add `#elif __JETSON__` -> `FfmpegEncoder` |
| 6 | `selfdrive/modeld/modeld.py` | 3 | Add `JETSON` import, set `DEV=CUDA` when Jetson |
| 7 | `selfdrive/modeld/dmonitoringmodeld.py` | 3 | Add `JETSON` import, set `DEV=CUDA` when Jetson |
| 8 | `selfdrive/modeld/SConscript` | 3 | Add `jarch64` entry with CUDA compile flags |
| 9 | `system/hardware/hardwared.py` | 6 | Import and instantiate `JetsonFanController` |

## Files to CREATE

| # | File | Phase | Description |
|---|------|-------|-------------|
| 1 | `system/hardware/jetson/__init__.py` | 1 | Empty Python module init |
| 2 | `system/hardware/jetson/hardware.py` | 1 | Jetson hardware class (~100 lines) |
| 3 | `system/hardware/jetson/hardware.h` | 1 | C++ HardwareJetson class (~30 lines) |
| 4 | `system/hardware/jetson/fan_controller.py` | 6 | PWM fan control via sysfs (~30 lines) |

## Symlinks to CREATE

| Source | Target |
|--------|--------|
| `third_party/acados/jarch64` | -> `aarch64` |
| `third_party/libyuv/jarch64` | -> `aarch64` |

## Files that need NO changes (portable)

### Core Control Logic
- `selfdrive/controls/controlsd.py`
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
