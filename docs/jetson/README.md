# DragonPilot Jetson AGX Xavier Port

Port do DragonPilot 0.10.3 para rodar nativamente na NVIDIA Jetson AGX Xavier.

![DragonPilot rodando na Jetson AGX Xavier](assets/ui_running_on_jetson.png)
*UI do DragonPilot rodando na Jetson AGX Xavier via replay demo*

## Status

| Componente | Status | Detalhes |
|------------|--------|----------|
| Build system (`jarch64`) | OK | scons compila sem erros, nvcc para CUDA kernels |
| Hardware abstraction | OK | Classe Jetson com thermal, fan, DFS, hugepages |
| Inferencia CUDA (modeld) | OK | 15.85ms median, 0% frame drops, 2.9x vs OpenCL |
| TensorRT (opcional) | OK | Script de instalacao pronto, 2-3x speedup esperado |
| DLA (dmonitoring) | OK | Fallback chain: DLA0→DLA1→GPU TensorRT→tinygrad |
| CUDA kernels nativos | OK | transform.cu + loadyuv.cu compilados via nvcc |
| CUDA zero-copy preprocessing | OK | Tensor.from_blob, default stream, lazy GPU init |
| UI (raylib/OpenGL) | OK | 60 FPS, fullscreen, VSync, render texture |
| Replay (NVDEC) | OK | Decode HW via V4L2/CUDA hwaccel |
| Encoder (NVENC) | OK | h264_nvmpi / h264_nvenc |
| Core affinity (8 cores) | OK | SCHED_FIFO, distribuicao otimizada |
| Power management | OK | MAXN 30W dirigindo, MODE_10W estacionado |
| Camera USB (webcam) | Pendente | webcamerad pronto, falta testar |
| Panda USB (CAN) | Pendente | pandad pronto, falta conectar hardware |

## Documentacao

| Documento | Descricao |
|-----------|-----------|
| [Setup Guide](SETUP_GUIDE.md) | Guia completo passo-a-passo para compilar e rodar |
| [Operational Guide](../../JETSON_SETUP.md) | Como rodar na Jetson (SSH, VNC, replay, benchmark) |
| [Optimization Details](../../JETSON_OPTIMIZATION.md) | Todas as otimizacoes, benchmarks, bugs, arquivos |
| [Porting Plan](PORTING_PLAN.md) | Plano detalhado do port (Fases 0-7) |
| [Architecture Analysis](ARCHITECTURE_ANALYSIS.md) | Analise de hardware abstraction, build system, GPU pipeline |
| [File Inventory](FILE_INVENTORY.md) | Inventario de todos os arquivos modificados/criados |
| [Hardware Specs](HARDWARE_SPECS.md) | Specs da Jetson AGX Xavier vs comma Tici |

## Arquitetura do Port

```
comma Tici (Snapdragon 845)          Jetson AGX Xavier (Volta)
─────────────────────────────        ─────────────────────────────
Qualcomm Spectra ISP (camera)   →    USB webcam / MIPI CSI-2
Adreno 618 GPU (OpenCL)        →    Volta GPU (CUDA sm_72)
ION memory allocator            →    Standard OpenCL (visionbuf_cl)
tinygrad DEV=QCOM               →    tinygrad DEV=CUDA FLOAT16=1
V4L2 encoder (Qualcomm)        →    NVENC (h264_nvmpi)
Qualcomm NVDEC                  →    NVDEC (V4L2 nvv4l2dec)
AGNOS (custom Android)          →    JetPack 5.x (Ubuntu 20.04)
```

## Quick Start

```bash
cd /data/openpilot
./scripts/jetson_replay.sh --demo
# VNC: conectar em 192.168.3.152:5900 (768x384)
```

Veja o [Operational Guide](../../JETSON_SETUP.md) para instrucoes completas.
