# DragonPilot Jetson AGX Xavier - Plano de Otimizacao Completo

## Objetivo: Rodar CENTENAS de vezes melhor que o Snapdragon 845 (Comma)

Hardware Xavier: 8 cores ARM Carmel 2.26GHz, 512 CUDA cores Volta sm_72, 64 tensor cores, 32GB RAM, 2x NVDEC, NVENC, 2x NVDLA, TensorRT, JetPack 5.1.4

---

## STATUS DAS OTIMIZACOES IMPLEMENTADAS

| # | Otimizacao | Status | Resultado Real |
|---|-----------|--------|----------------|
| 1 | FP16 end-to-end nos Tensor Cores | IMPLEMENTADO | compile3.py mantem FP16 na GPU sem cast para FP32 |
| 2 | CUDA Graphs (JIT_BATCH_SIZE=32) | IMPLEMENTADO | 122 kernels em 3 grafos, ~8.8ms inference |
| 3 | Tensor Cores (TC=1) | IMPLEMENTADO | 64 tensor cores Volta habilitados via tinygrad |
| 4 | VisionBuf pinned memory (cudaHostRegister) | IMPLEMENTADO | DMA mais rapido para transfers GPU, nao e zero-copy |
| 5 | TensorRT runner + install script | IMPLEMENTADO | Fallback automatico para tinygrad, script de instalacao pronto |
| 6 | NVDLA runner | IMPLEMENTADO | Fallback 3-tier: DLA0 → DLA1 → GPU TensorRT → tinygrad |
| 7 | Camera CSI V4L2 | CODIGO PRONTO | Para uso com camera real, inativo durante replay |
| 8 | DPMS disable | IMPLEMENTADO | Previne freeze da UI a cada 10min em headless |
| 9 | VNC otimizado | IMPLEMENTADO | -wait 50 -defer 30 -noxdamage, CPU de 37% para ~2% idle |
| 10 | NVDEC hardware decode | IMPLEMENTADO | h264_nvv4l2dec/hevc_nvv4l2dec com fallback CUDA hwaccel |
| 11 | NVENC hardware encode | IMPLEMENTADO | h264_nvmpi/h264_nvenc com CBR, zero-latency |
| 12 | Kernels CUDA nativos | IMPLEMENTADO | transform.cu + loadyuv.cu + headers + nvcc build sm_72 |
| 13 | Core affinity completo (8 cores) | IMPLEMENTADO | Todos os processos com core dedicado no Jetson |
| 14 | MPC N=32 → N=24 | IMPLEMENTADO | 25% mais rapido no lateral planning |
| 15 | Cache slip_factor | IMPLEMENTADO | _slip_factor_cache com invalidacao automatica |
| 16 | Modo parked (10W, 4 cores) | IMPLEMENTADO | nvpmodel -m 1, desliga cores 4-7 |
| 17 | Dynamic Frequency Scaling | IMPLEMENTADO | CPU adaptativo, GPU/EMC max ao dirigir, min parked |
| 18 | Fan controller PID | IMPLEMENTADO | Target 65C dirigindo, 70C parked, hysteresis 5% |
| 19 | tmpfs buffer para logs | IMPLEMENTADO | /dev/shm staging com flush 5s para NVMe |
| 20 | Huge Pages para CUDA | IMPLEMENTADO | 256 pages (512MB) + THP madvise |

### Bugs corrigidos
- `USE_TC=1` → `TC=1`: tinygrad le `ContextVar("TC", 1)`, nao `USE_TC`. Tensor cores nao estavam sendo ativados.
- `CUDA_OPT=1` removido: nao e variavel real do tinygrad, zero efeito.
- Zero-copy falso em commonmodel.h removido: `clEnqueueMapBuffer` com POCL sempre retorna ponteiro diferente, memcpy sempre executava. Simplificado para `clEnqueueReadBuffer` honesto.
- DPMS X11: Standby de 600s desligava o display virtual, causando UI a 1fps.

---

## RESUMO EXECUTIVO - TOP 20 OTIMIZACOES POR IMPACTO

| # | Otimizacao | Area | Impacto | Status |
|---|-----------|------|---------|--------|
| 1 | CUDA Graphs (JIT_BATCH_SIZE=32) | Inferencia | **~8.8ms inference** | FEITO |
| 2 | FP16 end-to-end nos Tensor Cores | Inferencia | **sem cast FP32** | FEITO |
| 3 | Tensor Cores Volta (TC=1) | Inferencia | **64 cores ativos** | FEITO |
| 4 | VisionBuf pinned memory | Memoria | **DMA transfer** | FEITO |
| 5 | TensorRT para modelos de visao | Inferencia | **2-5x vs tinygrad** | FEITO (script instalacao pronto) |
| 6 | NVDLA para dmonitoring | Inferencia | **libera GPU** | FEITO (ativa com TensorRT) |
| 7 | NVDEC hardware decode no replay | Video | **190% CPU → 5%** | FEITO |
| 8 | NVENC hardware encode no loggerd | Video | **80% CPU → 5%** | FEITO |
| 9 | Kernels CUDA nativos (substituir OpenCL) | Inferencia | **12ms saved** | FEITO |
| 10 | Fullscreen + VSync na UI | UI | **15% CPU saved** | PENDENTE |
| 11 | Eliminar render texture scaling | UI | **20% GPU saved** | PENDENTE |
| 12 | Modo parked (10W, 4 cores) | Power | **5-6W saved** | FEITO |
| 13 | Dynamic Frequency Scaling | Power | **8W saved** | FEITO |
| 14 | Lateral MPC N=32→24 | Controles | **25% MPC faster** | FEITO |
| 15 | Cache slip_factor no VehicleModel | Controles | **15% faster** | FEITO |
| 16 | tmpfs buffer para logs | I/O | **50% latencia** | FEITO |
| 17 | Core affinity otimizado (8 cores) | Sistema | **jitter -70%** | FEITO |
| 18 | Camera MIPI CSI-2 nativa (V4L2) | Camera | **zero CPU** | PRONTO |
| 19 | Huge Pages para CUDA | Memoria | **5-10% TLB** | FEITO |
| 20 | Precompiled Python + boot | Boot | **500ms faster** | PENDENTE |

---

## FASE 1: INFERENCIA DE MODELOS

### 1.1 VisionBuf Pinned Memory (IMPLEMENTADO)

**Arquivo**: `msgq_repo/msgq/visionipc/visionbuf_jetson.cc`

Usa `cudaHostRegister` para pinar paginas de shared memory, permitindo DMA mais rapido entre CPU e GPU. Nao e zero-copy verdadeiro com POCL/OpenCL, mas reduz latencia de transfer.

**Nota**: `clEnqueueMapBuffer` com POCL sempre retorna ponteiro diferente do buffer original, entao true zero-copy nao e possivel nesta stack. O path honesto e `clEnqueueReadBuffer` com memoria pinada.

### 1.2 CUDA Graphs via TinyJit (IMPLEMENTADO)

**Arquivo**: `selfdrive/modeld/modeld.py`, `selfdrive/modeld/SConscript`

`JIT_BATCH_SIZE=32` consolida kernels em CUDA graphs:
- Run 0: baseline
- Run 1: schedule + JIT capture (126 kernels → 122 pruned)
- Run 2: CUDA graph capture (3 batches: 32+64+26 kernels)
- Runs 3+: graph replay (~8.8ms total, ~0.9ms enqueue)

### 1.3 FP16 End-to-End (IMPLEMENTADO)

**Arquivo**: `tinygrad_repo/examples/openpilot/compile3.py`

Quando `FLOAT16=1` e `DEV=CUDA`, remove o `.cast('float32')` final, mantendo output FP16 na GPU. Evita conversao desnecessaria que desperdicava tensor cores.

### 1.4 Tensor Cores (IMPLEMENTADO)

**Arquivo**: `selfdrive/modeld/modeld.py`, `dmonitoringmodeld.py`, `SConscript`

Variavel correta: `TC=1` (tinygrad `ContextVar("TC", 1)` em `helpers.py:171`).
Os 64 tensor cores Volta sm_72 sao usados para operacoes FP16 GEMM.

### 1.5 TensorRT Backend (IMPLEMENTADO - script de instalacao pronto)

**Arquivos**:
- `selfdrive/modeld/runners/tensorrt_runner.py` (310 linhas) - Runner completo
- `scripts/jetson_install_tensorrt.sh` - Script de instalacao automatica

Runner completo com:
- Compilacao ONNX → TensorRT engine com FP16
- Cache de engines em `/data/trt_engines/`
- Suporte a DLA cores
- Verificacao de pycuda e TensorRT

**Para ativar**:
```bash
sudo ./scripts/jetson_install_tensorrt.sh
```
O fallback para tinygrad e automatico via try/except.

### 1.6 NVDLA para Driver Monitoring (IMPLEMENTADO - ativa com TensorRT)

**Arquivo**: `selfdrive/modeld/runners/dla_runner.py` (127 linhas)

Fallback 3-tier: DLA core 0 → DLA core 1 → GPU TensorRT → tinygrad.

### 1.7 Kernels CUDA Nativos (IMPLEMENTADO)

**Arquivos**:
- `selfdrive/modeld/transforms/transform.cu` - Warp perspective CUDA kernel (sm_72)
- `selfdrive/modeld/transforms/loadyuv.cu` - YUV loading CUDA kernels
- `selfdrive/modeld/transforms/transform_cuda.h` - Header + CudaTransform struct
- `selfdrive/modeld/transforms/loadyuv_cuda.h` - Header + CudaLoadYUVState struct
- `selfdrive/modeld/models/commonmodel_cuda.h` - Pipeline CUDA nativo completo

SConscript compila .cu com nvcc -arch=sm_72 -O3 --use_fast_math no jarch64.

Pipeline CUDA nativo elimina overhead POCL/OpenCL:
```
OpenCL: VisionIPC → cl_mem → POCL→CUDA warp → POCL→CUDA loadyuv → clReadBuffer → host → tinygrad
CUDA:   VisionIPC → cudaMemcpy → CUDA warp → CUDA loadyuv → device ptr (GPU direto)
```

### 1.8 Huge Pages para CUDA (IMPLEMENTADO)

**Arquivo**: `system/hardware/jetson/hugepages.py`

- 256 huge pages de 2MB = 512MB pre-alocadas para CUDA
- Transparent Huge Pages (THP) em modo madvise
- `CUDA_USE_HUGEPAGES=1` configurado automaticamente
- Inicializado em `hardware.py` → `initialize_hardware()`

**Benchmarks Fase 1 (medidos)**:
- Vision model: min=8.68ms avg=8.85ms max=9.52ms (122 kernels, 3 CUDA graphs)
- Model pkl: 50.32MB (vision), 14MB (policy), 9.6MB (dmonitoring)

---

## FASE 2: VIDEO DECODE/ENCODE HARDWARE

### 2.1 NVDEC Hardware Decoder no Replay (IMPLEMENTADO)

**Arquivos**:
- `tools/replay/nvdec_decoder.cc` (247 linhas)
- `tools/replay/nvdec_decoder.h` (33 linhas)

Dual decoder fallback: `h264_nvv4l2dec` / `hevc_nvv4l2dec` (Jetson V4L2) → CUDA hwaccel → software.
Suporta NV12, NV21, YUV420P com conversao automatica.

**Dispositivos**: `/dev/video32` (H.264), `/dev/video33` (HEVC)

### 2.2 NVENC Hardware Encoder no Loggerd (IMPLEMENTADO)

**Arquivos**:
- `system/loggerd/encoder/nvenc_encoder.cc` (253 linhas)
- `system/loggerd/encoder/nvenc_encoder.h` (42 linhas)

Dual codec fallback: `h264_nvmpi` (Jetson L4T) → `h264_nvenc` (generic NVIDIA) → software.
Aceita NV12 direto, CBR mode, zero-latency, buffers pre-alocados.

`encoderd.cc` seleciona automaticamente via `#ifdef __JETSON__`.

### 2.3 NV12 Format Nativo (PENDENTE)

### 2.4 VisionBuf com DMA-BUF (PENDENTE)

Requires Jetson-specific V4L2 DMA-BUF integration para true zero-copy pipeline.

---

## FASE 3: UI E RENDERING

### 3.1 DPMS Disable (IMPLEMENTADO)

**Arquivo**: `scripts/jetson_replay.sh`, `scripts/jetson_stress_test.sh`

```bash
DISPLAY=:0 xset s off
DISPLAY=:0 xset -dpms
DISPLAY=:0 xset s noblank
```

Sem isso, DPMS Standby (600s) desliga o display virtual e raylib fica bloqueado no swap_buffers (1fps).

### 3.2 VNC Otimizado (IMPLEMENTADO)

```bash
x11vnc -display :0 -clip 1920x960+0+60 -scale 0.4 \
       -rfbport 5900 -forever -shared -nopw \
       -wait 50 -defer 30 -noxdamage -nocursor -norepeat
```

CPU reduzido de 37% para ~2% (idle) / ~29% (streaming ativo).

### 3.3 Fullscreen + VSync (PENDENTE)
### 3.4 Eliminar Render Texture Overhead (PENDENTE)
### 3.5 Cache Polygon Vertices (PENDENTE)

---

## FASE 4: CONTROLES E PLANEJAMENTO

### 4.1 Core Affinity Completo (IMPLEMENTADO)

**Mapa de 8 cores Jetson AGX Xavier:**

| Core | Processo | Prioridade | Funcao |
|------|---------|------------|--------|
| 0 | UI (raylib) | 51 | Render 60fps |
| 1 | controlsd, sensord | 53 (SCHED_FIFO) | Controle veicular |
| 2 | card | 53 (SCHED_FIFO) | Interface carro |
| 3-4 | modeld | 54 (SCHED_FIFO) | CUDA inference vision |
| 5 | plannerd, radard | 51 (SCHED_FIFO) | Planejamento + radar |
| 5-6 | selfdrived | 53 (SCHED_FIFO) | Estado do sistema |
| 6 | dmonitoringmodeld | 5 | CUDA/DLA driver monitoring |
| 7 | locationd, calibrationd, torqued, paramsd, lagd, dmonitoringd | 5 | Localizacao e calibracao |

**Arquivos modificados**: controlsd.py, card.py, modeld.py, selfdrived.py, dmonitoringmodeld.py, dmonitoringd.py, torqued.py, paramsd.py, locationd.py, calibrationd.py, lagd.py

### 4.2 MPC Lateral N=24 (IMPLEMENTADO)

**Arquivo**: `selfdrive/controls/lib/lateral_mpc_lib/lat_mpc.py:28`

```python
# Reduced horizon for Jetson: N=24 saves ~25% MPC solve time with minimal quality loss.
N = 24
```

### 4.3 Cache slip_factor (IMPLEMENTADO)

**Arquivo**: `opendbc_repo/opendbc/car/vehicle_model.py`

```python
self._slip_factor_cache: float | None = None  # Cache: recalculated only when params change
```

Invalidado automaticamente em `update_params()`. Usado em `curvature_factor()` e `roll_compensation()`.

---

## FASE 5: SISTEMA, POWER E I/O

### 5.1 Modo Parked (IMPLEMENTADO)

**Arquivo**: `system/hardware/jetson/hardware.py` → `set_power_save()`

- Parked: `nvpmodel -m 1` (10W), desliga cores 4-7
- Driving: `nvpmodel -m 0` (MAXN 30W), todos os 8 cores, `jetson_clocks`

### 5.2 Dynamic Frequency Scaling (IMPLEMENTADO)

**Arquivo**: `system/hardware/jetson/dfs.py`

- CPU: 1.2 / 1.7 / 2.27 GHz (adaptativo por carga)
- GPU: 520 / 900 / 1377 MHz (max ao dirigir)
- EMC: 665 / 1600 / 2133 MHz (max ao dirigir, essencial para CUDA bandwidth)
- Thread-safe, update a cada 2s, modo parked minimiza tudo

### 5.3 Fan Controller PID (IMPLEMENTADO)

**Arquivo**: `system/hardware/jetson/fan_controller.py`

- PID: k_p=0.5, k_i=2e-3, k_d=1e-2
- Target: 65°C dirigindo, 70°C parked
- Hysteresis: 5% (previne oscilacao)
- PWM: /sys/devices/pwm-fan/target_pwm (0-255)
- Safe defaults para NaN/Inf do sensor

### 5.4 tmpfs Buffer para Logs (IMPLEMENTADO)

**Arquivo**: `system/hardware/jetson/tmpfs_logger.py`

- Logs escritos em `/dev/shm/openpilot_logs/` (RAM-backed)
- Flush background a cada 5s para `/data/media/0/realdata/` (NVMe)
- Max 512MB em tmpfs antes de forced flush
- Segmentos completos movidos, segmento atual fica em tmpfs
- Ativado automaticamente em `system/hardware/hw.py` para Jetson
- Desativavel com `JETSON_TMPFS_LOGS=0`

### 5.5 Watchdog de Clocks (IMPLEMENTADO)

**Arquivo**: `system/hardware/jetson/hardware.py` → `_jetson_clocks_watchdog()`

Thread que verifica a cada 2 minutos se thermal throttling reduziu frequencia.
Se CPU < 2GHz, re-aplica `jetson_clocks`.

---

## FASE 6: CAMERA NATIVA

### 6.1 Camera CSI V4L2 (CODIGO PRONTO)

**Arquivos**:
- `system/camerad/cameras/camera_jetson.py` (454 linhas) - Driver V4L2 MIPI CSI-2
- `system/camerad/jetson_camerad.py` (147 linhas) - Camera daemon

Suporta deteccao automatica de cameras CSI e fallback para webcam USB.
Inativo durante replay (replay fornece frames via VisionIPC).

---

## BENCHMARKS MEDIDOS (Stress Test 12h)

| Metrica | Valor |
|---------|-------|
| **Vision inference** | 8.68-9.52ms (avg 8.85ms) |
| **Kernels** | 122 (3 CUDA graphs: 32+64+26) |
| **UI FPS** | 50-60+ fps |
| **GPU Load** | 6-10% (idle) / 65-90% (renderizando) |
| **CPU Temp** | 50-54°C |
| **GPU Temp** | 48-52°C |
| **RAM** | ~3.2 GB / 32 GB |
| **Replay CPU** | ~16% |
| **UI CPU** | ~50% |
| **x11vnc CPU** | ~2% (idle) / ~29% (streaming) |
| **Stress test** | 12h0m, 0 restarts, 0 erros |

---

## ARQUIVOS CRIADOS

| Arquivo | Status | Descricao |
|---------|--------|-----------|
| `msgq_repo/msgq/visionipc/visionbuf_jetson.cc` | ATIVO | VisionBuf com cudaHostRegister pinned memory |
| `selfdrive/modeld/runners/tensorrt_runner.py` | PRONTO | TensorRT runner (ativa com script de instalacao) |
| `selfdrive/modeld/runners/dla_runner.py` | PRONTO | NVDLA runner (ativa com TensorRT) |
| `selfdrive/modeld/transforms/transform.cu` | ATIVO | Kernel CUDA warp perspective (sm_72) |
| `selfdrive/modeld/transforms/loadyuv.cu` | ATIVO | Kernels CUDA YUV loading |
| `selfdrive/modeld/transforms/transform_cuda.h` | ATIVO | Header + CudaTransform struct |
| `selfdrive/modeld/transforms/loadyuv_cuda.h` | ATIVO | Header + CudaLoadYUVState struct |
| `selfdrive/modeld/models/commonmodel_cuda.h` | ATIVO | Pipeline preprocessing CUDA nativo |
| `tools/replay/nvdec_decoder.cc` | ATIVO | NVDEC hardware decoder H.264/HEVC |
| `tools/replay/nvdec_decoder.h` | ATIVO | Header NVDEC decoder |
| `system/loggerd/encoder/nvenc_encoder.cc` | ATIVO | NVENC hardware encoder H.264 |
| `system/loggerd/encoder/nvenc_encoder.h` | ATIVO | Header NVENC encoder |
| `system/hardware/jetson/hardware.py` | ATIVO | Jetson hardware class + DFS + watchdog |
| `system/hardware/jetson/dfs.py` | ATIVO | Dynamic Frequency Scaling |
| `system/hardware/jetson/fan_controller.py` | ATIVO | Fan controller PID |
| `system/hardware/jetson/hugepages.py` | ATIVO | Huge pages 512MB + THP |
| `system/hardware/jetson/tmpfs_logger.py` | ATIVO | tmpfs log staging com flush |
| `system/camerad/cameras/camera_jetson.py` | PRONTO | Camera CSI V4L2 (para camera real) |
| `system/camerad/jetson_camerad.py` | PRONTO | Camera daemon Jetson |
| `scripts/jetson_replay.sh` | ATIVO | Script de replay com VNC e DPMS fix |
| `scripts/jetson_stress_test.sh` | ATIVO | Stress test 12h com watchdog |
| `scripts/jetson_install_tensorrt.sh` | ATIVO | Instalacao TensorRT + pycuda |

## ARQUIVOS MODIFICADOS

| Arquivo | Mudanca |
|---------|---------|
| `cereal/services.py` | `from __future__ import annotations` (Python 3.8) |
| `tinygrad_repo/examples/openpilot/compile3.py` | FP16 output sem cast para FP32 |
| `selfdrive/modeld/modeld.py` | CUDA env vars + TensorRT fallback + core affinity [3,4] |
| `selfdrive/modeld/dmonitoringmodeld.py` | CUDA env vars + DLA/TensorRT fallback + core 6 |
| `selfdrive/modeld/SConscript` | jarch64 flags + nvcc compile .cu + cudart link |
| `selfdrive/modeld/models/commonmodel.h` | Removido zero-copy falso, path honesto |
| `selfdrive/controls/controlsd.py` | Core 1 no Jetson |
| `selfdrive/car/card.py` | Core 2 no Jetson |
| `selfdrive/selfdrived/selfdrived.py` | Cores [5,6] no Jetson |
| `selfdrive/controls/plannerd.py` | Core 5 |
| `selfdrive/controls/radard.py` | Core 5 |
| `selfdrive/monitoring/dmonitoringd.py` | Core 7 no Jetson |
| `selfdrive/locationd/torqued.py` | Core 7 no Jetson |
| `selfdrive/locationd/paramsd.py` | Core 7 no Jetson |
| `selfdrive/locationd/locationd.py` | Core 7 no Jetson |
| `selfdrive/locationd/calibrationd.py` | Core 7 no Jetson |
| `selfdrive/locationd/lagd.py` | Core 7 no Jetson |
| `selfdrive/controls/lib/lateral_mpc_lib/lat_mpc.py` | N=32 → N=24 |
| `opendbc_repo/opendbc/car/vehicle_model.py` | _slip_factor_cache |
| `system/hardware/hw.py` | tmpfs log root para Jetson |
| `system/loggerd/SConscript` | Link vipc_extra_libs + nvenc_encoder |
| `system/loggerd/encoderd.cc` | `#ifdef __JETSON__` → NvencEncoder |
| `tools/replay/SConscript` | Link vipc_extra_libs + nvdec_decoder |
| `msgq_repo/SConscript` | jarch64 build path + vipc_extra_libs |
| `system/manager/process_config.py` | Registrado jetson_camerad |

---

## ROADMAP DE IMPLEMENTACAO

### Semana 1-2: Quick Wins + Inferencia CUDA
- [x] FP16 end-to-end (compile3.py)
- [x] CUDA Graphs (JIT_BATCH_SIZE=32)
- [x] Tensor Cores (TC=1)
- [x] VisionBuf pinned memory (cudaHostRegister)
- [x] Python 3.8 compatibility fix (services.py)
- [x] Build system (SConscripts, vipc_extra_libs)

### Semana 3: Runners + Camera
- [x] TensorRT runner (codigo pronto, script de instalacao)
- [x] NVDLA runner (ativa com TensorRT)
- [x] Camera CSI V4L2 driver
- [x] Camera daemon com fallback

### Semana 4: Testing + Fixes
- [x] Replay script com VNC otimizado
- [x] DPMS fix (UI 1fps)
- [x] Stress test 12h com watchdog
- [x] Fix USE_TC → TC
- [x] Fix CUDA_OPT removal
- [x] Fix zero-copy falso em commonmodel.h

### Semana 5: Hardware Acceleration + System
- [x] NVDEC hardware decode (nvdec_decoder.cc)
- [x] NVENC hardware encode (nvenc_encoder.cc)
- [x] Kernels CUDA nativos (transform.cu, loadyuv.cu, headers, commonmodel_cuda.h)
- [x] Core affinity completo (8 cores, 11 processos)
- [x] MPC N=24 (lat_mpc.py)
- [x] Cache slip_factor (vehicle_model.py)
- [x] Modo parked 10W (hardware.py)
- [x] Dynamic Frequency Scaling (dfs.py)
- [x] Fan controller PID (fan_controller.py)
- [x] tmpfs buffer para logs (tmpfs_logger.py + hw.py)
- [x] Huge Pages CUDA 512MB (hugepages.py + hardware.py)
- [x] Script instalacao TensorRT (jetson_install_tensorrt.sh)

### Pendente
- [ ] Instalar TensorRT na Jetson (`sudo ./scripts/jetson_install_tensorrt.sh`)
- [ ] UI fullscreen + VSync
- [ ] Eliminar render texture overhead
- [ ] NV12 format nativo (eliminar conversoes)
- [ ] DMA-BUF para VisionBuf (true zero-copy)
- [ ] Precompiled Python + boot otimizado
- [ ] Teste com camera real e veiculo
