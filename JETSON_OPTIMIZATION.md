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
| 5 | TensorRT runner | CODIGO PRONTO | Fallback automatico para tinygrad (python3-libnvinfer nao instalado) |
| 6 | NVDLA runner | CODIGO PRONTO | Depende do TensorRT, fallback para tinygrad |
| 7 | Camera CSI V4L2 | CODIGO PRONTO | Para uso com camera real, inativo durante replay |
| 8 | DPMS disable | IMPLEMENTADO | Previne freeze da UI a cada 10min em headless |
| 9 | VNC otimizado | IMPLEMENTADO | -wait 50 -defer 30 -noxdamage, CPU de 37% para ~2% idle |

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
| 5 | TensorRT para modelos de visao | Inferencia | **2-5x vs tinygrad** | PRONTO (falta instalar) |
| 6 | NVDLA para dmonitoring | Inferencia | **libera GPU** | PRONTO (falta instalar) |
| 7 | NVDEC hardware decode no replay | Video | **190% CPU → 5%** | PENDENTE |
| 8 | NVENC hardware encode no loggerd | Video | **80% CPU → 5%** | PENDENTE |
| 9 | Kernels CUDA nativos (substituir OpenCL) | Inferencia | **12ms saved** | PENDENTE |
| 10 | Fullscreen + VSync na UI | UI | **15% CPU saved** | PENDENTE |
| 11 | Eliminar render texture scaling | UI | **20% GPU saved** | PENDENTE |
| 12 | Modo parked (10W, 4 cores) | Power | **5-6W saved** | PENDENTE |
| 13 | Dynamic Frequency Scaling | Power | **8W saved** | PENDENTE |
| 14 | Lateral MPC N=32→24 | Controles | **25% MPC faster** | PENDENTE |
| 15 | Cache slip_factor no VehicleModel | Controles | **15% faster** | PENDENTE |
| 16 | tmpfs buffer para logs | I/O | **50% latencia** | PENDENTE |
| 17 | Core affinity otimizado (8 cores) | Sistema | **jitter -70%** | PARCIAL |
| 18 | Camera MIPI CSI-2 nativa (V4L2) | Camera | **zero CPU** | PRONTO |
| 19 | Huge Pages para CUDA | Memoria | **5-10% TLB** | PENDENTE |
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

### 1.5 TensorRT Backend (CODIGO PRONTO - falta instalar tensorrt)

**Arquivo**: `selfdrive/modeld/runners/tensorrt_runner.py` (301 linhas)

Runner completo com:
- Compilacao ONNX → TensorRT engine com FP16
- Cache de engines em `/data/trt_engines/`
- Suporte a DLA cores

**Para ativar**:
```bash
sudo apt-get install python3-libnvinfer python3-libnvinfer-dev
pip install pycuda
```
O fallback para tinygrad e automatico via try/except.

### 1.6 NVDLA para Driver Monitoring (CODIGO PRONTO - falta instalar tensorrt)

**Arquivo**: `selfdrive/modeld/runners/dla_runner.py` (127 linhas)

Fallback 3-tier: DLA core 0 → DLA core 1 → GPU TensorRT → tinygrad.

### 1.7 Kernels CUDA Nativos (PENDENTE)

Substituir `selfdrive/modeld/transforms/transform.cl` por kernel CUDA com texture memory cache.

### 1.8 CUDA Memory Pool (PENDENTE)

Pre-alocar memory pool para inference buffers em `ops_cuda.py`.

**Benchmarks Fase 1 (medidos)**:
- Vision model: min=8.68ms avg=8.85ms max=9.52ms (122 kernels, 3 CUDA graphs)
- Model pkl: 50.32MB (vision), 14MB (policy), 9.6MB (dmonitoring)

---

## FASE 2: VIDEO DECODE/ENCODE HARDWARE

### 2.1 NVDEC Hardware Decoder no Replay (PENDENTE)

**Problema**: replay usa FFmpeg software decode.

**Dispositivos disponíveis**: `/dev/video32` (H.264), `/dev/video33` (HEVC)

### 2.2 NVENC Hardware Encoder no Loggerd (PENDENTE)

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

### 4.1 Core Affinity (PARCIAL)

**Implementado**:
- modeld: cores 3-4, prioridade 54
- dmonitoringmodeld: core 6, prioridade 5

**Pendente**: mapping completo dos 8 cores.

### 4.2-4.4 MPC e Cache (PENDENTE)

---

## FASE 5: SISTEMA, POWER E I/O (PENDENTE)

---

## FASE 6: CAMERA NATIVA

### 6.1 Camera CSI V4L2 (CODIGO PRONTO)

**Arquivos**:
- `system/camerad/cameras/camera_jetson.py` (454 linhas) - Driver V4L2 MIPI CSI-2
- `system/camerad/jetson_camerad.py` (147 linhas) - Camera daemon

Suporta deteccao automatica de cameras CSI e fallback para webcam USB.
Inativo durante replay (replay fornece frames via VisionIPC).

---

## BENCHMARKS MEDIDOS (Stress Test)

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
| **Stress test** | 12h sem erros (com DPMS fix) |

---

## ARQUIVOS CRIADOS

| Arquivo | Status | Descricao |
|---------|--------|-----------|
| `msgq_repo/msgq/visionipc/visionbuf_jetson.cc` | ATIVO | VisionBuf com cudaHostRegister pinned memory |
| `selfdrive/modeld/runners/tensorrt_runner.py` | PRONTO | TensorRT runner (falta instalar tensorrt) |
| `selfdrive/modeld/runners/dla_runner.py` | PRONTO | NVDLA runner (falta instalar tensorrt) |
| `system/camerad/cameras/camera_jetson.py` | PRONTO | Camera CSI V4L2 (para camera real) |
| `system/camerad/jetson_camerad.py` | PRONTO | Camera daemon Jetson |
| `scripts/jetson_replay.sh` | ATIVO | Script de replay com VNC e DPMS fix |
| `scripts/jetson_stress_test.sh` | ATIVO | Stress test 12h com watchdog |

## ARQUIVOS MODIFICADOS

| Arquivo | Mudanca |
|---------|---------|
| `cereal/services.py` | `from __future__ import annotations` (Python 3.8) |
| `tinygrad_repo/examples/openpilot/compile3.py` | FP16 output sem cast para FP32 |
| `selfdrive/modeld/modeld.py` | CUDA env vars (TC=1, JIT_BATCH_SIZE=32, FLOAT16=1) + TensorRT fallback |
| `selfdrive/modeld/dmonitoringmodeld.py` | CUDA env vars + DLA/TensorRT fallback |
| `selfdrive/modeld/SConscript` | jarch64 flags: `DEV=CUDA FLOAT16=1 JIT_BATCH_SIZE=32 TC=1` |
| `selfdrive/modeld/models/commonmodel.h` | Removido zero-copy falso, path honesto com clEnqueueReadBuffer |
| `msgq_repo/SConscript` | jarch64 build path + vipc_extra_libs (OpenCL, cudart) |
| `system/loggerd/SConscript` | Link vipc_extra_libs |
| `tools/replay/SConscript` | Link vipc_extra_libs + always build replay for jarch64 |
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
- [x] TensorRT runner (codigo pronto, falta instalar)
- [x] NVDLA runner (codigo pronto, falta instalar)
- [x] Camera CSI V4L2 driver
- [x] Camera daemon com fallback

### Semana 4: Testing + Fixes
- [x] Replay script com VNC otimizado
- [x] DPMS fix (UI 1fps)
- [x] Stress test 12h com watchdog
- [x] Fix USE_TC → TC
- [x] Fix CUDA_OPT removal
- [x] Fix zero-copy falso em commonmodel.h

### Pendente
- [ ] Instalar TensorRT (`python3-libnvinfer`) para ativar runners
- [ ] NVDEC hardware decode no replay
- [ ] NVENC hardware encode no loggerd
- [ ] Kernels CUDA nativos (substituir OpenCL transforms)
- [ ] UI fullscreen + VSync
- [ ] Dynamic Frequency Scaling
- [ ] Modo parked (10W)
- [ ] MPC otimizacoes (N=24, iter_max=8)
- [ ] Cache slip_factor
- [ ] Core affinity completo (8 cores)
- [ ] Teste com camera real e veiculo
