# DragonPilot - Jetson AGX Xavier Setup

## Conexao SSH
```
ssh xavier@192.168.3.152
# senha: 14041404
```

## Configuracao de Display e VNC

### 1. Parar GDM e iniciar Xorg limpo (1920x1080 NVIDIA)
```bash
# Criar xorg.conf
cat > /tmp/xorg-op.conf << 'EOF'
Section "ServerLayout"
    Identifier     "Layout0"
    Screen      0  "Screen0"
EndSection
Section "Device"
    Identifier     "GPU0"
    Driver         "nvidia"
EndSection
Section "Monitor"
    Identifier     "Monitor0"
EndSection
Section "Screen"
    Identifier     "Screen0"
    Device         "GPU0"
    Monitor        "Monitor0"
    DefaultDepth    24
    SubSection     "Display"
        Depth       24
        Modes       "1920x1080"
    EndSubSection
EndSection
EOF

# Parar GDM e iniciar Xorg
sudo systemctl stop gdm3
sudo nohup Xorg :0 -config /tmp/xorg-op.conf -noreset vt7 > /tmp/xorg.log 2>&1 &
sleep 3
sudo DISPLAY=:0 xhost +local:
```

### 2. Desabilitar DPMS (previne freeze da UI)
```bash
DISPLAY=:0 xset s off
DISPLAY=:0 xset -dpms
DISPLAY=:0 xset s noblank
```
Sem isso, o X11 desliga o "display" apos 10min e a UI cai para 1fps.

### 3. Maximizar performance
```bash
sudo jetson_clocks
```

### 4. Iniciar Replay + UI (modo automatico)
```bash
cd /data/openpilot
./scripts/jetson_replay.sh --demo
```
O script `jetson_replay.sh` faz tudo automaticamente:
- Desabilita DPMS
- Inicia VNC na porta 5900 (scale 0.4 = 768x384)
- Limpa shared memory stale
- Inicia replay com os argumentos passados
- Aguarda VisionIPC ficar pronto
- Inicia a UI

### 4b. Iniciar manualmente (se necessario)
```bash
cd /data/openpilot && source launch_env.sh

# Desabilitar DPMS
DISPLAY=:0 xset s off && DISPLAY=:0 xset -dpms && DISPLAY=:0 xset s noblank

# VNC (scale 0.4, low-CPU mode)
x11vnc -display :0 -clip 1920x960+0+60 -scale 0.4 \
       -rfbport 5900 -forever -shared -nopw \
       -wait 50 -defer 30 -noxdamage -nocursor -norepeat \
       -bg -o /tmp/x11vnc.log

# Limpar shared memory
rm -f /dev/shm/msgq_* /tmp/visionipc_*

# Replay (rota demo, loop automatico)
export TERM=xterm-256color
nohup ./tools/replay/replay --demo > /tmp/replay.log 2>&1 &

# Aguardar VisionIPC
while [ ! -S /tmp/visionipc_camerad ]; do sleep 0.5; done

# UI
nohup env DISPLAY=:0 BIG=1 SCALE=0.889 .venv/bin/python3 -m selfdrive.ui.ui > /tmp/ui.log 2>&1 &
```

### 5. Stress Test (12h)
```bash
cd /data/openpilot
nohup bash scripts/jetson_stress_test.sh 12 > /tmp/stress_console.log 2>&1 &

# Monitorar
cat /tmp/stress_test/events.log     # Eventos e restarts
tail -5 /tmp/stress_test/metrics.csv # Metricas CSV
tail -5 /tmp/stress_test/ui.log     # FPS da UI
```
O stress test inclui watchdog que reinicia processos se morrerem ou a UI travar em 1fps.

**VNC Client**: conectar em `192.168.3.152:5900` (imagem 768x384)

## Parametros da UI
| Parametro | Valor | Descricao |
|-----------|-------|-----------|
| `BIG=1` | Ativa 2160x1080 base | Resolucao nativa openpilot |
| `SCALE=0.889` | 1920/2160 | Escala para caber em 1920x1080 |
| `DISPLAY=:0` | Xorg com NVIDIA | GPU acelerada (60 FPS) |
| x11vnc `-scale 0.4` | 768x384 | Tamanho compacto no VNC |
| x11vnc `-clip 1920x960+0+60` | Crop da UI | Captura so a area util |

## Flags de Compilacao Jetson (tinygrad)
| Variavel | Valor | Efeito |
|----------|-------|--------|
| `DEV=CUDA` | Backend CUDA | Usa GPU Volta |
| `FLOAT16=1` | FP16 end-to-end | Mantem FP16 na GPU sem cast para FP32 |
| `JIT_BATCH_SIZE=32` | CUDA Graphs | Agrupa kernels em grafos (~8.8ms inference) |
| `TC=1` | Tensor Cores | Habilita 64 tensor cores Volta sm_72 |

## Benchmarks Medidos

### modeld (CUDA Zero-Copy, benchmark 3min, 2639 frames)
| Metrica | Valor |
|---------|-------|
| **modeld execution (median)** | **15.85ms** |
| **modeld execution (P95)** | 16.64ms |
| **modeld execution (P99)** | 17.49ms |
| **Frame drops** | **0 (0.00%)** |
| **FPS (avg)** | 14.7 |
| **dmonitoringmodeld (median)** | 20.83ms |
| **Melhoria vs OpenCL** | **2.9x (46ms → 16ms)** |

### Sistema
| Metrica | Valor |
|---------|-------|
| **UI FPS** | 50-60+ fps |
| **GPU Load** | 8.1% median / 61.6% max |
| **CPU Load** | 16.4% median / 35.4% max |
| **GPU Temp** | 50.4°C avg / 51.0°C max |
| **CPU Temp** | 51.9°C avg / 53.0°C max |
| **RAM** | ~5.5 GB / 31 GB |
| **Replay CPU** | ~16% |
| **UI CPU** | ~50% |
| **x11vnc CPU** | ~2% (idle) / ~29% (streaming) |
| **Vision inference only** | ~8.8ms (122 kernels, 3 CUDA graphs) |

## Benchmark modeld
```bash
cd /data/openpilot
# Rodar com replay ativo e modeld rodando
PYTHONPATH=/data/openpilot:/data/openpilot/tinygrad_repo \
  .venv/bin/python3 -m tools.jetson.benchmark_modeld --duration 180 --label "cuda_zerocopy"
```
O benchmark mede: latencia de inferencia, frame drops, uso GPU/CPU/RAM, temperaturas.
Resultados salvos em `/tmp/benchmark_results.txt`.

## Instalar TensorRT (Opcional - 2-3x speedup na inferencia)

```bash
cd /data/openpilot
bash scripts/jetson_install_tensorrt.sh
```

O script instala `python3-libnvinfer`, `pycuda`, cria cache de engines e configura symlinks no venv.
Apos instalar, o modeld usa automaticamente TensorRT com fallback para tinygrad.

## Specs Confirmados
- **GPU**: NVIDIA Volta (GV10B), 512 CUDA cores, 64 tensor cores, driver tegra 35.6.2
- **Direct rendering**: Yes (NVIDIA GLX)
- **Replay**: loop automatico da rota demo, VisionIPC em `/tmp/visionipc_camerad`
- **NVDEC**: h264_nvv4l2dec, hevc_nvv4l2dec — decode hardware ativo em replay
- **NVENC**: h264_nvmpi / h264_nvenc — encode hardware ativo em loggerd
- **TensorRT**: disponivel via `scripts/jetson_install_tensorrt.sh` (fallback automatico para tinygrad)
- **DLA**: 2 cores disponíveis para dmonitoring model (fallback DLA0→DLA1→GPU→tinygrad)
- **CUDA kernels nativos**: transform.cu e loadyuv.cu compilados via nvcc (substituem OpenCL)
- **Huge Pages**: 512MB (256 x 2MB) para CUDA TLB — 5-10% melhoria
- **tmpfs logs**: Logs escritos em /dev/shm, flush para NVMe a cada 5s — 50% menos latencia I/O
- **Core affinity**: 8 cores mapeados — SCHED_FIFO real-time scheduling
