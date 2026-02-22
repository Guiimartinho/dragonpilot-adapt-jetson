# DragonPilot - Jetson AGX Xavier Operational Guide

Como rodar o DragonPilot na Jetson. Para detalhes tecnicos e benchmarks, veja [JETSON_OPTIMIZATION.md](JETSON_OPTIMIZATION.md).

> **Performance atual**: modeld 13.66ms median, 20 FPS, 0 errors | dmonitoringmodeld 20.65ms median | GPU 8.6% | CPU 48.9C | UI 60 FPS VSync | Display 1920x960 centered

## Conexao SSH
```
ssh xavier@192.168.3.152
# senha: 14041404
```

## Configuracao de Display e VNC

### 1. Parar GDM e iniciar Xorg limpo (1920x1080 NVIDIA)
```bash
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
cd /data/openpilot && source .venv/bin/activate

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

# modeld (CUDA zero-copy + tinygrad)
# CUDA_MODULE_LOADING=LAZY evita carregar todos os modulos CUDA de uma vez
# JIT_BATCH_SIZE=16 para driving (32 para dmonitoringmodeld)
# BEAM=2 faz autotuning de kernels (~90s na primeira vez, depois cached)
nohup env FLOAT16=1 TC=1 BEAM=2 JIT_BATCH_SIZE=16 CUDA_MODULE_LOADING=LAZY \
  python3 selfdrive/modeld/modeld.py --demo > /tmp/modeld.log 2>&1 &

# deviceState + pandaStates publisher (necessario para modo demo)
# Sem pandaStates, a UI nao entra em modo "started" e as linhas do modelo nao renderizam
nohup python3 -c "
import time, cereal.messaging as messaging
pm = messaging.PubMaster(['deviceState', 'pandaStates'])
while True:
    msg = messaging.new_message('deviceState')
    msg.deviceState.started = True
    msg.deviceState.deviceType = 4  # tici
    msg.deviceState.freeSpacePercent = 80.0
    msg.deviceState.memoryUsagePercent = 20
    msg.deviceState.cpuTempC = [45.0, 45.0, 45.0, 45.0]
    msg.deviceState.gpuTempC = [45.0]
    pm.send('deviceState', msg)
    msg2 = messaging.new_message('pandaStates', 1)
    msg2.pandaStates[0].pandaType = 3
    msg2.pandaStates[0].ignitionLine = True
    msg2.pandaStates[0].ignitionCan = True
    pm.send('pandaStates', msg2)
    time.sleep(0.5)
" > /tmp/devicestate.log 2>&1 &

# UI (centralizada no display 1920x1080)
nohup env DISPLAY=:0 BIG=1 SCALE=0.889 python3 selfdrive/ui/ui.py > /tmp/ui.log 2>&1 &
```

> **Nota**: Para as linhas do modelo (path, lane lines, lead indicators) renderizarem, sao necessarios:
> 1. `deviceState.started = True` — indica que o sistema esta ativo
> 2. `pandaStates` com `ignitionLine = True` — habilita `ui_state.ignition`
> 3. `liveCalibration` — publicado pelo replay, necessario para calibracao da camera
> 4. `modelV2` — publicado pelo modeld, contem dados de path/lanes/leads

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
| `BIG=1` | Ativa 2160x1080 base | Resolucao nativa openpilot (comma 3 = 2160x1080) |
| `SCALE=0.889` | 1920/2160 | Escala para caber em 1920x1080 HDMI |
| `DISPLAY=:0` | Xorg com NVIDIA | GPU acelerada, VSync, 60 FPS |
| `ENABLE_VSYNC=1` | Default | Sync vertical, sem tearing |
| x11vnc `-scale 0.4` | 768x384 | Tamanho compacto no VNC |
| x11vnc `-clip 1920x960+0+60` | Crop da UI | Captura so a area util (pula barras pretas) |

### Resolucao: Comma 3 vs Jetson

| | Comma 3 (original) | Jetson AGX Xavier |
|--|-------------------|-------------------|
| Display fisico | 2160x1080 (2:1 ultra-wide) | 1920x1080 (16:9 HDMI) |
| UI interna | 2160x1080 | 2160x1080 (igual) |
| SCALE | 1.0 (nativo) | 0.889 |
| Janela efetiva | 2160x1080 | 1920x960 |
| Posicionamento | Fullscreen | Centralizado (60px topo + 960px + 60px base) |
| Pixels | 100% | ~89% do original |

## Benchmark modeld
```bash
cd /data/openpilot
# Rodar com replay ativo e modeld rodando
PYTHONPATH=/data/openpilot:/data/openpilot/tinygrad_repo \
  .venv/bin/python3 -m tools.jetson.benchmark_modeld --duration 180 --label "cuda_zerocopy"
```
O benchmark mede: latencia de inferencia, frame drops, uso GPU/CPU/RAM, temperaturas.
Resultados salvos em `/tmp/benchmark_results.txt`.

## Instalar TensorRT (Opcional)

```bash
cd /data/openpilot
sudo bash scripts/jetson_install_tensorrt.sh
```

O script instala `python3-libnvinfer`, compila `trt_runtime.so`, cria cache de engines e configura symlinks no venv.
Apos instalar, o dmonitoringmodeld usa automaticamente TensorRT/DLA com fallback para tinygrad.
