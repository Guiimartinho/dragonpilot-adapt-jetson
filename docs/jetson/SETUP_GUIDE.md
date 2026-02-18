# Guia Completo de Setup: DragonPilot na Jetson AGX Xavier

Guia passo-a-passo **testado e validado** para compilar e rodar o DragonPilot 0.10.3 na Jetson AGX Xavier.

**Hardware**: Jetson AGX Xavier 32GB, JetPack 5.x (R35.6.2, Ubuntu 20.04, CUDA 11.4)

> **Nota**: Todos os passos abaixo foram validados em uma Jetson AGX Xavier real rodando JetPack R35.6.2.

---

## Passo 1: Preparar o NVMe e Diretorio de Trabalho

A eMMC interna da Jetson tem apenas **28GB** e e insuficiente para o build. Use um NVMe SSD.

```bash
# Verificar que o NVMe esta montado (tipicamente em /home)
df -h /home
# Deve mostrar /dev/nvme0n1p1 com centenas de GB livres

# Criar /data como symlink para o NVMe
sudo ln -s /home/$USER /data

# Verificar
ls -la /data
# Deve apontar para /home/xavier (ou seu usuario)
```

---

## Passo 2: Configurar o Sistema

```bash
# Configurar sudo sem senha (necessario para jetson_clocks, nvpmodel, etc.)
echo "$USER ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/$USER

# Configurar limites de real-time scheduling
echo "$USER - rtprio 99" | sudo tee -a /etc/security/limits.conf
echo "$USER - memlock unlimited" | sudo tee -a /etc/security/limits.conf
echo "$USER - nice -20" | sudo tee -a /etc/security/limits.conf

# Adicionar CUDA ao PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Criar diretorio tmp no NVMe (evita encher a eMMC)
mkdir -p /home/$USER/tmp
echo 'export TMPDIR=/home/$USER/tmp' >> ~/.bashrc

# Aplicar
source ~/.bashrc
```

---

## Passo 3: Criar Marker de Plataforma

```bash
sudo touch /JETSON
```

Este arquivo e detectado pelo `system/hardware/__init__.py` e `SConstruct` para ativar o modo Jetson (`jarch64`). Sem ele, o sistema roda em modo PC.

---

## Passo 4: Instalar Dependencias de Sistema (apt)

```bash
sudo apt-get update

# Compilador, ferramentas e todas as bibliotecas
sudo apt-get install -y \
  build-essential git wget curl cmake pkg-config gettext \
  clang \
  gcc-arm-none-eabi \
  libzmq3-dev \
  libcapnp-dev capnproto \
  libusb-1.0-0-dev \
  ffmpeg libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
  opencl-headers ocl-icd-opencl-dev ocl-icd-libopencl1 clinfo \
  pocl-opencl-icd libpocl2 \
  libglfw3-dev libglew-dev libgles2-mesa-dev \
  portaudio19-dev \
  libeigen3-dev \
  libsqlite3-dev \
  libzstd-dev libbz2-dev \
  libffi-dev libssl-dev \
  zlib1g-dev libncurses5-dev libncursesw5-dev \
  libreadline-dev libgdbm-dev liblzma-dev \
  tk-dev uuid-dev \
  libnlopt-dev

# Limpar cache apt (economiza espaco na eMMC)
sudo apt-get clean
```

### Notas Importantes

- **clang**: O SConstruct usa `clang/clang++` como compilador (nao gcc).
- **gcc-arm-none-eabi**: Necessario para compilar o firmware do Panda.
- **pocl-opencl-icd**: A Jetson **NAO** tem driver NVIDIA OpenCL. O POCL (Portable Computing Language) fornece OpenCL via CPU, suficiente para o preprocessing de imagens do VisionIPC.
- **gettext**: Necessario para `msgfmt` (compilacao de traducoes da UI).

---

## Passo 5: Compilar e Instalar Python 3.11

O JetPack 5.x vem com Python 3.8. O DragonPilot exige **Python >= 3.11, < 3.13**.

> **Importante**: O PPA `deadsnakes` NAO tem pacotes para `arm64` no Ubuntu 20.04. E necessario compilar do source.

```bash
cd /tmp
wget https://www.python.org/ftp/python/3.11.11/Python-3.11.11.tar.xz
tar xf Python-3.11.11.tar.xz
cd Python-3.11.11

# Configurar (sem --enable-optimizations para build mais rapido)
./configure --prefix=/usr/local

# Compilar com 8 cores (~5 minutos)
make -j8

# Instalar (altinstall para nao substituir o python3 do sistema)
sudo make altinstall

# Limpar
cd ~ && rm -rf /tmp/Python-3.11*

# Verificar
python3.11 --version
# Deve retornar: Python 3.11.11
```

---

## Passo 6: Verificar CUDA

CUDA ja vem instalado com o JetPack. Verificar:

```bash
nvcc --version
# Deve mostrar: Cuda compilation tools, release 11.4

ls /usr/local/cuda/lib64/libcudart.so
# Deve existir
```

---

## Passo 7: Verificar OpenCL

```bash
clinfo | head -10
# Deve mostrar:
#   Number of platforms: 1
#   Platform Name: Portable Computing Language
#   ...
#   Device Name: pthread-0x004
#   Max compute units: 8
```

> **Nota**: A plataforma OpenCL sera POCL (CPU), nao NVIDIA. Isso e correto - o OpenCL e usado apenas para preprocessing de imagens (loadyuv, transform), enquanto a inferencia dos modelos usa CUDA via tinygrad.

---

## Passo 8: Clonar o Repositorio

```bash
cd /data
git clone --recurse-submodules -b development \
  git@github.com:Guiimartinho/dragonpilot-adapt-jetson.git openpilot
cd openpilot

# Se os submodules nao foram clonados automaticamente:
git submodule update --init --recursive
```

---

## Passo 9: Criar Ambiente Virtual Python e Instalar Dependencias

```bash
cd /data/openpilot

# Criar venv
python3.11 -m venv .venv
source .venv/bin/activate

# Atualizar pip
pip install --upgrade pip setuptools wheel

# Instalar dependencias principais
pip install -e .

# Instalar submodules
pip install -e msgq_repo/
pip install -e opendbc_repo/
pip install -e panda/
pip install -e tinygrad_repo/

# Dependencias adicionais (nao listadas em pyproject.toml mas necessarias)
pip install opencv-python-headless  # webcamerad
pip install Pillow                   # UI emoji/text rendering
pip install jeepney                  # DBus/WiFi manager
pip install dbus-next                # DBus async
```

**Verificar que o venv esta ativo**:
```bash
which python3
# Deve mostrar: /data/openpilot/.venv/bin/python3
python3 --version
# Deve mostrar: Python 3.11.11
```

---

## Passo 10: Compilar o Projeto

```bash
cd /data/openpilot
source .venv/bin/activate

# Garantir CUDA e TMPDIR
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TMPDIR=/home/$USER/tmp

# Compilar com 8 cores
scons -j8
```

O build compila:
- Bibliotecas C++ (cereal, msgq, visionipc, common, rednose, etc.)
- Modulos Cython (params_pyx, visionipc_pyx, commonmodel_pyx, etc.)
- Firmware do Panda (via arm-none-eabi-gcc)
- **Modelos tinygrad para CUDA** (driving_vision, driving_policy, dmonitoring_model)
- Traducoes da UI

### Verificar Artefatos do Build

```bash
# Modelos CUDA compilados
ls -lh selfdrive/modeld/models/*_tinygrad.pkl
# driving_vision_tinygrad.pkl  (~50MB)
# driving_policy_tinygrad.pkl  (~14MB)
# dmonitoring_model_tinygrad.pkl (~10MB)

# Modulos Cython
ls selfdrive/modeld/models/commonmodel_pyx.so
ls msgq_repo/msgq/visionipc/visionipc_pyx.so
ls common/params_pyx.so
```

### Erros Comuns na Compilacao

| Erro | Causa | Solucao |
|------|-------|---------|
| `No space left on device` | eMMC cheia | Mover para NVMe, setar `TMPDIR` |
| `arm-none-eabi-gcc: not found` | Cross-compiler faltando | `sudo apt install gcc-arm-none-eabi` |
| `Error 127` em traducoes | `msgfmt` faltando | `sudo apt install gettext` |
| `clang: not found` | Compilador faltando | `sudo apt install clang` |
| `CL/cl.h not found` | OpenCL headers | `sudo apt install opencl-headers` |
| `jarch64 not found` (acados/libyuv) | Symlinks ausentes | Ja estao no repo (commitados) |

---

## Passo 11: Testar CUDA com tinygrad

```bash
source .venv/bin/activate
DEV=CUDA python3 -c "
from tinygrad import Tensor, Device
print('Devices:', list(Device.get_available_devices()))
print('Default:', Device.DEFAULT)
t = Tensor([1,2,3]).realize()
print('Sum:', t.sum().item())
print('CUDA tinygrad OK!')
"
# Deve imprimir:
#   Devices: ['CUDA', 'CPU']
#   Default: CUDA
#   Sum: 6
#   CUDA tinygrad OK!
```

---

## Passo 12: Testar Hardware Abstraction

```bash
source .venv/bin/activate
python3 -c "
from openpilot.system.hardware import HARDWARE, TICI, JETSON, PC
print(f'TICI={TICI}, JETSON={JETSON}, PC={PC}')
print(f'Device: {HARDWARE.get_device_type()}')
print(f'OS: {HARDWARE.get_os_version()}')
print(f'Serial: {HARDWARE.get_serial()}')
print(f'GPU: {HARDWARE.get_gpu_usage_percent()}%')
thermal = HARDWARE.get_thermal_config()
print(f'CPU zones: {[z.name for z in thermal.cpu]}')
print(f'GPU zones: {[z.name for z in thermal.gpu]}')
"
# Deve mostrar:
#   TICI=False, JETSON=True, PC=False
#   Device: pc
#   OS: # R35 (release)
#   Serial: <chip_uid>
#   GPU: 0.0%
#   CPU zones: ['CPU-therm']
#   GPU zones: ['GPU-therm']
```

---

## Passo 13: Instalar TensorRT (Opcional — 2-3x speedup)

TensorRT permite inferencia FP16 nos tensor cores Volta com 2-3x speedup sobre tinygrad.

```bash
cd /data/openpilot
bash scripts/jetson_install_tensorrt.sh
```

O script:
1. Instala `python3-libnvinfer` e `pycuda`
2. Cria diretorio de cache de engines
3. Configura symlinks TensorRT no venv
4. Verifica instalacao

Apos instalar, o modeld usa automaticamente TensorRT com fallback para tinygrad se indisponivel.

> **Nota**: Se nao instalar TensorRT, o sistema funciona normalmente com tinygrad CUDA (~19ms inferencia total).

---

## Passo 14: Configurar Performance da Jetson

```bash
# Modo MAXN (30W, todos os 8 cores, GPU max)
sudo nvpmodel -m 0

# Travar frequencias no maximo
sudo jetson_clocks

# Verificar
sudo nvpmodel -q
# Deve mostrar: NV Power Mode: MAXN
```

---

## Passo 15: Conectar Hardware

### 14.1 Camera USB (teste inicial)

```bash
# Conectar webcam USB (Logitech C920/C930 ou similar)
ls /dev/video*
# Deve mostrar /dev/video0

# Permissao
sudo usermod -aG video $USER
```

### 14.2 Panda USB (comunicacao veicular)

```bash
# Conectar comma Panda via USB
lsusb | grep -i "comma\|bbaa"

# Permissao USB (sem sudo)
sudo tee /etc/udev/rules.d/11-panda.rules << 'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="bbaa", ATTR{idProduct}=="ddcc", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="bbaa", ATTR{idProduct}=="ddee", MODE="0666"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

## Passo 16: Rodar o DragonPilot

```bash
cd /data/openpilot
source .venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TMPDIR=/home/$USER/tmp

# Com webcam USB
USE_WEBCAM=1 python3 -c "from openpilot.system.manager.manager import main; main()"
```

### Variaveis de Ambiente Uteis

| Variavel | Descricao |
|----------|-----------|
| `USE_WEBCAM=1` | Ativa webcamerad ao inves de camerad nativo |
| `ROAD_CAM=0` | Indice da camera (/dev/video0) |
| `NOSENSOR=1` | Desativa sensord (sem IMU) |
| `PASSIVE=1` | Modo passivo (sem controle do veiculo) |
| `SIMULATION=1` | Modo simulacao |
| `FINGERPRINT=TOYOTA_COROLLA_TSS2` | Forcar fingerprint de veiculo (para teste) |
| `DISPLAY=:0` | Necessario para a UI se rodando via SSH |

### Processos que devem aparecer verdes no manager

```
logmessaged timed ui deleter pandad hardwared tombstoned statsd dashy
```

> **Nota**: A UI (raylib/GLFW) requer um display X11. Se rodando via SSH sem display, a UI vai crashar repetidamente mas os outros processos continuam funcionando normalmente. Use `DISPLAY=:0` se a Jetson tem monitor conectado.

---

## Passo 17: Configurar Display X11 e VNC (Acesso Remoto)

A UI do DragonPilot usa raylib/OpenGL e requer um display X11. Para acessar remotamente via VNC, siga os passos abaixo.

### 16.1 Xorg

A Jetson AGX Xavier ja inicia o Xorg automaticamente no display `:0` via HDMI. Resolucao padrao: **1920x1080**.

```bash
# Verificar que o Xorg esta rodando
export DISPLAY=:0
xdpyinfo | grep dimensions
# Deve mostrar: dimensions: 1920x1080 pixels
```

### 16.2 Instalar x11vnc

```bash
sudo apt-get install -y x11vnc
```

### 16.3 Iniciar o Servidor VNC

Configuracao **exata** para o VNC casar perfeitamente com a janela da UI:

```bash
x11vnc -display :0 \
  -forever \
  -shared \
  -clip 1920x960+0+60 \
  -scale 0.5 \
  -rfbport 5900 \
  -bg \
  -o /tmp/x11vnc.log
```

**Parametros explicados**:

| Parametro | Valor | Descricao |
|-----------|-------|-----------|
| `-display` | `:0` | Display X11 do Xorg |
| `-forever` | - | Manter servidor rodando apos desconexao do cliente |
| `-shared` | - | Permitir multiplos clientes simultaneos |
| `-clip` | `1920x960+0+60` | Capturar apenas a area da UI: 1920x960 pixels, com offset Y=60 (pula a borda preta superior do fullscreen) |
| `-scale` | `0.5` | Escalar para 960x480 no cliente VNC |
| `-rfbport` | `5900` | Porta VNC padrao |
| `-bg` | - | Rodar em background |
| `-o` | `/tmp/x11vnc.log` | Arquivo de log |

> **Por que `-clip 1920x960+0+60`?**
> A UI renderiza em resolucao de design 2160x1080, escalada por 0.889 para 1920x960.
> Em modo fullscreen no display 1920x1080, o conteudo fica centralizado verticalmente
> com ~60px de borda preta em cima e embaixo. O `-clip` recorta exatamente a area visivel
> da UI, eliminando as bordas pretas.

**Resolucao resultante no cliente VNC**: **960x480** (exatamente o conteudo da UI, sem cortes, sem bordas pretas).

### 16.4 Iniciar a UI

```bash
cd /data/openpilot
source .venv/bin/activate
export DISPLAY=:0 BIG=1 SCALE=0.889

# Em foreground (para debug):
python3 -m selfdrive.ui.ui

# Em background (para uso normal):
nohup python3 -m selfdrive.ui.ui > /tmp/ui.log 2>&1 &
```

**Variaveis de ambiente da UI**:

| Variavel | Valor | Descricao |
|----------|-------|-----------|
| `DISPLAY` | `:0` | Display X11 |
| `BIG` | `1` | UI em resolucao grande (2160x1080 design, igual ao comma 3) |
| `SCALE` | `0.889` | Fator de escala: 2160x1080 * 0.889 = 1920x960 (cabe no display 1920x1080) |
| `ENABLE_VSYNC` | `1` | VSync ativado (padrao, 60 FPS suaves) |
| `SHOW_FPS` | `1` | (Opcional) Mostra contador de FPS no canto |

### 16.5 Iniciar o Replay (para teste sem carro)

```bash
cd /data/openpilot
source .venv/bin/activate
export TERM=xterm  # Necessario para ncurses do replay via SSH

./tools/replay/replay --demo
```

> **Nota**: O `TERM=xterm` evita o erro "Error opening terminal: unknown" ao rodar replay via SSH.

### 16.6 Conectar com Cliente VNC

No seu PC, abra um cliente VNC (TigerVNC, RealVNC, TightVNC, etc.) e conecte em:

```
192.168.3.152:5900
```

A janela VNC mostrara a UI do DragonPilot em **960x480**, centralizada e sem cortes.

### 16.7 Script de Inicializacao Completo (VNC + UI + Replay)

```bash
#!/bin/bash
# /data/openpilot/start_vnc_ui.sh
# Inicia VNC, UI e replay para teste/demo

cd /data/openpilot
source .venv/bin/activate
export DISPLAY=:0
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 1. Parar processos anteriores
killall x11vnc python3 2>/dev/null
sleep 1

# 2. Iniciar VNC (clip exato na area da UI)
x11vnc -display :0 -forever -shared \
  -clip 1920x960+0+60 -scale 0.5 \
  -rfbport 5900 -bg -o /tmp/x11vnc.log

# 3. Iniciar UI
export BIG=1 SCALE=0.889
nohup python3 -m selfdrive.ui.ui > /tmp/ui.log 2>&1 &
sleep 3

# 4. Iniciar Replay (opcional)
export TERM=xterm
nohup ./tools/replay/replay --demo > /tmp/replay.log 2>&1 &

echo "VNC rodando em porta 5900 (960x480)"
echo "UI PID: $(pgrep -f 'selfdrive.ui.ui')"
echo "Replay PID: $(pgrep -f 'replay')"
```

```bash
# Tornar executavel
chmod +x /data/openpilot/start_vnc_ui.sh

# Usar
/data/openpilot/start_vnc_ui.sh
```

---

## Troubleshooting

### "No space left on device"
```bash
# Verificar espaco
df -h
# A eMMC (/dev/mmcblk0p1) tem apenas 28GB
# Solucao: mover tudo para NVMe e setar TMPDIR
export TMPDIR=/home/$USER/tmp
sudo apt-get clean
```

### "PermissionError: SCHED_FIFO"
```bash
# Ja tratado no codigo (fallback gracioso)
# Se quiser real-time scheduling nativo:
echo "$USER - rtprio 99" | sudo tee -a /etc/security/limits.conf
# Fazer logout/login para aplicar
```

### "CUDA out of memory"
```bash
# Verificar uso de GPU
sudo tegrastats
# Matar processos usando GPU
sudo fuser -v /dev/nvhost-gpu
```

### "OpenCL platform not found" ou "0 platforms"
```bash
# Instalar POCL
sudo apt install -y pocl-opencl-icd libpocl2
# Verificar
clinfo | head -5
# Deve mostrar: Number of platforms: 1
```

### "Module not found" no Python
```bash
# Verificar que o venv esta ativo
which python3
# Deve mostrar: /data/openpilot/.venv/bin/python3

# Re-instalar pacotes
pip install -e .
pip install -e msgq_repo/ -e opendbc_repo/ -e panda/ -e tinygrad_repo/
```

---

## Referencia Rapida de Comandos

```bash
# Ativar ambiente
source /data/openpilot/.venv/bin/activate
cd /data/openpilot
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TMPDIR=/home/$USER/tmp

# Compilar
scons -j8

# Rodar
USE_WEBCAM=1 python3 -c "from openpilot.system.manager.manager import main; main()"

# Monitorar Jetson
sudo tegrastats  # CPU/GPU/RAM/temp em tempo real

# Performance maxima
sudo nvpmodel -m 0 && sudo jetson_clocks
```

---

## Otimizacoes Aplicadas (20 itens)

O port inclui 20 otimizacoes especificas para o Jetson AGX Xavier:

### Pipeline GPU (Critico)
- **OpenCL async transfer**: `commonmodel.h` usa `clEnqueueMapBuffer` nao-bloqueante com event sync
- **CUDA Graphs via TinyJit**: Grafo de kernels capturado e replayed automaticamente (JIT_BATCH_SIZE=32)
- **FP16 end-to-end**: Tensor cores Volta preservam FP16 nativo (sem conversao FP32)
- **CUDA kernels nativos**: `transform.cu` e `loadyuv.cu` compilados via nvcc -arch=sm_72 (substituem OpenCL)
- **Headers CUDA**: `transform_cuda.h`, `loadyuv_cuda.h`, `commonmodel_cuda.h` — pipeline completo
- **TensorRT** (opcional): `tensorrt_runner.py` com FP16 tensor cores — 2-3x speedup
- **DLA**: `dla_runner.py` para dmonitoring — 2 cores DLA com fallback chain
- **Huge Pages**: 256 x 2MB (512MB) via `hugepages.py` — 5-10% menos TLB misses

### Core Affinity (8 cores completos)
| Core | Processo | Prioridade |
|------|----------|------------|
| 0 | UI (raylib) | prio 51 |
| 1 | controlsd | SCHED_FIFO 53 |
| 2 | card | SCHED_FIFO 53 |
| 3-4 | modeld | SCHED_FIFO 54 |
| 5 | plannerd, radard | SCHED_FIFO 51 |
| 5-6 | selfdrived | SCHED_FIFO 53 |
| 6 | dmonitoringmodeld, camerad | prio 5 |
| 7 | locationd, calibrationd, torqued, paramsd, lagd, dmonitoringd | prio 5 |

### Hardware Acceleration
- **NVDEC**: Decode video via V4L2 nvv4l2dec com fallback CUDA hwaccel (247 lines)
- **NVENC**: Encode via h264_nvmpi com fallback h264_nvenc e software (253 lines)

### Power Management
- **DFS**: Frequencia CPU adaptativa (1.2-2.27 GHz), GPU/EMC sempre max ao dirigir
- **Fan PID**: Controle proporcional com hysteresis (5%), protecao NaN/Inf
- **Power modes**: MAXN (30W) ao dirigir, MODE_10W estacionado (4 cores offline)
- **Watchdog**: Re-aplica jetson_clocks a cada 2min se thermal throttling desfizer

### Controls & MPC
- **MPC N=24**: Horizonte lateral reduzido (era 32) — ~25% mais rapido com perda minima
- **Cache slip_factor**: `vehicle_model.py` evita recalculo de slip_factor a cada ciclo

### I/O
- **tmpfs log buffer**: Logs em /dev/shm, flush NVMe a cada 5s — 50% menos latencia escrita
- **Max 512MB tmpfs**: Forced flush se ultrapassar limite
