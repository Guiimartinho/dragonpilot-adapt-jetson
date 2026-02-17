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

## Passo 13: Configurar Performance da Jetson

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

## Passo 14: Conectar Hardware

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

## Passo 15: Rodar o DragonPilot

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
