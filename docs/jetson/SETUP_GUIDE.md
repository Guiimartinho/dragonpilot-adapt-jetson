# Guia Completo de Setup: DragonPilot na Jetson AGX Xavier

Guia passo-a-passo para compilar e rodar o DragonPilot 0.10.3 na Jetson AGX Xavier sem erros.

**Hardware**: Jetson AGX Xavier 32GB, JetPack 5.1.4 (Ubuntu 20.04, CUDA 11.4)

---

## Passo 1: Preparar o NVMe e Diretorio de Trabalho

```bash
# Verificar que o NVMe esta montado (deve estar em /home ou /data)
df -h /home

# Criar diretorio de trabalho no NVMe
sudo mkdir -p /data
sudo chown $USER:$USER /data
```

---

## Passo 2: Instalar Python 3.11

O JetPack 5.x vem com Python 3.8. O DragonPilot exige **Python >= 3.11, < 3.13**.

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-dev python3.11-venv python3.11-distutils

# Verificar
python3.11 --version
# Deve retornar: Python 3.11.x
```

---

## Passo 3: Instalar Dependencias de Sistema (apt)

### 3.1 Compilador e Ferramentas Essenciais

```bash
sudo apt install -y \
  clang \
  build-essential \
  ca-certificates \
  curl \
  git \
  git-lfs \
  gettext \
  locales
```

**IMPORTANTE**: O projeto usa `clang/clang++` como compilador (nao gcc). O `SConstruct` tem `CC='clang', CXX='clang++'` hardcoded.

### 3.2 Bibliotecas C/C++ (Build)

```bash
sudo apt install -y \
  libssl-dev \
  libcurl4-openssl-dev \
  libzmq3-dev \
  libcapnp-dev \
  capnproto \
  libusb-1.0-0-dev \
  libzstd-dev \
  libsqlite3-dev \
  libffi-dev \
  libbz2-dev \
  libeigen3-dev \
  libglib2.0-0 \
  libjpeg-dev \
  libncurses5-dev \
  libi2c-dev
```

### 3.3 FFmpeg (Encoder de Video)

```bash
sudo apt install -y \
  ffmpeg \
  libavformat-dev \
  libavcodec-dev \
  libavdevice-dev \
  libavutil-dev \
  libavfilter-dev
```

### 3.4 OpenCL

```bash
sudo apt install -y \
  opencl-headers \
  ocl-icd-libopencl1 \
  ocl-icd-opencl-dev \
  clinfo
```

**Verificar OpenCL**:
```bash
clinfo | head -20
# Deve mostrar: "NVIDIA CUDA" como plataforma, "Xavier" como device
```

O JetPack 5.x ja inclui o driver NVIDIA OpenCL 1.2. O `ocl-icd` e o loader que conecta ao driver NVIDIA.

### 3.5 OpenGL / EGL (UI)

```bash
sudo apt install -y \
  libgles2-mesa-dev \
  libglfw3-dev
```

### 3.6 Audio

```bash
sudo apt install -y portaudio19-dev
```

### 3.7 Panda Firmware (cross-compiler ARM)

```bash
sudo apt install -y gcc-arm-none-eabi
```

### 3.8 Qt5 (Opcional - ferramentas cabana/replay)

```bash
sudo apt install -y \
  qtbase5-dev \
  qtbase5-dev-tools \
  qttools5-dev-tools \
  libqt5charts5-dev \
  libqt5svg5-dev \
  libqt5serialbus5-dev \
  libqt5x11extras5-dev \
  libqt5opengl5-dev
```

---

## Passo 4: Verificar CUDA

CUDA ja vem instalado com o JetPack. Verificar:

```bash
nvcc --version
# Deve mostrar: Cuda compilation tools, release 11.4

ls /usr/local/cuda/lib64/libcudart.so
# Deve existir

# Testar compilacao CUDA
cat > /tmp/test_cuda.cu << 'EOF'
#include <stdio.h>
__global__ void hello() { printf("CUDA OK!\n"); }
int main() { hello<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
EOF
nvcc /tmp/test_cuda.cu -o /tmp/test_cuda && /tmp/test_cuda
# Deve imprimir: CUDA OK!
rm /tmp/test_cuda /tmp/test_cuda.cu
```

---

## Passo 5: Clonar o Repositorio

```bash
cd /data
git clone git@github.com:Guiimartinho/dragonpilot-adapt-jetson.git openpilot
cd openpilot
git checkout development
git submodule update --init --recursive
```

---

## Passo 6: Criar Marker de Plataforma

```bash
sudo touch /JETSON
```

Este arquivo e detectado pelo `system/hardware/__init__.py` e `SConstruct` para ativar o modo Jetson. Sem ele, o sistema roda em modo PC.

---

## Passo 7: Criar Ambiente Virtual Python

```bash
python3.11 -m venv /data/openpilot_venv
source /data/openpilot_venv/bin/activate

# Verificar que esta usando Python 3.11
python --version
# Deve retornar: Python 3.11.x

# Atualizar pip
pip install --upgrade pip setuptools wheel
```

**IMPORTANTE**: Sempre ative o venv antes de trabalhar:
```bash
source /data/openpilot_venv/bin/activate
```

---

## Passo 8: Instalar Dependencias Python

### 8.1 Opcao A: Via pip (mais simples)

```bash
cd /data/openpilot
pip install -e '.[dev]'
```

### 8.2 Opcao B: Via uv (mais rapido, metodo oficial)

```bash
# Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Sincronizar dependencias
cd /data/openpilot
uv sync --frozen --all-extras
source .venv/bin/activate
```

### 8.3 Pacotes que podem falhar no ARM64

Alguns pacotes podem precisar de build manual:

```bash
# Se pycapnp falhar:
pip install pycapnp==2.1.0 --no-binary :all:

# Se numpy falhar:
pip install numpy --no-binary :all:

# Se casadi falhar (MPC solver):
pip install casadi --no-binary :all:

# Se pyaudio falhar:
sudo apt install -y portaudio19-dev
pip install pyaudio
```

### 8.4 Nota sobre pyopencl

O `pyproject.toml` ja exclui `pyopencl` para `aarch64`:
```
"pyopencl; platform_machine != 'aarch64'"
```
Isso e intencional - o projeto usa OpenCL via C++ (nao via Python).

---

## Passo 9: Criar Symlinks para Third-Party

Os prebuilds `acados` e `libyuv` existem para `aarch64` e sao ABI-compativeis. O build system procura pelo nome da arquitetura `jarch64`:

```bash
cd /data/openpilot/third_party/acados && ln -sf aarch64 jarch64
cd /data/openpilot/third_party/libyuv && ln -sf aarch64 jarch64
```

**Verificar**:
```bash
ls -la /data/openpilot/third_party/acados/jarch64/
ls -la /data/openpilot/third_party/libyuv/jarch64/
# Ambos devem apontar para aarch64/
```

---

## Passo 10: Compilar o Projeto

```bash
cd /data/openpilot
source /data/openpilot_venv/bin/activate

# Compilar com todos os cores (Xavier tem 8)
scons -j8
```

### Erros Comuns na Compilacao

| Erro | Causa | Solucao |
|------|-------|---------|
| `arch "aarch64" not in assert` | `/JETSON` nao existe | `sudo touch /JETSON` |
| `clang: not found` | clang nao instalado | `sudo apt install clang` |
| `capnp/kj not found` | capnproto nao instalado | `sudo apt install libcapnp-dev capnproto` |
| `zmq.h not found` | libzmq nao instalado | `sudo apt install libzmq3-dev` |
| `usb.h not found` | libusb nao instalado | `sudo apt install libusb-1.0-0-dev` |
| `No module named 'scons'` | venv nao ativado | `source /data/openpilot_venv/bin/activate` |
| `third_party/acados/jarch64 not found` | symlink nao criado | ver Passo 9 |
| `CL/cl.h not found` | OpenCL headers | `sudo apt install opencl-headers` |
| `cuda.h not found` | CUDA path nao configurado | verificar `/usr/local/cuda/include/` |

---

## Passo 11: Compilar Modelos para CUDA

Apos as modificacoes de codigo (Fase 3 do PORTING_PLAN.md), os modelos ONNX serao compilados para CUDA pickle:

```bash
# Os modelos sao compilados automaticamente pelo scons via selfdrive/modeld/SConscript
# Com as flags: DEV=CUDA FLOAT16=1 JIT_BATCH_SIZE=0

# Para testar CUDA manualmente:
python3 -c "
import os; os.environ['DEV']='CUDA'
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
import numpy as np
t = Tensor(np.random.randn(1,12,128,256).astype(np.float16), dtype=dtypes.float16)
print('CUDA tensor shape:', t.shape, 'device:', t.device)
r = t.sum().realize()
print('Resultado:', r.numpy())
print('CUDA tinygrad OK!')
"
```

Se `DEV=CUDA` falhar, tentar `DEV=CL` como fallback.

---

## Passo 12: Configurar Performance da Jetson

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

## Passo 13: Conectar Hardware

### 13.1 Camera USB (teste inicial)

```bash
# Conectar webcam USB (Logitech C920/C930 ou similar)
ls /dev/video*
# Deve mostrar /dev/video0

# Testar camera
sudo apt install -y v4l-utils
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

### 13.2 Panda USB (comunicacao veicular)

```bash
# Conectar comma Panda via USB
lsusb | grep -i "comma\|bbaa"
# Deve mostrar o dispositivo Panda

# Permissao USB (sem sudo)
sudo tee /etc/udev/rules.d/11-panda.rules << 'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="bbaa", ATTR{idProduct}=="ddcc", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="bbaa", ATTR{idProduct}=="ddee", MODE="0666"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

## Passo 14: Rodar o DragonPilot

```bash
cd /data/openpilot
source /data/openpilot_venv/bin/activate

# Com webcam USB
USE_WEBCAM=1 python -m selfdrive.manager.manager
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

---

## Passo 15: Verificacao End-to-End

Lista de verificacao para confirmar que tudo esta funcionando:

```bash
# 1. Python 3.11
python3.11 --version  # Python 3.11.x

# 2. CUDA
nvcc --version  # CUDA 11.4

# 3. OpenCL
clinfo | grep "Device Name"  # Xavier

# 4. Compilacao
cd /data/openpilot && scons -j8  # sem erros

# 5. Marker
ls /JETSON  # deve existir

# 6. Symlinks
ls -la third_party/acados/jarch64  # -> aarch64
ls -la third_party/libyuv/jarch64  # -> aarch64

# 7. Camera
ls /dev/video0  # deve existir

# 8. Panda (se conectado)
lsusb | grep bbaa  # deve aparecer

# 9. CUDA tinygrad
python3 -c "import os; os.environ['DEV']='CUDA'; from tinygrad.tensor import Tensor; print(Tensor([1,2,3]).sum().realize().numpy())"
# Deve imprimir: 6

# 10. Manager
USE_WEBCAM=1 python -m selfdrive.manager.manager
# Deve iniciar sem crash
```

---

## Troubleshooting

### "No space left on device"
```bash
# Verificar espaco
df -h
# Limpar cache de compilacao
rm -rf /tmp/scons_cache /data/scons_cache
```

### "Permission denied" no /dev/video0
```bash
sudo usermod -aG video $USER
# Fazer logout/login ou:
newgrp video
```

### "CUDA out of memory"
```bash
# Verificar uso de GPU
sudo tegrastats
# Matar processos usando GPU
sudo fuser -v /dev/nvhost-gpu
```

### "OpenCL platform not found"
```bash
# Verificar ICD
ls /etc/OpenCL/vendors/
# Deve ter nvidia.icd

# Se nao existir:
sudo mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" | sudo tee /etc/OpenCL/vendors/nvidia.icd
```

### "Module not found" no Python
```bash
# Verificar que o venv esta ativo
which python
# Deve mostrar /data/openpilot_venv/bin/python

# Re-instalar pacotes
pip install -e '.[dev]'
```

### Build falha em msgq/visionbuf
```bash
# Verificar que OpenCL esta acessivel
pkg-config --libs OpenCL
# Se falhar:
sudo apt install ocl-icd-opencl-dev
```

---

## Referencia Rapida de Comandos

```bash
# Ativar ambiente
source /data/openpilot_venv/bin/activate
cd /data/openpilot

# Compilar
scons -j8

# Rodar
USE_WEBCAM=1 python -m selfdrive.manager.manager

# Replay de rota gravada (teste sem camera)
tools/replay/replay --demo-route

# Ver logs em tempo real
python -m cereal.messaging.bridge

# Monitorar Jetson
sudo tegrastats  # CPU/GPU/RAM/temp em tempo real

# Performance maxima
sudo nvpmodel -m 0 && sudo jetson_clocks
```

---

## Proximos Passos

Apos confirmar que o ambiente esta funcionando, seguir o [Plano de Port](PORTING_PLAN.md) para implementar as modificacoes de codigo (Fases 1-7).
