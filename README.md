![](dragonpilot/selfdrive/assets/dragonpilot.png)

[Leia em Portugues](#portuguese) | [Read in English](#english) | [DragonPilot Original (Chinese)](https://github.com/dragonpilot-community/dragonpilot) | [DragonPilot Original (English)](README_EN.md)

---

<a name="english"></a>

# DragonPilot - Jetson AGX Xavier Adaptation

![Platform](https://img.shields.io/badge/Platform-Jetson_AGX_Xavier-76B900?logo=nvidia&logoColor=white)
![JetPack](https://img.shields.io/badge/JetPack-5.x_(R35)-76B900?logo=nvidia)
![CUDA](https://img.shields.io/badge/CUDA-11.4-76B900?logo=nvidia)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![openpilot](https://img.shields.io/badge/openpilot-0.10.3-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)
![tinygrad](https://img.shields.io/badge/Inference-tinygrad_CUDA-orange)

**Bringing DragonPilot's autonomous driving capabilities to NVIDIA Jetson hardware.**

### Running on Jetson AGX Xavier

![DragonPilot running on Jetson AGX Xavier](docs/jetson/assets/ui_running_on_jetson.png)
*DragonPilot UI with real-time CUDA inference on Jetson AGX Xavier — modeld 15.85ms median (CUDA zero-copy, 2.9x faster than OpenCL)*

## About This Project

This is an adaptation of [DragonPilot](https://github.com/dragonpilot-community/dragonpilot) 0.10.3 to run on the **NVIDIA Jetson AGX Xavier** platform. DragonPilot is a community fork of [openpilot](https://github.com/commaai/openpilot), originally designed for comma.ai hardware (Snapdragon 845).

This project preserves the full DragonPilot history and features while adding native Jetson support.

## Why Jetson?

| Feature | comma 3X (Tici) | Jetson AGX Xavier |
|---------|----------------|-------------------|
| AI Performance | ~6 TOPS | **32 TOPS** |
| RAM | 4 GB | **32 GB** |
| GPU | Adreno 630 (OpenCL) | **512 CUDA + 64 Tensor Cores (Volta)** |
| Tensor Cores | None | **64 (FP16/INT8)** |
| Storage | 64 GB eMMC | **NVMe SSD (expandable)** |

The Jetson AGX Xavier offers **5x the AI performance** with **8x the RAM**, making it suitable for more advanced models and future development.

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Platform Detection (`/JETSON` + `jarch64`) | Working | SConstruct, hardware init |
| Hardware Abstraction (thermal, power, fan) | Working | Full Jetson class with INA3221 power, PWM fan, thermal zones |
| CUDA Model Compilation (tinygrad) | Working | driving_vision, driving_policy, dmonitoring_model |
| CUDA Zero-Copy Preprocessing | Working | 15.85ms median, 0% frame drops, Tensor.from_blob |
| scons Build (`-j8`) | Working | All C++/Cython targets compile cleanly |
| Core Processes (hardwared, pandad, loggerd...) | Working | All green on manager status line |
| UI (raylib) | Requires Display | Needs X11 DISPLAY or headless mode |
| Camera (webcamerad) | Ready | Requires USB webcam connected |

## Key Adaptations

- **Platform Identity**: New `jarch64` architecture with `-D__JETSON__` compile flag
- **GPU Compute**: Qualcomm QCOM/Adreno -> NVIDIA CUDA (Volta sm_72)
- **Model Inference**: tinygrad `DEV=QCOM` -> `DEV=CUDA` with FP16 support
- **Camera**: Qualcomm Spectra ISP -> USB webcam / MIPI CSI-2
- **Memory**: ION allocator -> Standard OpenCL `visionbuf_cl.cc` (POCL on CPU)
- **Hardware Class**: New `Jetson` class with thermal, power, and fan management
- **Real-Time Scheduling**: Graceful fallback when SCHED_FIFO not permitted

## Preserved Features

All DragonPilot features are preserved:
- ALKA (Always-on Lane Keeping Assist)
- ACM (Adaptive Coasting Mode)
- AEM (Adaptive Experimental Mode)
- DTSC (Dynamic Turn Speed Control)
- RED (Road Edge Detection)
- Full multilingual support (Chinese, English, Portuguese)
- 300+ supported vehicles

## Documentation

- [Setup Guide](docs/jetson/SETUP_GUIDE.md) - Complete step-by-step guide to build and run
- [Porting Plan](docs/jetson/PORTING_PLAN.md) - Complete phased implementation plan
- [Architecture Analysis](docs/jetson/ARCHITECTURE_ANALYSIS.md) - Platform analysis and GPU pipeline
- [File Inventory](docs/jetson/FILE_INVENTORY.md) - All files to modify/create
- [Hardware Specs](docs/jetson/HARDWARE_SPECS.md) - Jetson AGX Xavier specifications

## Quick Start (on Jetson)

See the full [Setup Guide](docs/jetson/SETUP_GUIDE.md) for detailed instructions.

```bash
# 1. Create platform marker and /data directory
sudo touch /JETSON
sudo mkdir -p /data && sudo chown $USER:$USER /data
# If eMMC is small, symlink /data to NVMe:
# ln -s /home/$USER /data

# 2. Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 3. Install system dependencies
sudo apt-get install -y build-essential git clang cmake pkg-config gettext \
  gcc-arm-none-eabi libzmq3-dev libcapnp-dev capnproto libusb-1.0-0-dev \
  ffmpeg libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
  opencl-headers ocl-icd-opencl-dev ocl-icd-libopencl1 pocl-opencl-icd \
  libglfw3-dev libglew-dev libgles2-mesa-dev portaudio19-dev \
  libeigen3-dev libsqlite3-dev libzstd-dev libbz2-dev libffi-dev libssl-dev \
  zlib1g-dev libreadline-dev liblzma-dev tk-dev

# 4. Build Python 3.11 from source (deadsnakes has no arm64 packages)
cd /tmp
wget https://www.python.org/ftp/python/3.11.11/Python-3.11.11.tar.xz
tar xf Python-3.11.11.tar.xz && cd Python-3.11.11
./configure --prefix=/usr/local && make -j8 && sudo make altinstall
cd ~ && rm -rf /tmp/Python-3.11*

# 5. Clone and setup
cd /data
git clone --recurse-submodules -b development \
  git@github.com:Guiimartinho/dragonpilot-adapt-jetson.git openpilot
cd openpilot
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e . && pip install -e msgq_repo/ -e opendbc_repo/ -e panda/ -e tinygrad_repo/
pip install opencv-python-headless Pillow jeepney dbus-next

# 6. Build
export TMPDIR=/home/$USER/tmp && mkdir -p $TMPDIR
scons -j8

# 7. Run (with USB webcam)
USE_WEBCAM=1 python3 -c "from openpilot.system.manager.manager import main; main()"
```

## Hardware Requirements

- **NVIDIA Jetson AGX Xavier** (32GB recommended)
- JetPack 5.x (R35, CUDA 11.4+)
- NVMe SSD (recommended - eMMC 28GB is too small for build)
- USB webcam (for initial testing) or MIPI CSI-2 camera
- [comma Panda](https://comma.ai/shop/panda) (for vehicle communication)
- Car harness for your supported vehicle

## Project Structure

```
dragonpilot-adapt-jetson/
  docs/jetson/          <- Jetson-specific documentation
  system/hardware/
    jetson/             <- Jetson hardware abstraction
      hardware.py       <- Python hardware class (thermal, power, GPU, serial)
      hardware.h        <- C++ hardware class
      fan_controller.py <- PID fan controller via /sys/devices/pwm-fan/target_pwm
    tici/               <- Original comma hardware (preserved)
    pc/                 <- PC/simulation (preserved)
  selfdrive/modeld/     <- Model inference (CUDA adaptation)
  common/realtime.py    <- Real-time scheduling (graceful fallback)
  SConstruct            <- Build system (jarch64 support)
```

## Model Performance (Jetson AGX Xavier)

### modeld End-to-End (CUDA Zero-Copy, 3min benchmark)

| Metric | Value |
|--------|-------|
| **modeld execution (median)** | **15.85ms** |
| modeld execution (P95) | 16.64ms |
| **Frame drops** | **0 (0.00%)** |
| dmonitoringmodeld (median) | 20.83ms |
| **Improvement vs OpenCL** | **2.9x faster (46ms → 16ms)** |

### Individual Model Inference (tinygrad CUDA Graphs)

| Model | Inference Time | Notes |
|-------|---------------|-------|
| driving_vision (50.3M) | **~8.7ms** | CUDA FP16, Volta tensor cores |
| driving_policy (7.0M) | **~3.3ms** | CUDA FP16 |
| dmonitoring_model (9.6M) | **~6.5ms** | CUDA FP16 |

All models compiled with `DEV=CUDA FLOAT16=1 JIT_BATCH_SIZE=32 TC=1` via tinygrad.

## Credits

- [DragonPilot Community](https://github.com/dragonpilot-community/dragonpilot) - Original project
- [comma.ai / openpilot](https://github.com/commaai/openpilot) - Base platform
- [xnxpilot](https://github.com/eFiniLan/xnxpilot) - Prior art for Jetson porting (openpilot 0.8.9)
- [tinygrad](https://github.com/tinygrad/tinygrad) - ML inference engine with CUDA support

## License

MIT License (same as openpilot). See [LICENSE](LICENSE).

---

<a name="portuguese"></a>

# DragonPilot - Adaptacao para Jetson AGX Xavier

![Plataforma](https://img.shields.io/badge/Plataforma-Jetson_AGX_Xavier-76B900?logo=nvidia&logoColor=white)
![JetPack](https://img.shields.io/badge/JetPack-5.x_(R35)-76B900?logo=nvidia)
![CUDA](https://img.shields.io/badge/CUDA-11.4-76B900?logo=nvidia)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![openpilot](https://img.shields.io/badge/openpilot-0.10.3-blue)
![Licenca](https://img.shields.io/badge/Licenca-MIT-green)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)
![tinygrad](https://img.shields.io/badge/Inferencia-tinygrad_CUDA-orange)

**Trazendo as capacidades de direcao autonoma do DragonPilot para hardware NVIDIA Jetson.**

### Rodando na Jetson AGX Xavier

![DragonPilot rodando na Jetson AGX Xavier](docs/jetson/assets/ui_running_on_jetson.png)
*UI do DragonPilot com inferencia CUDA em tempo real na Jetson AGX Xavier — modeld 15.85ms median (CUDA zero-copy, 2.9x mais rapido que OpenCL)*

## Sobre Este Projeto

Esta e uma adaptacao do [DragonPilot](https://github.com/dragonpilot-community/dragonpilot) 0.10.3 para rodar nativamente na plataforma **NVIDIA Jetson AGX Xavier**. O DragonPilot e um fork comunitario do [openpilot](https://github.com/commaai/openpilot), originalmente projetado para hardware comma.ai (Snapdragon 845).

Este projeto preserva todo o historico e funcionalidades do DragonPilot, adicionando suporte nativo para Jetson.

## Por que Jetson?

| Caracteristica | comma 3X (Tici) | Jetson AGX Xavier |
|----------------|----------------|-------------------|
| Performance IA | ~6 TOPS | **32 TOPS** |
| RAM | 4 GB | **32 GB** |
| GPU | Adreno 630 (OpenCL) | **512 CUDA + 64 Tensor Cores (Volta)** |
| Tensor Cores | Nenhum | **64 (FP16/INT8)** |
| Armazenamento | 64 GB eMMC | **NVMe SSD (expansivel)** |

O Jetson AGX Xavier oferece **5x a performance de IA** com **8x a RAM**, tornando-o adequado para modelos mais avancados e desenvolvimento futuro.

## Status Atual

| Componente | Status | Notas |
|------------|--------|-------|
| Deteccao de Plataforma (`/JETSON` + `jarch64`) | Funcionando | SConstruct, hardware init |
| Abstracao de Hardware (thermal, energia, fan) | Funcionando | Classe Jetson completa com INA3221, PWM fan, zonas termicas |
| Compilacao de Modelos CUDA (tinygrad) | Funcionando | driving_vision, driving_policy, dmonitoring_model |
| CUDA Zero-Copy Preprocessing | Funcionando | 15.85ms median, 0% frame drops, Tensor.from_blob |
| Build scons (`-j8`) | Funcionando | Todos os alvos C++/Cython compilam sem erros |
| Processos Core (hardwared, pandad, loggerd...) | Funcionando | Todos verdes na linha de status do manager |
| UI (raylib) | Requer Display | Precisa de DISPLAY X11 ou modo headless |
| Camera (webcamerad) | Pronto | Requer webcam USB conectada |

## Adaptacoes Principais

- **Identidade de Plataforma**: Nova arquitetura `jarch64` com flag `-D__JETSON__`
- **GPU Compute**: Qualcomm QCOM/Adreno -> NVIDIA CUDA (Volta sm_72)
- **Inferencia de Modelos**: tinygrad `DEV=QCOM` -> `DEV=CUDA` com suporte FP16
- **Camera**: Qualcomm Spectra ISP -> Webcam USB / MIPI CSI-2
- **Memoria**: Alocador ION -> OpenCL padrao `visionbuf_cl.cc` (POCL via CPU)
- **Classe Hardware**: Nova classe `Jetson` com gerenciamento termico, energia e ventilador
- **Scheduling Real-Time**: Fallback gracioso quando SCHED_FIFO nao permitido

## Funcionalidades Preservadas

Todas as funcionalidades do DragonPilot sao preservadas:
- ALKA (Assistencia de Manutencao de Faixa Sempre Ativa)
- ACM (Modo de Coasting Adaptativo)
- AEM (Modo Experimental Adaptativo)
- DTSC (Controle Dinamico de Velocidade em Curvas)
- RED (Deteccao de Borda de Pista)
- Suporte multilinguistico completo
- 300+ veiculos suportados

## Documentacao

- [Guia de Setup](docs/jetson/SETUP_GUIDE.md) - Guia completo passo-a-passo para compilar e rodar
- [Plano de Port](docs/jetson/PORTING_PLAN.md) - Plano completo de implementacao por fases
- [Analise de Arquitetura](docs/jetson/ARCHITECTURE_ANALYSIS.md) - Analise de plataforma e pipeline GPU
- [Inventario de Arquivos](docs/jetson/FILE_INVENTORY.md) - Todos os arquivos a modificar/criar
- [Specs do Hardware](docs/jetson/HARDWARE_SPECS.md) - Especificacoes do Jetson AGX Xavier

## Inicio Rapido (na Jetson)

Veja o [Guia de Setup](docs/jetson/SETUP_GUIDE.md) completo para instrucoes detalhadas.

```bash
# 1. Criar marker de plataforma e diretorio /data
sudo touch /JETSON
sudo mkdir -p /data && sudo chown $USER:$USER /data
# Se eMMC for pequena, facer symlink para NVMe:
# ln -s /home/$USER /data

# 2. Adicionar CUDA ao PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 3. Instalar dependencias do sistema
sudo apt-get install -y build-essential git clang cmake pkg-config gettext \
  gcc-arm-none-eabi libzmq3-dev libcapnp-dev capnproto libusb-1.0-0-dev \
  ffmpeg libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
  opencl-headers ocl-icd-opencl-dev ocl-icd-libopencl1 pocl-opencl-icd \
  libglfw3-dev libglew-dev libgles2-mesa-dev portaudio19-dev \
  libeigen3-dev libsqlite3-dev libzstd-dev libbz2-dev libffi-dev libssl-dev \
  zlib1g-dev libreadline-dev liblzma-dev tk-dev

# 4. Compilar Python 3.11 do source (deadsnakes nao tem arm64)
cd /tmp
wget https://www.python.org/ftp/python/3.11.11/Python-3.11.11.tar.xz
tar xf Python-3.11.11.tar.xz && cd Python-3.11.11
./configure --prefix=/usr/local && make -j8 && sudo make altinstall
cd ~ && rm -rf /tmp/Python-3.11*

# 5. Clonar e configurar
cd /data
git clone --recurse-submodules -b development \
  git@github.com:Guiimartinho/dragonpilot-adapt-jetson.git openpilot
cd openpilot
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e . && pip install -e msgq_repo/ -e opendbc_repo/ -e panda/ -e tinygrad_repo/
pip install opencv-python-headless Pillow jeepney dbus-next

# 6. Compilar
export TMPDIR=/home/$USER/tmp && mkdir -p $TMPDIR
scons -j8

# 7. Rodar (com webcam USB)
USE_WEBCAM=1 python3 -c "from openpilot.system.manager.manager import main; main()"
```

## Requisitos de Hardware

- **NVIDIA Jetson AGX Xavier** (32GB recomendado)
- JetPack 5.x (R35, CUDA 11.4+)
- SSD NVMe (recomendado - eMMC 28GB e insuficiente para o build)
- Webcam USB (para teste inicial) ou camera MIPI CSI-2
- [comma Panda](https://comma.ai/shop/panda) (para comunicacao veicular)
- Chicote para seu veiculo suportado

## Performance dos Modelos (Jetson AGX Xavier)

### modeld End-to-End (CUDA Zero-Copy, benchmark 3min)

| Metrica | Valor |
|---------|-------|
| **modeld execution (median)** | **15.85ms** |
| modeld execution (P95) | 16.64ms |
| **Frame drops** | **0 (0.00%)** |
| dmonitoringmodeld (median) | 20.83ms |
| **Melhoria vs OpenCL** | **2.9x mais rapido (46ms → 16ms)** |

### Inferencia Individual (tinygrad CUDA Graphs)

| Modelo | Tempo de Inferencia | Notas |
|--------|---------------------|-------|
| driving_vision (50.3M) | **~8.7ms** | CUDA FP16, tensor cores Volta |
| driving_policy (7.0M) | **~3.3ms** | CUDA FP16 |
| dmonitoring_model (9.6M) | **~6.5ms** | CUDA FP16 |

Todos os modelos compilados com `DEV=CUDA FLOAT16=1 JIT_BATCH_SIZE=32 TC=1` via tinygrad.

## Aviso de Seguranca

DragonPilot e um sistema de **assistencia** ao motorista, nao direcao autonoma completa. Voce deve permanecer alerta e pronto para assumir o controle a qualquer momento. Siga sempre as leis de transito locais.

## Creditos

- [Comunidade DragonPilot](https://github.com/dragonpilot-community/dragonpilot) - Projeto original
- [comma.ai / openpilot](https://github.com/commaai/openpilot) - Plataforma base
- [xnxpilot](https://github.com/eFiniLan/xnxpilot) - Referencia para port Jetson (openpilot 0.8.9)
- [tinygrad](https://github.com/tinygrad/tinygrad) - Motor de inferencia ML com suporte CUDA

## Licenca

MIT License (mesma do openpilot). Veja [LICENSE](LICENSE).
