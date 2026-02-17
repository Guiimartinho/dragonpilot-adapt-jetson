![](dragonpilot/selfdrive/assets/dragonpilot.png)

[Leia em Portugues](#portuguese) | [Read in English](#english) | [DragonPilot Original (Chinese)](https://github.com/dragonpilot-community/dragonpilot) | [DragonPilot Original (English)](README_EN.md)

---

<a name="english"></a>

# DragonPilot - Jetson AGX Xavier Adaptation

**Bringing DragonPilot's autonomous driving capabilities to NVIDIA Jetson hardware.**

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

## Key Adaptations

- **Platform Identity**: New `jarch64` architecture with `-D__JETSON__` compile flag
- **GPU Compute**: Qualcomm QCOM/Adreno -> NVIDIA CUDA (Volta sm_72)
- **Model Inference**: tinygrad `DEV=QCOM` -> `DEV=CUDA` with FP16 support
- **Camera**: Qualcomm Spectra ISP -> USB webcam / MIPI CSI-2
- **Memory**: ION allocator -> Standard OpenCL `visionbuf_cl.cc`
- **Hardware Class**: New `Jetson` class with thermal, power, and fan management

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
# 1. Clone this repo
git clone git@github.com:Guiimartinho/dragonpilot-adapt-jetson.git /data/openpilot
cd /data/openpilot

# 2. Create platform marker
sudo touch /JETSON

# 3. Setup Python 3.11 environment
sudo apt install python3.11 python3.11-dev python3.11-venv
python3.11 -m venv /data/openpilot_venv
source /data/openpilot_venv/bin/activate

# 4. Install dependencies
pip install -e '.[dev]'
scons -j8

# 5. Run (with USB webcam)
USE_WEBCAM=1 python -m selfdrive.manager.manager
```

## Hardware Requirements

- **NVIDIA Jetson AGX Xavier** (32GB recommended)
- JetPack 5.x (CUDA 11.4+)
- NVMe SSD (recommended for storage)
- USB webcam (for initial testing) or MIPI CSI-2 camera
- [comma Panda](https://comma.ai/shop/panda) (for vehicle communication)
- Car harness for your supported vehicle

## Project Structure

```
dragonpilot-adapt-jetson/
  docs/jetson/          <- Jetson-specific documentation
  system/hardware/
    jetson/             <- NEW: Jetson hardware abstraction
      hardware.py       <- Jetson hardware class
      hardware.h        <- C++ hardware class
      fan_controller.py <- Fan management
    tici/               <- Original comma hardware (preserved)
    pc/                 <- PC/simulation (preserved)
  selfdrive/modeld/     <- Model inference (CUDA adaptation)
  SConstruct            <- Build system (jarch64 support)
```

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

**Trazendo as capacidades de direcao autonoma do DragonPilot para hardware NVIDIA Jetson.**

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

## Adaptacoes Principais

- **Identidade de Plataforma**: Nova arquitetura `jarch64` com flag `-D__JETSON__`
- **GPU Compute**: Qualcomm QCOM/Adreno -> NVIDIA CUDA (Volta sm_72)
- **Inferencia de Modelos**: tinygrad `DEV=QCOM` -> `DEV=CUDA` com suporte FP16
- **Camera**: Qualcomm Spectra ISP -> Webcam USB / MIPI CSI-2
- **Memoria**: Alocador ION -> OpenCL padrao `visionbuf_cl.cc`
- **Classe Hardware**: Nova classe `Jetson` com gerenciamento termico, energia e ventilador

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
# 1. Clonar este repo
git clone git@github.com:Guiimartinho/dragonpilot-adapt-jetson.git /data/openpilot
cd /data/openpilot

# 2. Criar marker de plataforma
sudo touch /JETSON

# 3. Configurar ambiente Python 3.11
sudo apt install python3.11 python3.11-dev python3.11-venv
python3.11 -m venv /data/openpilot_venv
source /data/openpilot_venv/bin/activate

# 4. Instalar dependencias
pip install -e '.[dev]'
scons -j8

# 5. Rodar (com webcam USB)
USE_WEBCAM=1 python -m selfdrive.manager.manager
```

## Requisitos de Hardware

- **NVIDIA Jetson AGX Xavier** (32GB recomendado)
- JetPack 5.x (CUDA 11.4+)
- SSD NVMe (recomendado)
- Webcam USB (para teste inicial) ou camera MIPI CSI-2
- [comma Panda](https://comma.ai/shop/panda) (para comunicacao veicular)
- Chicote para seu veiculo suportado

## Aviso de Seguranca

DragonPilot e um sistema de **assistencia** ao motorista, nao direcao autonoma completa. Voce deve permanecer alerta e pronto para assumir o controle a qualquer momento. Siga sempre as leis de transito locais.

## Creditos

- [Comunidade DragonPilot](https://github.com/dragonpilot-community/dragonpilot) - Projeto original
- [comma.ai / openpilot](https://github.com/commaai/openpilot) - Plataforma base
- [xnxpilot](https://github.com/eFiniLan/xnxpilot) - Referencia para port Jetson (openpilot 0.8.9)

## Licenca

MIT License (mesma do openpilot). Veja [LICENSE](LICENSE).
