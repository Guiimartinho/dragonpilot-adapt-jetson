# DragonPilot Jetson AGX Xavier - Analise de Modelos e Inferencia

## Modelos Neurais do Projeto

### Localizacao
```
selfdrive/modeld/models/
├── driving_vision.onnx        (45 MB)  - Modelo principal de visao
├── driving_policy.onnx        (14 MB)  - Modelo de politica/planejamento
├── dmonitoring_model.onnx     (6.7 MB) - Monitoramento do motorista
├── big_driving_policy.onnx    → symlink para driving_policy.onnx
├── big_driving_vision.onnx    → symlink para driving_vision.onnx
├── driving_vision_metadata.pkl          (gerado pelo build)
├── driving_policy_metadata.pkl          (gerado pelo build)
├── dmonitoring_model_metadata.pkl       (gerado pelo build)
├── driving_vision_tinygrad.pkl  (50.3 MB) (compilado para CUDA)
├── driving_policy_tinygrad.pkl  (14 MB)   (compilado para CUDA)
└── dmonitoring_model_tinygrad.pkl (9.6 MB) (compilado para CUDA)
```

**Nota**: `big_driving_*.onnx` sao symlinks identicos aos modelos normais. Nao sao modelos separados.

Nao existem outros modelos no projeto (`.pt`, `.pth`, `.pb`, `.tflite`, `.dlc`, `.engine`, `.plan` - nenhum encontrado).

---

## 1. Arquitetura dos Modelos

### 1.1 driving_vision.onnx (45 MB, ~100-150M parametros)

**Funcao**: Extrair features visuais das cameras (road + wide road)

**Inputs**:
| Nome | Shape | Dtype | Descricao |
|------|-------|-------|-----------|
| road | (1, 12, 128, 256) | uint8 | 2 frames consecutivos YUV420, 6 canais cada |
| wide_road | (1, 12, 128, 256) | uint8 | 2 frames consecutivos YUV420, camera wide |

**YUV420 layout** (por frame, 6 canais × 128×256):
- Canais 0-3: Y full-res (Y[::2,::2], Y[::2,1::2], Y[1::2,::2], Y[1::2,1::2])
- Canal 4: U half-res
- Canal 5: V half-res

**Outputs** (1576 float16):
| Saida | Descricao |
|-------|-----------|
| pose | 6D pose do veiculo (posicao + rotacao + covariancia) |
| lane_lines | 4 faixas × 33 indices temporais × 2 (posicao + incerteza) |
| road_edges | 2 bordas × 33 indices temporais × 2 |
| lead | 2 veiculos a frente × 6 predicoes de trajetoria (MHP) |
| lead_prob | Probabilidade de deteccao de veiculo a frente |
| desire_pred | 4 predicoes × 8 categorias de desejo |
| meta | Engagement, desengajamento gas/freio/direcao, frenagem brusca |
| **hidden_state** | **512 features → alimenta o policy model** |

**Perfil de compute**: **COMPUTE-BOUND** (dominado por Conv2d, beneficia-se de tensor cores)

### 1.2 driving_policy.onnx (14 MB, ~15-25M parametros)

**Funcao**: Decisao temporal - converte features visuais + desejo em plano de direcao

**Inputs**:
| Nome | Shape | Dtype | Descricao |
|------|-------|-------|-----------|
| desire_pulse | (1, 100, 8) | float16 | 5s de contexto temporal (100 frames × 8 categorias) |
| features_buffer | (1, 100, 512) | float16 | Hidden state do vision model (100 frames × 512) |
| traffic_convention | (1, 2) | float16 | Mao esquerda/direita (one-hot) |
| lateral_control_params | (1, 2) | float16 | Velocidade e delay de direcao |
| prev_desired_curvatures | (1, 100) | float16 | Historico de curvaturas anteriores |

**Outputs**:
| Saida | Descricao |
|-------|-----------|
| plan | 33 indices temporais × 15 valores (posicao, velocidade, aceleracao, rotacao, taxa de orientacao) |
| desire_state | 8 probabilidades de desejo (softmax) |

**Perfil de compute**: **MEMORY-BOUND** (LSTM/GRU temporal, dependencias sequenciais)

**Dependencia critica**: policy depende do output do vision (hidden_state). Nao podem rodar em paralelo.

### 1.3 dmonitoring_model.onnx (6.7 MB, ~5-10M parametros)

**Funcao**: Monitoramento do motorista - deteccao facial, atencao, estado

**Inputs**:
| Nome | Shape | Dtype | Descricao |
|------|-------|-------|-----------|
| input_img | (1, 1, 960, 1440) | uint8 | Frame da camera do motorista (Y plane, 1.38M pixels) |
| calib | (1, 3) | float32 | Calibracao da camera (roll, pitch, yaw) |

**Outputs** (84 float32):
| Saida | Descricao |
|-------|-----------|
| face_descs (LHD/RHD) | Orientacao (pitch/yaw/roll), posicao, tamanho + desvios padrao |
| face_prob | Probabilidade de face visivel |
| eye outputs | Posicao dos olhos, probabilidade de olhos fechados |
| sunglasses_prob | Probabilidade de oculos de sol |
| using_phone_prob | Probabilidade de uso de celular |
| wheel_on_right | Deteccao de mao inglesa vs mao esquerda |

**Perfil de compute**: **BALANCEADO** (Conv2d backbone + regressao de landmarks)

**Independente**: Roda em processo separado, nao depende dos modelos de direcao.

---

## 2. Pipeline Atual de Inferencia

### Compilacao (ONNX → TinyJit Pickle)
```
ONNX → OnnxRunner(tinygrad) → TinyJit capture → CUDA Graphs → Pickle serializado
```

**Flags de compilacao Jetson**: `DEV=CUDA FLOAT16=1 JIT_BATCH_SIZE=32 TC=1`

**Processo de compilacao** (`compile3.py`):
1. Carrega ONNX via `OnnxRunner`
2. Com `FLOAT16=1 + DEV=CUDA`: mantem FP16 sem cast para FP32
3. Run 0: baseline
4. Run 1: JIT scheduling (prune ops nao usadas, 126 → 122 kernels)
5. Run 2: CUDA graph capture (`JIT_BATCH_SIZE=32` → 3 grafos: 32+64+26 kernels)
6. Runs 3+: graph replay (~8.8ms)
7. Serializa para pickle

### Execucao em runtime (`modeld.py`)
```
Camera YUV420 → OpenCL transform → clEnqueueReadBuffer (DMA pinado)
→ Tensor CUDA uint8 → Vision CUDA Graph replay (8.8ms)
→ parse hidden_state (512 features)
→ Policy CUDA Graph replay → parse plan + desire
```

### Benchmarks medidos (Jetson AGX Xavier)
| Modelo | Kernels | CUDA Graphs | Latencia | Enqueue |
|--------|---------|-------------|----------|---------|
| driving_vision | 122 | 3 (32+64+26) | 8.68-9.52ms (avg 8.85ms) | ~0.9ms |
| driving_policy | ~122 | 3 | Nao medido isolado | - |
| dmonitoring | ~122 | 3 | Nao medido isolado | - |

---

## 3. Comparacao de Formatos para Jetson AGX Xavier

### 3.1 Tabela Comparativa

| Formato | Latencia Vision (est.) | Fusion de Kernels | INT8 | DLA | Complexidade |
|---------|----------------------|-------------------|------|-----|--------------|
| **tinygrad TinyJit (atual)** | ~8.8ms | Nao | Nao | Nao | Ja implementado |
| **TensorRT FP16** | **~3-5ms** | Sim (agressivo) | Sim | Sim | Codigo pronto, falta instalar |
| **TensorRT INT8** | ~2.5-3.5ms | Sim | Sim | Sim | Precisa calibracao |
| **ONNX Runtime CUDA** | ~6-8ms | Parcial | Nao | Nao | Dependencia extra |
| **Apache TVM** | ~5-7ms | Auto-tuning | Nao | Nao | Tuning demorado |
| **torch2trt** | ~3-5ms | Via TRT | Sim | Sim | Requer PyTorch (nao temos) |
| **CUDA/cuDNN manual** | ~2-4ms | Manual | Sim | N/A | Meses de trabalho |

### 3.2 Analise Detalhada

#### tinygrad TinyJit + CUDA Graphs (ATUAL)

**Pros**:
- Ja integrado e funcionando
- Python-native, facil de debugar
- Mesmo framework do openpilot upstream (comma.ai)
- CUDA Graphs minimizam overhead de launch (0.9ms enqueue)
- FP16 + tensor cores ativos

**Contras**:
- Gera kernels CUDA genericos, NAO otimizados para Volta sm_72
- Sem fusion de camadas (Conv+BN+ReLU sao kernels separados)
- 122 kernels = 122 leituras/escritas intermediarias na memoria
- Sem suporte INT8
- Sem offload para DLA

#### TensorRT FP16 (RECOMENDADO)

**Pros**:
- **Fusion agressiva de camadas**: Conv+BN+ReLU viram 1 kernel (elimina escrita/leitura intermediaria)
- **Kernels hand-tuned para Volta sm_72**: TensorRT escolhe entre centenas de kernels otimizados
- **Reducao de kernels**: de 122 (tinygrad) para ~30-50 (fusionados)
- **Suporte a DLA** nativo para offload de modelos menores
- **Cache de engines**: build uma vez, reutiliza em todos os boots
- **Compativel com CUDA Graphs**: TRT pode capturar execucao em grafos

**Contras**:
- Build time: 30-120s por modelo (uma vez, cacheado)
- Engine nao e portavel (atrelado a versao TRT + arquitetura GPU)
- Requer `python3-libnvinfer` (disponivel no JetPack 5.1.4)

**Speedup esperado**:
- Vision: 8.8ms → **3-5ms** (40-65% reducao)
- Policy: ~3-4ms → **1-2ms**
- Total driving: ~12-13ms → **4-7ms** (1.8-3.2x mais rapido)

#### TensorRT INT8

**Speedup adicional**: 10-30% sobre FP16 no Volta

**Riscos para direcao autonoma**:
- PTQ (Post-Training Quantization) pode falhar catastroficamente em algumas arquiteturas
- Exemplo documentado: EfficientNet B0 caiu de 77.4% para 33.9% com PTQ
- Para seguranca: QAT (Quantization-Aware Training) e necessario, mas requer pipeline de treino

**Processo de calibracao**:
1. Coletar 500-1000 frames de direcao representativos
2. Forward pass FP32 para coletar distribuicao de ativacoes
3. TensorRT aplica calibracao entropy/minmax
4. Validar outputs contra FP16 em milhares de cenarios

**Veredicto**: Comecar com FP16. INT8 so depois de validacao extensiva.

#### ONNX Runtime / TVM / torch2trt

**Nao recomendados** para este projeto:
- ONNX Runtime: mais lento que TensorRT, sem DLA
- TVM: TensorRT supera consistentemente em hardware NVIDIA
- torch2trt: modelos sao ONNX, nao PyTorch

---

## 4. DLA (Deep Learning Accelerator) - Offload de Modelos

Xavier tem 2 engines DLA independentes (~5 TOPS INT8 / 2.5 TFLOPS FP16 cada, 0.5-1.5W).

### Adequacao por modelo

| Modelo | DLA? | Justificativa |
|--------|------|---------------|
| driving_vision (45MB) | **NAO** | Grande demais, latencia critica, GPU e muito mais rapido |
| driving_policy (14MB) | **NAO** | Depende do output da vision (GPU), transferencia GPU→DLA→GPU adiciona latencia |
| **dmonitoring (6.7MB)** | **SIM** | Menor modelo, arquitetura simples (conv+bn+relu), independente, libera GPU |

### Camadas suportadas pelo DLA
Convolution, Deconvolution, Fully Connected, Activation (ReLU, sigmoid, tanh), Pooling, BatchNorm, ElementWise, Concatenation, Resize (nearest), Softmax, Slice, Shuffle.

### Beneficios do DLA para dmonitoring
- **Libera GPU** inteiramente para driving_vision + driving_policy
- **Economia de energia**: ~1W (DLA) vs ~5-10W (GPU) para este workload
- **Paralelismo real**: DLA roda simultaneamente com GPU sem competir por recursos
- Ja implementado em `selfdrive/modeld/runners/dla_runner.py`

---

## 5. Otimizacoes de Memoria

### Xavier Unified Memory (32GB LPDDR4x, ~100GB/s)
CPU e GPU compartilham mesma memoria fisica. Nao precisa de copias explicitas CPU↔GPU com unified memory.

### Pipeline atual de transferencia
```
Camera ISP → shm mmap → cudaHostRegister (pinar paginas)
→ OpenCL transform (YUV→RGB+resize)
→ clEnqueueReadBuffer (DMA com memoria pinada)
→ Tensor CUDA (cpu→gpu copy)
→ Inferencia
```

### Otimizacoes possiveis com TensorRT
1. **Eliminar round-trip CPU**: Manter dados na GPU durante toda a pipeline
   - Atual: Tensor → numpy (CPU) → memcpy_htod (GPU) → TRT → memcpy_dtoh (CPU)
   - Otimizado: Buffer CUDA direto como input do TRT (economia ~0.5-1ms por modelo)
2. **Pre-alocar buffers de output**: Evitar alocacao de numpy a cada inferencia
3. **Multi-stream**: dmonitoring (DLA) + driving (GPU) em paralelo real

---

## 6. Quantizacao - Impacto na Precisao

### FP16 (ATUAL - SEGURO)

- Impacto na precisao: **< 0.2%** vs FP32 (documentado em 12 arquiteturas CNN)
- Range: 6×10⁻⁸ a 65,504 - suficiente para todos os pesos e ativacoes tipicos
- Tensor cores Volta fazem multiplicacao FP16 com acumulacao FP32 nativamente
- **Usado pela comma.ai em producao** no Snapdragon 845
- **Veredicto: Seguro e eficaz, sem necessidade de validacao extra**

### INT8 (OPCIONAL - REQUER VALIDACAO)

| Aspecto | Detalhe |
|---------|---------|
| Speedup sobre FP16 | 10-30% no Volta (menos dramatico que em Ampere/Ada) |
| Risco | PTQ pode falhar catastroficamente em algumas arquiteturas |
| Calibracao | 500-1000 frames representativos (dia/noite, chuva, urbano/rodovia) |
| Validacao | Comparar trajetorias, posicoes de faixa, comandos de aceleracao |
| Recomendacao | **Nao usar sem validacao extensiva. FP16 ja e suficiente.** |

### Mixed Precision (FP16 compute + FP32 accumulate)
**Ja e o comportamento padrao dos tensor cores Volta.** Multiplicacao em FP16, acumulacao em FP32. Nenhuma configuracao adicional necessaria.

---

## 7. Recomendacao Final por Modelo

| Modelo | Formato Atual | Formato Recomendado | Latencia Esperada | Acao |
|--------|--------------|--------------------|--------------------|------|
| driving_vision | tinygrad FP16 (~8.8ms) | **TensorRT FP16 GPU** | **3-5ms** | Instalar TRT, ativar USE_TENSORRT=1 |
| driving_policy | tinygrad FP16 | **TensorRT FP16 GPU** | **1-2ms** | Mesmo engine pipeline |
| dmonitoring | tinygrad FP16 | **TensorRT FP16 DLA core 0** | **3-5ms (DLA)** | Instalar TRT, ativar USE_DLA=1 |

### Roadmap de implementacao

**Fase 1 - Instalar TensorRT** (imediato):
```bash
sudo apt-get install python3-libnvinfer python3-libnvinfer-dev
pip install pycuda
```
O codigo ja existe (`tensorrt_runner.py`, `dla_runner.py`). Com os pacotes instalados, `USE_TENSORRT=1` e `USE_DLA=1` ativam automaticamente.

**Fase 2 - Profile e validacao**:
```bash
# Verificar fusion e timing
trtexec --onnx=driving_vision.onnx --fp16 --verbose
trtexec --onnx=driving_policy.onnx --fp16 --verbose
trtexec --onnx=dmonitoring_model.onnx --fp16 --useDLACore=0 --allowGPUFallback --verbose
```

**Fase 3 - Otimizar runner** (eliminar CPU round-trips):
- Usar CUDA device pointers direto como input do TRT
- Pre-alocar buffers de output

**Fase 4 - INT8 (opcional, so apos validacao)**:
- Coletar dataset de calibracao
- Build engine INT8 com calibracao entropy
- Validar numericamente contra FP16

---

## 8. Comandos de Build TensorRT

```bash
# FP16 para driving_vision (GPU)
trtexec --onnx=driving_vision.onnx --saveEngine=driving_vision_fp16.engine \
  --fp16 --workspace=1024 --verbose

# FP16 para driving_policy (GPU)
trtexec --onnx=driving_policy.onnx --saveEngine=driving_policy_fp16.engine \
  --fp16 --workspace=1024 --verbose

# FP16 para dmonitoring (DLA core 0 com fallback GPU)
trtexec --onnx=dmonitoring_model.onnx --saveEngine=dmonitoring_dla0_fp16.engine \
  --fp16 --useDLACore=0 --allowGPUFallback --workspace=512 --verbose

# INT8 com calibracao (se necessario)
trtexec --onnx=driving_vision.onnx --saveEngine=driving_vision_int8.engine \
  --int8 --fp16 --calib=calibration_cache.bin --workspace=1024 --verbose
```

---

## Fontes

- [Jetson Benchmarks - NVIDIA Developer](https://developer.nvidia.com/embedded/jetson-benchmarks)
- [TensorRT-Based Framework and Optimization for Jetson (ACM TECS)](https://dl.acm.org/doi/10.1145/3508391)
- [Benchmark FP32/FP16/INT8 with TensorRT on Xavier](https://github.com/kentaroy47/benchmark-FP32-FP16-INT8-with-TensorRT)
- [Fast INT8 Inference for Autonomous Vehicles with TensorRT](https://developer.nvidia.com/blog/int8-inference-autonomous-vehicles-tensorrt/)
- [Achieving FP32 Accuracy for INT8 Inference Using QAT](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)
- [Jetson DLA Tutorial - NVIDIA-AI-IOT](https://github.com/NVIDIA-AI-IOT/jetson_dla_tutorial)
- [Working with DLA - NVIDIA TensorRT Docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html)
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
- [Volta Tuning Guide - NVIDIA](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html)
