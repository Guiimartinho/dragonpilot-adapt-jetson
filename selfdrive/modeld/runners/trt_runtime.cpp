/**
 * Minimal TensorRT C runtime bridge for Python ctypes.
 * Bypasses the Python 3.8-only tensorrt package by exposing
 * plain C functions that wrap the TRT C++ API.
 *
 * Compile: g++ -shared -fPIC -O2 -o trt_runtime.so trt_runtime.cpp -lnvinfer -lcudart
 */
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <vector>

#define MAX_IO_TENSORS 32

class TRTLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING)
      fprintf(stderr, "[TRT] %s\n", msg);
  }
};
static TRTLogger sLogger;

struct IOTensor {
  char name[256];
  int is_input;
  int64_t size_bytes;
  void* d_mem;
  int dims[8];
  int ndims;
  int dtype;  // 0=fp32, 1=fp16, 2=int8, 3=int32, 4=uint8
};

struct TRTContext {
  nvinfer1::IRuntime* runtime;
  nvinfer1::ICudaEngine* engine;
  nvinfer1::IExecutionContext* context;
  cudaStream_t stream;
  int num_io;
  IOTensor tensors[MAX_IO_TENSORS];
};

extern "C" {

TRTContext* trt_load_engine(const char* engine_path) {
  std::ifstream f(engine_path, std::ios::binary);
  if (!f.good()) return nullptr;
  f.seekg(0, std::ios::end);
  size_t sz = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> data(sz);
  f.read(data.data(), sz);

  auto ctx = new TRTContext();
  ctx->runtime = nvinfer1::createInferRuntime(sLogger);
  if (!ctx->runtime) { delete ctx; return nullptr; }

  ctx->engine = ctx->runtime->deserializeCudaEngine(data.data(), sz);
  if (!ctx->engine) { delete ctx->runtime; delete ctx; return nullptr; }

  ctx->context = ctx->engine->createExecutionContext();
  cudaStreamCreate(&ctx->stream);

  ctx->num_io = ctx->engine->getNbIOTensors();
  if (ctx->num_io > MAX_IO_TENSORS) ctx->num_io = MAX_IO_TENSORS;

  for (int i = 0; i < ctx->num_io; i++) {
    auto& t = ctx->tensors[i];
    const char* name = ctx->engine->getIOTensorName(i);
    strncpy(t.name, name, 255);
    t.name[255] = '\0';
    t.is_input = (ctx->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) ? 1 : 0;

    auto dims = ctx->engine->getTensorShape(name);
    t.ndims = dims.nbDims;
    int64_t vol = 1;
    for (int d = 0; d < dims.nbDims; d++) {
      t.dims[d] = dims.d[d];
      vol *= dims.d[d];
    }

    auto dtype = ctx->engine->getTensorDataType(name);
    int elem_size = 4;
    t.dtype = 0;
    switch (dtype) {
      case nvinfer1::DataType::kFLOAT: elem_size = 4; t.dtype = 0; break;
      case nvinfer1::DataType::kHALF:  elem_size = 2; t.dtype = 1; break;
      case nvinfer1::DataType::kINT8:  elem_size = 1; t.dtype = 2; break;
      case nvinfer1::DataType::kINT32: elem_size = 4; t.dtype = 3; break;
      default: break;
    }
    t.size_bytes = vol * elem_size;
    cudaMalloc(&t.d_mem, t.size_bytes);
    ctx->context->setTensorAddress(name, t.d_mem);
  }
  return ctx;
}

int trt_num_io(TRTContext* ctx) { return ctx ? ctx->num_io : 0; }
const char* trt_tensor_name(TRTContext* ctx, int i) { return ctx->tensors[i].name; }
int trt_tensor_is_input(TRTContext* ctx, int i) { return ctx->tensors[i].is_input; }
int64_t trt_tensor_size(TRTContext* ctx, int i) { return ctx->tensors[i].size_bytes; }
int trt_tensor_dtype(TRTContext* ctx, int i) { return ctx->tensors[i].dtype; }
int trt_tensor_ndims(TRTContext* ctx, int i) { return ctx->tensors[i].ndims; }
int trt_tensor_dim(TRTContext* ctx, int i, int d) { return ctx->tensors[i].dims[d]; }

void trt_set_input(TRTContext* ctx, int i, const void* data, int64_t nbytes) {
  cudaMemcpyAsync(ctx->tensors[i].d_mem, data, nbytes, cudaMemcpyHostToDevice, ctx->stream);
}

void trt_execute(TRTContext* ctx) {
  ctx->context->enqueueV3(ctx->stream);
}

void trt_get_output(TRTContext* ctx, int i, void* data, int64_t nbytes) {
  cudaMemcpyAsync(data, ctx->tensors[i].d_mem, nbytes, cudaMemcpyDeviceToHost, ctx->stream);
}

void trt_sync(TRTContext* ctx) {
  cudaStreamSynchronize(ctx->stream);
}

void trt_destroy(TRTContext* ctx) {
  if (!ctx) return;
  for (int i = 0; i < ctx->num_io; i++)
    cudaFree(ctx->tensors[i].d_mem);
  cudaStreamDestroy(ctx->stream);
  delete ctx->context;
  delete ctx->engine;
  delete ctx->runtime;
  delete ctx;
}

// Build engine from ONNX using trtexec CLI (simplest approach)
int trt_build_engine(const char* onnx_path, const char* engine_path,
                     int fp16, int dla_core, int workspace_mb) {
  char cmd[2048];
  const char* trtexec = "/usr/src/tensorrt/bin/trtexec";

  if (dla_core >= 0) {
    // DLA requires FP16, always enable it
    snprintf(cmd, sizeof(cmd),
      "%s --onnx=%s --saveEngine=%s --fp16 --useDLACore=%d --allowGPUFallback --workspace=%d 2>&1",
      trtexec, onnx_path, engine_path, dla_core, workspace_mb);
  } else {
    snprintf(cmd, sizeof(cmd),
      "%s --onnx=%s --saveEngine=%s %s --workspace=%d 2>&1",
      trtexec, onnx_path, engine_path,
      fp16 ? "--fp16" : "",
      workspace_mb);
  }
  return system(cmd);
}

}  // extern "C"
