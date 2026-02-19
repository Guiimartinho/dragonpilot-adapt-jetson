#pragma once

/**
 * CUDA-native ModelFrame for Jetson AGX Xavier.
 * Replaces OpenCL-based preprocessing with direct CUDA kernels.
 *
 * Pipeline comparison:
 *   OpenCL path: VisionIPC → cl_mem → POCL→CUDA warp → POCL→CUDA loadyuv → clReadBuffer → host → tinygrad upload
 *   CUDA path:   VisionIPC → cudaMemcpy → CUDA warp → CUDA loadyuv → device ptr (stays on GPU)
 *
 * Eliminates: POCL interop overhead, GPU→CPU read, CPU→GPU upload
 */

#ifdef __JETSON__

#include <cassert>
#include <cstring>
#include <memory>

#include <cuda_runtime.h>

#include "common/mat.h"
#include "selfdrive/modeld/transforms/transform_cuda.h"
#include "selfdrive/modeld/transforms/loadyuv_cuda.h"

class CudaModelFrame {
public:
  /**
   * Uses default CUDA stream (0) to avoid stream invalidation.
   * tinygrad's CUDA graph compilation/execution manipulates CUDA contexts,
   * which invalidates custom streams. The default stream is always valid.
   */
  CudaModelFrame() : stream_(0), gpu_initialized_(false) {}

  virtual ~CudaModelFrame() {
    // Default stream (0) doesn't need to be destroyed
  }

  virtual uint8_t* prepare(const uint8_t* yuv_data, int frame_width, int frame_height,
                           int frame_stride, int frame_uv_offset, const mat3& projection) = 0;

  uint8_t* get_host_buffer(int buffer_size) {
    if (!h_output_) {
      h_output_.reset(new uint8_t[buffer_size]);
    }
    cudaMemcpy(h_output_.get(), d_output_, buffer_size, cudaMemcpyDeviceToHost);
    return h_output_.get();
  }

  uint8_t* get_device_ptr() const { return d_output_; }
  cudaStream_t get_stream() const { return stream_; }

  int MODEL_WIDTH;
  int MODEL_HEIGHT;
  int MODEL_FRAME_SIZE;
  int buf_size;

protected:
  cudaStream_t stream_;      // Always 0 (default stream)
  bool gpu_initialized_;
  uint8_t* d_input_ = nullptr;
  uint8_t* d_y_ = nullptr;
  uint8_t* d_u_ = nullptr;
  uint8_t* d_v_ = nullptr;
  uint8_t* d_output_ = nullptr;
  std::unique_ptr<uint8_t[]> h_output_;

  CudaTransform transform_;
  CudaLoadYUVState loadyuv_;

  int deferred_model_width_ = 0;
  int deferred_model_height_ = 0;
  int deferred_input_size_ = 0;
  int deferred_output_size_ = 0;

  void init_cuda(int model_width, int model_height, int input_size, int output_size) {
    MODEL_WIDTH = model_width;
    MODEL_HEIGHT = model_height;
    deferred_model_width_ = model_width;
    deferred_model_height_ = model_height;
    deferred_input_size_ = input_size;
    deferred_output_size_ = output_size;
  }

  /** Lazily allocate GPU memory on first prepare() (after tinygrad CUDA context is active). */
  void ensure_gpu_initialized() {
    if (gpu_initialized_) return;

    cudaError_t err;
    err = cudaMalloc(&d_input_, deferred_input_size_);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA ERROR cudaMalloc d_input_: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&d_y_, deferred_model_width_ * deferred_model_height_);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA ERROR cudaMalloc d_y_: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&d_u_, (deferred_model_width_ / 2) * (deferred_model_height_ / 2));
    if (err != cudaSuccess) { fprintf(stderr, "CUDA ERROR cudaMalloc d_u_: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&d_v_, (deferred_model_width_ / 2) * (deferred_model_height_ / 2));
    if (err != cudaSuccess) { fprintf(stderr, "CUDA ERROR cudaMalloc d_v_: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&d_output_, deferred_output_size_);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA ERROR cudaMalloc d_output_: %s\n", cudaGetErrorString(err)); return; }

    // Use default stream (0) for all sub-objects
    transform_.init(stream_);
    loadyuv_.init(deferred_model_width_, deferred_model_height_, stream_);

    // Only set initialized after ALL allocations succeed
    gpu_initialized_ = true;
    fprintf(stderr, "CudaModelFrame: GPU initialized (default stream, lazy alloc)\n");
  }

  void deinit_cuda() {
    if (!gpu_initialized_) return;
    loadyuv_.destroy();
    transform_.destroy();
    if (d_output_) cudaFree(d_output_);
    if (d_v_) cudaFree(d_v_);
    if (d_u_) cudaFree(d_u_);
    if (d_y_) cudaFree(d_y_);
    if (d_input_) cudaFree(d_input_);
  }

  void upload_and_transform(const uint8_t* yuv_data, int frame_width, int frame_height,
                            int frame_stride, int frame_uv_offset, const mat3& projection) {
    int input_size = frame_uv_offset + (frame_stride * frame_height / 2);
    cudaMemcpy(d_input_, yuv_data, input_size, cudaMemcpyHostToDevice);

    transform_.queue(d_input_, frame_width, frame_height, frame_stride, frame_uv_offset,
                     d_y_, d_u_, d_v_, MODEL_WIDTH, MODEL_HEIGHT, projection);
  }
};

class CudaDrivingModelFrame : public CudaModelFrame {
public:
  CudaDrivingModelFrame(int temporal_skip) : temporal_skip_(temporal_skip) {
    const int mw = 512, mh = 256;
    MODEL_WIDTH = mw;
    MODEL_HEIGHT = mh;
    MODEL_FRAME_SIZE = mw * mh * 3 / 2;
    buf_size = MODEL_FRAME_SIZE * 2;

    int max_input_size = 2048 * 1208 * 2;
    init_cuda(mw, mh, max_input_size, buf_size);

    ring_size_ = temporal_skip + 1;
  }

  ~CudaDrivingModelFrame() {
    if (d_ring_buffer_) cudaFree(d_ring_buffer_);
    deinit_cuda();
  }

  uint8_t* prepare(const uint8_t* yuv_data, int frame_width, int frame_height,
                   int frame_stride, int frame_uv_offset, const mat3& projection) override {
    if (!gpu_initialized_) {
      ensure_gpu_initialized();
      if (!gpu_initialized_) return nullptr;  // allocation failed
      cudaError_t err = cudaMalloc(&d_ring_buffer_, ring_size_ * MODEL_FRAME_SIZE);
      if (err != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR cudaMalloc ring buffer: %s\n", cudaGetErrorString(err));
        return nullptr;
      }
      cudaMemset(d_ring_buffer_, 0, ring_size_ * MODEL_FRAME_SIZE);
      d_last_img_ = d_ring_buffer_ + temporal_skip_ * MODEL_FRAME_SIZE;
      fprintf(stderr, "CudaDrivingModelFrame: ring buffer allocated, ring_size=%d\n", ring_size_);
    }

    upload_and_transform(yuv_data, frame_width, frame_height, frame_stride, frame_uv_offset, projection);

    // Shift ring buffer: single memmove shifts all frames down by one slot
    // (replaces sequential loop of temporal_skip_ individual cudaMemcpy calls)
    if (temporal_skip_ > 0) {
      cudaMemcpy(
        d_ring_buffer_,
        d_ring_buffer_ + MODEL_FRAME_SIZE,
        temporal_skip_ * MODEL_FRAME_SIZE, cudaMemcpyDeviceToDevice);
    }

    // Load Y/U/V into interleaved format at last slot
    loadyuv_.queue(d_y_, d_u_, d_v_, d_last_img_);

    // Copy oldest + newest to output
    loadyuv_.copy(d_ring_buffer_, d_output_, 0, 0, MODEL_FRAME_SIZE);
    loadyuv_.copy(d_last_img_, d_output_, 0, MODEL_FRAME_SIZE, MODEL_FRAME_SIZE);

    cudaDeviceSynchronize();
    return d_output_;
  }

private:
  int temporal_skip_;
  int ring_size_;
  uint8_t* d_ring_buffer_ = nullptr;
  uint8_t* d_last_img_ = nullptr;
};

class CudaMonitoringModelFrame : public CudaModelFrame {
public:
  CudaMonitoringModelFrame() {
    const int mw = 1440, mh = 960;
    MODEL_WIDTH = mw;
    MODEL_HEIGHT = mh;
    MODEL_FRAME_SIZE = mw * mh;
    buf_size = MODEL_FRAME_SIZE;

    int max_input_size = 1928 * 1208 * 3 / 2;
    init_cuda(mw, mh, max_input_size, buf_size);
  }

  ~CudaMonitoringModelFrame() {
    deinit_cuda();
  }

  uint8_t* prepare(const uint8_t* yuv_data, int frame_width, int frame_height,
                   int frame_stride, int frame_uv_offset, const mat3& projection) override {
    if (!gpu_initialized_) {
      ensure_gpu_initialized();
      if (!gpu_initialized_) return nullptr;  // allocation failed
    }
    upload_and_transform(yuv_data, frame_width, frame_height, frame_stride, frame_uv_offset, projection);

    cudaMemcpy(d_output_, d_y_, MODEL_FRAME_SIZE, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    return d_output_;
  }
};

#endif  // __JETSON__
