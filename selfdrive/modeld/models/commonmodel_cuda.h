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
  CudaModelFrame() {
    cudaStreamCreate(&stream_);
  }

  virtual ~CudaModelFrame() {
    cudaStreamDestroy(stream_);
  }

  /**
   * Prepare frame from host memory (VisionIPC shared memory).
   * Uploads to GPU, runs transform and loadyuv, returns device pointer.
   * The returned pointer is valid until the next prepare() call.
   */
  virtual uint8_t* prepare(const uint8_t* yuv_data, int frame_width, int frame_height,
                           int frame_stride, int frame_uv_offset, const mat3& projection) = 0;

  /**
   * Get the output buffer as host memory (for compatibility with existing pipeline).
   * Copies from device to host. Use get_device_ptr() to avoid this copy.
   */
  uint8_t* get_host_buffer(int buffer_size) {
    if (!h_output_) {
      h_output_.reset(new uint8_t[buffer_size]);
    }
    cudaMemcpyAsync(h_output_.get(), d_output_, buffer_size, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    return h_output_.get();
  }

  /** Get device pointer directly (for tinygrad CUDA backend - zero copy). */
  uint8_t* get_device_ptr() const { return d_output_; }

  cudaStream_t get_stream() const { return stream_; }

  int MODEL_WIDTH;
  int MODEL_HEIGHT;
  int MODEL_FRAME_SIZE;
  int buf_size;

protected:
  cudaStream_t stream_;
  uint8_t* d_input_ = nullptr;    // Device: raw YUV input
  uint8_t* d_y_ = nullptr;        // Device: transformed Y plane
  uint8_t* d_u_ = nullptr;        // Device: transformed U plane
  uint8_t* d_v_ = nullptr;        // Device: transformed V plane
  uint8_t* d_output_ = nullptr;   // Device: final output (loadyuv result)
  std::unique_ptr<uint8_t[]> h_output_;  // Host: output (for compatibility)

  CudaTransform transform_;
  CudaLoadYUVState loadyuv_;

  void init_cuda(int model_width, int model_height, int input_size, int output_size) {
    MODEL_WIDTH = model_width;
    MODEL_HEIGHT = model_height;

    cudaMalloc(&d_input_, input_size);
    cudaMalloc(&d_y_, model_width * model_height);
    cudaMalloc(&d_u_, (model_width / 2) * (model_height / 2));
    cudaMalloc(&d_v_, (model_width / 2) * (model_height / 2));
    cudaMalloc(&d_output_, output_size);

    transform_.init(stream_);
    loadyuv_.init(model_width, model_height, stream_);
  }

  void deinit_cuda() {
    loadyuv_.destroy();
    transform_.destroy();
    cudaFree(d_output_);
    cudaFree(d_v_);
    cudaFree(d_u_);
    cudaFree(d_y_);
    cudaFree(d_input_);
  }

  void upload_and_transform(const uint8_t* yuv_data, int frame_width, int frame_height,
                            int frame_stride, int frame_uv_offset, const mat3& projection) {
    int input_size = frame_stride * frame_height * 3 / 2;
    cudaMemcpyAsync(d_input_, yuv_data, input_size, cudaMemcpyHostToDevice, stream_);

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

    // Allocate for largest expected input (1928x1208 HEVC from replay)
    int max_input_size = 1928 * 1208 * 3 / 2;
    init_cuda(mw, mh, max_input_size, buf_size);

    // Additional buffers for temporal frames
    cudaMalloc(&d_img_buffer_20hz_, MODEL_FRAME_SIZE);
    cudaMalloc(&d_last_img_, MODEL_FRAME_SIZE);
    frame_count_ = 0;
  }

  ~CudaDrivingModelFrame() {
    cudaFree(d_last_img_);
    cudaFree(d_img_buffer_20hz_);
    deinit_cuda();
  }

  uint8_t* prepare(const uint8_t* yuv_data, int frame_width, int frame_height,
                   int frame_stride, int frame_uv_offset, const mat3& projection) override {
    upload_and_transform(yuv_data, frame_width, frame_height, frame_stride, frame_uv_offset, projection);

    // Load Y/U/V into interleaved format
    loadyuv_.queue(d_y_, d_u_, d_v_, d_img_buffer_20hz_);

    // Temporal: keep current and last frame
    if (frame_count_ % temporal_skip_ == 0) {
      // Copy current to output[0:frame_size]
      loadyuv_.copy(d_img_buffer_20hz_, d_output_, 0, 0, MODEL_FRAME_SIZE);
      // Copy last to output[frame_size:2*frame_size]
      loadyuv_.copy(d_last_img_, d_output_, 0, MODEL_FRAME_SIZE, MODEL_FRAME_SIZE);
      // Save current as last
      cudaMemcpyAsync(d_last_img_, d_img_buffer_20hz_, MODEL_FRAME_SIZE, cudaMemcpyDeviceToDevice, stream_);
    }

    frame_count_++;
    cudaStreamSynchronize(stream_);
    return d_output_;
  }

private:
  int temporal_skip_;
  int frame_count_;
  uint8_t* d_img_buffer_20hz_;
  uint8_t* d_last_img_;
};

class CudaMonitoringModelFrame : public CudaModelFrame {
public:
  CudaMonitoringModelFrame() {
    const int mw = 1440, mh = 960;
    MODEL_WIDTH = mw;
    MODEL_HEIGHT = mh;
    MODEL_FRAME_SIZE = mw * mh;  // Y plane only for monitoring
    buf_size = MODEL_FRAME_SIZE;

    int max_input_size = 1928 * 1208 * 3 / 2;
    init_cuda(mw, mh, max_input_size, buf_size);
  }

  ~CudaMonitoringModelFrame() {
    deinit_cuda();
  }

  uint8_t* prepare(const uint8_t* yuv_data, int frame_width, int frame_height,
                   int frame_stride, int frame_uv_offset, const mat3& projection) override {
    upload_and_transform(yuv_data, frame_width, frame_height, frame_stride, frame_uv_offset, projection);

    // Monitoring model uses only Y plane
    cudaMemcpyAsync(d_output_, d_y_, MODEL_FRAME_SIZE, cudaMemcpyDeviceToDevice, stream_);
    cudaStreamSynchronize(stream_);
    return d_output_;
  }
};

#endif  // __JETSON__
