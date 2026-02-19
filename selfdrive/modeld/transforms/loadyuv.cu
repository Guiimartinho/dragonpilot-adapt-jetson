/**
 * CUDA YUV loading kernels for Jetson AGX Xavier.
 * Replaces loadyuv.cl OpenCL kernels to eliminate OpenCL→CUDA interop overhead.
 * Same pixel layout as the OpenCL version for compatibility.
 */

#include <cuda_runtime.h>
#include <cstdint>

__global__ void loadys_kernel(
    const uint8_t* __restrict__ Y,
    uint8_t* __restrict__ out,
    int out_offset,
    int transformed_width,
    int uv_size)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ois = gid * 8;
    const int oy = ois / transformed_width;
    const int ox = ois % transformed_width;

    // Load 8 bytes at once
    const uint8_t y0 = Y[gid * 8 + 0];
    const uint8_t y1 = Y[gid * 8 + 1];
    const uint8_t y2 = Y[gid * 8 + 2];
    const uint8_t y3 = Y[gid * 8 + 3];
    const uint8_t y4 = Y[gid * 8 + 4];
    const uint8_t y5 = Y[gid * 8 + 5];
    const uint8_t y6 = Y[gid * 8 + 6];
    const uint8_t y7 = Y[gid * 8 + 7];

    // Layout: 02 / 13 (even/odd rows go to different output planes)
    uint8_t* outy0;
    uint8_t* outy1;
    if ((oy & 1) == 0) {
        outy0 = out + out_offset;                    // y0
        outy1 = out + out_offset + uv_size * 2;     // y2
    } else {
        outy0 = out + out_offset + uv_size;          // y1
        outy1 = out + out_offset + uv_size * 3;     // y3
    }

    int base = (oy / 2) * (transformed_width / 2) + ox / 2;
    outy0[base + 0] = y0;  // s0
    outy0[base + 1] = y2;  // s2
    outy0[base + 2] = y4;  // s4
    outy0[base + 3] = y6;  // s6

    outy1[base + 0] = y1;  // s1
    outy1[base + 1] = y3;  // s3
    outy1[base + 2] = y5;  // s5
    outy1[base + 3] = y7;  // s7
}

__global__ void loaduv_kernel(
    const uint8_t* __restrict__ in,
    uint8_t* __restrict__ out,
    int out_offset,
    int total_bytes)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int byte_offset = gid * 8;
    if (byte_offset >= total_bytes) return;

    if (byte_offset + 8 <= total_bytes) {
        // Full 8-byte aligned copy
        const uint64_t* src = (const uint64_t*)(in + byte_offset);
        uint64_t* dst = (uint64_t*)(out + out_offset + byte_offset);
        *dst = *src;
    } else {
        // Handle remainder bytes (last <8 bytes)
        for (int i = byte_offset; i < total_bytes; i++) {
            out[out_offset + i] = in[i];
        }
    }
}

__global__ void copy_kernel(
    const uint8_t* __restrict__ in,
    uint8_t* __restrict__ out,
    int in_offset,
    int out_offset,
    int total_bytes)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int byte_offset = gid * 8;
    if (byte_offset >= total_bytes) return;

    if (byte_offset + 8 <= total_bytes) {
        // Full 8-byte aligned copy
        const uint64_t* src = (const uint64_t*)(in + in_offset + byte_offset);
        uint64_t* dst = (uint64_t*)(out + out_offset + byte_offset);
        *dst = *src;
    } else {
        // Handle remainder bytes (last <8 bytes)
        for (int i = byte_offset; i < total_bytes; i++) {
            out[out_offset + i] = in[in_offset + i];
        }
    }
}

extern "C" {

void cuda_loadys(const uint8_t* Y, uint8_t* out, int out_offset,
                 int transformed_width, int transformed_height, cudaStream_t stream) {
    int uv_size = (transformed_width / 2) * (transformed_height / 2);
    int total_elements = (transformed_width * transformed_height) / 8;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    loadys_kernel<<<blocks, threads, 0, stream>>>(Y, out, out_offset, transformed_width, uv_size);
}

void cuda_loaduv(const uint8_t* in, uint8_t* out, int out_offset,
                 int total_bytes, cudaStream_t stream) {
    int total_elements = (total_bytes + 7) / 8;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    loaduv_kernel<<<blocks, threads, 0, stream>>>(in, out, out_offset, total_bytes);
}

void cuda_copy(const uint8_t* in, uint8_t* out, int in_offset, int out_offset,
               int total_bytes, cudaStream_t stream) {
    int total_elements = (total_bytes + 7) / 8;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    copy_kernel<<<blocks, threads, 0, stream>>>(in, out, in_offset, out_offset, total_bytes);
}

}  // extern "C"
