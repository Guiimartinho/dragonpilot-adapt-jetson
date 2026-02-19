/**
 * CUDA warp perspective kernel for Jetson AGX Xavier.
 * Replaces transform.cl OpenCL kernel to eliminate OpenCL→CUDA interop overhead.
 * Bilinear interpolation with sub-pixel accuracy (same algorithm as OpenCL version).
 */

#include <cuda_runtime.h>
#include <cstdint>

#define INTER_BITS 5
#define INTER_TAB_SIZE (1 << INTER_BITS)
#define INTER_SCALE (1.f / INTER_TAB_SIZE)

#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << INTER_REMAP_COEF_BITS)

__global__ void warpPerspectiveKernel(
    const uint8_t* __restrict__ src,
    int src_row_stride, int src_px_stride, int src_offset, int src_rows, int src_cols,
    uint8_t* __restrict__ dst,
    int dst_row_stride, int dst_offset, int dst_rows, int dst_cols,
    const float* __restrict__ M)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < dst_cols && dy < dst_rows) {
        float X0 = M[0] * dx + M[1] * dy + M[2];
        float Y0 = M[3] * dx + M[4] * dy + M[5];
        float W  = M[6] * dx + M[7] * dy + M[8];
        W = (W != 0.0f) ? (float)INTER_TAB_SIZE / W : 0.0f;
        int X = __float2int_rn(X0 * W);
        int Y = __float2int_rn(Y0 * W);

        int sx = X >> INTER_BITS;
        int sy = Y >> INTER_BITS;

        // Clamp coordinates
        int sx_clamp    = min(max(sx, 0), src_cols - 1);
        int sx_p1_clamp = min(max(sx + 1, 0), src_cols - 1);
        int sy_clamp    = min(max(sy, 0), src_rows - 1);
        int sy_p1_clamp = min(max(sy + 1, 0), src_rows - 1);

        // Use __ldg() for read-only data cache path (optimized for scattered reads on Volta sm_72)
        int v0 = __ldg(&src[sy_clamp    * src_row_stride + src_offset + sx_clamp    * src_px_stride]);
        int v1 = __ldg(&src[sy_clamp    * src_row_stride + src_offset + sx_p1_clamp * src_px_stride]);
        int v2 = __ldg(&src[sy_p1_clamp * src_row_stride + src_offset + sx_clamp    * src_px_stride]);
        int v3 = __ldg(&src[sy_p1_clamp * src_row_stride + src_offset + sx_p1_clamp * src_px_stride]);

        int ay = Y & (INTER_TAB_SIZE - 1);
        int ax = X & (INTER_TAB_SIZE - 1);
        float taby = INTER_SCALE * ay;
        float tabx = INTER_SCALE * ax;

        int itab0 = __float2int_rn((1.0f - taby) * (1.0f - tabx) * INTER_REMAP_COEF_SCALE);
        int itab1 = __float2int_rn((1.0f - taby) * tabx * INTER_REMAP_COEF_SCALE);
        int itab2 = __float2int_rn(taby * (1.0f - tabx) * INTER_REMAP_COEF_SCALE);
        int itab3 = __float2int_rn(taby * tabx * INTER_REMAP_COEF_SCALE);

        int val = v0 * itab0 + v1 * itab1 + v2 * itab2 + v3 * itab3;
        val = (val + (1 << (INTER_REMAP_COEF_BITS - 1))) >> INTER_REMAP_COEF_BITS;

        dst[dy * dst_row_stride + dst_offset + dx] = (uint8_t)min(max(val, 0), 255);
    }
}

extern "C" {

void cuda_warp_perspective(
    const uint8_t* src, int src_row_stride, int src_px_stride, int src_offset, int src_rows, int src_cols,
    uint8_t* dst, int dst_row_stride, int dst_offset, int dst_rows, int dst_cols,
    const float* M, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dst_cols + block.x - 1) / block.x, (dst_rows + block.y - 1) / block.y);

    warpPerspectiveKernel<<<grid, block, 0, stream>>>(
        src, src_row_stride, src_px_stride, src_offset, src_rows, src_cols,
        dst, dst_row_stride, dst_offset, dst_rows, dst_cols,
        M);
}

}  // extern "C"
