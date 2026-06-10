/***************************************************************************
 *
 * GPU real-space weighted back-projection (Route B) for
 * relion_tomo_reconstruct_tomogram.  See real_backprojection_gpu.h.
 *
 * The kernel mirrors, line for line, the CPU reference:
 *   RealSpaceBackprojection::backproject  (real_backprojection.h:135-197)
 *   Interpolation::linearXY_clip          (interpolation.h)
 * using double arithmetic throughout (as the CPU d4Matrix/d4Vector path does),
 * so the result is numerically equivalent to the CPU volume. Texture-memory /
 * single-precision optimisation is deliberately deferred; correctness first.
 *
 ***************************************************************************/
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "src/jaz/cuda/real_backprojection_gpu.h"

#define RLN_CUDA_CHECK(call) do {                                              \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        fprintf(stderr, "ERROR: CUDA failure %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));                   \
        exit(1);                                                               \
    }                                                                          \
} while (0)


__global__ void wbp_backproject_kernel(
        const float*  __restrict__ stack,
        const double* __restrict__ projRows,
        int fc, int W, int H,
        int outX, int outY, int zCount, int zBase,
        double ox, double oy, double oz, double spacing,
        float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;   // slab-local z in [0,zCount)
    if (x >= outX || y >= outY || z >= zCount) return;

    // pw = origin + (x,y, zBase+z) * spacing   (w = 1); zBase = global z of this slab
    const double pwx = ox + x * spacing;
    const double pwy = oy + y * spacing;
    const double pwz = oz + (zBase + z) * spacing;

    double sum = 0.0;
    double wgh = 0.0;

    for (int f = 0; f < fc; ++f)
    {
        const double* P = projRows + (size_t) f * 8;
        // pi = proj[f] * pw  (only the x,y components are needed)
        const double pix = P[0] * pwx + P[1] * pwy + P[2] * pwz + P[3];
        const double piy = P[4] * pwx + P[5] * pwy + P[6] * pwz + P[7];

        if (pix >= 0.0 && pix < W && piy >= 0.0 && piy < H)
        {
            // Interpolation::linearXY_clip
            int x0 = (int) floor(pix);
            int y0 = (int) floor(piy);
            const double xf = pix - x0;
            const double yf = piy - y0;
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            if (x0 < 0) x0 = 0;  if (x0 >= W) x0 = W - 1;
            if (x1 < 0) x1 = 0;  if (x1 >= W) x1 = W - 1;
            if (y0 < 0) y0 = 0;  if (y0 >= H) y0 = H - 1;
            if (y1 < 0) y1 = 0;  if (y1 >= H) y1 = H - 1;

            const float* S = stack + (size_t) f * H * W;
            const double vx0 = (1.0 - xf) * S[(size_t) y0 * W + x0] + xf * S[(size_t) y0 * W + x1];
            const double vx1 = (1.0 - xf) * S[(size_t) y1 * W + x0] + xf * S[(size_t) y1 * W + x1];
            sum += (1.0 - yf) * vx0 + yf * vx1;
            wgh += 1.0;
        }
    }

    // dest(x,y,z) = sum/wgh  (out pre-zeroed via cudaMemset; matches CPU += on a
    // zero-initialised destination, leaving wgh==0 voxels at 0)
    if (wgh > 0.0)
        out[((size_t) z * outY + y) * outX + x] = (float) (sum / wgh);
}


// Fast variant: single precision + a layered 2D texture with hardware bilinear filtering
// (one tex fetch replaces 4 global loads + manual interpolation). addressMode=Clamp
// reproduces linearXY_clip's edge clamping; the in-bounds test on pi is done BEFORE the
// fetch (as the CPU does). Equivalent to the CPU only to ~routeB FSC tolerance (texture
// filtering uses 9-bit fractional weights and accumulation is in float).
__global__ void wbp_backproject_kernel_tex(
        cudaTextureObject_t tilts,
        const float* __restrict__ projRows,
        int fc, int W, int H,
        int outX, int outY, int zCount, int zBase,
        float ox, float oy, float oz, float spacing,
        float* __restrict__ out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= outX || y >= outY || z >= zCount) return;

    const float pwx = ox + x * spacing;
    const float pwy = oy + y * spacing;
    const float pwz = oz + (zBase + z) * spacing;

    float sum = 0.0f, wgh = 0.0f;
    for (int f = 0; f < fc; ++f)
    {
        const float* P = projRows + (size_t) f * 8;
        const float pix = P[0] * pwx + P[1] * pwy + P[2] * pwz + P[3];
        const float piy = P[4] * pwx + P[5] * pwy + P[6] * pwz + P[7];
        if (pix >= 0.0f && pix < W && piy >= 0.0f && piy < H)
        {
            // +0.5 -> texel-centre addressing for cudaFilterModeLinear
            sum += tex2DLayered<float>(tilts, pix + 0.5f, piy + 0.5f, f);
            wgh += 1.0f;
        }
    }
    if (wgh > 0.0f)
        out[((size_t) z * outY + y) * outX + x] = sum / wgh;
}


void wbpBackprojectGPU(
        const float*  stack,
        const double* projRows,
        int fc, int W, int H,
        int outX, int outY, int outZ,
        double ox, double oy, double oz, double spacing,
        float* out,
        int tileZ,
        bool fast,
        int device)
{
    if (device >= 0) RLN_CUDA_CHECK(cudaSetDevice(device));

    const size_t stackN = (size_t) fc * H * W;
    const size_t projN  = (size_t) fc * 8;

    // Output tiled in z; the tilt stack stays resident, so volumes > VRAM still fit
    // (one z-slab at a time). tileZ<=0 or >=outZ means a single slab (whole volume).
    const int tz = (tileZ > 0 && tileZ < outZ) ? tileZ : outZ;
    const size_t slabMax = (size_t) outX * outY * tz;
    const dim3 block(8, 8, 8);

    float* d_out = nullptr;
    RLN_CUDA_CHECK(cudaMalloc(&d_out, slabMax * sizeof(float)));

    if (!fast)
    {
        // ---- exact: double precision, global-memory gather ----
        float*  d_stack = nullptr;
        double* d_proj  = nullptr;
        RLN_CUDA_CHECK(cudaMalloc(&d_stack, stackN * sizeof(float)));
        RLN_CUDA_CHECK(cudaMalloc(&d_proj,  projN  * sizeof(double)));
        RLN_CUDA_CHECK(cudaMemcpy(d_stack, stack,    stackN * sizeof(float),  cudaMemcpyHostToDevice));
        RLN_CUDA_CHECK(cudaMemcpy(d_proj,  projRows, projN  * sizeof(double), cudaMemcpyHostToDevice));

        for (int z0 = 0; z0 < outZ; z0 += tz)
        {
            const int zc = (outZ - z0 < tz) ? (outZ - z0) : tz;
            const size_t slabN = (size_t) outX * outY * zc;
            RLN_CUDA_CHECK(cudaMemset(d_out, 0, slabN * sizeof(float)));
            const dim3 grid((outX+block.x-1)/block.x, (outY+block.y-1)/block.y, (zc+block.z-1)/block.z);
            wbp_backproject_kernel<<<grid, block>>>(
                    d_stack, d_proj, fc, W, H, outX, outY, zc, z0, ox, oy, oz, spacing, d_out);
            RLN_CUDA_CHECK(cudaGetLastError());
            RLN_CUDA_CHECK(cudaDeviceSynchronize());
            RLN_CUDA_CHECK(cudaMemcpy(out + (size_t) z0 * outX * outY, d_out,
                                      slabN * sizeof(float), cudaMemcpyDeviceToHost));
        }
        cudaFree(d_stack);
        cudaFree(d_proj);
    }
    else
    {
        // ---- fast: single precision, layered-texture hardware bilinear ----
        std::vector<float> hproj(projN);
        for (size_t i = 0; i < projN; ++i) hproj[i] = (float) projRows[i];
        float* d_proj = nullptr;
        RLN_CUDA_CHECK(cudaMalloc(&d_proj, projN * sizeof(float)));
        RLN_CUDA_CHECK(cudaMemcpy(d_proj, hproj.data(), projN * sizeof(float), cudaMemcpyHostToDevice));

        cudaArray_t arr = nullptr;
        cudaChannelFormatDesc ch = cudaCreateChannelDesc<float>();
        RLN_CUDA_CHECK(cudaMalloc3DArray(&arr, &ch, make_cudaExtent(W, H, fc), cudaArrayLayered));
        cudaMemcpy3DParms cp = {};
        cp.srcPtr   = make_cudaPitchedPtr((void*) stack, W * sizeof(float), W, H);
        cp.dstArray = arr;
        cp.extent   = make_cudaExtent(W, H, fc);
        cp.kind     = cudaMemcpyHostToDevice;
        RLN_CUDA_CHECK(cudaMemcpy3D(&cp));

        cudaResourceDesc rd = {}; rd.resType = cudaResourceTypeArray; rd.res.array.array = arr;
        cudaTextureDesc td = {};
        td.addressMode[0] = cudaAddressModeClamp;   // match linearXY_clip edge clamping
        td.addressMode[1] = cudaAddressModeClamp;
        td.filterMode     = cudaFilterModeLinear;   // hardware bilinear
        td.readMode       = cudaReadModeElementType;
        td.normalizedCoords = 0;
        cudaTextureObject_t tex = 0;
        RLN_CUDA_CHECK(cudaCreateTextureObject(&tex, &rd, &td, nullptr));

        for (int z0 = 0; z0 < outZ; z0 += tz)
        {
            const int zc = (outZ - z0 < tz) ? (outZ - z0) : tz;
            const size_t slabN = (size_t) outX * outY * zc;
            RLN_CUDA_CHECK(cudaMemset(d_out, 0, slabN * sizeof(float)));
            const dim3 grid((outX+block.x-1)/block.x, (outY+block.y-1)/block.y, (zc+block.z-1)/block.z);
            wbp_backproject_kernel_tex<<<grid, block>>>(
                    tex, d_proj, fc, W, H, outX, outY, zc, z0,
                    (float) ox, (float) oy, (float) oz, (float) spacing, d_out);
            RLN_CUDA_CHECK(cudaGetLastError());
            RLN_CUDA_CHECK(cudaDeviceSynchronize());
            RLN_CUDA_CHECK(cudaMemcpy(out + (size_t) z0 * outX * outY, d_out,
                                      slabN * sizeof(float), cudaMemcpyDeviceToHost));
        }
        cudaDestroyTextureObject(tex);
        cudaFreeArray(arr);
        cudaFree(d_proj);
    }

    cudaFree(d_out);
}
