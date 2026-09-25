#ifndef CUDA_COARSE_GEMM_CUH_
#define CUDA_COARSE_GEMM_CUH_

/*
 * The coarse pass of a global search as one matrix product per particle (opt-in).
 *
 * For every orientation o and translation t, cuda_kernel_diff2_coarse adds
 *     sum_p c_p |r_op - x_tp|^2,   c_p = corr_p / 2,
 * to the particle's weights, with r_op the reference projected at orientation o and x_tp the image
 * shifted by translation t. Expanded, that is
 *     sum_p c_p |r_op|^2  -  2 sum_p c_p (Re r_op Re x_tp + Im r_op Im x_tp)  +  sum_p c_p |x_tp|^2,
 * the product of
 *     R[o] = [ Re r_o | Im r_o | |r_o|^2 | 1 ]                         (3P + 1 values)
 *     Q[t] = [ -2 c Re x_t | -2 c Im x_t | c | sum_p c_p |x_tp|^2 ]   (3P + 1 values).
 * In a global search every particle uses the same orientations, so R is built once per class and
 * image size per iteration, and each particle only builds Q (T columns) and runs one GEMM that adds
 * Q^T R into the weights in their own layout (o * T + t).
 *
 * The sums are reordered, so the scores agree with the kernel's to float rounding, not bit for bit
 * (the kernel's own float atomics already make it run-to-run nondeterministic in the same way).
 *
 * RELION_COARSE_GEMM=1 enables it for global 3D searches of 2D images, one image per particle, no
 * cross-correlation (1: fp32; 2: TF32 tensor cores). With RELION_COARSE_GEMM_CHECK=<n> as well, every
 * n-th eligible image and class is scored both ways, the largest difference is printed, and the
 * kernel's scores are kept for that one; all others use the GEMM.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <atomic>
#include <mutex>
#include <unistd.h>
#include "src/acc/cuda/cuda_settings.h"
#include "src/acc/cuda/cuda_kernels/cuda_device_utils.cuh"
#include "src/acc/acc_projectorkernel_impl.h"

#define COARSE_GEMM_BLOCK 128

namespace CoarseGemm
{

// Settings, read once per process.
inline int mode()
{
	static const int m = getenv("RELION_COARSE_GEMM") ? atoi(getenv("RELION_COARSE_GEMM")) : 0;
	return m;
}

inline int checkCount()
{
	static const int n = getenv("RELION_COARSE_GEMM_CHECK") ? atoi(getenv("RELION_COARSE_GEMM_CHECK")) : 0;
	return n;
}

// R for one class and image size: row o (3P + 1 floats) = [Re r_o | Im r_o | |r_o|^2 | 1].
// One block per orientation; pixels indexed as in cuda_kernel_diff2_coarse.
static __global__ void cuda_kernel_coarse_gemm_R(
		XFLOAT *g_eulers,
		AccProjectorKernel projector,
		XFLOAT *R,
		int image_size,
		size_t ldr)
{
	__shared__ XFLOAT e[9];
	const size_t o = blockIdx.x;
	if (threadIdx.x < 9)
		e[threadIdx.x] = g_eulers[o * 9 + threadIdx.x];
	__syncthreads();

	XFLOAT *row = R + o * ldr;
	for (int p = threadIdx.x; p < image_size; p += blockDim.x)
	{
		int x = p % projector.imgX;
		int y = floorfracf(p, projector.imgX);
		if (y > projector.maxR)
			y -= projector.imgY;
		XFLOAT re, im;
		projector.project3Dmodel(x, y, e[0], e[1], e[3], e[4], e[6], e[7], re, im);
		row[p] = re;
		row[image_size + p] = im;
		row[2 * image_size + p] = re * re + im * im;
	}
	if (threadIdx.x == 0)
		row[3 * image_size] = (XFLOAT) 1.;
}

// Q for one particle: column t (3P + 1 floats) = [-2 c Re x_t | -2 c Im x_t | c | sum_p c_p |x_tp|^2],
// with x_t shifted by the kernel's own translatePixel. One block per translation.
static __global__ void cuda_kernel_coarse_gemm_Q(
		XFLOAT *trans_x,
		XFLOAT *trans_y,
		XFLOAT *img_re,
		XFLOAT *img_im,
		XFLOAT *corr,
		AccProjectorKernel projector,
		XFLOAT *Q,
		int image_size,
		size_t ldq)
{
	__shared__ XFLOAT s_sum[COARSE_GEMM_BLOCK];
	const size_t t = blockIdx.x;
	const XFLOAT tx = trans_x[t], ty = trans_y[t];
	XFLOAT *col = Q + t * ldq;
	XFLOAT part = 0.;
	for (int p = threadIdx.x; p < image_size; p += blockDim.x)
	{
		int x = p % projector.imgX;
		int y = floorfracf(p, projector.imgX);
		if (y > projector.maxR)
			y -= projector.imgY;
		XFLOAT re, im;
		translatePixel(x, y, tx, ty, img_re[p], img_im[p], re, im);
		const XFLOAT c = corr[p] / 2;
		col[p] = -2 * c * re;
		col[image_size + p] = -2 * c * im;
		col[2 * image_size + p] = c;
		part += c * (re * re + im * im);
	}
	s_sum[threadIdx.x] = part;
	__syncthreads();
	for (int s = COARSE_GEMM_BLOCK / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
			s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
		__syncthreads();
	}
	if (threadIdx.x == 0)
		col[3 * image_size] = s_sum[0];
}

// Counts, printed by each process at exit when the GEMM path is enabled.
struct Totals
{
	std::atomic<long long> gemm{0}, kernel{0}, checked{0}, seen{0};
	~Totals()
	{
		if (mode() > 0)
			printf("coarse-gemm pid %d: %lld image-class scorings by GEMM (mode %d), %lld by the kernel, %lld checked\n",
			       (int) getpid(), (long long) gemm, mode(), (long long) kernel, (long long) checked);
		fflush(stdout);
	}
};
static Totals totals;

inline void handleCublas(cublasStatus_t s, const char *what)
{
	if (s != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "ERROR: cuBLAS %s failed with status %d\n", what, (int) s);
		exit(EXIT_FAILURE);
	}
}

} // namespace CoarseGemm

#endif
