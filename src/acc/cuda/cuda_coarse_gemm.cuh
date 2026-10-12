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
 *     R[o] = [ Re r_o | Im r_o | |r_o|^2 | 1 | 1 ]                        (3P + 2 values)
 *     Q[t] = [ -2 c Re x_t | -2 c Im x_t | c | s_t hi | s_t lo ]         (3P + 2 values),
 * with s_t = sum_p c_p |x_tp|^2 split into a part with only TF32's 10 mantissa bits (hi) and the rest
 * (lo). s_t is most of each score, so in TF32 mode a single rounding of it would offset all of a
 * translation's scores by up to 1 part in 2^11; split, both parts pass through TF32 almost exactly.
 * In a global search every particle uses the same orientations, so R is built once per class and
 * image size per iteration, and each particle only builds Q (T columns) and runs one GEMM that adds
 * Q^T R into the weights in their own layout (o * T + t).
 *
 * The sums are reordered, so the scores agree with the kernel's to float rounding, not bit for bit
 * (the kernel's own float atomics already make it run-to-run nondeterministic in the same way).
 *
 * RELION_COARSE_GEMM=1 enables it for global 3D searches of 2D images, one image per particle, no
 * cross-correlation (1: fp32; 2: TF32 tensor cores; 3: fp16 tensor cores, 4: bf16 tensor cores, both with
 * fp32 data and fp32 accumulation, cuBLAS rounding the inputs to 16 bits inside the product). fp16 keeps
 * TF32's 10 mantissa bits but has a range of only 6.5e4, so large R and Q entries become infinite; bf16
 * keeps fp32's range with 7 mantissa bits. bf16 needs sm_80 or newer; older GPUs run mode 4 as mode 1.
 *
 * RELION_COARSE_GEMM_BATCH=<B> (B >= 2, with the GEMM on) scores B particles of one thread in one GEMM:
 * each particle's translation columns side by side in Q, the product into a scratch matrix, and each
 * particle's block added into its own weights (accDoExpectationParticleBatch). Threads then take tasks of
 * B particles.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mutex>
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

inline int batch()
{
	static const int b = getenv("RELION_COARSE_GEMM_BATCH") ? atoi(getenv("RELION_COARSE_GEMM_BATCH")) : 1;
	return b;
}

// Columns of R and Q (K of the product).
__host__ __device__ inline size_t kdim(size_t image_size)
{
	return 3 * image_size + 2;
}

// R for one class and image size: row o (3P + 2 floats) = [Re r_o | Im r_o | |r_o|^2 | 1 | 1].
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
	{
		row[3 * image_size] = (XFLOAT) 1.;
		row[3 * image_size + 1] = (XFLOAT) 1.;
	}
}

// Q for one particle: column t (3P + 2 floats) = [-2 c Re x_t | -2 c Im x_t | c | s_t hi | s_t lo],
// s_t = sum_p c_p |x_tp|^2,
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
	{
		const float s = s_sum[0];
		const float hi = __uint_as_float(__float_as_uint(s) & 0xffffe000u);  // exact in TF32
		col[3 * image_size] = hi;
		col[3 * image_size + 1] = s - hi;  // exact in float
	}
}

// The product of a batch, C (column-major, nP * T rows by O columns: particle p's scores are rows
// p * T ... p * T + T - 1), added into each particle's weights in the kernel's layout: dst[p][o * T + t].
// Grid (O, number of particles), particles p0, p0 + 1, ...
static __global__ void cuda_kernel_coarse_gemm_scatter(
		const XFLOAT *C,
		XFLOAT **dst,
		int nP,
		int T,
		int p0)
{
	const size_t o = blockIdx.x;
	const int p = p0 + blockIdx.y;
	const size_t ldc = (size_t) nP * T;
	XFLOAT *d = dst[blockIdx.y];
	for (int t = threadIdx.x; t < T; t += blockDim.x)
		d[o * T + t] += C[o * ldc + (size_t) p * T + t];
}

inline void handleCublas(cublasStatus_t s, const char *what)
{
	if (s != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "ERROR: cuBLAS %s failed with status %d\n", what, (int) s);
		exit(EXIT_FAILURE);
	}
}

// The GEMM in the selected precision: C = alpha op(A) op(B) + beta C, all fp32 in memory.
// The mode in effect: bf16 (4) needs sm_80 or newer and runs as fp32 (1) on older GPUs.
inline int effectiveMode()
{
	static const int eff = [] {
		int m_ = mode();
		if (m_ == 4)
		{
			int dev = 0, major = 0;
			cudaGetDevice(&dev);
			cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
			if (major < 8)
			{
				printf("coarse-gemm: bf16 needs sm_80 or newer; this GPU runs the GEMM in fp32 (mode 1)\n");
				return 1;
			}
		}
		return m_;
	}();
	return eff;
}
inline bool sixteenBit() { return effectiveMode() == 3 || effectiveMode() == 4; }
// The rank-1 split (cross term K = 2P, |r|^2 c and s in fp32 afterwards): always for 16-bit, opt-in for fp32 and
// tf32 with RELION_COARSE_GEMM_SPLIT=1 (then without scaling).
inline bool split()
{
	static const bool s = sixteenBit() ||
		(getenv("RELION_COARSE_GEMM_SPLIT") != NULL && atoi(getenv("RELION_COARSE_GEMM_SPLIT")) > 0);
	return s;
}

// The GEMM in the selected precision: C = alpha op(A) op(B) + beta C, all fp32 in memory.
inline cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
		const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
	if (sixteenBit())
		return cublasGemmEx(h, ta, tb, m, n, k, alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, beta, C, CUDA_R_32F, ldc,
				effectiveMode() == 3 ? CUBLAS_COMPUTE_32F_FAST_16F : CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	return cublasSgemm(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

/*
 * 16-bit modes (3, 4). The 16-bit product takes only the cross term: rows 0..2P-1 of R and Q
 * (Re and Im), K = 2P. The two rank-1 terms stay in fp32 and are added afterwards: sum_p c_p |r_op|^2
 * depends only on the orientation (for a particle), so it is V = R[2P:3P]^T c, one fp32 GEMM with K = P
 * whose columns are the particles; s_t (Q rows 3P, 3P+1) depends only on the translation. Each operand's
 * cross rows are scaled by one power of two chosen from its own maximum in the same pass (R when it is
 * built, Q per call), leaving a bit of headroom below fp16's 65504; the GEMM's alpha (a device value)
 * undoes both exactly. A non-finite score after this is a bug, so the add kernel stops the run.
 */
// Largest |M[c*ld + r]| over r < rows, c < cols, into *out (float bits; *out zeroed by the caller).
static __global__ void cuda_kernel_coarse_gemm_maxabs(const XFLOAT *M, size_t ld, size_t rows, size_t cols, unsigned int *out)
{
	float m = 0.f;
	for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < rows * cols; i += (size_t) gridDim.x * blockDim.x)
		m = fmaxf(m, fabsf(M[(i / rows) * ld + i % rows]));
	for (int off = 16; off > 0; off >>= 1)
		m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
	if ((threadIdx.x & 31) == 0)
		atomicMax(out, __float_as_uint(m));
}
// The power of two that brings max to at most 2^14.
__device__ inline float pow2Scale(unsigned int maxbits)
{
	const float mx = __uint_as_float(maxbits);
	if (!(mx > 0.f) || isinf(mx))
		return 1.f;
	int e;
	frexpf(16384.f / mx, &e);
	return ldexpf(1.f, e - 1);
}
// M[c*ld + r] *= scale for r < rows; the scale also into *scaleOut.
static __global__ void cuda_kernel_coarse_gemm_scale(XFLOAT *M, size_t ld, size_t rows, size_t cols,
		const unsigned int *maxbits, float *scaleOut)
{
	const float a = pow2Scale(*maxbits);
	if (blockIdx.x == 0 && threadIdx.x == 0 && scaleOut != NULL)
		*scaleOut = a;
	for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < rows * cols; i += (size_t) gridDim.x * blockDim.x)
		M[(i / rows) * ld + i % rows] *= a;
}
// v[0] = 1 / (aR * aQ) (exact: powers of two), v[1] = 1, v[2] = 0: the GEMM's alpha and betas.
static __global__ void cuda_kernel_coarse_gemm_alpha(const float *aR, const float *aQ, float *v)
{
	v[0] = (aR != NULL && aQ != NULL) ? 1.f / (*aR * *aQ) : 1.f;   // NULL: unscaled
	v[1] = 1.f;
	v[2] = 0.f;
}
// C[o*ldc + i] += V[o + (i / T) * O] + s_i (s_i = Q[i*ld + 3P] + Q[i*ld + 3P + 1]); i < nT, o < O. Stops the run on
// a non-finite score.
static __global__ void cuda_kernel_coarse_gemm_addback(XFLOAT *C, size_t ldc, const XFLOAT *V, const XFLOAT *Q,
		size_t ld, size_t P, size_t nT, size_t T, size_t O)
{
	for (size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x; j < nT * O; j += (size_t) gridDim.x * blockDim.x)
	{
		const size_t i = j % nT, o = j / nT;
		const float v = C[o * ldc + i] + V[o + (i / T) * O] + Q[i * ld + 3 * P] + Q[i * ld + 3 * P + 1];
		if (!isfinite(v))
		{
			printf("coarse-gemm 16-bit: non-finite score (row %llu, orientation %llu): a bug; stopping\n",
			       (unsigned long long) i, (unsigned long long) o);
			__trap();
		}
		C[o * ldc + i] = v;
	}
}
// R's scale lives just past the matrix (the cache allocates it).
__host__ __device__ inline float *rScale(XFLOAT *R, size_t O, size_t ld) { return (float *) (R + O * ld); }
const size_t R_EXTRA = 32;

} // namespace CoarseGemm

#endif
