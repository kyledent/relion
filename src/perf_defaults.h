/***************************************************************************
 *
 * Default settings of the opt-in performance switches, chosen at start-up from the run type and the GPU.
 *
 * Each switch is an environment variable read where it is used (RELION_COARSE_GEMM, RELION_KEEP_GPU_POOL, ...).
 * For a CUDA run on the GPU, applyPerformanceDefaults() sets every such variable that is not already set to the
 * value measured to be best for this run type and GPU, then prints the configuration and where each value came
 * from. A variable set by the user always wins; RELION_PERF_DEFAULTS=0 sets nothing (every switch off, as before).
 *
 * The run type is known from the options: a gradient run (ab initio), auto-refine, 2D classification (no
 * reference and no --denovo_3dref) or 3D classification (everything else, including 2D classification against
 * given references). The GPU is the lowest compute capability among the visible devices.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 ***************************************************************************/
#ifndef PERF_DEFAULTS_H_
#define PERF_DEFAULTS_H_

#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#ifdef _CUDA_ENABLED
#include <cuda_runtime.h>
#endif

enum PerfRunType { PERF_GRADIENT, PERF_CLASS2D, PERF_CLASS3D, PERF_REFINE3D };

inline const char *perfRunTypeName(PerfRunType t)
{
	switch (t)
	{
	case PERF_GRADIENT: return "gradient (ab initio)";
	case PERF_CLASS2D: return "2D classification";
	case PERF_CLASS3D: return "3D classification";
	default: return "auto-refine";
	}
}

// The defaults, from measurements of ab initio, 2D and 3D classification and auto-refine on four GPU types:
//   coarse pass as a GEMM, B = 8 particles per GEMM: gradient runs and 3D classification (in 2D classification and
//     auto-refine the batching costs 2-6 %); tf32 on sm_80 and newer; on sm_7x fp16 for gradient runs (gated on
//     the best sample and on the maps) and fp32 for 3D classification; off below sm_70 (not measured);
//   kept GPU pool, GPU FT map, gradient threads, significance indices, prior cache: on;
//   an allocator per expectation thread (512 MB): on, except on L40S GPUs, where it costs 9 %;
//   threaded FFTW plans outside the expectation: off (1 thread; it costs 1-3 % almost everywhere).
inline void applyPerformanceDefaults(PerfRunType type, bool do_gpu, bool verbose)
{
	const char *master = getenv("RELION_PERF_DEFAULTS");
	if (master != NULL && atoi(master) == 0)
	{
		if (verbose)
			std::cout << " Performance defaults: off (RELION_PERF_DEFAULTS=0)" << std::endl;
		return;
	}
#ifdef _CUDA_ENABLED
	if (!do_gpu)
		return;

	int count = 0, cc = 0;
	bool l40s = false;
	std::string gpus;
	if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
	{
		cudaGetLastError();
		return;   // e.g. an MPI leader on a node without a GPU: it runs no GPU code
	}
	for (int i = 0; i < count; i++)
	{
		cudaDeviceProp p;
		if (cudaGetDeviceProperties(&p, i) != cudaSuccess)
			continue;
		int c = 10 * p.major + p.minor;
		cc = (cc == 0) ? c : std::min(cc, c);
		if (strstr(p.name, "L40S") != NULL)
			l40s = true;
		if (gpus.find(p.name) == std::string::npos)
			gpus += std::string(gpus.empty() ? "" : ", ") + p.name;
	}
	if (cc == 0)
		return;

	const bool gemm_type = (type == PERF_GRADIENT || type == PERF_CLASS3D);
	int gemm = 0;
	if (gemm_type)
	{
		if (cc >= 80)
			gemm = 2;
		else if (cc >= 70)
			gemm = (type == PERF_GRADIENT) ? 3 : 1;
	}
	struct Setting { const char *name; std::string value; };
	std::vector<Setting> settings = {
		{"RELION_COARSE_GEMM", std::to_string(gemm)},
		{"RELION_COARSE_GEMM_BATCH", gemm > 0 ? "8" : "1"},
		{"RELION_KEEP_GPU_POOL", "1"},
		{"RELION_GRAD_GPU_FTMAP", "1"},
		{"RELION_GRAD_THREADS", "1"},
		{"RELION_SIG_COMPACT", "1"},
		{"RELION_PDF_CACHE", "1"},
		{"RELION_THREAD_ALLOC_MB", l40s ? "0" : "512"},
		{"RELION_FFTW_THREADS", "1"},
	};
	std::string line;
	for (const Setting &s : settings)
	{
		const char *user = getenv(s.name);
		if (user == NULL)
			setenv(s.name, s.value.c_str(), 0);
		line += std::string("  ") + s.name + "=" + (user == NULL ? s.value : std::string(user) + " (set)");
	}
	if (verbose)
		std::cout << " Performance defaults for " << perfRunTypeName(type) << " on sm_" << cc << " (" << gpus
		          << "); RELION_PERF_DEFAULTS=0 turns them off:" << std::endl << line << std::endl;
#endif
}

#endif /* PERF_DEFAULTS_H_ */
