#ifndef CUDA_PINNED_UPLOAD_H_
#define CUDA_PINNED_UPLOAD_H_

/*
 * Small host-to-device uploads through a pinned ring buffer (opt-in).
 *
 * An asynchronous copy from pageable host memory makes the calling thread wait: the driver first
 * synchronises the stream, then stages the data through a buffer that all threads of the process
 * share. A particle makes a few dozen such uploads, so the threads of one process spend much of their
 * time waiting behind each other's GPU work. With RELION_PINNED_UPLOAD=1, an upload of at most a
 * quarter of the ring is copied into a pinned ring buffer on the host and sent from there, which
 * returns at once. An event recorded after each copy marks when its part of the ring may be reused.
 * The data sent are the same, and the caller may reuse its buffer as soon as the call returns, as
 * with pageable memory.
 *
 * Rings (RELION_PINNED_UPLOAD_MB, default 16, per ring) are made on first use and kept for the whole
 * run: a thread borrows one from a shared pool and gives it back when it ends, so threads that are
 * made again for each iteration do not allocate pinned memory again.
 */

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

namespace PinnedUpload
{

inline bool enabled()
{
	static const bool e = getenv("RELION_PINNED_UPLOAD") != NULL && atoi(getenv("RELION_PINNED_UPLOAD")) > 0;
	return e;
}

struct Ring
{
	char *buf = NULL;
	size_t cap = 0, head = 0;
	struct Slot { size_t begin, end; cudaEvent_t ev; };
	std::deque<Slot> inflight;
	std::vector<cudaEvent_t> spare;
};

struct Pool
{
	std::mutex mutex;
	std::vector<Ring *> free;
};

inline Pool &pool()
{
	static Pool *p = new Pool();  // never destroyed: rings outlive threads and may be in use at exit
	return *p;
}

// This thread's ring, borrowed from the pool on first use and returned when the thread ends.
struct Holder
{
	Ring *ring = NULL;
	~Holder()
	{
		if (ring != NULL)
		{
			std::lock_guard<std::mutex> lock(pool().mutex);
			pool().free.push_back(ring);
		}
	}
};

inline Ring *threadRing()
{
	thread_local Holder holder;
	if (holder.ring == NULL)
	{
		{
			std::lock_guard<std::mutex> lock(pool().mutex);
			if (!pool().free.empty())
			{
				holder.ring = pool().free.back();
				pool().free.pop_back();
			}
		}
		if (holder.ring == NULL)
		{
			Ring *r = new Ring();
			const char *mb = getenv("RELION_PINNED_UPLOAD_MB");
			r->cap = (size_t) (mb != NULL && atoi(mb) > 0 ? atoi(mb) : 16) << 20;
			if (cudaMallocHost((void **) &r->buf, r->cap) != cudaSuccess)
			{
				cudaGetLastError();  // clear it; this thread then uses the normal path
				r->buf = NULL;
				r->cap = 0;
			}
			holder.ring = r;
		}
	}
	return holder.ring;
}

// Send bytes from h to d on stream through this thread's ring. Returns false, having done nothing, if
// the ring cannot take it; the caller then makes the normal copy.
inline bool upload(const void *h, void *d, size_t bytes, cudaStream_t stream)
{
	Ring *r = threadRing();
	const size_t need = (bytes + 255) & ~(size_t) 255;
	if (r->buf == NULL || bytes == 0 || need > r->cap / 4)
		return false;
	const size_t start = (r->head + need <= r->cap) ? r->head : 0;
	const size_t end = start + need;
	// The ring's parts in flight are one circular run, oldest first, so what [start, end) overlaps is at the front.
	while (!r->inflight.empty() && r->inflight.front().begin < end && r->inflight.front().end > start)
	{
		if (cudaEventSynchronize(r->inflight.front().ev) != cudaSuccess)
			return false;
		r->spare.push_back(r->inflight.front().ev);
		r->inflight.pop_front();
	}
	cudaEvent_t ev;
	if (!r->spare.empty())
	{
		ev = r->spare.back();
		r->spare.pop_back();
	}
	else if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess)
	{
		cudaGetLastError();
		return false;
	}
	memcpy(r->buf + start, h, bytes);
	if (cudaMemcpyAsync(d, r->buf + start, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
	    cudaEventRecord(ev, stream) != cudaSuccess)
	{
		r->spare.push_back(ev);
		return false;  // the error stays set, and the normal copy reports it
	}
	r->inflight.push_back({start, end, ev});
	r->head = end;
	return true;
}

} // namespace PinnedUpload

#endif
