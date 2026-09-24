#ifndef ACC_OPERAND_DUMP_H_
#define ACC_OPERAND_DUMP_H_

// Opt-in dump of the coarse scoring pass's operands, for offline analysis of its numerics (for
// example, how reduced-precision arithmetic would change the scores). Measurement
// instrumentation, not a feature.
//
// With RELION_OPERAND_DUMP=<dir> set, the coarse scoring pass writes, for the first
// RELION_OPERAND_DUMP_MAX particles (default 8) each process scores in each iteration, what the
// coarse difference kernel reads -- the CTF-divided image, the weight image, the raw image, the
// CTF, 1/sigma2, the translations and the orientation matrices -- and the diff2 it returned for
// every (orientation, translation). The projector model of each class is written once per
// iteration by each process that dumps. Unset, the cost is one getenv per process.
//
// Each record is <dir>/<stem>.json (scalars, and an index of the arrays: dtype, element count,
// byte offset) plus <stem>.bin (the arrays, raw, native little-endian).

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <unistd.h>

namespace OperandDump
{

inline const char *dir()
{
	static const char *d = std::getenv("RELION_OPERAND_DUMP");
	return (d && *d) ? d : nullptr;
}

inline int maxParticles()
{
	static const char *m = std::getenv("RELION_OPERAND_DUMP_MAX");
	static const int n = m ? std::atoi(m) : 8;
	return n;
}

// True for the first maxParticles() particles this process scores in iteration `iter`.
inline bool claimParticle(int iter)
{
	static std::mutex mu;
	static int current = -1, count = 0;
	std::lock_guard<std::mutex> lock(mu);
	if (iter != current)
	{
		current = iter;
		count = 0;
	}
	if (count >= maxParticles())
		return false;
	count++;
	return true;
}

// True the first time this process asks about (iter, iclass).
inline bool claimModel(int iter, int iclass)
{
	static std::mutex mu;
	static std::set<std::pair<int, int> > seen;
	std::lock_guard<std::mutex> lock(mu);
	return seen.insert(std::make_pair(iter, iclass)).second;
}

template <typename T> inline const char *dtype();
template <> inline const char *dtype<float>() { return "<f4"; }
template <> inline const char *dtype<double>() { return "<f8"; }
template <> inline const char *dtype<int>() { return "<i4"; }

class Writer
{
	std::string stem;
	std::ofstream bin;
	std::ostringstream meta, index;
	size_t offset = 0;
	bool firstValue = true, firstArray = true;

public:
	explicit Writer(const std::string &name)
	{
		std::ostringstream s;
		s << dir() << "/" << name << "_pid" << getpid();
		stem = s.str();
		bin.open(stem + ".bin", std::ios::binary);
		meta.precision(17);   // scalars such as highres_Xi2 are compared at full precision
	}

	template <typename T> Writer &value(const char *key, T v)
	{
		meta << (firstValue ? "" : ",\n") << "  \"" << key << "\": " << v;
		firstValue = false;
		return *this;
	}

	// `n` elements of T; complex data is written as 2n interleaved (re, im) values of T.
	template <typename T> Writer &array(const char *key, const T *p, size_t n)
	{
		bin.write(reinterpret_cast<const char *>(p), n * sizeof(T));
		index << (firstArray ? "" : ",\n") << "    \"" << key << "\": {\"dtype\": \""
		      << dtype<T>() << "\", \"count\": " << n << ", \"offset\": " << offset << "}";
		offset += n * sizeof(T);
		firstArray = false;
		return *this;
	}

	~Writer()
	{
		bin.close();
		std::ofstream js(stem + ".json");
		js << "{\n" << meta.str() << (firstValue ? "" : ",\n")
		   << "  \"arrays\": {\n" << index.str() << "\n  }\n}\n";
	}
};

} // namespace OperandDump

#endif
