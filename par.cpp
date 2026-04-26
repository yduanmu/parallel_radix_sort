#include <cstdint>
#include <omp.h>
#include <vector>
#include <array>
#include <immintrin.h>

using std::vector;
using std::array;
using std::uint32_t;
using std::uint8_t;

struct Histogram {
	alignas(32) array<uint32_t, 256> count0 = {0};
	alignas(32) array<uint32_t, 256> count1 = {0};
	alignas(32) array<uint32_t, 256> count2 = {0};
	alignas(32) array<uint32_t, 256> count3 = {0};
};

// ====================================================================================
// ====================================================================================
vector<uint32_t> radix_sort(const vector<uint32_t>& zcodes, int num_thr, bool scalar) {
	Histogram g_hist;	//global histogram. One array per pass
	
	size_t n = zcodes.size();
	vector<uint32_t> sorted;
	sorted.resize(n);

	size_t lim = n - (n % 8);
	Histogram t_hist;	//thread-level histogram
	vector<uint8_t> key0, key1, key2, key3;	//32-bit zcode keys broken into 4 octets
	key0.resize(n);
	key1.resize(n);
	key2.resize(n);
	key3.resize(n);
	if(!scalar) {
		// #pragma omp parallel for private(t_hist, key0, key1, key2, key3)
		// for(size_t i = 0; i < lim; i += 8) {
		// 	/* AVX2 does not have a dedicated instruction to truncate 32-bit ints to 8-int.
		// 	 * We can implement it by packing from 32->16->8 and then restore linear order
		// 	 * with _mm256_permute4x64_epi64, which crosses lanes.
		// 	 * @TODO This will have to be microbenchmarked.
		// 	 * https://stackoverflow.com/questions/51778721/how-to-convert-32-bit-float-to-8-bit-signed-char-41-packing-of-int32-to-int8
		// 	 */
		//
		// 	__m256i k = _mm256_loadu_epi32(zcodes.data() + i);	//z-order keys
		// 	__m256i mask = _mm256_set1_epi32(0x000000FF);
		//
		// 	//mask keys into octets to prepare for 4 passes of 8 bits each
		// 	_mm256_packus_epi16(_mm256_cvtepi32_epi16(_mm256_and_epi32(k, mask)));
		// 	__m256i k0 = _mm256_and_epi32(k, mask);
		// 	__m256i k1 = _mm256_and_epi32(_mm256_srli_epi32(k, 8), mask);
		// 	__m256i k2 = _mm256_and_epi32(_mm256_srli_epi32(k, 16), mask);
		// 	__m256i k3 = _mm256_and_epi32(_mm256_srli_epi32(k, 32), mask);
		//
		// 	//histogram each of the octet arrays to prepare for the 4 passes
		// }
	} else {
		#pragma omp parallel private(t_hist, key0, key1, key2, key3)
		{
			//@TODO: change to reduction
			#pragma omp for schedule(static) nowait
			for(size_t i = 0; i < n; ++i ) {
				uint32_t key = zcodes[i];
				
				//mask keys into octets to prepare for 4 passes of 8 bits each
				key0[i] = key & 0xFF;
				key1[i] = (key >> 8) & 0xFF;
				key2[i] = (key >> 16) & 0xFF;
				key3[i] = (key >> 24) & 0xFF;

				//histogram each of the octet arrays to prepare for the 4 passes
				++t_hist.count0[key0[i]];
				++t_hist.count1[key1[i]];
				++t_hist.count2[key2[i]];
				++t_hist.count3[key3[i]];
			}

			#pragma omp atomic critical
			{
				//
			}
		}
	}

	return sorted;
}

