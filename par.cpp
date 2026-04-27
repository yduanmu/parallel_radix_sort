#include <cstdint>
#include <omp.h>
#include <vector>
#include <array>
#include <immintrin.h>

//used in OpenMP vector summation reduction declaration
// #include <algorithm>	
// #include <functional>

using std::vector;
using std::array;
using std::uint32_t;
using std::uint8_t;

// Custom reduction (summation) of vectors.
// #pragma omp declare reduction(vec_ui32_add : vector<uint32_t> : 					\
// 							  std::transform(omp_out.begin(), omp_out.end(),		\
// 					   		  omp_in.begin(), omp_out.begin(),						\
// 					   		  std::plus<uint32_t>()))								\
// 					initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

// ====================================================================================
// Radix sort parallelized from eloj's radix_sort_u32.c implementation.
// ====================================================================================
vector<uint32_t> radix_sort(const vector<uint32_t>& zcodes, int num_thr, bool scalar) {
	//global histogram. One array per pass
	alignas(32) array<uint32_t, 256> hist_count0 = {0};
	alignas(32) array<uint32_t, 256> hist_count1 = {0};
	alignas(32) array<uint32_t, 256> hist_count2 = {0};
	alignas(32) array<uint32_t, 256> hist_count3 = {0};
	
	size_t n = zcodes.size();
	vector<uint32_t> sorted;
	sorted.resize(n);

	omp_set_num_threads(num_thr);
	// ---------------------------------------------------------------------------
	// Vectorized approach to generating histograms.
	// @TODO benchmark this if scalar approach doesn't beat sequential.
	// ---------------------------------------------------------------------------
	/* AVX2 does not have a dedicated instruction to truncate 32-bit ints to
	 * 8-int. We can implement it by packing from 32->16->8 and then restore
	 * linear order with _mm256_permute4x64_epi64, which crosses lanes. */
	// size_t lim = n - (n % 8);
	// #pragma omp parallel for private(t_hist, key0, key1, key2, key3)
	// for(size_t i = 0; i < lim; i += 8) {
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
	
	// ---------------------------------------------------------------------------
	// Scalar approach to generating histograms.
	// Will become main approach if parallelization beats sequential.
	// ---------------------------------------------------------------------------
	#pragma omp parallel
	{
		// --------------------- Generate histograms. ----------------------------
		/* Each thread operates on its private copy of the histogram arrays and,
		 * once complete, sums each element with the global arrays. For t threads,
		 * this is (256 * t * 4) summations. OpenMP reduction can parallelize
		 * this efficiently. */
		#pragma omp for schedule(static) reduction(+ : hist_count0[:256]		 \
												   + : hist_count1[:256]		 \
												   + : hist_count2[:256]		 \
												   + : hist_count3[:256])
		for(size_t i = 0; i < n; ++i ) {
			uint32_t key = zcodes[i];
			
			//mask keys into octets to prepare for 4 passes of 8 bits each
			uint8_t key0 = key & 0xFF;
			uint8_t key1 = (key >> 8) & 0xFF;
			uint8_t key2 = (key >> 16) & 0xFF;
			uint8_t key3 = (key >> 24) & 0xFF;

			//histogram each of the octet arrays to prepare for the 4 passes
			++hist_count0[key0];
			++hist_count1[key1];
			++hist_count2[key2];
			++hist_count3[key3];
		}

		// ------------------- Calculate prefix sums. ---------------------------
		/* Histogram of digit counts is now tranformed to memory address offsets
		 * for the output array. In other words, for digit d, d = |d| + |p| where
		 * p is all the previous digits. */
		#pragma omp single
		{
			/* We are performing additions on 4KB of data, so only 1 thread is
			 * required. This is fast; other threads don't spin long. */
			size_t a0 = 0;
			size_t a1 = 0;
			size_t a2 = 0;
			size_t a3 = 0;
			for (int j = 0 ; j < 256 ; ++j) {
				size_t b0 = hist_count0[j];
				size_t b1 = hist_count1[j];
				size_t b2 = hist_count2[j];
				size_t b3 = hist_count3[j];
				hist_count0[j] = a0;
				hist_count1[j] = a1;
				hist_count2[j] = a2;
				hist_count3[j] = a3;
				a0 += b0;
				a1 += b1;
				a2 += b2;
				a3 += b3;
			}
		}

		// ----------------- Sort in 4 passes in LSB order. --------------------
		/* Each thread sorts only according to its own offset. */

	}

	return sorted;
}

