#include <cstdint>
#include <omp.h>
#include <vector>
#include <immintrin.h>
#include <array>

using std::vector;
using std::uint32_t;
using std::uint8_t;
using std::array;

struct alignas(64) Count {
	array<uint32_t, 256> local = {0};
};

// ====================================================================================
// Radix sort parallelized from eloj's radix_sort_u32.c implementation.
// ====================================================================================
vector<uint32_t> radix_sort(vector<uint32_t>& zcodes, size_t num_thr, bool scalar) {
	size_t n = zcodes.size();
	vector<uint32_t> zcodes_aux;
	zcodes_aux.resize(n);

	/* Per-thread histograms. Aligned to prevent false sharing. One vector per pass;
	 * within each vector is the per-thread histogram vector. */
	vector<Count> count0;
	vector<Count> count1;
	vector<Count> count2;
	vector<Count> count3;
	count0.resize(num_thr);
	count1.resize(num_thr);
	count2.resize(num_thr);
	count3.resize(num_thr);

	omp_set_num_threads(num_thr);
	// --------------------------------------------------------------------------------
	// Vectorized approach to generating histograms.
	// @TODO benchmark this if scalar approach doesn't beat sequential.
	// --------------------------------------------------------------------------------
	/* AVX2 does not have a dedicated instruction to truncate 32-bit ints to 8-int.
	 * We can work around this by packing from 32->16->8 and then restore linear order
	 * with _mm256_permute4x64_epi64, but that crosses lanes and might not be worth
	 * the overhead. */
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
	
	// --------------------------------------------------------------------------------
	// Scalar approach to generating histograms.
	// Will become main approach if parallelization beats sequential.
	// --------------------------------------------------------------------------------
	size_t a0 = 0;	//used in calculating prefix sums
	size_t a1 = 0;
	size_t a2 = 0;
	size_t a3 = 0;
	array<uint32_t, 256> global_starts;
	#pragma omp parallel
	{
		// ---------------------- Generate histograms. --------------------------------
		/* Each thread operates on its private copy of the histogram arrays and, once
		 * complete, sums each element with the global arrays. For t threads, this is
		 * (256 * t * 4) summations. OpenMP reduction can parallelize this efficiently.
		 */
		int tid = omp_get_thread_num();
		#pragma omp for schedule(static)
		for(size_t i = 0; i < n; ++i ) {
			uint32_t key = zcodes[i];
			
			/* Mask keys into octets to prepare for 4 passes of 8 bits each, then
			 * histogram each of the octet arrays to prepare for the 4 passes. */
			++count0[tid].local[key & 0xFF];
			++count1[tid].local[(key >> 8) & 0xFF];
			++count2[tid].local[(key >> 16) & 0xFF];
			++count3[tid].local[(key >> 24) & 0xFF];
		}

		// --------------------- Calculate prefix sums. -------------------------------
		/* Histogram of digit counts is now tranformed to memory address offsets for
		 * the output array. In other words, prefix sum. */
		#pragma omp for schedule(static) firstprivate(a0, a1, a2, a3)
		for(size_t i = 0; i < 256; ++i) {
			size_t b0 = count0[tid].local[i];
			size_t b1 = count1[tid].local[i];
			size_t b2 = count2[tid].local[i];
			size_t b3 = count3[tid].local[i];
			count0[tid].local[i] = a0;
			count1[tid].local[i] = a1;
			count2[tid].local[i] = a2;
			count3[tid].local[i] = a3;
			a0 += b0;
			a1 += b1;
			a2 += b2;
			a3 += b3;
		}

		#pragma omp single
		{
			for(size_t i = 0; i < 256; ++i) {
				for(size_t j = 0; j < tid - 1; ++j) {
					a0 = count0[j].local[i];
					a1 = count1[j].local[i];
					a2 = count2[j].local[i];
					a3 = count3[j].local[i];
					global_starts[i] += a0 + a1 + a2 + a3;
				}
			}
		}

		// -------------------- Sort in 4 passes in LSB order. ------------------------
		#pragma for schedule(static)
		for(size_t i = 0; i < n; ++i) {
			uint32_t key = zcodes[i];
			uint8_t digit = key & 0xFF;
			size_t offset = count0[tid].local[digit] + global_starts[digit] ++;
			zcodes_aux[offset] = zcodes[i];
		}

		#pragma for schedule(static)
		for(size_t i = 0; i < n; ++i) {
			uint32_t key = zcodes_aux[i];
			uint8_t digit = (key >> 8) & 0xFF;
			size_t offset = count1[tid].local[digit] + global_starts[digit] ++;
			zcodes[offset] = zcodes_aux[i];
		}


		#pragma for schedule(static)
		for(size_t i = 0; i < n; ++i) {
			uint32_t key = zcodes[i];
			uint8_t digit = (key >> 16) & 0xFF;
			size_t offset = count2[tid].local[digit] + global_starts[digit] ++;
			zcodes_aux[offset] = zcodes[i];
		}

		#pragma for schedule(static)
		for(size_t i = 0; i < n; ++i) {
			uint32_t key = zcodes_aux[i];
			uint8_t digit = (key >> 24) & 0xFF;
			size_t offset = count3[tid].local[digit] + global_starts[digit] ++;
			zcodes[offset] = zcodes_aux[i];
		}

	}

	return zcodes;
}

