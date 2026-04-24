#include <immintrin.h>
#include <cstdint>
#include <iostream>
#include <omp.h>
#include <vector>
#include <array>

using std::uint32_t;
using std::vector;
using std::array;

// =====================================================================================
// documentation TBA
// =====================================================================================
vector<uint32_t> radix_sort(vector<uint32_t>& zcodes, vector<uint32_t> aux, size_t n) {
	// ---------------------------------------------------------------------------------
	// generate histograms
	// ---------------------------------------------------------------------------------
	array<size_t, 256> count0 = {0};
	array<size_t, 256> count1 = {0};
	array<size_t, 256> count3 = {0};
	array<size_t, 256> count4 = {0};

	size_t lim = n - (n % 8);
	for(size_t i = 0; i < lim; i += 8) {
		//32-bit Z-order code keys
		__m256i k = _mm256_loadu_epi32(zcodes.data() + i);
		__m256i mask = _mm256_set1_epi32(0x000000FF);

		__m256i k0 = _mm256_and_epi32(k, mask);
		__m256i k1 = _mm256_and_epi32(_mm256_srli_epi32(k, 8), mask);
		__m256i k2 = _mm256_and_epi32(_mm256_srli_epi32(k, 16), mask);
		__m256i k3 = _mm256_and_epi32(_mm256_srli_epi32(k, 32), mask);

	}
}


// =====================================================================================
// Main
// =====================================================================================


