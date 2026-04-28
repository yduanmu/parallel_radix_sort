#include <cstdint>
#include <omp.h>
#include <vector>
#include <array>

//for main
#include <fstream>
#include <bitset>
#include <iostream>

using std::vector;
using std::uint32_t;
using std::uint8_t;
using std::array;

struct alignas(64) Count {
	array<size_t, 256> local = {0};
};

// ====================================================================================
// Radix sort parallelized from eloj's radix_sort_u32.c implementation.
// ====================================================================================
void prefix_sums(vector<Count>& offset0, vector<Count>& offset1,
				 vector<Count>& offset2, vector<Count>& offset3,
				 vector<Count>& count0, vector<Count>& count1,
				 vector<Count>& count2, vector<Count>& count3,
				 size_t num_thr){
	//starts are the beginning offsets of each bucket.
	size_t start0 = 0;
	size_t start1 = 0;
	size_t start2 = 0;
	size_t start3 = 0;

	for(size_t bucket = 0; bucket < 256; ++bucket) {
		//runs are the per-thread offsets of each bucket.
		size_t run0 = start0;
		size_t run1 = start1;
		size_t run2 = start2;
		size_t run3 = start3;

		for(size_t t = 0; t < num_thr; ++t){
			/* Offsets = runs. They decide which slice of the current bucket
			 * that each thread owns. */
			offset0[t].local[bucket] = run0;
			offset1[t].local[bucket] = run1;
			offset2[t].local[bucket] = run2;
			offset3[t].local[bucket] = run3;

			/* Increase runs by amount of elements in each histogram bucket to
			 * prepare for the next thread's offset calculation. */
			run0 += count0[t].local[bucket];
			run1 += count1[t].local[bucket];
			run2 += count2[t].local[bucket];
			run3 += count3[t].local[bucket];
		}

		//prepare for offset calculation of next bucket
		start0 = run0;
		start1 = run1;
		start2 = run2;
		start3 = run3;
	}
}

void radix_sort(vector<uint32_t>& zcodes, size_t num_thr) {
	size_t n = zcodes.size();
	vector<uint32_t> zcodes_aux;
	zcodes_aux.resize(n);

	/* Per-thread histograms. Aligned to prevent false sharing. One vector per pass;
	 * within each vector is the per-thread histogram vector. */
	vector<Count> count0, count1, count2, count3;
	count0.resize(num_thr);
	count1.resize(num_thr);
	count2.resize(num_thr);
	count3.resize(num_thr);

	//offset arrays
	vector<Count> offset0, offset1, offset2, offset3;
	offset0.resize(num_thr);
	offset1.resize(num_thr);
	offset2.resize(num_thr);
	offset3.resize(num_thr);

	omp_set_dynamic(0);
	omp_set_num_threads(num_thr);
	
	// --------------------------------------------------------------------------------
	// Sort in 4 passes in LSB order.
	// --------------------------------------------------------------------------------
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		// ------------------------------- Pass 0. ------------------------------------
		/* Each thread only mutates in its own slice. These offsets are calculated by
		 * histogramming the current pass and taking the prefix sums. */

		/* Each thread operates on its private copy of the histogram arrays and, once
		 * complete, sums each element with the global arrays. For t threads, this is
		 * (256 * t * 4) summations. */
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

		/* Prefix sum: histogram of digit counts is now tranformed to memory address
		 * offsets for the output array. */
		#pragma omp single
		{
			prefix_sums(offset0, offset1, offset2, offset3,
						count0, count1, count2, count3, num_thr);
		}

		/* Now sort the cureent pass. */
		#pragma omp for schedule(static)
		for(size_t i = 0; i < n; ++i) {
			uint32_t key = zcodes[i];
			uint8_t digit = key & 0xFF;
			size_t dest = offset0[tid].local[digit] ++;
			zcodes_aux[dest] = zcodes[i];
		}
		
		// ------------------------------- Pass 1. ------------------------------------
		/* The histogram and prefix sums must be recalculated in order for per-thread
		 * write offsets to be accurate. */
		count1[tid].local = {0};
		#pragma omp for schedule(static)
		for(size_t i = 0; i < n; ++i ) {
			uint32_t key = zcodes_aux[i];
			++count1[tid].local[(key >> 8) & 0xFF];
		}

		#pragma omp single
		{
			prefix_sums(offset0, offset1, offset2, offset3,
						count0, count1, count2, count3, num_thr);
		}

		#pragma omp for schedule(static)
		for(size_t i = 0; i < n; ++i) {
			uint32_t key = zcodes_aux[i];
			uint8_t digit = (key >> 8) & 0xFF;
			size_t dest = offset1[tid].local[digit] ++;
			zcodes[dest] = zcodes_aux[i];
		}

		// ------------------------------- Pass 2. ------------------------------------
		count2[tid].local = {0};
		#pragma omp for schedule(static)
		for(size_t i = 0; i < n; ++i ) {
			uint32_t key = zcodes[i];
			++count2[tid].local[(key >> 16) & 0xFF];
		}

		#pragma omp single
		{
			prefix_sums(offset0, offset1, offset2, offset3,
						count0, count1, count2, count3, num_thr);
		}

		#pragma omp for schedule(static)
		for(size_t i = 0; i < n; ++i) {
			uint32_t key = zcodes[i];
			uint8_t digit = (key >> 16) & 0xFF;
			size_t dest = offset2[tid].local[digit] ++;
			zcodes_aux[dest] = zcodes[i];
		}

		// ------------------------------- Pass 3. ------------------------------------
		count3[tid].local = {0};
		#pragma omp for schedule(static)
		for(size_t i = 0; i < n; ++i ) {
			uint32_t key = zcodes_aux[i];
			++count3[tid].local[(key >> 24) & 0xFF];
		}

		#pragma omp single
		{
			prefix_sums(offset0, offset1, offset2, offset3,
						count0, count1, count2, count3, num_thr);
		}

		#pragma omp for schedule(static)
		for(size_t i = 0; i < n; ++i) {
			uint32_t key = zcodes_aux[i];
			uint8_t digit = (key >> 24) & 0xFF;
			size_t dest = offset3[tid].local[digit] ++;
			zcodes[dest] = zcodes_aux[i];
		}

	}
}

// ====================================================================================
// Main.
// Minor test for correctness for now; comparison and microbenchmark TBA.
// ====================================================================================
int main() {
	vector<uint32_t> zcodes = {
		0b00000111111100011001010100110111,
		0b00010011111011111111011011000110,
		0b00011000011011010000001001011001,
		0b00111111111100111010111101010111,
		0b00101000000100000000001101000101,
		0b00101101111101100100100101110111,
		0b00000111101100011010001110111000,
		0b00011111111110111111110110011110,
		0b00000100111001100111001110110001,
		0b00001100010110011011000100110101,
		0b00001100000111000111011010110000,
		0b00000010111000000011100111111100,
		0b00000000111101011011010110110011,
		0b00100010110100100000110010100110,
		0b00101011101011011101001001111001
	};

	radix_sort(zcodes, 2);

	std::ofstream output("tests/par.txt");
	if(output.is_open()) {
		for(uint32_t key : zcodes) {
			output << std::bitset<32>(key) << "\n";
		}
		output.close();
	} else {
		std::cerr << "Unable to open tests/par.txt";
	}

	return 0;
}

