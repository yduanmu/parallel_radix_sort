#include <cstdint>
#include <omp.h>
#include <vector>
#include <array>

//for main
#include <fstream>
#include <bitset>
#include <iostream>
#include <random>
#include <chrono>

using std::vector;
using std::uint32_t;
using std::uint8_t;
using std::array;

//for main
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::steady_clock;

struct alignas(64) Count {
	array<size_t, 256> local = {0};
};

// ====================================================================================
// Radix sort parallelized from eloj's radix_sort_u32.c implementation.
// ====================================================================================
void prefix_sums(vector<Count>& offset, vector<Count>& count,
				 size_t num_thr){
	//starts are the beginning offsets of each bucket.
	size_t start = 0;

	for(size_t bucket = 0; bucket < 256; ++bucket) {
		//runs are the per-thread offsets of each bucket.
		size_t run = start;

		for(size_t t = 0; t < num_thr; ++t){
			/* Offsets = runs. They decide which slice of the current bucket
			 * that each thread owns. */
			offset[t].local[bucket] = run;

			/* Increase runs by amount of elements in each histogram bucket to
			 * prepare for the next thread's offset calculation. */
			run += count[t].local[bucket];
		}

		//prepare for offset calculation of next bucket
		start = run;
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
			prefix_sums(offset0, count0, num_thr);
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
			prefix_sums(offset1, count1, num_thr);
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
			prefix_sums(offset2, count2, num_thr);
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
			prefix_sums(offset3, count3, num_thr);
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
	size_t n = 10000000;
	vector<uint32_t> zcodes;
	zcodes.resize(n);
	std::mt19937 prng(0);
	uint32_t maxcode = (1 << 30) - 1;
	std::uniform_int_distribution<uint32_t> dist(0, maxcode);
	for(size_t i = 0; i < n; ++i) {
		zcodes[i] = dist(prng);
	}

	size_t num_thr = 30;

	auto t0 = steady_clock::now();
	radix_sort(zcodes, num_thr);
	auto t1 = steady_clock::now();
	auto elapsed = duration_cast<milliseconds>(t1 - t0);
	std::cout << "par elapsed: " << elapsed.count() << std::endl;

	std::ofstream output("tests/par.txt");
	if(output.is_open()) {
		for(uint32_t key : zcodes) {
			output << std::bitset<32>(key) << "\n";
		}
		output.close();
	} else {
		std::cerr << "Unable to open tests/par.txt" << std::endl;
	}

	return 0;
}

