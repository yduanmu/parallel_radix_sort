/* 
 * MIT License
 * Copyright (c) 2018-2021 Eddy L O Jansson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


#include <cstdint>
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

// ====================================================================================
// Sequential radix sort from eloj's radix_sort_u32.c, which I converted to C++.
// ====================================================================================
void radix_sort(vector<uint32_t>& zcodes) {
	size_t n = zcodes.size();
	vector<uint32_t> zcodes_aux;
	zcodes_aux.resize(n);

	//Histograms. 1 vector per pass.
	array<size_t, 256> count0;
	array<size_t, 256> count1;
	array<size_t, 256> count2;
	array<size_t, 256> count3;

	// --------------------------------------------------------------------------------
	// Generate histograms.
	// --------------------------------------------------------------------------------
	for(size_t i = 0; i < 256; ++i) {
		uint32_t key = zcodes[i];
		++count0[key & 0xFF];
		++count1[(key >> 8) & 0xFF];
		++count2[(key >> 16) & 0xFF];
		++count3[(key >> 24) & 0xFF];
	}
	
	// --------------------------------------------------------------------------------
	// Calculate prefix sums.
	// --------------------------------------------------------------------------------
	size_t a0 = 0;
	size_t a1 = 0;
	size_t a2 = 0;
	size_t a3 = 0;
	for(size_t i = 0; i < 256; ++i) {
		size_t b0 = count0[i];
		size_t b1 = count1[i];
		size_t b2 = count2[i];
		size_t b3 = count3[i];
		count0[i] = a0;
		count1[i] = a1;
		count2[i] = a2;
		count3[i] = a3;
		a0 += b0;
		a1 += b1;
		a2 += b2;
		a3 += b3;
	}

	// --------------------------------------------------------------------------------
	// Sort in 4 passes in LSB order.
	// --------------------------------------------------------------------------------
	for(size_t i = 0; i < n; ++i) {
		uint32_t key = zcodes[i];
		size_t dest = count0[key & 0xFF]++;
		zcodes_aux[dest] = zcodes[i];
	}

	for(size_t i = 0; i < n; ++i) {
		uint32_t key = zcodes_aux[i];
		size_t dest = count1[(key >> 8) & 0xFF]++;
		zcodes[dest] = zcodes_aux[i];
	}

	for(size_t i = 0; i < n; ++i) {
		uint32_t key = zcodes[i];
		size_t dest = count2[(key >> 16) & 0xFF]++;
		zcodes_aux[dest] = zcodes[i];
	}

	for(size_t i = 0; i < n; ++i) {
		uint32_t key = zcodes_aux[i];
		size_t dest = count3[(key >> 24) & 0xFF]++;
		zcodes[dest] = zcodes_aux[i];
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

	radix_sort(zcodes);

	std::ofstream output("tests/seq.txt");
	if(output.is_open()) {
		for(uint32_t key : zcodes) {
			output << std::bitset<32>(key) << "\n";
		}
		output.close();
	} else {
		std::cerr << "Unable to open tests/seq.txt";
	}

	return 0;
}

