# parallel_radix_sort
Implementation of least-significant digit (LSD) parallel radix sort in C++ using OpenMP. Sorts an arbitrary-length vector of $`32`$-bit `int` Z-order (Morton) codes. Result is MSb-first.

Parallelized from eloj's [radix-sorting](https://github.com/eloj/radix-sorting). CPU implementation of [Harada & Howes 2011](https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf).

I use it in my [LBVH project](https://github.com/yduanmu/lbvh-cpu).

## Table of contents
- [Usage](#usage)
- [Todo](#todo)
- [Implementation](#implementation)
- [Testing](#testing)
- [Sources](#sources)

## Usage

```
g++ -O3 -march=native -fopenmp -Wall -Werror -Wextra -Wpedantic par.cpp -o par && ./par
g++ -O3 -march=native -Wall -Werror -Wextra -Wpedantic seq.cpp -o seq && ./seq
```

Takes in a `vector<uint32_t>` and outputs a sorted `vector<uint32_t>`. The max count per histogram is $`2^{32} - 1`$.

## Todo

In descending order of importance.

- [ ] Microbenchmarking.
- [ ] More detail about correctness in writeup.
- [ ] A better prefix sum algorithm.
- [ ] SIMD.

## Implementation

Once again, parallelized from eloj's [radix-sorting](https://github.com/eloj/radix-sorting).

I have a fair amount of comments in the code, so it might be worthwhile to jump to that instead. This is more of a summary.

For $`t`$ threads, input is split into $`t`$ thread-local contiguous chunks. We use an $`8`$-bit radix, meaning $`4`$ passes per each `uint32_t` key (it is split into $`1`$-byte chunks), and $`256`$ histogram buckets per pass. Histograms are built per-thread.

> Generating this per-thread histogram is done in scalar. AVX2 does not have a dedicated instruction to truncate $`32`$-bit ints to $`8`$-int. We can work around this by packing from $`32`$ to $`16`$ to $`8`$ and then restore linear order with `_mm256_permute4x64_epi64`, but that crosses lanes and might not be worth the overhead for my purposes. Time permitting, I will try it.

Once the histograms have been calculated, global prefix sums are calculated using the histograms, which are used as the memory offsets of the keys in the complete `vector` of Morton codes. This is important because thread will read and write only from its own slice of the complete `vector`. The prefix sums are currently $`O(256 * t)`$ on `uint32_t`s, and performed sequentially. This is faster than computing per-thread prefix sums and then summing them together by ~$`200`$% even with a large amount of threads. The current `count` array is small enough to stay hot in the cache, and memory access is predictable. Prefix scan can be [performed more efficiently with SIMD](https://en.algorithmica.org/hpc/algorithms/prefix/) and [a much better prefix scan algorithm](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda).

Once prefix sums are calculated, the pass is sorted in LSD order. The input of next pass is directly the output of the recently-completed pass.

This entire histogram -> prefix sum -> sorting procedure is done $`4`$ times (passes) in total, once per $`1`$-byte chunk of the $`32`$-bit key. Once all passes are complete, the `vector` has been sorted.

Remember to pin threads.

## Testing

Parallelization successful; $`~200`$% speedup when sorting $`10M`$ codes compared to sequential when using $`10`$ threads; $`~450`$% speedup using $`30`$ threads. Difference in output was checked for using `diff`. More rigorous testing TBA.

I will benchmark this against eloj's [radix-sorting](https://github.com/eloj/radix-sorting#-c-implementation). The testing machine is an [Intel Xeon Gold 5128](https://www.intel.com/content/www/us/en/products/sku/192444/intel-xeon-gold-5218-processor-22m-cache-2-30-ghz/specifications.html).

Parallelism only expected to pay off at a large $`n`$ number of keys to sort. Expected to scale poorly over a certain amount of threads because of memory bandwidth saturation.

## Sources
- [radix-sorting](https://github.com/eloj/radix-sorting).
- [Harada & Howes 2011](https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf).
