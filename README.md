# parallel_radix_sort
Implementation of least-significant digit (LSD) parallel radix sort. Sorts an arbitrary-length vector of $`32`$-bit `int` Z-order (Morton) codes. Result is MSb-first.

Parallelized from eloj's [radix-sorting](https://github.com/eloj/radix-sorting).

I use it in my [LBVH project](https://github.com/yduanmu/lbvh-cpu).

## Usage

```
g++ -O3 -march=native -fopenmp -Wall -Werror -Wextra -Wpedantic par.cpp -o par && ./par
```

Takes in a `vector<uint32_t>` and outputs a `vector<uint32_t>`. For best usage, the output should be a pre-allocated buffer.

> [!NOTE]
> The current max count per histogram is $`2^{32} - 1`$. If you expect a larger bucket size, or can get away with $`2^{16}-1`$ count per bucket, change it accordingly.

## Implementation

Once again, parallelized from eloj's [radix-sorting](https://github.com/eloj/radix-sorting).

For $`t`$ threads, input is split into $`t`$ thread-local contiguous chunks. We use an $`8`$-bit radix, meaning $`4`$ passes per each `uint32_t` key, and $`256`$ histogram buckets in total. Histograms are built per-thread and combined at the end with OpenMP reduction, which is lock-free.

Once the offset per bucket of the global histogram has been calculated, we now calculate the per-thread global histogram bucket offsets. This is a prefix sum of $`4 * 256`$ `uint32_t`s, or `4KB` of data. It can be performed quickly when sequential and scalar.

Lastly, once prefix sums are calculated, sort in $`4`$ passes (LSD order). The input of next pass is directly the output of the recently-completed pass.

Remember to pin threads and use cache-friendly bucket layout.

Required for vectorization of generating histograms [bookmark](https://stackoverflow.com/questions/51778721/how-to-convert-32-bit-float-to-8-bit-signed-char-41-packing-of-int32-to-int8).

## Testing

I benchmark this against eloj's [radix-sorting](https://github.com/eloj/radix-sorting#-c-implementation). The testing machine is an [Intel Xeon Gold 5128](https://www.intel.com/content/www/us/en/products/sku/192444/intel-xeon-gold-5218-processor-22m-cache-2-30-ghz/specifications.html).

Parallelism only expected to pay off at a large $`n`$ number of keys to sort. Expected to scale poorly over a certain amount of threads because of memory bandwidth saturation.

