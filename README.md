# radix-sort
Implementation of least-significant bit (LSb) forward-order sequential and parallel radix sort. Sorts an arbitrary-length vector of $`32`$-bit `int` Z-order (Morton) codes. Result is in MSb.

I use both in my [LBVH project](https://github.com/yduanmu/lbvh-cpu).

## Sequential:

```
g++ -O3 -mavx2 -march=native -mbmi2 -Wall -Werror -Wextra -Wpedantic seq.cpp -o seq && ./seq
```

Has data-level parallelism (SIMD using AVX2) but uses the sequential algorithm where nodes are created after ancestors. $`8`$-bit wide radixes (radix-256), though I may test against $`11`$-bit radixes. Uses counting sort.

Vectorized from eloj's [radix-sort](https://github.com/eloj/radix-sorting).

## Parallel:

```
g++ -O3 -mavx2 -march=native -mbmi2 -fopenmp -Wall -Werror -Wextra -Wpedantic par.cpp -o par && ./par
```

