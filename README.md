# radix-sort
Implementation of least-significant bit (LSb) parallel radix sort. Sorts an arbitrary-length vector of $`32`$-bit `int` Z-order (Morton) codes. Result is in MSb.

I use it in my [LBVH project](https://github.com/yduanmu/lbvh-cpu).

## Parallel:

```
g++ -O3 -mavx2 -march=native -mbmi2 -fopenmp -Wall -Werror -Wextra -Wpedantic par.cpp -o par && ./par
```

