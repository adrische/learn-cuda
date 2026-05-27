# Vector - vector addition

Single precision float, 2^25 elements, GTX 1050 Ti, CC 6.1

Kernel name                                                          | Duration kernel (ms)  | Duration incl. copy operations (ms)
---------------------------------------------------------------------|-----------------------|-----------------------------------
[CuBLAS Saxpy](vCublas.cu)                                           |                   4.1 |    51
[Pytorch A + B](vPytorch.ipynb)                                      |                   4.9 |    112
[v1](v1.cu) (naive 1 thread = 1 element)                             |                   4.1 |    52
[v2](v2.cu) (overlapping data movement and computation with streams) |                   4.6 |    31
[v3](v3.cu) (more data load operations per thread, grid-stride loop) |                   4.1 |    52

Practical limit is given by host <-> device transfer: bandwidthTest sample gives 12.2 GB/s. Hence the practical minimum total runtime including copy operations is

2^25 (number of elements) * 4 (number of bytes in a float) * 3 (2 host -> device, 1 device -> host transfers) / 1e9 (B -> GB) / 12.2 GB/s  = 33 ms.

The difference to v2 above seem to be time measurement or rounding errors.


### Further approaches

All the following items try modify the kernel's memory access pattern by trying to hide its load and store operations from global memory behind the concurrently ongoing host -> device or device -> host transfers. This would hide the 4 ms kernel execution time (which is mainly global memory access). Since 2 streams are sufficient to achieve the ~40% overall speedup, the kernel would be distributed across 2 streams for a maximum of 2ms savings.

#### Things that didn't work:

* [v4](v4.cu) - [Vectorized memory access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/), memory transfer seems to be saturated already.
* [Explicitly manipulating L2 cache](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html#l2-cache-control). The tools for this seem to be available only for compute capability 8.0.
* [v5](v5.cu) - Implicitly manipulating L2 cache: In my understanding, host to device copies go through L2 cache before they end up in global memory. When accessing data in global memory, again the request gets sent first to L2 cache. Hence, by copying sufficiently small pieces from host to device, the data should already be present in L2 and be faster accessible than global memory. I did not manage to go above 50% L2 hit rate though.
* [Explicily synchronously copying](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#asynchronous-data-copies) ("prefetching") from global to shared memory: This again seems only to be available for compute capability 8.0.

#### Things that did seem to work:

* [v6](v6.cu) -  Implicitly prefetching from global to shared memory: It seems to be possible to concurrently execute host -> device, device -> host copies, and a kernel copying global -> shared memory.