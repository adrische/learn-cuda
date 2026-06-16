# Fast vector addition on A100 - 80 GB

[This](submission.py) is my solution to the float16 vector addition problem https://www.gpumode.com/leaderboard/543?tab=rankings

At time of writing I'm listed at 6th place (tied with places 6-9), and I have run out of ideas what to try. The currently best solution is 0.27% (2.4 μs) faster.

My solution is based on interpreting 8 float16 as 4 float32 (or one float4, to be precise) to saturate memory bandwidth, doing the calculations using the half2 vector type to maximize arithmetic efficiency, and finding the [fastest load / store operations](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#low-level-load-and-store-functions), and the finding best launch configuration.


#### Some observations

Sometimes I would get different times for the same kernel, e.g., 958 μs and 895μs for the exact same submission, but I could not figure out why.

The relative performance between different kernels may not be consistent between different GPUs. I had access to an A100 - 40 GB, and what was fastest for this GPU was not necessarily fastest for the benchmarked A100 - 80 GB (mainly the launch configuration).

There was an inconsistency how nvcc would compile a local cu file, vs how load inline would compile inline cuda code. I tried to make the two methods consistent by including the following code in the submission:

```python
from torch.utils.cpp_extension import COMMON_NVCC_FLAGS

# List the specific flags you want to remove
flags_to_remove = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
    '--expt-relaxed-constexpr'
]

for flag in flags_to_remove:
    try:
        COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
        pass  # Flag was already absent

add_module = load_inline(
    name='add_cudafloat4',
    cpp_sources=add_cpp_source,
    cuda_sources=add_cuda_source,
    functions=['add_cuda'],
    verbose=True,
    extra_cuda_cflags=["-gencode=arch=compute_80,code=sm_80", '--use_fast_math', '-O3']
)
```


#### What did not work (for me):

A custom struct consisting of 8 half numbers, for a total of 128 bit. The compiler would not generate v4 load or store instructions.


Doing part of the calculation on the CPU:
- The idea is that in principle one could transfer a very small part of the data back to the CPU, while the GPU calculates, then doing the vector addition for that small part of the data on the CPU, and transferring the data back. In principle this could save a fraction of host-device-speed/global-memory-speed of overall execution time.
- However, transferring part of the data asyncronously to the CPU requires streams, and stream creation overhead is > 10ms, much too large for this small kernel.
- Additionally, the calculation is then CPU compute bound -> further reduction in possible benefit.


Trying custom [asm load and store operations](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st) did not give better performance.
For example, 
```cpp
__stwt(&reinterpret_cast<float4*>(C)[i], c);
```
could be accessed as
```cpp
asm("st.global.wt.v4.f32 	[%0], {%1, %2, %3, %4};"
: 
: "l"(&reinterpret_cast<float4*>(C)[i]), "f"(c.x), "f"(c.y), "f"(c.z),"f"(c.w)
);
```
and then be replaced by a more specific operation.


#### What turned out to be not relevant:

Interestingly, the solution does not use any features that would be specific to Ampere or compute capability 8.0.

Size-specific kernels / launch configurations, e.g., changing the number of blocks for smaller problems. The leaderboard was based on the performance for only one problem. Otherwise one could try to profile / ncu all problem sizes individually - I briefly tried this, and the same kernel and same launch specification would show very different characteristics for different problem sizes (e.g., memory utilization much lower).


####  Resources

Code for some other leaderboard entries:
* https://github.com/NitishNaineni/gpumode/tree/master/pmpp_v2/vectoradd_py
* https://github.com/CaptnJackSparrow/reference-kernels/blob/20260209/problems/pmpp_v2/vectoradd_py/solutions/correct/submission_cuda_inline_A100.py

Some interesting related papers:
* Better Performance at Lower Occupancy (instruction-level parallelism) https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf
* Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis (analysis of cycles for selected instructions, and memory latency benchmarking) https://arxiv.org/pdf/2208.11174
* Ampere architecture whitepaper https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
