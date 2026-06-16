import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t

add_cuda_source = r"""
#include <cuda_fp16.h>


typedef struct __align__(16) {
   __half a1;
   __half a2;
   __half a3;
   __half a4;
   __half a5;
   __half a6;
   __half a7;
   __half a8;
} __half8;



// https://forums.developer.nvidia.com/t/how-to-use-the-ex2-approx-f16x2-instruction/304825

// extract least significant 16 bits of an unsigned int into an unsigned short
__forceinline__ __device__ unsigned short uint2loushort (unsigned int arg)
{
    unsigned short res;
    asm ("{\n\t"
         ".reg .b16 lo, hi;\n\t"
         "mov.b32 {lo, hi}, %1;\n\t"
         "mov.b16 %0, lo;\n\t"
         "}\n\t"
         : "=h"(res) : "r"(arg));
    return res;
}

// extract most significant 16 bits of an unsigned int into an unsigned short 
__forceinline__ __device__ unsigned short uint2hiushort (unsigned int arg)
{
    unsigned short res;
    asm ("{\n\t"
         ".reg .b16 lo, hi;\n\t"
         "mov.b32 {lo, hi}, %1;\n\t"
         "mov.b16 %0, hi;\n\t"
         "}\n\t"
         : "=h"(res) : "r"(arg));
    return res;
}

__forceinline__ __device__ unsigned int __half22uint (__half2 arg) {

    __half hi, lo;
    unsigned short ilo, ihi;
    unsigned int out;

    lo = __low2half (arg);
    hi = __high2half (arg);
    ilo = __half_as_ushort (lo);
    ihi = __half_as_ushort (hi);
    out = ((unsigned int)ihi << 16) | ((unsigned int)ilo);

    return out;
}


template <typename scalar_t>
__global__ void add_kernel(const scalar_t* __restrict__ A, 
                           const scalar_t* __restrict__ B, 
                           scalar_t* __restrict__ C, 
                           int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N) return;

    __half8 anext = reinterpret_cast<const __half8*>(A)[idx];
    __half8 bnext = reinterpret_cast<const __half8*>(B)[idx];

    for(int i = idx; i < N/8; i += blockDim.x * gridDim.x) {
        // reinterpret_cast<__half8*>(C)[i] = reinterpret_cast<const __half8*>(A)[i] + reinterpret_cast<const __half8*>(B)[i];
        
        __half8 a = anext;
        __half8 b = bnext;
        
        int inext = i + blockDim.x * gridDim.x;
        if (inext < N/8) {
            __half8 anext = reinterpret_cast<const __half8*>(A)[inext];
            __half8 bnext = reinterpret_cast<const __half8*>(B)[inext];
        }

        __half2 a1 = __half2(a.a1, a.a2);
        __half2 a2 = __half2(a.a3, a.a4);
        __half2 a3 = __half2(a.a5, a.a6);
        __half2 a4 = __half2(a.a7, a.a8);
        __half2 b1 = __half2(b.a1, b.a2);
        __half2 b2 = __half2(b.a3, b.a4);
        __half2 b3 = __half2(b.a5, b.a6);
        __half2 b4 = __half2(b.a7, b.a8);

        __half2 c1 = __hadd2(a1, b1);
        __half2 c2 = __hadd2(a2, b2);
        __half2 c3 = __hadd2(a3, b3);
        __half2 c4 = __hadd2(a4, b4);


        // asm("{add.f16.x2 t1, %0, %1};\n\t"
        //     "{add.f16.x2 t2, %3, %4};\n\t"
        //     "{add.f16.x2 t3, %5, %6};\n\t"
        //     "{add.f16.x2 t4, %7, %8};\n\t"
        //     "st.global.v4.u32 	[%0], {t1, t2, t3, t4};\n\t"
        //     :
        //     : 
        //     )


        //reinterpret_cast<half8*>(C)[i] = out; gives: st.global.v4.u32 	[%rd10], {%r17, %r20, %r23, %r26};
        // .wb, .cg, .cs, .wt, e.g., st.global.wt.v4.u32  -> not much impact
        asm("st.global.wt.v4.u32 	[%0], {%1, %2, %3, %4};"
            : 
            : "l"(&reinterpret_cast<const __half8*>(C)[i].a1), "r"(__half22uint(c1)), "r"(__half22uint(c2)), "r"(__half22uint(c3)),"r"(__half22uint(c4))

    );
    
    }

    // remaining elements if N is not divisible by 8
    int tail_start = (N/8) * 8;

    for (int i = tail_start + idx; i < N; i += blockDim.x * gridDim.x) { 
        C[i] = A[i] + B[i];
    }

}






torch::Tensor add_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.device().is_cuda(), "Tensor A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "Tensor B must be a CUDA tensor");
    TORCH_CHECK(C.device().is_cuda(), "Tensor C must be a CUDA tensor");
    TORCH_CHECK(A.sizes() == B.sizes(), "Input tensors must have the same size");
    
    int N = A.numel();  

    const int threads = 256; 
    const int blocks = (N + threads - 1) / threads;  
    
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "add_kernel", ([&] {
        add_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            N
        );
    }));
    
    /*
    const __half* a = reinterpret_cast<const __half*>(A.data_ptr<at::Half>());
    const __half* b = reinterpret_cast<const __half*>(B.data_ptr<at::Half>());
    __half* c = reinterpret_cast<__half*>(C.data_ptr<at::Half>());
    add_kernel<__half><<<blocks, threads>>>(
            a,
            b,
            c,
            N
        );
    */

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}
"""

add_cpp_source = """
#include <torch/extension.h>

torch::Tensor add_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);
"""

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0+PTX"

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
    name='add_cuda',
    cpp_sources=add_cpp_source,
    cuda_sources=add_cuda_source,
    functions=['add_cuda'],
    verbose=True,
    extra_cuda_cflags=["-arch=sm_80", "-gencode=arch=compute_80,code=sm_80", "-O3", "--relocatable-device-code=false"]
)

def add(A, B, C):
    if not A.is_cuda or not B.is_cuda or not C.is_cuda:
        raise RuntimeError("All tensors must be on GPU")
    return add_module.add_cuda(A, B, C)

def custom_kernel(data: input_t) -> output_t:
    """
    Custom implementation of vector addition using CUDA.
    Args:
        inputs: List of pairs of tensors [A, B] to be added.
    Returns:
        Tensor containing element-wise sum.
    """
    A, B, C = data

    assert A.is_cuda and B.is_cuda and C.is_cuda, "Input/output tensors must be on GPU"
    assert A.shape == B.shape, "Input tensors must have the same shape"
    assert C.shape == A.shape, "Output tensor and input tensors must have the same shape"
    assert A.dtype == torch.float16 and B.dtype == torch.float16 and C.dtype == torch.float16, "Input/output tensors must be float16"
    
    # Simply reuse the existing add function we already defined
    # This avoids the compilation issues with the inline kernel
    return add(A, B, C)

A = torch.rand((2, 2), dtype=torch.float16, device="cuda")
B = torch.rand((2, 2), dtype=torch.float16, device="cuda")
C = torch.empty((2, 2), dtype=torch.float16, device="cuda")
torch.cuda.synchronize()
C = add(A, B, C)
torch.cuda.synchronize()
del A, B, C
