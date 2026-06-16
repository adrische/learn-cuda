# from task import input_t, output_t


# def custom_kernel(data: input_t) -> output_t:
#     A, B, output = data
#     output[...] = A + B
#     return output

# A100_vectorized_float4_with_preload.py on leaderboard


import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t

add_cuda_source = r"""
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void add_kernel(
    const scalar_t* __restrict__ A, 
    const scalar_t* __restrict__ B, 
    scalar_t* __restrict__ C, 
    const int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float4 anext = __ldg(&reinterpret_cast<const float4*>(A)[idx]);
    float4 bnext = __ldg(&reinterpret_cast<const float4*>(B)[idx]);
    
    int stride = gridDim.x * blockDim.x;

    float4 aovernext;
    float4 bovernext;
    
    if (idx + stride < N/8) {
        aovernext = __ldg(&reinterpret_cast<const float4*>(A)[idx + stride]);
        bovernext = __ldg(&reinterpret_cast<const float4*>(B)[idx + stride]);
    }

    for (int i = idx; i < N/8; i += stride) {
        int inext = i + stride;
        int iovernext = i + 2*stride;
        
        float4 a = anext;
        float4 b = bnext;

        if (inext < N/8) {
            anext = aovernext;
            bnext = bovernext;
        }
        if (iovernext < N/8) {
            aovernext = __ldg(&reinterpret_cast<const float4*>(A)[iovernext]);
            bovernext = __ldg(&reinterpret_cast<const float4*>(B)[iovernext]);
        }

        __half2 * a_half2 = reinterpret_cast<__half2 *>(&a);
        __half2 * b_half2 = reinterpret_cast<__half2 *>(&b);

        float4 c;

        for (int j = 0; j < 4; j++) {
            reinterpret_cast<__half2*>(&c)[j] = __hadd2(a_half2[j], b_half2[j]);
        }
        
        // reinterpret_cast<float4*>(C)[i] = c;
        __stwt(&reinterpret_cast<float4*>(C)[i], c); // st.global.wt.v4.f32

        // asm("st.global.wt.v4.f32 	[%0], {%1, %2, %3, %4};"
        // asm("st.global.L1::no_allocate.v4.f32 	[%0], {%1, %2, %3, %4};"
        // : 
        // : "l"(&reinterpret_cast<float4*>(C)[i]), "f"(c.x), "f"(c.y), "f"(c.z),"f"(c.w)
        // );

    }

    for (int i = idx + 8*(N/8); i < N; i += stride) {
        C[i] = A[i] + B[i]; // could save some half operations by doing up to 3 __half2 ones
    }

}

torch::Tensor add_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // TORCH_CHECK(A.device().is_cuda(), "Tensor A must be a CUDA tensor");
    // TORCH_CHECK(B.device().is_cuda(), "Tensor B must be a CUDA tensor");
    // TORCH_CHECK(C.device().is_cuda(), "Tensor C must be a CUDA tensor");
    // TORCH_CHECK(A.sizes() == B.sizes(), "Input tensors must have the same size");
    
    int N = A.numel();  
    
    // Best values depend on GPU, optimal values may differ even between A100-40GB and A100-80GB
    // const int threads = 128; 
    const int threads = 256; // fastest
    // const int threads = 512; 
    // const int threads = 1024;
    
    // const int blocks = 108 * 2048 / threads; 
    const int blocks = ((N/8) + threads - 1) / threads; // fastest
    // const int blocks = (N + threads - 1) / threads;
    // const int blocks = 54 * 2048 / threads; 

    // No effect:    
    // cudaFuncSetAttribute(add_kernel<at::Half>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    // cudaFuncSetAttribute(add_kernel<c10::Half>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);

    
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
    // add_kernel<__half><<<blocks, threads>>>(
    add_kernel<<<blocks, threads>>>(
            a,
            b,
            c,
            N
        );
    */
    
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     throw std::runtime_error(cudaGetErrorString(err));
    // }

    return C;
}
"""

add_cpp_source = r"""
#include <torch/extension.h>

torch::Tensor add_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);
"""

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

def add(A, B, C):
    # if not A.is_cuda or not B.is_cuda or not C.is_cuda:
    #     raise RuntimeError("All tensors must be on GPU")
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

    # assert A.is_cuda and B.is_cuda and C.is_cuda, "Input/output tensors must be on GPU"
    # assert A.shape == B.shape, "Input tensors must have the same shape"
    # assert C.shape == A.shape, "Output tensor and input tensors must have the same shape"
    # assert A.dtype == torch.float16 and B.dtype == torch.float16 and C.dtype == torch.float16, "Input/output tensors must be float16"
    
    # Simply reuse the existing add function we already defined
    # This avoids the compilation issues with the inline kernel
    return add(A, B, C)
