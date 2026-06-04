// nvcc -o test v0_A100_vectorized_half8_preload_ptxstore.cu -arch=sm_80 -lineinfo
// ncu --print-details header -o ncu_output.out -f --set full ./test --clock-control none 
// nsys profile --force-overwrite=true -o profile.out --stats=true --trace=cuda --cuda-memory-usage=true ./test

#include <stdio.h>
#include <time.h>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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

/* extract least significant 16 bits of an unsigned int into an unsigned short*/
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

/* extract most significant 16 bits of an unsigned int into an unsigned short */
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



__global__ void add(
    const __half* __restrict__ A, 
    const __half* __restrict__ B, 
    __half* __restrict__ C, 
    const int N) {

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

        // I was trying to do the half2 addition and store to global operation
        // asm("{add.f16.x2 t1, %0, %1};\n\t"
        //     "{add.f16.x2 t2, %3, %4};\n\t"
        //     "{add.f16.x2 t3, %5, %6};\n\t"
        //     "{add.f16.x2 t4, %7, %8};\n\t"
        //     "st.global.v4.u32 	[%0], {t1, t2, t3, t4};\n\t"
        //     :
        //     : 
        //     )


        //reinterpret_cast<__half8*>(C)[i] = out; gives: st.global.v4.u32 	[%rd10], {%r17, %r20, %r23, %r26};
        // Here only the store to global memory operation
        // .wb, .cg, .cs, .wt, e.g., st.global.wt.v4.u32  -> not much impact
        asm("st.global.wt.v4.u32 	[%0], {%1, %2, %3, %4};"
            : 
            : "l"(&reinterpret_cast<const __half8*>(C)[i].a1), "r"(__half22uint(c1)), "r"(__half22uint(c2)), "r"(__half22uint(c3)),"r"(__half22uint(c4))

    );
    
    }

    // remaining elements if N is not divisible by 8
    int tail_start = (N/8) * 8;

    for (int i = tail_start + idx; i < N; i += blockDim.x * gridDim.x) { 
        C[i] = __hadd(A[i], B[i]);
    }

}
 





int main() {
    int N = 16384 * 16384;
    int threads = 256;
    int blocks = 108 * 2048 / threads; 

    size_t size = N * sizeof(__half);

    __half *A = (__half *)malloc(size);
    __half *B = (__half *)malloc(size);
    __half *C = (__half *)malloc(size);
    __half *C_cpu = (__half *)malloc(size);


    for (int i = 0; i < N; i++) {
        __half r1 = static_cast <__half> (rand()) / static_cast <__half> (RAND_MAX);
        __half r2 = static_cast <__half> (rand()) / static_cast <__half> (RAND_MAX);
        A[i] = r1;
        B[i] = r2;
        C[i] = __float2half(0.0f);
        C_cpu[i] = A[i] + B[i];
        // A[i] = __float2half(1.0f);
        // B[i] = __float2half(2.0f);
        // C[i] = __float2half(0.0f);
    }
    
    __half *d_A, *d_B, *d_C;
    gpuErrchk(cudaMalloc(&d_A, size));
    gpuErrchk(cudaMalloc(&d_B, size));
    gpuErrchk(cudaMalloc(&d_C, size));

    gpuErrchk(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaEventRecord(start));
    
    add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    
    gpuErrchk(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    cudaDeviceSynchronize();

    int errors = 0;
    for (int i = 0; i < N; i++) {
        // float val = __half2float(C[i]);
        // if (std::fabs(val - 3.0f) > 0.1f) {
        //     printf("Mismatch at i = %i\n", i);
        //     printf("A[i] = %f\n", __half2float(A[i])); 
        //     printf("B[i] = %f\n", __half2float(B[i])); 
        //     printf("C[i] = %f\n", val);
        //     errors++;
        //     break;
        // }
        float diff = __half2float(C[i] - C_cpu[i]);
        if (std::fabs(diff) > 0.01f) {
            printf("Mismatch at i = %i\n", i);
            printf("A[i] = %f\n", __half2float(A[i])); 
            printf("B[i] = %f\n", __half2float(B[i])); 
            printf("C[i] = %f\n", __half2float(C[i]));
            errors++;
            break;
        }
    }

    if (errors == 0) {
        printf("Verification passed!\n");
    }

    free(A); free(B); free(C);
    gpuErrchk(cudaFree(d_A)); gpuErrchk(cudaFree(d_B)); gpuErrchk(cudaFree(d_C));
    gpuErrchk(cudaEventDestroy(start)); gpuErrchk(cudaEventDestroy(stop));

    printf("Kernel execution time: %f ms\n", milliseconds);
    return 0;
}
