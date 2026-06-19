#include <stdio.h>
#include <time.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

const int THREADS = 256;
const int BLOCKS = 108 * 2048 / THREADS;

// Compile with -arch=sm_80
// Adapted from
// https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/
template<typename T>
__global__ void add_v3(const T * __restrict__ A, const T * __restrict__ B, T * __restrict__ C, size_t size) {
    
    const int num_stages = 2;
    
    
    // Read blockDim.x type T elements per pipeline stage
    __shared__ alignas(128) float4 A_smem[num_stages][THREADS];
    __shared__ alignas(128) float4 B_smem[num_stages][THREADS];
    

    // Grid stride loop:
    int offset = blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;

    // No pipeline::shared_state needed
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
 
    // Load all pipeline stages.
    for (int stage = 0; stage < num_stages; ++stage) {
        pipe.producer_acquire();
        size_t idx = offset + stage * stride + threadIdx.x;
        if (idx < size / 8) {
            cuda::memcpy_async(&A_smem[stage][threadIdx.x], &reinterpret_cast<const float4 *>(A)[idx], sizeof(float4), pipe);
            cuda::memcpy_async(&B_smem[stage][threadIdx.x], &reinterpret_cast<const float4 *>(B)[idx], sizeof(float4), pipe);
        }
        pipe.producer_commit();
    }

    // At this point, there are `num_stages` commited into the pipeline. This is a loop
    // invariant that is upheld throughout the loop.
    int stage = 0;
    for (size_t block_idx = offset; block_idx < size / 8; block_idx += stride) {
        // Wait for the first stage to have completed loading, or equivalently: wait until
        // at most `num_stages - 1` stages are still loading.
        cuda::pipeline_consumer_wait_prior<num_stages-1>(pipe);

        __syncthreads();
        
        // load from smem + compute + save to global memory
        if (block_idx + threadIdx.x < size / 8) {
            float4 a = A_smem[stage][threadIdx.x];
            float4 b = B_smem[stage][threadIdx.x];

            __half2 * a_half2 = reinterpret_cast<__half2 *>(&a);
            __half2 * b_half2 = reinterpret_cast<__half2 *>(&b);

            float4 c;

            for (int j = 0; j < 4; j++) {
                reinterpret_cast<__half2*>(&c)[j] = __hadd2(a_half2[j], b_half2[j]);
            }

            __stwt(&reinterpret_cast<float4 *>(C)[block_idx + threadIdx.x], c); // st.global.wt.v4.f32 ?? TODO check
        }
        
        __syncthreads();
          
        // Release the consumed stage.
        pipe.consumer_release();
 
        // Pre-load data for `num_stages` into the future.
        pipe.producer_acquire();
        // To ensure that the number of commited stages into the pipeline remains constant,
        // producer_acquire and producer_commit are called even if the load is out-of-bounds.
        size_t idx = block_idx + num_stages * stride + threadIdx.x;
        if (idx < size / 8) {
            cuda::memcpy_async(&A_smem[stage][threadIdx.x], &reinterpret_cast<const float4 *>(A)[idx], cuda::aligned_size_t<128>(sizeof(float4)), pipe);
            cuda::memcpy_async(&B_smem[stage][threadIdx.x], &reinterpret_cast<const float4 *>(B)[idx], cuda::aligned_size_t<128>(sizeof(float4)), pipe);
        }
 
        pipe.producer_commit();
 
        stage = (stage + 1) % num_stages;
    }


    for (int i = offset + threadIdx.x + 8*(size/8); i < size; i += stride) {
        C[i] = A[i] + B[i]; // could save some half operations by doing up to 3 __half2 ones
    }
}



int main() {
    // int Ns[] = {1024, 2048, 4096, 8192, 16384}; // benchmark
    // int Ns[] = {127, 128, 129, 256, 512, 1024, 2048, 4096, 8192, 16384}; // full
    // int Ns[] = {127, 128, 129, 256, 512, 1024, 2048, 4096, 8192}; // fast
    // int Ns[] = {127, 128, 129, 256, 512}; // test
    // int Ns[] = {4096};
    int Ns[] = {16384}; // the only timed shape

	int num = sizeof( Ns )/sizeof( Ns[0] );
	
	for(int idxN = 0; idxN < num; idxN++) {
        int N = Ns[idxN] * Ns[idxN];
        printf("\n------------- N: %i ---------\n", Ns[idxN]);
        
        size_t size = N * sizeof(__half);
        
        __half *A = (__half *)malloc(size);
        __half *B = (__half *)malloc(size);
        __half *C = (__half *)malloc(size);
        __half *C_cpu = (__half *)malloc(size);
        
        
        for (int i = 0; i < N; i++) {
            float r1 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            float r2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            
            A[i] = __float2half(r1);
            B[i] = __float2half(r2);
            // A[i] = __float2half(1.0f);
            // B[i] = __float2half(2.0f);
            C[i] = __float2half(0.0f);
            C_cpu[i] = __float2half(__half2float(A[i]) + __half2float(B[i]));
            // C[i] = __float2half(0.0f);
        }
        
        __half *d_A, *d_B, *d_C;
        gpuErrchk(cudaMalloc(&d_A, size));
        gpuErrchk(cudaMalloc(&d_B, size));
        gpuErrchk(cudaMalloc(&d_C, size));
        
        gpuErrchk(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
        
        // cudaFuncSetAttribute(add, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
        
        cudaEvent_t start, stop;
        gpuErrchk(cudaEventCreate(&start));
        gpuErrchk(cudaEventCreate(&stop));

        gpuErrchk(cudaEventRecord(start));
        
        ////////////////////////////////////////////////////////
        printf("Threads: %i, blocks: %i\n", THREADS, BLOCKS);
    
        add_v3<__half><<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);
        ////////////////////////////////////////////////////////
        
        gpuErrchk(cudaGetLastError());

        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaEventSynchronize(stop));
        
        gpuErrchk(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

        float milliseconds = 0;
        gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

        cudaDeviceSynchronize();

        int errors = 0;
        for (int i = 0; i < N; i++) {
            float diff = __half2float(C[i]) - __half2float(C_cpu[i]);
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

        gpuErrchk(cudaEventDestroy(start)); gpuErrchk(cudaEventDestroy(stop));
        
        printf("Kernel execution time: %f ms\n", milliseconds);
        printf("-------------");

        free(A); free(B); free(C);
        gpuErrchk(cudaFree(d_A)); gpuErrchk(cudaFree(d_B)); gpuErrchk(cudaFree(d_C));
    }

    return 0;
}