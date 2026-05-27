// Naive Cublas saxpy version
// nvcc -o vCublas vCublas.cu -lineinfo -arch=sm_61 -lcublas
// nvprof ./vCublas
// Hypothesis: faster kernel
// -> roughly same data processing time (code unchanged from v1, synchronous host <-> device transfers)
// -> kernel seems(?) a bit slower than v1: 4.15ms vs 4.10ms
// -> warm-up (second call to memcopies and saxpy) does not seem to influence the kernel execution time
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    int N = 1<<25;

    int size = N * sizeof(float);

    float *A, *B, *C;
    float *d_A, *d_B;

    A = (float *)malloc(size);
    B = (float *)malloc(size);
    C = (float *)malloc(size);

    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 0.0f;
    }
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;

    // warm-up (does not seem to influence the kernel execution time)
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cublasSaxpy(handle, N, &alpha, d_A, 1, d_B, 1);

    cudaMemcpy(C, d_B, size, cudaMemcpyDeviceToHost);

    // second call
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cublasSaxpy(handle, N, &alpha, d_A, 1, d_B, 1);

    cudaMemcpy(C, d_B, size, cudaMemcpyDeviceToHost);


    cublasDestroy(handle);
    free(A); free(B); free(C);
    cudaFree(d_A); cudaFree(d_B);

    return 0;
}