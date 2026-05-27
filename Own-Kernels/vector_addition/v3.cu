// Naive version + more data-load instructions per thread (grid-stride loop)
// Hypothesis: SM utilization to increase beyond ~20% -> NO
// -> Memory bandwidth seems to increase from 81% to 88% but without noticeable impact on runtime
#include <stdio.h>
#include <time.h>
#include <cmath>


__global__ void add_v3(float *A, float *B, float *C, int N) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    while (idx < N) {
        C[idx] = A[idx] + B[idx];
        idx += gridDim.x * blockDim.x;
    }

}


int main() {
    int N = 1<<25;
    int threads = 256;
    int max_threads_per_SM = 2048;
    int max_blocks_per_SM = max_threads_per_SM / threads;
    int blocks = max_blocks_per_SM * 60; // device-specific: 1050 Ti has 6 SMs

    int size = N * sizeof(float);

    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

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
    cudaMalloc(&d_C, size);


    clock_t t0, t1;
    double tdiff=0.0;

    t0 = clock();

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    add_v3<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    t1 = clock();
    
    tdiff = ((double)(t1-t0))/1000;

    for (int i = 0; i < N; i++) {
        if (std::fabs(C[i] - 3.0f) > 0.1f) {
            printf("Mismatch at i = %i\n", i);
            printf("A[i] = %f\n", A[i]);
            printf("B[i] = %f\n", B[i]);
            printf("C[i] = %f\n", C[i]);
            break;
        }
    }

    free(A); free(B); free(C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    printf ("Time including copying to device: %f ms\n", tdiff);

    return 0;
}