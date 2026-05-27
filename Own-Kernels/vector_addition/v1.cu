// Naive version
#include <stdio.h>
#include <time.h>

__global__ void add_v1(float *A, float *B, float *C, int N) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx <  N) {
        C[idx] = A[idx] + B[idx];
    }
}


int main() {
    int N = 1<<25;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    int size = N * sizeof(float);

    clock_t t0, t1, t2, t3;
    double tdiff_inclmemcpy=0.0, tdiff_inclallocandfree=0.0;


    t0 = clock();

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


    
    t1 = clock();

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    add_v1<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    t2 = clock();
    

    free(A); free(B); free(C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    t3 = clock();

    tdiff_inclmemcpy = ((double)(t2-t1))/1000;
    tdiff_inclallocandfree = ((double)(t3-t0))/1000;
    
    printf ("Time including copying to device: %f ms\n", tdiff_inclmemcpy);
    printf ("Time including allocating, copying, and freeing: %f ms\n", tdiff_inclallocandfree);

    return 0;
}