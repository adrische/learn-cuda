// Measure launch overhead of starting an empty kernel
// https://forums.developer.nvidia.com/t/any-way-to-measure-the-latency-of-a-kernel-launch/221413
// ~ 0.75 ms for 128 launches with 1024 blocks and 256 threads per block, with no synchronization after a kernel
// and ~1.5 ms with cudaDeviceSynchronize after each kernel call

#include <stdio.h>
#include <time.h>

__global__ void emptyKernel() {
}


int main() {
    int N_launches = 128;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    for (int i = 0; i < N_launches; i++) {
        emptyKernel<<<1024, 256>>>();
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // time in ms
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf ("Total time of %i kernel launches: %f ms\n", N_launches, elapsedTime);

    return 0;
}