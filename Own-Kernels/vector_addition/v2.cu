// With streams to overlap host -> device and device -> host transfer
// Hypothesis: We have 3 sequential data transfers of the same size: 
// 2 host -> device and one device -> host transfer. 
// In principle (up to epsilon), the host -> device
// and device -> host transfer should overlap,
// saving 1/3 of the sequential memory transfer time.
// -> The argument is false, but that this speedup was actually achieved is a coincidence:
// Vector addition is 'embarassingly parallel', i.e., it is bound by host - device throughput.
#include <stdio.h>
#include <time.h>
#include <cmath>

__global__ void add_v2(float *A, float *B, float *C, int N) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx <  N) {
        C[idx] = A[idx] + B[idx];
    }
}


int main() {
    int N = 1<<25;
    int threads = 256;

    int n_workpackages = 128; // should divide N
    int n_streams = 2;

    cudaStream_t streams[n_streams];
    for (int i = 0; i < n_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }


    int size = N * sizeof(float);

    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    cudaHostAlloc(&A, size, cudaHostAllocDefault);
    cudaHostAlloc(&B, size, cudaHostAllocDefault);
    cudaHostAlloc(&C, size, cudaHostAllocDefault);

    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 0.0f;
    }
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaDeviceSynchronize();


    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);


    int offset;
    int package_length = N / n_workpackages;
    int package_size = sizeof(float) * package_length;
    int blocks = (package_length + threads -1)/threads;
    
    cudaEventRecord(start);
    
    for (int i = 0; i < n_workpackages; i++) {
        offset = i * package_length;
        cudaMemcpyAsync(&d_A[offset], &A[offset], package_size, cudaMemcpyHostToDevice, streams[i % n_streams]);
        cudaMemcpyAsync(&d_B[offset], &B[offset], package_size, cudaMemcpyHostToDevice, streams[i % n_streams]);

        add_v2<<<blocks, threads, 0, streams[i % n_streams]>>>(&d_A[offset], &d_B[offset], &d_C[offset], package_length);

        cudaMemcpyAsync(&C[offset], &d_C[offset], package_size, cudaMemcpyDeviceToHost, streams[i % n_streams]);
    
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // time in ms
    cudaEventDestroy(start); cudaEventDestroy(stop);


    for (int i = 0; i < N; i++) {
        if (std::fabs(C[i] - 3.0f) > 0.1f) {
            printf("Mismatch at i = %i\n", i);
            printf("A[i] = %f\n", A[i]);
            printf("B[i] = %f\n", B[i]);
            printf("C[i] = %f\n", C[i]);
            break;
        }
    }

    
    cudaFreeHost(A); cudaFreeHost(B); cudaFreeHost(C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    for (int i = 0; i < n_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    

    printf ("Time including copying to device: %f ms\n", elapsedTime);

    return 0;
}