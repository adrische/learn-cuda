// Attempt to copy host to device, but then access the data in L2
// The below code only gives 50% L2 Hit Rate, for both kernel calls.
// - the warm up does not seem to have an effect on the second kernel call
// - not 100% as expected of A and B are in L2 when calling the kernel the first time
// - I also tried overwriting the L2 cache by copying a large amount of irrelevant data host -> device,
//   expecting 0% L2 hit rate after this. 
//   However, this was not achieved as well, hit rate was still around 10%.

const int L2bytes = 1048576; // from deviceQuery, L2 Cache Size

__global__ void access(float *A, float *B, int N) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    float sum = 0.0f;
    if (idx <  N) {
        // B[idx] = A[idx];
        sum += A[idx];
        sum += B[idx];
    }

    if (idx == 0) {
        atomicAdd(&B[0], sum);
    }
}


int main() {
    
    int N = L2bytes / 2 / sizeof(float); // save both A and B in L2
    
    int size = N * sizeof(float);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float *A, *B, *Big;
    float *d_A, *d_B, *d_Big;

    int Nbig = 1000 * N;
    int sizebig = 1000 * size;

    A = (float *)malloc(size);
    B = (float *)malloc(size);
    Big = (float *)malloc(sizebig);

    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 0.0f;
    }

    for (int i=0 ; i < Nbig; i++) {
        Big[i] = 10.0f;
    }
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_Big, sizebig);

    // After this copy operation, L2 should contain A, it is small enough - and there should be space left for B
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // cudaDeviceSynchronize();    

    access<<<blocks, threads>>>(d_A, d_B, N);
    
    
    // "warm cache"?
    cudaDeviceSynchronize();    
    access<<<blocks, threads>>>(d_A, d_B, N);
    
    cudaDeviceSynchronize();

    free(A); free(B); free(Big);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_Big);

    
    return 0;
}