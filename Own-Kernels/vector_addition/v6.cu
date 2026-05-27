// Try to overlap host -> device, device -> host, global -> shared memory copies
// -> Visually (in Nsight Systems), the two copy operations and the kernel overlap.

const int shared_mem_per_block_in_bytes = 49152;
const int N_shared = shared_mem_per_block_in_bytes / sizeof(float);


__global__ void access_shared(float *A, int N) {
    __shared__ float smem[N_shared];
    float sum = 0.0;

    for (int i = 0; i < N; i++) {
        smem[i] = A[i];
    }

    for (int i = 0; i < N; i++) {
        sum += smem[i];
    }

    smem[0] = sum; // do something with the value so the instructions get actually executed
}


int main() {
    int N = 1<<25;

    int n_streams = 3; // HtD, DtH, GtS

    cudaStream_t streams[n_streams];
    for (int i = 0; i < n_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int size = N * sizeof(float);

    float *HtD, *DtH, *GtS;
    float *d_HtD, *d_DtH, *d_GtS;

    cudaHostAlloc(&HtD, size, cudaHostAllocDefault);
    cudaHostAlloc(&DtH, size, cudaHostAllocDefault);
    cudaHostAlloc(&GtS, shared_mem_per_block_in_bytes, cudaHostAllocDefault);

    for (int i = 0; i < N; i++) {
        HtD[i] = 1.0f;
        DtH[i] = 2.0f;
    }

    for (int i = 0; i < N_shared; i++) {
        GtS[i] = 3.0f;
    }

    
    cudaMalloc(&d_HtD, size);
    cudaMalloc(&d_DtH, size);
    cudaMalloc(&d_GtS, shared_mem_per_block_in_bytes);

    // prepare
    cudaMemcpyAsync(d_DtH, DtH, size, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_GtS, GtS, shared_mem_per_block_in_bytes, cudaMemcpyHostToDevice, 0);

    cudaDeviceSynchronize();


    // HtD
    cudaMemcpyAsync(d_HtD, HtD, size, cudaMemcpyHostToDevice, streams[0]);
    
    // DtH
    cudaMemcpyAsync(DtH, d_DtH, size, cudaMemcpyDeviceToHost, streams[1]);

    // GtS - do everything in one thread so the kernel takes long to execute
    access_shared<<<1, 1, 0, streams[2]>>>(d_GtS, N_shared);


    cudaDeviceSynchronize();


    cudaFreeHost(HtD); cudaFreeHost(DtH); cudaFreeHost(GtS);
    cudaFree(d_HtD); cudaFree(d_DtH); cudaFree(d_GtS);
    for (int i = 0; i < n_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}