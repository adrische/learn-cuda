// Graph creation vs kernel launch overhead for a number of streams and a number of kernel launches per stream
// adjusted from https://github.com/olcf/cuda-training-series/blob/master/exercises/hw13/Solutions/axpy_stream_capture_solution.cu
//
// -> starting around 8 empty kernels launched per stream, for 8 streams, 
//    the graph creation + launch time is less than the regular kernel launches:
//
// Graph creation time: 0.271986 ms.
// Graph launch time: 0.060696 ms.
// Normal kernel launch incl. overhead: 0.337254 ms.

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <ctime>
#include <ratio>
#include <chrono>
#include <iostream>


__global__ void emptyKernel() {

}

int main() {
    using namespace std::chrono;

    int N_streams = 8;
    int N_workPerStream = 8;

    // We don't count stream creation overhead
    cudaStream_t streams[N_streams];
    for (int s=0; s < N_streams; s++) cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking);
    cudaDeviceSynchronize();


    high_resolution_clock::time_point t0 = high_resolution_clock::now();
    
    // Graph creation overhead
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    for (int s = 0; s < N_streams; s++) cudaStreamBeginCapture(streams[s], cudaStreamCaptureModeGlobal);
    
    for (int w = 0; w < N_workPerStream; w++) {
        for (int s = 0; s < N_streams; s++) {
            emptyKernel<<<1024, 256, 0, streams[s]>>>();
        }
    }
    
    for (int s=0; s < N_streams; s++) cudaStreamEndCapture(streams[s], &graph);
    
    cudaGraphExec_t instance;
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    cudaDeviceSynchronize();
    // Graph creation overhead - end
    
    
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    

    
    // Graph launch
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();
    // Graph launch - end

    high_resolution_clock::time_point t2 = high_resolution_clock::now();



    // Regular launch time
    for (int w = 0; w < N_workPerStream; w++) {
        for (int s = 0; s < N_streams; s++) {
            emptyKernel<<<1024, 256, 0, streams[s]>>>();
        }
    }
    cudaDeviceSynchronize();
    // Regular launch time - end

    high_resolution_clock::time_point t3 = high_resolution_clock::now();



    duration<double> total_time_graph_creation_overhead = duration_cast<duration<double>>(t1 - t0);
    duration<double> total_time_graph_launch = duration_cast<duration<double>>(t2 - t1);
    duration<double> total_time_normal_launch = duration_cast<duration<double>>(t3 - t2);

    std::cout << "Graph creation time: " << total_time_graph_creation_overhead.count() * 1000
              << " ms.\nGraph launch time: " << total_time_graph_launch.count() * 1000 
              << " ms.\nNormal kernel launch incl. overhead: " << total_time_normal_launch.count() * 1000 << " ms." << std::endl;

    
}