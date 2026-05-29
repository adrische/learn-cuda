#include <iostream>

/*
    Result is:
    
    ld.global.u32 %r2, [%rd2+4];
    ld.global.u32 %r3, [%rd2+8];
    add.s32 %r1, %r2, %r3;
    st.global.u32 [%rd2], %r1
*/

__global__ void add(int * arr) {
    asm("add.s32 %0, %1, %2;"
        : "=r"(arr[0])
        : "r"(arr[1]), "r"(arr[2])
    );
}

int main() {
    int i[3] = {0, 1, 2};

    int* d_i;
    cudaMalloc(&d_i, sizeof(int) * 3);
    cudaMemcpy(d_i, i, sizeof(int) * 3, cudaMemcpyHostToDevice);

    add<<<1,1>>>(d_i);

    cudaMemcpy(i, d_i, sizeof(int) * 3, cudaMemcpyDeviceToHost);

    printf("i[0] = %i\n", i[0]);

}