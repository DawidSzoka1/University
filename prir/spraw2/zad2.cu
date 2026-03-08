#include "cuda.h"
#include <iostream>
#include <stdio.h>

#define N 10

__global__ void add(int *dev_A, int *dev_B, int *dev_C, int *dev_D) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        dev_D[i * N + j] = dev_A[i * N + j] + dev_B[i * N + j] + dev_C[i * N + j];
    }
}

int main() {
    int *A = (int*)malloc(N*N*sizeof(int));
    int *B = (int*)malloc(N*N*sizeof(int));
    int *C = (int*)malloc(N*N*sizeof(int));
    int *D = (int*)malloc(N*N*sizeof(int));

    for (int i=0; i<N; i++) {
        for (int j = 0; j<N; j++) {
            A[i*N + j] = rand() % 10;
            B[i*N + j] = rand() % 10;
            C[i*N + j] = rand() % 10;
        }
    }
    int *dev_A, *dev_B, *dev_C, *dev_D;
    cudaMalloc((void**)&dev_A, N*N*sizeof(int));
    cudaMalloc((void**)&dev_B, N*N*sizeof(int));
    cudaMalloc((void**)&dev_C, N*N*sizeof(int));
    cudaMalloc((void**)&dev_D, N*N*sizeof(int));

    cudaMemcpy(dev_A, A, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, C, N*N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPreBlock(N, N, 1);
    dim3 dimBlock(1, 1, 1);

    add<<<dimBlock, threadsPreBlock>>>(dev_A, dev_B, dev_C, dev_D);
    cudaMemcpy(D, dev_D, N*N*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            printf("%d ", D[i*N + j]);
        }
    }
    free(A);
    free(B);
    free(C);
    free(D);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFree(dev_D);

    return 0;
}