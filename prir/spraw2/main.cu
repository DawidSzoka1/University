#include <stdio.h>
#define N 3

__global__ void add(int *dev_A, int *dev_B) {
    // i to jest wiersz
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    // j to jest kolumna
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += dev_A[i * N + k]; // po wierszu sumowanie
        }

        for (int k = 0; k < N; k++) {
            sum += dev_A[k * N + j]; // po kolumnie
        }

        dev_B[i * N + j] = sum;
    }
}


int main() {
    int *A = (int *)malloc(N * N * sizeof(int));
    int *B = (int *)malloc(N * N * sizeof(int));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = rand() % 10;
        }
    }
    int *dev_A, *dev_B;
    cudaMalloc((void**)&dev_A, N * N * sizeof(int));
    cudaMalloc((void**)&dev_B, N * N * sizeof(int));
    cudaMemcpy(dev_A, A, N *N * sizeof(int), cudaMemcpyHostToDevice);

    // watki
    dim3 threadsPerBlock(N, N, 1);
    // podzial na bloki
    dim3 numBlocks(ceil(N/8.0), ceil(N/8.0), 1);


    add<<<numBlocks, threadsPerBlock>>>(dev_A, dev_B);
    cudaMemcpy(B, dev_B, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("B = %d", B[90]);
    free(A); free(B);
    cudaFree(dev_A);
    cudaFree(dev_B);

    return 0;
}