#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#include <fstream>
#include <iostream>

#define ALPHA 0.1f
#define ITERATIONS 50

// Funkcja CPU
void heat_cpu(float* current, float* next, int N) {
    for (int y = 1; y < N - 1; y++) {
        for (int x = 1; x < N - 1; x++) {
            int idx = y * N + x;
            next[idx] = current[idx] + ALPHA * (
                current[idx + 1] + current[idx - 1] +
                current[idx + N] + current[idx - N] - 4 * current[idx]
            );
        }
    }
}

// Kernel GPU
__global__ void heat_gpu_kernel(float* current, float* next, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < N - 1 && y > 0 && y < N - 1) {
        int idx = y * N + x;
        next[idx] = current[idx] + ALPHA * (
            current[idx + 1] + current[idx - 1] +
            current[idx + N] + current[idx - N] - 4 * current[idx]
        );
    }
}

void run_benchmark(std::ofstream &file, int N) {
    size_t size = N * N * sizeof(float);
    float *h_curr = (float*)malloc(size);
    float *h_next = (float*)malloc(size);

    // Inicjalizacja
    for(int i=0; i<N*N; i++) h_curr[i] = 0.0f;
    h_curr[(N/2)*N + (N/2)] = 100.0f;

    // --- TEST CPU ---
    clock_t start_c = clock();
    for(int i=0; i<ITERATIONS; i++) {
        heat_cpu(h_curr, h_next, N);
        float* t = h_curr; h_curr = h_next; h_next = t;
    }
    double time_cpu = (double)(clock() - start_c) / CLOCKS_PER_SEC / ITERATIONS;

    // --- TEST GPU ---
    float *d_curr, *d_next;
    cudaMalloc(&d_curr, size);
    cudaMalloc(&d_next, size);

    int block_sizes[] = {8, 16, 32};
    double gpu_results[3];

    for(int b=0; b<3; b++) {
        int bs = block_sizes[b];
        cudaMemcpy(d_curr, h_curr, size, cudaMemcpyHostToDevice);

        dim3 block(bs, bs);
        dim3 grid((N + bs - 1) / bs, (N + bs - 1) / bs);

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);

        for(int i=0; i<ITERATIONS; i++) {
            heat_gpu_kernel<<<grid, block>>>(d_curr, d_next, N);
            float* t = d_curr; d_curr = d_next; d_next = t;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        gpu_results[b] = (ms / 1000.0) / ITERATIONS;
    }

    // Zapis do CSV: N, CPU, GPU_8, GPU_16, GPU_32
    file << N << "," << time_cpu << "," << gpu_results[0] << "," << gpu_results[1] << "," << gpu_results[2] << "\n";
    printf("N=%d completed.\n", N);

    cudaFree(d_curr); cudaFree(d_next);
    free(h_curr); free(h_next);
}

int main() {
    std::ofstream file("results.csv");
    file << "N,CPU,GPU_8,GPU_16,GPU_32\n";

    int sizes[] = {128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072, 4096, 8192, 16384};
    for(int i=0; i<12; i++) {
        run_benchmark(file, sizes[i]);
    }

    file.close();
    return 0;
}