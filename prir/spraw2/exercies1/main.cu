#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// --- KONFIGURACJA ---
#define MAX_ITER 1000
#define C_REAL -0.7f
#define C_IMAG 0.27015f

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Błąd CUDA: %s:%d, kod:%d\n", __FILE__, __LINE__, error); \
        exit(1); \
    } \
}

// Funkcja pomocnicza do zapisu pliku obrazu
void saveRawImage(const char* filename, unsigned char* data, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (f) {
        fwrite(data, 1, (size_t)width * height, f);
        fclose(f);
    } else {
        printf("Blad zapisu pliku obrazu: %s\n", filename);
    }
}

// --- CPU FUNCTIONS ---
void computeMandelbrotCPU(unsigned char* img, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float jx = 1.5f * (float)(x - width / 2) / (0.5f * width);
            float jy = (float)(y - height / 2) / (0.5f * height);
            float zx = 0.0f, zy = 0.0f;
            int iter = 0;
            while ((zx * zx + zy * zy < 4.0f) && (iter < MAX_ITER)) {
                float temp = zx * zx - zy * zy + jx;
                zy = 2.0f * zx * zy + jy;
                zx = temp;
                iter++;
            }
            img[y * width + x] = (unsigned char)(iter % 256);
        }
    }
}

void computeJuliaCPU(unsigned char* img, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float zx = 1.5f * (float)(x - width / 2) / (0.5f * width);
            float zy = (float)(y - height / 2) / (0.5f * height);
            int iter = 0;
            while ((zx * zx + zy * zy < 4.0f) && (iter < MAX_ITER)) {
                float temp = zx * zx - zy * zy + C_REAL;
                zy = 2.0f * zx * zy + C_IMAG;
                zx = temp;
                iter++;
            }
            img[y * width + x] = (unsigned char)(iter % 256);
        }
    }
}

// --- GPU KERNELS ---
__global__ void mandelbrotKernel(unsigned char* img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float jx = 1.5f * (float)(x - width / 2) / (0.5f * width);
        float jy = (float)(y - height / 2) / (0.5f * height);
        float zx = 0.0f, zy = 0.0f;
        int iter = 0;
        while ((zx * zx + zy * zy < 4.0f) && (iter < MAX_ITER)) {
            float temp = zx * zx - zy * zy + jx;
            zy = 2.0f * zx * zy + jy;
            zx = temp;
            iter++;
        }
        img[y * width + x] = (unsigned char)(iter % 256);
    }
}

__global__ void juliaKernel(unsigned char* img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float zx = 1.5f * (float)(x - width / 2) / (0.5f * width);
        float zy = (float)(y - height / 2) / (0.5f * height);
        int iter = 0;
        while ((zx * zx + zy * zy < 4.0f) && (iter < MAX_ITER)) {
            float temp = zx * zx - zy * zy + C_REAL;
            zy = 2.0f * zx * zy + C_IMAG;
            zx = temp;
            iter++;
        }
        img[y * width + x] = (unsigned char)(iter % 256);
    }
}

int main() {
    // 1. OTWIERAMY PLIK CSV OD RAZU NA POCZĄTKU
    std::ofstream csv("results.csv");
    // Wpisujemy nagłówek
    csv << "type,device,width,height,pixels,timeMs" << std::endl;

    int resolutions[][2] = {
        {640, 480},
        {800, 600},
        {1024, 768},
        {1280, 720},
        {1920, 1080},
        {2560, 1440},
        {3840, 2160},
        {7680, 4320},
        {15360, 8640},
        {30000, 30000},
    };

    printf("--- Start Benchmarku (Zapis CSV w czasie rzeczywistym) ---\n");

    for (auto& res : resolutions) {
        int w = res[0];
        int h = res[1];
        size_t size = (size_t)w * h * sizeof(unsigned char);
        long long numPixels = (long long)w * h;

        unsigned char* d_img;
        CHECK(cudaMalloc((void**)&d_img, size));
        unsigned char* h_img = (unsigned char*)malloc(size);

        // --- 1. TEST CPU ---
        if (numPixels <= 2073600) {
            // A. Mandelbrot CPU
            printf("[CPU] Mandelbrot %dx%d... ", w, h); fflush(stdout);
            clock_t start = clock();
            computeMandelbrotCPU(h_img, w, h);
            float timeMs = ((float)(clock() - start)) / CLOCKS_PER_SEC * 1000.0f;
            printf("%.2f ms\n", timeMs);

            // ZAPIS DO CSV NATYCHMIAST
            csv << "Mandelbrot,CPU," << w << "," << h << "," << numPixels << "," << timeMs << std::endl;

            // B. Julia CPU
            printf("[CPU] Julia      %dx%d... ", w, h); fflush(stdout);
            start = clock();
            computeJuliaCPU(h_img, w, h);
            timeMs = ((float)(clock() - start)) / CLOCKS_PER_SEC * 1000.0f;
            printf("%.2f ms\n", timeMs);

            // ZAPIS DO CSV NATYCHMIAST
            csv << "Julia,CPU," << w << "," << h << "," << numPixels << "," << timeMs << std::endl;
        } else {
            printf("[CPU] %dx%d POMINIETE (zbyt duze dla CPU)\n", w, h);
        }

        // --- 2. TEST GPU ---
        dim3 threads(16, 16);
        dim3 grid((w + 15) / 16, (h + 15) / 16);
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        // A. Mandelbrot GPU
        cudaEventRecord(start);
        mandelbrotKernel<<<grid, threads>>>(d_img, w, h);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float timeMs = 0;
        cudaEventElapsedTime(&timeMs, start, stop);
        printf("[GPU] Mandelbzxzxrot %dx%d: %.2f ms\n", w, h, timeMs);

        // ZAPIS DO CSV NATYCHMIAST
        csv << "Mandelbrot,GPU," << w << "," << h << "," << numPixels << "," << timeMs << std::endl;

        // B. Julia GPU
        cudaEventRecord(start);
        juliaKernel<<<grid, threads>>>(d_img, w, h);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeMs, start, stop);
        printf("[GPU] Julia      %dx%d: %.2f ms -> ZAPIS OBRAZU\n", w, h, timeMs);

        // ZAPIS DO CSV NATYCHMIAST
        csv << "Julia,GPU," << w << "," << h << "," << numPixels << "," << timeMs << std::endl;


        CHECK(cudaFree(d_img));
        free(h_img);
    }

    // Zamykamy plik na końcu
    csv.close();
    printf("Koniec benchmarku. Wyniki w results.csv są bezpieczne.\n");
    // ==========================================
    // BENCHMARK 2: ROZMIAR BLOKU (Liczba wątków)
    // ==========================================
    std::ofstream blockFile("benchmark_blocks.csv");
    blockFile << "label,block_side,total_threads,gpu_time_ms\n";

    int bw = 7680, bh = 4320; // Test na 8K
    size_t b_size = (size_t)bw * bh * sizeof(unsigned char);
    unsigned char* d_test;
    CHECK(cudaMalloc(&d_test, b_size));

    std::vector<int> block_sides = {2, 4, 6, 8, 12, 16, 20, 24, 28, 32};

    printf("\n--- Start Benchmarku 2: Rozmiar Bloku (8K) ---\n");

    for(int b : block_sides) {
        dim3 threads(b, b);
        dim3 grid((bw + b - 1) / b, (bh + b - 1) / b);

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        cudaEventRecord(start);
        mandelbrotKernel<<<grid, threads>>>(d_test, bw, bh);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        int total_threads = b * b;
        std::string label = std::to_string(b) + "x" + std::to_string(b);
        printf("Blok %5s | Czas: %8.4f ms\n", label.c_str(), ms);
        blockFile << label << "," << b << "," << total_threads << "," << ms << "\n";
    }

    blockFile.close();
    CHECK(cudaFree(d_test));
    printf("\nKoniec benchmarku. Wyniki zapisano do CSV.\n");

    return 0;
}