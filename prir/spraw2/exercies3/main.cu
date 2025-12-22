#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, code:%d, reason: %s\n", __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// ==========================================
// 1. ALGORYTMY CPU
// ==========================================

void sepiaCPU(unsigned char* img, unsigned char* out, int width, int height, int channels) {
    for (int i = 0; i < width * height; ++i) {
        int idx = i * channels;
        float r = img[idx];
        float g = img[idx+1];
        float b = img[idx+2];
        out[idx]     = (unsigned char)fminf((r * .393) + (g *.769) + (b * .189), 255.0f);
        out[idx+1]   = (unsigned char)fminf((r * .349) + (g *.686) + (b * .168), 255.0f);
        out[idx+2]   = (unsigned char)fminf((r * .272) + (g *.534) + (b * .131), 255.0f);
        if (channels == 4) out[idx+3] = img[idx+3];
    }
}

void gaussianCPU(unsigned char* img, unsigned char* out, int width, int height, int channels) {
    float kernel[5][5] = {{1,4,7,4,1},{4,16,26,16,4},{7,26,41,26,7},{4,16,26,16,4},{1,4,7,4,1}};
    for (int y = 2; y < height - 2; ++y) {
        for (int x = 2; x < width - 2; ++x) {
            float r=0,g=0,b=0;
            for(int ky=-2; ky<=2; ++ky) {
                for(int kx=-2; kx<=2; ++kx) {
                    int idx = ((y+ky)*width + (x+kx))*channels;
                    float w = kernel[ky+2][kx+2];
                    r += img[idx]*w; g += img[idx+1]*w; b += img[idx+2]*w;
                }
            }
            int o = (y*width+x)*channels;
            out[o] = r/273.0f; out[o+1] = g/273.0f; out[o+2] = b/273.0f;
            if(channels==4) out[o+3] = img[o+3];
        }
    }
}

void sobelCPU(unsigned char* img, unsigned char* out, int width, int height, int channels) {
    int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    int Gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float gx=0, gy=0;
            for(int ky=-1; ky<=1; ++ky) for(int kx=-1; kx<=1; ++kx) {
                int idx = ((y+ky)*width + (x+kx))*channels;
                float val = img[idx];
                gx += val * Gx[ky+1][kx+1]; gy += val * Gy[ky+1][kx+1];
            }
            unsigned char mag = (unsigned char)fminf(sqrtf(gx*gx + gy*gy), 255.0f);
            int o = (y*width+x)*channels;
            out[o] = mag; out[o+1] = mag; out[o+2] = mag;
            if(channels==4) out[o+3] = 255;
        }
    }
}

// ==========================================
// 2. KERNELS GPU
// ==========================================

__global__ void sepiaKernel(unsigned char* img, unsigned char* out, int w, int h, int c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        int i = (y * w + x) * c;
        float r = img[i], g = img[i+1], b = img[i+2];
        out[i]   = fminf((r*.393) + (g*.769) + (b*.189), 255.0f);
        out[i+1] = fminf((r*.349) + (g*.686) + (b*.168), 255.0f);
        out[i+2] = fminf((r*.272) + (g*.534) + (b*.131), 255.0f);
        if(c==4) out[i+3] = img[i+3];
    }
}

__global__ void gaussianKernel(unsigned char* img, unsigned char* out, int w, int h, int c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float k[25] = {1,4,7,4,1, 4,16,26,16,4, 7,26,41,26,7, 4,16,26,16,4, 1,4,7,4,1};
    if (x >= 2 && x < w-2 && y >= 2 && y < h-2) {
        float r=0,g=0,b=0;
        for(int ky=-2; ky<=2; ++ky) for(int kx=-2; kx<=2; ++kx) {
            int i = ((y+ky)*w + (x+kx))*c;
            float wgt = k[(ky+2)*5+(kx+2)];
            r+=img[i]*wgt; g+=img[i+1]*wgt; b+=img[i+2]*wgt;
        }
        int o = (y*w+x)*c;
        out[o] = r/273.0f; out[o+1] = g/273.0f; out[o+2] = b/273.0f;
        if(c==4) out[o+3] = img[o+3];
    }
}

__global__ void sobelKernel(unsigned char* img, unsigned char* out, int w, int h, int c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int Gx[9] = {-1,0,1, -2,0,2, -1,0,1};
    int Gy[9] = {-1,-2,-1, 0,0,0, 1,2,1};
    if (x >= 1 && x < w-1 && y >= 1 && y < h-1) {
        float gx=0, gy=0;
        for(int ky=-1; ky<=1; ++ky) for(int kx=-1; kx<=1; ++kx) {
            int i = ((y+ky)*w + (x+kx))*c;
            float v = img[i];
            gx += v * Gx[(ky+1)*3+(kx+1)];
            gy += v * Gy[(ky+1)*3+(kx+1)];
        }
        unsigned char mag = fminf(sqrtf(gx*gx+gy*gy), 255.0f);
        int o = (y*w+x)*c;
        out[o] = mag; out[o+1] = mag; out[o+2] = mag;
        if(c==4) out[o+3] = 255;
    }
}

// ==========================================
// 3. MAIN (CLI)
// ==========================================

void runAlgorithmGPU(int algo, unsigned char* d_img, unsigned char* d_out, int w, int h, int c, dim3 grid, dim3 block) {
    if (algo == 1) sepiaKernel<<<grid, block>>>(d_img, d_out, w, h, c);
    else if (algo == 2) gaussianKernel<<<grid, block>>>(d_img, d_out, w, h, c);
    else if (algo == 3) sobelKernel<<<grid, block>>>(d_img, d_out, w, h, c);
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
    // Usage: ./image_proc <mode> <filepath> <algorithm>
    // mode: 1 = Single Process (save result), 2 = Benchmark Single File (return stats)

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <mode> <file> <algo>\n";
        return 1;
    }

    int mode = std::stoi(argv[1]);
    std::string filename = argv[2];
    int algo = std::stoi(argv[3]);

    int w, h, c;
    unsigned char* img = stbi_load(filename.c_str(), &w, &h, &c, 0);
    if (!img) {
        std::cerr << "ERR: Failed to load image: " << filename << "\n";
        return 1;
    }

    size_t size = w * h * c;
    unsigned char* h_out = (unsigned char*)malloc(size);
    unsigned char *d_img, *d_out;
    CHECK(cudaMalloc(&d_img, size));
    CHECK(cudaMalloc(&d_out, size));
    CHECK(cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice));

    // --- MODE 1: Pojedyncze przetwarzanie i zapis ---
    if (mode == 1) {
        dim3 block(16, 16);
        dim3 grid((w + 15)/16, (h + 15)/16);

        runAlgorithmGPU(algo, d_img, d_out, w, h, c, grid, block); // Warmup

        auto start = std::chrono::high_resolution_clock::now();
        runAlgorithmGPU(algo, d_img, d_out, w, h, c, grid, block);
        auto end = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(end - start).count();

        CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
        stbi_write_png("wynik.png", w, h, c, h_out, w * c);

        std::cout << "SUCCESS " << ms << " wynik.png\n";
    }

    // --- MODE 2: Benchmark pojedynczego pliku (dla folderu) ---
    else if (mode == 2) {
        // 1. CPU Time
        auto start = std::chrono::high_resolution_clock::now();
        if (algo == 1) sepiaCPU(img, h_out, w, h, c);
        else if (algo == 2) gaussianCPU(img, h_out, w, h, c);
        else if (algo == 3) sobelCPU(img, h_out, w, h, c);
        auto end = std::chrono::high_resolution_clock::now();
        float cpu_ms = std::chrono::duration<float, std::milli>(end - start).count();

        // 2. GPU Time (Standard 16x16)
        dim3 block(16, 16);
        dim3 grid((w + 15)/16, (h + 15)/16);

        // Warmup
        runAlgorithmGPU(algo, d_img, d_out, w, h, c, grid, block);

        start = std::chrono::high_resolution_clock::now();
        runAlgorithmGPU(algo, d_img, d_out, w, h, c, grid, block);
        end = std::chrono::high_resolution_clock::now();
        float gpu_ms = std::chrono::duration<float, std::milli>(end - start).count();

        // Wypisujemy dane w formacie łatwym dla Pythona
        // Format: RES_DATA <width> <height> <cpu_ms> <gpu_ms>
        std::cout << "RES_DATA " << w << " " << h << " " << cpu_ms << " " << gpu_ms << "\n";

        // 3. Block Size Benchmark (Wątki)
        int blocks[] = {4, 8, 16, 32};
        for (int b : blocks) {
            dim3 blk(b, b);
            dim3 grd((w + b - 1)/b, (h + b - 1)/b);

            start = std::chrono::high_resolution_clock::now();
            runAlgorithmGPU(algo, d_img, d_out, w, h, c, grd, blk);
            end = std::chrono::high_resolution_clock::now();
            float b_ms = std::chrono::duration<float, std::milli>(end - start).count();

            // Format: BLOCK_DATA <threads_per_block> <ms>
            std::cout << "BLOCK_DATA " << b*b << " " << b_ms << "\n";
        }
    }

    cudaFree(d_img); cudaFree(d_out);
    free(h_out); stbi_image_free(img);
    return 0;
}