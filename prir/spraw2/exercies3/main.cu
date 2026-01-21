#define STB_IMAGE_IMPLEMENTATION       // Definicja implementacji biblioteki stb_image (wymagane przez nagłówek single-file)
#define STB_IMAGE_WRITE_IMPLEMENTATION // Definicja implementacji biblioteki stb_image_write (do zapisu plików)
#include "stb_image.h"                 // Dołączenie nagłówka biblioteki do wczytywania obrazów
#include "stb_image_write.h"           // Dołączenie nagłówka biblioteki do zapisywania obrazów

#include <iostream>      // Dołączenie biblioteki strumieni wejścia/wyjścia (std::cout, std::cerr)
#include <vector>        // Dołączenie biblioteki wektorów (std::vector)
#include <string>        // Dołączenie biblioteki ciągów znaków (std::string)
#include <chrono>        // Dołączenie biblioteki do mierzenia czasu (std::chrono)
#include <cuda_runtime.h> // Dołączenie nagłówków CUDA (funkcje cudaMalloc, cudaMemcpy itp.)
#include <cmath>         // Dołączenie biblioteki matematycznej (fminf, sqrtf)
#include <algorithm>     // Dołączenie biblioteki algorytmów

// Makro do sprawdzania błędów CUDA
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, code:%d, reason: %s\n", __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}
// Powyższe makro sprawdza kod błędu zwracany przez funkcje CUDA.
// Jeśli wystąpi błąd, wypisuje plik, linię, kod błędu i jego opis, a następnie kończy program.

// ==========================================
// 1. ALGORYTMY CPU
// ==========================================

// Funkcja wykonująca efekt sepii na CPU
void sepiaCPU(unsigned char* img, unsigned char* out, int width, int height, int channels) {
    for (int i = 0; i < width * height; ++i) { // Pętla po wszystkich pikselach obrazu
        int idx = i * channels; // Obliczenie indeksu pierwszego kanału (R) dla danego piksela
        float r = img[idx];     // Pobranie wartości kanału czerwonego
        float g = img[idx+1];   // Pobranie wartości kanału zielonego
        float b = img[idx+2];   // Pobranie wartości kanału niebieskiego
        // Obliczenie nowych wartości RGB według wzoru na sepię i ograniczenie do 255 (fminf)
        out[idx]     = (unsigned char)fminf((r * .393) + (g *.769) + (b * .189), 255.0f); // Nowy R
        out[idx+1]   = (unsigned char)fminf((r * .349) + (g *.686) + (b * .168), 255.0f); // Nowy G
        out[idx+2]   = (unsigned char)fminf((r * .272) + (g *.534) + (b * .131), 255.0f); // Nowy B
        if (channels == 4) out[idx+3] = img[idx+3]; // Jeśli obraz ma kanał alfa (przezroczystość), kopiujemy go bez zmian
    }
}

// Funkcja wykonująca rozmycie Gaussa na CPU
void gaussianCPU(unsigned char* img, unsigned char* out, int width, int height, int channels) {
    // Definicja macierzy (jądra) splotu 5x5 dla rozmycia Gaussa
    float kernel[5][5] = {{1,4,7,4,1},{4,16,26,16,4},{7,26,41,26,7},{4,16,26,16,4},{1,4,7,4,1}};
    for (int y = 2; y < height - 2; ++y) { // Pętla po wierszach (pomijając marginesy 2 pikseli)
        for (int x = 2; x < width - 2; ++x) { // Pętla po kolumnach (pomijając marginesy)
            float r=0,g=0,b=0; // Inicjalizacja akumulatorów kolorów
            for(int ky=-2; ky<=2; ++ky) { // Pętla wewnętrzna po jądrze (oś Y)
                for(int kx=-2; kx<=2; ++kx) { // Pętla wewnętrzna po jądrze (oś X)
                    int idx = ((y+ky)*width + (x+kx))*channels; // Obliczenie indeksu piksela sąsiedniego
                    float w = kernel[ky+2][kx+2]; // Pobranie wagi z macierzy kernel
                    r += img[idx]*w; g += img[idx+1]*w; b += img[idx+2]*w; // Ważona suma kolorów
                }
            }
            int o = (y*width+x)*channels; // Indeks piksela wyjściowego
            // Normalizacja wyniku (dzielenie przez sumę wag = 273) i zapis
            out[o] = r/273.0f; out[o+1] = g/273.0f; out[o+2] = b/273.0f;
            if(channels==4) out[o+3] = img[o+3]; // Kopiowanie kanału alfa
        }
    }
}

// Funkcja wykonująca detekcję krawędzi (Sobel) na CPU
void sobelCPU(unsigned char* img, unsigned char* out, int width, int height, int channels) {
    // Definicja masek Sobela dla kierunku X i Y
    int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    int Gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    for (int y = 1; y < height - 1; ++y) { // Pętla po wierszach (margines 1 px)
        for (int x = 1; x < width - 1; ++x) { // Pętla po kolumnach (margines 1 px)
            float gx=0, gy=0; // Akumulatory gradientów
            for(int ky=-1; ky<=1; ++ky) for(int kx=-1; kx<=1; ++kx) { // Pętla po otoczeniu 3x3
                int idx = ((y+ky)*width + (x+kx))*channels; // Indeks sąsiada
                float val = img[idx]; // Pobranie jasności (uproszczenie: bierzemy tylko kanał R lub traktujemy jako grayscale)
                gx += val * Gx[ky+1][kx+1]; // Splot z maską Gx
                gy += val * Gy[ky+1][kx+1]; // Splot z maską Gy
            }
            unsigned char mag = (unsigned char)fminf(sqrtf(gx*gx + gy*gy), 255.0f); // Obliczenie magnitudy gradientu
            int o = (y*width+x)*channels; // Indeks wyjściowy
            out[o] = mag; out[o+1] = mag; out[o+2] = mag; // Zapis wyniku (obraz czarno-biały)
            if(channels==4) out[o+3] = 255; // Pełna nieprzezroczystość dla alfy
        }
    }
}

// ==========================================
// 2. KERNELS GPU
// ==========================================

// Kernel CUDA do efektu sepii
__global__ void sepiaKernel(unsigned char* img, unsigned char* out, int w, int h, int c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Obliczenie globalnej współrzędnej X wątku
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Obliczenie globalnej współrzędnej Y wątku
    if (x < w && y < h) { // Sprawdzenie, czy wątek znajduje się w granicach obrazu
        int i = (y * w + x) * c; // Obliczenie indeksu piksela w jednowymiarowej tablicy
        float r = img[i], g = img[i+1], b = img[i+2]; // Odczyt wartości RGB
        // Obliczenia sepii z zabezpieczeniem zakresu (fminf)
        out[i]   = fminf((r*.393) + (g*.769) + (b*.189), 255.0f);
        out[i+1] = fminf((r*.349) + (g*.686) + (b*.168), 255.0f);
        out[i+2] = fminf((r*.272) + (g*.534) + (b*.131), 255.0f);
        if(c==4) out[i+3] = img[i+3]; // Kopiowanie alfy
    }
}

// Kernel CUDA do rozmycia Gaussa
__global__ void gaussianKernel(unsigned char* img, unsigned char* out, int w, int h, int c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Globalny X
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Globalny Y
    // Spłaszczona tablica jądra filtru (dostęp w pamięci GPU jest szybszy w ten sposób)
    float k[25] = {1,4,7,4,1, 4,16,26,16,4, 7,26,41,26,7, 4,16,26,16,4, 1,4,7,4,1};
    if (x >= 2 && x < w-2 && y >= 2 && y < h-2) { // Warunek brzegowy (omijamy krawędzie)
        float r=0,g=0,b=0; // Zmienne na sumę
        for(int ky=-2; ky<=2; ++ky) for(int kx=-2; kx<=2; ++kx) { // Pętla po oknie 5x5
            int i = ((y+ky)*w + (x+kx))*c; // Indeks piksela w oknie
            float wgt = k[(ky+2)*5+(kx+2)]; // Pobranie wagi z tablicy k
            r+=img[i]*wgt; g+=img[i+1]*wgt; b+=img[i+2]*wgt; // Akumulacja
        }
        int o = (y*w+x)*c; // Indeks wyjściowy
        out[o] = r/273.0f; out[o+1] = g/273.0f; out[o+2] = b/273.0f; // Normalizacja i zapis
        if(c==4) out[o+3] = img[o+3]; // Alfa
    }
}

// Kernel CUDA do detekcji krawędzi (Sobel)
__global__ void sobelKernel(unsigned char* img, unsigned char* out, int w, int h, int c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Globalny X
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Globalny Y
    // Spłaszczone maski Sobela
    int Gx[9] = {-1,0,1, -2,0,2, -1,0,1};
    int Gy[9] = {-1,-2,-1, 0,0,0, 1,2,1};
    if (x >= 1 && x < w-1 && y >= 1 && y < h-1) { // Warunek brzegowy (omijamy skrajne piksele)
        float gx=0, gy=0; // Zmienne gradientu
        for(int ky=-1; ky<=1; ++ky) for(int kx=-1; kx<=1; ++kx) { // Pętla po oknie 3x3
            int i = ((y+ky)*w + (x+kx))*c; // Indeks sąsiada
            float v = img[i]; // Pobranie wartości (uproszczone R)
            gx += v * Gx[(ky+1)*3+(kx+1)]; // Splot X
            gy += v * Gy[(ky+1)*3+(kx+1)]; // Splot Y
        }
        unsigned char mag = fminf(sqrtf(gx*gx+gy*gy), 255.0f); // Obliczenie długości wektora gradientu
        int o = (y*w+x)*c; // Indeks wyjściowy
        out[o] = mag; out[o+1] = mag; out[o+2] = mag; // Zapis wyniku
        if(c==4) out[o+3] = 255; // Alfa
    }
}

// ==========================================
// 3. MAIN (CLI)
// ==========================================

// Funkcja pomocnicza do uruchamiania odpowiedniego kernela
void runAlgorithmGPU(int algo, unsigned char* d_img, unsigned char* d_out, int w, int h, int c, dim3 grid, dim3 block) {
    if (algo == 1) sepiaKernel<<<grid, block>>>(d_img, d_out, w, h, c); // Wywołanie kernela Sepia
    else if (algo == 2) gaussianKernel<<<grid, block>>>(d_img, d_out, w, h, c); // Wywołanie kernela Gaussian
    else if (algo == 3) sobelKernel<<<grid, block>>>(d_img, d_out, w, h, c); // Wywołanie kernela Sobel
    cudaDeviceSynchronize(); // Oczekiwanie na zakończenie obliczeń przez GPU
}

int main(int argc, char** argv) {
    // Sposób użycia: ./image_proc <tryb> <ścieżka_do_pliku> <algorytm>
    // tryb: 1 = Pojedyncze przetwarzanie (zapis wyniku), 2 = Benchmark (zwrot statystyk)

    if (argc < 4) { // Sprawdzenie liczby argumentów
        std::cerr << "Usage: " << argv[0] << " <mode> <file> <algo>\n"; // Komunikat o użyciu
        return 1; // Błąd, wyjście
    }

    int mode = std::stoi(argv[1]); // Konwersja pierwszego argumentu na int (tryb)
    std::string filename = argv[2]; // Drugi argument to nazwa pliku
    int algo = std::stoi(argv[3]); // Trzeci argument to ID algorytmu

    int w, h, c; // Zmienne na szerokość, wysokość i liczbę kanałów
    unsigned char* img = stbi_load(filename.c_str(), &w, &h, &c, 0); // Wczytanie obrazu za pomocą stb_image
    if (!img) { // Jeśli wskaźnik jest pusty (błąd wczytywania)
        std::cerr << "ERR: Failed to load image: " << filename << "\n"; // Wypisanie błędu
        return 1; // Wyjście
    }

    size_t size = w * h * c; // Obliczenie rozmiaru bufora w bajtach
    unsigned char* h_out = (unsigned char*)malloc(size); // Alokacja pamięci RAM na wynik
    unsigned char *d_img, *d_out; // Wskaźniki na pamięć GPU
    CHECK(cudaMalloc(&d_img, size)); // Alokacja pamięci na GPU dla obrazu wejściowego
    CHECK(cudaMalloc(&d_out, size)); // Alokacja pamięci na GPU dla obrazu wyjściowego
    CHECK(cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice)); // Kopiowanie danych obrazu z RAM do GPU

    // --- MODE 1: Pojedyncze przetwarzanie i zapis ---
    if (mode == 1) {
        dim3 block(16, 16); // Definicja bloku wątków (16x16 = 256 wątków)
        dim3 grid((w + 15)/16, (h + 15)/16); // Obliczenie liczby bloków potrzebnych do pokrycia obrazu

        runAlgorithmGPU(algo, d_img, d_out, w, h, c, grid, block); // Warmup (pierwsze uruchomienie, aby rozgrzać GPU)

        auto start = std::chrono::high_resolution_clock::now(); // Start pomiaru czasu
        runAlgorithmGPU(algo, d_img, d_out, w, h, c, grid, block); // Właściwe uruchomienie algorytmu
        auto end = std::chrono::high_resolution_clock::now(); // Stop pomiaru czasu
        float ms = std::chrono::duration<float, std::milli>(end - start).count(); // Obliczenie czasu trwania w ms

        CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost)); // Kopiowanie wyniku z GPU do RAM
        stbi_write_png("wynik.png", w, h, c, h_out, w * c); // Zapis wyniku do pliku PNG

        std::cout << "SUCCESS " << ms << " wynik.png\n"; // Wypisanie sukcesu i czasu wykonania
    }

    // --- MODE 2: Benchmark pojedynczego pliku (dla folderu) ---
    else if (mode == 2) {
        // 1. CPU Time - pomiar czasu procesora
        auto start = std::chrono::high_resolution_clock::now(); // Start zegara CPU
        if (algo == 1) sepiaCPU(img, h_out, w, h, c); // Wywołanie Sepii CPU
        else if (algo == 2) gaussianCPU(img, h_out, w, h, c); // Wywołanie Gaussa CPU
        else if (algo == 3) sobelCPU(img, h_out, w, h, c); // Wywołanie Sobela CPU
        auto end = std::chrono::high_resolution_clock::now(); // Stop zegara CPU
        float cpu_ms = std::chrono::duration<float, std::milli>(end - start).count(); // Obliczenie czasu CPU

        // 2. GPU Time (Standard 16x16) - pomiar czasu karty graficznej
        dim3 block(16, 16); // Blok 16x16
        dim3 grid((w + 15)/16, (h + 15)/16); // Siatka bloków

        // Warmup
        runAlgorithmGPU(algo, d_img, d_out, w, h, c, grid, block); // Rozgrzewka

        start = std::chrono::high_resolution_clock::now(); // Start zegara GPU
        runAlgorithmGPU(algo, d_img, d_out, w, h, c, grid, block); // Właściwe obliczenia
        end = std::chrono::high_resolution_clock::now(); // Stop zegara GPU
        float gpu_ms = std::chrono::duration<float, std::milli>(end - start).count(); // Obliczenie czasu GPU

        // Wypisujemy dane w formacie łatwym dla Pythona
        // Format: RES_DATA <szerokosc> <wysokosc> <czas_cpu> <czas_gpu>
        std::cout << "RES_DATA " << w << " " << h << " " << cpu_ms << " " << gpu_ms << "\n";

        // 3. Block Size Benchmark (Testowanie różnych liczb wątków)
        int blocks[] = {4, 8, 16, 32}; // Tablica rozmiarów boku bloku (4x4, 8x8, itd.)
        for (int b : blocks) { // Pętla po rozmiarach
            dim3 blk(b, b); // Ustawienie wymiaru bloku
            dim3 grd((w + b - 1)/b, (h + b - 1)/b); // Przeliczenie siatki dla nowego bloku

            start = std::chrono::high_resolution_clock::now(); // Start zegara
            runAlgorithmGPU(algo, d_img, d_out, w, h, c, grd, blk); // Uruchomienie kernela
            end = std::chrono::high_resolution_clock::now(); // Stop zegara
            float b_ms = std::chrono::duration<float, std::milli>(end - start).count(); // Obliczenie czasu

            // Format: BLOCK_DATA <liczba_watkow> <czas_ms>
            std::cout << "BLOCK_DATA " << b*b << " " << b_ms << "\n";
        }
    }

    cudaFree(d_img); cudaFree(d_out); // Zwolnienie pamięci GPU
    free(h_out); stbi_image_free(img); // Zwolnienie pamięci RAM i bufora obrazu
    return 0; // Zakończenie programu
}