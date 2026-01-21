#include <stdio.h>                      // Dołączenie standardowej biblioteki wejścia/wyjścia (printf)
#include <cuda_runtime.h>               // Dołączenie biblioteki uruchomieniowej CUDA (zarządzanie pamięcią, wątkami)
#include <device_launch_parameters.h>   // Dołączenie definicji zmiennych wbudowanych (blockIdx, threadIdx itp.)
#include <time.h>                       // Dołączenie biblioteki do obsługi czasu (clock_t)
#include <fstream>                      // Dołączenie biblioteki do obsługi plików (std::ofstream)
#include <iostream>                     // Dołączenie biblioteki strumieni wejścia/wyjścia C++

#define ALPHA 0.1f                      // Definicja stałej współczynnika dyfuzji ciepła
#define ITERATIONS 50                   // Definicja liczby iteracji symulacji w każdym teście

// Funkcja CPU
void heat_cpu(float* current, float* next, int N) { // Deklaracja funkcji wykonującej obliczenia na procesorze
    for (int y = 1; y < N - 1; y++) {               // Pętla iterująca po wierszach (pomijając krawędzie)
        for (int x = 1; x < N - 1; x++) {           // Pętla iterująca po kolumnach (pomijając krawędzie)
            int idx = y * N + x;                    // Obliczenie jednowymiarowego indeksu w tablicy dla współrzędnych (x, y)
            next[idx] = current[idx] + ALPHA * (    // Obliczenie nowej wartości temperatury na podstawie sąsiadów
                current[idx + 1] + current[idx - 1] + // Dodanie sąsiada prawego i lewego
                current[idx + N] + current[idx - N] - 4 * current[idx] // Dodanie sąsiada dolnego, górnego i odjęcie 4x środka
            );                                      // Zamknięcie nawiasu wzoru
        }                                           // Koniec pętli po x
    }                                               // Koniec pętli po y
}                                                   // Koniec funkcji CPU

// Kernel GPU
__global__ void heat_gpu_kernel(float* current, float* next, int N) { // Deklaracja kernela - funkcji uruchamianej na karcie graficznej
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // Obliczenie globalnej współrzędnej x wątku w siatce
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // Obliczenie globalnej współrzędnej y wątku w siatce

    if (x > 0 && x < N - 1 && y > 0 && y < N - 1) { // Sprawdzenie, czy wątek znajduje się wewnątrz obszaru (pomijając ramkę)
        int idx = y * N + x;                        // Obliczenie spłaszczonego indeksu w pamięci globalnej
        next[idx] = current[idx] + ALPHA * (        // Zastosowanie tego samego wzoru dyfuzji co na CPU
            current[idx + 1] + current[idx - 1] +   // Sąsiedzi poziomi
            current[idx + N] + current[idx - N] - 4 * current[idx] // Sąsiedzi pionowi i punkt centralny
        );                                          // Zamknięcie wzoru
    }                                               // Koniec warunku if
}                                                   // Koniec kernela

void run_benchmark(std::ofstream &file, int N) {    // Funkcja przeprowadzająca test wydajności dla zadanego rozmiaru N
    size_t size = N * N * sizeof(float);            // Obliczenie rozmiaru pamięci potrzebnej dla macierzy NxN w bajtach
    float *h_curr = (float*)malloc(size);           // Alokacja pamięci RAM dla tablicy obecnej (Host)
    float *h_next = (float*)malloc(size);           // Alokacja pamięci RAM dla tablicy następnej (Host)

    // Inicjalizacja
    for(int i=0; i<N*N; i++) h_curr[i] = 0.0f;      // Wyzerowanie całej tablicy w pętli
    h_curr[(N/2)*N + (N/2)] = 100.0f;               // Ustawienie gorącego punktu (100 stopni) na środku siatki

    // --- TEST CPU ---
    clock_t start_c = clock();                      // Pobranie czasu startu dla CPU
    for(int i=0; i<ITERATIONS; i++) {               // Pętla wykonująca zadaną liczbę iteracji symulacji
        heat_cpu(h_curr, h_next, N);                // Wywołanie funkcji obliczeniowej CPU
        float* t = h_curr; h_curr = h_next; h_next = t; // Zamiana wskaźników (ping-pong buffering) - wynik staje się wejściem
    }                                               // Koniec pętli CPU
    double time_cpu = (double)(clock() - start_c) / CLOCKS_PER_SEC / ITERATIONS; // Obliczenie średniego czasu jednej iteracji w sekundach

    // --- TEST GPU ---
    float *d_curr, *d_next;                         // Deklaracja wskaźników na pamięć GPU (Device)
    cudaMalloc(&d_curr, size);                      // Alokacja pamięci VRAM na tablicę obecną
    cudaMalloc(&d_next, size);                      // Alokacja pamięci VRAM na tablicę następną

    int block_sizes[] = {8, 16, 32};                // Tablica z testowanymi rozmiarami bloków wątków (8x8, 16x16, 32x32)
    double gpu_results[3];                          // Tablica na wyniki czasowe dla każdego rozmiaru bloku

    for(int b=0; b<3; b++) {                        // Pętla po trzech konfiguracjach bloków
        int bs = block_sizes[b];                    // Pobranie bieżącego rozmiaru boku bloku
        cudaMemcpy(d_curr, h_curr, size, cudaMemcpyHostToDevice); // Skopiowanie danych początkowych z RAM do VRAM (reset stanu)

        dim3 block(bs, bs);                         // Definicja wymiarów bloku wątków (bs x bs)
        dim3 grid((N + bs - 1) / bs, (N + bs - 1) / bs); // Obliczenie liczby bloków w siatce, aby pokryć cały obraz NxN

        cudaEvent_t start, stop;                    // Deklaracja zdarzeń CUDA do precyzyjnego pomiaru czasu
        cudaEventCreate(&start); cudaEventCreate(&stop); // Inicjalizacja zdarzeń
        cudaEventRecord(start);                     // Zapisanie znacznika czasu startu

        for(int i=0; i<ITERATIONS; i++) {           // Pętla iteracji symulacji na GPU
            heat_gpu_kernel<<<grid, block>>>(d_curr, d_next, N); // Uruchomienie kernela na karcie graficznej
            float* t = d_curr; d_curr = d_next; d_next = t; // Zamiana wskaźników w pamięci GPU
        }                                           // Koniec pętli GPU

        cudaEventRecord(stop);                      // Zapisanie znacznika czasu końca
        cudaEventSynchronize(stop);                 // Czekanie na zakończenie wszystkich operacji GPU do momentu 'stop'
        float ms = 0;                               // Zmienna na czas w milisekundach
        cudaEventElapsedTime(&ms, start, stop);     // Obliczenie upływu czasu między zdarzeniami start i stop
        gpu_results[b] = (ms / 1000.0) / ITERATIONS; // Przeliczenie czasu na sekundy i uśrednienie na jedną iterację
    }                                               // Koniec pętli po rozmiarach bloków

    // Zapis do CSV: N, CPU, GPU_8, GPU_16, GPU_32
    file << N << "," << time_cpu << "," << gpu_results[0] << "," << gpu_results[1] << "," << gpu_results[2] << "\n"; // Zapis wyników do pliku
    printf("N=%d completed.\n", N);                 // Wypisanie informacji o postępie w konsoli

    cudaFree(d_curr); cudaFree(d_next);             // Zwolnienie pamięci na karcie graficznej
    free(h_curr); free(h_next);                     // Zwolnienie pamięci RAM
}                                                   // Koniec funkcji benchmarku

int main() {                                        // Funkcja główna programu
    std::ofstream file("results.csv");              // Utworzenie i otwarcie pliku results.csv do zapisu
    file << "N,CPU,GPU_8,GPU_16,GPU_32\n";          // Zapisanie nagłówka kolumn w pliku CSV

    int sizes[] = {128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072, 4096, 8192, 16384}; // Tablica z rozmiarami siatek do przetestowania
    for(int i=0; i<12; i++) {                       // Pętla iterująca po wszystkich zdefiniowanych rozmiarach
        run_benchmark(file, sizes[i]);              // Uruchomienie benchmarku dla danego rozmiaru
    }                                               // Koniec pętli głównej

    file.close();                                   // Zamknięcie pliku CSV
    return 0;                                       // Zakończenie programu z kodem sukcesu 0
}                                                   // Koniec funkcji main