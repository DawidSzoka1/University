#include <stdio.h>              // Dołączanie standardowej biblioteki wejścia/wyjścia (printf itp.)
#include <stdlib.h>             // Dołączanie biblioteki standardowej (malloc, exit, itp.)
#include <math.h>               // Dołączanie biblioteki matematycznej (funkcje matematyczne)
#include <time.h>               // Dołączanie biblioteki do obsługi czasu (clock, clock_t)
#include <cuda_runtime.h>       // Dołączanie biblioteki uruchomieniowej CUDA (funkcje cuda*)
#include <iostream>             // Dołączanie strumieni wejścia/wyjścia C++ (std::cout)
#include <fstream>              // Dołączanie obsługi plików w C++ (std::ofstream)
#include <vector>               // Dołączanie kontenera vector z biblioteki STL
#include <string>               // Dołączanie obsługi łańcuchów znaków (std::string)

// --- KONFIGURACJA ---
#define MAX_ITER 1000           // Definicja stałej preprocesora określającej maksymalną liczbę iteracji
#define C_REAL -0.7f            // Definicja stałej: część rzeczywista liczby zespolonej dla zbioru Julii
#define C_IMAG 0.27015f         // Definicja stałej: część urojona liczby zespolonej dla zbioru Julii

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Błąd CUDA: %s:%d, kod:%d\n", __FILE__, __LINE__, error); \
        exit(1); \
    } \
}
// Powyżej makro CHECK: sprawdza kod błędu funkcji CUDA. Jeśli wystąpi błąd, wypisuje go i kończy program.

// Funkcja pomocnicza do zapisu pliku obrazu
void saveRawImage(const char* filename, unsigned char* data, int width, int height) {
    FILE* f = fopen(filename, "wb"); // Otwiera plik o nazwie 'filename' w trybie zapisu binarnego ("wb")
    if (f) {                         // Sprawdza, czy plik został poprawnie otwarty
        fwrite(data, 1, (size_t)width * height, f); // Zapisuje dane z bufora 'data' do pliku
        fclose(f);                   // Zamyka plik, zwalniając zasoby
    } else {                         // Jeśli nie udało się otworzyć pliku
        printf("Blad zapisu pliku obrazu: %s\n", filename); // Wypisuje komunikat o błędzie
    }
}

// --- CPU FUNCTIONS ---
void computeMandelbrotCPU(unsigned char* img, int width, int height) {
    for (int y = 0; y < height; y++) { // Pętla iterująca po wszystkich wierszach obrazu (współrzędna y)
        for (int x = 0; x < width; x++) { // Pętla iterująca po wszystkich kolumnach obrazu (współrzędna x)
            float jx = 1.5f * (float)(x - width / 2) / (0.5f * width); // Mapowanie piksela x na współrzędną rzeczywistą fraktala
            float jy = (float)(y - height / 2) / (0.5f * height);      // Mapowanie piksela y na współrzędną urojoną fraktala
            float zx = 0.0f, zy = 0.0f; // Inicjalizacja zmiennych Z (część rzeczywista i urojona) na 0
            int iter = 0;               // Licznik iteracji ustawiony na 0
            while ((zx * zx + zy * zy < 4.0f) && (iter < MAX_ITER)) { // Pętla while: dopóki moduł liczby zespolonej < 2 (kwadrat < 4) i nie przekroczono limitu iteracji
                float temp = zx * zx - zy * zy + jx; // Obliczenie nowej części rzeczywistej Z (formuła Mandelbrota: Z^2 + C)
                zy = 2.0f * zx * zy + jy;            // Obliczenie nowej części urojonej Z
                zx = temp;                           // Aktualizacja części rzeczywistej Z
                iter++;                              // Inkrementacja licznika iteracji
            }
            img[y * width + x] = (unsigned char)(iter % 256); // Zapisanie wyniku (liczba iteracji modulo 256) do tablicy obrazu
        }
    }
}

void computeJuliaCPU(unsigned char* img, int width, int height) {
    for (int y = 0; y < height; y++) { // Pętla iterująca po wierszach obrazu
        for (int x = 0; x < width; x++) { // Pętla iterująca po kolumnach obrazu
            float zx = 1.5f * (float)(x - width / 2) / (0.5f * width); // Mapowanie x na część rzeczywistą Z (dla Julii Z jest punktem startowym)
            float zy = (float)(y - height / 2) / (0.5f * height);      // Mapowanie y na część urojoną Z
            int iter = 0;              // Reset licznika iteracji
            while ((zx * zx + zy * zy < 4.0f) && (iter < MAX_ITER)) { // Pętla warunkowa (ucieczka do nieskończoności lub limit)
                float temp = zx * zx - zy * zy + C_REAL; // Obliczenie nowej części rzeczywistej (stała C jest zdefiniowana globalnie)
                zy = 2.0f * zx * zy + C_IMAG;            // Obliczenie nowej części urojonej
                zx = temp;                               // Aktualizacja Zx
                iter++;                                  // Zwiększenie licznika
            }
            img[y * width + x] = (unsigned char)(iter % 256); // Zapis wyniku do tablicy pikseli
        }
    }
}

// --- GPU KERNELS ---
__global__ void mandelbrotKernel(unsigned char* img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Obliczenie globalnej współrzędnej x wątku
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Obliczenie globalnej współrzędnej y wątku
    if (x < width && y < height) { // Sprawdzenie, czy wątek znajduje się w granicach obrazu
        float jx = 1.5f * (float)(x - width / 2) / (0.5f * width); // Mapowanie współrzędnej x na układ Mandelbrota
        float jy = (float)(y - height / 2) / (0.5f * height);      // Mapowanie współrzędnej y na układ Mandelbrota
        float zx = 0.0f, zy = 0.0f; // Inicjalizacja Z na 0
        int iter = 0;               // Licznik iteracji
        while ((zx * zx + zy * zy < 4.0f) && (iter < MAX_ITER)) { // Pętla sprawdzająca warunek zbieżności
            float temp = zx * zx - zy * zy + jx; // Obliczenie nowej wartości rzeczywistej
            zy = 2.0f * zx * zy + jy;            // Obliczenie nowej wartości urojonej
            zx = temp;                           // Aktualizacja Zx
            iter++;                              // Zwiększenie licznika
        }
        img[y * width + x] = (unsigned char)(iter % 256); // Zapisanie wyniku przez dany wątek do pamięci globalnej GPU
    }
}

__global__ void juliaKernel(unsigned char* img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Obliczenie globalnego indeksu x wątku
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Obliczenie globalnego indeksu y wątku
    if (x < width && y < height) { // Zabezpieczenie przed wyjściem poza zakres obrazu
        float zx = 1.5f * (float)(x - width / 2) / (0.5f * width); // Inicjalizacja Zx na podstawie pozycji piksela
        float zy = (float)(y - height / 2) / (0.5f * height);      // Inicjalizacja Zy na podstawie pozycji piksela
        int iter = 0;              // Reset licznika iteracji
        while ((zx * zx + zy * zy < 4.0f) && (iter < MAX_ITER)) { // Pętla obliczeniowa
            float temp = zx * zx - zy * zy + C_REAL; // Część rzeczywista (stała C jest parametrem zbioru Julii)
            zy = 2.0f * zx * zy + C_IMAG;            // Część urojona
            zx = temp;                               // Aktualizacja Zx
            iter++;                                  // Inkrementacja
        }
        img[y * width + x] = (unsigned char)(iter % 256); // Zapis wyniku do pamięci
    }
}

int main() {
    // 1. OTWIERAMY PLIK CSV OD RAZU NA POCZĄTKU
    std::ofstream csv("results.csv"); // Tworzenie obiektu strumienia plikowego i otwarcie pliku "results.csv"
    // Wpisujemy nagłówek
    csv << "type,device,width,height,pixels,timeMs" << std::endl; // Zapis nagłówków kolumn do pliku CSV

    int resolutions[][2] = { // Deklaracja tablicy z rozdzielczościami do przetestowania
        {640, 480},          // Rozdzielczość 1
        {800, 600},          // Rozdzielczość 2
        {1024, 768},         // Rozdzielczość 3
        {1280, 720},         // Rozdzielczość 4
        {1920, 1080},        // Full HD
        {2560, 1440},        // 2K / QHD
        {3840, 2160},        // 4K
        {7680, 4320},        // 8K
        {15360, 8640},       // 16K
        {30000, 30000},      // Ekstremalnie duża rozdzielczość
    };

    printf("--- Start Benchmarku (Zapis CSV w czasie rzeczywistym) ---\n"); // Wypisanie informacji powitalnej na konsolę

    for (auto& res : resolutions) { // Pętla iterująca po każdej zdefiniowanej rozdzielczości
        int w = res[0]; // Pobranie szerokości
        int h = res[1]; // Pobranie wysokości
        size_t size = (size_t)w * h * sizeof(unsigned char); // Obliczenie rozmiaru bufora w bajtach
        long long numPixels = (long long)w * h; // Obliczenie całkowitej liczby pikseli

        unsigned char* d_img; // Wskaźnik na pamięć urządzenia (GPU)
        CHECK(cudaMalloc((void**)&d_img, size)); // Alokacja pamięci na GPU z obsługą błędów
        unsigned char* h_img = (unsigned char*)malloc(size); // Alokacja pamięci w RAM (CPU)

        // --- 1. TEST CPU ---
        if (numPixels <= 2073600) { // Sprawdzenie, czy obraz nie jest zbyt duży dla testu CPU (limit ok. 2Mpix / FullHD)
            // A. Mandelbrot CPU
            printf("[CPU] Mandelbrot %dx%d... ", w, h); fflush(stdout); // Wypisanie informacji o starcie testu CPU
            clock_t start = clock(); // Pobranie czasu startu
            computeMandelbrotCPU(h_img, w, h); // Wywołanie funkcji obliczeniowej na CPU
            float timeMs = ((float)(clock() - start)) / CLOCKS_PER_SEC * 1000.0f; // Obliczenie czasu trwania w milisekundach
            printf("%.2f ms\n", timeMs); // Wypisanie czasu wykonania

            // ZAPIS DO CSV NATYCHMIAST
            csv << "Mandelbrot,CPU," << w << "," << h << "," << numPixels << "," << timeMs << std::endl; // Zapis wyniku do pliku CSV

            // B. Julia CPU
            printf("[CPU] Julia      %dx%d... ", w, h); fflush(stdout); // Info o starcie testu Julii na CPU
            start = clock(); // Reset zegara
            computeJuliaCPU(h_img, w, h); // Obliczenia na CPU
            timeMs = ((float)(clock() - start)) / CLOCKS_PER_SEC * 1000.0f; // Obliczenie czasu
            printf("%.2f ms\n", timeMs); // Wypisanie czasu

            // ZAPIS DO CSV NATYCHMIAST
            csv << "Julia,CPU," << w << "," << h << "," << numPixels << "," << timeMs << std::endl; // Zapis do CSV
        } else {
            printf("[CPU] %dx%d POMINIETE (zbyt duze dla CPU)\n", w, h); // Komunikat o pominięciu testu CPU dla dużych rozdzielczości
        }

        // --- 2. TEST GPU ---
        dim3 threads(16, 16); // Definicja wymiarów bloku wątków (16x16 = 256 wątków na blok)
        dim3 grid((w + 15) / 16, (h + 15) / 16); // Obliczenie wymiarów siatki (grid), aby pokryć cały obraz
        cudaEvent_t start, stop; // Deklaracja zdarzeń CUDA do mierzenia czasu
        cudaEventCreate(&start); cudaEventCreate(&stop); // Inicjalizacja zdarzeń

        // A. Mandelbrot GPU
        cudaEventRecord(start); // Rejestracja zdarzenia startowego
        mandelbrotKernel<<<grid, threads>>>(d_img, w, h); // Uruchomienie kernela na GPU
        cudaEventRecord(stop); // Rejestracja zdarzenia końcowego
        cudaEventSynchronize(stop); // Oczekiwanie na zakończenie wszystkich operacji do momentu 'stop'
        float timeMs = 0; // Zmienna na czas
        cudaEventElapsedTime(&timeMs, start, stop); // Obliczenie czasu między zdarzeniami start i stop
        printf("[GPU] Mandelbzxzxrot %dx%d: %.2f ms\n", w, h, timeMs); // Wypisanie wyniku

        // ZAPIS DO CSV NATYCHMIAST
        csv << "Mandelbrot,GPU," << w << "," << h << "," << numPixels << "," << timeMs << std::endl; // Zapis wyniku do CSV

        // B. Julia GPU
        cudaEventRecord(start); // Start pomiaru czasu
        juliaKernel<<<grid, threads>>>(d_img, w, h); // Uruchomienie kernela Julii na GPU
        cudaEventRecord(stop); // Stop pomiaru czasu
        cudaEventSynchronize(stop); // Synchronizacja
        cudaEventElapsedTime(&timeMs, start, stop); // Obliczenie czasu
        printf("[GPU] Julia      %dx%d: %.2f ms -> ZAPIS OBRAZU\n", w, h, timeMs); // Wypisanie wyniku

        // ZAPIS DO CSV NATYCHMIAST
        csv << "Julia,GPU," << w << "," << h << "," << numPixels << "," << timeMs << std::endl; // Zapis do CSV


        CHECK(cudaFree(d_img)); // Zwolnienie pamięci na GPU
        free(h_img); // Zwolnienie pamięci RAM
    }

    // Zamykamy plik na końcu
    csv.close(); // Zamknięcie pliku CSV z wynikami
    printf("Koniec benchmarku. Wyniki w results.csv są bezpieczne.\n"); // Komunikat końcowy
    // ==========================================
    // BENCHMARK 2: ROZMIAR BLOKU (Liczba wątków)
    // ==========================================
    std::ofstream blockFile("benchmark_blocks.csv"); // Utworzenie drugiego pliku CSV na testy bloków
    blockFile << "label,block_side,total_threads,gpu_time_ms\n"; // Zapis nagłówka

    int bw = 7680, bh = 4320; // Test na 8K - ustawienie sztywnej rozdzielczości
    size_t b_size = (size_t)bw * bh * sizeof(unsigned char); // Obliczenie rozmiaru pamięci
    unsigned char* d_test; // Wskaźnik na pamięć GPU
    CHECK(cudaMalloc(&d_test, b_size)); // Alokacja pamięci GPU

    std::vector<int> block_sides = {2, 4, 6, 8, 12, 16, 20, 24, 28, 32}; // Wektor zawierający różne rozmiary boku bloku do przetestowania

    printf("\n--- Start Benchmarku 2: Rozmiar Bloku (8K) ---\n"); // Info o starcie drugiego testu

    for(int b : block_sides) { // Pętla po rozmiarach bloków
        dim3 threads(b, b); // Ustawienie wymiarów bloku (b x b)
        dim3 grid((bw + b - 1) / b, (bh + b - 1) / b); // Przeliczenie rozmiaru siatki dla nowego bloku

        cudaEvent_t start, stop; // Zmienne czasowe CUDA
        cudaEventCreate(&start); cudaEventCreate(&stop); // Tworzenie zdarzeń

        cudaEventRecord(start); // Start zegara
        mandelbrotKernel<<<grid, threads>>>(d_test, bw, bh); // Uruchomienie kernela z danym rozmiarem bloku
        cudaEventRecord(stop); // Stop zegara
        cudaEventSynchronize(stop); // Czekanie na koniec

        float ms = 0; // Zmienna na czas
        cudaEventElapsedTime(&ms, start, stop); // Obliczenie czasu trwania

        int total_threads = b * b; // Całkowita liczba wątków w bloku
        std::string label = std::to_string(b) + "x" + std::to_string(b); // Etykieta np. "16x16"
        printf("Blok %5s | Czas: %8.4f ms\n", label.c_str(), ms); // Wypisanie wyniku na ekran
        blockFile << label << "," << b << "," << total_threads << "," << ms << "\n"; // Zapis wyniku do pliku
    }

    blockFile.close(); // Zamknięcie pliku CSV
    CHECK(cudaFree(d_test)); // Zwolnienie pamięci testowej GPU
    printf("\nKoniec benchmarku. Wyniki zapisano do CSV.\n"); // Komunikat końcowy

    return 0; // Zakończenie programu
}