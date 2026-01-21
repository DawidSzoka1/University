#include <iostream>             // Dołączanie biblioteki strumieni wejścia/wyjścia (std::cout)
#include <vector>               // Dołączanie biblioteki wektorów (std::vector)
#include <cmath>                // Dołączanie biblioteki matematycznej (sqrtf, sinf, cosf)
#include <chrono>               // Dołączanie biblioteki do mierzenia czasu (std::chrono)
#include <cuda_runtime.h>       // Dołączanie nagłówków CUDA (funkcje cudaMalloc, cudaMemcpy itp.)
#include <fstream>              // Dołączanie biblioteki do obsługi plików (std::ofstream)
#include <string>               // Dołączanie biblioteki do obsługi napisów (std::string)

// --- KONFIGURACJA ---
#define INF 2e10f               // Definicja stałej "nieskończoność" dla ray tracingu (bardzo duża liczba)

struct Vec3 {                   // Definicja struktury wektora 3D
    float x, y, z;              // Składowe wektora: x, y, z
    __host__ __device__ Vec3 operator+(const Vec3& b) const { return {x + b.x, y + b.y, z + b.z}; } // Przeciążenie operatora dodawania wektorów
    __host__ __device__ Vec3 operator-(const Vec3& b) const { return {x - b.x, y - b.y, z - b.z}; } // Przeciążenie operatora odejmowania wektorów
    __host__ __device__ Vec3 operator*(float b) const { return {x * b, y * b, z * b}; } // Przeciążenie operatora mnożenia wektora przez skalar
    __host__ __device__ float dot(const Vec3& b) const { return x * b.x + y * b.y + z * b.z; } // Funkcja obliczająca iloczyn skalarny
    __host__ __device__ Vec3 normalize() { // Funkcja normalizująca wektor (zmienia długość na 1)
        float k = 1.0f / sqrtf(x * x + y * y + z * z); // Obliczenie odwrotności długości wektora
        return {x * k, y * k, z * k}; // Zwrócenie znormalizowanego wektora
    }
};

struct Sphere {                 // Definicja struktury kuli (sfery)
    Vec3 center;                // Środek kuli (wektor 3D)
    float radius;               // Promień kuli
    Vec3 color;                 // Kolor kuli (RGB jako wektor 3D)
};

// --- BARDZO GĘSTE TWORZENIE INICJAŁÓW "DS" ---
void createDSInitials(std::vector<Sphere>& spheres) { // Funkcja generująca sfery układające się w inicjały "DS"
    // Zmniejszamy promień i odstępy, żeby mieć WIĘCEJ PUNKTÓW
    // To sprawi, że S będzie gładkie jak linia
    float r = 2.5f;       // Ustawienie małego promienia dla pojedynczej kuli
    float step = 0.8f;    // Ustawienie małego kroku (gęste upakowanie)

    // --- LITERA D (Po lewej, Czerwona) ---
    float dx = -100.0f;   // Pozycja X dla litery D (przesunięcie w lewo)

    // 1. Pionowa kreska (Gęsta)
    for (float y = -100.0f; y <= 100.0f; y += step) { // Pętla generująca pionową linię litery D
        spheres.push_back({{dx, y, 0}, r, {1.0f, 0.2f, 0.2f}}); // Dodanie czerwonej kuli do wektora
    }

    // 2. Brzuszek D (Gęsty)
    // Kąt od -90 stopni do +90 stopni
    for (float angle = -1.57f; angle <= 1.57f; angle += 0.005f) { // Pętla po kącie dla łuku (bardzo mały krok 0.005f)
        float x = dx + 80.0f * cosf(angle); // Obliczenie współrzędnej X na elipsie
        float y = 100.0f * sinf(angle);     // Obliczenie współrzędnej Y na elipsie

        // Rysujemy tylko prawą połówkę elipsy, żeby zamknąć D
        if (x >= dx) { // Warunek sprawdzający, czy jesteśmy po prawej stronie pionowej kreski
            spheres.push_back({{x, y, 0}, r, {1.0f, 0.2f, 0.2f}}); // Dodanie kuli tworzącej łuk D
        }
    }

    // --- LITERA S (Po prawej, Niebieska - NAPRAWIONA CIĄGŁOŚĆ) ---
    float sx = 100.0f;        // Pozycja bazowa X dla litery S (przesunięcie w prawo)
    float radius_s = 45.0f;   // Promień łuków tworzących literę S
    float y_top = radius_s;   // Współrzędna Y środka górnego łuku
    float y_bot = -radius_s;  // Współrzędna Y środka dolnego łuku

    // Matematyka S: Dwa łuki.
    // Górny: od prawej strony (0 rad), przez górę, do lewej strony (PI rad)
    // Dolny: od prawej strony (0 rad), przez dół, do lewej strony (PI rad)
    // Ale musimy je obrócić i połączyć.

    // 1. Górny łuk S (od prawej góry do środka)
    // Kąt od ok. 30 stopni do 270 stopni (w układzie zegarowym)
    // Używamy parametryzacji, żeby stykały się idealnie w punkcie (sx, 0)

    // Górna część (niebieska)
    for (float angle = 0.5f; angle <= 4.713f; angle += 0.005f) { // Pętla generująca górny łuk S
        float x = sx + radius_s * cosf(angle); // Obliczenie X
        float y = y_top + radius_s * sinf(angle); // Obliczenie Y (względem górnego środka)
        spheres.push_back({{x, y, 0}, r, {0.2f, 0.5f, 1.0f}}); // Dodanie niebieskiej kuli
    }

    // Dolna część (niebieska)
    // Musi zaczynać się tam gdzie górna skończyła (mniej więcej w środku S)
    for (float angle = 1.5708f; angle >= -2.6f; angle -= 0.005f) { // Pętla generująca dolny łuk S (iteracja wsteczna)
        float x = sx + radius_s * cosf(angle); // Obliczenie X (ten sam środek X co góra)
        float y = y_bot + radius_s * sinf(angle); // Obliczenie Y (względem dolnego środka)
        spheres.push_back({{x, y, 0}, r, {0.2f, 0.5f, 1.0f}}); // Dodanie niebieskiej kuli
    }
}

// Funkcja przecięcia
__host__ __device__ float hit(const Sphere& s, const Vec3& rayOrigin, const Vec3& rayDir) { // Funkcja obliczająca przecięcie promienia z kulą
    Vec3 oc = rayOrigin - s.center; // Wektor od środka kuli do początku promienia
    float a = rayDir.dot(rayDir);   // Współczynnik A równania kwadratowego (długość wektora kierunkowego do kwadratu)
    float b = 2.0f * oc.dot(rayDir); // Współczynnik B równania kwadratowego
    float c = oc.dot(oc) - s.radius * s.radius; // Współczynnik C równania kwadratowego
    float disc = b * b - 4 * a * c; // Obliczenie delty (wyróżnika)
    if (disc < 0) return -1.0f;     // Jeśli delta ujemna, brak przecięcia -> zwracamy -1
    return (-b - sqrtf(disc)) / (2.0f * a); // Obliczenie mniejszego pierwiastka (bliższego punktu przecięcia)
}

// Ray Tracing
__host__ __device__ void computeColor(int x, int y, int width, int height, Sphere* spheres, int num_spheres, unsigned char* buffer) { // Funkcja obliczająca kolor piksela
    float u = (float)x / width;     // Normalizacja współrzędnej x do zakresu 0-1
    float v = (float)y / height;    // Normalizacja współrzędnej y do zakresu 0-1
    float aspect = (float)width / height; // Obliczenie proporcji obrazu

    Vec3 rayOrigin = {0, 0, 600};   // Ustawienie pozycji kamery (punktu początkowego promieni)
    float screenX = (u * 2.0f - 1.0f) * aspect * 300.0f; // Przeliczenie współrzędnej ekranowej na świat 3D (X)
    float screenY = (1.0f - v * 2.0f) * 300.0f;          // Przeliczenie współrzędnej ekranowej na świat 3D (Y)
    Vec3 rayDir = (Vec3{screenX, screenY, 0} - rayOrigin).normalize(); // Obliczenie znormalizowanego wektora kierunkowego promienia

    float closest_t = INF;          // Inicjalizacja najbliższego trafienia jako nieskończoność
    int idx = -1;                   // Indeks trafionej sfery (-1 oznacza brak trafienia)

    for (int i = 0; i < num_spheres; i++) { // Pętla po wszystkich sferach w scenie
        float t = hit(spheres[i], rayOrigin, rayDir); // Sprawdzenie przecięcia z i-tą sferą
        if (t > 0.0f && t < closest_t) { // Jeśli trafienie jest poprawne i bliższe niż poprzednie
            closest_t = t;          // Aktualizacja najbliższej odległości
            idx = i;                // Zapisanie indeksu najbliższej sfery
        }
    }

    int r = 0, g = 0, b = 0;        // Domyślny kolor tła (czarny)
    if (idx != -1) {                // Jeśli promień w coś trafił
        Sphere s = spheres[idx];    // Pobranie trafionej sfery
        Vec3 hitPoint = rayOrigin + rayDir * closest_t; // Obliczenie punktu przecięcia w 3D
        Vec3 normal = (hitPoint - s.center).normalize(); // Obliczenie wektora normalnego w punkcie trafienia
        Vec3 lightDir = Vec3{-0.5f, 0.5f, 1.0f}.normalize(); // Zdefiniowanie kierunku światła

        // Prosty shading diffuse
        float diff = fmaxf(normal.dot(lightDir), 0.1f); // Obliczenie oświetlenia (iloczyn skalarny normalnej i światła), min 0.1
        Vec3 result = s.color * diff; // Obliczenie koloru wynikowego (kolor sfery * natężenie światła)

        // Dodajmy trochę "połysku" żeby kulki były ładniejsze
        r = (int)(fminf(result.x * 255.0f, 255.0f)); // Konwersja składowej R na int (0-255) z przycięciem
        g = (int)(fminf(result.y * 255.0f, 255.0f)); // Konwersja składowej G na int (0-255) z przycięciem
        b = (int)(fminf(result.z * 255.0f, 255.0f)); // Konwersja składowej B na int (0-255) z przycięciem
    }

    int pIdx = (y * width + x) * 3; // Obliczenie indeksu w płaskim buforze obrazu
    buffer[pIdx] = r;               // Zapisanie składowej R
    buffer[pIdx+1] = g;             // Zapisanie składowej G
    buffer[pIdx+2] = b;             // Zapisanie składowej B
}

__global__ void render_kernel(unsigned char* buffer, int w, int h, Sphere* s, int n) { // Kernel CUDA (funkcja uruchamiana na GPU)
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Obliczenie globalnej współrzędnej x wątku
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Obliczenie globalnej współrzędnej y wątku
    if (x < w && y < h) computeColor(x, y, w, h, s, n, buffer); // Wywołanie funkcji liczącej kolor, jeśli wątek wewnątrz obrazu
}

void render_cpu(unsigned char* buffer, int w, int h, Sphere* s, int n) { // Funkcja renderująca na CPU (dla porównania)
    for (int y = 0; y < h; y++) {       // Pętla po wierszach
        for (int x = 0; x < w; x++) {   // Pętla po kolumnach
            computeColor(x, y, w, h, s, n, buffer); // Obliczenie koloru dla piksela
        }
    }
}

int main() {
    // 1. GENEROWANIE GEOMETRII (TERAZ GĘSTEJ)
    std::vector<Sphere> host_spheres;   // Utworzenie wektora na sfery (pamięć RAM)
    createDSInitials(host_spheres);     // Wygenerowanie inicjałów DS
    int num_spheres = host_spheres.size(); // Pobranie liczby wygenerowanych sfer

    // Informacja dla Ciebie ile punktów powstało
    std::cout << "[INFO] Wygenerowano model skladajacy sie z " << num_spheres << " sfer (punktow).\n"; // Wypisanie liczby sfer

    // 2. ZAPIS GEOMETRII DO PLIKU
    std::ofstream sceneFile("scene_data.csv"); // Otwarcie pliku CSV do zapisu danych sceny
    sceneFile << "x,y,z,r,g,b,radius\n";       // Zapis nagłówka CSV
    for(const auto& s : host_spheres) {        // Pętla po wszystkich sferach
        sceneFile << s.center.x << "," << s.center.y << "," << s.center.z << "," // Zapis współrzędnych
                  << s.color.x << "," << s.color.y << "," << s.color.z << "," << s.radius << "\n"; // Zapis kolorów i promienia
    }
    sceneFile.close();                         // Zamknięcie pliku

    // 3. PRZYGOTOWANIE GPU
    Sphere* d_spheres;                         // Wskaźnik na sfery w pamięci GPU
    cudaMalloc(&d_spheres, num_spheres * sizeof(Sphere)); // Alokacja pamięci na karcie graficznej
    cudaMemcpy(d_spheres, host_spheres.data(), num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice); // Kopiowanie sfer z RAM do GPU

    // ==========================================
    // BENCHMARK 1: ROZDZIELCZOŚCI
    // ==========================================
    std::ofstream benchFile("benchmark_resolution.csv"); // Otwarcie pliku na wyniki benchmarku rozdzielczości
    benchFile << "label,width,height,pixels,cpu_time_ms,gpu_time_ms\n"; // Zapis nagłówka

    std::vector<std::pair<std::string, std::pair<int, int>>> resolutions = { // Lista rozdzielczości do przetestowania
        {"VGA 480p", {640, 480}},
                {"SVGA 600p", {800, 600}},
                {"HD 720p", {1280, 720}},
                {"WXGA", {1366, 768}},
                {"HD+", {1600, 900}},
                {"FHD 1080p", {1920, 1080}},
                {"QHD 1440p", {2560, 1440}},
                {"UHD 4K", {3840, 2160}},
                {"5K", {5120, 2880}},
                {"UHD 8K", {7680, 4320}},
    };

    std::cout << "\n--- Start Benchmarku Rozdzielczosci ---\n"; // Info startowe

    for (auto item : resolutions) {      // Pętla po rozdzielczościach
        std::string label = item.first;  // Pobranie nazwy rozdzielczości
        int w = item.second.first;       // Pobranie szerokości
        int h = item.second.second;      // Pobranie wysokości
        size_t size = (size_t)w * h * 3; // Obliczenie rozmiaru bufora obrazu w bajtach

        unsigned char* d_img;            // Wskaźnik na obraz w pamięci GPU
        cudaMalloc(&d_img, size);        // Alokacja pamięci na obraz na GPU

        unsigned char* h_img = nullptr;  // Wskaźnik na obraz w RAM
        float cpu_ms = 0;                // Zmienna na czas CPU

        // CPU Benchmark (Tylko do FHD, bo przy tylu sferach CPU umrze)
        // Im więcej sfer, tym CPU wolniejsze!
        if (w <= 1920) {                 // Ograniczenie testu CPU do 1920 pikseli szerokości
            h_img = (unsigned char*)malloc(size); // Alokacja pamięci RAM na obraz
            auto start = std::chrono::high_resolution_clock::now(); // Start pomiaru czasu CPU
            render_cpu(h_img, w, h, host_spheres.data(), num_spheres); // Renderowanie na CPU
            auto end = std::chrono::high_resolution_clock::now();   // Stop pomiaru czasu
            cpu_ms = std::chrono::duration<float, std::milli>(end - start).count(); // Obliczenie czasu w ms
            free(h_img);                 // Zwolnienie pamięci RAM
            std::cout << label << " CPU: " << cpu_ms << " ms\n"; // Wypisanie wyniku CPU
        } else {
            std::cout << label << " CPU: POMINIETO (za dlugo)\n"; // Komunikat o pominięciu
        }

        // GPU Benchmark
        dim3 bs(16, 16);                 // Konfiguracja bloku wątków (16x16)
        dim3 grid((w + 15)/16, (h + 15)/16); // Konfiguracja siatki bloków pokrywającej obraz

        cudaEvent_t start, stop;         // Zdarzenia CUDA do pomiaru czasu
        cudaEventCreate(&start); cudaEventCreate(&stop); // Utworzenie zdarzeń
        cudaEventRecord(start);          // Rejestracja startu

        render_kernel<<<grid, bs>>>(d_img, w, h, d_spheres, num_spheres); // Uruchomienie kernela renderującego

        cudaEventRecord(stop);           // Rejestracja stopu
        cudaEventSynchronize(stop);      // Czekanie na zakończenie GPU
        float gpu_ms = 0;                // Zmienna na czas GPU
        cudaEventElapsedTime(&gpu_ms, start, stop); // Obliczenie czasu GPU
        std::cout << label << " GPU: " << gpu_ms << " ms\n"; // Wypisanie wyniku GPU

        benchFile << label << "," << w << "," << h << "," << (long long)w*h << "," << cpu_ms << "," << gpu_ms << "\n"; // Zapis wyniku do CSV
        cudaFree(d_img);                 // Zwolnienie pamięci obrazu na GPU
    }
    benchFile.close();                   // Zamknięcie pliku benchmarku

    // ==========================================
    // BENCHMARK 2: WĄTKI (BLOKI)
    // ==========================================
    // ==========================================
        // BENCHMARK 2: ROZMIAR BLOKU (Liczba wątków)
        // ==========================================
    std::ofstream blockFile("benchmark_blocks.csv"); // Otwarcie pliku na wyniki testu bloków
    blockFile << "label,block_side,total_threads,gpu_time_ms\n"; // Zapis nagłówka

    // Testujemy na wysokiej rozdzielczości 4K, aby GPU miało co robić
    int bw = 7680, bh = 4320;            // Ustawienie rozdzielczości testowej 8K
    size_t b_size = (size_t)bw * bh * 3; // Obliczenie rozmiaru bufora
    unsigned char* d_test;               // Wskaźnik GPU
    cudaMalloc(&d_test, b_size);         // Alokacja

    // Rozszerzona lista rozmiarów bloków (kwadratowe: side x side)
    // 2x2, 4x4, 8x8, 16x16, 20x20, 24x24, 28x28, 32x32
    std::vector<int> block_sides = {2, 4, 6, 8, 12, 16, 20, 24, 28, 32}; // Definicja rozmiarów boków bloku

    std::cout << "\n--- Start Rozszerzonego Benchmarku Bloków (4K) ---\n"; // Info startowe
    std::cout << "Testowanie wydajnosci w zaleznosci od zageszczenia watkow...\n"; // Info dodatkowe

    for(int b : block_sides) {           // Pętla po rozmiarach bloków
        dim3 bs(b, b);                   // Ustawienie wymiarów bloku
        dim3 grid((bw + b - 1) / b, (bh + b - 1) / b); // Przeliczenie siatki dla danego bloku

        cudaEvent_t start, stop;         // Zdarzenia czasowe
        cudaEventCreate(&start); cudaEventCreate(&stop); // Utworzenie
        cudaEventRecord(start);          // Start zegara

        // Wykonujemy kilka iteracji, aby usrednic wynik
        for(int i = 0; i < 10; i++) {    // Pętla uśredniająca (10 powtórzeń)
            render_kernel<<<grid, bs>>>(d_test, bw, bh, d_spheres, num_spheres); // Uruchomienie kernela
        }

        cudaEventRecord(stop);           // Stop zegara
        cudaEventSynchronize(stop);      // Czekanie
        float ms = 0;                    // Zmienna czasu
        cudaEventElapsedTime(&ms, start, stop); // Obliczenie czasu
        ms /= 10.0f; // Sredni czas z 10 prob // Podział przez liczbę iteracji

        int total_threads = b * b;       // Całkowita liczba wątków w bloku
        std::string label = std::to_string(b) + "x" + std::to_string(b); // Etykieta tekstowa

        printf("Blok %5s | Watkow: %4d | Czas: %8.4f ms\n", label.c_str(), total_threads, ms); // Wypisanie wyniku
        blockFile << label << "," << b << "," << total_threads << "," << ms << "\n"; // Zapis do pliku
    }
    blockFile.close();                   // Zamknięcie pliku

    cudaFree(d_test);                    // Zwolnienie pamięci testowej
    cudaFree(d_spheres);                 // Zwolnienie pamięci sfer
    return 0;                            // Zakończenie programu
}