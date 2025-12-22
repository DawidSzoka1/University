#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <string>

// --- KONFIGURACJA ---
#define INF 2e10f

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3 operator+(const Vec3& b) const { return {x + b.x, y + b.y, z + b.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& b) const { return {x - b.x, y - b.y, z - b.z}; }
    __host__ __device__ Vec3 operator*(float b) const { return {x * b, y * b, z * b}; }
    __host__ __device__ float dot(const Vec3& b) const { return x * b.x + y * b.y + z * b.z; }
    __host__ __device__ Vec3 normalize() {
        float k = 1.0f / sqrtf(x * x + y * y + z * z);
        return {x * k, y * k, z * k};
    }
};

struct Sphere {
    Vec3 center;
    float radius;
    Vec3 color;
};

// --- BARDZO GĘSTE TWORZENIE INICJAŁÓW "DS" ---
void createDSInitials(std::vector<Sphere>& spheres) {
    // Zmniejszamy promień i odstępy, żeby mieć WIĘCEJ PUNKTÓW
    // To sprawi, że S będzie gładkie jak linia
    float r = 2.5f;       // Małe kuleczki
    float step = 0.8f;    // Kuleczki nachodzą na siebie (gęściej niż ich średnica)

    // --- LITERA D (Po lewej, Czerwona) ---
    float dx = -100.0f;

    // 1. Pionowa kreska (Gęsta)
    for (float y = -100.0f; y <= 100.0f; y += step) {
        spheres.push_back({{dx, y, 0}, r, {1.0f, 0.2f, 0.2f}});
    }

    // 2. Brzuszek D (Gęsty)
    // Kąt od -90 stopni do +90 stopni
    for (float angle = -1.57f; angle <= 1.57f; angle += 0.005f) { // Bardzo mały krok kąta!
        float x = dx + 80.0f * cosf(angle);
        float y = 100.0f * sinf(angle);

        // Rysujemy tylko prawą połówkę elipsy, żeby zamknąć D
        if (x >= dx) {
            spheres.push_back({{x, y, 0}, r, {1.0f, 0.2f, 0.2f}});
        }
    }

    // --- LITERA S (Po prawej, Niebieska - NAPRAWIONA CIĄGŁOŚĆ) ---
    float sx = 100.0f;
    float radius_s = 45.0f;
    float y_top = radius_s;   // Środek górnego: 45.0
    float y_bot = -radius_s;

    // Matematyka S: Dwa łuki.
    // Górny: od prawej strony (0 rad), przez górę, do lewej strony (PI rad)
    // Dolny: od prawej strony (0 rad), przez dół, do lewej strony (PI rad)
    // Ale musimy je obrócić i połączyć.

    // 1. Górny łuk S (od prawej góry do środka)
    // Kąt od ok. 30 stopni do 270 stopni (w układzie zegarowym)
    // Używamy parametryzacji, żeby stykały się idealnie w punkcie (sx, 0)

    // Górna część (niebieska)
    for (float angle = 0.5f; angle <= 4.713f; angle += 0.005f) {
        float x = sx + radius_s * cosf(angle);
        float y = y_top + radius_s * sinf(angle);
        spheres.push_back({{x, y, 0}, r, {0.2f, 0.5f, 1.0f}});
    }

    // Dolna część (niebieska)
    // Musi zaczynać się tam gdzie górna skończyła (mniej więcej w środku S)
    for (float angle = 1.5708f; angle >= -2.6f; angle -= 0.005f) {
        float x = sx + radius_s * cosf(angle); // Ten sam X
        float y = y_bot + radius_s * sinf(angle); // Y przesunięty w dół
        spheres.push_back({{x, y, 0}, r, {0.2f, 0.5f, 1.0f}});
    }
}

// Funkcja przecięcia
__host__ __device__ float hit(const Sphere& s, const Vec3& rayOrigin, const Vec3& rayDir) {
    Vec3 oc = rayOrigin - s.center;
    float a = rayDir.dot(rayDir);
    float b = 2.0f * oc.dot(rayDir);
    float c = oc.dot(oc) - s.radius * s.radius;
    float disc = b * b - 4 * a * c;
    if (disc < 0) return -1.0f;
    return (-b - sqrtf(disc)) / (2.0f * a);
}

// Ray Tracing
__host__ __device__ void computeColor(int x, int y, int width, int height, Sphere* spheres, int num_spheres, unsigned char* buffer) {
    float u = (float)x / width;
    float v = (float)y / height;
    float aspect = (float)width / height;

    Vec3 rayOrigin = {0, 0, 600};
    float screenX = (u * 2.0f - 1.0f) * aspect * 300.0f;
    float screenY = (1.0f - v * 2.0f) * 300.0f;
    Vec3 rayDir = (Vec3{screenX, screenY, 0} - rayOrigin).normalize();

    float closest_t = INF;
    int idx = -1;

    for (int i = 0; i < num_spheres; i++) {
        float t = hit(spheres[i], rayOrigin, rayDir);
        if (t > 0.0f && t < closest_t) {
            closest_t = t;
            idx = i;
        }
    }

    int r = 0, g = 0, b = 0;
    if (idx != -1) {
        Sphere s = spheres[idx];
        Vec3 hitPoint = rayOrigin + rayDir * closest_t;
        Vec3 normal = (hitPoint - s.center).normalize();
        Vec3 lightDir = Vec3{-0.5f, 0.5f, 1.0f}.normalize(); // Światło z przodu/góry

        // Prosty shading diffuse
        float diff = fmaxf(normal.dot(lightDir), 0.1f);
        Vec3 result = s.color * diff;

        // Dodajmy trochę "połysku" żeby kulki były ładniejsze
        r = (int)(fminf(result.x * 255.0f, 255.0f));
        g = (int)(fminf(result.y * 255.0f, 255.0f));
        b = (int)(fminf(result.z * 255.0f, 255.0f));
    }

    int pIdx = (y * width + x) * 3;
    buffer[pIdx] = r;
    buffer[pIdx+1] = g;
    buffer[pIdx+2] = b;
}

__global__ void render_kernel(unsigned char* buffer, int w, int h, Sphere* s, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) computeColor(x, y, w, h, s, n, buffer);
}

void render_cpu(unsigned char* buffer, int w, int h, Sphere* s, int n) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            computeColor(x, y, w, h, s, n, buffer);
        }
    }
}

int main() {
    // 1. GENEROWANIE GEOMETRII (TERAZ GĘSTEJ)
    std::vector<Sphere> host_spheres;
    createDSInitials(host_spheres);
    int num_spheres = host_spheres.size();

    // Informacja dla Ciebie ile punktów powstało
    std::cout << "[INFO] Wygenerowano model skladajacy sie z " << num_spheres << " sfer (punktow).\n";

    // 2. ZAPIS GEOMETRII DO PLIKU
    std::ofstream sceneFile("scene_data.csv");
    sceneFile << "x,y,z,r,g,b,radius\n";
    for(const auto& s : host_spheres) {
        sceneFile << s.center.x << "," << s.center.y << "," << s.center.z << ","
                  << s.color.x << "," << s.color.y << "," << s.color.z << "," << s.radius << "\n";
    }
    sceneFile.close();

    // 3. PRZYGOTOWANIE GPU
    Sphere* d_spheres;
    cudaMalloc(&d_spheres, num_spheres * sizeof(Sphere));
    cudaMemcpy(d_spheres, host_spheres.data(), num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice);

    // ==========================================
    // BENCHMARK 1: ROZDZIELCZOŚCI
    // ==========================================
    std::ofstream benchFile("benchmark_resolution.csv");
    benchFile << "label,width,height,pixels,cpu_time_ms,gpu_time_ms\n";

    std::vector<std::pair<std::string, std::pair<int, int>>> resolutions = {
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

    std::cout << "\n--- Start Benchmarku Rozdzielczosci ---\n";

    for (auto item : resolutions) {
        std::string label = item.first;
        int w = item.second.first;
        int h = item.second.second;
        size_t size = (size_t)w * h * 3;

        unsigned char* d_img;
        cudaMalloc(&d_img, size);

        unsigned char* h_img = nullptr;
        float cpu_ms = 0;

        // CPU Benchmark (Tylko do FHD, bo przy tylu sferach CPU umrze)
        // Im więcej sfer, tym CPU wolniejsze!
        if (w <= 1920) {
            h_img = (unsigned char*)malloc(size);
            auto start = std::chrono::high_resolution_clock::now();
            render_cpu(h_img, w, h, host_spheres.data(), num_spheres);
            auto end = std::chrono::high_resolution_clock::now();
            cpu_ms = std::chrono::duration<float, std::milli>(end - start).count();
            free(h_img);
            std::cout << label << " CPU: " << cpu_ms << " ms\n";
        } else {
            std::cout << label << " CPU: POMINIETO (za dlugo)\n";
        }

        // GPU Benchmark
        dim3 bs(16, 16);
        dim3 grid((w + 15)/16, (h + 15)/16);

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);

        render_kernel<<<grid, bs>>>(d_img, w, h, d_spheres, num_spheres);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpu_ms = 0;
        cudaEventElapsedTime(&gpu_ms, start, stop);
        std::cout << label << " GPU: " << gpu_ms << " ms\n";

        benchFile << label << "," << w << "," << h << "," << (long long)w*h << "," << cpu_ms << "," << gpu_ms << "\n";
        cudaFree(d_img);
    }
    benchFile.close();

    // ==========================================
    // BENCHMARK 2: WĄTKI (BLOKI)
    // ==========================================
    std::ofstream blockFile("benchmark_blocks.csv");
    blockFile << "label,block_side,total_threads,gpu_time_ms\n";

    int bw = 3840, bh = 2160; // Testujemy na 4K
    size_t b_size = (size_t)bw * bh * 3;
    unsigned char* d_test;
    cudaMalloc(&d_test, b_size);

    std::vector<int> block_sides = {4, 8, 16, 32};

    std::cout << "\n--- Start Benchmarku Watkow (4K) ---\n";

    for(int b : block_sides) {
        dim3 bs(b, b);
        dim3 grid((bw + b - 1)/b, (bh + b - 1)/b);

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);

        render_kernel<<<grid, bs>>>(d_test, bw, bh, d_spheres, num_spheres);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        int total_threads = b * b;
        std::string label = std::to_string(b) + "x" + std::to_string(b);
        std::cout << "Blok " << label << " (" << total_threads << " watkow): " << ms << " ms\n";
        blockFile << label << "," << b << "," << total_threads << "," << ms << "\n";
    }
    blockFile.close();

    cudaFree(d_test);
    cudaFree(d_spheres);
    return 0;
}