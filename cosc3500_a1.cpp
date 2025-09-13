// brians_fast.cpp
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <thread>
#include <immintrin.h>
#include <cstring>

#ifdef ENABLE_GRAPHICS
#include <SFML/Graphics.hpp>
#endif

constexpr int WIDTH = 200;
constexpr int HEIGHT = 200;
constexpr int MAX_FRAMES = 100;
constexpr int CELL_SIZE = 6; // Pixels per cell

using u8 = uint8_t;

static inline void build_extended_row(const u8* row, int width, u8* ext_row) {
    // ext_row length = width + 2
    ext_row[0] = row[width - 1];
    std::memcpy(ext_row + 1, row, width);
    ext_row[width + 1] = row[0];
}

void step_avx2(const u8* on_grid, const u8* dying_grid, u8* next_on_grid, u8* next_dying_grid, int width, int height) {
    const int stride = width;

    const int VEC_BYTES = 32; // AVX2 processes 32 bytes per register
    const int vec_iters = width / VEC_BYTES;
    const int tail = width % VEC_BYTES;

    // temporaries for extended rows
    std::vector<u8> ext_prev(width + 2);
    std::vector<u8> ext_cur(width + 2);
    std::vector<u8> ext_next(width + 2);

    // constants
    __m256i one8      = _mm256_set1_epi8(1);
    __m256i two8      = _mm256_set1_epi8(2);
    __m256i zero8     = _mm256_setzero_si256();
    __m256i ff8       = _mm256_set1_epi8(static_cast<char>(0xFF));

    for (int y = 0; y < height; ++y) {
        int y_prev = (y - 1 + height) % height;
        int y_next = (y + 1) % height;

        const u8* row_prev = on_grid + y_prev * stride;
        const u8* row_cur  = on_grid + y * stride;
        const u8* row_next = on_grid + y_next * stride;

        const u8* dy_row_cur = dying_grid + y * stride;

        // Build extended rows (wrap-around) so left/center/right loads are simple
        build_extended_row(row_prev, width, ext_prev.data());
        build_extended_row(row_cur,  width, ext_cur.data());
        build_extended_row(row_next, width, ext_next.data());

        // Also extend dying row for center checks (we only need center)
        // But we can load dying center from row_cur directly (no ext needed for dying center)

        int x = 0;
        // Vectorised main loop (process 32 cells per iteration)
        for (int vi = 0; vi < vec_iters; ++vi, x += VEC_BYTES) {
            // For prev row: load left/center/right
            __m256i p_left   = _mm256_loadu_si256((__m256i*)(ext_prev.data() + x + 0));
            __m256i p_center = _mm256_loadu_si256((__m256i*)(ext_prev.data() + x + 1));
            __m256i p_right  = _mm256_loadu_si256((__m256i*)(ext_prev.data() + x + 2));

            // For cur row: left and right (we skip center for neighbor sum)
            __m256i c_left   = _mm256_loadu_si256((__m256i*)(ext_cur.data() + x + 0));
            __m256i c_right  = _mm256_loadu_si256((__m256i*)(ext_cur.data() + x + 2));
            // We also need cur center for state checks
            __m256i c_center = _mm256_loadu_si256((__m256i*)(ext_cur.data() + x + 1));

            // For next row: left/center/right
            __m256i n_left   = _mm256_loadu_si256((__m256i*)(ext_next.data() + x + 0));
            __m256i n_center = _mm256_loadu_si256((__m256i*)(ext_next.data() + x + 1));
            __m256i n_right  = _mm256_loadu_si256((__m256i*)(ext_next.data() + x + 2));

            // Sum bytes: start with zeros and add components
            __m256i sum = _mm256_add_epi8(p_left, p_center);
            sum = _mm256_add_epi8(sum, p_right);
            sum = _mm256_add_epi8(sum, c_left);
            sum = _mm256_add_epi8(sum, c_right);
            sum = _mm256_add_epi8(sum, n_left);
            sum = _mm256_add_epi8(sum, n_center);
            sum = _mm256_add_epi8(sum, n_right);
            // sum now contains neighbor counts (0..8) in each byte lane

            // Compare sum == 2
            __m256i eq2_mask = _mm256_cmpeq_epi8(sum, two8); // 0xFF where sum==2

            // Build masks for is_on and is_dying
            // c_center currently contains bytes 0/1 for cur center; convert to mask
            __m256i is_on_mask = _mm256_cmpeq_epi8(c_center, one8);     // 0xFF if on
            // load dying center directly from dying grid
            __m256i d_center = _mm256_loadu_si256((__m256i*)(dy_row_cur + x)); // row bytes are 0/1
            __m256i is_dying_mask = _mm256_cmpeq_epi8(d_center, one8); // 0xFF if dying

            // is_off_mask = ~(is_on_mask | is_dying_mask)
            __m256i on_or_dying = _mm256_or_si256(is_on_mask, is_dying_mask);
            __m256i is_off_mask = _mm256_andnot_si256(on_or_dying, ff8); // 0xFF where off

            // next_on = is_off_mask & eq2_mask -> then convert 0xFF -> 1 by AND with one8
            __m256i candidate_on = _mm256_and_si256(is_off_mask, eq2_mask);
            __m256i next_on_vec = _mm256_and_si256(candidate_on, one8); // becomes 1 or 0

            // next_dying = is_on_mask -> convert 0xFF -> 1
            __m256i next_dying_vec = _mm256_and_si256(is_on_mask, one8);

            // Store results (store 32 bytes)
            _mm256_storeu_si256((__m256i*)(next_on_grid + y * stride + x), next_on_vec);
            _mm256_storeu_si256((__m256i*)(next_dying_grid + y * stride + x), next_dying_vec);
        }

        // Tail scalar for remaining columns (if width not divisible by 32)
        for (; x < width; ++x) {
            // left/center/right indices in ext arrays: x, x+1, x+2
            int p_sum = ext_prev[x + 0] + ext_prev[x + 1] + ext_prev[x + 2];
            int c_sum = ext_cur[x + 0] + ext_cur[x + 2]; // skip center
            int n_sum = ext_next[x + 0] + ext_next[x + 1] + ext_next[x + 2];
            int total = p_sum + c_sum + n_sum;
            u8 cur_on = row_cur[x];
            u8 cur_dying = dy_row_cur[x];
            if (!cur_on && !cur_dying) {
                next_on_grid[y * stride + x] = (total == 2) ? 1 : 0;
            } else {
                next_on_grid[y * stride + x] = 0;
            }
            next_dying_grid[y * stride + x] = cur_on ? 1 : 0;
        }
    } // for each row
}



// A small initializer and timing harness (no graphics)
int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    const int width = WIDTH;
    const int height = HEIGHT;
    const int stride = width;

    std::vector<u8> on_grid(width * height);
    std::vector<u8> dying_grid(width * height);

    // init ~10% ON randomly
    for (int i = 0; i < width * height; ++i) {
        if (std::rand() % 10 == 0) on_grid[i] = 1;
        else on_grid[i] = 0;
        dying_grid[i] = 0;
    }

    std::vector<u8> next_on(width * height);
    std::vector<u8> next_dying(width * height);

    const int MAX_ITER = MAX_FRAMES;
    std::vector<double> frameTimes;
    frameTimes.reserve(MAX_ITER);

#ifdef ENABLE_GRAPHICS
    sf::RenderWindow window(sf::VideoMode(sf::Vector2u(WIDTH * CELL_SIZE, HEIGHT * CELL_SIZE)), "COSC3500 - s4800993 A1");
    sf::RectangleShape cellShape(sf::Vector2f(CELL_SIZE, CELL_SIZE));
    
    int iter = 0;
    while (window.isOpen() && iter < MAX_ITER) {
#else
    for (int iter = 0; iter < MAX_ITER; ++iter) {
#endif
        auto t0 = std::chrono::high_resolution_clock::now();

        step_avx2(on_grid.data(), dying_grid.data(), next_on.data(), next_dying.data(), width, height);

        // swap
        on_grid.swap(next_on);
        dying_grid.swap(next_dying);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        frameTimes.push_back(ms);

#ifdef ENABLE_GRAPHICS
        // Handle events
        while (auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();
        }
        
        // Draw grid using flat array indexing
        window.clear(sf::Color::Black);
        for(int y = 0; y < HEIGHT; y++) {
            for(int x = 0; x < WIDTH; x++) {
                int idx = y * stride + x;  // Flat array index
                if(on_grid[idx]) {
                    // ON state = Cyan
                    cellShape.setFillColor(sf::Color::Cyan);
                    cellShape.setPosition(sf::Vector2f(x * CELL_SIZE, y * CELL_SIZE));
                    window.draw(cellShape);
                } else if(dying_grid[idx]) {
                    // DYING state = Dark blue
                    cellShape.setFillColor(sf::Color(0, 128, 255));
                    cellShape.setPosition(sf::Vector2f(x * CELL_SIZE, y * CELL_SIZE));
                    window.draw(cellShape);
                }
                // OFF state = black (no drawing needed)
            }
        }

        window.display();
        
        // Add a small delay for visualization (not counted in benchmarking)
        sf::sleep(sf::milliseconds(50));
        
        iter++;
    }
#else
    }
#endif

    // write stats
    double sum = 0;
    for (double v : frameTimes) sum += v;
    double avg = sum / frameTimes.size();
    double mn = *std::min_element(frameTimes.begin(), frameTimes.end());
    double mx = *std::max_element(frameTimes.begin(), frameTimes.end());

    std::ofstream out("frame_timing_stats.txt");
    for (size_t i = 0; i < frameTimes.size(); ++i) out << "Frame " << (i+1) << ": " << frameTimes[i] << " ms\n";
    out.close();

    std::cout << "Saved frame_timing_stats.txt\n";
    std::cout << "Avg: " << avg << " ms  min: " << mn << " ms  max: " << mx << " ms\n";

    return 0;
}
