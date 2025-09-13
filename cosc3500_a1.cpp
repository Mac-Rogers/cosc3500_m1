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
constexpr int WARMUP_FRAMES = 30;   // Frames to discard for warmup
constexpr int MAX_FRAMES = 100;     // Frames to record after warmup
constexpr int CELL_SIZE = 6; // Pixels per cell

using u8 = uint8_t;

static inline void build_extended_row(const u8* row, int width, u8* ext_row) {
    // ext_row length = width + 2
    ext_row[0] = row[width - 1];
    std::memcpy(ext_row + 1, row, width);
    ext_row[width + 1] = row[0];
}

// step_avx2 function removed - now inlined in main loop for performance



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

    const int TOTAL_ITER = WARMUP_FRAMES + MAX_FRAMES;
    std::vector<double> frameTimes;
    frameTimes.reserve(MAX_FRAMES);
    

#ifdef ENABLE_GRAPHICS
    sf::RenderWindow window(sf::VideoMode(sf::Vector2u(WIDTH * CELL_SIZE, HEIGHT * CELL_SIZE)), "COSC3500 - s4800993 A1");
    sf::RectangleShape cellShape(sf::Vector2f(CELL_SIZE, CELL_SIZE));
    
    int iter = 0;
    while (window.isOpen() && iter < TOTAL_ITER) {
#else
    // Pre-allocate all temporary vectors outside the loop
    std::vector<u8> ext_prev(width + 2);
    std::vector<u8> ext_cur(width + 2);  
    std::vector<u8> ext_next(width + 2);
    
    // Pre-compute constants
    const int VEC_BYTES = 32;
    const int vec_iters = width / VEC_BYTES;
    const int tail = width % VEC_BYTES;
    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i two8 = _mm256_set1_epi8(2);
    const __m256i ff8 = _mm256_set1_epi8(static_cast<char>(0xFF));
    
    for (int iter = 0; iter < TOTAL_ITER; ++iter) {
#endif
        auto t0 = std::chrono::high_resolution_clock::now();

        // INLINED step_avx2 with optimizations
        const u8* on_ptr = on_grid.data();
        const u8* dying_ptr = dying_grid.data();
        u8* next_on_ptr = next_on.data();
        u8* next_dying_ptr = next_dying.data();
        
        // Unroll outer loop by 4 rows for better cache utilization
        int y = 0;
        for (; y < height - 3; y += 4) {
            // Process 4 rows at once
            for (int row_offset = 0; row_offset < 4; ++row_offset) {
                const int cur_y = y + row_offset;
                const int y_prev = (cur_y - 1 + height) % height;
                const int y_next = (cur_y + 1) % height;
                
                const u8* row_prev = on_ptr + y_prev * width;
                const u8* row_cur = on_ptr + cur_y * width;
                const u8* row_next = on_ptr + y_next * width;
                const u8* dy_row_cur = dying_ptr + cur_y * width;
                
                // Build extended rows inline
                ext_prev[0] = row_prev[width - 1];
                std::memcpy(ext_prev.data() + 1, row_prev, width);
                ext_prev[width + 1] = row_prev[0];
                
                ext_cur[0] = row_cur[width - 1];
                std::memcpy(ext_cur.data() + 1, row_cur, width);
                ext_cur[width + 1] = row_cur[0];
                
                ext_next[0] = row_next[width - 1];
                std::memcpy(ext_next.data() + 1, row_next, width);
                ext_next[width + 1] = row_next[0];
                
                // Unrolled vectorized loop (process 2 vectors at once)
                int x = 0;
                for (int vi = 0; vi < vec_iters - 1; vi += 2, x += 2 * VEC_BYTES) {
                    // First vector
                    __m256i p_left1 = _mm256_loadu_si256((__m256i*)(ext_prev.data() + x));
                    __m256i p_center1 = _mm256_loadu_si256((__m256i*)(ext_prev.data() + x + 1));
                    __m256i p_right1 = _mm256_loadu_si256((__m256i*)(ext_prev.data() + x + 2));
                    
                    __m256i c_left1 = _mm256_loadu_si256((__m256i*)(ext_cur.data() + x));
                    __m256i c_right1 = _mm256_loadu_si256((__m256i*)(ext_cur.data() + x + 2));
                    __m256i c_center1 = _mm256_loadu_si256((__m256i*)(ext_cur.data() + x + 1));
                    
                    __m256i n_left1 = _mm256_loadu_si256((__m256i*)(ext_next.data() + x));
                    __m256i n_center1 = _mm256_loadu_si256((__m256i*)(ext_next.data() + x + 1));
                    __m256i n_right1 = _mm256_loadu_si256((__m256i*)(ext_next.data() + x + 2));
                    
                    // Second vector
                    __m256i p_left2 = _mm256_loadu_si256((__m256i*)(ext_prev.data() + x + VEC_BYTES));
                    __m256i p_center2 = _mm256_loadu_si256((__m256i*)(ext_prev.data() + x + VEC_BYTES + 1));
                    __m256i p_right2 = _mm256_loadu_si256((__m256i*)(ext_prev.data() + x + VEC_BYTES + 2));
                    
                    __m256i c_left2 = _mm256_loadu_si256((__m256i*)(ext_cur.data() + x + VEC_BYTES));
                    __m256i c_right2 = _mm256_loadu_si256((__m256i*)(ext_cur.data() + x + VEC_BYTES + 2));
                    __m256i c_center2 = _mm256_loadu_si256((__m256i*)(ext_cur.data() + x + VEC_BYTES + 1));
                    
                    __m256i n_left2 = _mm256_loadu_si256((__m256i*)(ext_next.data() + x + VEC_BYTES));
                    __m256i n_center2 = _mm256_loadu_si256((__m256i*)(ext_next.data() + x + VEC_BYTES + 1));
                    __m256i n_right2 = _mm256_loadu_si256((__m256i*)(ext_next.data() + x + VEC_BYTES + 2));
                    
                    // Process both vectors in parallel
                    __m256i sum1 = _mm256_add_epi8(_mm256_add_epi8(p_left1, p_center1), p_right1);
                    sum1 = _mm256_add_epi8(sum1, _mm256_add_epi8(c_left1, c_right1));
                    sum1 = _mm256_add_epi8(sum1, _mm256_add_epi8(n_left1, n_center1));
                    sum1 = _mm256_add_epi8(sum1, n_right1);
                    
                    __m256i sum2 = _mm256_add_epi8(_mm256_add_epi8(p_left2, p_center2), p_right2);
                    sum2 = _mm256_add_epi8(sum2, _mm256_add_epi8(c_left2, c_right2));
                    sum2 = _mm256_add_epi8(sum2, _mm256_add_epi8(n_left2, n_center2));
                    sum2 = _mm256_add_epi8(sum2, n_right2);
                    
                    // Process states
                    __m256i eq2_mask1 = _mm256_cmpeq_epi8(sum1, two8);
                    __m256i eq2_mask2 = _mm256_cmpeq_epi8(sum2, two8);
                    
                    __m256i is_on_mask1 = _mm256_cmpeq_epi8(c_center1, one8);
                    __m256i is_on_mask2 = _mm256_cmpeq_epi8(c_center2, one8);
                    
                    __m256i d_center1 = _mm256_loadu_si256((__m256i*)(dy_row_cur + x));
                    __m256i d_center2 = _mm256_loadu_si256((__m256i*)(dy_row_cur + x + VEC_BYTES));
                    __m256i is_dying_mask1 = _mm256_cmpeq_epi8(d_center1, one8);
                    __m256i is_dying_mask2 = _mm256_cmpeq_epi8(d_center2, one8);
                    
                    __m256i on_or_dying1 = _mm256_or_si256(is_on_mask1, is_dying_mask1);
                    __m256i on_or_dying2 = _mm256_or_si256(is_on_mask2, is_dying_mask2);
                    __m256i is_off_mask1 = _mm256_andnot_si256(on_or_dying1, ff8);
                    __m256i is_off_mask2 = _mm256_andnot_si256(on_or_dying2, ff8);
                    
                    __m256i next_on_vec1 = _mm256_and_si256(_mm256_and_si256(is_off_mask1, eq2_mask1), one8);
                    __m256i next_on_vec2 = _mm256_and_si256(_mm256_and_si256(is_off_mask2, eq2_mask2), one8);
                    __m256i next_dying_vec1 = _mm256_and_si256(is_on_mask1, one8);
                    __m256i next_dying_vec2 = _mm256_and_si256(is_on_mask2, one8);
                    
                    _mm256_storeu_si256((__m256i*)(next_on_ptr + cur_y * width + x), next_on_vec1);
                    _mm256_storeu_si256((__m256i*)(next_on_ptr + cur_y * width + x + VEC_BYTES), next_on_vec2);
                    _mm256_storeu_si256((__m256i*)(next_dying_ptr + cur_y * width + x), next_dying_vec1);
                    _mm256_storeu_si256((__m256i*)(next_dying_ptr + cur_y * width + x + VEC_BYTES), next_dying_vec2);
                }
                
                // Handle remaining single vectors and tail
                for (int vi = x / VEC_BYTES; vi < vec_iters; ++vi, x += VEC_BYTES) {
                    // Single vector processing (original code)
                    __m256i p_left = _mm256_loadu_si256((__m256i*)(ext_prev.data() + x));
                    __m256i p_center = _mm256_loadu_si256((__m256i*)(ext_prev.data() + x + 1));
                    __m256i p_right = _mm256_loadu_si256((__m256i*)(ext_prev.data() + x + 2));
                    __m256i c_left = _mm256_loadu_si256((__m256i*)(ext_cur.data() + x));
                    __m256i c_right = _mm256_loadu_si256((__m256i*)(ext_cur.data() + x + 2));
                    __m256i c_center = _mm256_loadu_si256((__m256i*)(ext_cur.data() + x + 1));
                    __m256i n_left = _mm256_loadu_si256((__m256i*)(ext_next.data() + x));
                    __m256i n_center = _mm256_loadu_si256((__m256i*)(ext_next.data() + x + 1));
                    __m256i n_right = _mm256_loadu_si256((__m256i*)(ext_next.data() + x + 2));
                    
                    __m256i sum = _mm256_add_epi8(p_left, p_center);
                    sum = _mm256_add_epi8(sum, p_right);
                    sum = _mm256_add_epi8(sum, c_left);
                    sum = _mm256_add_epi8(sum, c_right);
                    sum = _mm256_add_epi8(sum, n_left);
                    sum = _mm256_add_epi8(sum, n_center);
                    sum = _mm256_add_epi8(sum, n_right);
                    
                    __m256i eq2_mask = _mm256_cmpeq_epi8(sum, two8);
                    __m256i is_on_mask = _mm256_cmpeq_epi8(c_center, one8);
                    __m256i d_center = _mm256_loadu_si256((__m256i*)(dy_row_cur + x));
                    __m256i is_dying_mask = _mm256_cmpeq_epi8(d_center, one8);
                    __m256i on_or_dying = _mm256_or_si256(is_on_mask, is_dying_mask);
                    __m256i is_off_mask = _mm256_andnot_si256(on_or_dying, ff8);
                    __m256i next_on_vec = _mm256_and_si256(_mm256_and_si256(is_off_mask, eq2_mask), one8);
                    __m256i next_dying_vec = _mm256_and_si256(is_on_mask, one8);
                    
                    _mm256_storeu_si256((__m256i*)(next_on_ptr + cur_y * width + x), next_on_vec);
                    _mm256_storeu_si256((__m256i*)(next_dying_ptr + cur_y * width + x), next_dying_vec);
                }
                
                // Scalar tail processing
                for (; x < width; ++x) {
                    int p_sum = ext_prev[x] + ext_prev[x + 1] + ext_prev[x + 2];
                    int c_sum = ext_cur[x] + ext_cur[x + 2];
                    int n_sum = ext_next[x] + ext_next[x + 1] + ext_next[x + 2];
                    int total = p_sum + c_sum + n_sum;
                    u8 cur_on = row_cur[x];
                    u8 cur_dying = dy_row_cur[x];
                    next_on_ptr[cur_y * width + x] = (!cur_on && !cur_dying && total == 2) ? 1 : 0;
                    next_dying_ptr[cur_y * width + x] = cur_on;
                }
            }
        }
        
        // Handle remaining rows
        for (; y < height; ++y) {
            const int y_prev = (y - 1 + height) % height;
            const int y_next = (y + 1) % height;
            const u8* row_prev = on_ptr + y_prev * width;
            const u8* row_cur = on_ptr + y * width;
            const u8* row_next = on_ptr + y_next * width;
            const u8* dy_row_cur = dying_ptr + y * width;
            
            ext_prev[0] = row_prev[width - 1];
            std::memcpy(ext_prev.data() + 1, row_prev, width);
            ext_prev[width + 1] = row_prev[0];
            ext_cur[0] = row_cur[width - 1];
            std::memcpy(ext_cur.data() + 1, row_cur, width);
            ext_cur[width + 1] = row_cur[0];
            ext_next[0] = row_next[width - 1];
            std::memcpy(ext_next.data() + 1, row_next, width);
            ext_next[width + 1] = row_next[0];
            
            int x = 0;
            for (int vi = 0; vi < vec_iters; ++vi, x += VEC_BYTES) {
                __m256i p_left = _mm256_loadu_si256((__m256i*)(ext_prev.data() + x));
                __m256i p_center = _mm256_loadu_si256((__m256i*)(ext_prev.data() + x + 1));
                __m256i p_right = _mm256_loadu_si256((__m256i*)(ext_prev.data() + x + 2));
                __m256i c_left = _mm256_loadu_si256((__m256i*)(ext_cur.data() + x));
                __m256i c_right = _mm256_loadu_si256((__m256i*)(ext_cur.data() + x + 2));
                __m256i c_center = _mm256_loadu_si256((__m256i*)(ext_cur.data() + x + 1));
                __m256i n_left = _mm256_loadu_si256((__m256i*)(ext_next.data() + x));
                __m256i n_center = _mm256_loadu_si256((__m256i*)(ext_next.data() + x + 1));
                __m256i n_right = _mm256_loadu_si256((__m256i*)(ext_next.data() + x + 2));
                
                __m256i sum = _mm256_add_epi8(p_left, p_center);
                sum = _mm256_add_epi8(sum, p_right);
                sum = _mm256_add_epi8(sum, c_left);
                sum = _mm256_add_epi8(sum, c_right);
                sum = _mm256_add_epi8(sum, n_left);
                sum = _mm256_add_epi8(sum, n_center);
                sum = _mm256_add_epi8(sum, n_right);
                
                __m256i eq2_mask = _mm256_cmpeq_epi8(sum, two8);
                __m256i is_on_mask = _mm256_cmpeq_epi8(c_center, one8);
                __m256i d_center = _mm256_loadu_si256((__m256i*)(dy_row_cur + x));
                __m256i is_dying_mask = _mm256_cmpeq_epi8(d_center, one8);
                __m256i on_or_dying = _mm256_or_si256(is_on_mask, is_dying_mask);
                __m256i is_off_mask = _mm256_andnot_si256(on_or_dying, ff8);
                __m256i next_on_vec = _mm256_and_si256(_mm256_and_si256(is_off_mask, eq2_mask), one8);
                __m256i next_dying_vec = _mm256_and_si256(is_on_mask, one8);
                
                _mm256_storeu_si256((__m256i*)(next_on_ptr + y * width + x), next_on_vec);
                _mm256_storeu_si256((__m256i*)(next_dying_ptr + y * width + x), next_dying_vec);
            }
            
            for (; x < width; ++x) {
                int p_sum = ext_prev[x] + ext_prev[x + 1] + ext_prev[x + 2];
                int c_sum = ext_cur[x] + ext_cur[x + 2];
                int n_sum = ext_next[x] + ext_next[x + 1] + ext_next[x + 2];
                int total = p_sum + c_sum + n_sum;
                u8 cur_on = row_cur[x];
                u8 cur_dying = dy_row_cur[x];
                next_on_ptr[y * width + x] = (!cur_on && !cur_dying && total == 2) ? 1 : 0;
                next_dying_ptr[y * width + x] = cur_on;
            }
        }

        // swap
        on_grid.swap(next_on);
        dying_grid.swap(next_dying);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        // Only record timing after warmup period
        if (iter >= WARMUP_FRAMES) {
            frameTimes.push_back(ms);
        }

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
            }
        }

        window.display();
        
        // small delay for visualization
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
