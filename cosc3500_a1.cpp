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

const int WIDTH = 200;
const int HEIGHT = 200;
const int CELL_SIZE = 6; // Pixels per cell

// State encoding using 2 binary grids:
// OFF:   on=0, dying=0
// ON:    on=1, dying=0  
// DYING: on=0, dying=1

// Count ON neighbors with toroidal wrap-around (scalar version)
int countOnNeighbors(const std::vector<std::vector<uint8_t>>& on_grid, int x, int y) {
    int count = 0;
    for(int dx = -1; dx <= 1; dx++) {
        for(int dy = -1; dy <= 1; dy++) {
            if(dx == 0 && dy == 0) continue;
            int nx = (x + dx + WIDTH) % WIDTH;
            int ny = (y + dy + HEIGHT) % HEIGHT;
            if(on_grid[nx][ny]) count++;
        }
    }
    return count;
}




int main() {
    std::srand(std::time(0));

    // Initialize 2 binary grids for tri-state system
    std::vector<std::vector<uint8_t>> on_grid(WIDTH, std::vector<uint8_t>(HEIGHT, 0));
    std::vector<std::vector<uint8_t>> dying_grid(WIDTH, std::vector<uint8_t>(HEIGHT, 0));
    
    
    // Initialize with random ON/OFF cells (~10% ON)
    for(int x = 0; x < WIDTH; x++) {
        for(int y = 0; y < HEIGHT; y++) {
            if(std::rand() % 10 == 0) {
                on_grid[x][y] = 1;
                dying_grid[x][y] = 0;
            }
        }
    }

#ifdef ENABLE_GRAPHICS
    sf::RenderWindow window(sf::VideoMode(sf::Vector2u(WIDTH * CELL_SIZE, HEIGHT * CELL_SIZE)), "COSC3500 - s4800993 A1");
    sf::RectangleShape cellShape(sf::Vector2f(CELL_SIZE, CELL_SIZE));
#endif
    
    // Timing data collection
    std::vector<double> frameTimes;
    int frameCount = 0;
    const int MAX_FRAMES = 100;

#ifdef ENABLE_GRAPHICS
    while (window.isOpen()) {
#else
    while (frameCount < MAX_FRAMES) {
#endif
        auto frameStart = std::chrono::high_resolution_clock::now();
        
#ifdef ENABLE_GRAPHICS
        while (auto event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();
        }
#endif

        // MAXIMUM SPEED VERSION: Pure SIMD processing
        std::vector<std::vector<uint8_t>> next_on_grid(HEIGHT, std::vector<uint8_t>(WIDTH, 0));
        std::vector<std::vector<uint8_t>> next_dying_grid(HEIGHT, std::vector<uint8_t>(WIDTH, 0));
        
        
        for(int x = 0; x < WIDTH; x++) {
            for(int y = 0; y < HEIGHT; y++) {
                bool is_on = on_grid[x][y];
                bool is_dying = dying_grid[x][y];
                bool is_off = !is_on && !is_dying;
                
                if(is_off) {
                    // OFF -> ON if exactly 2 ON neighbors
                    if(countOnNeighbors(on_grid, x, y) == 2) {
                        next_on_grid[x][y] = 1;
                        next_dying_grid[x][y] = 0;
                    }
                } else if(is_on) {
                    // ON -> DYING always
                    next_on_grid[x][y] = 0;
                    next_dying_grid[x][y] = 1;
                } else if(is_dying) {
                    // DYING -> OFF always
                    next_on_grid[x][y] = 0;
                    next_dying_grid[x][y] = 0;
                }
            }
        }
        
        // Update grids
        on_grid = next_on_grid;
        dying_grid = next_dying_grid;

#ifdef ENABLE_GRAPHICS
        // Draw grid using 2-grid system
        window.clear(sf::Color::Black);
        for(int y = 0; y < HEIGHT; y++) {
            for(int x = 0; x < WIDTH; x++) {
                if(on_grid[y][x]) {
                    // ON state = Cyan
                    cellShape.setFillColor(sf::Color::Cyan);
                    cellShape.setPosition(sf::Vector2f(x * CELL_SIZE, y * CELL_SIZE));
                    window.draw(cellShape);
                } else if(dying_grid[y][x]) {
                    // DYING state = Dark blue
                    cellShape.setFillColor(sf::Color(0, 128, 255));
                    cellShape.setPosition(sf::Vector2f(x * CELL_SIZE, y * CELL_SIZE));
                    window.draw(cellShape);
                }
                // OFF state = black (no drawing needed)
            }
        }

        window.display();
#endif
        
        auto frameEnd = std::chrono::high_resolution_clock::now();
        auto frameTime = std::chrono::duration_cast<std::chrono::microseconds>(frameEnd - frameStart);
        double frameTimeMs = frameTime.count() / 1000.0;
        
        frameCount++;
        if (frameCount <= MAX_FRAMES) {
            frameTimes.push_back(frameTimeMs);
            
            // Generate statistics after 100 frames
            if (frameCount == MAX_FRAMES) {
                double sum = 0;
                for (double time : frameTimes) {
                    sum += time;
                }
                double avgTime = sum / frameTimes.size();
                double minTime = *std::min_element(frameTimes.begin(), frameTimes.end());
                double maxTime = *std::max_element(frameTimes.begin(), frameTimes.end());
                
                // Write to file
                std::ofstream outFile("frame_timing_stats.txt");
                for (size_t i = 0; i < frameTimes.size(); i++) {
                    outFile << "Frame " << (i + 1) << ": " << frameTimes[i] << " ms\n";
                }
                outFile.close();
                
                std::cout << "\nStatistics saved to frame_timing_stats.txt" << std::endl;
                std::cout << "Average: " << avgTime << " ms, Min: " << minTime << " ms, Max: " << maxTime << " ms" << std::endl;
            }
        }
        
#ifdef ENABLE_GRAPHICS
        sf::sleep(sf::milliseconds(50)); // control simulation speed
#else
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // control simulation speed
#endif
    }

    return 0;
}
