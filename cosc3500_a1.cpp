#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <thread>

#ifdef ENABLE_GRAPHICS
#include <SFML/Graphics.hpp>
#endif

const int WIDTH = 200;
const int HEIGHT = 200;
const int CELL_SIZE = 6; // Pixels per cell

enum CellState { OFF, ON, DYING };

// Count ON neighbors with toroidal wrap-around
int countOnNeighbors(const std::vector<std::vector<CellState>>& grid, int x, int y) {
    int count = 0;
    for(int dx = -1; dx <= 1; dx++) {
        for(int dy = -1; dy <= 1; dy++) {
            if(dx == 0 && dy == 0) continue;
            int nx = (x + dx + WIDTH) % WIDTH;
            int ny = (y + dy + HEIGHT) % HEIGHT;
            if(grid[nx][ny] == ON) count++;
        }
    }
    return count;
}

int main() {
    std::srand(std::time(0));

    // Initialize grid with random ON/OFF cells
    std::vector<std::vector<CellState>> grid(WIDTH, std::vector<CellState>(HEIGHT, OFF));
    for(int x = 0; x < WIDTH; x++) {
        for(int y = 0; y < HEIGHT; y++) {
            grid[x][y] = (std::rand() % 10 == 0) ? ON : OFF; // ~10% ON
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

        // Compute next grid state
        std::vector<std::vector<CellState>> nextGrid = grid;
        for(int x = 0; x < WIDTH; x++) {
            for(int y = 0; y < HEIGHT; y++) {
                if(grid[x][y] == OFF) {
                    if(countOnNeighbors(grid, x, y) == 2) nextGrid[x][y] = ON;
                } else if(grid[x][y] == ON) {
                    nextGrid[x][y] = DYING;
                } else if(grid[x][y] == DYING) {
                    nextGrid[x][y] = OFF;
                }
            }
        }
        grid = nextGrid;

#ifdef ENABLE_GRAPHICS
        // Draw grid
        window.clear(sf::Color::Black);
        for(int x = 0; x < WIDTH; x++) {
            for(int y = 0; y < HEIGHT; y++) {
                if(grid[x][y] == ON) cellShape.setFillColor(sf::Color::Cyan);
                else if(grid[x][y] == DYING) cellShape.setFillColor(sf::Color(0, 128, 255)); // darker blue
                else continue; // OFF = black

                cellShape.setPosition(sf::Vector2f(x * CELL_SIZE, y * CELL_SIZE));
                window.draw(cellShape);
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
