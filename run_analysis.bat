@echo off
REM Batch script to compile and analyze frame timing

g++ -mavx2 -O3 -march=native -o main .\cosc3500_a1.cpp
if %errorlevel% neq 0 exit /b 1

set PATH=%PATH%;.\SFML
.\main.exe
if %errorlevel% neq 0 exit /b 1

python .\analyze_frame_times.py
