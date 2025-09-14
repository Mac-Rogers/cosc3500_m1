#!/usr/bin/env python3
"""
Frame Timing Analysis Tool
Analyzes frame timing data from frame_timing_stats.txt and provides comprehensive statistics
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import statistics
from typing import List, Tuple

def parse_frame_times(filename: str) -> List[float]:
    """
    Parse frame times from the timing stats file
    
    Args:
        filename: Path to the frame_timing_stats.txt file
    
    Returns:
        List of frame times in milliseconds
    """
    frame_times = []
    
    try:
        with open(filename, 'r') as file:
            content = file.read()
            
            # Use regex to find all frame times in format "Frame X: Y.YYY ms"
            pattern = r'Frame \d+: ([\d.]+) ms'
            matches = re.findall(pattern, content)
            
            frame_times = [float(time) for time in matches]
            
        return frame_times
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'")
        return []
    except Exception as e:
        print(f"Error parsing file: {e}")
        return []

def calculate_statistics(frame_times: List[float]) -> dict:
    """
    Calculate comprehensive statistics for frame times
    
    Args:
        frame_times: List of frame times in milliseconds
    
    Returns:
        Dictionary containing all statistical measures
    """
    if not frame_times:
        return {}
    
    frame_array = np.array(frame_times)
    
    stats = {
        'count': len(frame_times),
        'mean': np.mean(frame_array),
        'median': np.median(frame_array),
        'std_dev': np.std(frame_array, ddof=1),  # Sample standard deviation
        'variance': np.var(frame_array, ddof=1),
        'min': np.min(frame_array),
        'max': np.max(frame_array),
        'range': np.max(frame_array) - np.min(frame_array),
        'q1': np.percentile(frame_array, 25),
        'q3': np.percentile(frame_array, 75),
        'iqr': np.percentile(frame_array, 75) - np.percentile(frame_array, 25)
    }
    
    return stats

def print_statistics(stats: dict):
    """
    Print formatted statistics
    
    Args:
        stats: Dictionary of statistical measures
    """
    if not stats:
        print("No statistics to display")
        return
    
    print(f"Count:           {stats['count']} frames")
    print(f"Mean:            {stats['mean']:.4f} ms")
    print(f"Median:          {stats['median']:.4f} ms")
    print(f"Standard Dev:    {stats['std_dev']:.4f} ms")
    print(f"Variance:        {stats['variance']:.6f} ms²")

    print(f"Minimum:         {stats['min']:.4f} ms")
    print(f"Maximum:         {stats['max']:.4f} ms")
    print(f"Range:           {stats['range']:.4f} ms")
    
def create_distribution_plots(frame_times: List[float], stats: dict):
    """
    Create comprehensive distribution plots
    
    Args:
        frame_times: List of frame times in milliseconds
        stats: Dictionary of statistical measures
    """
    if not frame_times:
        print("No data to plot")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('48009939 cosc3500 Benchmarking stats', fontsize=16, fontweight='bold')
    
    # 1. Histogram with distribution curve
    ax1.hist(frame_times, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.3f}ms')
    ax1.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["median"]:.3f}ms')
    ax1.set_xlabel('Frame Time (ms)')
    ax1.set_ylabel('Density')
    ax1.set_title('Frame Time Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time series plot
    ax2.plot(range(1, len(frame_times) + 1), frame_times, 'b-', alpha=0.7, linewidth=1)
    ax2.axhline(stats['mean'], color='red', linestyle='--', alpha=0.8, label=f'Mean: {stats["mean"]:.3f}ms')
    ax2.fill_between(range(1, len(frame_times) + 1), 
                     stats['mean'] - stats['std_dev'], 
                     stats['mean'] + stats['std_dev'], 
                     alpha=0.2, color='red', label=f'±1 Std Dev')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Frame Time (ms)')
    ax2.set_title('Frame Time Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('frame_timing_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nDistribution plots saved as 'frame_timing_analysis.png'")
    
    # Show the plot
    # plt.show()

def save_detailed_report(frame_times: List[float], stats: dict, filename: str = 'detailed_frame_analysis.txt'):
    """
    Save a detailed analysis report to a text file
    
    Args:
        frame_times: List of frame times in milliseconds
        stats: Dictionary of statistical measures
        filename: Output filename
    """
    try:
        with open(filename, 'w') as f:
            f.write("DETAILED FRAME TIMING ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            # Basic statistics
            f.write("STATISTICAL SUMMARY\n")
            f.write("-" * 20 + "\n")
            for key, value in stats.items():
                f.write(f"{key.replace('_', ' ').title():15}: {value:.6f}\n")
            
            f.write(f"\nPERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average FPS     : {1000/stats['mean']:.2f}\n")
            f.write(f"Min FPS         : {1000/stats['max']:.2f}\n")
            f.write(f"Max FPS         : {1000/stats['min']:.2f}\n")
            f.write(f"Coeff of Var    : {(stats['std_dev']/stats['mean']*100):.2f}%\n")
            
            # Raw data
            f.write(f"\nRAW FRAME TIMES (ms)\n")
            f.write("-" * 20 + "\n")
            for i, time in enumerate(frame_times, 1):
                f.write(f"Frame {i:3d}: {time:.3f} ms\n")
        
        print(f"Detailed report saved as '{filename}'")
        
    except Exception as e:
        print(f"Error saving report: {e}")

def main():
    """
    Main function to run the frame timing analysis
    """
    
    # Parse frame times from file
    filename = "frame_timing_stats.txt"
    frame_times = parse_frame_times(filename)
    
    if not frame_times:
        print("No frame timing data found. Exiting.")
        return
    
    # Calculate statistics
    stats = calculate_statistics(frame_times)
    
    # Print statistics
    print_statistics(stats)
    
    # Create plots
    try:
        create_distribution_plots(frame_times, stats)
    except ImportError:
        print("\nNote: matplotlib not available for plotting")
    except Exception as e:
        print(f"\nError creating plots: {e}")
    
    # Save detailed report
    save_detailed_report(frame_times, stats)
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()
