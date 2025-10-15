import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Assume 10 potholes, distributed among low, medium, high severity
    # Example distribution: 4 low, 3 medium, 3 high
    severity_levels = ["low", "medium", "high"]
    pothole_counts = [4, 1, 3]  # Number of potholes in each severity

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(severity_levels, pothole_counts, color=["green", "yellow", "orange"], edgecolor='black')

    # Add value labels on top of each bar
    for bar, count in zip(bars, pothole_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count}",
                 ha='center', va='bottom', fontsize=12)

    plt.title('Number of Potholes by Severity Level')
    plt.xlabel('Severity Level')
    plt.ylabel('Number of Potholes')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show() 