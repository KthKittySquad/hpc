import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ["Standard", "Cython", "GPU", "Multi-Proc"]
runs = ["Run 1", "Run 2", "Run 3", "Run 4", "Run 5"]

data = {
    "Standard": [244.9840034, 229.09, 224.32, 223.85, 241.43],
    "Cython": [228.2047827, 223.3687992, 239.8631124, 223.9217951, 230.5736975],
    "GPU": [96.66132279, 96.58154736, 96.45082072, 95.44598346, 95.88996547],
    "Multi-Proc": [422.1742069, 406.7512059, 405.1418552, 410.9026767, 410.7935586],
}

# Define bar width and positions for each group
bar_width = 0.2
r = np.arange(len(runs))
r1 = r
r2 = r1 + bar_width
r3 = r2 + bar_width
r4 = r3 + bar_width

# Create the plot
plt.figure(figsize=(10, 6))
plt.bar(r1, data["Standard"], width=bar_width, label="Standard")
plt.bar(r2, data["Cython"], width=bar_width, label="Cython")
plt.bar(r3, data["GPU"], width=bar_width, label="GPU")
plt.bar(r4, data["Multi-Proc"], width=bar_width, label="Multi-Proc")

# Add labels and title
plt.xlabel("Runs")
plt.ylabel("Time in seconds")
plt.title("Method Performance across Runs")
plt.xticks(r + 1.5 * bar_width, runs)
plt.legend()

plt.tight_layout()
plt.show()


# Methods and their average values and standard deviations
methods = ["Standard", "Cython", "GPU", "Multi-Proc"]
averages = [232.7343912, 229.1864374, 96.20592796, 411.1527007]
stdev = [9.857587631, 6.679088601, 0.5210073853, 6.655233168]

# Positions for the bars on the x-axis
x_pos = np.arange(len(methods))
bar_width = 0.5

# Create the bar chart with error bars
plt.figure(figsize=(8, 6))
bars = plt.bar(
    x_pos, averages, width=bar_width, yerr=stdev, capsize=5, edgecolor="black"
)

# Add labels, title, and custom x-axis tick labels
plt.xlabel("Methods")
plt.ylabel("Average Value")
plt.title("Average Results with Standard Deviation")
plt.xticks(x_pos, methods)

plt.tight_layout()
plt.show()
