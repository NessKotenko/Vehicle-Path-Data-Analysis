# Path Data Processing & Analysis

This project provides a suite of tools for processing, analyzing, and visualizing vehicle path data and traffic light locations. The project focuses on data cleaning, smoothing, and finding the closest path points to specific locations using efficient algorithms.

## Features

  * **Data Loading & Cleaning**: Loads raw vehicle path and traffic light data, handling `NaN` and `Inf` values to ensure data integrity.
  * **Path Smoothing**: Implements a moving average filter to smooth noisy path data, and interpolation polynominal smoothing creating a more representative vehicle trajectory.
  * **Closest Neighbor Search**: Uses the KDTree,BallTree & Brute-Force algorithm for an efficient approximate nearest neighbor search to find the closest point on the smoothed path for each traffic light.
  * **Performance Analysis**: Measures and compares the performance of different search algorithms and analyzes execution time as the dataset size increases.
  * **Visualization**: Generates clear, illustrative plots to visualize the raw and smoothed paths, traffic lights, and the results of the nearest neighbor search.

## Getting Started

### Prerequisites

You need Python 3.7 or newer. The required packages are listed in the `requirements.txt` file.

### Installation

1.  Clone this repository:
    ```
    git clone https://github.com/NessKotenko/Vehicle-Path-Data-Analysis.git
    ```
2.  Navigate to the project directory:
    ```
    cd your-repo-name
    ```
3.  Install the required packages:
    ```
    (CAN BE DONE AUTMOATICALLY FROM path_processing.py SCRIPT)
    pip install -r requirements.txt
    ```

### Usage

Run the main script from the root directory to access the command-line menu:

```
python path_processing.py
```

A console menu will guide you through the various analysis options.

