# TDAR-IDW Sensor Synchronization

A robust Python implementation of the **Time-Distance and Acceleration-Rate Inverse Distance Weighting (TDAR-IDW)** algorithm. This tool synchronizes high-frequency IMU data (e.g., 100Hz) with low-frequency GPS timestamps (e.g., 1Hz) for precise trajectory alignment.

## Features

*   **Data Preprocessing**: Automatic timestamp alignment, rolling average smoothing, and coordinate transformations.
*   **Robust Outlier Removal**: Uses Median Absolute Deviation (MAD) to filter sensor noise before processing.
*   **Dynamic Scene Partitioning**: Automatically detects "gradual" vs. "rapid" movement phases based on acceleration change rates to adjust algorithm sensitivity.
*   **Adaptive Weighting**: Dynamically adjusts interpolation weights ($\lambda_1, \lambda_2$) based on the detected scene type to balance time-proximity and signal stability.

## Requirements

*   Python 3.x
*   `numpy`
*   `pandas`

## Usage

1.  **Install dependencies**:
    ```bash
    pip install numpy pandas
    ```

2.  **Run the script**:
    The script includes a synthetic data generator for demonstration purposes.
    ```bash
    python tdar_sync.py
    ```

## Algorithm Overview

The algorithm aligns data by calculating weights based on two primary factors:

1.  **Time Distance**: How close the IMU sample is to the target GPS timestamp.
2.  **Acceleration Rate**: How stable the sensor data is at that specific moment (penalizing high-instability points during interpolation).

It uses a weighted average of the $N$ nearest neighbors to estimate **Pitch**, **Roll**, and **Total Acceleration** aligned to the exact GPS time.

## Configuration

You can adjust the algorithm parameters in the `TDAR_IDW_Sync` config dictionary:

*   `alpha`, `beta`: Decay coefficients for time and acceleration weights.
*   `lambda_gradual`, `lambda_rapid`: Adaptive coupling factors for different dynamic scenes.
*   `window_size`: Window size (in seconds) for smoothing and rate-of-change calculation.
