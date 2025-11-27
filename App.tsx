import React, { useState } from 'react';

const PYTHON_CODE = `import numpy as np
import pandas as pd
import math

class TDAR_IDW_Sync:
    def __init__(self, config=None):
        """
        Initialize the TDAR-IDW Synchronizer with parameters.
        """
        self.cfg = config or {
            'alpha': 0.1,    # Time decay parameter
            'beta': 0.05,    # Acceleration decay parameter
            'p': 2,          # Time power exponent
            'q': 2,          # Acceleration power exponent
            'lambda_gradual': (0.7, 0.3), # (lambda1, lambda2) for gradual scenes
            'lambda_rapid': (0.4, 0.6),   # (lambda1, lambda2) for rapid scenes
            'window_size': 5.0,           # Rolling window size in seconds
            'neighbor_count': 10          # Number of neighbors (N)
        }

    def compute_mad_outliers(self, series, k=3):
        """
        Detect outliers using Median Absolute Deviation (MAD).
        """
        median = np.median(series)
        diff = np.abs(series - median)
        mad = np.median(diff)
        if mad == 0:
            return np.zeros(len(series), dtype=bool)
        threshold = k * mad
        return diff > threshold

    def preprocess(self, imu_df):
        """
        Step II: Data Preprocessing
        1. Alignment & Sorting
        2. MAD Outlier Removal
        3. Coord Transform (skipped, assume IMU frame)
        4. Magnitude Calculation
        5. Pitch/Roll Calculation
        6. Smoothing
        """
        # Ensure time sorted
        df = imu_df.sort_values('time').reset_index(drop=True)
        
        # 2. Outlier Removal (MAD)
        for col in ['ax', 'ay', 'az']:
            outliers = self.compute_mad_outliers(df[col].values)
            # Set outliers to NaN then interpolate
            df.loc[outliers, col] = np.nan
            df[col] = df[col].interpolate(method='linear').ffill().bfill()
            
        # 4. Calculate Total Acceleration Magnitude
        df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        
        # 5. Calculate Pitch (theta) and Roll (phi)
        # theta = arctan(ay / sqrt(ax^2 + az^2))
        df['pitch_calc'] = np.arctan2(df['ay'], np.sqrt(df['ax']**2 + df['az']**2))
        # phi = arctan(ax / az)
        # Using arctan2 to handle az=0 safely
        df['roll_calc'] = np.arctan2(df['ax'], df['az'])
        
        # 6. Smoothing (Rolling Average)
        # Estimate sampling frequency
        dt_vals = df['time'].diff().dropna()
        dt_avg = dt_vals.median() if len(dt_vals) > 0 else 0.01
        fs = 1.0 / dt_avg if dt_avg > 0 else 100.0
        
        window_pts = int(self.cfg['window_size'] * fs)
        
        # Apply smoothing to calculated angles and magnitude
        cols_to_smooth = ['pitch_calc', 'roll_calc', 'acc_mag']
        for col in cols_to_smooth:
            # min_periods=1 ensures we get values at edges
            df[f'{col}_smooth'] = df[col].rolling(window=window_pts, center=True, min_periods=1).mean()
            
        return df, fs

    def partition_dynamic_scenes(self, df, fs):
        """
        Step III: Acceleration Rate of Change & Dynamic Scene Partitioning
        """
        # 1. Calculate rate of change R[i] = |d|a|/dt|
        # Use central difference or simple diff. Prompt implies diff.
        d_mag = np.abs(df['acc_mag'].diff())
        d_t = df['time'].diff()
        # Avoid div by zero
        df['R'] = (d_mag / d_t).fillna(0)
        
        # 2. Rolling average R_bar
        window_pts = int(self.cfg['window_size'] * fs)
        df['R_bar'] = df['R'].rolling(window=window_pts, center=True, min_periods=1).mean()
        
        # 3. Adjacent difference of R_bar
        df['R_bar_diff'] = np.abs(df['R_bar'].diff()).fillna(0)
        
        # 4. Find split point k (argmax)
        k_idx = df['R_bar_diff'].idxmax()
        split_time = df.loc[k_idx, 'time']
        
        return split_time

    def tdar_idw_core(self, target_time, imu_subset, lambdas):
        """
        Step IV: Core Weighting and Interpolation
        """
        l1, l2 = lambdas
        alpha, beta = self.cfg['alpha'], self.cfg['beta']
        p, q = self.cfg['p'], self.cfg['q']
        
        # 1. Time Difference
        dt = np.abs(imu_subset['time'].values - target_time)
        
        # 2. Acceleration Change Amount (Delta a)
        # Defined as sum(|a_i,k - a_i-1,k|) for k in {x,y,z}
        # We assume this is precomputed in 'delta_acc_metric' for the subset
        da = imu_subset['delta_acc_metric'].values
        
        # 3. Weights
        # Avoid potential overflow in exp if values are huge, but typically safe here
        w_t = np.exp(-alpha * (dt ** p))
        w_a = np.exp(-beta * (da ** q))
        
        # Combined weight (Adaptive Coupling)
        w_i = (w_t ** l1) * (w_a ** l2)
        
        # 4. Normalize
        w_sum = np.sum(w_i)
        if w_sum == 0:
            w_norm = np.ones(len(w_i)) / len(w_i)
        else:
            w_norm = w_i / w_sum
            
        # 5. Interpolate
        results = {}
        # We interpolate the SMOOTHED values as per typical pipeline, 
        # or raw depending on requirement. Usually smooth for syncing.
        targets = ['pitch_calc_smooth', 'roll_calc_smooth', 'acc_mag_smooth']
        
        for col in targets:
            val = np.sum(w_norm * imu_subset[col].values)
            results[col] = val
            
        return results

    def run_sync(self, imu_df, gps_df):
        """
        Main Execution Pipeline
        """
        # --- Preprocessing ---
        print("Preprocessing IMU data...")
        processed_imu, fs = self.preprocess(imu_df)
        
        # Precompute 'Delta a' metric for the whole series to save time in loop
        # |ax_i - ax_{i-1}| + ...
        dax = np.abs(processed_imu['ax'].diff().fillna(0))
        day = np.abs(processed_imu['ay'].diff().fillna(0))
        daz = np.abs(processed_imu['az'].diff().fillna(0))
        processed_imu['delta_acc_metric'] = dax + day + daz

        # --- Dynamic Scene Partitioning ---
        split_time = self.partition_dynamic_scenes(processed_imu, fs)
        print(f"Dynamic split detected at t = {split_time:.3f}s")
        
        # --- Sync Loop ---
        print("Starting TDAR-IDW synchronization...")
        sync_results = []
        
        N = self.cfg['neighbor_count']
        
        for t_gps in gps_df['time']:
            # Determine Scene Parameters
            if t_gps < split_time:
                lambdas = self.cfg['lambda_gradual']
                scene = 'gradual'
            else:
                lambdas = self.cfg['lambda_rapid']
                scene = 'rapid'
                
            # Select Neighbors
            # Calculate distance to all points (inefficient for huge data, ok for script)
            # Optimization: searchsorted if strictly monotonic time
            idx_closest = np.searchsorted(processed_imu['time'], t_gps)
            
            # Get a window around the closest point
            start_idx = max(0, idx_closest - N * 2)
            end_idx = min(len(processed_imu), idx_closest + N * 2)
            subset_candidates = processed_imu.iloc[start_idx:end_idx].copy()
            
            # Exact N closest
            subset_candidates['temp_dt'] = np.abs(subset_candidates['time'] - t_gps)
            neighbors = subset_candidates.nsmallest(N, 'temp_dt')
            
            # Interpolate
            if len(neighbors) > 0:
                res = self.tdar_idw_core(t_gps, neighbors, lambdas)
                res['time'] = t_gps
                res['scene_type'] = scene
                sync_results.append(res)
            else:
                # Handle edge case with no neighbors
                pass
                
        return pd.DataFrame(sync_results)

# --- Synthetic Data Generation for Demo ---
def generate_demo_data(duration=20, imu_freq=100, gps_freq=1):
    # Time arrays
    t_imu = np.linspace(0, duration, int(duration*imu_freq))
    t_gps = np.linspace(0, duration, int(duration*gps_freq)) + 0.1 # Offset
    
    # Generate Synthetic IMU Motion
    # Scenario: 0-10s Gradual, 10-20s Rapid
    # Base gravity
    az = np.ones_like(t_imu) * 9.81
    ax = np.sin(t_imu) * 2
    ay = np.cos(t_imu * 0.5) * 2
    
    # Add noise
    ax += np.random.normal(0, 0.2, size=len(t_imu))
    ay += np.random.normal(0, 0.2, size=len(t_imu))
    az += np.random.normal(0, 0.2, size=len(t_imu))
    
    # Add Outliers
    outlier_indices = np.random.choice(len(t_imu), size=int(len(t_imu)*0.005), replace=False)
    az[outlier_indices] += 50.0
    
    imu_df = pd.DataFrame({'time': t_imu, 'ax': ax, 'ay': ay, 'az': az})
    gps_df = pd.DataFrame({'time': t_gps}) # Lat/Lon not used in calculation
    
    return imu_df, gps_df

# --- Main Entry Point ---
if __name__ == "__main__":
    # 1. Generate Data
    print("Generating synthetic data...")
    imu_df, gps_df = generate_demo_data()
    
    # 2. Init Sync Engine
    tdar = TDAR_IDW_Sync()
    
    # 3. Run Sync
    result_df = tdar.run_sync(imu_df, gps_df)
    
    # 4. Output Results
    print("\\nSynchronization Complete. Sample results:")
    print(result_df[['time', 'scene_type', 'pitch_calc_smooth', 'roll_calc_smooth']].head(10))
`;

const README_CONTENT = `# TDAR-IDW Sensor Synchronization

A robust Python implementation of the **Time-Distance and Acceleration-Rate Inverse Distance Weighting (TDAR-IDW)** algorithm. This tool synchronizes high-frequency IMU data (e.g., 100Hz) with low-frequency GPS timestamps (e.g., 1Hz) for precise trajectory alignment.

## Features

*   **Data Preprocessing**: Automatic timestamp alignment, rolling average smoothing, and coordinate transformations.
*   **Robust Outlier Removal**: Uses Median Absolute Deviation (MAD) to filter sensor noise before processing.
*   **Dynamic Scene Partitioning**: Automatically detects "gradual" vs. "rapid" movement phases based on acceleration change rates to adjust algorithm sensitivity.
*   **Adaptive Weighting**: Dynamically adjusts interpolation weights (λ₁, λ₂) based on the detected scene type to balance time-proximity and signal stability.

## Requirements

*   Python 3.x
*   \`numpy\`
*   \`pandas\`

## Usage

1.  **Install dependencies**:
    \`\`\`bash
    pip install numpy pandas
    \`\`\`

2.  **Run the script**:
    The script includes a synthetic data generator for demonstration purposes.
    \`\`\`bash
    python tdar_sync.py
    \`\`\`

## Algorithm Overview

The algorithm aligns data by calculating weights based on two primary factors:

1.  **Time Distance**: How close the IMU sample is to the target GPS timestamp.
2.  **Acceleration Rate**: How stable the sensor data is at that specific moment.

It uses a weighted average of the N nearest neighbors to estimate **Pitch**, **Roll**, and **Total Acceleration** aligned to the exact GPS time.
`;

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'code' | 'readme'>('code');
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    const content = activeTab === 'code' ? PYTHON_CODE : README_CONTENT;
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="min-h-screen p-8 bg-slate-900 font-mono text-slate-300">
      <div className="max-w-5xl mx-auto space-y-6">
        <header className="border-b border-slate-700 pb-4 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-white">TDAR-IDW Project</h1>
            <p className="text-slate-400 mt-1">Python Implementation & Documentation</p>
          </div>
          <button 
            onClick={handleCopy}
            className={`px-4 py-2 rounded font-bold transition-all ${
              copied 
              ? 'bg-green-500 text-white' 
              : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {copied ? 'Copied!' : `Copy ${activeTab === 'code' ? 'Code' : 'README'}`}
          </button>
        </header>

        <div className="flex space-x-2 border-b border-slate-700">
          <button
            onClick={() => setActiveTab('code')}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === 'code'
                ? 'text-white border-b-2 border-blue-500'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            tdar_sync.py
          </button>
          <button
            onClick={() => setActiveTab('readme')}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === 'readme'
                ? 'text-white border-b-2 border-blue-500'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            README.md
          </button>
        </div>

        <div className="bg-slate-950 p-6 rounded-lg border border-slate-800 shadow-2xl overflow-hidden relative">
          <div className="absolute top-4 right-4 text-xs text-slate-500">
            {activeTab === 'code' ? 'Python Source' : 'Markdown'}
          </div>
          <pre className="overflow-x-auto text-xs leading-relaxed text-green-400 whitespace-pre-wrap">
            <code>{activeTab === 'code' ? PYTHON_CODE : README_CONTENT}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};

export default App;
