import { ProcessedIMUData, RawIMUData, TDARParams, SyncResult, GPSData } from '../types';

// --- Helper Math Functions ---

const getMedian = (values: number[]): number => {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

// --- II. Preprocessing & III. Data Preparation ---

/**
 * 3. MAD Outlier Detection
 * Replaces outliers with linear interpolation of neighbors
 * Rule: |x - median| > k * MAD
 */
export const removeOutliersMAD = (data: RawIMUData[], k = 3): RawIMUData[] => {
  const axes = ['ax', 'ay', 'az'] as const;
  // Deep copy to avoid mutating source
  const processed = data.map(d => ({...d}));

  axes.forEach((axis) => {
    const values = processed.map(d => d[axis]);
    const median = getMedian(values);
    
    // Calculate MAD (Median Absolute Deviation)
    const deviations = values.map(v => Math.abs(v - median));
    const mad = getMedian(deviations);
    
    // Threshold
    const threshold = k * mad;

    // Filter and basic interpolation repair
    for (let i = 0; i < processed.length; i++) {
      if (Math.abs(processed[i][axis] - median) > threshold) {
        // Simple repair: avg of prev and next (if available)
        const prev = i > 0 ? processed[i - 1][axis] : median;
        const next = i < processed.length - 1 ? processed[i + 1][axis] : median;
        processed[i][axis] = (prev + next) / 2;
      }
    }
  });

  return processed;
};

/**
 * 5. & 6. Calculate Magnitude and Angles (Pitch/Roll)
 * Formula:
 * |a| = sqrt(ax^2 + ay^2 + az^2)
 * Theta (Pitch) = atan(ay / sqrt(ax^2 + az^2))
 * Phi (Roll) = atan(ax / az)
 */
export const calculateDerivedData = (data: RawIMUData[]): ProcessedIMUData[] => {
  const derived: ProcessedIMUData[] = [];
  
  for (let i = 0; i < data.length; i++) {
    const { ax, ay, az, timestamp } = data[i];
    const accMagnitude = Math.sqrt(ax * ax + ay * ay + az * az);
    
    // Using atan2 for better quadrant handling, though paper specifies atan
    const calcPitch = Math.atan(ay / Math.sqrt(ax * ax + az * az)); 
    // Handle singularity if az is 0
    const calcRoll = az !== 0 ? Math.atan(ax / az) : 0; 

    // Calculate Delta Accel (Step IV definition)
    // sum(|a_i,k - a_i-1,k|)
    let deltaAcc = 0;
    if (i > 0) {
      deltaAcc = Math.abs(ax - data[i-1].ax) + 
                 Math.abs(ay - data[i-1].ay) + 
                 Math.abs(az - data[i-1].az);
    }

    derived.push({
      ...data[i],
      accMagnitude,
      calcPitch,
      calcRoll,
      deltaAcc
    });
  }

  return derived;
};

/**
 * 7. Smoothing (Rolling Average)
 * Window W = 5s. For 100Hz, that's 500 points.
 * Applied to Pitch, Roll, and Magnitude.
 */
export const applyRollingAverage = (data: ProcessedIMUData[], freq: number, windowSeconds = 5): ProcessedIMUData[] => {
  const windowSize = Math.floor(windowSeconds * freq);
  if (windowSize <= 1) return data;

  const result = [...data];
  const keys = ['accMagnitude', 'calcPitch', 'calcRoll'] as const;

  for (let i = 0; i < data.length; i++) {
    const start = Math.max(0, i - Math.floor(windowSize / 2));
    const end = Math.min(data.length, i + Math.floor(windowSize / 2));
    const count = end - start;

    keys.forEach(key => {
      let sum = 0;
      for (let j = start; j < end; j++) {
        sum += data[j][key];
      }
      result[i][key] = sum / count;
    });
  }
  return result;
};


// --- IV. TDAR-IDW Core Algorithm ---

/**
 * Main Sync Function
 */
export const runTDARSync = (
  gpsData: GPSData[],
  imuData: ProcessedIMUData[],
  params: TDARParams
): SyncResult[] => {
  const results: SyncResult[] = [];

  // For every GPS timestamp
  for (const gpsPoint of gpsData) {
    const T_new = gpsPoint.timestamp;

    // 1. Find Neighbors
    // Optimization: Since data is sorted by time, we can binary search or scan.
    // Given the constraints, a simple sort by distance is robust.
    
    // Map indices to distance
    const candidates = imuData.map((d, index) => ({
      index,
      timeDiff: Math.abs(d.timestamp - T_new)
    }));

    // Sort by time difference and take top N
    candidates.sort((a, b) => a.timeDiff - b.timeDiff);
    const neighbors = candidates.slice(0, params.neighborCount);

    // 2. Calculate Weights
    let sumWeights = 0;
    const neighborWeights: number[] = [];

    for (const neighbor of neighbors) {
      const idx = neighbor.index;
      const dataPoint = imuData[idx];

      const dt = neighbor.timeDiff; // Delta t
      // Delta a is already calculated in preprocessing as `deltaAcc`
      // Paper handles i=0 case, here we treat deltaAcc as property of the point
      const da = dataPoint.deltaAcc || 0; 

      // w_t = exp(-alpha * dt^p)
      const w_t = Math.exp(-params.alpha * Math.pow(dt, params.p));

      // w_a = exp(-beta * da^q)
      const w_a = Math.exp(-params.beta * Math.pow(da, params.q));

      // Adaptive Coupling: w = (w_t ^ lambda1) * (w_a ^ lambda2)
      const w_i = Math.pow(w_t, params.lambda1) * Math.pow(w_a, params.lambda2);

      neighborWeights.push(w_i);
      sumWeights += w_i;
    }

    // 3. Normalization & 4. Interpolation
    let interpPitch = 0;
    let interpRoll = 0;
    let interpMag = 0;

    for (let k = 0; k < neighbors.length; k++) {
      const idx = neighbors[k].index;
      const rawW = neighborWeights[k];
      const normW = sumWeights > 0 ? rawW / sumWeights : (1 / neighbors.length); // Fallback if 0

      interpPitch += normW * imuData[idx].calcPitch;
      interpRoll += normW * imuData[idx].calcRoll;
      interpMag += normW * imuData[idx].accMagnitude;
    }

    results.push({
      timestamp: T_new,
      pitch: interpPitch,
      roll: interpRoll,
      accMagnitude: interpMag,
      method: 'TDAR-IDW'
    });
  }

  return results;
};

export const runLinearSync = (gpsData: GPSData[], imuData: ProcessedIMUData[]): SyncResult[] => {
    const results: SyncResult[] = [];
    
    for (const gps of gpsData) {
        // Find 2 closest points (before and after)
        // Simple implementation for comparison
        const t = gps.timestamp;
        const nextIdx = imuData.findIndex(d => d.timestamp >= t);
        
        let pitch = 0, roll = 0, mag = 0;

        if (nextIdx > 0 && nextIdx < imuData.length) {
            const p1 = imuData[nextIdx - 1];
            const p2 = imuData[nextIdx];
            const factor = (t - p1.timestamp) / (p2.timestamp - p1.timestamp);
            
            pitch = p1.calcPitch + factor * (p2.calcPitch - p1.calcPitch);
            roll = p1.calcRoll + factor * (p2.calcRoll - p1.calcRoll);
            mag = p1.accMagnitude + factor * (p2.accMagnitude - p1.accMagnitude);
        } else if (nextIdx === 0) {
           pitch = imuData[0].calcPitch;
           roll = imuData[0].calcRoll;
           mag = imuData[0].accMagnitude;
        } else {
           const last = imuData[imuData.length-1];
           pitch = last.calcPitch;
           roll = last.calcRoll;
           mag = last.accMagnitude;
        }

        results.push({
            timestamp: t,
            pitch,
            roll,
            accMagnitude: mag,
            method: 'Linear'
        });
    }
    return results;
}
