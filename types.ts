export interface RawIMUData {
  timestamp: number; // Seconds
  ax: number;
  ay: number;
  az: number;
  // Optional pre-calculated angles
  pitch?: number;
  roll?: number;
}

export interface ProcessedIMUData extends RawIMUData {
  accMagnitude: number; // |a|
  calcPitch: number;    // Calculated Pitch (theta)
  calcRoll: number;     // Calculated Roll (phi)
  
  // Algorithmic helpers
  deltaAcc?: number;    // Sum of abs diffs (accel change)
  isOutlier?: boolean;
}

export interface GPSData {
  timestamp: number; // Seconds
  lat?: number;
  lon?: number;
}

export interface SyncResult {
  timestamp: number;
  pitch: number;
  roll: number;
  accMagnitude: number;
  method: string;
}

export interface TDARParams {
  alpha: number;   // Time decay coeff (e.g., 0.1)
  beta: number;    // Accel decay coeff (e.g., 0.05)
  p: number;       // Time power (e.g., 2)
  q: number;       // Accel power (e.g., 2)
  lambda1: number; // Time weight influence (0.2 - 0.8)
  lambda2: number; // Accel weight influence (1 - lambda1)
  neighborCount: number; // N (e.g., 10)
}

export interface SimulationConfig {
  duration: number; // seconds
  imuFreq: number;  // Hz
  gpsFreq: number;  // Hz
}