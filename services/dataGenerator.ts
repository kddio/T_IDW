import { GPSData, RawIMUData, SimulationConfig } from '../types';

export const generateSyntheticData = (config: SimulationConfig) => {
  const imuData: RawIMUData[] = [];
  const gpsData: GPSData[] = [];

  const totalIMUSamples = Math.floor(config.duration * config.imuFreq);
  const totalGPSSamples = Math.floor(config.duration * config.gpsFreq);

  // Normalize time to start at 0 for the algorithm
  const startTime = 0; 

  // Generate IMU Data (100Hz)
  for (let i = 0; i < totalIMUSamples; i++) {
    const t = i / config.imuFreq;
    const timestamp = startTime + t;

    // Simulate "Gradual" (0-33%, 66-100%) vs "Rapid" (33-66%) movement
    let intensity = 1;
    let freq = 1;

    if (t > config.duration * 0.33 && t < config.duration * 0.66) {
      intensity = 8; // Higher amplitude
      freq = 5;      // Higher frequency noise
    }

    // Base Gravity (approx 9.8 on Z) + Sine wave motion + Random Noise
    const ax = Math.sin(t * freq) * intensity + (Math.random() - 0.5) * 0.2;
    const ay = Math.cos(t * freq * 0.5) * intensity + (Math.random() - 0.5) * 0.2;
    // az has gravity
    let az = 9.81 + Math.sin(t * freq * 2) * (intensity * 0.5) + (Math.random() - 0.5) * 0.2;

    // Add occasional outlier (spike) for MAD testing (> 50m/s^2 or similar)
    if (Math.random() > 0.998) {
      az += 60; // Massive spike
    }

    imuData.push({
      timestamp,
      ax,
      ay,
      az,
    });
  }

  // Generate GPS Data (1Hz) 
  for (let j = 0; j < totalGPSSamples; j++) {
    const t = j / config.gpsFreq;
    // Add a random offset (0.1 - 0.5s) to ensure misalignment with IMU ticks
    const timeOffset = 0.12 + (Math.random() * 0.05); 
    const timestamp = startTime + t + timeOffset; 
    
    if (timestamp < config.duration) {
      gpsData.push({
        timestamp,
        lat: 0,
        lon: 0
      });
    }
  }

  return { imuData, gpsData };
};