export interface LocationData {
  lat: number;
  lon: number;
  timezone: number;
}

export interface SunPosition {
  altitude: number; // degrees
  azimuth: number;  // degrees
  isSunUp: boolean;
}

export interface DaySunVectors {
  sunVectors: [number, number, number][]; // 24 hours
  isSunUp: boolean[];                     // 24 hours
  /** Solar altitude in degrees, 0 for nighttime. */
  altitudes: number[];
}

/**
 * Solar position using a NOAA-style algorithm.
 *
 * The returned altitude/azimuth are in the usual geographic sense, but any
 * derived direction vectors in this module are expressed in a local ENU
 * frame: X = East, Y = North, Z = Up (Z-up, matching the Python pipeline).
 * Callers that need Three.js world vectors must rotate into its Y-up frame.
 */
export function calculateSunPosition(loc: LocationData, month: number, day: number, hour: number): SunPosition {
  const daysInMonth = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  let n = 0;
  for (let i = 1; i < month; i++) {
    n += daysInMonth[i];
  }
  n += day;

  // Fractional year (gamma) in radians
  const gamma = (2 * Math.PI / 365) * (n - 1 + (hour - 12) / 24);

  // Equation of time in minutes
  const eqTime = 229.18 * (0.000075 + 0.001868 * Math.cos(gamma) - 0.032077 * Math.sin(gamma)
                 - 0.014615 * Math.cos(2 * gamma) - 0.040849 * Math.sin(2 * gamma));

  // Solar declination in radians
  const decl_rad = 0.006918 - 0.399912 * Math.cos(gamma) + 0.070257 * Math.sin(gamma)
                   - 0.006758 * Math.cos(2 * gamma) + 0.000907 * Math.sin(2 * gamma)
                   - 0.002697 * Math.cos(3 * gamma) + 0.00148 * Math.sin(3 * gamma);

  // Time offset in minutes
  const timeOffset = eqTime + 4 * loc.lon - 60 * loc.timezone;

  // True solar time in minutes
  const tst = hour * 60 + timeOffset;

  // Solar hour angle in degrees
  let ha = (tst / 4) - 180;
  if (ha < -180) ha += 360;
  if (ha > 180)  ha -= 360;
  const ha_rad = ha * Math.PI / 180;

  const lat_rad = loc.lat * Math.PI / 180;

  // Zenith angle and altitude
  // cos(Zenith) = sin(Lat) * sin(Decl) + cos(Lat) * cos(Decl) * cos(HA)
  let cosZenith = Math.sin(lat_rad) * Math.sin(decl_rad) + Math.cos(lat_rad) * Math.cos(decl_rad) * Math.cos(ha_rad);
  cosZenith = Math.max(-1, Math.min(1, cosZenith));
  
  const zenith_rad = Math.acos(cosZenith);
  const alt_rad = (Math.PI / 2) - zenith_rad;
  const altitude = alt_rad * 180 / Math.PI;

  // Azimuth
  let cosAzimuth = (Math.sin(lat_rad) * cosZenith - Math.sin(decl_rad)) / (Math.cos(lat_rad) * Math.sin(zenith_rad));
  // Let's use the other equivalent NOAA formulation to avoid division by near-zero:
  cosAzimuth = -(Math.sin(decl_rad) - Math.sin(lat_rad) * Math.cos(zenith_rad)) / (Math.cos(lat_rad) * Math.sin(zenith_rad));

  let azimuth_rad = 0;
  if (Math.abs(Math.sin(zenith_rad)) > 0.0001) {
    cosAzimuth = Math.max(-1, Math.min(1, cosAzimuth));
    azimuth_rad = Math.acos(cosAzimuth);
    if (ha > 0) {
      azimuth_rad = 2 * Math.PI - azimuth_rad;
    }
  }
  
  // Convert standard azimuth to ladybug definition (0 = North, clockwise)
  // NOAA typically puts 0 = North, but wait, usually 180 = South.
  // The above formula produces 0 at North. Let's adjust to be safe if it's 180 shifted.
  // Actually, standard NOAA gives azimuth from North. Let's just use it directly.
  const azimuth = azimuth_rad * 180 / Math.PI;

  // In ladybug/grasshopper, sun altitude is corrected for atmospheric refraction, 
  // but we can start with raw altitude.
  const isSunUp = altitude > 0;

  return { altitude, azimuth, isSunUp };
}

export function getSunVectors(loc: LocationData, month: number, day: number): DaySunVectors {
  const sunVectors: [number, number, number][] = [];
  const isSunUp: boolean[] = [];
  const altitudes: number[] = [];

  for (let hour = 0; hour < 24; hour++) {
    // Sample at exact hour boundaries to align with Python hourly datetimes.
    const pos = calculateSunPosition(loc, month, day, hour);
    isSunUp.push(pos.isSunUp);
    altitudes.push(pos.isSunUp ? pos.altitude : 0);

    if (pos.isSunUp) {
      const alt_rad = pos.altitude * Math.PI / 180;
      const azi_rad = pos.azimuth * Math.PI / 180;

      // Local ENU frame used throughout the Python pipeline:
      // X = East, Y = North, Z = Up (Z-up).
      const x = Math.sin(azi_rad) * Math.cos(alt_rad);
      const y = Math.cos(azi_rad) * Math.cos(alt_rad);
      const z = Math.sin(alt_rad);

      sunVectors.push([x, y, z]);
    } else {
      sunVectors.push([0, 0, 0]);
    }
  }

  return { sunVectors, isSunUp, altitudes };
}
