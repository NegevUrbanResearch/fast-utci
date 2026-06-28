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
 * Solar position using the same Julian-century NOAA equations used by Ladybug.
 *
 * The returned altitude/azimuth are in the usual geographic sense, but any
 * derived direction vectors in this module are expressed in a local ENU
 * frame: X = East, Y = North, Z = Up (Z-up, matching the Python pipeline).
 * Callers that need Three.js world vectors must rotate into its Y-up frame.
 */
export function calculateSunPosition(loc: LocationData, month: number, day: number, hour: number): SunPosition {
  const year = 2017;
  const julianDay = daysFrom010119(year, month, day) + 2415018.5 + roundHalfEven(hour * 60 / 1440, 2) - loc.timezone / 24;
  const julianCentury = (julianDay - 2451545) / 36525;

  const geomMeanLong = normalizeDegrees(280.46646 + julianCentury * (36000.76983 + julianCentury * 0.0003032));
  const geomMeanAnom = 357.52911 + julianCentury * (35999.05029 - 0.0001537 * julianCentury);
  const earthEccentricity = 0.016708634 - julianCentury * (0.000042037 + 0.0000001267 * julianCentury);
  const sunEqCenter = Math.sin(degToRad(geomMeanAnom)) * (1.914602 - julianCentury * (0.004817 + 0.000014 * julianCentury))
    + Math.sin(degToRad(2 * geomMeanAnom)) * (0.019993 - 0.000101 * julianCentury)
    + Math.sin(degToRad(3 * geomMeanAnom)) * 0.000289;
  const sunTrueLong = geomMeanLong + sunEqCenter;
  const sunAppLong = sunTrueLong - 0.00569 - 0.00478 * Math.sin(degToRad(125.04 - 1934.136 * julianCentury));
  const meanObliq = 23 + (26 + ((21.448 - julianCentury * (46.815 + julianCentury * (0.00059 - julianCentury * 0.001813)))) / 60) / 60;
  const obliqCorr = meanObliq + 0.00256 * Math.cos(degToRad(125.04 - 1934.136 * julianCentury));
  const declRad = Math.asin(Math.sin(degToRad(obliqCorr)) * Math.sin(degToRad(sunAppLong)));

  const varY = Math.tan(degToRad(obliqCorr) / 2) ** 2;
  const eqTime = 4 * radToDeg(
    varY * Math.sin(2 * degToRad(geomMeanLong))
    - 2 * earthEccentricity * Math.sin(degToRad(geomMeanAnom))
    + 4 * earthEccentricity * varY * Math.sin(degToRad(geomMeanAnom)) * Math.cos(2 * degToRad(geomMeanLong))
    - 0.5 * varY * varY * Math.sin(4 * degToRad(geomMeanLong))
    - 1.25 * earthEccentricity * earthEccentricity * Math.sin(2 * degToRad(geomMeanAnom))
  );

  const solarTimeHours = positiveModulo(hour * 60 + eqTime + 4 * loc.lon - 60 * loc.timezone, 1440) / 60;
  const solarTimeMinutes = solarTimeHours * 60;
  const hourAngle = solarTimeMinutes < 0 ? solarTimeMinutes / 4 + 180 : solarTimeMinutes / 4 - 180;

  const latRad = degToRad(loc.lat);
  const haRad = degToRad(hourAngle);
  let cosZenith = Math.sin(latRad) * Math.sin(declRad) + Math.cos(latRad) * Math.cos(declRad) * Math.cos(haRad);
  cosZenith = Math.max(-1, Math.min(1, cosZenith));

  const zenith = radToDeg(Math.acos(cosZenith));
  const rawAltitude = 90 - zenith;
  const altitude = rawAltitude + atmosphericRefractionCorrection(rawAltitude);

  const zenithRad = degToRad(zenith);
  const azInit = ((Math.sin(latRad) * Math.cos(zenithRad)) - Math.sin(declRad)) / (Math.cos(latRad) * Math.sin(zenithRad));
  const azimuth = Math.abs(azInit) <= 1
    ? hourAngle > 0
      ? (radToDeg(Math.acos(azInit)) + 180) % 360
      : (540 - radToDeg(Math.acos(azInit))) % 360
    : 180;

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

function daysFrom010119(year: number, month: number, day: number): number {
  let total = 0;
  for (let y = 1900; y < year; y++) {
    total += isLeapYear(y) ? 366 : 365;
  }

  const daysInMonth = [31, isLeapYear(year) ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  for (let m = 1; m < month; m++) {
    total += daysInMonth[m - 1];
  }

  return total + day + 1;
}

function isLeapYear(year: number): boolean {
  return year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
}

function normalizeDegrees(value: number): number {
  return positiveModulo(value, 360);
}

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function degToRad(degrees: number): number {
  return degrees * Math.PI / 180;
}

function radToDeg(radians: number): number {
  return radians * 180 / Math.PI;
}

function roundHalfEven(value: number, decimals: number): number {
  const factor = 10 ** decimals;
  const scaled = value * factor;
  const floor = Math.floor(scaled);
  const fraction = scaled - floor;

  if (Math.abs(fraction - 0.5) < Number.EPSILON * 10) {
    return (floor % 2 === 0 ? floor : floor + 1) / factor;
  }

  return Math.round(scaled) / factor;
}

function atmosphericRefractionCorrection(altitude: number): number {
  if (altitude > 85) return 0;

  const tanElevation = Math.tan(degToRad(altitude));
  if (altitude > 5) {
    return (58.1 / tanElevation - 0.07 / (tanElevation ** 3) + 0.000086 / (tanElevation ** 5)) / 3600;
  }

  if (altitude > -0.575) {
    return (1735 - 518.2 * altitude + 103.4 * altitude ** 2 - 12.79 * altitude ** 3 + 0.711 * altitude ** 4) / 3600;
  }

  return (-20.772 / tanElevation) / 3600;
}
