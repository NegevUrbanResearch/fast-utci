import { describe, it, expect } from 'vitest';
import { calculateSunPosition, getSunVectors } from '$lib/compute/core/sunpath';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

// Beer Sheva, Israel: lat=31.25, lon=34.79, timezone=2
const BEER_SHEVA = { lat: 31.25, lon: 34.79, timezone: 2 };

interface FixtureSunPosition {
  hour: number;
  altitude: number;
  azimuth: number;
  is_up: boolean;
  vector: [number, number, number];
}

interface SunpathFixture {
  location: {
    latitude: number;
    longitude: number;
    timezone: number;
  };
  sun_positions: FixtureSunPosition[];
}

const fixture = JSON.parse(
  readFileSync(
    resolveFixturePath(),
    'utf-8'
  )
) as SunpathFixture;

const FIXTURE_LOCATION = {
  lat: fixture.location.latitude,
  lon: fixture.location.longitude,
  timezone: fixture.location.timezone
};

const PARITY_TOLERANCE = 1e-6;

function resolveFixturePath(): string {
  const fromViewer = resolve(process.cwd(), '../data/analyses/Ben-Gurion/20250815_grid_2m_fullday.json');
  const fromRepoRoot = resolve(process.cwd(), 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday.json');
  return existsSync(fromViewer) ? fromViewer : fromRepoRoot;
}

describe('Sunpath', () => {
  describe('calculateSunPosition', () => {
    it('should calculate correct altitude for Beer Sheva summer noon', () => {
      // June 15, noon — sun should be near zenith in Beer Sheva
      const pos = calculateSunPosition(BEER_SHEVA, 6, 15, 12);
      expect(pos.altitude).toBeGreaterThan(75); // ~80° in June
      expect(pos.altitude).toBeLessThan(85);
    });

    it('should calculate correct altitude for Beer Sheva winter noon', () => {
      const pos = calculateSunPosition(BEER_SHEVA, 1, 15, 12);
      expect(pos.altitude).toBeGreaterThan(30); // ~34° in January
      expect(pos.altitude).toBeLessThan(42);
    });

    it('should return isSunUp=false for nighttime', () => {
      const pos = calculateSunPosition(BEER_SHEVA, 6, 15, 2); // 2 AM
      expect(pos.isSunUp).toBe(false);
    });

    it('should return isSunUp=true for daytime', () => {
      const pos = calculateSunPosition(BEER_SHEVA, 6, 15, 12);
      expect(pos.isSunUp).toBe(true);
    });

    it('normalizes solar time with positive modulo for multi-day negative offsets', () => {
      const wrappedLoc = { lat: 10, lon: -540, timezone: 14 };

      const wrapped = calculateSunPosition(wrappedLoc, 3, 20, 0);

      expect(Number.isFinite(wrapped.altitude)).toBe(true);
      expect(Number.isFinite(wrapped.azimuth)).toBe(true);
      expect(wrapped.azimuth).toBeGreaterThanOrEqual(0);
      expect(wrapped.azimuth).toBeLessThan(360);
    });
  });

  describe('getSunVectors', () => {
    it('should return 24 vectors for a single day', () => {
      const vectors = getSunVectors(BEER_SHEVA, 6, 15);
      expect(vectors.sunVectors.length).toBe(24);
      expect(vectors.isSunUp.length).toBe(24);
    });

    it('should return unit vectors', () => {
      const vectors = getSunVectors(BEER_SHEVA, 6, 15);
      for (let i = 0; i < 24; i++) {
        if (vectors.isSunUp[i]) {
          const v = vectors.sunVectors[i];
          const mag = Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2);
          expect(mag).toBeCloseTo(1.0, 4);
        }
      }
    });
  });

  describe('Ladybug parity', () => {
    it('matches exported Ladybug Ben-Gurion sun positions for every hour on August 15', () => {
      const vectors = getSunVectors(FIXTURE_LOCATION, 8, 15);

      expect(fixture.sun_positions).toHaveLength(24);
      for (const expected of fixture.sun_positions) {
        const actual = calculateSunPosition(FIXTURE_LOCATION, 8, 15, expected.hour);

        expect(actual.altitude).toBeCloseTo(expected.altitude, 6);
        expect(actual.azimuth).toBeCloseTo(expected.azimuth, 6);
        expect(actual.isSunUp).toBe(expected.is_up);
        expect(vectors.isSunUp[expected.hour]).toBe(expected.is_up);

        for (let axis = 0; axis < 3; axis++) {
          expect(Math.abs(vectors.sunVectors[expected.hour][axis] - expected.vector[axis])).toBeLessThan(
            PARITY_TOLERANCE
          );
        }
      }
    });
  });
});
