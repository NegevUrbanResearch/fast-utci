import { describe, it, expect } from 'vitest';
import { calculateSunPosition, getSunVectors } from '$lib/compute/sunpath';

// Beer Sheva, Israel: lat=31.25, lon=34.79, timezone=2
const BEER_SHEVA = { lat: 31.25, lon: 34.79, timezone: 2 };

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
});
