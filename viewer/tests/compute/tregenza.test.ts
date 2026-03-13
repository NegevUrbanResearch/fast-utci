import { describe, it, expect } from 'vitest';
import { getTregenzaDome } from '$lib/compute/tregenza';

describe('Tregenza Dome', () => {
  it('should return exactly 145 direction vectors', () => {
    const dome = getTregenzaDome();
    expect(dome.vectors).toHaveLength(145);
  });

  it('should return exactly 145 weights', () => {
    const dome = getTregenzaDome();
    expect(dome.weights).toHaveLength(145);
  });

  it('should have unit-length direction vectors', () => {
    const dome = getTregenzaDome();
    dome.vectors.forEach(v => {
      const mag = Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2);
      expect(mag).toBeCloseTo(1.0, 4);
    });
  });

  it('should have weights that sum to approximately 145', () => {
    // Note: Python script used dome_patch_weights(1) which sums to 145 (one per patch equivalent).
    const dome = getTregenzaDome();
    const sum = dome.weights.reduce((a, b) => a + b, 0);
    // ladybug dome_patch_weights sum to 145
    expect(sum).toBeCloseTo(145.0, 1);
  });

  it('all vectors should point upward (z > 0)', () => {
    const dome = getTregenzaDome();
    dome.vectors.forEach(v => {
      // Allow slight floating point tolerance around 0, but Tregenza doesn't have exactly z=0 horizons.
      // The lowest band is alt=6deg -> z = sin(6deg) = 0.104
      expect(v[2]).toBeGreaterThan(0.05); 
    });
  });

  it('sky exposure weights can be normalized to view factor', () => {
    const dome = getTregenzaDome();
    const total = dome.weights.reduce((a, b) => a + b, 0);

    // In the current representation, weights sum to ~145 (equivalent visible
    // patch count). A normalized sky view factor in [0,1] is obtained by
    // dividing any accumulated sky_exposure by this total.
    const normalizedAllSky = dome.weights.reduce((a, b) => a + b / total, 0);
    expect(normalizedAllSky).toBeCloseTo(1.0, 3);
  });
});
