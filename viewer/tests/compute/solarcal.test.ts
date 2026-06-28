import { describe, it, expect } from 'vitest';
import { computeSolarCal } from '$lib/compute/core/solarcal';

describe('SolarCal MRT', () => {
  it('should compute ERF=0 when fully shaded (no direct, no diffuse from sky context)', () => {
    // Note: fully shaded still has some diffuse and ground reflected if skyViewFactor > 0, 
    // but the test in the plan implies ERF should be very low or 0 if everything is 0.
    const result = computeSolarCal({
      directNormalRad: 0,
      diffuseHorizRad: 0,
      horizInfrared: 400, // W/m2, typical longwave
      solarAltitude: 45,
      solarExposure: 0,
      skyViewFactor: 0,
      groundReflectance: 0.2,
			airTemp: 30,
		});
		expect(result.shortwaveErf).toBeCloseTo(0, 1);
		expect(result.shortwaveDeltaMRT).toBeCloseTo(0, 1);
	});

  it('should compute positive ERF in direct sun', () => {
    const result = computeSolarCal({
      directNormalRad: 800,
      diffuseHorizRad: 200,
      horizInfrared: 400,
      solarAltitude: 60,
      solarExposure: 1,
      skyViewFactor: 0.8,
      groundReflectance: 0.2,
			airTemp: 30,
		});
		expect(result.shortwaveErf).toBeGreaterThan(0);
		expect(result.shortwaveDeltaMRT).toBeGreaterThan(0);
		expect(Number.isFinite(result.longwaveErf)).toBe(true);
		expect(Number.isFinite(result.longwaveDeltaMRT)).toBe(true);
		expect(result.outdoorMRT).toBeGreaterThan(30);
	});
});
