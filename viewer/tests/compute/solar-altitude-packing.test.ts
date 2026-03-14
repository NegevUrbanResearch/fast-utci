import { describe, it, expect } from 'vitest';
import { calculateSunPosition, getSunVectors } from '$lib/compute/sunpath';

describe('Sun altitude packing', () => {
	const BEER_SHEVA = { lat: 31.25, lon: 34.79, timezone: 2 };

	it('getSunVectors should also return altitudes for each hour', () => {
		const result = getSunVectors(BEER_SHEVA, 8, 15);
		expect(result.altitudes).toBeDefined();
		expect(result.altitudes).toHaveLength(24);
	});

	it('daytime altitudes should be positive, nighttime should be zero', () => {
		const result = getSunVectors(BEER_SHEVA, 8, 15);
		for (let h = 0; h < 24; h++) {
			if (result.isSunUp[h]) {
				expect(result.altitudes[h]).toBeGreaterThan(0);
			} else {
				expect(result.altitudes[h]).toBe(0);
			}
		}
	});

	it('noon altitude in August Beer Sheva should be ~75-82 degrees', () => {
		const result = getSunVectors(BEER_SHEVA, 8, 15);
		expect(result.altitudes[12]).toBeGreaterThan(70);
		expect(result.altitudes[12]).toBeLessThan(85);
	});

	it('hourly packing should match exact-hour sun position (no +0.5h shift)', () => {
		const result = getSunVectors(BEER_SHEVA, 8, 15);
		const noon = calculateSunPosition(BEER_SHEVA, 8, 15, 12);
		expect(result.altitudes[12]).toBeCloseTo(noon.isSunUp ? noon.altitude : 0, 6);
	});
});
