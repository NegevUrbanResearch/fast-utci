import { describe, it, expect } from 'vitest';
import { calculateUTCI, calculateBoundaryAveragedUtciSeries } from '$lib/compute/core/utci';

describe('UTCI domain policy parity', () => {
	it('returns NaN outside validity domain in strict-domain mode', () => {
		expect(Number.isNaN(calculateUTCI(60, 60, 1, 50, { policy: 'strict-domain' }))).toBe(true);
		expect(Number.isNaN(calculateUTCI(20, 120, 1, 50, { policy: 'strict-domain' }))).toBe(true);
	});

	it('clamps outside-domain values in clamped-domain mode', () => {
		const clamped = calculateUTCI(60, 60, 1, 50, { policy: 'clamped-domain' });
		const equivalentAtBoundary = calculateUTCI(50, 50, 1, 50, { policy: 'clamped-domain' });
		expect(Number.isFinite(clamped)).toBe(true);
		expect(clamped).toBeCloseTo(equivalentAtBoundary, 6);
	});

	it('keeps boundary averaging semantics with policy applied', () => {
		const airTemps = [25, 26, 27];
		const mrts = [30, 31, 32];
		const windSpeeds = [1, 1, 1];
		const relativeHumidities = [50, 50, 50];

		const series = calculateBoundaryAveragedUtciSeries({
			airTemps,
			mrts,
			windSpeeds,
			relativeHumidities,
			policy: 'strict-domain'
		});
		expect(series).toHaveLength(3);
		expect(series[2]).toBeCloseTo(calculateUTCI(27, 32, 1, 50, { policy: 'strict-domain' }), 6);
	});
});
