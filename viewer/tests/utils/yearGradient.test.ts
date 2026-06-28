import { describe, it, expect } from 'vitest';
import { getYearRingConicGradient, YEAR_GRADIENT_STOPS } from '$lib/utils/yearGradient';

describe('yearGradient', () => {
	it('returns a conic-gradient string', () => {
		const g = getYearRingConicGradient();
		expect(g).toMatch(/^conic-gradient/);
		expect(g).toContain('#8B6CC7');
	});

	it('has 17 gradient stops', () => {
		expect(YEAR_GRADIENT_STOPS.length).toBe(17);
	});
});
