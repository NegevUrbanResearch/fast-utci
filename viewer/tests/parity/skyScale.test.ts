import { describe, it, expect } from 'vitest';
import { TOTAL_TREGENZA_WEIGHT, normalizeSkyExposureToViewFactor } from '$lib/parity/skyScale';

describe('skyScale', () => {
	it('normalizes raw sky to 0–1', () => {
		const raw = [0, 72.6244, 145.2488, 200];
		const out = normalizeSkyExposureToViewFactor(raw);
		expect(out[0]).toBe(0);
		expect(out[1]).toBeCloseTo(0.5);
		expect(out[2]).toBeCloseTo(1);
		expect(out[3]).toBeCloseTo(1);
	});

	it('is idempotent when input is already in 0..1', () => {
		const once = normalizeSkyExposureToViewFactor([0, 0.5, 1]);
		const twice = normalizeSkyExposureToViewFactor(once);
		expect(twice).toEqual(once);
	});
});
