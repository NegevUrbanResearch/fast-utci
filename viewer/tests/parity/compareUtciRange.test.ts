import { describe, it, expect } from 'vitest';
import { compareUtciRange } from '$lib/parity/compareUtciRange';

describe('compareUtciRange', () => {
	it('returns pass when within tolerance', () => {
		const result = compareUtciRange({
			ref: { min: 22, max: 39, mean: 28 },
			webgpu: { min: 22.5, max: 38.5, mean: 28.2 },
			toleranceMin: 1,
			toleranceMax: 1,
			toleranceMean: 0.5,
		});
		expect(result.pass).toBe(true);
	});

	it('returns fail when min diff exceeds tolerance', () => {
		const result = compareUtciRange({
			ref: { min: 22, max: 39, mean: 28 },
			webgpu: { min: 20, max: 39, mean: 28 },
			toleranceMin: 1,
			toleranceMax: 1,
			toleranceMean: 0.5,
		});
		expect(result.pass).toBe(false);
		expect(result.minDiff).toBe(2);
	});
});
