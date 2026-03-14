import { describe, it, expect } from 'vitest';
import { compareIntermediates } from '$lib/parity/compareIntermediates';

describe('strict pointwise parity', () => {
	it('fails immediately on length mismatch', () => {
		expect(() =>
			compareIntermediates({
				ref: new Float32Array([1, 2, 3]),
				webgpu: new Float32Array([1, 2]),
				tolerance: 1e-5
			})
		).toThrow(/Length mismatch/);
	});
});
