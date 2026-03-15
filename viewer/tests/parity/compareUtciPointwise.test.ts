import { describe, expect, it } from 'vitest';
import { compareUtciPointwise } from '$lib/parity/compareUtciPointwise';

describe('compareUtciPointwise', () => {
	it('fails strict mode when any UTCI cell exceeds tolerance', () => {
		const ref = [
			[25, 26],
			[27, 28]
		];
		const webgpu = [
			[25, 26],
			[27, 30]
		];
		const result = compareUtciPointwise({ ref, webgpu, tolerance: 0.5 });
		expect(result.pass).toBe(false);
		expect(result.maxError).toBe(2);
		expect(result.worst.hour).toBe(1);
		expect(result.worst.pointIndex).toBe(1);
	});

	it('throws on hour-count mismatch', () => {
		expect(() =>
			compareUtciPointwise({
				ref: [[1, 2]],
				webgpu: [
					[1, 2],
					[1, 2]
				],
				tolerance: 0.1
			})
		).toThrow(/hour count mismatch/i);
	});
});
