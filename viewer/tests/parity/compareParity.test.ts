import { describe, it, expect } from 'vitest';
import { compareParity, type ParityResult } from '$lib/parity/compareParity';

describe('compareParity', () => {
	it('returns pass when reference and webgpu UTCI match within tolerance', () => {
		const ref = new Float32Array([25, 26, 27]);
		const webgpu = new Float32Array([25.1, 25.9, 27.05]);
		const result = compareParity({ utciRef: ref, utciWebgpu: webgpu, toleranceC: 0.5 });
		expect(result.pass).toBe(true);
		expect(result.maxError).toBeLessThanOrEqual(0.5);
	});

	it('returns fail when any point exceeds tolerance', () => {
		const ref = new Float32Array([25, 26]);
		const webgpu = new Float32Array([25, 30]);
		const result = compareParity({ utciRef: ref, utciWebgpu: webgpu, toleranceC: 1 });
		expect(result.pass).toBe(false);
		expect(result.maxError).toBe(4);
	});

	it('throws when lengths differ', () => {
		const ref = new Float32Array(3);
		const webgpu = new Float32Array(5);
		expect(() => compareParity({ utciRef: ref, utciWebgpu: webgpu })).toThrow(/Length mismatch/);
	});

	it('computes rmse and withinTolerancePct', () => {
		const ref = new Float32Array([20, 22, 24]);
		const webgpu = new Float32Array([20, 22.5, 25]);
		const result = compareParity({ utciRef: ref, utciWebgpu: webgpu, toleranceC: 1 });
		expect(result.rmse).toBeGreaterThan(0);
		expect(result.withinTolerancePct).toBeGreaterThanOrEqual(0);
		expect(result.withinTolerancePct).toBeLessThanOrEqual(100);
	});
});
