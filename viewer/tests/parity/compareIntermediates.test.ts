import { describe, it, expect } from 'vitest';
import {
	compareIntermediates,
	compareIntermediatesStats,
	analyzeDiffs
} from '$lib/parity/compareIntermediates';

describe('compareIntermediatesStats', () => {
	it('passes when mean and max within tolerance (different lengths ok)', () => {
		const ref = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]); // mean 0.3, max 0.5
		const webgpu = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.3]); // n=6, mean 0.3, max 0.5
		const r = compareIntermediatesStats({ ref, webgpu, toleranceMean: 0.02, toleranceMax: 0.02 });
		expect(r.pass).toBe(true);
		expect(r.refStats.mean).toBeCloseTo(0.3);
		expect(r.refStats.n).toBe(5);
		expect(r.webgpuStats.n).toBe(6);
	});

	it('fails when mean exceeds tolerance', () => {
		const ref = new Float32Array([0.2, 0.4]);
		const webgpu = new Float32Array([0.5, 0.7]); // mean 0.6 vs ref 0.3
		const r = compareIntermediatesStats({ ref, webgpu, toleranceMean: 0.1, toleranceMax: 1 });
		expect(r.pass).toBe(false);
		expect(r.meanDiff).toBeCloseTo(0.3);
	});
});

describe('compareIntermediates', () => {
	it('returns pass when arrays match within tolerance', () => {
		const ref = new Float32Array([0.0, 0.5, 1.0]);
		const webgpu = new Float32Array([0.0, 0.50001, 1.0]);
		const r = compareIntermediates({ ref, webgpu, tolerance: 1e-4 });
		expect(r.pass).toBe(true);
		expect(r.maxError).toBeLessThanOrEqual(1e-4);
	});

	it('returns fail when any value exceeds tolerance', () => {
		const ref = new Float32Array([0.5]);
		const webgpu = new Float32Array([0.6]);
		const r = compareIntermediates({ ref, webgpu, tolerance: 0.05 });
		expect(r.pass).toBe(false);
		expect(r.maxError).toBeCloseTo(0.1);
	});

	it('throws when lengths differ', () => {
		expect(() =>
			compareIntermediates({ ref: new Float32Array(3), webgpu: new Float32Array(5), tolerance: 0.01 })
		).toThrow(/Length mismatch/);
	});
});

describe('analyzeDiffs', () => {
	it('returns sameLength false when lengths differ', () => {
		const r = analyzeDiffs({
			ref: new Float32Array([0, 1]),
			webgpu: new Float32Array([0, 1, 2])
		});
		expect(r.sameLength).toBe(false);
		expect(r.n).toBe(2);
	});

	it('returns diffStats and worstIndices when same length', () => {
		const ref = new Float32Array([1, 2, 3, 4, 5]);
		const webgpu = new Float32Array([1, 2.5, 3, 4, 10]); // diffs: 0, 0.5, 0, 0, 5
		const r = analyzeDiffs({ ref, webgpu, maxWorst: 3 });
		expect(r.sameLength).toBe(true);
		expect(r.n).toBe(5);
		expect(r.diffStats).toBeDefined();
		expect(r.diffStats!.mean).toBeCloseTo(1.1);
		expect(r.diffStats!.max).toBe(5);
		expect(r.diffStats!.min).toBe(0);
		expect(r.worstIndices!.length).toBeLessThanOrEqual(3);
		expect(r.worstIndices![0].index).toBe(4);
		expect(r.worstIndices![0].diff).toBe(5);
	});
});
