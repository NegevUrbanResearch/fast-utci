import { describe, expect, it } from 'vitest';
import { compareShadingIndex } from './compareShadingIndex';

describe('compareShadingIndex', () => {
	it('passes when all per-position shading index values are within tolerance', () => {
		const result = compareShadingIndex({
			python: [0, 0.5, 1],
			webgpu: [0, 0.5000002, 1],
			tolerance: 1e-5
		});

		expect(result.pass).toBe(true);
		expect(result.strictPass).toBe(true);
		expect(result.maxAbsoluteError).toBeCloseTo(0.0000002);
		expect(result.mismatchCountAboveTolerance).toBe(0);
		expect(result.worstCells[0]).toMatchObject({
			pointIndex: 1,
			pythonValue: 0.5,
			webgpuValue: 0.5000002
		});
	});

	it('reports mean absolute error, mismatch count, positions, and worst cells', () => {
		const result = compareShadingIndex({
			python: [0.25, 0.75, 1],
			webgpu: [0.25, 0.5, 0.25],
			positions: [1, 2, 3, 4, 5, 6, 7, 8, 9],
			tolerance: 0.01,
			maxWorstCells: 2
		});

		expect(result.pass).toBe(false);
		expect(result.strictPass).toBe(false);
		expect(result.maxAbsoluteError).toBe(0.75);
		expect(result.meanAbsoluteError).toBeCloseTo((0 + 0.25 + 0.75) / 3);
		expect(result.mismatchCountAboveTolerance).toBe(2);
		expect(result.worstCells).toEqual([
			{
				pointIndex: 2,
				position: { x: 7, y: 8, z: 9 },
				pythonValue: 1,
				webgpuValue: 0.25,
				absoluteError: 0.75,
				solarBitMismatchCount: null,
				attributedToSolarBitFlip: false
			},
			{
				pointIndex: 1,
				position: { x: 4, y: 5, z: 6 },
				pythonValue: 0.75,
				webgpuValue: 0.5,
				absoluteError: 0.25,
				solarBitMismatchCount: null,
				attributedToSolarBitFlip: false
			}
		]);
	});

	it('accepts mismatches explained by known solar ray bit flips with a caveat', () => {
		const result = compareShadingIndex({
			python: [0.5, 0.75],
			webgpu: [0.25, 0.75],
			tolerance: 1e-6,
			sunUpCount: 4,
			solarBitMismatchCounts: [1, 0]
		});

		expect(result.strictPass).toBe(false);
		expect(result.pass).toBe(true);
		expect(result.caveats).toEqual([
			'1 shading-index mismatch(es) above tolerance are attributable to known solar ray bit flips.'
		]);
		expect(result.solarBitFlipAttributedMismatchCount).toBe(1);
		expect(result.worstCells[0]).toMatchObject({
			pointIndex: 0,
			solarBitMismatchCount: 1,
			attributedToSolarBitFlip: true
		});
	});

	it('fails hard when the python reference contains a non-finite value', () => {
		const result = compareShadingIndex({
			python: [0.5, Number.NaN, 0.75],
			webgpu: [0.5, 0.25, 0.75],
			tolerance: 1e-6
		});

		expect(result.pass).toBe(false);
		expect(result.strictPass).toBe(false);
		expect(result.mismatchCountAboveTolerance).toBe(1);
		expect(result.nonFinitePythonValueCount).toBe(1);
		expect(result.nonFiniteWebgpuValueCount).toBe(0);
		expect(result.caveats).toEqual([]);
		expect(result.worstCells[0]).toMatchObject({
			pointIndex: 1,
			pythonValue: Number.NaN,
			webgpuValue: 0.25,
			absoluteError: Number.POSITIVE_INFINITY
		});
	});

	it('fails hard when the webgpu output contains a non-finite value', () => {
		const result = compareShadingIndex({
			python: [0.5, 0.25, 0.75],
			webgpu: [0.5, Number.POSITIVE_INFINITY, 0.75],
			tolerance: 1e-6,
			sunUpCount: 4,
			solarBitMismatchCounts: [0, 1, 0]
		});

		expect(result.pass).toBe(false);
		expect(result.strictPass).toBe(false);
		expect(result.mismatchCountAboveTolerance).toBe(1);
		expect(result.nonFinitePythonValueCount).toBe(0);
		expect(result.nonFiniteWebgpuValueCount).toBe(1);
		expect(result.solarBitFlipAttributedMismatchCount).toBe(0);
		expect(result.caveats).toEqual([]);
		expect(result.worstCells[0]).toMatchObject({
			pointIndex: 1,
			pythonValue: 0.25,
			webgpuValue: Number.POSITIVE_INFINITY,
			absoluteError: Number.POSITIVE_INFINITY
		});
	});

	it('throws on shape mismatches', () => {
		expect(() =>
			compareShadingIndex({
				python: [0, 1],
				webgpu: [0]
			})
		).toThrow(/length mismatch/i);
	});
});
