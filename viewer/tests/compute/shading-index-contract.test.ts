import { describe, expect, it } from 'vitest';

function computeReferenceShadingIndex(params: {
	exposure: number[][];
	isSunUp: boolean[];
	epsilon?: number;
}): number[] {
	const epsilon = params.epsilon ?? 1e-6;
	const sunlightIndices = params.isSunUp
		.map((value, index) => (value ? index : -1))
		.filter((index) => index >= 0);
	if (sunlightIndices.length === 0) {
		return params.exposure.map(() => 1);
	}
	return params.exposure.map((pointExposure) => {
		let shaded = 0;
		for (const hourIndex of sunlightIndices) {
			if (pointExposure[hourIndex] <= epsilon) shaded += 1;
		}
		return shaded / sunlightIndices.length;
	});
}

describe('shading index contract', () => {
	it('returns 1 for fully shaded sun-up exposure', () => {
		const result = computeReferenceShadingIndex({
			exposure: [[0, 0, 0]],
			isSunUp: [true, true, true]
		});

		expect(result).toEqual([1]);
	});

	it('returns 0 for fully exposed sun-up exposure', () => {
		const result = computeReferenceShadingIndex({
			exposure: [[1, 0, 1, 1]],
			isSunUp: [true, false, true, true]
		});

		expect(result).toEqual([0]);
	});

	it('returns the ratio of shaded sun-up hours', () => {
		const result = computeReferenceShadingIndex({
			exposure: [
				[0, 1, 0, 1],
				[1, 0, 0, 1]
			],
			isSunUp: [true, true, true, true]
		});

		expect(result).toEqual([0.5, 0.5]);
	});

	it('does not treat partial direct exposure as shaded', () => {
		const result = computeReferenceShadingIndex({
			exposure: [[0, 0.25, 1]],
			isSunUp: [true, true, true]
		});

		expect(result).toEqual([1 / 3]);
	});

	it('treats exposure at or below epsilon as shaded', () => {
		const result = computeReferenceShadingIndex({
			exposure: [[1e-6, 1.1e-6, 0]],
			isSunUp: [true, true, true]
		});

		expect(result[0]).toBeCloseTo(2 / 3, 12);
	});

	it('returns 1 for every point when there are no sun-up hours', () => {
		const result = computeReferenceShadingIndex({
			exposure: [
				[1, 1, 1],
				[0.5, 0.25, 0]
			],
			isSunUp: [false, false, false]
		});

		expect(result).toEqual([1, 1]);
	});
});
