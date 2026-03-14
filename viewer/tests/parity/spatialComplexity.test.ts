import { describe, expect, it } from 'vitest';
import { computeSpatialComplexity, inferRectGridShapeFromPositions } from '$lib/parity/spatialComplexity';

describe('spatial complexity diagnostics', () => {
	it('computes non-zero complexity for a varying field', () => {
		const width = 3;
		const height = 3;
		const field = [
			0, 1, 2,
			1, 2, 3,
			2, 3, 4
		];
		const metrics = computeSpatialComplexity(field, width, height);
		expect(metrics.gradientEnergy).toBeGreaterThan(0);
		expect(metrics.variance).toBeGreaterThan(0);
		expect(metrics.entropy).toBeGreaterThan(0);
	});

	it('infers rectangular shape from canonical xyz positions', () => {
		const positions = [
			0, 0.9, 0,  0, 0.9, 2,
			2, 0.9, 0,  2, 0.9, 2
		];
		const shape = inferRectGridShapeFromPositions(positions);
		expect(shape).toEqual({ width: 2, height: 2 });
	});
});
