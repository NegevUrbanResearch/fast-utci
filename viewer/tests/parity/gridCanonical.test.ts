import { describe, it, expect } from 'vitest';
import { canonicalGridChecksum, canonicalGridPoints } from '$lib/parity/gridCanonical';

describe('canonical grid ordering', () => {
	it('produces stable checksum for fixed bounds/grid', () => {
		const checksum = canonicalGridChecksum({
			bounds: { x_min: 0, x_max: 4, y_min: 0, y_max: 4, z: 0.9 },
			gridSize: 2,
			coordinateSystem: 'xy_ground'
		});
		expect(checksum).toBe('633353b7');
	});

	it('uses deterministic x-major then z-major ordering', () => {
		const { points, numPoints } = canonicalGridPoints({
			bounds: { x_min: 0, x_max: 2, y_min: 0, y_max: 2, z: 1.25 },
			gridSize: 1,
			coordinateSystem: 'xy_ground'
		});

		expect(numPoints).toBe(9);
		expect(Array.from(points.slice(0, 6))).toEqual([0, 1.25, -2, 0, 1.25, -1]);
		expect(Array.from(points.slice(-6))).toEqual([2, 1.25, -1, 2, 1.25, 0]);
	});
});
