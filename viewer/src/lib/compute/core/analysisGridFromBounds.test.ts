import { describe, it, expect } from 'vitest';
import { analysisBoundsToRectangularGrid } from './analysisGridFromBounds';

describe('analysisBoundsToRectangularGrid', () => {
	it('xy_ground: bounds 0,2,0,2,z=1, gridSize 1 → 9 points, first (0,1,-2), last (2,1,0)', () => {
		const { points, normals } = analysisBoundsToRectangularGrid({
			bounds: { x_min: 0, x_max: 2, y_min: 0, y_max: 2, z: 1 },
			gridSize: 1,
			coordinateSystem: 'xy_ground'
		});
		expect(points.length).toBe(9);
		expect(normals.length).toBe(9);
		// Grid order: x outer, z inner; minZ=-y_max=-2, maxZ=-y_min=0
		expect(points[0].x).toBe(0);
		expect(points[0].y).toBe(1);
		expect(points[0].z).toBe(-2);
		expect(points[8].x).toBe(2);
		expect(points[8].y).toBe(1);
		expect(points[8].z).toBe(0);
		expect(normals[0].x).toBe(0);
		expect(normals[0].y).toBe(1);
		expect(normals[0].z).toBe(0);
	});
});
