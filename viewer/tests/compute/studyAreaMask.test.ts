import { describe, expect, it } from 'vitest';
import {
	buildStudyAreaMaskFromProjectedTriangles
} from '$lib/compute/core/studyAreaMask';
import {
	canonicalGridPoints,
	canonicalGridPointsForActiveIndices
} from '$lib/compute/core/canonicalGrid';
import type { ProjectedTriangle2D } from '$lib/types/analysis';

describe('buildStudyAreaMaskFromProjectedTriangles', () => {
	it('marks the full canonical grid active when the footprint covers the full bounds', () => {
		const result = buildStudyAreaMaskFromProjectedTriangles({
			bounds: { x_min: 0, x_max: 4, y_min: 0, y_max: 4, z: 1.5 },
			gridSize: 2,
			coordinateSystem: 'xy_ground',
			triangles: [
				[0, -4, 4, -4, 0, 0],
				[4, -4, 4, 0, 0, 0]
			]
		});

		expect(result.width).toBe(3);
		expect(result.height).toBe(3);
		expect(result.canonicalPointCount).toBe(9);
		expect([...result.activeCanonicalIndices]).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
		expect(result.activePointCount).toBe(9);
		expect(Array.from(result.mask)).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1]);
		expect(result.maskChecksum).toMatch(/^[0-9a-f]{8}$/);
	});

	it('keeps canonical cells outside a partial triangle inactive', () => {
		const result = buildStudyAreaMaskFromProjectedTriangles({
			bounds: { x_min: 0, x_max: 4, y_min: 0, y_max: 4, z: 1.5 },
			gridSize: 2,
			coordinateSystem: 'xy_ground',
			triangles: [[0, -4, 2, -4, 0, -2]]
		});

		expect(result.canonicalPointCount).toBe(9);
		expect(result.activeCanonicalIndices).toEqual(new Uint32Array([1, 2, 5]));
		expect(result.activePointCount).toBe(3);
		expect(Array.from(result.mask)).toEqual([0, 1, 1, 0, 0, 1, 0, 0, 0]);
		expect(result.mask.some((active) => !active)).toBe(true);
	});

	it('uses the same descending canonical row ordering as canonicalGridPoints for xy_ground', () => {
		const result = buildStudyAreaMaskFromProjectedTriangles({
			bounds: { x_min: 0, x_max: 4, y_min: 0, y_max: 4, z: 1.5 },
			gridSize: 2,
			coordinateSystem: 'xy_ground',
			triangles: [[0, -2, 2, -2, 0, 0]]
		});

		expect(result.activeCanonicalIndices).toEqual(new Uint32Array([0, 1, 4]));
		expect(Array.from(result.mask)).toEqual([1, 1, 0, 0, 1, 0, 0, 0, 0]);

		const { points } = canonicalGridPointsForActiveIndices({
			bounds: { x_min: 0, x_max: 4, y_min: 0, y_max: 4, z: 1.5 },
			gridSize: 2,
			coordinateSystem: 'xy_ground',
			activeCanonicalIndices: result.activeCanonicalIndices
		});

		expect(Array.from(points)).toEqual([
			0, 1.5, 0,
			0, 1.5, -2,
			2, 1.5, -2
		]);
	});

	it('maps asymmetric xy_ground projected footprints through metadata bounds', () => {
		const result = buildStudyAreaMaskFromProjectedTriangles({
			bounds: { x_min: 0, x_max: 4, y_min: 0, y_max: 4, z: 1.5 },
			gridSize: 2,
			coordinateSystem: 'xy_ground',
			triangles: [[2, 0, 4, 0, 4, -2]]
		});

		expect(result.activeCanonicalIndices).toEqual(new Uint32Array([3, 6, 7]));
		expect(Array.from(result.mask)).toEqual([0, 0, 0, 1, 0, 0, 1, 1, 0]);
	});

	it('uses ascending canonical row ordering for xz_ground', () => {
		const result = buildStudyAreaMaskFromProjectedTriangles({
			bounds: { x_min: 0, x_max: 4, y_min: 0, y_max: 4, z: 1.5 },
			gridSize: 2,
			coordinateSystem: 'xz_ground',
			triangles: [[0, 0, 2, 0, 0, 2]]
		});

		expect(result.activeCanonicalIndices).toEqual(new Uint32Array([0, 1, 3]));
		expect(Array.from(result.mask)).toEqual([1, 1, 0, 1, 0, 0, 0, 0, 0]);
	});

	const inclusiveStepCases: Array<{
		coordinateSystem: 'xy_ground' | 'xz_ground';
		bounds: { x_min: number; x_max: number; y_min: number; y_max: number; z: number };
		triangles: ProjectedTriangle2D[];
	}> = [
		{
			coordinateSystem: 'xy_ground',
			bounds: { x_min: 0, x_max: 3.99999995, y_min: 0, y_max: 3.99999995, z: 1.5 },
			triangles: [
				[0, -3.99999995, 3.99999995, -3.99999995, 0, 0],
				[3.99999995, -3.99999995, 3.99999995, 0, 0, 0]
			]
		},
		{
			coordinateSystem: 'xz_ground',
			bounds: { x_min: 0, x_max: 3.99999995, y_min: 0, y_max: 3.99999995, z: 1.5 },
			triangles: [
				[0, 0, 3.99999995, 0, 0, 3.99999995],
				[3.99999995, 0, 3.99999995, 3.99999995, 0, 3.99999995]
			]
		}
	];

	it.each(inclusiveStepCases)(
		'matches canonical metadata-bounds point counts for %s near inclusive-step edges',
		({ coordinateSystem, bounds, triangles }) => {
			const canonical = canonicalGridPoints({
				bounds,
				gridSize: 2,
				coordinateSystem
			});
			const result = buildStudyAreaMaskFromProjectedTriangles({
				bounds,
				gridSize: 2,
				coordinateSystem,
				triangles
			});

			expect(canonical.numPoints).toBe(4);
			expect(result.width).toBe(2);
			expect(result.height).toBe(2);
			expect(result.canonicalPointCount).toBe(canonical.numPoints);
			expect(result.activeCanonicalIndices).toEqual(new Uint32Array([0, 1, 2, 3]));
			expect(() =>
				canonicalGridPointsForActiveIndices({
					bounds,
					gridSize: 2,
					coordinateSystem,
					activeCanonicalIndices: result.activeCanonicalIndices
				})
			).not.toThrow();
		}
	);
});

describe('canonicalGridPointsForActiveIndices', () => {
	it('projects only the requested canonical indices using canonical ordering', () => {
		const { points, numPoints } = canonicalGridPointsForActiveIndices({
			bounds: { x_min: 0, x_max: 4, y_min: 0, y_max: 4, z: 1.5 },
			gridSize: 2,
			coordinateSystem: 'xy_ground',
			activeCanonicalIndices: new Uint32Array([1, 3, 8])
		});

		expect(numPoints).toBe(3);
		expect(Array.from(points)).toEqual([
			0, 1.5, -2,
			2, 1.5, 0,
			4, 1.5, -4
		]);
	});
});
