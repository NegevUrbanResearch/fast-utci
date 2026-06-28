import { describe, expect, it } from 'vitest';
import {
	buildClassifiedStudyAreaMaskFromProjectedTriangles,
	buildStudyAreaMaskFromProjectedTriangles
} from '$lib/compute/core/studyAreaMask';
import {
	canonicalGridPoints,
	canonicalGridPointsForActiveIndices
} from '$lib/compute/core/canonicalGrid';
import {
	parseSurfaceFlags,
	SURFACE_FLAGS,
	type ClassifiedProjectedTriangle2D,
	type ProjectedTriangle2D
} from '$lib/types/analysis';

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

describe('buildClassifiedStudyAreaMaskFromProjectedTriangles', () => {
	it('keeps [0,1,2] active and treats a sample center on the shared triangle edge as inside both sampled-surface and building-footprint rasterization', () => {
		const result = buildClassifiedStudyAreaMaskFromProjectedTriangles({
			bounds: { x_min: 0, x_max: 6, y_min: 0, y_max: 0, z: 1.5 },
			gridSize: 2,
			coordinateSystem: 'xz_ground',
			triangles: [
				{
					flags: SURFACE_FLAGS.ground,
					triangle: [0, 0, 4, 0, 0, 2]
				},
				{
					flags: SURFACE_FLAGS.ground,
					triangle: [4, 0, 4, 2, 0, 2]
				},
				{
					flags: SURFACE_FLAGS.streetSurface,
					triangle: [1, 0, 3, 0, 2, 1]
				},
				{
					flags: SURFACE_FLAGS.buildingFootprint,
					triangle: [3, 0, 5, 0, 4, 1]
				}
			]
		});

		expect(Array.from(result.activeMask.activeCanonicalIndices)).toEqual([0, 1, 2]);
		expect(Array.from(result.surfaceFlagsByActiveCell)).toEqual([
			SURFACE_FLAGS.ground,
			SURFACE_FLAGS.ground | SURFACE_FLAGS.streetSurface,
			SURFACE_FLAGS.ground | SURFACE_FLAGS.buildingFootprint
		]);
	});

	it('keeps building-only cells outside sampled ground or street geometry inactive and never leaves a classified active row without a sampled-surface flag', () => {
		const result = buildClassifiedStudyAreaMaskFromProjectedTriangles({
			bounds: { x_min: 0, x_max: 6, y_min: 0, y_max: 0, z: 1.5 },
			gridSize: 2,
			coordinateSystem: 'xz_ground',
			triangles: [
				{
					flags: SURFACE_FLAGS.ground,
					triangle: [0, 0, 4, 0, 0, 2]
				},
				{
					flags: SURFACE_FLAGS.ground,
					triangle: [4, 0, 4, 2, 0, 2]
				},
				{
					flags: SURFACE_FLAGS.streetSurface,
					triangle: [3, 0, 5, 0, 4, 1]
				},
				{
					flags: SURFACE_FLAGS.buildingFootprint,
					triangle: [3, 0, 5, 0, 4, 1]
				},
				{
					flags: SURFACE_FLAGS.buildingFootprint,
					triangle: [5, 0, 7, 0, 6, 1]
				}
			]
		});

		expect(Array.from(result.activeMask.activeCanonicalIndices)).toEqual([0, 1, 2]);
		expect(Array.from(result.surfaceFlagsByActiveCell)).toEqual([
			SURFACE_FLAGS.ground,
			SURFACE_FLAGS.ground,
			SURFACE_FLAGS.ground | SURFACE_FLAGS.streetSurface | SURFACE_FLAGS.buildingFootprint
		]);
		expect(
			Array.from(result.surfaceFlagsByActiveCell).every((flags) => {
				const sampledSurfaceFlags = flags & (SURFACE_FLAGS.ground | SURFACE_FLAGS.streetSurface);
				return sampledSurfaceFlags !== 0;
			})
		).toBe(true);
	});

	it('preserves street-family plus building-footprint overlap as multi-hot flags', () => {
		const result = buildClassifiedStudyAreaMaskFromProjectedTriangles({
			bounds: { x_min: 0, x_max: 6, y_min: 0, y_max: 0, z: 1.5 },
			gridSize: 2,
			coordinateSystem: 'xz_ground',
			triangles: [
				{
					flags: SURFACE_FLAGS.ground,
					triangle: [0, 0, 4, 0, 0, 2]
				},
				{
					flags: SURFACE_FLAGS.ground,
					triangle: [4, 0, 4, 2, 0, 2]
				},
				{
					flags: SURFACE_FLAGS.streetSurface,
					triangle: [1, 0, 3, 0, 2, 1]
				},
				{
					flags: SURFACE_FLAGS.streetSurface,
					triangle: [3, 0, 5, 0, 4, 1]
				},
				{
					flags: SURFACE_FLAGS.buildingFootprint,
					triangle: [3, 0, 5, 0, 4, 1]
				}
			]
		});

		expect(Array.from(result.activeMask.activeCanonicalIndices)).toEqual([0, 1, 2]);
		expect(Array.from(result.surfaceFlagsByActiveCell)).toEqual([
			SURFACE_FLAGS.ground,
			SURFACE_FLAGS.ground | SURFACE_FLAGS.streetSurface,
			SURFACE_FLAGS.ground | SURFACE_FLAGS.streetSurface | SURFACE_FLAGS.buildingFootprint
		]);
		const includeInPublicRealmStats = Array.from(result.surfaceFlagsByActiveCell).map((flags) => {
			return (
				(flags & SURFACE_FLAGS.streetSurface) !== 0 &&
				(flags & SURFACE_FLAGS.buildingFootprint) === 0
			);
		});
		expect(includeInPublicRealmStats).toEqual([false, true, false]);
	});

	it('rejects arbitrary surface flag numbers before they enter classified rasterization', () => {
		expect(() => parseSurfaceFlags(1 << 7)).toThrowError(/unknown bits/i);

		const triangles: ClassifiedProjectedTriangle2D[] = [
			{
				flags: SURFACE_FLAGS.ground,
				triangle: [0, 0, 2, 0, 0, 2]
			}
		];
		expect(
			buildClassifiedStudyAreaMaskFromProjectedTriangles({
				bounds: { x_min: 0, x_max: 0, y_min: 0, y_max: 0, z: 1.5 },
				gridSize: 2,
				coordinateSystem: 'xz_ground',
				triangles
			}).surfaceFlagsByActiveCell
		).toEqual(new Uint8Array([SURFACE_FLAGS.ground]));
	});
});
