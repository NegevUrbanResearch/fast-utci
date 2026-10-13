import { describe, it, expect } from 'vitest';
import { canonicalGridChecksum, canonicalGridPoints } from '$lib/parity/gridCanonical';
import { canonicalGridPointsForActiveIndices, type CanonicalGridParams } from '$lib/compute/core/canonicalGrid';

// Independent legacy traversal: accumulate numbers, then round once to Float32.
function legacyPoints(params: CanonicalGridParams): Float32Array {
	const { bounds, gridSize, coordinateSystem, originOffset } = params;
	const values: number[] = [];
	const height = params.zHeight ?? bounds.z ?? 0;
	for (let x = bounds.x_min; x <= bounds.x_max + 1e-9; x += gridSize) {
		for (let ground = bounds.y_min; ground <= bounds.y_max + 1e-9; ground += gridSize) {
			values.push(x + (originOffset?.x ?? 0), height + (originOffset?.y ?? 0),
				(coordinateSystem === 'xy_ground' ? -ground : ground) + (originOffset?.z ?? 0));
		}
	}
	return new Float32Array(values);
}

function bytes(points: Float32Array): Uint8Array {
	return new Uint8Array(points.buffer, points.byteOffset, points.byteLength);
}

describe('canonical grid ordering', () => {
	for (const coordinateSystem of ['xy_ground', 'xz_ground'] as const) {
		it(`preserves legacy bytes with fractional steps, height, and offsets (${coordinateSystem})`, () => {
			const params: CanonicalGridParams = {
				bounds: { x_min: -0.3, x_max: 0.6, y_min: -0.2, y_max: 0.7, z: 9 },
				gridSize: 0.1, coordinateSystem, zHeight: 1.123456789,
				originOffset: { x: 123.456789, y: -0.5, z: -17.7654321 }
			};
			const result = canonicalGridPoints(params);
			expect(bytes(result.points)).toEqual(bytes(legacyPoints(params)));
			expect(result.numPoints).toBe(100);
			const indices = new Uint32Array([99, 0, 17, 17]);
			const active = canonicalGridPointsForActiveIndices({ ...params, activeCanonicalIndices: indices });
			const expected = new Float32Array(Array.from(indices).flatMap(index => Array.from(result.points.slice(index * 3, index * 3 + 3))));
			expect(bytes(active.points)).toEqual(bytes(expected));
		});
	}

	it('preserves signed zero in a degenerate grid', () => {
		const { points, numPoints } = canonicalGridPoints({
			bounds: { x_min: -0, x_max: 0, y_min: 0, y_max: 0, z: -0 },
			gridSize: 1, coordinateSystem: 'xy_ground', originOffset: { x: -0, y: -0, z: -0 }
		});
		expect(numPoints).toBe(1);
		expect(Array.from(new Uint32Array(points.buffer))).toEqual([0x80000000, 0x80000000, 0x80000000]);
	});

	for (const coordinateSystem of ['xy_ground', 'xz_ground'] as const) {
		for (const bounds of [
			{ x_min: 2, x_max: 1, y_min: 0, y_max: 1 },
			{ x_min: 0, x_max: 1, y_min: 2, y_max: 1 }
		]) {
			it(`returns an empty grid for inverted bounds (${coordinateSystem}, ${JSON.stringify(bounds)})`, () => {
				const params = { bounds, gridSize: 1, coordinateSystem };
				expect(canonicalGridPoints(params)).toEqual({ points: new Float32Array(), numPoints: 0 });
				expect(canonicalGridChecksum(params)).toBe('811c9dc5');
			});
		}
	}
	it('produces stable checksum for fixed bounds/grid', () => {
		const checksum = canonicalGridChecksum({
			bounds: { x_min: 0, x_max: 4, y_min: 0, y_max: 4, z: 0.9 },
			gridSize: 2,
			coordinateSystem: 'xy_ground'
		});
		expect(checksum).toBe('30bf93b7');
	});

	it('uses deterministic x-major then z-major ordering', () => {
		const { points, numPoints } = canonicalGridPoints({
			bounds: { x_min: 0, x_max: 2, y_min: 0, y_max: 2, z: 1.25 },
			gridSize: 1,
			coordinateSystem: 'xy_ground'
		});

		expect(numPoints).toBe(9);
		expect(Array.from(points.slice(0, 6))).toEqual([0, 1.25, 0, 0, 1.25, -1]);
		expect(Array.from(points.slice(-6))).toEqual([2, 1.25, -1, 2, 1.25, -2]);
	});
});
