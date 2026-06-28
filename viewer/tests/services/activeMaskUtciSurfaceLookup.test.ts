import { describe, expect, it } from 'vitest';
import {
	createActiveMaskUtciSurfaceLookup,
	getActiveMaskCanonicalIndexFromSurfaceCell,
	resolveActiveMaskUtciSurfaceCellLookup
} from '$lib/services/activeMaskUtciSurfaceLookup';
import type { ActiveCellsUtciGridLayout } from '$lib/services/utciGridLayoutTopology';

function createActiveLayout(params: {
	width: number;
	height: number;
	coordinateSystem: 'xy_ground' | 'xz_ground';
	activeCanonicalIndices: number[];
}): ActiveCellsUtciGridLayout {
	return {
		renderTopology: 'active-cells',
		width: params.width,
		height: params.height,
		gridSize: 1,
		coordinateSystem: params.coordinateSystem,
		numPositions: params.activeCanonicalIndices.length,
		minX: 0,
		minZ: 0,
		minY: 0,
		maxY: 0,
		centerX: (params.width - 1) / 2,
		centerZ: (params.height - 1) / 2,
		baseY: 0,
		renderCellCount: params.activeCanonicalIndices.length,
		canonicalCellCount: params.width * params.height,
		activeCanonicalIndices: new Uint32Array(params.activeCanonicalIndices),
		activeMaskSignature: 'active-mask-lookup-test'
	};
}

describe('active mask UTCI sparse surface lookup', () => {
	it('resolves active xz cells to active point indices and returns no-data for inactive cells', () => {
		const layout = createActiveLayout({
			width: 3,
			height: 2,
			coordinateSystem: 'xz_ground',
			activeCanonicalIndices: [0, 3, 5]
		});
		const lookup = createActiveMaskUtciSurfaceLookup(layout);

		expect(resolveActiveMaskUtciSurfaceCellLookup(lookup, { row: 0, column: 0 })).toMatchObject({
			canonicalIndex: 0,
			positionIndex: 0,
			inactiveCell: false
		});
		expect(resolveActiveMaskUtciSurfaceCellLookup(lookup, { row: 1, column: 1 })).toMatchObject({
			canonicalIndex: 3,
			positionIndex: 1,
			inactiveCell: false
		});
		expect(resolveActiveMaskUtciSurfaceCellLookup(lookup, { row: 1, column: 2 })).toMatchObject({
			canonicalIndex: 5,
			positionIndex: 2,
			inactiveCell: false
		});
		expect(resolveActiveMaskUtciSurfaceCellLookup(lookup, { row: 0, column: 1 })).toMatchObject({
			canonicalIndex: 2,
			positionIndex: null,
			inactiveCell: true
		});
	});

	it('converts xy surface rows back to canonical indices with the canonical row flip', () => {
		const layout = createActiveLayout({
			width: 2,
			height: 3,
			coordinateSystem: 'xy_ground',
			activeCanonicalIndices: [0, 1, 4, 5]
		});
		const lookup = createActiveMaskUtciSurfaceLookup(layout);

		expect(getActiveMaskCanonicalIndexFromSurfaceCell({ layout, row: 2, column: 0 })).toBe(0);
		expect(getActiveMaskCanonicalIndexFromSurfaceCell({ layout, row: 1, column: 0 })).toBe(1);
		expect(getActiveMaskCanonicalIndexFromSurfaceCell({ layout, row: 1, column: 1 })).toBe(4);
		expect(getActiveMaskCanonicalIndexFromSurfaceCell({ layout, row: 0, column: 1 })).toBe(5);
		expect(resolveActiveMaskUtciSurfaceCellLookup(lookup, { row: 1, column: 1 })).toMatchObject({
			canonicalIndex: 4,
			positionIndex: 2,
			inactiveCell: false
		});
		expect(resolveActiveMaskUtciSurfaceCellLookup(lookup, { row: 0, column: 0 })).toMatchObject({
			canonicalIndex: 2,
			positionIndex: null,
			inactiveCell: true
		});
	});

	it('keeps lookup storage compact and reuses activeCanonicalIndices as the sorted source of truth', () => {
		const layout = createActiveLayout({
			width: 1_000,
			height: 1_000,
			coordinateSystem: 'xz_ground',
			activeCanonicalIndices: [7, 31, 999_999]
		});

		const lookup = createActiveMaskUtciSurfaceLookup(layout);

		expect(lookup.activeCanonicalIndices).toBe(layout.activeCanonicalIndices);
		expect(lookup.byteLength).toBe(12);
		expect(lookup.activePointCount).toBe(3);
		expect(lookup.canonicalCellCount).toBe(1_000_000);
		expect(lookup).not.toHaveProperty('cellToPointIndex');
		expect(lookup).not.toHaveProperty('indexToRow');
		expect(lookup).not.toHaveProperty('indexToColumn');
	});

	it('documents and enforces the ascending activeCanonicalIndices invariant used by binary search', () => {
		const layout = createActiveLayout({
			width: 3,
			height: 2,
			coordinateSystem: 'xz_ground',
			activeCanonicalIndices: [3, 0, 5]
		});

		expect(() => createActiveMaskUtciSurfaceLookup(layout)).toThrow(
			'activeCanonicalIndices must be sorted in ascending canonical order'
		);
	});

	it('returns the active point index from the render source rather than the canonical index', () => {
		const layout = createActiveLayout({
			width: 101,
			height: 2,
			coordinateSystem: 'xz_ground',
			activeCanonicalIndices: [2, 100, 200]
		});
		const lookup = createActiveMaskUtciSurfaceLookup(layout);

		expect(resolveActiveMaskUtciSurfaceCellLookup(lookup, { row: 0, column: 50 })).toMatchObject({
			canonicalIndex: 100,
			positionIndex: 1,
			inactiveCell: false
		});
	});
});
