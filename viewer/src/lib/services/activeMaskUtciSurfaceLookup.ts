import type { ActiveCellsUtciGridLayout } from '$lib/services/utciGridLayoutTopology';

export type ActiveMaskUtciSurfaceLookup = {
	layout: ActiveCellsUtciGridLayout;
	activeCanonicalIndices: Uint32Array;
	width: number;
	height: number;
	canonicalCellCount: number;
	activePointCount: number;
	byteLength: number;
};

export type ActiveMaskUtciSurfaceCellLookupResult = {
	canonicalIndex: number | null;
	positionIndex: number | null;
	inactiveCell: boolean;
};

const lookupCache = new WeakMap<ActiveCellsUtciGridLayout, ActiveMaskUtciSurfaceLookup>();

function assertSortedActiveCanonicalIndices(layout: ActiveCellsUtciGridLayout): void {
	const { activeCanonicalIndices, canonicalCellCount } = layout;
	for (let index = 0; index < activeCanonicalIndices.length; index += 1) {
		const canonicalIndex = activeCanonicalIndices[index];
		if (canonicalIndex === undefined || canonicalIndex >= canonicalCellCount) {
			throw new RangeError(
				`activeCanonicalIndices contains out-of-range canonical index at active point ${index}.`
			);
		}
		if (index > 0 && canonicalIndex <= activeCanonicalIndices[index - 1]!) {
			throw new Error(
				'activeCanonicalIndices must be sorted in ascending canonical order for sparse active lookup.'
			);
		}
	}
}

export function createActiveMaskUtciSurfaceLookup(
	layout: ActiveCellsUtciGridLayout
): ActiveMaskUtciSurfaceLookup {
	assertSortedActiveCanonicalIndices(layout);
	return {
		layout,
		activeCanonicalIndices: layout.activeCanonicalIndices,
		width: layout.width,
		height: layout.height,
		canonicalCellCount: layout.canonicalCellCount,
		activePointCount: layout.activeCanonicalIndices.length,
		byteLength: layout.activeCanonicalIndices.byteLength
	};
}

export function getCachedActiveMaskUtciSurfaceLookup(
	layout: ActiveCellsUtciGridLayout
): ActiveMaskUtciSurfaceLookup {
	const cached = lookupCache.get(layout);
	if (cached) {
		return cached;
	}
	const lookup = createActiveMaskUtciSurfaceLookup(layout);
	lookupCache.set(layout, lookup);
	return lookup;
}

export function getActiveMaskCanonicalIndexFromSurfaceCell(params: {
	layout: Pick<ActiveCellsUtciGridLayout, 'coordinateSystem' | 'height'>;
	row: number;
	column: number;
}): number {
	const canonicalRow =
		params.layout.coordinateSystem === 'xy_ground'
			? params.layout.height - 1 - params.row
			: params.row;
	return params.column * params.layout.height + canonicalRow;
}

function binarySearchActivePointIndex(
	activeCanonicalIndices: Uint32Array,
	canonicalIndex: number
): number {
	let low = 0;
	let high = activeCanonicalIndices.length - 1;

	while (low <= high) {
		const mid = low + Math.floor((high - low) / 2);
		const candidate = activeCanonicalIndices[mid]!;
		if (candidate === canonicalIndex) {
			return mid;
		}
		if (candidate < canonicalIndex) {
			low = mid + 1;
		} else {
			high = mid - 1;
		}
	}

	return -1;
}

export function resolveActiveMaskUtciSurfaceCellLookup(
	lookup: ActiveMaskUtciSurfaceLookup,
	cell: { row: number; column: number }
): ActiveMaskUtciSurfaceCellLookupResult {
	if (
		cell.row < 0 ||
		cell.row >= lookup.height ||
		cell.column < 0 ||
		cell.column >= lookup.width
	) {
		return {
			canonicalIndex: null,
			positionIndex: null,
			inactiveCell: false
		};
	}

	const canonicalIndex = getActiveMaskCanonicalIndexFromSurfaceCell({
		layout: lookup.layout,
		row: cell.row,
		column: cell.column
	});
	const positionIndex = binarySearchActivePointIndex(
		lookup.activeCanonicalIndices,
		canonicalIndex
	);
	return {
		canonicalIndex,
		positionIndex: positionIndex >= 0 ? positionIndex : null,
		inactiveCell: positionIndex === -1
	};
}
