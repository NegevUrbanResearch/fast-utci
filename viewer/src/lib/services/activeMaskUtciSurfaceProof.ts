import { getActiveMaskUtciCanonicalCellCenter } from '$lib/services/activeMaskUtciSurfaceGeometry';
import {
	getActiveMaskGridCell,
	type ActiveCellsUtciGridLayout
} from '$lib/services/utciGridLayoutTopology';
import { assertTslInstanceIndexSupport } from '$lib/services/utciSurfaceRenderStrategy';

export const MAX_ACTIVE_SURFACE_PROOF_CANONICAL_CELLS = 10_000;

export type ActiveInstancedUtciSurfaceProofInstance = {
	instanceIndex: number;
	pointIndex: number;
	canonicalIndex: number;
	row: number;
	column: number;
	center: { x: number; z: number };
	value?: number;
};

export type ActiveInstancedUtciSurfaceProof = {
	pointIndexSource: 'instanceIndex';
	activeCanonicalIndexSource: 'activeCanonicalIndices[instanceIndex]';
	instanceCount: number;
	canonicalCellCount: number;
	instances: ActiveInstancedUtciSurfaceProofInstance[];
	inactiveCanonicalIndices: number[];
};

export function createActiveInstancedUtciSurfaceProof(params: {
	layout: ActiveCellsUtciGridLayout;
	values?: Float32Array;
}): ActiveInstancedUtciSurfaceProof {
	assertTslInstanceIndexSupport();
	if (params.layout.canonicalCellCount > MAX_ACTIVE_SURFACE_PROOF_CANONICAL_CELLS) {
		throw new Error(
			`Active UTCI surface proof only supports small layouts up to ${MAX_ACTIVE_SURFACE_PROOF_CANONICAL_CELLS} canonical cells.`
		);
	}
	if (params.values && params.values.length !== params.layout.numPositions) {
		throw new Error(
			`Active UTCI proof values length ${params.values.length} does not match active point count ${params.layout.numPositions}.`
		);
	}

	const activeCanonicalSet = new Set<number>();
	const instances = Array.from(
		params.layout.activeCanonicalIndices,
		(canonicalIndex, instanceIndexValue): ActiveInstancedUtciSurfaceProofInstance => {
			activeCanonicalSet.add(canonicalIndex);
			const { row, col } = getActiveMaskGridCell({
				canonicalIndex,
				width: params.layout.width,
				height: params.layout.height,
				coordinateSystem: params.layout.coordinateSystem
			});
			return {
				instanceIndex: instanceIndexValue,
				pointIndex: instanceIndexValue,
				canonicalIndex,
				row,
				column: col,
				center: getActiveMaskUtciCanonicalCellCenter({
					layout: params.layout,
					canonicalIndex
				}),
				value: params.values?.[instanceIndexValue]
			};
		}
	);
	const inactiveCanonicalIndices: number[] = [];
	for (
		let canonicalIndex = 0;
		canonicalIndex < params.layout.canonicalCellCount;
		canonicalIndex += 1
	) {
		if (!activeCanonicalSet.has(canonicalIndex)) {
			inactiveCanonicalIndices.push(canonicalIndex);
		}
	}

	return {
		pointIndexSource: 'instanceIndex',
		activeCanonicalIndexSource: 'activeCanonicalIndices[instanceIndex]',
		instanceCount: params.layout.activeCanonicalIndices.length,
		canonicalCellCount: params.layout.canonicalCellCount,
		instances,
		inactiveCanonicalIndices
	};
}
