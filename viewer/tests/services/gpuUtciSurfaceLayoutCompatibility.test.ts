import { describe, expect, it } from 'vitest';
import {
	evaluateComputeBufferUtciSurfaceLayoutCompatibility,
	evaluateUtciGridLayoutsPointCompatibility,
	getComputeBufferSurfaceVertexCount
} from '$lib/services/gpuUtciSurfaceLayoutCompatibility';
import type {
	ActiveCellsUtciGridLayout,
	DenseUtciGridLayout
} from '$lib/services/utciGridLayoutTopology';

function createDenseLayout(): DenseUtciGridLayout {
	return {
		renderTopology: 'dense-grid',
		width: 2,
		height: 2,
		gridSize: 1,
		coordinateSystem: 'xz_ground',
		numPositions: 4,
		minX: 0,
		minZ: 0,
		minY: 0,
		maxY: 0,
		centerX: 0.5,
		centerZ: 0.5,
		baseY: 0,
		renderCellCount: 4,
		canonicalCellCount: 4,
		indexToRow: new Uint32Array([0, 0, 1, 1]),
		indexToColumn: new Uint32Array([0, 1, 0, 1]),
		indexToTexel: new Uint32Array([0, 1, 2, 3]),
		cellToPointIndex: new Int32Array([0, 1, 2, 3]),
		colorBuffer: new Uint8Array(16)
	};
}

function createActiveLayout(params?: {
	activeCanonicalIndices?: number[];
	activeMaskSignature?: string;
}): ActiveCellsUtciGridLayout {
	const activeCanonicalIndices = params?.activeCanonicalIndices ?? [0, 3];
	return {
		renderTopology: 'active-cells',
		width: 2,
		height: 2,
		gridSize: 1,
		coordinateSystem: 'xz_ground',
		numPositions: activeCanonicalIndices.length,
		minX: 0,
		minZ: 0,
		minY: 0,
		maxY: 0,
		centerX: 0.5,
		centerZ: 0.5,
		baseY: 0,
		renderCellCount: activeCanonicalIndices.length,
		canonicalCellCount: 4,
		activeCanonicalIndices: new Uint32Array(activeCanonicalIndices),
		activeMaskSignature: params?.activeMaskSignature ?? 'active-mask-a'
	};
}

describe('compute-buffer UTCI surface layout compatibility', () => {
	it('keeps dense-to-dense layouts compatible by shared-grid topology and mapping', () => {
		const layout = createDenseLayout();

		expect(
			evaluateComputeBufferUtciSurfaceLayoutCompatibility({
				state: {
					source: 'compute-buffer-selected-hour',
					renderTopology: 'dense-grid',
					width: layout.width,
					height: layout.height,
					gridSize: layout.gridSize,
					vertexCount: getComputeBufferSurfaceVertexCount(layout),
					storageCount: layout.numPositions
				},
				previousLayout: layout,
				nextLayout: layout,
				allowExpensiveMappingComparison: false
			})
		).toMatchObject({
			compatible: true,
			vertexCountMatch: true,
			storageCountMatch: true,
			pointCompatibility: {
				compatible: true,
				performedExpensiveMappingComparison: false
			}
		});
	});

	it('keeps active-to-active layouts compatible by topology, signature, and layout without active-index scans', () => {
		const previousLayout = createActiveLayout({
			activeCanonicalIndices: [0, 3],
			activeMaskSignature: 'same-mask'
		});
		const nextLayout = createActiveLayout({
			activeCanonicalIndices: [1, 2],
			activeMaskSignature: 'same-mask'
		});

		expect(
			evaluateUtciGridLayoutsPointCompatibility(previousLayout, nextLayout, {
				allowExpensiveMappingComparison: false
			})
		).toEqual({
			compatible: true,
			cellToPointMappingMatch: true,
			requiredExpensiveMappingComparison: false,
			performedExpensiveMappingComparison: false
		});
		expect(
			evaluateComputeBufferUtciSurfaceLayoutCompatibility({
				state: {
					source: 'compute-buffer-selected-hour',
					renderTopology: 'active-cells',
					activeMaskSignature: 'same-mask',
					width: previousLayout.width,
					height: previousLayout.height,
					gridSize: previousLayout.gridSize,
					vertexCount: 4,
					storageCount: previousLayout.numPositions
				},
				previousLayout,
				nextLayout,
				allowExpensiveMappingComparison: false
			})
		).toMatchObject({
			compatible: true,
			vertexCountMatch: true,
			storageCountMatch: true,
			pointCompatibility: {
				compatible: true,
				performedExpensiveMappingComparison: false
			}
		});
	});

	it('rejects active-to-dense topology reuse', () => {
		const activeLayout = createActiveLayout();
		const denseLayout = createDenseLayout();

		expect(
			evaluateComputeBufferUtciSurfaceLayoutCompatibility({
				state: {
					source: 'compute-buffer-selected-hour',
					renderTopology: 'active-cells',
					activeMaskSignature: activeLayout.activeMaskSignature,
					width: activeLayout.width,
					height: activeLayout.height,
					gridSize: activeLayout.gridSize,
					vertexCount: 4,
					storageCount: activeLayout.numPositions
				},
				previousLayout: activeLayout,
				nextLayout: denseLayout
			})
		).toMatchObject({
			compatible: false,
			pointCompatibility: {
				compatible: false,
				performedExpensiveMappingComparison: false
			}
		});
	});

	it('rejects active-to-active reuse when the active-mask signature changes', () => {
		const previousLayout = createActiveLayout({
			activeCanonicalIndices: [0, 3],
			activeMaskSignature: 'mask-a'
		});
		const nextLayout = createActiveLayout({
			activeCanonicalIndices: [0, 3],
			activeMaskSignature: 'mask-b'
		});

		expect(
			evaluateComputeBufferUtciSurfaceLayoutCompatibility({
				state: {
					source: 'compute-buffer-selected-hour',
					renderTopology: 'active-cells',
					activeMaskSignature: 'mask-a',
					width: previousLayout.width,
					height: previousLayout.height,
					gridSize: previousLayout.gridSize,
					vertexCount: 4,
					storageCount: previousLayout.numPositions
				},
				previousLayout,
				nextLayout,
				allowExpensiveMappingComparison: false
			})
		).toMatchObject({
			compatible: false,
			pointCompatibility: {
				compatible: false,
				requiredExpensiveMappingComparison: false,
				performedExpensiveMappingComparison: false
			}
		});
	});
});
