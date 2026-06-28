import { afterEach, describe, expect, it, vi } from 'vitest';
import {
	assertTslInstanceIndexSupport,
	buildUtciRenderAllocationPreflight
} from '$lib/services/utciSurfaceRenderStrategy';
import {
	createActiveInstancedUtciSurfaceProof,
	MAX_ACTIVE_SURFACE_PROOF_CANONICAL_CELLS
} from '$lib/services/activeMaskUtciSurfaceProof';
import type { ActiveCellsUtciGridLayout } from '$lib/services/utciGridLayoutTopology';

function createActiveLayout(params: {
	width: number;
	height: number;
	gridSize?: number;
	coordinateSystem: 'xy_ground' | 'xz_ground';
	activeCanonicalIndices: number[];
	activeMaskSignature?: string;
}): ActiveCellsUtciGridLayout {
	const gridSize = params.gridSize ?? 1;
	return {
		renderTopology: 'active-cells',
		width: params.width,
		height: params.height,
		gridSize,
		coordinateSystem: params.coordinateSystem,
		numPositions: params.activeCanonicalIndices.length,
		minX: 0,
		minZ: 0,
		minY: 0,
		maxY: 0,
		centerX: ((params.width - 1) * gridSize) / 2,
		centerZ: ((params.height - 1) * gridSize) / 2,
		baseY: 0,
		renderCellCount: params.activeCanonicalIndices.length,
		canonicalCellCount: params.width * params.height,
		activeCanonicalIndices: new Uint32Array(params.activeCanonicalIndices),
		activeMaskSignature: params.activeMaskSignature ?? 'active-proof-mask'
	};
}

describe('UTCI surface render strategy', () => {
	afterEach(() => {
		vi.doUnmock('three/tsl');
		vi.resetModules();
	});

	it('proves local TSL exposes vertex-stage instanceIndex for active instancing', () => {
		expect(assertTslInstanceIndexSupport()).toEqual({
			available: true,
			nodeType: 'uint',
			scope: 'instance',
			vertexBuiltin: 'instance_index'
		});
	});

	it('proves active instanced UTCI lookup and placement use instanceIndex over compact active cells', () => {
		const layout = createActiveLayout({
			width: 3,
			height: 2,
			gridSize: 2,
			coordinateSystem: 'xz_ground',
			activeCanonicalIndices: [0, 3, 5]
		});

		const proof = createActiveInstancedUtciSurfaceProof({
			layout,
			values: new Float32Array([11, 22, 33])
		});

		expect(proof.pointIndexSource).toBe('instanceIndex');
		expect(proof.activeCanonicalIndexSource).toBe('activeCanonicalIndices[instanceIndex]');
		expect(proof.instances).toEqual([
			{
				instanceIndex: 0,
				pointIndex: 0,
				canonicalIndex: 0,
				row: 0,
				column: 0,
				center: { x: -2, z: -1 },
				value: 11
			},
			{
				instanceIndex: 1,
				pointIndex: 1,
				canonicalIndex: 3,
				row: 1,
				column: 1,
				center: { x: 0, z: 1 },
				value: 22
			},
			{
				instanceIndex: 2,
				pointIndex: 2,
				canonicalIndex: 5,
				row: 1,
				column: 2,
				center: { x: 2, z: 1 },
				value: 33
			}
		]);
		expect(proof.inactiveCanonicalIndices).toEqual([1, 2, 4]);
		expect(proof.instanceCount).toBe(3);
		expect(proof.canonicalCellCount).toBe(6);
	});

	it('rejects active instanced proof scans for large canonical layouts', () => {
		const layout = createActiveLayout({
			width: MAX_ACTIVE_SURFACE_PROOF_CANONICAL_CELLS + 1,
			height: 1,
			coordinateSystem: 'xz_ground',
			activeCanonicalIndices: [0]
		});

		expect(() => createActiveInstancedUtciSurfaceProof({ layout })).toThrow(
			'Active UTCI surface proof only supports small layouts'
		);
	});

	it('builds active render preflight from the shared strategy estimate without dense allocations', () => {
		const layout = createActiveLayout({
			width: 3,
			height: 2,
			gridSize: 2,
			coordinateSystem: 'xz_ground',
			activeCanonicalIndices: [0, 3, 5]
		});

		const preflight = buildUtciRenderAllocationPreflight({
			layout,
			utciStorageBytes: 12,
			rendererLimits: {
				maxBufferSize: 1024,
				maxStorageBufferBindingSize: 1024
			}
		});

		expect(preflight).toMatchObject({
			status: 'passed',
			renderTopology: 'active-cells',
			renderCellCount: 3,
			canonicalCellCount: 6,
			activePointCount: 3,
			estimatedRenderGeometryBytes: 84,
			estimatedLargestSingleRenderAllocationBytes: 48,
			activeRenderStrategy: 'active-instanced-quads',
			activeRenderInstanceCount: 3,
			activeRenderSharedVertexCount: 4,
			activeRenderSharedIndexCount: 6,
			activeCanonicalIndexBufferBytes: 12,
			forbiddenDenseAllocationProof: {
				noDenseCellToPointStorageAttribute: true,
				noDenseColorBuffer: true,
				noWidthHeightRenderGeometry: true,
				noPerActiveCellDuplicatedVertexBuffer: true,
				noPerActiveCellDuplicatedIndexBuffer: true,
				sharedQuadVertexIndexBuffersConstantSize: true,
				instanceCountEqualsActivePointCount: true,
				noFullDenseTooltipReverseMapWithoutExplicitApprovalAndByteAccounting: true
			}
		});
		expect(preflight.estimatedDenseRectGeometryBytes).toBeGreaterThan(
			preflight.estimatedRenderGeometryBytes
		);
	});

	it('fails active preflight before allocation when the active strategy exceeds renderer limits', () => {
		const layout = createActiveLayout({
			width: 3,
			height: 2,
			coordinateSystem: 'xz_ground',
			activeCanonicalIndices: [0, 3, 5]
		});

		const preflight = buildUtciRenderAllocationPreflight({
			layout,
			utciStorageBytes: 12,
			rendererLimits: {
				maxBufferSize: 11,
				maxStorageBufferBindingSize: 11
			}
		});

		expect(preflight.status).toBe('failed');
		expect(preflight.failureReasons).toEqual([
			'active canonical index buffer exceeds renderer maxBufferSize',
			'selected-hour UTCI storage exceeds renderer maxBufferSize',
			'selected-hour UTCI storage exceeds renderer maxStorageBufferBindingSize'
		]);
		expect(preflight.activeRenderStrategy).toBe('active-instanced-quads');
	});

	it('fails active render preflight before allocation when the shared instance-index proof fails', async () => {
		vi.resetModules();
		vi.doMock('three/tsl', () => ({
			instanceIndex: {
				isIndexNode: false,
				nodeType: 'float',
				scope: 'vertex'
			}
		}));
		const { buildUtciRenderAllocationPreflight: buildPreflightWithMockedProof } =
			await import('$lib/services/utciSurfaceRenderStrategy');
		const layout = createActiveLayout({
			width: 3,
			height: 2,
			coordinateSystem: 'xz_ground',
			activeCanonicalIndices: [0, 3, 5]
		});

		const preflight = buildPreflightWithMockedProof({
			layout,
			utciStorageBytes: 12,
			rendererLimits: {
				maxBufferSize: 1024,
				maxStorageBufferBindingSize: 1024
			}
		});

		expect(preflight.status).toBe('failed');
		expect(preflight.failureReasons).toContain(
			'active instanced rendering requires Three TSL instanceIndex support: Three TSL instanceIndex is unavailable or not an instance uint node.'
		);
		expect(preflight.activeRenderStrategy).toBe('active-instanced-quads');
	});

	it('fails active render preflight when the largest JS typed-array allocation exceeds the default limit', () => {
		const layout = createActiveLayout({
			width: 3,
			height: 2,
			coordinateSystem: 'xz_ground',
			activeCanonicalIndices: [0, 3, 5]
		});

		const preflight = buildUtciRenderAllocationPreflight({
			layout,
			utciStorageBytes: 268_435_457,
			renderEstimate: {
				renderTopology: 'active-cells',
				geometryBytes: 84,
				selectedHourUtciStorageBytes: 268_435_457,
				cellToPointStorageBytes: 0,
				activeCanonicalIndexAttributeBytes: 12,
				colorLutBytes: 4096,
				totalBytes: 268_439_637
			}
		});

		expect(preflight.status).toBe('failed');
		expect(preflight.jsLargestTypedArrayByteLimit).toBe(268_435_456);
		expect(preflight.estimatedLargestJsTypedArrayBytes).toBe(268_435_457);
		expect(preflight.failureReasons).toContain(
			'largest render JS typed-array allocation exceeds conservative limit'
		);
	});
});
