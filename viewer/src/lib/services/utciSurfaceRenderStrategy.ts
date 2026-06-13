import type {
	WebgpuLargeBufferDeviceLimits,
	WebgpuLargeBufferRequiredLimits
} from '$lib/compute/gpu/webgpuDeviceLimits';
import { instanceIndex } from 'three/tsl';
import {
	INNOVATION_DISTRICT_05M_ACTIVE_MASK_SURFACE_SHAPE,
	estimateActiveMaskUtciInstancedGeometryBytes
} from '$lib/services/activeMaskUtciSurfaceGeometry';
import {
	buildInnovationDistrict05mUtciSurfaceBudgetDecision,
	type ActiveMaskUtciSurfaceBudgetDecision
} from '$lib/services/activeMaskUtciSurfaceBudget';
import { estimateActiveMaskUtciSurfaceStrategies } from '$lib/services/activeMaskUtciSurfaceBudgetMath';
import type { UtciGridLayout, ActiveCellsUtciGridLayout } from '$lib/services/utciGridLayoutTopology';
import { COMPUTE_BUFFER_COLOR_LUT_BYTES } from '$lib/services/computeBufferMetricColorPolicy';
import type { SelectedHourRenderAllocationPreflight } from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';

export type ActiveMaskUtciSurfaceRenderStrategyDecision =
	ActiveMaskUtciSurfaceBudgetDecision;

export type ActiveMaskUtciSurfaceRenderStrategyInputs = {
	activePointCount?: number;
	canonicalCellCount?: number;
	requestedLimits?: WebgpuLargeBufferRequiredLimits;
	deviceLimits?: WebgpuLargeBufferDeviceLimits;
};

export type TslInstanceIndexSupportProof = {
	available: true;
	nodeType: 'uint';
	scope: 'instance';
	vertexBuiltin: 'instance_index';
};

export type UtciSurfaceRenderStrategyEstimate = {
	renderTopology: UtciGridLayout['renderTopology'];
	geometryBytes: number;
	selectedHourUtciStorageBytes: number;
	cellToPointStorageBytes: number;
	activeCanonicalIndexAttributeBytes: number;
	colorLutBytes: number;
	totalBytes: number;
};

export type UtciRenderAllocationPreflightParams = {
	layout: UtciGridLayout;
	utciStorageBytes: number;
	rendererLimits?: {
		maxBufferSize?: number;
		maxStorageBufferBindingSize?: number;
	};
	jsLargestTypedArrayByteLimit?: number;
	renderEstimate?: UtciSurfaceRenderStrategyEstimate;
};

export const DEFAULT_RENDER_JS_LARGEST_TYPED_ARRAY_BYTE_LIMIT = 268_435_456;

function matchesKnownInnovationDistrict05mShape(
	inputs: ActiveMaskUtciSurfaceRenderStrategyInputs
): boolean {
	return (
		inputs.activePointCount ===
			INNOVATION_DISTRICT_05M_ACTIVE_MASK_SURFACE_SHAPE.activePointCount &&
		inputs.canonicalCellCount ===
			INNOVATION_DISTRICT_05M_ACTIVE_MASK_SURFACE_SHAPE.canonicalCellCount
	);
}

export function buildKnownActiveMaskUtciSurfaceRenderStrategyDecision(
	inputs: ActiveMaskUtciSurfaceRenderStrategyInputs
): ActiveMaskUtciSurfaceRenderStrategyDecision | undefined {
	if (!matchesKnownInnovationDistrict05mShape(inputs)) {
		return undefined;
	}

	return buildInnovationDistrict05mUtciSurfaceBudgetDecision({
		requestedLimits: inputs.requestedLimits,
		deviceLimits: inputs.deviceLimits
	});
}

export function assertTslInstanceIndexSupport(): TslInstanceIndexSupportProof {
	const node = instanceIndex as {
		isIndexNode?: boolean;
		nodeType?: string;
		scope?: string;
	};
	if (
		node?.isIndexNode !== true ||
		node.nodeType !== 'uint' ||
		node.scope !== 'instance'
	) {
		throw new Error('Three TSL instanceIndex is unavailable or not an instance uint node.');
	}

	return {
		available: true,
		nodeType: 'uint',
		scope: 'instance',
		vertexBuiltin: 'instance_index'
	};
}

export function estimateComputeBufferUtciSurfaceRenderStrategy(params: {
	layout: UtciGridLayout;
	geometryBytes?: number;
	utciStorageBytes: number;
	cellToPointStorageBytes?: number;
}): UtciSurfaceRenderStrategyEstimate {
	if (params.layout.renderTopology === 'active-cells') {
		const geometryEstimate = estimateActiveMaskUtciInstancedGeometryBytes(params.layout);
		const geometryBytes = params.geometryBytes ?? geometryEstimate.totalBytes;
		return {
			renderTopology: 'active-cells',
			geometryBytes,
			selectedHourUtciStorageBytes: params.utciStorageBytes,
			cellToPointStorageBytes: 0,
			activeCanonicalIndexAttributeBytes:
				geometryEstimate.activeCanonicalIndexAttributeBytes,
			colorLutBytes: COMPUTE_BUFFER_COLOR_LUT_BYTES,
			totalBytes:
				geometryBytes + params.utciStorageBytes + COMPUTE_BUFFER_COLOR_LUT_BYTES
		};
	}

	const cellToPointStorageBytes = params.cellToPointStorageBytes ?? 0;
	const geometryBytes = params.geometryBytes ?? 0;
	return {
		renderTopology: 'dense-grid',
		geometryBytes,
		selectedHourUtciStorageBytes: params.utciStorageBytes,
		cellToPointStorageBytes,
		activeCanonicalIndexAttributeBytes: 0,
		colorLutBytes: COMPUTE_BUFFER_COLOR_LUT_BYTES,
		totalBytes:
			geometryBytes +
			params.utciStorageBytes +
			cellToPointStorageBytes +
			COMPUTE_BUFFER_COLOR_LUT_BYTES
	};
}

function fitsLimit(value: number, limit: number | undefined): boolean {
	return limit === undefined || value <= limit;
}

function getEstimatedDenseRectGeometryBytes(layout: ActiveCellsUtciGridLayout): number {
	const { estimates } = estimateActiveMaskUtciSurfaceStrategies({
		label: 'active-render-preflight',
		activePointCount: layout.numPositions,
		canonicalCellCount: layout.canonicalCellCount,
		canonicalWidth: layout.width,
		canonicalHeight: layout.height
	});
	const denseEstimate = estimates.find(
		(estimate) => estimate.strategy === 'dense-indexed-rect'
	);
	return denseEstimate
		? denseEstimate.vertexBufferBytes + denseEstimate.indexBufferBytes
		: 0;
}

export function buildUtciRenderAllocationPreflight(
	params: UtciRenderAllocationPreflightParams
): SelectedHourRenderAllocationPreflight {
	const renderEstimate =
		params.renderEstimate ??
		estimateComputeBufferUtciSurfaceRenderStrategy({
			layout: params.layout,
			utciStorageBytes: params.utciStorageBytes
		});
	const rendererMaxBufferSize = params.rendererLimits?.maxBufferSize;
	const rendererMaxStorageBufferBindingSize =
		params.rendererLimits?.maxStorageBufferBindingSize;

	if (params.layout.renderTopology === 'active-cells') {
		const geometryEstimate = estimateActiveMaskUtciInstancedGeometryBytes(params.layout);
		const jsLargestTypedArrayByteLimit =
			params.jsLargestTypedArrayByteLimit ??
			DEFAULT_RENDER_JS_LARGEST_TYPED_ARRAY_BYTE_LIMIT;
		const estimatedLargestJsTypedArrayBytes = Math.max(
			geometryEstimate.vertexBufferBytes,
			geometryEstimate.indexBufferBytes,
			renderEstimate.activeCanonicalIndexAttributeBytes,
			renderEstimate.selectedHourUtciStorageBytes
		);
		const failureReasons: string[] = [];
		try {
			assertTslInstanceIndexSupport();
		} catch (error) {
			failureReasons.push(
				`active instanced rendering requires Three TSL instanceIndex support: ${
					error instanceof Error ? error.message : String(error)
				}`
			);
		}
		if (!fitsLimit(estimatedLargestJsTypedArrayBytes, jsLargestTypedArrayByteLimit)) {
			failureReasons.push(
				'largest render JS typed-array allocation exceeds conservative limit'
			);
		}
		if (
			!fitsLimit(
				renderEstimate.activeCanonicalIndexAttributeBytes,
				rendererMaxBufferSize
			)
		) {
			failureReasons.push(
				'active canonical index buffer exceeds renderer maxBufferSize'
			);
		}
		if (!fitsLimit(renderEstimate.selectedHourUtciStorageBytes, rendererMaxBufferSize)) {
			failureReasons.push('selected-hour UTCI storage exceeds renderer maxBufferSize');
		}
		if (
			!fitsLimit(
				renderEstimate.selectedHourUtciStorageBytes,
				rendererMaxStorageBufferBindingSize
			)
		) {
			failureReasons.push(
				'selected-hour UTCI storage exceeds renderer maxStorageBufferBindingSize'
			);
		}

		return {
			status: failureReasons.length === 0 ? 'passed' : 'failed',
			renderTopology: 'active-cells',
			renderCellCount: params.layout.renderCellCount,
			canonicalCellCount: params.layout.canonicalCellCount,
			activePointCount: params.layout.numPositions,
			estimatedRenderGeometryBytes: renderEstimate.geometryBytes,
			estimatedLargestSingleRenderAllocationBytes: Math.max(
				geometryEstimate.vertexBufferBytes,
				geometryEstimate.indexBufferBytes,
				renderEstimate.activeCanonicalIndexAttributeBytes,
				renderEstimate.selectedHourUtciStorageBytes
			),
			estimatedDenseRectGeometryBytes: getEstimatedDenseRectGeometryBytes(params.layout),
			estimatedLargestJsTypedArrayBytes,
			jsLargestTypedArrayByteLimit,
			rendererMaxBufferSize,
			rendererMaxStorageBufferBindingSize,
			activeRenderStrategy: 'active-instanced-quads',
			activeRenderInstanceCount: params.layout.numPositions,
			activeRenderSharedVertexCount: 4,
			activeRenderSharedIndexCount: 6,
			activeCanonicalIndexBufferBytes:
				renderEstimate.activeCanonicalIndexAttributeBytes,
			failureReasons: failureReasons.length > 0 ? failureReasons : undefined,
			forbiddenDenseAllocationProof: {
				noDenseCellToPointStorageAttribute: true,
				noDenseColorBuffer: true,
				noWidthHeightRenderGeometry: true,
				noPerActiveCellDuplicatedVertexBuffer: true,
				noPerActiveCellDuplicatedIndexBuffer: true,
				sharedQuadVertexIndexBuffersConstantSize: true,
				instanceCountEqualsActivePointCount:
					params.layout.activeCanonicalIndices.length === params.layout.numPositions,
				noFullDenseTooltipReverseMapWithoutExplicitApprovalAndByteAccounting: true
			}
		};
	}

	return {
		status: 'passed',
		renderTopology: 'dense-grid',
		renderCellCount: params.layout.renderCellCount,
		canonicalCellCount: params.layout.canonicalCellCount,
		estimatedRenderGeometryBytes: renderEstimate.geometryBytes,
		estimatedLargestSingleRenderAllocationBytes: Math.max(
			renderEstimate.geometryBytes,
			renderEstimate.selectedHourUtciStorageBytes,
			renderEstimate.cellToPointStorageBytes
		),
		rendererMaxBufferSize,
		rendererMaxStorageBufferBindingSize
	};
}
