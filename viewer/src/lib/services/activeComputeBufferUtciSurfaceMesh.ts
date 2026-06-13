import * as THREE from 'three';
import { MeshBasicNodeMaterial, StorageBufferAttribute } from 'three/webgpu';
import {
	attribute,
	clamp,
	float,
	instanceIndex,
	positionLocal,
	storage,
	texture,
	uint,
	uniform,
	vec2,
	vec3
} from 'three/tsl';
import type { F32MetricType } from '$lib/compute/on-demand/onDemandOutputFormat';
import type { SelectedHourRenderSurfaceMeshTrace } from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';
import { createActiveMaskUtciInstancedSurfaceGeometry } from '$lib/services/activeMaskUtciSurfaceGeometry';
import {
	COMPUTE_BUFFER_COLOR_LUT_SIZE,
	createComputeBufferMetricColorLutTexture,
	resolveComputeBufferMetricColorPolicy
} from '$lib/services/computeBufferMetricColorPolicy';
import { GPU_NATIVE_SURFACE_STATE_KEY } from '$lib/services/gpuUtciSurfaceLayoutCompatibility';
import {
	assertTslInstanceIndexSupport,
	estimateComputeBufferUtciSurfaceRenderStrategy,
	type UtciSurfaceRenderStrategyEstimate
} from '$lib/services/utciSurfaceRenderStrategy';
import type { ActiveCellsUtciGridLayout } from '$lib/services/utciGridLayoutTopology';

const DEFAULT_SURFACE_OPACITY = 0.9;

export interface ActiveComputeBufferUtciSurfaceState {
	source: 'compute-buffer-selected-hour';
	renderTopology: 'active-cells';
	activeMaskSignature: string;
	utciStorageAttribute: StorageBufferAttribute;
	activeCanonicalIndexAttribute: THREE.InstancedBufferAttribute;
	width: number;
	height: number;
	gridSize: number;
	vertexCount: number;
	utciRange: { min: number; max: number };
	valueRange: { min: number; max: number };
	metricType: F32MetricType;
	minUniform: ReturnType<typeof uniform>;
	maxUniform: ReturnType<typeof uniform>;
	colorLutTexture: THREE.DataTexture;
	colorLutMetricType: F32MetricType;
	renderEstimate: UtciSurfaceRenderStrategyEstimate;
}

export interface ActiveComputeBufferUtciSurfaceMeshOptions {
	layout: ActiveCellsUtciGridLayout;
	utciBuffer: GPUBuffer;
	utciRange: { min: number; max: number };
	metricType?: F32MetricType;
	valueRange?: { min: number; max: number };
	renderEstimate?: UtciSurfaceRenderStrategyEstimate;
	trace?: SelectedHourRenderSurfaceMeshTrace;
	now?: () => number;
}

function addCreateSurfaceTraceTiming(
	trace: SelectedHourRenderSurfaceMeshTrace | undefined,
	key: Exclude<
		keyof SelectedHourRenderSurfaceMeshTrace,
		'action' | 'totalMs' | 'recreateDecision'
	>,
	value: number
): void {
	if (!trace) return;
	trace[key] = (trace[key] ?? 0) + value;
}

function getGeometryGpuAttributeBytes(geometry: THREE.BufferGeometry): number {
	let total = 0;
	for (const attribute of Object.values(geometry.attributes)) {
		total += (attribute as THREE.BufferAttribute).array.byteLength;
	}
	return total + (geometry.index?.array.byteLength ?? 0);
}

function disabledActiveInstancedUtciSurfaceRaycast(): void {
	return;
}

function createActiveUtciRenderNodes(
	layout: ActiveCellsUtciGridLayout,
	utciStorageAttribute: StorageBufferAttribute,
	minUniform: ReturnType<typeof uniform>,
	maxUniform: ReturnType<typeof uniform>,
	colorLutTexture: THREE.DataTexture
) {
	const utciStorage = storage(utciStorageAttribute, 'float', utciStorageAttribute.count).toReadOnly();
	const pointIndex = instanceIndex;
	const activeCanonicalIndex = attribute('activeCanonicalIndex', 'uint');
	const canonicalRow = activeCanonicalIndex.mod(uint(layout.height));
	const column = activeCanonicalIndex.div(uint(layout.height));
	const row =
		layout.coordinateSystem === 'xy_ground'
			? uint(layout.height - 1).sub(canonicalRow)
			: canonicalRow;
	const x = float(column)
		.add(0.5)
		.mul(layout.gridSize)
		.sub((layout.width * layout.gridSize) / 2);
	const z = float(row)
		.add(0.5)
		.mul(layout.gridSize)
		.sub((layout.height * layout.gridSize) / 2);
	const value = utciStorage.element(pointIndex);
	const t = clamp(
		value.sub(minUniform).div(maxUniform.sub(minUniform).max(float(0.001))),
		0,
		1
	);
	const lutU = t.mul(
		(COMPUTE_BUFFER_COLOR_LUT_SIZE - 1) / COMPUTE_BUFFER_COLOR_LUT_SIZE
	).add(0.5 / COMPUTE_BUFFER_COLOR_LUT_SIZE);

	return {
		colorNode: texture(colorLutTexture, vec2(lutU, 0.5)).rgb,
		opacityNode: float(DEFAULT_SURFACE_OPACITY),
		positionNode: positionLocal.add(vec3(x, float(0), z))
	};
}

export function createActiveComputeBufferUtciSurfaceMesh(
	options: ActiveComputeBufferUtciSurfaceMeshOptions
): THREE.Mesh {
	assertTslInstanceIndexSupport();
	const now = options.now ?? performance.now.bind(performance);
	const { metricType, valueRange } = resolveComputeBufferMetricColorPolicy(options);
	const geometry = createActiveMaskUtciInstancedSurfaceGeometry(options.layout);
	const vertexCount = geometry.getAttribute('position').count;
	const utciArrayStartedAt = now();
	const utciArray = new Float32Array(options.layout.numPositions);
	addCreateSurfaceTraceTiming(
		options.trace,
		'createComputeBufferSurfaceUtciStorageAllocMs',
		now() - utciArrayStartedAt
	);
	const utciStorageAttribute = new StorageBufferAttribute(utciArray, 1);
	const activeCanonicalIndexAttribute = geometry.getAttribute(
		'activeCanonicalIndex'
	) as THREE.InstancedBufferAttribute;
	const minUniform = uniform(valueRange.min);
	const maxUniform = uniform(valueRange.max);
	const colorLutStartedAt = now();
	const colorLutTexture = createComputeBufferMetricColorLutTexture(metricType);
	addCreateSurfaceTraceTiming(
		options.trace,
		'createComputeBufferSurfaceColorLutSetupMs',
		now() - colorLutStartedAt
	);
	const materialStartedAt = now();
	const { colorNode, opacityNode, positionNode } = createActiveUtciRenderNodes(
		options.layout,
		utciStorageAttribute,
		minUniform,
		maxUniform,
		colorLutTexture
	);

	const material = new MeshBasicNodeMaterial({
		side: THREE.FrontSide,
		transparent: true,
		depthTest: true,
		depthWrite: false
	});
	material.colorNode = colorNode;
	material.opacityNode = opacityNode;
	material.positionNode = positionNode;
	material.toneMapped = false;
	addCreateSurfaceTraceTiming(
		options.trace,
		'createComputeBufferSurfaceMaterialSetupMs',
		now() - materialStartedAt
	);

	const meshConstructStartedAt = now();
	const mesh = new THREE.Mesh(geometry, material);
	mesh.name = 'UTCI GPU Resident Active Instanced Surface Overlay';
	mesh.renderOrder = 2;
	mesh.raycast = disabledActiveInstancedUtciSurfaceRaycast;
	mesh.userData.raycastDisabledReason =
		'active-instanced UTCI surface uses sparse tooltip lookup instead of Mesh.raycast';
	mesh.userData.pendingComputeBufferUtciSource = options.utciBuffer;
	mesh.userData.utciLayout = options.layout;
	const renderEstimate =
		options.renderEstimate ??
		estimateComputeBufferUtciSurfaceRenderStrategy({
			layout: options.layout,
			geometryBytes: getGeometryGpuAttributeBytes(geometry),
			utciStorageBytes: utciArray.byteLength
		});
	mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY] = {
		renderTopology: 'active-cells',
		activeMaskSignature: options.layout.activeMaskSignature,
		utciStorageAttribute,
		activeCanonicalIndexAttribute,
		width: options.layout.width,
		height: options.layout.height,
		gridSize: options.layout.gridSize,
		vertexCount,
		source: 'compute-buffer-selected-hour',
		utciRange: { ...valueRange },
		valueRange: { ...valueRange },
		metricType,
		minUniform,
		maxUniform,
		colorLutTexture,
		colorLutMetricType: metricType,
		renderEstimate
	} satisfies ActiveComputeBufferUtciSurfaceState;
	addCreateSurfaceTraceTiming(
		options.trace,
		'createComputeBufferSurfaceMeshConstructMs',
		now() - meshConstructStartedAt
	);

	const byteAccountingStartedAt = now();
	mesh.userData.renderOwnedSelectedHourBytes = renderEstimate.totalBytes;
	addCreateSurfaceTraceTiming(
		options.trace,
		'createComputeBufferSurfaceByteAccountingMs',
		now() - byteAccountingStartedAt
	);
	if (options.trace) {
		options.trace.createComputeBufferSurfaceGeometryBytes =
			renderEstimate.geometryBytes;
		options.trace.createComputeBufferSurfaceUtciStorageBytes =
			renderEstimate.selectedHourUtciStorageBytes;
		options.trace.createComputeBufferSurfaceCellToPointBytes =
			renderEstimate.cellToPointStorageBytes;
		options.trace.createComputeBufferSurfaceColorLutBytes = renderEstimate.colorLutBytes;
	}

	return mesh;
}
