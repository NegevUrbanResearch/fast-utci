import * as THREE from 'three';
import { MeshBasicNodeMaterial, StorageBufferAttribute } from 'three/webgpu';
import {
	clamp,
	float,
	positionLocal,
	storage,
	texture,
	uint,
	uniform,
	vec2,
	vertexIndex
} from 'three/tsl';
import type { UtciGridLayout } from './pointCloudService';
import type { F32MetricType } from '$lib/compute/on-demand/onDemandOutputFormat';
import type { SelectedHourRenderSurfaceMeshTrace } from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';
import {
	COMPUTE_BUFFER_COLOR_LUT_BYTES,
	COMPUTE_BUFFER_COLOR_LUT_SIZE,
	createComputeBufferMetricColorLutTexture,
	resolveComputeBufferMetricColorPolicy
} from '$lib/services/computeBufferMetricColorPolicy';
import {
	GPU_NATIVE_SURFACE_STATE_KEY,
	createCellToPointIndexArray,
	evaluateComputeBufferUtciSurfaceLayoutCompatibility,
	type ComputeBufferUtciSurfaceLayoutCompatibilityEvaluation,
	type ComputeBufferUtciSurfaceLayoutCompatibilityStateSnapshot,
	type GpuNativeUtciSurfaceSource
} from './gpuUtciSurfaceLayoutCompatibility';
export {
	areUtciGridLayoutsPointCompatible,
	createCellToPointIndexArray,
	createVertexToPointIndexArray,
	evaluateComputeBufferUtciSurfaceLayoutCompatibility,
	evaluateUtciGridLayoutsPointCompatibility,
	type ComputeBufferUtciSurfaceLayoutCompatibilityEvaluation,
	type ComputeBufferUtciSurfaceLayoutCompatibilityStateSnapshot,
	type GpuNativeUtciSurfaceSource,
	type UtciGridLayoutPointCompatibilityEvaluation
} from './gpuUtciSurfaceLayoutCompatibility';

export type SyntheticGpuUtciBridge = {
	group: THREE.Group;
	dispose: () => void;
};

type SyntheticGpuUtciBridgeOptions = {
	center: THREE.Vector3;
	size: THREE.Vector3;
};

export interface GpuNativeUtciSurfaceMeshOptions {
	layout: UtciGridLayout;
	colors: Float32Array;
	opacity?: number;
}

export interface ComputeBufferUtciSurfaceMeshOptions {
	layout: UtciGridLayout;
	utciBuffer: GPUBuffer;
	utciRange: { min: number; max: number };
	metricType?: F32MetricType;
	valueRange?: { min: number; max: number };
	opacity?: number;
	compatibilityEvaluation?: ComputeBufferUtciSurfaceLayoutCompatibilityEvaluation;
	trace?: SelectedHourRenderSurfaceMeshTrace;
	now?: () => number;
}

type ComputeBufferUtciSurfaceUpdateTrace = {
	updateComputeBufferSurfaceRangeUniformMs?: number;
	updateComputeBufferSurfacePendingSourceMs?: number;
	updateComputeBufferSurfaceLayoutUserDataMs?: number;
	updateComputeBufferSurfaceByteAccountingMs?: number;
};

interface GpuNativeUtciSurfaceState {
	colorStorageAttribute: StorageBufferAttribute;
	width: number;
	height: number;
	gridSize: number;
	vertexCount: number;
	source: GpuNativeUtciSurfaceSource;
}

interface ComputeBufferUtciSurfaceState extends GpuNativeUtciSurfaceState {
	source: 'compute-buffer-selected-hour';
	utciStorageAttribute: StorageBufferAttribute;
	cellToPointStorageAttribute: StorageBufferAttribute;
	utciRange: { min: number; max: number };
	valueRange: { min: number; max: number };
	metricType: F32MetricType;
	minUniform: ReturnType<typeof uniform>;
	maxUniform: ReturnType<typeof uniform>;
	colorLutTexture: THREE.DataTexture;
	colorLutMetricType: F32MetricType;
}

const MIN_SPAN = 12;
const GRID_SEGMENTS = 14;
const PROTOTYPE_ELEVATION = 1.5;
const DEFAULT_SURFACE_OPACITY = 0.9;
const SURFACE_VERTICES_PER_CELL = 6;
const SURFACE_COLOR_COMPONENTS = 4;
const workingColor = new THREE.Color();

function getGeometryGpuAttributeBytes(geometry: THREE.BufferGeometry): number {
	let total = 0;
	const attributes = geometry.attributes;
	for (const attribute of Object.values(attributes)) {
		const array = (attribute as THREE.BufferAttribute).array;
		total += array.byteLength;
	}
	const indexArray = geometry.index?.array;
	return total + (indexArray?.byteLength ?? 0);
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

function createStorageBackedColorArray(geometry: THREE.BufferGeometry, span: number): Float32Array {
	const positionAttribute = geometry.getAttribute('position');
	if (!positionAttribute) {
		throw new Error('Synthetic bridge geometry is missing positions.');
	}

	const colors = new Float32Array(positionAttribute.count * 4);
	for (let index = 0; index < positionAttribute.count; index += 1) {
		const x = positionAttribute.getX(index) / span;
		const y = positionAttribute.getY(index) / span;
		const offset = index * 4;

		colors[offset] = 0.25 + 0.75 * (x + 0.5);
		colors[offset + 1] = 0.2 + 0.8 * (y + 0.5);
		colors[offset + 2] = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin((x - y) * Math.PI * 3));
		colors[offset + 3] = 0.92;
	}

	return colors;
}

export function createSyntheticGpuUtciBridge(
	options: SyntheticGpuUtciBridgeOptions
): SyntheticGpuUtciBridge {
	const span = Math.max(options.size.x, options.size.z, MIN_SPAN);
	const geometry = new THREE.PlaneGeometry(span, span, GRID_SEGMENTS, GRID_SEGMENTS).toNonIndexed();
	const colorArray = createStorageBackedColorArray(geometry, span);
	const colorStorageAttribute = new StorageBufferAttribute(colorArray, 4);
	const colorStorage = storage(colorStorageAttribute, 'vec4', colorStorageAttribute.count).toReadOnly();

	const material = new MeshBasicNodeMaterial({
		side: THREE.DoubleSide,
		transparent: true
	});
	material.colorNode = colorStorage.element(vertexIndex).xyz;
	material.opacityNode = colorStorage.element(vertexIndex).w;

	const mesh = new THREE.Mesh(geometry, material);
	mesh.name = 'SyntheticGpuUtciBridgeMesh';
	mesh.frustumCulled = false;
	mesh.rotation.x = -Math.PI / 2;
	mesh.position.copy(options.center);
	mesh.position.y += Math.max(options.size.y * 0.05, PROTOTYPE_ELEVATION);

	const group = new THREE.Group();
	group.name = 'SyntheticGpuUtciBridge';
	group.add(mesh);

	return {
		group,
		dispose: () => {
			group.removeFromParent();
			geometry.dispose();
			material.dispose();
		}
	};
}

export function createGpuNativeUtciSurfaceMesh(
	options: GpuNativeUtciSurfaceMeshOptions
): THREE.Mesh {
	const geometry = createGpuNativeSurfaceGeometry(options.layout);
	const vertexCount = geometry.getAttribute('position').count;
	const colorArray = new Float32Array(vertexCount * SURFACE_COLOR_COMPONENTS);
	const opacity = options.opacity ?? DEFAULT_SURFACE_OPACITY;

	fillGpuNativeSurfaceVertexColors(colorArray, options.layout, options.colors, opacity);

	const colorStorageAttribute = new StorageBufferAttribute(colorArray, SURFACE_COLOR_COMPONENTS);
	const colorStorage = storage(
		colorStorageAttribute,
		'vec4',
		colorStorageAttribute.count
	).toReadOnly();

	const material = new MeshBasicNodeMaterial({
		side: THREE.FrontSide,
		transparent: true,
		depthTest: true,
		depthWrite: false
	});
	material.colorNode = colorStorage.element(vertexIndex).xyz;
	material.opacityNode = colorStorage.element(vertexIndex).w;
	material.toneMapped = false;

	const mesh = new THREE.Mesh(geometry, material);
	mesh.name = 'GpuNativeUtciSurfaceMesh';
	mesh.renderOrder = 2;
	mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY] = {
		colorStorageAttribute,
		width: options.layout.width,
		height: options.layout.height,
		gridSize: options.layout.gridSize,
		vertexCount,
		source: 'cpu-uploaded-selected-hour'
	} satisfies GpuNativeUtciSurfaceState;
	mesh.userData.renderOwnedSelectedHourBytes =
		getGeometryGpuAttributeBytes(geometry) + colorArray.byteLength;

	return mesh;
}

export function createComputeBufferUtciSurfaceMesh(
	options: ComputeBufferUtciSurfaceMeshOptions
): THREE.Mesh {
	const now = options.now ?? performance.now.bind(performance);
	const { metricType, valueRange } = resolveComputeBufferMetricColorPolicy(options);
	const geometry = createIndexedGridSurfaceGeometry(options.layout, {
		trace: options.trace,
		now
	});
	const vertexCount = geometry.getAttribute('position').count;
	const utciArrayStartedAt = now();
	const utciArray = new Float32Array(options.layout.numPositions);
	addCreateSurfaceTraceTiming(
		options.trace,
		'createComputeBufferSurfaceUtciStorageAllocMs',
		now() - utciArrayStartedAt
	);
	const utciStorageAttribute = new StorageBufferAttribute(utciArray, 1);
	const cellToPointStartedAt = now();
	const cellToPointArray = createCellToPointIndexArray(options.layout);
	const cellToPointStorageAttribute = new StorageBufferAttribute(cellToPointArray, 1);
	addCreateSurfaceTraceTiming(
		options.trace,
		'createComputeBufferSurfaceCellToPointAllocFillMs',
		now() - cellToPointStartedAt
	);
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
	const { colorNode, opacityNode } = createUtciColorNode(
		options.layout,
		utciStorageAttribute,
		cellToPointStorageAttribute,
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
	material.toneMapped = false;
	addCreateSurfaceTraceTiming(
		options.trace,
		'createComputeBufferSurfaceMaterialSetupMs',
		now() - materialStartedAt
	);

	const meshConstructStartedAt = now();
	const mesh = new THREE.Mesh(geometry, material);
	mesh.name = 'UTCI GPU Resident Surface Overlay';
	mesh.renderOrder = 2;
	mesh.userData.pendingComputeBufferUtciSource = options.utciBuffer;
	mesh.userData.utciLayout = options.layout;
	mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY] = {
		colorStorageAttribute: utciStorageAttribute,
		utciStorageAttribute,
		cellToPointStorageAttribute,
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
		colorLutMetricType: metricType
	} satisfies ComputeBufferUtciSurfaceState;
	addCreateSurfaceTraceTiming(
		options.trace,
		'createComputeBufferSurfaceMeshConstructMs',
		now() - meshConstructStartedAt
	);
	const byteAccountingStartedAt = now();
	const geometryBytes = getGeometryGpuAttributeBytes(geometry);
	mesh.userData.renderOwnedSelectedHourBytes =
		geometryBytes +
		utciArray.byteLength +
		cellToPointArray.byteLength +
		COMPUTE_BUFFER_COLOR_LUT_BYTES;
	addCreateSurfaceTraceTiming(
		options.trace,
		'createComputeBufferSurfaceByteAccountingMs',
		now() - byteAccountingStartedAt
	);
	if (options.trace) {
		options.trace.createComputeBufferSurfaceGeometryBytes = geometryBytes;
		options.trace.createComputeBufferSurfaceUtciStorageBytes = utciArray.byteLength;
		options.trace.createComputeBufferSurfaceCellToPointBytes = cellToPointArray.byteLength;
		options.trace.createComputeBufferSurfaceColorLutBytes =
			COMPUTE_BUFFER_COLOR_LUT_BYTES;
	}

	return mesh;
}

export function updateComputeBufferUtciSurfaceMesh(
	mesh: THREE.Mesh,
	options: ComputeBufferUtciSurfaceMeshOptions & {
		trace?: ComputeBufferUtciSurfaceUpdateTrace;
		now?: () => number;
	}
): boolean {
	const compatibilityEvaluation =
		options.compatibilityEvaluation ??
		evaluateComputeBufferUtciSurfaceLayoutCompatibility({
			state: getComputeBufferUtciSurfaceLayoutCompatibilityState(mesh),
			previousLayout: mesh?.userData.utciLayout as UtciGridLayout | undefined,
			nextLayout: options.layout,
			metricType: options.metricType ?? 'utci',
			allowExpensiveMappingComparison: true
		});
	if ((compatibilityEvaluation.compatible ?? false) !== true) {
		return false;
	}
	const state = mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY] as ComputeBufferUtciSurfaceState;

	const now = options.now ?? performance.now.bind(performance);
	const { metricType, valueRange } = resolveComputeBufferMetricColorPolicy(options);
	if (state.colorLutMetricType !== metricType) {
		return false;
	}
	const rangeUniformStartedAt = now();
	state.utciRange = { ...valueRange };
	state.valueRange = { ...valueRange };
	state.metricType = metricType;
	state.minUniform.value = valueRange.min;
	state.maxUniform.value = valueRange.max;
	if (options.trace) {
		options.trace.updateComputeBufferSurfaceRangeUniformMs =
			now() - rangeUniformStartedAt;
	}
	const pendingSourceStartedAt = now();
	mesh.userData.pendingComputeBufferUtciSource = options.utciBuffer;
	if (options.trace) {
		options.trace.updateComputeBufferSurfacePendingSourceMs =
			now() - pendingSourceStartedAt;
	}
	const layoutUserDataStartedAt = now();
	mesh.userData.utciLayout = options.layout;
	if (options.trace) {
		options.trace.updateComputeBufferSurfaceLayoutUserDataMs =
			now() - layoutUserDataStartedAt;
	}
	const byteAccountingStartedAt = now();
	mesh.userData.renderOwnedSelectedHourBytes =
		getGeometryGpuAttributeBytes(mesh.geometry) +
		state.utciStorageAttribute.array.byteLength +
		state.cellToPointStorageAttribute.array.byteLength +
		COMPUTE_BUFFER_COLOR_LUT_BYTES;
	if (options.trace) {
		options.trace.updateComputeBufferSurfaceByteAccountingMs =
			now() - byteAccountingStartedAt;
	}
	return true;
}

export function isComputeBufferUtciSurfaceLayoutCompatible(
	mesh: THREE.Mesh | null | undefined,
	layout: UtciGridLayout,
	metricType?: F32MetricType
): boolean {
	const previousLayout = mesh?.userData.utciLayout as UtciGridLayout | undefined;
	return (
		evaluateComputeBufferUtciSurfaceLayoutCompatibility({
			state: getComputeBufferUtciSurfaceLayoutCompatibilityState(mesh),
			previousLayout,
			nextLayout: layout,
			metricType,
			allowExpensiveMappingComparison: true
		}).compatible ?? false
	);
}

export function getComputeBufferUtciSurfaceLayoutCompatibilityState(
	mesh: THREE.Mesh | null | undefined
): ComputeBufferUtciSurfaceLayoutCompatibilityStateSnapshot | null {
	const state = mesh?.userData[GPU_NATIVE_SURFACE_STATE_KEY] as
		| ComputeBufferUtciSurfaceState
		| undefined;
	if (!state) {
		return null;
	}

	return {
		source: state.source,
		metricType: state.metricType,
		width: state.width,
		height: state.height,
		gridSize: state.gridSize,
		vertexCount: state.vertexCount,
		storageCount: state.utciStorageAttribute.count
	};
}

export function updateGpuNativeUtciSurfaceMesh(
	mesh: THREE.Mesh,
	options: GpuNativeUtciSurfaceMeshOptions
): boolean {
	const state = mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY] as GpuNativeUtciSurfaceState | undefined;
	if (!state) {
		console.warn('[UTCI] Missing gpuNative surface state. Recreate the mesh.');
		return false;
	}
	if (state.source !== 'cpu-uploaded-selected-hour') {
		console.warn(
			`[UTCI] gpuNative surface source ${state.source} is not CPU-uploaded; recreate the mesh.`
		);
		return false;
	}

	const expectedVertexCount = options.layout.width * options.layout.height * SURFACE_VERTICES_PER_CELL;
	if (
		state.width !== options.layout.width ||
		state.height !== options.layout.height ||
		state.gridSize !== options.layout.gridSize ||
		state.vertexCount !== expectedVertexCount
	) {
		console.warn('[UTCI] gpuNative surface layout changed; recreate the mesh.');
		return false;
	}

	const colorArray = state.colorStorageAttribute.array as Float32Array;
	fillGpuNativeSurfaceVertexColors(
		colorArray,
		options.layout,
		options.colors,
		options.opacity ?? DEFAULT_SURFACE_OPACITY
	);
	state.colorStorageAttribute.needsUpdate = true;
	return true;
}

export function disposeGpuNativeUtciSurfaceMesh(mesh: THREE.Mesh): void {
	mesh.removeFromParent();

	const material = mesh.material;
	if (Array.isArray(material)) {
		for (const entry of material) {
			entry.dispose();
		}
	} else {
		material.dispose();
	}

	const state = mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY] as
		| ComputeBufferUtciSurfaceState
		| GpuNativeUtciSurfaceState
		| undefined;
	if (state && 'colorLutTexture' in state) {
		state.colorLutTexture.dispose();
	}

	mesh.geometry.dispose();
	delete mesh.userData.pendingComputeBufferUtciSource;
	delete mesh.userData.renderOwnedSelectedHourBytes;
	delete mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY];
}

export function getGpuNativeUtciSurfaceSource(
	mesh: THREE.Mesh
): GpuNativeUtciSurfaceSource | undefined {
	return (mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY] as GpuNativeUtciSurfaceState | undefined)
		?.source;
}

export function getComputeBufferUtciStorageAttribute(
	mesh: THREE.Mesh
): StorageBufferAttribute | undefined {
	const state = mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY] as
		| ComputeBufferUtciSurfaceState
		| undefined;
	if (!state || state.source !== 'compute-buffer-selected-hour') {
		return undefined;
	}

	return state.utciStorageAttribute;
}

function createUtciColorNode(
	layout: UtciGridLayout,
	utciStorageAttribute: StorageBufferAttribute,
	cellToPointStorageAttribute: StorageBufferAttribute,
	minUniform: ReturnType<typeof uniform>,
	maxUniform: ReturnType<typeof uniform>,
	colorLutTexture: THREE.DataTexture
) {
	const utciStorage = storage(utciStorageAttribute, 'float', utciStorageAttribute.count).toReadOnly();
	const cellToPointStorage = storage(
		cellToPointStorageAttribute,
		'uint',
		cellToPointStorageAttribute.count
	).toReadOnly();
	const halfWidth = float((layout.width * layout.gridSize) / 2);
	const halfHeight = float((layout.height * layout.gridSize) / 2);
	const gridSize = float(layout.gridSize);
	const column = uint(
		clamp(positionLocal.x.add(halfWidth).div(gridSize).floor(), 0, layout.width - 1)
	);
	const row = uint(
		clamp(positionLocal.z.add(halfHeight).div(gridSize).floor(), 0, layout.height - 1)
	);
	const cellIndex = row.mul(uint(layout.width)).add(column);
	const pointIndex = cellToPointStorage.element(cellIndex);
	const hasPoint = pointIndex.lessThan(uint(layout.numPositions));
	const safePointIndex = hasPoint.select(pointIndex, uint(0));
	const value = utciStorage.element(safePointIndex);
	const t = clamp(
		value.sub(minUniform).div(maxUniform.sub(minUniform).max(float(0.001))),
		0,
		1
	);
	const lutU = t.mul(
		(COMPUTE_BUFFER_COLOR_LUT_SIZE - 1) / COMPUTE_BUFFER_COLOR_LUT_SIZE
	).add(0.5 / COMPUTE_BUFFER_COLOR_LUT_SIZE);
	const colorNode = texture(colorLutTexture, vec2(lutU, 0.5)).rgb;

	return {
		colorNode,
		opacityNode: hasPoint.select(float(DEFAULT_SURFACE_OPACITY), float(0))
	};
}

function createGpuNativeSurfaceGeometry(layout: UtciGridLayout): THREE.BufferGeometry {
	const geometry = new THREE.BufferGeometry();
	const planeWidth = layout.width * layout.gridSize;
	const planeHeight = layout.height * layout.gridSize;
	const halfWidth = planeWidth / 2;
	const halfHeight = planeHeight / 2;
	const positions = new Float32Array(
		layout.width * layout.height * SURFACE_VERTICES_PER_CELL * 3
	);

	let offset = 0;
	for (let row = 0; row < layout.height; row += 1) {
		const z0 = -halfHeight + row * layout.gridSize;
		const z1 = z0 + layout.gridSize;

		for (let col = 0; col < layout.width; col += 1) {
			const x0 = -halfWidth + col * layout.gridSize;
			const x1 = x0 + layout.gridSize;

			positions[offset++] = x0;
			positions[offset++] = 0;
			positions[offset++] = z1;
			positions[offset++] = x1;
			positions[offset++] = 0;
			positions[offset++] = z1;
			positions[offset++] = x0;
			positions[offset++] = 0;
			positions[offset++] = z0;

			positions[offset++] = x1;
			positions[offset++] = 0;
			positions[offset++] = z1;
			positions[offset++] = x1;
			positions[offset++] = 0;
			positions[offset++] = z0;
			positions[offset++] = x0;
			positions[offset++] = 0;
			positions[offset++] = z0;
		}
	}

	geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
	geometry.computeBoundingBox();
	geometry.computeBoundingSphere();
	return geometry;
}

function setIndexedGridSurfaceAnalyticBounds(
	geometry: THREE.BufferGeometry,
	params: { halfWidth: number; halfHeight: number }
): void {
	geometry.boundingBox = new THREE.Box3(
		new THREE.Vector3(-params.halfWidth, 0, -params.halfHeight),
		new THREE.Vector3(params.halfWidth, 0, params.halfHeight)
	);
	geometry.boundingSphere = new THREE.Sphere(
		new THREE.Vector3(0, 0, 0),
		Math.hypot(params.halfWidth, params.halfHeight)
	);
}

function createIndexedGridSurfaceGeometry(
	layout: UtciGridLayout,
	options?: {
		trace?: SelectedHourRenderSurfaceMeshTrace;
		now?: () => number;
	}
): THREE.BufferGeometry {
	const now = options?.now ?? performance.now.bind(performance);
	const geometry = new THREE.BufferGeometry();
	const planeWidth = layout.width * layout.gridSize;
	const planeHeight = layout.height * layout.gridSize;
	const halfWidth = planeWidth / 2;
	const halfHeight = planeHeight / 2;
	const vertexWidth = layout.width + 1;
	const vertexHeight = layout.height + 1;
	const positionAllocStartedAt = now();
	const positions = new Float32Array(vertexWidth * vertexHeight * 3);
	addCreateSurfaceTraceTiming(
		options?.trace,
		'createComputeBufferSurfacePositionArrayAllocMs',
		now() - positionAllocStartedAt
	);
	const indexAllocStartedAt = now();
	const indices = new Uint32Array(layout.width * layout.height * SURFACE_VERTICES_PER_CELL);
	addCreateSurfaceTraceTiming(
		options?.trace,
		'createComputeBufferSurfaceIndexArrayAllocMs',
		now() - indexAllocStartedAt
	);

	const positionFillStartedAt = now();
	let positionOffset = 0;
	for (let row = 0; row < vertexHeight; row += 1) {
		const z = -halfHeight + row * layout.gridSize;
		for (let col = 0; col < vertexWidth; col += 1) {
			const x = -halfWidth + col * layout.gridSize;
			positions[positionOffset++] = x;
			positions[positionOffset++] = 0;
			positions[positionOffset++] = z;
		}
	}
	addCreateSurfaceTraceTiming(
		options?.trace,
		'createComputeBufferSurfacePositionArrayFillMs',
		now() - positionFillStartedAt
	);

	const indexFillStartedAt = now();
	let indexOffset = 0;
	for (let row = 0; row < layout.height; row += 1) {
		for (let col = 0; col < layout.width; col += 1) {
			const v00 = row * vertexWidth + col;
			const v10 = v00 + 1;
			const v01 = v00 + vertexWidth;
			const v11 = v01 + 1;
			indices[indexOffset++] = v01;
			indices[indexOffset++] = v11;
			indices[indexOffset++] = v00;
			indices[indexOffset++] = v11;
			indices[indexOffset++] = v10;
			indices[indexOffset++] = v00;
		}
	}
	addCreateSurfaceTraceTiming(
		options?.trace,
		'createComputeBufferSurfaceIndexArrayFillMs',
		now() - indexFillStartedAt
	);

	const attributeStartedAt = now();
	geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
	geometry.setIndex(new THREE.BufferAttribute(indices, 1));
	addCreateSurfaceTraceTiming(
		options?.trace,
		'createComputeBufferSurfaceGeometryAttributeAttachMs',
		now() - attributeStartedAt
	);
	const boundsStartedAt = now();
	setIndexedGridSurfaceAnalyticBounds(geometry, { halfWidth, halfHeight });
	addCreateSurfaceTraceTiming(
		options?.trace,
		'createComputeBufferSurfaceBoundsMs',
		now() - boundsStartedAt
	);
	return geometry;
}

function fillGpuNativeSurfaceVertexColors(
	target: Float32Array,
	layout: UtciGridLayout,
	colors: Float32Array,
	opacity: number
): void {
	target.fill(0);

	const cellColors = new Float32Array(layout.width * layout.height * SURFACE_COLOR_COMPONENTS);
	for (let index = 0; index < layout.numPositions; index += 1) {
		const cellIndex = layout.indexToRow[index] * layout.width + layout.indexToColumn[index];
		const colorOffset = cellIndex * SURFACE_COLOR_COMPONENTS;
		const sourceOffset = index * 3;
		const linearColor = workingColor.setRGB(
			colors[sourceOffset],
			colors[sourceOffset + 1],
			colors[sourceOffset + 2],
			THREE.SRGBColorSpace
		);

		cellColors[colorOffset] = linearColor.r;
		cellColors[colorOffset + 1] = linearColor.g;
		cellColors[colorOffset + 2] = linearColor.b;
		cellColors[colorOffset + 3] = opacity;
	}

	let targetOffset = 0;
	for (let row = 0; row < layout.height; row += 1) {
		for (let col = 0; col < layout.width; col += 1) {
			const cellOffset = (row * layout.width + col) * SURFACE_COLOR_COMPONENTS;
			const r = cellColors[cellOffset];
			const g = cellColors[cellOffset + 1];
			const b = cellColors[cellOffset + 2];
			const a = cellColors[cellOffset + 3];

			for (let vertex = 0; vertex < SURFACE_VERTICES_PER_CELL; vertex += 1) {
				target[targetOffset++] = r;
				target[targetOffset++] = g;
				target[targetOffset++] = b;
				target[targetOffset++] = a;
			}
		}
	}
}
