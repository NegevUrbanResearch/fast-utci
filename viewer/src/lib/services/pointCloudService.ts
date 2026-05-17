/**
 * Point Cloud Service
 * 
 * Service for creating and updating UTCI point cloud geometries
 */

import * as THREE from 'three';
import type { Analysis, AnalysisMetadata, UTCIData } from '$lib/types/analysis';
import type { MetricType } from '$lib/types/viewer';
import { getUTCIForHour, getShadingIndex } from '$lib/services/dataLoader';
import { mapUTCIToColor, mapShadingIndexToColor } from '$lib/services/colorScale';
import { getAnchorOffset, isNormalizationEnabled } from '$lib/config/viewerConfig';
import { calculateScenarioOrigin } from '$lib/utils/coordinates';
import { getUtciRangeForDisplay } from '$lib/utils/effectiveHourIndex';
import type {
	SelectedHourRenderLayoutBuildTrace,
	SelectedHourRenderLayoutConstructionMode,
	SelectedHourRenderLayoutNormalizationSignature,
	SelectedHourRenderLayoutReuseProofTrace
} from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';
import {
	createGpuNativeUtciSurfaceMesh,
	disposeGpuNativeUtciSurfaceMesh,
	evaluateComputeBufferUtciSurfaceLayoutCompatibility,
	evaluateUtciGridLayoutsPointCompatibility,
	getGpuNativeUtciSurfaceSource,
	type UtciGridLayoutPointCompatibilityEvaluation,
	updateGpuNativeUtciSurfaceMesh
} from './gpuUtciRenderBridge';

// Vertical separation between the UTCI overlay and underlying geometry.
// Use a small negative offset so the UTCI plane sits just below the sampled
// ground grid; it will still be visible through the semi-transparent base
// mesh but remain behind buildings in depth.
const VISUAL_LAYER_OFFSET = -0.05;
const DEFAULT_OPACITY = 0.9;
const TEXTURE_ALPHA = 255;
const AMBIGUOUS_CELL_POINT_INDEX = -2;

export interface UtciGridLayout {
	width: number;
	height: number;
	gridSize: number;
	coordinateSystem: 'xy_ground' | 'xz_ground';
	numPositions: number;
	minX: number;
	minZ: number;
	minY: number;
	maxY: number;
	centerX: number;
	centerZ: number;
	baseY: number;
	indexToRow: Uint32Array;
	indexToColumn: Uint32Array;
	cellToPointIndex?: Int32Array;
	indexToTexel: Uint32Array;
	colorBuffer: Uint8Array;
	positionsIdentityId?: number;
	constructionMode?: SelectedHourRenderLayoutConstructionMode;
	normalizationSignature?: SelectedHourRenderLayoutNormalizationSignature;
	texture?: THREE.DataTexture;
}

export type UtciGridLayoutBuildDiagnostics = Partial<SelectedHourRenderLayoutBuildTrace>;
export type UtciGridLayoutReuseProofDiagnostics = SelectedHourRenderLayoutReuseProofTrace;
export type UtciLayoutReuseKey = {
	analysisId: string;
	layoutSourceSignature: string;
	gridSize: number;
	pointCount: number;
	coordinateSystem: string;
	normalizationSignature: string;
	constructionMode: 'world-positions' | 'metadata-bounds-fallback' | 'invalid';
	width: number;
	height: number;
	centerX: number;
	centerZ: number;
	baseY: number;
	utciSurfaceSource: 'compute-buffer-selected-hour';
	rendererBackend: 'webgpu';
};
export type UtciLayoutReuseKeyDiagnostics = {
	keyBuildMs?: number;
	layoutSourceSignatureMs?: number;
	positionsSourceSignatureMs?: number;
	positionsSourceSignatureCacheHit?: boolean;
	frameCacheLookupMs?: number;
	frameDerivationMs?: number;
	frameCacheHit?: boolean;
	frameCacheKind?: 'analysis-object' | 'structural' | 'miss';
};
export type UtciLayoutReusePublicationState = {
	proof: UtciGridLayoutReuseProofDiagnostics;
	key: UtciLayoutReuseKey;
	layoutIdentity: string;
	requestId: number | null;
	selectionKey: string | null;
};
export type UtciLayoutReuseDecisionReason =
	| 'reuse-safe'
	| 'missing-previous-layout'
	| 'proof-not-safe'
	| 'canonical-mismatch'
	| 'mapping-unsafe'
	| 'hover-proof-missing'
	| 'backend-or-source-mismatch'
	| 'layout-key-mismatch'
	| 'diagnostics-missing';
export type UtciLayoutReuseDecision = {
	action: 'reuse-candidate' | 'build-required';
	reason: UtciLayoutReuseDecisionReason;
	keyMatch: boolean;
};
export type UtciLayoutPublicationPlan =
	| {
			action: 'reuse-existing';
			layout: UtciGridLayout;
			reason: 'reuse-safe';
			keyMatch: true;
	  }
	| {
			action: 'build-new';
			reason:
				| UtciLayoutReuseDecisionReason
				| 'initial-publication';
			keyMatch: boolean;
	  };
const positionsIdentityIds = new WeakMap<Float32Array, number>();
let positionsSourceSignatureCache = new WeakMap<Float32Array, string>();
let utciLayoutFrameCache = new WeakMap<
	Analysis,
	{
		layoutSourceSignature: string;
		frame: DerivedUtciLayoutFrame;
	}
>();
const STRUCTURAL_FRAME_CACHE_LIMIT = 8;
const structuralUtciLayoutFrameCache = new Map<
	string,
	{
		frame: DerivedUtciLayoutFrame;
		layoutSourceSignature: string;
	}
>();
let nextPositionsIdentityId = 1;
let positionsSourceSignatureComputationCount = 0;

function rememberStructuralUtciLayoutFrame(
	key: string,
	entry: {
		frame: DerivedUtciLayoutFrame;
		layoutSourceSignature: string;
	}
): void {
	if (
		!structuralUtciLayoutFrameCache.has(key) &&
		structuralUtciLayoutFrameCache.size >= STRUCTURAL_FRAME_CACHE_LIMIT
	) {
		const oldestKey = structuralUtciLayoutFrameCache.keys().next().value;
		if (oldestKey !== undefined) {
			structuralUtciLayoutFrameCache.delete(oldestKey);
		}
	}
	structuralUtciLayoutFrameCache.set(key, entry);
}

function getPositionsIdentityId(positions: Float32Array): number {
	const existing = positionsIdentityIds.get(positions);
	if (existing !== undefined) {
		return existing;
	}
	const created = nextPositionsIdentityId++;
	positionsIdentityIds.set(positions, created);
	return created;
}

export type UtciSurfaceBackendType = 'dataTexture' | 'gpuNative';

export interface UtciSurfaceMeshOptions {
	analysis: Analysis;
	hourIndex?: number;
	colorMode?: 'normalized' | 'discrete';
	metricType?: MetricType;
	rangeOverride?: UtciRangeOverride;
	monthIndex?: number;
	backend?: UtciSurfaceBackendType;
}

interface ResolvedUtciSurfaceMeshOptions {
	analysis: Analysis;
	hourIndex: number;
	colorMode: 'normalized' | 'discrete';
	metricType: MetricType;
	rangeOverride?: UtciRangeOverride;
	monthIndex: number;
	backend: UtciSurfaceBackendType;
}

interface UtciSurfaceBackend {
	type: UtciSurfaceBackendType;
	createMesh: (layout: UtciGridLayout, colors: Float32Array) => THREE.Mesh;
	updateMesh: (mesh: THREE.Mesh, layout: UtciGridLayout, colors: Float32Array) => boolean;
	disposeMesh: (mesh: THREE.Mesh) => void;
}

const DEFAULT_UTCI_SURFACE_BACKEND: UtciSurfaceBackendType = 'dataTexture';

/**
 * Create UTCI point cloud geometry and material
 * @param analysis - Analysis data
 * @param hourIndex - Hour index (default: 0, ignored for Shading Index)
 * @param colorMode - Color mode ('normalized' or 'discrete', ignored for Shading Index)
 * @param metricType - Metric type ('utci' or 'shading_index')
 * @returns Object with geometry and material
 */
export function createPointCloudGeometry(
	analysis: Analysis,
	hourIndex: number = 0,
	colorMode: 'normalized' | 'discrete' = 'normalized',
	metricType: MetricType = 'utci'
): { geometry: THREE.BufferGeometry; material: THREE.PointsMaterial } {
	const { data, metadata } = analysis;
	const numPositions = data.numPositions;

	// Create position attribute without baked vertical offset
	const positions = new Float32Array(data.positions.length);
	for (let i = 0; i < numPositions; i++) {
		positions[i * 3] = data.positions[i * 3];
		positions[i * 3 + 1] = data.positions[i * 3 + 1];
		positions[i * 3 + 2] = data.positions[i * 3 + 2];
	}

	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

	// Create color attribute
	const colors = createColors(analysis, hourIndex, colorMode, metricType);
	geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

	// Create material
	const material = new THREE.PointsMaterial({
		size: 8,
		vertexColors: true,
		sizeAttenuation: false,
		transparent: true,
		opacity: 0.95,
		depthTest: true,
		depthWrite: false,
		alphaTest: 0.1
	});

	return { geometry, material };
}

/**
 * Optional range override for unified color scales during comparison
 */
export interface UtciRangeOverride {
	utciMin: number;
	utciMax: number;
}

const utciSurfaceBackends: Record<UtciSurfaceBackendType, UtciSurfaceBackend> = {
	dataTexture: {
		type: 'dataTexture',
		createMesh(layout, colors) {
			fillColorBuffer(layout, colors);

			const texture = new THREE.DataTexture(
				layout.colorBuffer,
				layout.width,
				layout.height,
				THREE.RGBAFormat,
				THREE.UnsignedByteType
			);
			texture.needsUpdate = true;
			texture.flipY = false;
			texture.generateMipmaps = false;
			texture.magFilter = THREE.NearestFilter;
			texture.minFilter = THREE.NearestFilter;
			texture.colorSpace = THREE.SRGBColorSpace;

			layout.texture = texture;

			const material = new THREE.MeshBasicMaterial({
				map: texture,
				transparent: true,
				opacity: DEFAULT_OPACITY,
				side: THREE.DoubleSide,
				depthTest: true,
				depthWrite: false,
				polygonOffset: false,
				toneMapped: false
			});

			const geometry = createUtciSurfacePlaneGeometry(layout);
			const mesh = new THREE.Mesh(geometry, material);
			mesh.name = 'UTCI Texture Overlay';
			applySurfaceMeshState(mesh, layout, 'dataTexture');
			incrementDataTextureBuildCount(mesh);
			return mesh;
		},
		updateMesh(mesh, layout, colors) {
			const material = mesh.material as THREE.MeshBasicMaterial;
			const texture = layout.texture ?? (material.map as THREE.DataTexture | null);
			if (!texture) {
				console.warn('[UTCI] Missing texture on dataTexture surface mesh. Recreate the mesh.');
				return false;
			}

			fillColorBuffer(layout, colors);
			texture.needsUpdate = true;
			layout.texture = texture;
			applySurfaceMeshState(mesh, layout, 'dataTexture');
			incrementDataTextureBuildCount(mesh);
			return true;
		},
		disposeMesh(mesh) {
			disposeSurfaceMeshAssets(mesh);
		}
	},
	gpuNative: {
		type: 'gpuNative',
		createMesh(layout, colors) {
			const mesh = createGpuNativeUtciSurfaceMesh({ layout, colors, opacity: DEFAULT_OPACITY });
			applySurfaceMeshState(mesh, layout, 'gpuNative');
			incrementSelectedHourTransferCount(mesh);
			mesh.name = 'UTCI GPU Surface Overlay';
			return mesh;
		},
		updateMesh(mesh, layout, colors) {
			const updated = updateGpuNativeUtciSurfaceMesh(mesh, {
				layout,
				colors,
				opacity: DEFAULT_OPACITY
			});
			if (!updated) {
				return false;
			}

			applySurfaceMeshState(mesh, layout, 'gpuNative');
			incrementSelectedHourTransferCount(mesh);
			return true;
		},
		disposeMesh(mesh) {
			disposeGpuNativeUtciSurfaceMesh(mesh);
		}
	}
};

/**
 * Create color array for point cloud
 * @param analysis - Analysis data
 * @param hourIndex - Hour index (ignored for Shading Index)
 * @param colorMode - Color mode (ignored for Shading Index)
 * @param metricType - Metric type ('utci' or 'shading_index')
 * @param rangeOverride - Optional UTCI range override for unified comparison scales
 * @param monthIndex - Month index for multi-month analyses (used for "full day" = selected month's 24h)
 * @returns Color Float32Array
 */
export function createColors(
	analysis: Analysis,
	hourIndex: number,
	colorMode: 'normalized' | 'discrete',
	metricType: MetricType = 'utci',
	rangeOverride?: UtciRangeOverride,
	monthIndex: number = 7
): Float32Array {
	const { data, metadata } = analysis;
	const numPositions = data.numPositions;

	// Create color attribute
	const colors = new Float32Array(numPositions * 3);

	if (metricType === 'shading_index') {
		// Shading Index: use full-day aggregated value (ignores hourIndex)
		const shadingIndexValues = getShadingIndex(data);
		
		if (!shadingIndexValues) {
			// Fallback to UTCI if Shading Index not available (backward compatibility)
			console.warn('[PointCloud] Shading Index requested but not available, falling back to UTCI');
			return createColors(analysis, hourIndex, colorMode, 'utci', rangeOverride, monthIndex);
		}

		// Get Shading Index range from metadata
		const shadingIndexMin = metadata.shading_index_range?.min ?? 0;
		const shadingIndexMax = metadata.shading_index_range?.max ?? 1;

		for (let i = 0; i < numPositions; i++) {
			const shadingIndex = shadingIndexValues[i];
			const color = mapShadingIndexToColor(shadingIndex, shadingIndexMin, shadingIndexMax);
			colors[i * 3] = color.r;
			colors[i * 3 + 1] = color.g;
			colors[i * 3 + 2] = color.b;
		}
	} else {
		// UTCI: use hour-specific value
		const utciValues = getUTCIForHour(data, hourIndex);
		
		// Use override range if provided (for comparison mode), otherwise resolve from metadata
		const { utciMin, utciMax } =
			rangeOverride ?? getUtciRangeForDisplay(metadata, colorMode, hourIndex, monthIndex);

		for (let i = 0; i < numPositions; i++) {
			const utci = utciValues[i];
			const color = mapUTCIToColor(utci, utciMin, utciMax);
			colors[i * 3] = color.r;
			colors[i * 3 + 1] = color.g;
			colors[i * 3 + 2] = color.b;
		}
	}

	return colors;
}

/**
 * Update point cloud colors
 * @param pointCloud - Three.js Points object
 * @param analysis - Analysis data
 * @param hourIndex - Hour index (ignored for Shading Index)
 * @param colorMode - Color mode (ignored for Shading Index)
 * @param metricType - Metric type ('utci' or 'shading_index')
 */
export function updatePointCloudColors(
	pointCloud: THREE.Points,
	analysis: Analysis,
	hourIndex: number,
	colorMode: 'normalized' | 'discrete',
	metricType: MetricType = 'utci'
): void {
	const { data } = analysis;

	// For UTCI, check if we can update (full day analysis only)
	if (metricType === 'utci' && data.numHours === 1) {
		console.warn('[WARN] Cannot update colors for single hour analysis');
		return;
	}

	const colors = createColors(analysis, hourIndex, colorMode, metricType);
	const colorAttribute = pointCloud.geometry.getAttribute('color') as THREE.BufferAttribute;

	for (let i = 0; i < colors.length / 3; i++) {
		colorAttribute.setXYZ(i, colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]);
	}

	colorAttribute.needsUpdate = true;
}

/**
 * Create a textured ground-aligned mesh that visualizes UTCI or Shading Index values.
 * @param analysis - Analysis data
 * @param hourIndex - Hour index (default: 0)
 * @param colorMode - Color mode ('normalized' or 'discrete')
 * @param metricType - Metric type ('utci' or 'shading_index')
 * @param rangeOverride - Optional UTCI range override for unified comparison scales
 * @param monthIndex - Month index for multi-month (full day = selected month's 24h)
 */
export function createUtciSurfaceMesh(options: UtciSurfaceMeshOptions): THREE.Mesh;
export function createUtciSurfaceMesh(
	analysis: Analysis,
	hourIndex?: number,
	colorMode?: 'normalized' | 'discrete',
	metricType?: MetricType,
	rangeOverride?: UtciRangeOverride,
	monthIndex?: number
): THREE.Mesh;
export function createUtciSurfaceMesh(
	optionsOrAnalysis: Analysis | UtciSurfaceMeshOptions,
	hourIndex: number = 0,
	colorMode: 'normalized' | 'discrete' = 'normalized',
	metricType: MetricType = 'utci',
	rangeOverride?: UtciRangeOverride,
	monthIndex: number = 7
): THREE.Mesh {
	const options = isUtciSurfaceMeshOptions(optionsOrAnalysis)
		? resolveUtciSurfaceMeshOptions(optionsOrAnalysis)
		: resolveUtciSurfaceMeshOptions(
				optionsOrAnalysis,
				hourIndex,
				colorMode,
				metricType,
				rangeOverride,
				monthIndex
			);

	return createUtciSurfaceMeshInternal(options);
}

export function updateUtciSurfaceMesh(mesh: THREE.Mesh, options: UtciSurfaceMeshOptions): boolean {
	const layout: UtciGridLayout | undefined = mesh.userData.utciLayout;
	if (!layout) {
		console.warn('[UTCI] Missing layout on surface mesh. Recreate the mesh.');
		return false;
	}

	const currentBackendType = getSurfaceBackendType(mesh);
	const resolved = resolveUtciSurfaceMeshOptions({
		...options,
		backend: options.backend ?? currentBackendType
	});
	if (resolved.backend !== currentBackendType) {
		console.warn(
			`[UTCI] Surface backend cannot switch from ${currentBackendType} to ${resolved.backend} in place. Recreate the mesh.`
		);
		return false;
	}

	const nextLayout = buildUtciGridLayout(resolved.analysis);
	if (!canReuseUtciSurfaceLayout(layout, nextLayout)) {
		console.warn('[UTCI] Surface layout changed; recreate the mesh to realign surface assets.');
		return false;
	}

	nextLayout.texture = layout.texture;
	const colors = createColors(
		resolved.analysis,
		resolved.hourIndex,
		resolved.colorMode,
		resolved.metricType,
		resolved.rangeOverride,
		resolved.monthIndex
	);

	return getUtciSurfaceBackend(resolved.backend).updateMesh(mesh, nextLayout, colors);
}

/**
 * Update the UTCI texture mesh with new colors for the specified hour/mode/metric.
 * @param mesh - The UTCI surface mesh to update
 * @param analysis - Analysis data
 * @param hourIndex - Hour index
 * @param colorMode - Color mode ('normalized' or 'discrete')
 * @param metricType - Metric type ('utci' or 'shading_index')
 * @param rangeOverride - Optional UTCI range override for unified comparison scales
 */
export function updateUtciSurfaceTexture(
	mesh: THREE.Mesh,
	analysis: Analysis,
	hourIndex: number,
	colorMode: 'normalized' | 'discrete',
	metricType: MetricType = 'utci',
	rangeOverride?: UtciRangeOverride,
	monthIndex: number = 7
): boolean {
	return updateUtciSurfaceMesh(mesh, {
		analysis,
		hourIndex,
		colorMode,
		metricType,
		rangeOverride,
		monthIndex
	});
}

export function disposeUtciSurfaceMesh(mesh: THREE.Mesh | null): void {
	if (!mesh) {
		return;
	}

	getUtciSurfaceBackend(getSurfaceBackendType(mesh)).disposeMesh(mesh);
	delete mesh.userData.utciLayout;
	delete mesh.userData.utciSurfaceBackend;
}

function createUtciSurfaceMeshInternal(options: ResolvedUtciSurfaceMeshOptions): THREE.Mesh {
	const layout = buildUtciGridLayout(options.analysis);
	const colors = createColors(
		options.analysis,
		options.hourIndex,
		options.colorMode,
		options.metricType,
		options.rangeOverride,
		options.monthIndex
	);

	return getUtciSurfaceBackend(options.backend).createMesh(layout, colors);
}

function resolveUtciSurfaceMeshOptions(
	options: UtciSurfaceMeshOptions
): ResolvedUtciSurfaceMeshOptions;
function resolveUtciSurfaceMeshOptions(
	analysis: Analysis,
	hourIndex?: number,
	colorMode?: 'normalized' | 'discrete',
	metricType?: MetricType,
	rangeOverride?: UtciRangeOverride,
	monthIndex?: number
): ResolvedUtciSurfaceMeshOptions;
function resolveUtciSurfaceMeshOptions(
	optionsOrAnalysis: Analysis | UtciSurfaceMeshOptions,
	hourIndex: number = 0,
	colorMode: 'normalized' | 'discrete' = 'normalized',
	metricType: MetricType = 'utci',
	rangeOverride?: UtciRangeOverride,
	monthIndex: number = 7
): ResolvedUtciSurfaceMeshOptions {
	if (isUtciSurfaceMeshOptions(optionsOrAnalysis)) {
		return {
			analysis: optionsOrAnalysis.analysis,
			hourIndex: optionsOrAnalysis.hourIndex ?? 0,
			colorMode: optionsOrAnalysis.colorMode ?? 'normalized',
			metricType: optionsOrAnalysis.metricType ?? 'utci',
			rangeOverride: optionsOrAnalysis.rangeOverride,
			monthIndex: optionsOrAnalysis.monthIndex ?? 7,
			backend: optionsOrAnalysis.backend ?? DEFAULT_UTCI_SURFACE_BACKEND
		};
	}

	return {
		analysis: optionsOrAnalysis,
		hourIndex,
		colorMode,
		metricType,
		rangeOverride,
		monthIndex,
		backend: DEFAULT_UTCI_SURFACE_BACKEND
	};
}

function isUtciSurfaceMeshOptions(
	optionsOrAnalysis: Analysis | UtciSurfaceMeshOptions
): optionsOrAnalysis is UtciSurfaceMeshOptions {
	return 'analysis' in optionsOrAnalysis;
}

function getUtciSurfaceBackend(type: UtciSurfaceBackendType): UtciSurfaceBackend {
	return utciSurfaceBackends[type];
}

function getSurfaceBackendType(mesh: THREE.Mesh): UtciSurfaceBackendType {
	return (mesh.userData.utciSurfaceBackend as UtciSurfaceBackendType | undefined) ?? DEFAULT_UTCI_SURFACE_BACKEND;
}

function canReuseUtciSurfaceLayout(current: UtciGridLayout, next: UtciGridLayout): boolean {
	return (
		current.numPositions === next.numPositions &&
		current.width === next.width &&
		current.height === next.height &&
		current.gridSize === next.gridSize &&
		current.coordinateSystem === next.coordinateSystem
	);
}

function createUtciSurfacePlaneGeometry(layout: UtciGridLayout): THREE.PlaneGeometry {
	const planeWidth = layout.width * layout.gridSize;
	const planeHeight = layout.height * layout.gridSize;
	const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
	geometry.rotateX(-Math.PI / 2);
	return geometry;
}

export function applySurfaceMeshState(
	mesh: THREE.Mesh,
	layout: UtciGridLayout,
	backend: UtciSurfaceBackendType
): void {
	mesh.position.set(layout.centerX, layout.baseY, layout.centerZ);
	mesh.renderOrder = 2;
	if (backend !== 'gpuNative') {
		mesh.frustumCulled = false;
	}
	mesh.userData.utciLayout = layout;
	mesh.userData.utciSurfaceBackend = backend;
	if (backend === 'gpuNative') {
		mesh.userData.utciSurfaceSource = getGpuNativeUtciSurfaceSource(mesh);
		mesh.userData.dataTextureBuildCount = 0;
	} else {
		delete mesh.userData.utciSurfaceSource;
		delete mesh.userData.selectedHourTransferCount;
	}
}

function incrementSelectedHourTransferCount(mesh: THREE.Mesh): void {
	mesh.userData.selectedHourTransferCount =
		((mesh.userData.selectedHourTransferCount as number | undefined) ?? 0) + 1;
}

function incrementDataTextureBuildCount(mesh: THREE.Mesh): void {
	mesh.userData.dataTextureBuildCount =
		((mesh.userData.dataTextureBuildCount as number | undefined) ?? 0) + 1;
}

function disposeSurfaceMeshAssets(mesh: THREE.Mesh): void {
	mesh.removeFromParent();

	const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
	for (const material of materials) {
		(material as THREE.Material & { map?: THREE.Texture | null }).map?.dispose();
		material.dispose();
	}

	mesh.geometry.dispose();
}

function createNormalizationSignature(params: {
	enabled: boolean;
	offset: THREE.Vector3;
}): SelectedHourRenderLayoutNormalizationSignature {
	return {
		enabled: params.enabled,
		offset: {
			x: params.offset.x,
			y: params.offset.y,
			z: params.offset.z
		},
		provenance: params.enabled ? 'anchor-offset-minus-origin' : 'normalization-disabled'
	};
}

function getLayoutNormalizationSignature(
	layout: UtciGridLayout
): SelectedHourRenderLayoutNormalizationSignature {
	return (
		layout.normalizationSignature ?? {
			enabled: false,
			offset: { x: 0, y: 0, z: 0 },
			provenance: 'normalization-disabled'
		}
	);
}

function normalizationSignaturesMatch(
	left: SelectedHourRenderLayoutNormalizationSignature,
	right: SelectedHourRenderLayoutNormalizationSignature
): boolean {
	return (
		left.enabled === right.enabled &&
		left.provenance === right.provenance &&
		left.offset.x === right.offset.x &&
		left.offset.y === right.offset.y &&
		left.offset.z === right.offset.z
	);
}

function estimateRetainedCpuLayoutBytes(layout: UtciGridLayout): number {
	return (
		layout.indexToRow.byteLength +
		layout.indexToColumn.byteLength +
		(layout.cellToPointIndex?.byteLength ?? 0) +
		layout.indexToTexel.byteLength +
		layout.colorBuffer.byteLength
	);
}

function appendHashNumber(hash: number, value: number): number {
	const normalized = Number.isFinite(value) ? Object.is(value, -0) ? 0 : value : 0;
	const encoded = Number.isInteger(normalized)
		? (normalized >>> 0)
		: new Uint32Array(new Float64Array([normalized]).buffer)[0] ?? 0;
	return Math.imul(hash ^ encoded, 16777619) >>> 0;
}

function appendHashString(hash: number, value: string): number {
	let nextHash = hash;
	for (let index = 0; index < value.length; index += 1) {
		nextHash = Math.imul(nextHash ^ value.charCodeAt(index), 16777619) >>> 0;
	}
	return nextHash;
}

function appendHashUint32Array(hash: number, values: Uint32Array): number {
	let nextHash = hash;
	for (let index = 0; index < values.length; index += 1) {
		nextHash = Math.imul(nextHash ^ values[index], 16777619) >>> 0;
	}
	return nextHash;
}

function appendHashFloat32Array(hash: number, values: Float32Array): number {
	return appendHashUint32Array(hash, new Uint32Array(values.buffer, values.byteOffset, values.length));
}

function appendHashInt32Array(hash: number, values: Int32Array): number {
	let nextHash = hash;
	for (let index = 0; index < values.length; index += 1) {
		nextHash = Math.imul(nextHash ^ (values[index] >>> 0), 16777619) >>> 0;
	}
	return nextHash;
}

type DerivedUtciLayoutFrame = {
	gridSize: number;
	coordinateSystem: 'xy_ground' | 'xz_ground';
	pointCount: number;
	width: number;
	height: number;
	minX: number;
	minZ: number;
	minY: number;
	maxY: number;
	centerX: number;
	centerZ: number;
	baseY: number;
	constructionMode: 'world-positions' | 'metadata-bounds-fallback';
	normalizationSignature: SelectedHourRenderLayoutNormalizationSignature;
};

type UtciLayoutBaseParams = {
	gridSize: number;
	coordinateSystem: 'xy_ground' | 'xz_ground';
	numPositions: number;
	normalizationOffset: THREE.Vector3;
};

function resolveUtciLayoutBaseParams(analysis: Analysis): UtciLayoutBaseParams {
	const coordinateSystem = analysis.metadata.coordinate_system || 'xy_ground';
	const gridSize = analysis.metadata.grid_size || 1;
	const numPositions = analysis.data.numPositions;
	const normalizationEnabled = isNormalizationEnabled();
	let normalizationOffset = new THREE.Vector3(0, 0, 0);

	if (normalizationEnabled) {
		const scenarioOrigin = calculateScenarioOrigin(analysis.metadata as any);
		const anchorOffset = getAnchorOffset();

		let transformedOrigin: THREE.Vector3;
		if (coordinateSystem === 'xy_ground') {
			transformedOrigin = new THREE.Vector3(
				scenarioOrigin.x,
				scenarioOrigin.z,
				-scenarioOrigin.y
			);
		} else {
			transformedOrigin = scenarioOrigin.clone();
		}

		normalizationOffset = anchorOffset.clone().sub(transformedOrigin);
		if (normalizationOffset.lengthSq() <= 0.001) {
			normalizationOffset.set(0, 0, 0);
		}
	}

	return {
		gridSize,
		coordinateSystem,
		numPositions,
		normalizationOffset
	};
}

function deriveUtciLayoutFrame(analysis: Analysis): DerivedUtciLayoutFrame {
	const { data, metadata } = analysis;
	const { gridSize, coordinateSystem, numPositions, normalizationOffset } =
		resolveUtciLayoutBaseParams(analysis);
	const transformed = new THREE.Vector3();

	let minX = Infinity;
	let minZ = Infinity;
	let maxX = -Infinity;
	let maxZ = -Infinity;
	let minY = Infinity;
	let maxY = -Infinity;

	for (let i = 0; i < numPositions; i += 1) {
		const x = data.positions[i * 3];
		const y = data.positions[i * 3 + 1];
		const z = data.positions[i * 3 + 2];

		transformToWorld(x, y, z, coordinateSystem, transformed);
		transformed.add(normalizationOffset);

		if (transformed.x < minX) minX = transformed.x;
		if (transformed.x > maxX) maxX = transformed.x;
		if (transformed.z < minZ) minZ = transformed.z;
		if (transformed.z > maxZ) maxZ = transformed.z;
		if (transformed.y < minY) minY = transformed.y;
		if (transformed.y > maxY) maxY = transformed.y;
	}

	let width = 1;
	let height = 1;
	let usingBoundsFallback = false;
	let layoutValid =
		Number.isFinite(minX) &&
		Number.isFinite(maxX) &&
		Number.isFinite(minZ) &&
		Number.isFinite(maxZ) &&
		Number.isFinite(minY) &&
		Number.isFinite(maxY);

	const bounds = metadata.bounds as
		| { x_min: number; x_max: number; y_min: number; y_max: number; z?: number }
		| undefined;
	if (!layoutValid && bounds && (analysis as { __source?: string }).__source === 'webgpu') {
		usingBoundsFallback = true;
		if (coordinateSystem === 'xy_ground') {
			minX = bounds.x_min;
			maxX = bounds.x_max;
			minZ = -bounds.y_max;
			maxZ = -bounds.y_min;
			minY = maxY = bounds.z ?? 0;
		} else {
			minX = bounds.x_min;
			maxX = bounds.x_max;
			minZ = bounds.y_min;
			maxZ = bounds.y_max;
			minY = maxY = bounds.z ?? 0;
		}
		const spanX = maxX - minX;
		const spanZ = maxZ - minZ;
		width = Math.max(1, Math.round(spanX / gridSize) + 1);
		height = Math.max(1, Math.round(spanZ / gridSize) + 1);
		layoutValid = true;
	} else if (layoutValid) {
		for (let i = 0; i < numPositions; i += 1) {
			const x = data.positions[i * 3];
			const y = data.positions[i * 3 + 1];
			const z = data.positions[i * 3 + 2];

			transformToWorld(x, y, z, coordinateSystem, transformed);
			transformed.add(normalizationOffset);

			const col = Math.round((transformed.x - minX) / gridSize);
			const row = Math.round((transformed.z - minZ) / gridSize);
			if (col + 1 > width) width = col + 1;
			if (row + 1 > height) height = row + 1;
		}
	}

	const centerX = layoutValid ? minX + ((width - 1) * gridSize) / 2 : 0;
	const centerZ = layoutValid ? minZ + ((height - 1) * gridSize) / 2 : 0;
	const baseY = layoutValid ? minY + VISUAL_LAYER_OFFSET : 0;

	return {
		gridSize,
		coordinateSystem,
		pointCount: numPositions,
		width,
		height,
		minX,
		minZ,
		minY,
		maxY,
		centerX,
		centerZ,
		baseY,
		constructionMode: usingBoundsFallback
			? 'metadata-bounds-fallback'
			: 'world-positions',
		normalizationSignature: createNormalizationSignature({
			enabled: isNormalizationEnabled(),
			offset: normalizationOffset
		})
	};
}

function getAnalysisIdentity(analysis: Analysis): string {
	return (
		analysis.metadata.source_analysis_id ??
		analysis.metadata.model_file ??
		'unknown-analysis'
	);
}

function getPositionsSourceSignature(
	positions: Float32Array,
	diagnostics?: UtciLayoutReuseKeyDiagnostics
): string {
	const startedAt = diagnostics ? performance.now() : undefined;
	const cached = positionsSourceSignatureCache.get(positions);
	if (cached !== undefined) {
		if (diagnostics && startedAt !== undefined) {
			diagnostics.positionsSourceSignatureCacheHit = true;
			diagnostics.positionsSourceSignatureMs = performance.now() - startedAt;
		}
		return cached;
	}

	positionsSourceSignatureComputationCount += 1;
	const hash = appendHashFloat32Array(2166136261, positions);
	const signature = `pos:${hash.toString(16)}`;
	positionsSourceSignatureCache.set(positions, signature);
	if (diagnostics && startedAt !== undefined) {
		diagnostics.positionsSourceSignatureCacheHit = false;
		diagnostics.positionsSourceSignatureMs = performance.now() - startedAt;
	}
	return signature;
}

function getLayoutSourceSignature(
	analysis: Analysis,
	diagnostics?: UtciLayoutReuseKeyDiagnostics
): string {
	const startedAt = diagnostics ? performance.now() : undefined;
	let hash = 2166136261;
	hash = appendHashString(hash, (analysis as { __source?: string }).__source ?? 'loaded');
	hash = appendHashString(hash, getAnalysisIdentity(analysis));
	hash = appendHashString(
		hash,
		getPositionsSourceSignature(analysis.data.positions, diagnostics)
	);
	const bounds = analysis.metadata.bounds;
	if (bounds) {
		hash = appendHashNumber(hash, bounds.x_min);
		hash = appendHashNumber(hash, bounds.x_max);
		hash = appendHashNumber(hash, bounds.y_min);
		hash = appendHashNumber(hash, bounds.y_max);
		hash = appendHashNumber(hash, bounds.z ?? 0);
	}
	const signature = `v1:${hash.toString(16)}`;
	if (diagnostics && startedAt !== undefined) {
		diagnostics.layoutSourceSignatureMs = performance.now() - startedAt;
	}
	return signature;
}

export function getUtciLayoutReuseSignatureDiagnosticsForTest(): {
	positionsSourceSignatureComputationCount: number;
} {
	return {
		positionsSourceSignatureComputationCount
	};
}

export function resetUtciLayoutFrameCachesForTest(): void {
	utciLayoutFrameCache = new WeakMap();
	structuralUtciLayoutFrameCache.clear();
	positionsSourceSignatureCache = new WeakMap();
}

export function getUtciLayoutFrameCacheDiagnosticsForTest(): {
	structuralFrameCacheSize: number;
	structuralFrameCacheLimit: number;
} {
	return {
		structuralFrameCacheSize: structuralUtciLayoutFrameCache.size,
		structuralFrameCacheLimit: STRUCTURAL_FRAME_CACHE_LIMIT
	};
}

export function deriveUtciLayoutFrameForTest(analysis: Analysis): DerivedUtciLayoutFrame {
	return deriveUtciLayoutFrame(analysis);
}

function getStructuralFrameCachePreKey(params: {
	analysis: Analysis;
	layoutSourceSignature: string;
	gridSize: number;
	pointCount: number;
	coordinateSystem: string;
	normalizationSignature: SelectedHourRenderLayoutNormalizationSignature;
}): string {
	return [
		params.layoutSourceSignature,
		params.gridSize,
		params.pointCount,
		params.coordinateSystem,
		serializeNormalizationSignature(params.normalizationSignature)
	].join('|');
}

function isDerivedUtciLayoutFrameCompatibleWithBaseParams(
	frame: DerivedUtciLayoutFrame,
	params: {
		gridSize: number;
		coordinateSystem: string;
		pointCount: number;
		normalizationSignature: SelectedHourRenderLayoutNormalizationSignature;
	}
): boolean {
	return (
		frame.gridSize === params.gridSize &&
		frame.coordinateSystem === params.coordinateSystem &&
		frame.pointCount === params.pointCount &&
		serializeNormalizationSignature(frame.normalizationSignature) ===
			serializeNormalizationSignature(params.normalizationSignature)
	);
}

function deriveCachedUtciLayoutFrame(
	analysis: Analysis,
	diagnostics?: UtciLayoutReuseKeyDiagnostics
): {
	frame: DerivedUtciLayoutFrame;
	layoutSourceSignature: string;
} {
	const keyBuildStartedAt = performance.now();
	const layoutSourceSignature = getLayoutSourceSignature(analysis, diagnostics);
	const lookupStartedAt = performance.now();
	const objectCached = utciLayoutFrameCache.get(analysis);
	if (objectCached && objectCached.layoutSourceSignature === layoutSourceSignature) {
		if (diagnostics) {
			diagnostics.frameCacheHit = true;
			diagnostics.frameCacheKind = 'analysis-object';
			diagnostics.frameCacheLookupMs = performance.now() - lookupStartedAt;
			diagnostics.frameDerivationMs = 0;
			diagnostics.keyBuildMs = performance.now() - keyBuildStartedAt;
		}
		return {
			frame: objectCached.frame,
			layoutSourceSignature
		};
	}

	const { gridSize, coordinateSystem, numPositions, normalizationOffset } =
		resolveUtciLayoutBaseParams(analysis);
	const normalizationSignature = createNormalizationSignature({
		enabled: isNormalizationEnabled(),
		offset: normalizationOffset
	});
	const structuralPreKey = getStructuralFrameCachePreKey({
		analysis,
		layoutSourceSignature,
		gridSize,
		pointCount: numPositions,
		coordinateSystem,
		normalizationSignature
	});

	const structuralCached = structuralUtciLayoutFrameCache.get(structuralPreKey);
	if (structuralCached && structuralCached.layoutSourceSignature === layoutSourceSignature) {
		if (
			isDerivedUtciLayoutFrameCompatibleWithBaseParams(structuralCached.frame, {
				gridSize,
				coordinateSystem,
				pointCount: numPositions,
				normalizationSignature
			})
		) {
			utciLayoutFrameCache.set(analysis, structuralCached);
			if (diagnostics) {
				diagnostics.frameCacheHit = true;
				diagnostics.frameCacheKind = 'structural';
				diagnostics.frameCacheLookupMs = performance.now() - lookupStartedAt;
				diagnostics.frameDerivationMs = 0;
				diagnostics.keyBuildMs = performance.now() - keyBuildStartedAt;
			}
			return {
				frame: structuralCached.frame,
				layoutSourceSignature
			};
		}
		structuralUtciLayoutFrameCache.delete(structuralPreKey);
	}

	const frameStartedAt = performance.now();
	const frame = deriveUtciLayoutFrame(analysis);
	const frameDerivationMs = performance.now() - frameStartedAt;
	const entry = {
		layoutSourceSignature,
		frame
	};
	utciLayoutFrameCache.set(analysis, entry);
	rememberStructuralUtciLayoutFrame(structuralPreKey, entry);
	if (diagnostics) {
		diagnostics.frameCacheHit = false;
		diagnostics.frameCacheKind = 'miss';
		diagnostics.frameCacheLookupMs = performance.now() - lookupStartedAt;
		diagnostics.frameDerivationMs = frameDerivationMs;
		diagnostics.keyBuildMs = performance.now() - keyBuildStartedAt;
	}
	return {
		frame,
		layoutSourceSignature
	};
}

function serializeNormalizationSignature(
	signature: SelectedHourRenderLayoutNormalizationSignature
): string {
	return [
		signature.enabled ? '1' : '0',
		signature.provenance,
		signature.offset.x,
		signature.offset.y,
		signature.offset.z
	].join('|');
}

export function areUtciLayoutReuseKeysEqual(
	left: UtciLayoutReuseKey,
	right: UtciLayoutReuseKey
): boolean {
	return (
		left.analysisId === right.analysisId &&
		left.layoutSourceSignature === right.layoutSourceSignature &&
		left.gridSize === right.gridSize &&
		left.pointCount === right.pointCount &&
		left.coordinateSystem === right.coordinateSystem &&
		left.normalizationSignature === right.normalizationSignature &&
		left.constructionMode === right.constructionMode &&
		left.width === right.width &&
		left.height === right.height &&
		left.centerX === right.centerX &&
		left.centerZ === right.centerZ &&
		left.baseY === right.baseY &&
		left.utciSurfaceSource === right.utciSurfaceSource &&
		left.rendererBackend === right.rendererBackend
	);
}

export function createUtciLayoutReuseKey(params: {
	analysis: Analysis;
	layout: UtciGridLayout;
	utciSurfaceSource: 'compute-buffer-selected-hour';
	rendererBackend: 'webgpu';
}): UtciLayoutReuseKey {
	const constructionMode = params.layout.constructionMode ?? 'invalid';
	return {
		analysisId: getAnalysisIdentity(params.analysis),
		layoutSourceSignature: getLayoutSourceSignature(params.analysis),
		gridSize: params.layout.gridSize,
		pointCount: params.layout.numPositions,
		coordinateSystem: params.layout.coordinateSystem,
		normalizationSignature: serializeNormalizationSignature(
			getLayoutNormalizationSignature(params.layout)
		),
		constructionMode,
		width: params.layout.width,
		height: params.layout.height,
		centerX: params.layout.centerX,
		centerZ: params.layout.centerZ,
		baseY: params.layout.baseY,
		utciSurfaceSource: params.utciSurfaceSource,
		rendererBackend: params.rendererBackend
	};
}

export function createUtciLayoutReuseKeyForAnalysis(params: {
	analysis: Analysis;
	utciSurfaceSource: 'compute-buffer-selected-hour';
	rendererBackend: 'webgpu';
	diagnostics?: UtciLayoutReuseKeyDiagnostics;
}): UtciLayoutReuseKey {
	const { frame, layoutSourceSignature } = deriveCachedUtciLayoutFrame(
		params.analysis,
		params.diagnostics
	);
	return {
		analysisId: getAnalysisIdentity(params.analysis),
		layoutSourceSignature,
		gridSize: frame.gridSize,
		pointCount: frame.pointCount,
		coordinateSystem: frame.coordinateSystem,
		normalizationSignature: serializeNormalizationSignature(
			frame.normalizationSignature
		),
		constructionMode: frame.constructionMode,
		width: frame.width,
		height: frame.height,
		centerX: frame.centerX,
		centerZ: frame.centerZ,
		baseY: frame.baseY,
		utciSurfaceSource: params.utciSurfaceSource,
		rendererBackend: params.rendererBackend
	};
}

export function getUtciLayoutIdentity(key: UtciLayoutReuseKey): string {
	return [
		key.analysisId,
		key.layoutSourceSignature,
		key.width,
		key.height,
		key.centerX,
		key.centerZ,
		key.baseY
	].join('|');
}

export function createUtciLayoutReusePublicationState(params: {
	proof: UtciGridLayoutReuseProofDiagnostics;
	key: UtciLayoutReuseKey;
	requestId: number | null;
	selectionKey: string | null;
}): UtciLayoutReusePublicationState {
	return {
		proof: params.proof,
		key: params.key,
		layoutIdentity: getUtciLayoutIdentity(params.key),
		requestId: params.requestId,
		selectionKey: params.selectionKey
	};
}

export function resolveUtciLayoutReusePublicationStateAfterSync(params: {
	currentState: UtciLayoutReusePublicationState | null;
	pendingState: UtciLayoutReusePublicationState;
	syncResult: 'complete' | 'failed' | 'superseded' | 'already-released';
}): UtciLayoutReusePublicationState | null {
	return params.syncResult === 'complete'
		? params.pendingState
		: params.currentState;
}

export function isUtciLayoutReuseProofSafe(
	proof: UtciGridLayoutReuseProofDiagnostics | null | undefined
): boolean {
	return (
		proof?.decision === 'reuse-safe' &&
		proof.canonicalRuntimeCompatibilityWouldReuse === true &&
		proof.proofMatchesCanonicalRuntimeCompatibility === true &&
		proof.constructionModeMatch === true &&
		proof.dimensionsMatch === true &&
		proof.placementMatch === true &&
		proof.cellToPointMappingMatch === true &&
		proof.hoverCellLookupProofStatus === 'same-point-confirmed'
	);
}

export function planUtciLayoutReuseCandidate(params: {
	previousLayout: UtciGridLayout | null;
	proof: UtciGridLayoutReuseProofDiagnostics | null;
	previousKey: UtciLayoutReuseKey | null;
	currentKey: UtciLayoutReuseKey;
}): UtciLayoutReuseDecision {
	if (!params.previousLayout) {
		return { action: 'build-required', reason: 'missing-previous-layout', keyMatch: false };
	}
	if (!params.proof || !params.previousKey) {
		return { action: 'build-required', reason: 'diagnostics-missing', keyMatch: false };
	}
	if (
		params.previousKey.utciSurfaceSource !== params.currentKey.utciSurfaceSource ||
		params.previousKey.rendererBackend !== params.currentKey.rendererBackend
	) {
		return { action: 'build-required', reason: 'backend-or-source-mismatch', keyMatch: false };
	}
	const keyMatch = areUtciLayoutReuseKeysEqual(params.previousKey, params.currentKey);
	if (!keyMatch) {
		return { action: 'build-required', reason: 'layout-key-mismatch', keyMatch };
	}
	if (
		params.proof.canonicalRuntimeCompatibilityWouldReuse !== true ||
		params.proof.proofMatchesCanonicalRuntimeCompatibility !== true
	) {
		return { action: 'build-required', reason: 'canonical-mismatch', keyMatch };
	}
	if (params.proof.cellToPointMappingMatch !== true) {
		return { action: 'build-required', reason: 'mapping-unsafe', keyMatch };
	}
	if (params.proof.hoverCellLookupProofStatus !== 'same-point-confirmed') {
		return { action: 'build-required', reason: 'hover-proof-missing', keyMatch };
	}
	if (!isUtciLayoutReuseProofSafe(params.proof)) {
		return { action: 'build-required', reason: 'proof-not-safe', keyMatch };
	}
	return { action: 'reuse-candidate', reason: 'reuse-safe', keyMatch };
}

export function planUtciLayoutPublication(params: {
	previousLayout: UtciGridLayout | null;
	previousProof: UtciGridLayoutReuseProofDiagnostics | null;
	previousKey: UtciLayoutReuseKey | null;
	currentKey: UtciLayoutReuseKey;
	currentSurfaceSource: string | null;
	currentRendererBackend: string | null;
	publicationPhase: 'initial' | 'scrub';
}): UtciLayoutPublicationPlan {
	if (params.publicationPhase !== 'scrub') {
		return {
			action: 'build-new',
			reason: 'initial-publication',
			keyMatch: false
		};
	}
	if (
		params.currentSurfaceSource !== 'compute-buffer-selected-hour' ||
		params.currentRendererBackend !== 'webgpu'
	) {
		return {
			action: 'build-new',
			reason: 'backend-or-source-mismatch',
			keyMatch: false
		};
	}

	const candidate = planUtciLayoutReuseCandidate({
		previousLayout: params.previousLayout,
		proof: params.previousProof,
		previousKey: params.previousKey,
		currentKey: params.currentKey
	});
	if (candidate.action !== 'reuse-candidate' || !params.previousLayout) {
		return {
			action: 'build-new',
			reason: candidate.reason,
			keyMatch: candidate.keyMatch
		};
	}

	return {
		action: 'reuse-existing',
		layout: params.previousLayout,
		reason: 'reuse-safe',
		keyMatch: true
	};
}

export function buildUtciGridLayoutReuseProofDiagnostics(params: {
	previousLayout?: UtciGridLayout | null;
	nextLayout: UtciGridLayout;
	skipExpensiveMappingComparison?: boolean;
	canonicalRuntimeCompatibilityWouldReuse?: boolean | null;
	canonicalPointCompatibility?: UtciGridLayoutPointCompatibilityEvaluation | null;
}): UtciGridLayoutReuseProofDiagnostics {
	const startedAt = performance.now();
	const previousLayout = params.previousLayout ?? null;
	const nextNormalizationSignature = getLayoutNormalizationSignature(params.nextLayout);
	const nextConstructionMode =
		params.nextLayout.constructionMode ?? 'world-positions';
	const proof: UtciGridLayoutReuseProofDiagnostics = {
		decision: 'rebuild-required',
		hoverCellLookupProofStatus: 'proof-inconclusive',
		previousLayoutPresent: previousLayout != null,
		canonicalRuntimeCompatibilityWouldReuse: null,
		proofMatchesCanonicalRuntimeCompatibility: null,
		positionsReferenceMatch: null,
		pointCountMatch: null,
		gridSizeMatch: null,
		coordinateSystemMatch: null,
		normalizationSignature: nextNormalizationSignature,
		previousNormalizationSignature: previousLayout
			? getLayoutNormalizationSignature(previousLayout)
			: null,
		normalizationSignatureMatch: null,
		constructionMode: nextConstructionMode,
		previousConstructionMode: previousLayout?.constructionMode ?? null,
		constructionModeMatch: null,
		dimensionsMatch: null,
		placementMatch: null,
		cellToPointMappingMatch: null,
		proofCostMs: null,
		estimatedRetainedCpuLayoutBytes: estimateRetainedCpuLayoutBytes(params.nextLayout)
	};

	if (!previousLayout) {
		proof.proofCostMs = performance.now() - startedAt;
		return proof;
	}

	const positionsReferenceMatch =
		previousLayout.positionsIdentityId != null &&
		params.nextLayout.positionsIdentityId != null
			? previousLayout.positionsIdentityId === params.nextLayout.positionsIdentityId
			: null;
	const pointCountMatch = previousLayout.numPositions === params.nextLayout.numPositions;
	const gridSizeMatch = previousLayout.gridSize === params.nextLayout.gridSize;
	const coordinateSystemMatch =
		previousLayout.coordinateSystem === params.nextLayout.coordinateSystem;
	const normalizationSignatureMatch = normalizationSignaturesMatch(
		getLayoutNormalizationSignature(previousLayout),
		nextNormalizationSignature
	);
	const constructionModeMatch =
		(previousLayout.constructionMode ?? 'world-positions') === nextConstructionMode;
	const dimensionsMatch =
		previousLayout.width === params.nextLayout.width &&
		previousLayout.height === params.nextLayout.height;
	const placementMatch =
		previousLayout.centerX === params.nextLayout.centerX &&
		previousLayout.centerZ === params.nextLayout.centerZ &&
		previousLayout.baseY === params.nextLayout.baseY;
	const layoutPointCompatibility =
		params.canonicalPointCompatibility ??
		evaluateUtciGridLayoutsPointCompatibility(previousLayout, params.nextLayout, {
			allowExpensiveMappingComparison: !params.skipExpensiveMappingComparison
		});
	const runtimeCompatibilityWouldReuse =
		params.canonicalRuntimeCompatibilityWouldReuse ?? null;
	const cellToPointMappingMatch = layoutPointCompatibility.cellToPointMappingMatch;

	proof.positionsReferenceMatch = positionsReferenceMatch;
	proof.pointCountMatch = pointCountMatch;
	proof.gridSizeMatch = gridSizeMatch;
	proof.coordinateSystemMatch = coordinateSystemMatch;
	proof.normalizationSignatureMatch = normalizationSignatureMatch;
	proof.constructionModeMatch = constructionModeMatch;
	proof.dimensionsMatch = dimensionsMatch;
	proof.placementMatch = placementMatch;
	proof.cellToPointMappingMatch = cellToPointMappingMatch;
	proof.canonicalRuntimeCompatibilityWouldReuse =
		runtimeCompatibilityWouldReuse;

	const proofWouldReuse =
		pointCountMatch &&
		gridSizeMatch &&
		coordinateSystemMatch &&
		normalizationSignatureMatch &&
		constructionModeMatch &&
		dimensionsMatch &&
		placementMatch &&
		cellToPointMappingMatch === true;
	const proofDefinitelyRejects =
		pointCountMatch === false ||
		gridSizeMatch === false ||
		coordinateSystemMatch === false ||
		normalizationSignatureMatch === false ||
		constructionModeMatch === false ||
		dimensionsMatch === false ||
		placementMatch === false ||
		cellToPointMappingMatch === false;

	if (
		typeof runtimeCompatibilityWouldReuse === 'boolean' &&
		typeof proofWouldReuse === 'boolean'
	) {
		proof.proofMatchesCanonicalRuntimeCompatibility =
			proofWouldReuse === runtimeCompatibilityWouldReuse;
	}

	if (proofWouldReuse && runtimeCompatibilityWouldReuse === true) {
		proof.decision = 'reuse-safe';
	} else if (
		proofDefinitelyRejects ||
		runtimeCompatibilityWouldReuse === false
	) {
		proof.decision = 'rebuild-required';
	} else {
		proof.decision = 'proof-inconclusive';
	}

	if (
		proof.decision === 'reuse-safe' &&
		proof.proofMatchesCanonicalRuntimeCompatibility === false
	) {
		proof.decision = 'proof-inconclusive';
	}

	proof.hoverCellLookupProofStatus =
		proof.decision === 'reuse-safe'
			? 'same-point-confirmed'
			: proof.decision === 'rebuild-required'
				? 'not-compatible'
				: 'proof-inconclusive';

	proof.proofCostMs = performance.now() - startedAt;
	return proof;
}

export function buildUtciGridLayout(
	analysis: Analysis,
	options?: {
		diagnostics?: UtciGridLayoutBuildDiagnostics;
	}
): UtciGridLayout {
	const { data, metadata } = analysis;
	const coordinateSystem = metadata.coordinate_system || 'xy_ground';
	const gridSize = metadata.grid_size || 1;
	const numPositions = data.numPositions;
	const diagnostics = options?.diagnostics;
	const startedAt = diagnostics ? performance.now() : undefined;
	const normalizationEnabled = isNormalizationEnabled();

	const allocationStartedAt = diagnostics ? performance.now() : undefined;
	const xs = new Float32Array(numPositions);
	const ys = new Float32Array(numPositions);
	const zs = new Float32Array(numPositions);

	const indexToTexel = new Uint32Array(numPositions);
	const rows = new Uint32Array(numPositions);
	const cols = new Uint32Array(numPositions);
	if (diagnostics && allocationStartedAt !== undefined) {
		diagnostics.arrayAllocationMs = performance.now() - allocationStartedAt;
	}

	let minX = Infinity;
	let minZ = Infinity;
	let maxX = -Infinity;
	let maxZ = -Infinity;
	let minY = Infinity;
	let maxY = -Infinity;

	const transformed = new THREE.Vector3();

	// Calculate normalization offset if enabled
	let normalizationOffset = new THREE.Vector3(0, 0, 0);
	if (normalizationEnabled) {
		const scenarioOrigin = calculateScenarioOrigin(metadata as any);
		const anchorOffset = getAnchorOffset();
		
		// Transform scenario origin to world space to match the coordinate system
		// For xy_ground: transformToWorld does (x, y, z) → (x, z, -y)
		let transformedOrigin: THREE.Vector3;
		if (coordinateSystem === 'xy_ground') {
			// Transform origin to world space: (x, y, z) → (x, z, -y)
			transformedOrigin = new THREE.Vector3(scenarioOrigin.x, scenarioOrigin.z, -scenarioOrigin.y);
		} else {
			transformedOrigin = scenarioOrigin.clone();
		}
		
		// Calculate offset in world space (where anchorOffset already is)
		normalizationOffset = anchorOffset.clone().sub(transformedOrigin);
		
		if (normalizationOffset.lengthSq() > 0.001) {
			console.log(`[UTCI] Applying normalization offset to analysis data:`, normalizationOffset);
		} else {
			normalizationOffset.set(0, 0, 0);
		}
	}

	const transformBoundsStartedAt = diagnostics ? performance.now() : undefined;
	for (let i = 0; i < numPositions; i++) {
		const x = data.positions[i * 3];
		const y = data.positions[i * 3 + 1];
		const z = data.positions[i * 3 + 2];

		transformToWorld(x, y, z, coordinateSystem, transformed);
		
		// Apply normalization offset
		transformed.add(normalizationOffset);

		xs[i] = transformed.x;
		ys[i] = transformed.y;
		zs[i] = transformed.z;

		if (transformed.x < minX) minX = transformed.x;
		if (transformed.x > maxX) maxX = transformed.x;
		if (transformed.z < minZ) minZ = transformed.z;
		if (transformed.z > maxZ) maxZ = transformed.z;
		if (transformed.y < minY) minY = transformed.y;
		if (transformed.y > maxY) maxY = transformed.y;
	}
	if (diagnostics && transformBoundsStartedAt !== undefined) {
		diagnostics.transformBoundsPassMs = performance.now() - transformBoundsStartedAt;
	}

	let width = 1;
	let height = 1;
	let usingBoundsFallback = false;

	// Guard against invalid layout (e.g. NaN positions or wrong coordinate space)
	let layoutValid =
		Number.isFinite(minX) &&
		Number.isFinite(maxX) &&
		Number.isFinite(minZ) &&
		Number.isFinite(maxZ) &&
		Number.isFinite(minY) &&
		Number.isFinite(maxY);

	// Fallback for live/WebGPU rectangular grid: use metadata.bounds so overlay is placed in scene
	const bounds = metadata.bounds as { x_min: number; x_max: number; y_min: number; y_max: number; z?: number } | undefined;
	if (!layoutValid && bounds && (analysis as any).__source === 'webgpu') {
		usingBoundsFallback = true;
		if (coordinateSystem === 'xy_ground') {
			minX = bounds.x_min;
			maxX = bounds.x_max;
			minZ = -bounds.y_max;
			maxZ = -bounds.y_min;
			minY = maxY = bounds.z ?? 0;
		} else {
			minX = bounds.x_min;
			maxX = bounds.x_max;
			minZ = bounds.y_min;
			maxZ = bounds.y_max;
			minY = maxY = bounds.z ?? 0;
		}
		// Recompute width/height from bounds so plane size matches grid
		const spanX = maxX - minX;
		const spanZ = maxZ - minZ;
		width = Math.max(1, Math.round(spanX / gridSize) + 1);
		height = Math.max(1, Math.round(spanZ / gridSize) + 1);
		layoutValid = true;
		console.log('[UTCI] Using metadata.bounds fallback for live overlay placement.', { minX, maxX, minZ, maxZ, width, height });
	} else if (!layoutValid) {
		const source = (analysis as any).__source;
		console.warn(
			'[UTCI] Invalid grid layout (non-finite bounds) – overlay may not show. Source:',
			source ?? 'unknown',
			{ minX, maxX, minZ, maxZ, minY, maxY }
		);
	}

	if (layoutValid) {
		const coordinateAssignmentStartedAt = diagnostics ? performance.now() : undefined;
		if (usingBoundsFallback) {
			assignFallbackGridCoordinates(rows, cols, numPositions, width, height);
		} else {
			const dimensions = assignGridCoordinatesFromWorldPositions({
				xs,
				zs,
				minX,
				minZ,
				gridSize,
				rows,
				cols,
				numPositions
			});
			width = dimensions.width;
			height = dimensions.height;
		}
		if (diagnostics && coordinateAssignmentStartedAt !== undefined) {
			diagnostics.coordinateAssignmentMs =
				performance.now() - coordinateAssignmentStartedAt;
		}
	}
	const normalizationSignature = createNormalizationSignature({
		enabled: normalizationEnabled,
		offset: normalizationOffset
	});

	const indexToTexelStartedAt = diagnostics ? performance.now() : undefined;
	for (let i = 0; i < numPositions; i++) {
		const flippedRow = height - 1 - rows[i];
		const texelIndex = flippedRow * width + cols[i];
		indexToTexel[i] = texelIndex;
	}
	if (diagnostics && indexToTexelStartedAt !== undefined) {
		diagnostics.indexToTexelFillMs = performance.now() - indexToTexelStartedAt;
	}

	const cellToPointIndexStartedAt = diagnostics ? performance.now() : undefined;
	const cellToPointIndex = createCellToPointIndex({
		rows,
		cols,
		numPositions,
		width,
		height,
		layoutValid
	});
	if (diagnostics && cellToPointIndexStartedAt !== undefined) {
		diagnostics.cellToPointIndexBuildMs = performance.now() - cellToPointIndexStartedAt;
	}
	const colorBufferAllocationStartedAt = diagnostics ? performance.now() : undefined;
	const colorBuffer = new Uint8Array(width * height * 4);
	if (
		diagnostics &&
		colorBufferAllocationStartedAt !== undefined &&
		startedAt !== undefined
	) {
		diagnostics.colorBufferAllocationMs =
			performance.now() - colorBufferAllocationStartedAt;
		diagnostics.totalMs = performance.now() - startedAt;
	}
	// centerX/Z and baseY are in viewer space: we already applied transformToWorld(analysis)+normalizationOffset above to get min/max.
	const centerX = layoutValid ? minX + ((width - 1) * gridSize) / 2 : 0;
	const centerZ = layoutValid ? minZ + ((height - 1) * gridSize) / 2 : 0;
	const baseY = layoutValid ? minY + VISUAL_LAYER_OFFSET : 0;

	return {
		width,
		height,
		gridSize,
		coordinateSystem,
		numPositions,
		minX,
		minZ,
		minY,
		maxY,
		centerX,
		centerZ,
		baseY,
		indexToRow: rows,
		indexToColumn: cols,
		cellToPointIndex,
		indexToTexel,
		colorBuffer,
		positionsIdentityId: getPositionsIdentityId(data.positions),
		constructionMode: usingBoundsFallback
			? 'metadata-bounds-fallback'
			: 'world-positions',
		normalizationSignature
	};
}

function assignGridCoordinatesFromWorldPositions(params: {
	xs: Float32Array;
	zs: Float32Array;
	minX: number;
	minZ: number;
	gridSize: number;
	rows: Uint32Array;
	cols: Uint32Array;
	numPositions: number;
}): { width: number; height: number } {
	const { xs, zs, minX, minZ, gridSize, rows, cols, numPositions } = params;
	let maxRow = 0;
	let maxCol = 0;
	const invGrid = 1 / gridSize;

	for (let i = 0; i < numPositions; i++) {
		const col = Math.round((xs[i] - minX) * invGrid);
		const row = Math.round((zs[i] - minZ) * invGrid);

		cols[i] = col;
		rows[i] = row;

		if (col > maxCol) maxCol = col;
		if (row > maxRow) maxRow = row;
	}

	return {
		width: Math.max(1, maxCol + 1),
		height: Math.max(1, maxRow + 1)
	};
}

function assignFallbackGridCoordinates(
	rows: Uint32Array,
	cols: Uint32Array,
	numPositions: number,
	width: number,
	height: number
): void {
	const expectedPositions = width * height;
	if (numPositions !== expectedPositions) {
		console.warn(
			`[UTCI] WebGPU bounds fallback expected ${expectedPositions} grid points but received ${numPositions}. Mapping sequentially within the fallback grid.`
		);
	}

	for (let i = 0; i < numPositions; i++) {
		const col = Math.min(width - 1, Math.floor(i / height));
		const row = Math.min(height - 1, i % height);
		cols[i] = col;
		rows[i] = row;
	}
}

function createCellToPointIndex(params: {
	rows: Uint32Array;
	cols: Uint32Array;
	numPositions: number;
	width: number;
	height: number;
	layoutValid: boolean;
}): Int32Array {
	const { rows, cols, numPositions, width, height, layoutValid } = params;
	const cellToPointIndex = new Int32Array(width * height);
	cellToPointIndex.fill(-1);

	if (!layoutValid) {
		return cellToPointIndex;
	}

	for (let pointIndex = 0; pointIndex < numPositions; pointIndex += 1) {
		const row = rows[pointIndex];
		const col = cols[pointIndex];
		if (row >= height || col >= width) {
			continue;
		}

		const cellIndex = row * width + col;
		const existingPointIndex = cellToPointIndex[cellIndex];
		if (existingPointIndex === -1) {
			cellToPointIndex[cellIndex] = pointIndex;
			continue;
		}

		if (existingPointIndex !== pointIndex) {
			cellToPointIndex[cellIndex] = AMBIGUOUS_CELL_POINT_INDEX;
		}
	}

	return cellToPointIndex;
}

function fillColorBuffer(layout: UtciGridLayout, colors: Float32Array): void {
	layout.colorBuffer.fill(0);

	for (let i = 0; i < layout.numPositions; i++) {
		const texelIndex = layout.indexToTexel[i];
		const colorOffset = texelIndex * 4;
		const colorIndex = i * 3;

		layout.colorBuffer[colorOffset] = Math.floor(colors[colorIndex] * 255);
		layout.colorBuffer[colorOffset + 1] = Math.floor(colors[colorIndex + 1] * 255);
		layout.colorBuffer[colorOffset + 2] = Math.floor(colors[colorIndex + 2] * 255);
		layout.colorBuffer[colorOffset + 3] = TEXTURE_ALPHA;
	}
}

function transformToWorld(
	x: number,
	y: number,
	z: number,
	coordinateSystem: 'xy_ground' | 'xz_ground',
	target: THREE.Vector3
): THREE.Vector3 {
	if (coordinateSystem === 'xy_ground') {
		target.set(x, z, -y);
	} else {
		target.set(x, y, z);
	}
	return target;
}
