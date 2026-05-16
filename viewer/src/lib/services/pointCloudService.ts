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
import {
	createGpuNativeUtciSurfaceMesh,
	disposeGpuNativeUtciSurfaceMesh,
	getGpuNativeUtciSurfaceSource,
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
	texture?: THREE.DataTexture;
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

export function buildUtciGridLayout(analysis: Analysis): UtciGridLayout {
	const { data, metadata } = analysis;
	const coordinateSystem = metadata.coordinate_system || 'xy_ground';
	const gridSize = metadata.grid_size || 1;
	const numPositions = data.numPositions;

	const xs = new Float32Array(numPositions);
	const ys = new Float32Array(numPositions);
	const zs = new Float32Array(numPositions);

	const indexToTexel = new Uint32Array(numPositions);
	const rows = new Uint32Array(numPositions);
	const cols = new Uint32Array(numPositions);

	let minX = Infinity;
	let minZ = Infinity;
	let maxX = -Infinity;
	let maxZ = -Infinity;
	let minY = Infinity;
	let maxY = -Infinity;

	const transformed = new THREE.Vector3();

	// Calculate normalization offset if enabled
	let normalizationOffset = new THREE.Vector3(0, 0, 0);
	if (isNormalizationEnabled()) {
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
	}

	for (let i = 0; i < numPositions; i++) {
		const flippedRow = height - 1 - rows[i];
		const texelIndex = flippedRow * width + cols[i];
		indexToTexel[i] = texelIndex;
	}

	const cellToPointIndex = createCellToPointIndex({
		rows,
		cols,
		numPositions,
		width,
		height,
		layoutValid
	});
	const colorBuffer = new Uint8Array(width * height * 4);
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
		colorBuffer
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
