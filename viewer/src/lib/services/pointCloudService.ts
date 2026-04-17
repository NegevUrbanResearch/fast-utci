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

// Vertical separation between the UTCI overlay and underlying geometry.
// Use a small negative offset so the UTCI plane sits just below the sampled
// ground grid; it will still be visible through the semi-transparent base
// mesh but remain behind buildings in depth.
const VISUAL_LAYER_OFFSET = -0.05;
const DEFAULT_OPACITY = 0.9;
const TEXTURE_ALPHA = 255;

interface UtciGridLayout {
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
	indexToTexel: Uint32Array;
	colorBuffer: Uint8Array;
	texture?: THREE.DataTexture;
}

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
export function createUtciSurfaceMesh(
	analysis: Analysis,
	hourIndex: number = 0,
	colorMode: 'normalized' | 'discrete' = 'normalized',
	metricType: MetricType = 'utci',
	rangeOverride?: UtciRangeOverride,
	monthIndex: number = 7
): THREE.Mesh {
	const layout = buildUtciGridLayout(analysis);
	const colors = createColors(analysis, hourIndex, colorMode, metricType, rangeOverride, monthIndex);
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
		// No polygon offset: we rely on geometry placement (slightly below
		// ground) plus the semi-transparent base mesh for visual layering.
		polygonOffset: false,
		toneMapped: false
	});

	const planeWidth = layout.width * layout.gridSize;
	const planeHeight = layout.height * layout.gridSize;

	const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
	geometry.rotateX(-Math.PI / 2);

	const mesh = new THREE.Mesh(geometry, material);
	mesh.name = 'UTCI Texture Overlay';
	mesh.position.set(layout.centerX, layout.baseY, layout.centerZ);
	mesh.renderOrder = 2;
	mesh.frustumCulled = false;

	mesh.userData.utciLayout = layout;
	return mesh;
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
): void {
	const layout: UtciGridLayout | undefined = mesh.userData.utciLayout;
	if (!layout) {
		console.warn('[UTCI] Missing layout on surface mesh. Recreating mesh.');
		return;
	}

	if (layout.numPositions !== analysis.data.numPositions) {
		console.warn('[UTCI] Analysis position count changed; recreate mesh to realign texture.');
		return;
	}

	const material = mesh.material as THREE.MeshBasicMaterial;
	const texture = layout.texture ?? (material.map as THREE.DataTexture | null);
	if (!texture) {
		console.warn('[UTCI] Missing texture on surface mesh. Recreating mesh.');
		return;
	}

	const colors = createColors(analysis, hourIndex, colorMode, metricType, rangeOverride, monthIndex);
	fillColorBuffer(layout, colors);

	texture.needsUpdate = true;
}

function buildUtciGridLayout(analysis: Analysis): UtciGridLayout {
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

	let width = Math.max(1, maxCol + 1);
	let height = Math.max(1, maxRow + 1);

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

	for (let i = 0; i < numPositions; i++) {
		const flippedRow = height - 1 - rows[i];
		const texelIndex = flippedRow * width + cols[i];
		indexToTexel[i] = texelIndex;
	}

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
		indexToTexel,
		colorBuffer
	};
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
