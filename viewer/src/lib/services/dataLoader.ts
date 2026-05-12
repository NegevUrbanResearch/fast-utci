/**
 * UTCI Data Loader for Binary Format
 * 
 * Loads and parses binary UTCI data files exported by export_for_viewer.py
 * Supports both single hour and full day analysis formats.
 */

import { base } from '$app/paths';
import type {
	SingleHourData,
	FullDayData,
	UTCIData,
	Analysis,
	Position,
	HourStatistics
} from '$lib/types/analysis';
import {
	isSingleHourData,
	hasCompactUtciStorage,
	hasDecodedUtciByHour
} from '$lib/types/analysis';
import { canonicalGridPoints } from '$lib/compute/core/canonicalGrid';

// Data base path: strip /viewer/build from base path to get project root
// e.g., /fast-utci/viewer/build -> /fast-utci
const getDataBasePath = () => {
	if (typeof window === 'undefined') return ''; // SSR
	const basePath = base || '';
	// Remove /viewer/build from the end if present
	return basePath.replace(/\/viewer\/build$/, '');
};

/**
 * Parse single hour binary data
 * 
 * Binary structure:
 *   [4 bytes: num_positions (uint32)]
 *   [num_positions × 12 bytes: positions as float32 x,y,z]
 *   [num_positions × 4 bytes: utci values as float32]
 * 
 * @param buffer - Binary data buffer
 * @returns Parsed data with positions and utci arrays
 */
export function parseSingleHourBinary(buffer: ArrayBuffer): SingleHourData {
	const dataView = new DataView(buffer);
	let offset = 0;
	
	// Read header
	const numPositions = dataView.getUint32(offset, true); // little-endian
	offset += 4;
	
	// Read positions (3 floats per position)
	const positions = new Float32Array(numPositions * 3);
	for (let i = 0; i < numPositions * 3; i++) {
		positions[i] = dataView.getFloat32(offset, true);
		offset += 4;
	}
	
	// Read UTCI values (1 float per position)
	const utciValues = new Float32Array(numPositions);
	for (let i = 0; i < numPositions; i++) {
		utciValues[i] = dataView.getFloat32(offset, true);
		offset += 4;
	}
	
	return {
		numPositions,
		positions,
		utciValues,
		numHours: 1
	};
}

/**
 * Parse full day binary data
 * 
 * Binary structure (with optional Shading Index):
 *   [8 bytes: num_positions (uint32), num_hours (uint32)]
 *   [num_positions × 12 bytes: positions as float32 x,y,z]
 *   [4 bytes: has_shading_index (uint32, 0 or 1)] - NEW format only
 *   [IF has_shading_index == 1: num_positions × 4 bytes: shading_index as float32]
 *   [num_positions × 4 bytes: utci values hour 0 as float32]
 *   [num_positions × 4 bytes: utci values hour 1 as float32]
 *   ...
 *   [num_positions × 4 bytes: utci values hour 23 as float32]
 * 
 * OLD format (backward compatibility):
 *   [8 bytes: num_positions (uint32), num_hours (uint32)]
 *   [num_positions × 12 bytes: positions as float32 x,y,z]
 *   [num_positions × 4 bytes: utci values hour 0 as float32]
 *   ... (no has_shading_index flag, no Shading Index array)
 * 
 * @param buffer - Binary data buffer
 * @param metadata - Optional metadata to determine format (checks has_shading_index)
 * @returns Parsed data with positions and utci arrays organized by hour
 */
export function parseFullDayBinary(buffer: ArrayBuffer, metadata?: any): FullDayData {
	const dataView = new DataView(buffer);
	let offset = 0;
	
	// Read header
	const numPositions = dataView.getUint32(offset, true);
	offset += 4;
	const numHours = dataView.getUint32(offset, true);
	offset += 4;
	
	// Read positions (3 floats per position)
	const positions = new Float32Array(numPositions * 3);
	for (let i = 0; i < numPositions * 3; i++) {
		positions[i] = dataView.getFloat32(offset, true);
		offset += 4;
	}
	
	// Determine if this is new format (has Shading Index flag)
	// Calculate expected size for old format (without Shading Index)
	const oldFormatSize = 8 + (numPositions * 12) + (numPositions * 4 * numHours);
	const isNewFormat = buffer.byteLength > oldFormatSize;
	
	// Read has_shading_index flag and Shading Index array if new format
	let shadingIndex: Float32Array | undefined;
	let hasShadingIndex = false;
	
	if (isNewFormat) {
		// New format: read the flag
		if (offset + 4 > buffer.byteLength) {
			// Safety check: shouldn't happen if size calculation is correct
			console.warn('[WARN] Binary file size suggests new format but cannot read flag');
		} else {
			hasShadingIndex = dataView.getUint32(offset, true) === 1;
			offset += 4;
			
			// Read Shading Index array if present
			if (hasShadingIndex) {
				shadingIndex = new Float32Array(numPositions);
				for (let i = 0; i < numPositions; i++) {
					if (offset + 4 > buffer.byteLength) {
						// Safety check: if we go out of bounds, stop reading
						console.warn('[WARN] Binary file truncated while reading Shading Index');
						shadingIndex = undefined;
						break;
					}
					shadingIndex[i] = dataView.getFloat32(offset, true);
					offset += 4;
				}
			}
		}
	}
	// If old format, skip the flag and Shading Index - offset is already correct
	
	// Read UTCI values hour by hour
	const utciByHour: Float32Array[] = [];
	for (let hour = 0; hour < numHours; hour++) {
		const hourValues = new Float32Array(numPositions);
		for (let i = 0; i < numPositions; i++) {
			if (offset + 4 > buffer.byteLength) {
				throw new Error(`Binary file is truncated: expected ${numPositions} UTCI values for hour ${hour}, but reached end of file`);
			}
			hourValues[i] = dataView.getFloat32(offset, true);
			offset += 4;
		}
		utciByHour.push(hourValues);
	}
	
	const result: FullDayData = {
		numPositions,
		numHours,
		positions,
		utciByHour
	};
	
	// Add Shading Index if present
	if (shadingIndex) {
		result.shadingIndex = shadingIndex;
	}
	
	return result;
}

/**
 * Load binary UTCI data file
 * @param binaryPath - Path to binary data file
 * @param analysisType - "single_hour" or "full_day"
 * @param metadata - Optional metadata to help determine binary format
 * @returns Parsed data object
 */
export async function loadBinaryData(
	binaryPath: string, 
	analysisType: 'single_hour' | 'full_day',
	metadata?: any
): Promise<UTCIData> {
	const response = await fetch(binaryPath);
	if (!response.ok) {
		throw new Error(`Failed to load binary data: ${response.statusText}`);
	}
	
	const buffer = await response.arrayBuffer();
	
	if (analysisType === 'single_hour') {
		return parseSingleHourBinary(buffer);
	} else {
		return parseFullDayBinary(buffer, metadata);
	}
}

/**
 * Load analysis metadata from JSON file
 * @param metadataPath - Path to metadata JSON file
 * @returns Metadata object
 */
export async function loadMetadata(metadataPath: string) {
	const response = await fetch(metadataPath);
	if (!response.ok) {
		throw new Error(`Failed to load metadata: ${response.statusText}`);
	}
	return await response.json();
}

/**
 * Load complete UTCI analysis (metadata + binary data)
 * @param analysisId - Analysis identifier (e.g., "20250815_grid_2m_fullday")
 * @param dataDir - Base directory for data files (default: uses project root + "/data/analyses")
 * @returns Complete analysis data with metadata and binary data
 */
export async function loadAnalysis(analysisId: string, dataDir?: string): Promise<Analysis> {
	// Use provided dataDir or construct from data base path (project root, not viewer/build)
	const dataBasePath = getDataBasePath();
	const baseDataDir = dataDir || `${dataBasePath}/data/analyses`;
	const metadataPath = `${baseDataDir}/${analysisId}.json`;
	const binaryPath = `${baseDataDir}/${analysisId}.bin`;
	
	console.log(`[LOAD] Loading analysis: ${analysisId}`);
	
	// Load metadata
	const metadata = {
		...(await loadMetadata(metadataPath)),
		source_analysis_id: analysisId
	};
	console.log(`[OK] Metadata loaded: ${metadata.num_positions} positions, ${metadata.hours.length} hours`);
	
	// Load binary data (pass metadata for format detection)
	const binaryData = await loadBinaryData(binaryPath, metadata.analysis_type, metadata);
	console.log(`[OK] Binary data loaded: ${(binaryData.positions.length * 4 / 1024).toFixed(1)} KB`);
	
	return {
		metadata,
		data: binaryData
	};
}

/**
 * Load analysis metadata without fetching UTCI binary data.
 */
export async function loadAnalysisMetadataOnly(
	analysisId: string,
	dataDir?: string
): Promise<Analysis> {
	const dataBasePath = getDataBasePath();
	const baseDataDir = dataDir || `${dataBasePath}/data/analyses`;
	const metadataPath = `${baseDataDir}/${analysisId}.json`;

	console.log(`[LOAD] Loading analysis metadata: ${analysisId}`);

	const metadata = {
		...(await loadMetadata(metadataPath)),
		source_analysis_id: analysisId
	};
	const positions =
		metadata.bounds && metadata.grid_size > 0
			? canonicalGridPointsToAnalysisPositions({
					bounds: metadata.bounds,
					gridSize: metadata.grid_size,
					coordinateSystem: metadata.coordinate_system,
					zHeight: metadata.bounds.z
				})
			: new Float32Array(0);
	const derivedNumPositions = positions.length / 3;
	if (derivedNumPositions > 0 && derivedNumPositions !== metadata.num_positions) {
		throw new Error(
			`Metadata-derived grid point count (${derivedNumPositions}) does not match metadata num_positions (${metadata.num_positions}) for ${analysisId}`
		);
	}

	return {
		metadata,
		data: {
			numPositions: derivedNumPositions,
			numHours: metadata.hours.length,
			positions,
			utciByHour: []
		}
	};
}

function canonicalGridPointsToAnalysisPositions(params: {
	bounds: NonNullable<Analysis['metadata']['bounds']>;
	gridSize: number;
	coordinateSystem: Analysis['metadata']['coordinate_system'];
	zHeight?: number;
}): Float32Array {
	const canonicalGrid = canonicalGridPoints(params);
	if (params.coordinateSystem !== 'xy_ground') {
		return canonicalGrid.points;
	}

	const points = canonicalGrid.points;
	const analysisPositions = new Float32Array(points.length);
	for (let i = 0; i < points.length; i += 3) {
		analysisPositions[i] = points[i];
		analysisPositions[i + 1] = -points[i + 2];
		analysisPositions[i + 2] = points[i + 1];
	}
	return analysisPositions;
}

/**
 * Get UTCI value for a specific position and hour
 * @param data - UTCI data (single hour or full day)
 * @param positionIndex - Position index (0 to numPositions-1)
 * @param hourIndex - Hour index (0 to numHours-1), defaults to 0
 * @returns UTCI value in Celsius
 */
export function getUTCIValue(data: UTCIData, positionIndex: number, hourIndex: number = 0): number {
	if (isSingleHourData(data)) {
		return data.utciValues[positionIndex];
	}
	if (hasCompactUtciStorage(data)) {
		const { buffer, numPoints, scale, numSlices } = data.utciStorage;
		const sliceIndex = Math.max(0, Math.min(hourIndex, numSlices - 1));
		const idx = sliceIndex * numPoints + positionIndex;
		return buffer[idx] / scale;
	}
	if (hasDecodedUtciByHour(data)) {
		const hourValues = data.utciByHour[hourIndex];
		if (hourValues) {
			return hourValues[positionIndex];
		}
	}
	throw new Error('Full-day UTCI data is missing utciStorage and decoded utciByHour slices');
}

/**
 * Get position coordinates for a specific position
 * @param data - UTCI data
 * @param positionIndex - Position index (0 to numPositions-1)
 * @returns Position {x, y, z}
 */
export function getPosition(data: UTCIData, positionIndex: number): Position {
	const i = positionIndex * 3;
	return {
		x: data.positions[i],
		y: data.positions[i + 1],
		z: data.positions[i + 2]
	};
}

/**
 * Get all UTCI values for a specific hour/slice.
 * For utciStorage, decodes on demand (allocates Float32Array).
 * @param data - UTCI data
 * @param hourIndex - Hour/slice index (0 to numHours-1), defaults to 0
 * @returns Array of UTCI values
 */
export function getUTCIForHour(data: UTCIData, hourIndex: number = 0): Float32Array {
	if (isSingleHourData(data)) {
		return data.utciValues;
	}
	if (hasCompactUtciStorage(data)) {
		const { buffer, numPoints, scale, numSlices } = data.utciStorage;
		// Clamp to valid slice index to avoid reading wrong month data
		const sliceIndex = Math.max(0, Math.min(hourIndex, numSlices - 1));
		const out = new Float32Array(numPoints);
		const base = sliceIndex * numPoints;
		for (let i = 0; i < numPoints; i++) {
			out[i] = buffer[base + i] / scale;
		}
		return out;
	}
	if (hasDecodedUtciByHour(data)) {
		const hourValues = data.utciByHour[hourIndex];
		if (hourValues) {
			return hourValues;
		}
	}
	throw new Error('Full-day UTCI data is missing utciStorage and decoded utciByHour slices');
}

/**
 * Get UTCI data as number[][] for export (e.g. parity artifact).
 * For utciStorage, decodes all slices incrementally.
 */
export function getUtciByHourForExport(data: FullDayData): number[][] {
	if (data.utciByHour && data.utciByHour.length > 0) {
		return data.utciByHour.map((arr) => Array.from(arr));
	}
	if (data.utciStorage) {
		const result: number[][] = [];
		for (let i = 0; i < data.utciStorage.numSlices; i++) {
			result.push(Array.from(getUTCIForHour(data, i)));
		}
		return result;
	}
	return [];
}

/**
 * Get Shading Index array from data
 * @param data - UTCI data (single hour or full day)
 * @returns Shading Index array if available, null otherwise
 */
export function getShadingIndex(data: UTCIData): Float32Array | null {
	return data.shadingIndex || null;
}

/**
 * Calculate statistics for UTCI data
 * @param utciValues - Array of UTCI values
 * @returns Statistics {min, max, mean, count}
 */
export function calculateStatistics(utciValues: Float32Array): HourStatistics & { count: number } {
	let min = Infinity;
	let max = -Infinity;
	let sum = 0;
	let count = 0;
	
	for (let i = 0; i < utciValues.length; i++) {
		const val = utciValues[i];
		if (!isNaN(val) && isFinite(val)) {
			min = Math.min(min, val);
			max = Math.max(max, val);
			sum += val;
			count++;
		}
	}
	
	return {
		min,
		max,
		mean: count > 0 ? sum / count : 0,
		count
	};
}


