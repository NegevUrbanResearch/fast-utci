/**
 * Type definitions for UTCI Analysis data structures
 */

/**
 * Single hour UTCI data structure
 */
export interface SingleHourData {
	numPositions: number;
	numHours: 1;
	positions: Float32Array;
	utciValues: Float32Array;
	shadingIndex?: Float32Array; // Optional Shading Index values
}

/**
 * Compact UTCI storage for large multi-month analyses.
 * Values stored as Int16 with scale factor (value_c = stored / scale).
 * Scale 100 gives 0.01 C precision; typical UTCI range -40..+50 C fits in Int16.
 * Use when present to avoid holding 288 Float32Arrays in memory.
 */
export interface UtciStorage {
	buffer: Int16Array;
	numPoints: number;
	numSlices: number;
	scale: number;
}

/**
 * Full day UTCI data structure
 */
export interface FullDayData {
	numPositions: number;
	numHours: number;
	positions: Float32Array;
	/** UTCI values per slice. Omit when utciStorage is used. */
	utciByHour?: Float32Array[];
	/** Compact storage for multi-month data; decode via dataLoader helpers */
	utciStorage?: UtciStorage;
	shadingIndex?: Float32Array; // Optional Shading Index values
}

/**
 * UTCI data (union type for single hour or full day)
 */
export type UTCIData = SingleHourData | FullDayData;

function hasFloat32ArrayValues(value: unknown): value is Float32Array {
	return value instanceof Float32Array;
}

function hasUtciStorageShape(value: unknown): value is UtciStorage {
	if (typeof value !== 'object' || value === null) return false;

	const candidate = value as Partial<UtciStorage>;
	return (
		candidate.buffer instanceof Int16Array &&
		typeof candidate.numPoints === 'number' &&
		typeof candidate.numSlices === 'number' &&
		typeof candidate.scale === 'number'
	);
}

export function isSingleHourData(data: UTCIData): data is SingleHourData {
	return data.numHours === 1 && hasFloat32ArrayValues((data as { utciValues?: unknown }).utciValues);
}

export function isFullDayData(data: UTCIData): data is FullDayData {
	return !isSingleHourData(data) && (hasDecodedUtciByHour(data) || hasCompactUtciStorage(data));
}

export function hasDecodedUtciByHour(
	data: UTCIData
): data is UTCIData & { utciByHour: Float32Array[] } {
	const candidate = (data as { utciByHour?: unknown }).utciByHour;
	return (
		Array.isArray(candidate) &&
		candidate.length > 0 &&
		candidate.every((slice) => hasFloat32ArrayValues(slice))
	);
}

export function hasCompactUtciStorage(
	data: UTCIData
): data is UTCIData & { utciStorage: UtciStorage } {
	return hasUtciStorageShape((data as { utciStorage?: unknown }).utciStorage);
}

/**
 * UTCI value range
 */
export interface UTCIRange {
	min: number;
	max: number;
}

/**
 * Shading Index value range
 */
export interface ShadingIndexRange {
	min: number;
	max: number;
}

/**
 * Statistics for a single hour
 */
export interface HourStatistics {
	min: number;
	max: number;
	mean: number;
}

/**
 * Hour metadata
 */
export interface HourMetadata {
	hour: string; // Format: "HH:00"
	index: number; // 0-23
}

/**
 * Sun position data
 */
export interface SunPosition {
	altitude: number;
	azimuth: number;
	is_up: boolean;
	vector: [number, number, number]; // [x, y, z] in Python coordinate system
}

/**
 * Analysis metadata
 */
export type AnalysisCoordinateSystem = 'xy_ground' | 'xz_ground';

export interface AnalysisRectangularBounds {
	x_min: number;
	x_max: number;
	y_min: number;
	y_max: number;
	z?: number;
}

export type ProjectedTriangle2D = readonly [number, number, number, number, number, number];

export interface StudyAreaMaskSummary {
	canonicalPointCount: number;
	activePointCount: number;
	width: number;
	height: number;
	footprintChecksum: string;
	maskChecksum: string;
	signature: string;
}

export interface StudyAreaMask extends StudyAreaMaskSummary {
	mask: Uint8Array;
	activeCanonicalIndices: Uint32Array;
}

export interface AnalysisActiveMask {
	source: 'base' | 'base+road';
	canonicalPointCount: number;
	activePointCount: number;
	inactivePointCount: number;
	activePointRatio: number;
	activeMaskChecksum: string;
	activeCanonicalIndices: Uint32Array;
	signature?: string;
}

export interface AnalysisMetadata {
	analysis_type: 'single_hour' | 'full_day';
	num_positions: number;
	hours: string[]; // Array of hour strings like ["00:00", "01:00", ...]
	utci_range: UTCIRange;
	grid_size: number;
	coordinate_system: AnalysisCoordinateSystem;
	model_file: string;
	/** Analysis id used to load this metadata, including project prefix when available. */
	source_analysis_id?: string;
	sun_positions?: SunPosition[];
	hour_statistics?: HourStatistics[];
	location?: {
		latitude: number;
		longitude: number;
	};
	epw_file?: string;
	date?: string;
	has_shading_index?: boolean; // Whether Shading Index data is available
	shading_index_range?: ShadingIndexRange; // Shading Index value range
	/** Analysis bounds for grid generation (x_min, x_max, y_min, y_max, z). Grid is always built from these bounds. */
	bounds?: AnalysisRectangularBounds;
	/** Runtime active study-area mask for compact live WebGPU outputs mapped into canonical cells. */
	activeMask?: AnalysisActiveMask;
	/** Number of representative months when analysis has multi-month data (e.g. 12 for full year). */
	num_months?: number;
}

/**
 * Complete UTCI analysis
 */
export interface Analysis {
	metadata: AnalysisMetadata;
	data: UTCIData;
}

/**
 * Position coordinates
 */
export interface Position {
	x: number;
	y: number;
	z: number;
}
