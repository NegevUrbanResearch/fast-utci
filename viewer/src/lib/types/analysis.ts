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
 * Full day UTCI data structure
 */
export interface FullDayData {
	numPositions: number;
	numHours: number;
	positions: Float32Array;
	utciByHour: Float32Array[];
	shadingIndex?: Float32Array; // Optional Shading Index values
}

/**
 * UTCI data (union type for single hour or full day)
 */
export type UTCIData = SingleHourData | FullDayData;

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
export interface AnalysisMetadata {
	analysis_type: 'single_hour' | 'full_day';
	num_positions: number;
	hours: string[]; // Array of hour strings like ["00:00", "01:00", ...]
	utci_range: UTCIRange;
	grid_size: number;
	coordinate_system: 'xy_ground' | 'xz_ground';
	model_file: string;
	sun_positions?: SunPosition[];
	hour_statistics?: HourStatistics[];
	location?: {
		latitude: number;
		longitude: number;
	};
	date?: string;
	has_shading_index?: boolean; // Whether Shading Index data is available
	shading_index_range?: ShadingIndexRange; // Shading Index value range
	/** Analysis bounds for grid generation (x_min, x_max, y_min, y_max, z). Grid is always built from these bounds. */
	bounds?: { x_min: number; x_max: number; y_min: number; y_max: number; z?: number };
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


