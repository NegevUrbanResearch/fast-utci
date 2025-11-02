/**
 * Validation Service
 * 
 * Service for loading and comparing Grasshopper validation data with analysis results
 */

import { base } from '$app/paths';
import { loadBinaryData, calculateStatistics } from './dataLoader';
import type { Analysis } from '$lib/types/analysis';

/**
 * Validation data structure (same as analysis data)
 */
export interface ValidationData {
	numPositions: number;
	numHours: number;
	positions: Float32Array;
	utciByHour: Float32Array[];
}

/**
 * Comparison statistics
 */
export interface ComparisonStats {
	analysis: {
		min: number;
		max: number;
		mean: number;
	};
	validation: {
		min: number;
		max: number;
		mean: number;
	};
	comparison: {
		minDiff: number;
		maxDiff: number;
		meanDiff: number;
	};
}

/**
 * Load Grasshopper validation data
 * @param validationPath - Path to validation binary file (optional, defaults to base path + "/data/validation/...")
 * @returns Promise with validation data
 */
export async function loadValidationData(
	validationPath?: string
): Promise<ValidationData> {
	const path = validationPath || `${base}/data/validation/grasshopper_aug15_fullday.bin`;
	console.log('[LOAD] Loading Grasshopper validation data...');
	const data = await loadBinaryData(path, 'full_day');
	console.log('[OK] Validation data loaded');
	return data;
}

/**
 * Calculate statistics differences between two datasets
 */
function calculateStatisticsDifferences(
	stats1: { min: number; max: number; mean: number },
	stats2: { min: number; max: number; mean: number }
): { minDiff: number; maxDiff: number; meanDiff: number } {
	return {
		minDiff: stats1.min - stats2.min,
		maxDiff: stats1.max - stats2.max,
		meanDiff: stats1.mean - stats2.mean
	};
}

/**
 * Calculate average mean difference across all 24 hours
 */
export function calculateAvgMeanDiffAllHours(
	analysis: Analysis,
	validation: ValidationData
): number | null {
	if (analysis.metadata.analysis_type !== 'full_day') {
		return null;
	}

	let totalMeanDiff = 0;
	let validHours = 0;

	for (let hourIndex = 0; hourIndex < 24; hourIndex++) {
		const analysisValues = analysis.data.utciByHour[hourIndex];
		const validationValues = validation.utciByHour[hourIndex];

		if (analysisValues && validationValues) {
			const analysisStats = calculateStatistics(analysisValues);
			const validationStats = calculateStatistics(validationValues);
			const diff = analysisStats.mean - validationStats.mean;
			totalMeanDiff += diff;
			validHours++;
		}
	}

	return validHours > 0 ? totalMeanDiff / validHours : null;
}

/**
 * Compare analysis with validation data
 */
export function compareWithValidation(
	analysis: Analysis,
	validation: ValidationData,
	hourIndex: number = 0
): ComparisonStats {
	// Get UTCI values for this hour
	let analysisValues: Float32Array;
	let validationHourIndex = hourIndex;

	if (analysis.data.numHours === 1) {
		analysisValues = analysis.data.utciValues;
		// For single hour analysis, use the specific hour from metadata
		validationHourIndex = analysis.metadata.hours[0] as number;
	} else {
		analysisValues = analysis.data.utciByHour[hourIndex];
		validationHourIndex = hourIndex;
	}

	const validationValues = validation.utciByHour[validationHourIndex];

	// Calculate statistics for both full datasets
	const analysisStats = calculateStatistics(analysisValues);
	const validationStats = calculateStatistics(validationValues);

	// Calculate statistics differences
	const statsDiff = calculateStatisticsDifferences(analysisStats, validationStats);

	return {
		analysis: analysisStats,
		validation: validationStats,
		comparison: {
			minDiff: statsDiff.minDiff,
			maxDiff: statsDiff.maxDiff,
			meanDiff: statsDiff.meanDiff
		}
	};
}

