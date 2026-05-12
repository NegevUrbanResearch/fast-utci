/**
 * Validation Service
 * 
 * Service for loading and comparing Grasshopper validation data with analysis results
 */

import { base } from '$app/paths';
import { loadBinaryData, calculateStatistics, getUTCIForHour } from './dataLoader';
import { hasDecodedUtciByHour, isSingleHourData } from '$lib/types/analysis';
import type { Analysis } from '$lib/types/analysis';

// Data base path: strip /viewer/build from base path to get project root
const getDataBasePath = () => {
	if (typeof window === 'undefined') return ''; // SSR
	const basePath = base || '';
	return basePath.replace(/\/viewer\/build$/, '');
};

/**
 * Validation data structure (same as analysis data)
 */
export interface ValidationData {
	numPositions: number;
	numHours: number;
	positions: Float32Array;
	utciByHour: Float32Array[];
}

function resolveValidationHourIndex(hourLabel: string | undefined): number | null {
	if (hourLabel == null) return 0;
	if (/^\d+$/.test(hourLabel)) {
		const numericHour = Number(hourLabel);
		return Number.isInteger(numericHour) ? numericHour : null;
	}

	const hourMatch = /^(\d{1,2}):/.exec(hourLabel);
	if (!hourMatch) return null;

	const parsedHour = Number(hourMatch[1]);
	return Number.isInteger(parsedHour) ? parsedHour : null;
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
 * @param validationPath - Path to validation binary file (optional, defaults to project root + "/data/validation/...")
 * @returns Promise with validation data
 */
export async function loadValidationData(
	validationPath?: string
): Promise<ValidationData> {
	const dataBasePath = getDataBasePath();
	const path = validationPath || `${dataBasePath}/data/validation/grasshopper_aug15_fullday.bin`;
	console.log('[LOAD] Loading Grasshopper validation data...');
	const data = await loadBinaryData(path, 'full_day');
	console.log('[OK] Validation data loaded');
	if (!hasDecodedUtciByHour(data)) {
		throw new Error('Validation data must include decoded utciByHour slices');
	}
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

	const numHours = Math.min(24, analysis.data.numHours);
	for (let hourIndex = 0; hourIndex < numHours; hourIndex++) {
		const analysisValues = getUTCIForHour(analysis.data, hourIndex);
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

	if (isSingleHourData(analysis.data)) {
		analysisValues = analysis.data.utciValues;
		const resolvedHourIndex = resolveValidationHourIndex(analysis.metadata.hours[0]);
		if (resolvedHourIndex == null) {
			throw new Error(
				`Unable to resolve validation hour index from analysis hour label: ${analysis.metadata.hours[0] ?? 'undefined'}`
			);
		}
		validationHourIndex = resolvedHourIndex;
	} else {
		analysisValues = getUTCIForHour(analysis.data, hourIndex);
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
