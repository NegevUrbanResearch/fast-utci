import type { Analysis, AnalysisMetadata, HourStatistics } from '$lib/types/analysis';

/**
 * Returns the UTCI slice index for the given hour and month.
 * For single-month analyses (e.g. .bin): returns hourIndex.
 * For 12-month analyses (live WebGPU): returns monthIndex*24 + hourIndex.
 */
export function getEffectiveHourIndex(
	analysis: Analysis | null,
	hourIndex: number,
	monthIndex: number
): number {
	if (!analysis?.metadata?.num_months || analysis.metadata.num_months <= 1) {
		return hourIndex;
	}
	return monthIndex * 24 + hourIndex;
}

/**
 * Returns the UTCI range for display (legend, color mapping).
 * When num_months > 1 and colorMode is 'normalized', "Full day" means the 24 hours
 * of the selected month only - not the global range across all 12 months (which
 * would include winter values like -3 C in August view).
 */
export function getUtciRangeForDisplay(
	metadata: AnalysisMetadata,
	colorMode: 'normalized' | 'discrete',
	hourIndex: number,
	monthIndex: number
): { utciMin: number; utciMax: number } {
	const stats = metadata.hour_statistics;
	const fallback = {
		utciMin: metadata.utci_range.min,
		utciMax: metadata.utci_range.max
	};

	if (colorMode === 'discrete') {
		const numMonthsVal = metadata.num_months ?? 1;
		const idx = numMonthsVal > 1 ? monthIndex * 24 + hourIndex : hourIndex;
		const h = stats?.[idx];
		if (h) return { utciMin: h.min, utciMax: h.max };
		return fallback;
	}

	// Normalized: use full-day range. For multi-month, use selected month's 24h only.
	const numMonthsVal = metadata.num_months ?? 1;
	if (numMonthsVal > 1 && stats && stats.length >= 24) {
		const base = monthIndex * 24;
		let min = Number.POSITIVE_INFINITY;
		let max = Number.NEGATIVE_INFINITY;
		for (let i = 0; i < 24 && base + i < stats.length; i++) {
			const s: HourStatistics = stats[base + i];
			if (s.min < min) min = s.min;
			if (s.max > max) max = s.max;
		}
		if (Number.isFinite(min) && Number.isFinite(max)) {
			return { utciMin: min, utciMax: max };
		}
	}

	return fallback;
}
