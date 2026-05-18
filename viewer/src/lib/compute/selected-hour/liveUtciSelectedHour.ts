import type { Analysis } from '$lib/types/analysis';
import { getUtciRangeForDisplay } from '$lib/utils/effectiveHourIndex';

const DEFAULT_LIVE_UTCI_DISPLAY_RANGE = { min: -20, max: 60 };

export function resolveLiveSelectedHourTimeIndex(params: {
	monthIndex: number;
	hourIndex: number;
	numHours?: number;
}): number {
	const numHours = params.numHours ?? 24;
	return params.monthIndex * numHours + params.hourIndex;
}

export function buildSelectedHourLiveAnalysis(params: {
	base: Analysis;
	utciValues: Float32Array;
	utciRange?: { min: number; max: number } | null;
	monthIndex: number;
	timeIndex: number;
}): Analysis {
	const suppliedRange =
		params.utciRange &&
		Number.isFinite(params.utciRange.min) &&
		Number.isFinite(params.utciRange.max) &&
		params.utciRange.max > params.utciRange.min
			? params.utciRange
			: null;
	let derivedRange: { min: number; max: number } | null = null;
	if (!suppliedRange) {
		let min = Number.POSITIVE_INFINITY;
		let max = Number.NEGATIVE_INFINITY;
		for (const value of params.utciValues) {
			if (value < min) min = value;
			if (value > max) max = value;
		}
		derivedRange = Number.isFinite(min) && Number.isFinite(max) ? { min, max } : null;
	}
	const range = suppliedRange ?? derivedRange ?? params.base.metadata.utci_range;

	return {
		metadata: {
			...params.base.metadata,
			analysis_type: 'single_hour',
			num_positions: params.utciValues.length,
			num_months: 1,
			utci_range: range
		},
		data: {
			numPositions: params.utciValues.length,
			numHours: 1,
			positions: params.base.data.positions,
			utciValues: params.utciValues,
			utciByHour: [params.utciValues],
			shadingIndex:
				'shadingIndex' in params.base.data ? params.base.data.shadingIndex : undefined,
			selectedMonthIndex: params.monthIndex,
			selectedTimeIndex: params.timeIndex
		} as Analysis['data']
	};
}

function resolveFiniteRange(range: { min: number; max: number } | null): {
	min: number;
	max: number;
} {
	if (
		range &&
		Number.isFinite(range.min) &&
		Number.isFinite(range.max) &&
		range.max > range.min
	) {
		return range;
	}

	return { ...DEFAULT_LIVE_UTCI_DISPLAY_RANGE };
}

function getUtciValuesRange(values: Float32Array | undefined): { min: number; max: number } | null {
	if (!values?.length) return null;

	let min = Number.POSITIVE_INFINITY;
	let max = Number.NEGATIVE_INFINITY;
	for (const value of values) {
		if (!Number.isFinite(value)) continue;
		if (value < min) min = value;
		if (value > max) max = value;
	}

	return resolveFiniteRange(Number.isFinite(min) && Number.isFinite(max) ? { min, max } : null);
}

export function resolveAcceptedGpuResidentUtciRange(params: {
	base: Analysis;
	monthIndex: number;
	hourIndex: number;
	colorMode: 'normalized' | 'discrete';
	selectedHourUtci?: Float32Array;
}): { min: number; max: number } {
	if (params.colorMode === 'discrete') {
		const selectedRange = getUtciValuesRange(params.selectedHourUtci);
		if (selectedRange) return selectedRange;
	}

	const range = getUtciRangeForDisplay(
		params.base.metadata,
		params.colorMode,
		params.hourIndex,
		params.monthIndex
	);
	return {
		min: range.utciMin,
		max: range.utciMax
	};
}

export function resolveLiveGpuResidentUtciRange(params: {
	selectedHourUtci?: Float32Array;
	selectedHourUtciRange?: { min: number; max: number } | null;
	selectedDayUtciRange?: { min: number; max: number } | null;
	colorMode?: 'normalized' | 'discrete';
}): { min: number; max: number } {
	if (params.colorMode === 'normalized') {
		return resolveFiniteRange(params.selectedDayUtciRange ?? null);
	}

	if (
		params.selectedHourUtciRange &&
		Number.isFinite(params.selectedHourUtciRange.min) &&
		Number.isFinite(params.selectedHourUtciRange.max) &&
		params.selectedHourUtciRange.max > params.selectedHourUtciRange.min
	) {
		return params.selectedHourUtciRange;
	}

	const selectedRange = getUtciValuesRange(params.selectedHourUtci);
	if (selectedRange) return selectedRange;

	return { ...DEFAULT_LIVE_UTCI_DISPLAY_RANGE };
}
