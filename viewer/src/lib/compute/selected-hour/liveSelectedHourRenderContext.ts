import type { Analysis } from '$lib/types/analysis';
import type { MetricType } from '$lib/types/viewer';
import { getEffectiveHourIndex } from '$lib/utils/effectiveHourIndex';

export type LiveSelectedHourRangeOverride = {
	utciMin: number;
	utciMax: number;
};

export type LiveSelectedHourPublishedRenderContext = {
	analysis: Analysis;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	selectionKey: string;
	publicationPhase: 'initial' | 'scrub';
	colorMode: 'normalized' | 'discrete';
	metricType: MetricType;
	rangeOverride: LiveSelectedHourRangeOverride | null;
};

export type LiveSelectedHourSceneRenderState = {
	analysis: Analysis;
	hourIndex: number;
	monthIndex: number;
	colorMode: 'normalized' | 'discrete';
	metricType: MetricType;
	rangeOverride: LiveSelectedHourRangeOverride | undefined;
};

export type LiveSelectedHourViewerStateLike = {
	currentHour: number;
	currentMonth?: number | null;
	colorMode?: 'normalized' | 'discrete';
	metricType?: MetricType | null;
};

function resolveMetricType(metricType: MetricType | null | undefined): MetricType {
	return metricType === 'shading_index' ? 'shading_index' : 'utci';
}

function resolveRangeOverride(
	metricType: MetricType,
	rangeOverride: LiveSelectedHourRangeOverride | null | undefined
): LiveSelectedHourRangeOverride | null {
	return metricType === 'utci' ? (rangeOverride ?? null) : null;
}

export function createLiveSelectedHourPublishedRenderContext(params: {
	analysis: Analysis;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	selectionKey: string;
	publicationPhase: 'initial' | 'scrub';
	colorMode: 'normalized' | 'discrete';
	metricType?: MetricType;
	rangeOverride?: LiveSelectedHourRangeOverride | null;
}): LiveSelectedHourPublishedRenderContext {
	const metricType = resolveMetricType(params.metricType);
	return {
		analysis: params.analysis,
		monthIndex: params.monthIndex,
		hourIndex: params.hourIndex,
		timeIndex: params.timeIndex,
		selectionKey: params.selectionKey,
		publicationPhase: params.publicationPhase,
		colorMode: params.colorMode,
		metricType,
		rangeOverride: resolveRangeOverride(metricType, params.rangeOverride)
	};
}

export function resolveLiveSelectedHourSurfaceRenderState(params: {
	analysis: Analysis | null | undefined;
	viewerState: LiveSelectedHourViewerStateLike;
	publishedRenderContext?: LiveSelectedHourPublishedRenderContext | null;
	rangeOverride?: LiveSelectedHourRangeOverride | null | undefined;
}): LiveSelectedHourSceneRenderState | null {
	const publishedRenderContext = params.publishedRenderContext;
	if (publishedRenderContext != null) {
		return {
			analysis: publishedRenderContext.analysis,
			hourIndex: getEffectiveHourIndex(
				publishedRenderContext.analysis,
				publishedRenderContext.hourIndex,
				publishedRenderContext.monthIndex
			),
			monthIndex: publishedRenderContext.monthIndex,
			colorMode: publishedRenderContext.colorMode,
			metricType: publishedRenderContext.metricType,
			rangeOverride:
				publishedRenderContext.metricType === 'utci'
					? (publishedRenderContext.rangeOverride ?? undefined)
					: undefined
		};
	}

	if (params.analysis == null) {
		return null;
	}

	const monthIndex = params.viewerState.currentMonth ?? 7;
	const hourIndex = params.viewerState.currentHour ?? 0;
	const metricType = resolveMetricType(params.viewerState.metricType);
	return {
		analysis: params.analysis,
		hourIndex: getEffectiveHourIndex(params.analysis, hourIndex, monthIndex),
		monthIndex,
		colorMode: params.viewerState.colorMode ?? 'normalized',
		metricType,
		rangeOverride:
			metricType === 'utci' ? (params.rangeOverride ?? undefined) : undefined
	};
}
