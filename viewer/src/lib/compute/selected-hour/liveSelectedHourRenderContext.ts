import type { Analysis } from '$lib/types/analysis';
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
	colorMode: 'normalized' | 'discrete';
	metricType: 'utci';
	rangeOverride: LiveSelectedHourRangeOverride | null;
};

export type LiveSelectedHourSceneRenderState = {
	analysis: Analysis;
	hourIndex: number;
	monthIndex: number;
	colorMode: 'normalized' | 'discrete';
	metricType: 'utci';
	rangeOverride: LiveSelectedHourRangeOverride | undefined;
};

export type LiveSelectedHourViewerStateLike = {
	currentHour: number;
	currentMonth?: number | null;
	colorMode?: 'normalized' | 'discrete';
	metricType?: string | null;
};

export function createLiveSelectedHourPublishedRenderContext(params: {
	analysis: Analysis;
	monthIndex: number;
	hourIndex: number;
	timeIndex: number;
	selectionKey: string;
	colorMode: 'normalized' | 'discrete';
	rangeOverride?: LiveSelectedHourRangeOverride | null;
}): LiveSelectedHourPublishedRenderContext {
	return {
		analysis: params.analysis,
		monthIndex: params.monthIndex,
		hourIndex: params.hourIndex,
		timeIndex: params.timeIndex,
		selectionKey: params.selectionKey,
		colorMode: params.colorMode,
		metricType: 'utci',
		rangeOverride: params.rangeOverride ?? null
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
			rangeOverride: publishedRenderContext.rangeOverride ?? undefined
		};
	}

	if (params.analysis == null) {
		return null;
	}

	const monthIndex = params.viewerState.currentMonth ?? 7;
	const hourIndex = params.viewerState.currentHour ?? 0;
	return {
		analysis: params.analysis,
		hourIndex: getEffectiveHourIndex(params.analysis, hourIndex, monthIndex),
		monthIndex,
		colorMode: params.viewerState.colorMode ?? 'normalized',
		metricType: 'utci',
		rangeOverride: params.rangeOverride ?? undefined
	};
}
