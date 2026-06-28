import { describe, expect, it } from 'vitest';
import { createFullDayAnalysis } from './live-selected-hour-route-host.test-support';
import {
	createLiveSelectedHourPublishedRenderContext,
	resolveLiveSelectedHourSurfaceRenderState,
	type LiveSelectedHourPublishedRenderContext
} from '$lib/compute/selected-hour/liveSelectedHourRenderContext';

describe('liveSelectedHourRenderContext', () => {
	it('prefers the published visible render context over pending viewer-state changes', () => {
		const analysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});
		const publishedRenderContext: LiveSelectedHourPublishedRenderContext = {
			analysis,
			monthIndex: 7,
			hourIndex: 12,
			timeIndex: 180,
			selectionKey: 'Ben-Gurion/base|7|12|180',
			publicationPhase: 'scrub',
			colorMode: 'discrete',
			metricType: 'utci',
			rangeOverride: { utciMin: 17, utciMax: 42 }
		};

		const result = resolveLiveSelectedHourSurfaceRenderState({
			analysis,
			viewerState: {
				currentHour: 3,
				currentMonth: 1,
				colorMode: 'normalized',
				metricType: 'utci'
			},
			publishedRenderContext
		});

		expect(result).toEqual({
			analysis,
			hourIndex: 180,
			monthIndex: 7,
			colorMode: 'discrete',
			metricType: 'utci',
			rangeOverride: { utciMin: 17, utciMax: 42 }
		});
	});

	it('falls back to the current viewer state when no published visible render context exists', () => {
		const analysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});

		const result = resolveLiveSelectedHourSurfaceRenderState({
			analysis,
			viewerState: {
				currentHour: 3,
				currentMonth: 1,
				colorMode: 'normalized',
				metricType: 'utci'
			},
			rangeOverride: { utciMin: 18, utciMax: 30 }
		});

		expect(result).toEqual({
			analysis,
			hourIndex: 27,
			monthIndex: 1,
			colorMode: 'normalized',
			metricType: 'utci',
			rangeOverride: { utciMin: 18, utciMax: 30 }
		});
	});

	it('publishes shading-index render contexts without UTCI range overrides', () => {
		const analysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});

		const publishedRenderContext = createLiveSelectedHourPublishedRenderContext({
			analysis,
			monthIndex: 7,
			hourIndex: 0,
			timeIndex: 168,
			selectionKey: 'Ben-Gurion/base|shading_index|7',
			publicationPhase: 'initial',
			colorMode: 'normalized',
			metricType: 'shading_index',
			rangeOverride: { utciMin: 18, utciMax: 30 }
		});

		expect(publishedRenderContext.metricType).toBe('shading_index');
		expect(publishedRenderContext.rangeOverride).toBeNull();
	});

	it('uses viewer metric type for fallback render state and suppresses UTCI range overrides for shading', () => {
		const analysis = createFullDayAnalysis({
			label: 'base',
			sourceAnalysisId: 'Ben-Gurion/base',
			baseMin: 18,
			baseMax: 30
		});

		const result = resolveLiveSelectedHourSurfaceRenderState({
			analysis,
			viewerState: {
				currentHour: 12,
				currentMonth: 7,
				colorMode: 'normalized',
				metricType: 'shading_index'
			},
			rangeOverride: { utciMin: 18, utciMax: 30 }
		});

		expect(result).toMatchObject({
			analysis,
			hourIndex: 180,
			monthIndex: 7,
			colorMode: 'normalized',
			metricType: 'shading_index'
		});
		expect(result?.rangeOverride).toBeUndefined();
	});
});
