import { describe, expect, it } from 'vitest';

import {
	getMainRouteTooltipHoverPolicy,
	resolveMainRouteTooltipTarget
} from '../../src/routes/main/tooltip';
import {
	createTooltipMotionSuppressionState,
	setTooltipMotionPointerDown
} from '$lib/services/tooltipMotionSuppression';

describe('main route tooltip helpers', () => {
	it('suppresses tooltip hover work while pointer interaction suppression is active', () => {
		const suppressionState = setTooltipMotionPointerDown(
			createTooltipMotionSuppressionState(),
			true,
			100
		);

		const policy = getMainRouteTooltipHoverPolicy({
			tooltipMotionSuppression: suppressionState,
			now: 110,
			lastTooltipUpdate: 0,
			throttleMs: 16
		});

		expect(policy).toEqual({
			shouldSuppress: true,
			shouldThrottle: false,
			nextTooltipUpdate: 0
		});
	});

	it('routes comparison-side hover to the comparison surface when the cursor crosses the curtain', () => {
		const target = resolveMainRouteTooltipTarget({
			baseMesh: { id: 'base' },
			baseAnalysis: { id: 'base-analysis' },
			baseSceneTimeIndex: 10,
			comparisonMesh: { id: 'comparison' },
			comparisonAnalysis: { id: 'comparison-analysis' },
			comparisonSceneTimeIndex: 22,
			useLiveUtciOnMainRoute: false,
			isComparing: true,
			mouseClientX: 750,
			mainViewportRect: { left: 0, width: 1000 },
			curtainPosition: 0.5,
			viewerCurrentHour: 14
		});

		expect(target).toEqual({
			meshToRaycast: { id: 'comparison' },
			analysisToUse: { id: 'comparison-analysis' },
			tooltipHourIndex: 14
		});
	});

	it('prefers the live selected-hour analysis and render-context time index when active', () => {
		const target = resolveMainRouteTooltipTarget({
			baseMesh: { id: 'base' },
			baseAnalysis: { id: 'display-analysis' },
			baseSceneTimeIndex: 123,
			comparisonMesh: null,
			comparisonAnalysis: null,
			comparisonSceneTimeIndex: undefined,
			useLiveUtciOnMainRoute: true,
			isComparing: false,
			mouseClientX: 100,
			mainViewportRect: null,
			curtainPosition: 0.5,
			viewerCurrentHour: 9
		});

		expect(target).toEqual({
			meshToRaycast: { id: 'base' },
			analysisToUse: { id: 'display-analysis' },
			tooltipHourIndex: 123
		});
	});

	it('uses event-time comparison mesh when the cursor is past the curtain', () => {
		const baseMesh = { id: 'base' };
		const comparisonMesh = { id: 'comparison' };
		const baseAnalysis = { id: 'base-analysis' };
		const comparisonAnalysis = { id: 'comparison-analysis' };

		const target = resolveMainRouteTooltipTarget({
			baseMesh,
			baseAnalysis,
			baseSceneTimeIndex: 180,
			comparisonMesh,
			comparisonAnalysis,
			comparisonSceneTimeIndex: 181,
			useLiveUtciOnMainRoute: true,
			isComparing: true,
			mouseClientX: 75,
			mainViewportRect: { left: 0, width: 100 },
			curtainPosition: 0.5,
			viewerCurrentHour: 12
		});

		expect(target.meshToRaycast).toBe(comparisonMesh);
		expect(target.analysisToUse).toBe(comparisonAnalysis);
		expect(target.tooltipHourIndex).toBe(181);
	});
});
