import {
	shouldSuppressTooltipMotion,
	type TooltipMotionSuppressionState,
} from '$lib/services/tooltipMotionSuppression';

export function getMainRouteTooltipHoverPolicy(params: {
	tooltipMotionSuppression: TooltipMotionSuppressionState;
	now: number;
	lastTooltipUpdate: number;
	throttleMs: number;
}): {
	shouldSuppress: boolean;
	shouldThrottle: boolean;
	nextTooltipUpdate: number;
} {
	if (
		shouldSuppressTooltipMotion(params.tooltipMotionSuppression, params.now)
	) {
		return {
			shouldSuppress: true,
			shouldThrottle: false,
			nextTooltipUpdate: params.lastTooltipUpdate,
		};
	}

	if (params.now - params.lastTooltipUpdate < params.throttleMs) {
		return {
			shouldSuppress: false,
			shouldThrottle: true,
			nextTooltipUpdate: params.lastTooltipUpdate,
		};
	}

	return {
		shouldSuppress: false,
		shouldThrottle: false,
		nextTooltipUpdate: params.now,
	};
}

export function resolveMainRouteTooltipTarget<TMesh, TAnalysis>(params: {
	baseMesh: TMesh;
	baseAnalysis: TAnalysis;
	baseSceneTimeIndex?: number | null;
	comparisonMesh: TMesh | null;
	comparisonAnalysis: TAnalysis | null | undefined;
	comparisonSceneTimeIndex?: number | null;
	useLiveUtciOnMainRoute: boolean;
	isComparing: boolean;
	mouseClientX: number;
	mainViewportRect: { left: number; width: number } | null;
	curtainPosition: number;
	viewerCurrentHour: number;
}): {
	meshToRaycast: TMesh;
	analysisToUse: TAnalysis;
	tooltipHourIndex: number;
} {
	let meshToRaycast = params.baseMesh;
	let analysisToUse = params.baseAnalysis;
	let tooltipHourIndex = params.useLiveUtciOnMainRoute
		? (params.baseSceneTimeIndex ?? params.viewerCurrentHour)
		: params.viewerCurrentHour;

	if (
		params.isComparing &&
		params.mainViewportRect &&
		params.mainViewportRect.width > 0
	) {
		const mouseXRelative =
			(params.mouseClientX - params.mainViewportRect.left) /
			params.mainViewportRect.width;

		if (
			mouseXRelative > params.curtainPosition &&
			params.comparisonMesh &&
			params.comparisonAnalysis
		) {
			meshToRaycast = params.comparisonMesh;
			analysisToUse = params.comparisonAnalysis;
			tooltipHourIndex = params.useLiveUtciOnMainRoute
				? (params.comparisonSceneTimeIndex ?? params.viewerCurrentHour)
				: params.viewerCurrentHour;
		}
	}

	return {
		meshToRaycast,
		analysisToUse,
		tooltipHourIndex,
	};
}
