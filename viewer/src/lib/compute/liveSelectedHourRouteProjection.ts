import type { Analysis } from '$lib/types/analysis';
import type { LiveSelectedHourPublishedRenderContext } from '$lib/compute/liveSelectedHourRenderContext';
import type { LiveSelectedHourRouteState } from '$lib/compute/liveSelectedHourRouteHost';
import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/liveSelectedHourSurfaceIdentity';
import type { SelectedHourGpuResidentOutput } from '$lib/compute/liveUtciSelectedHourSession';

export type MainRouteLiveSceneProjection = {
	baseDisplayedAnalysis: Analysis | null;
	comparisonRendererDisplayAnalysis: Analysis | null | undefined;
	baseLiveReady: boolean;
	comparisonLiveReady: boolean;
	baseHasVisibleLiveSurface: boolean;
	comparisonHasVisibleLiveSurface: boolean;
	baseSceneAnalysis: Analysis | null;
	comparisonSceneAnalysis: Analysis | null | undefined;
	baseSceneRenderContext: LiveSelectedHourPublishedRenderContext | null;
	comparisonSceneRenderContext: LiveSelectedHourPublishedRenderContext | null | undefined;
	baseSceneSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null;
	comparisonSceneSurfaceIdentity: LiveSelectedHourSurfaceIdentity | null | undefined;
	basePendingGpuResidentOutput: SelectedHourGpuResidentOutput | null;
	comparisonPendingGpuResidentOutput: SelectedHourGpuResidentOutput | null;
	basePendingRenderUpdateStartedAt: number | undefined;
	comparisonPendingRenderUpdateStartedAt: number | undefined;
};

export function projectMainRouteLiveSceneState(params: {
	useLiveUtciOnMainRoute: boolean;
	isComparing: boolean;
	baseAnalysis: Analysis | null;
	comparisonAnalysis: Analysis | null;
	liveRouteState: LiveSelectedHourRouteState;
}): MainRouteLiveSceneProjection {
	const { useLiveUtciOnMainRoute, isComparing, baseAnalysis, comparisonAnalysis, liveRouteState } =
		params;

	const baseDisplayedAnalysis = useLiveUtciOnMainRoute
		? liveRouteState.baseDisplayAnalysis
		: baseAnalysis;
	const comparisonRendererDisplayAnalysis = !isComparing
		? undefined
		: useLiveUtciOnMainRoute
			? liveRouteState.comparisonDisplayAnalysis
			: comparisonAnalysis;
	const baseLiveReady = useLiveUtciOnMainRoute
		? liveRouteState.baseReady
		: baseAnalysis != null;
	const comparisonLiveReady = !isComparing
		? true
		: useLiveUtciOnMainRoute
			? liveRouteState.comparisonReady
			: comparisonAnalysis != null;
	const baseHasVisibleLiveSurface =
		useLiveUtciOnMainRoute && liveRouteState.baseHasVisibleLiveSurface;
	const comparisonHasVisibleLiveSurface =
		isComparing &&
		useLiveUtciOnMainRoute &&
		liveRouteState.comparisonHasVisibleLiveSurface;
	const baseBootstrapSurfaceIdentity = useLiveUtciOnMainRoute
		? liveRouteState.baseSceneSurfaceIdentity
		: null;
	const basePublishedSurfaceIdentity = useLiveUtciOnMainRoute
		? liveRouteState.baseSurfaceIdentity
		: null;
	const baseSceneSurfaceIdentity = baseBootstrapSurfaceIdentity ?? basePublishedSurfaceIdentity;
	const basePendingGpuResidentOutput =
		baseSceneSurfaceIdentity?.acceptedGpuResidentOutput ?? null;
	const baseHasBootstrapSceneSurface =
		useLiveUtciOnMainRoute &&
		baseSceneSurfaceIdentity != null &&
		liveRouteState.baseRenderContext != null;
	const baseSceneRenderContext =
		useLiveUtciOnMainRoute && (baseHasVisibleLiveSurface || baseHasBootstrapSceneSurface)
			? liveRouteState.baseRenderContext
			: null;
	const baseBootstrapAnalysis = baseSceneRenderContext?.analysis ?? baseDisplayedAnalysis;
	const baseSceneAnalysis = !useLiveUtciOnMainRoute
		? baseAnalysis
		: baseHasVisibleLiveSurface
			? baseDisplayedAnalysis
			: baseHasBootstrapSceneSurface
				? baseBootstrapAnalysis
				: null;
	const comparisonBootstrapSurfaceIdentity =
		isComparing && useLiveUtciOnMainRoute
			? liveRouteState.comparisonSceneSurfaceIdentity
			: undefined;
	const comparisonPublishedSurfaceIdentity =
		isComparing && useLiveUtciOnMainRoute
			? liveRouteState.comparisonSurfaceIdentity
			: undefined;
	const comparisonSceneSurfaceIdentity = !isComparing
		? undefined
		: comparisonBootstrapSurfaceIdentity ?? comparisonPublishedSurfaceIdentity ?? null;
	const comparisonPendingGpuResidentOutput =
		comparisonSceneSurfaceIdentity?.acceptedGpuResidentOutput ?? null;
	const comparisonHasBootstrapSceneSurface =
		isComparing &&
		useLiveUtciOnMainRoute &&
		comparisonSceneSurfaceIdentity != null &&
		liveRouteState.comparisonRenderContext != null;
	const comparisonSceneRenderContext = !isComparing
		? undefined
		: useLiveUtciOnMainRoute &&
			  (comparisonHasVisibleLiveSurface || comparisonHasBootstrapSceneSurface)
			? liveRouteState.comparisonRenderContext
			: null;
	const comparisonBootstrapAnalysis =
		comparisonSceneRenderContext?.analysis ?? comparisonRendererDisplayAnalysis;
	const comparisonSceneAnalysis = !isComparing
		? undefined
		: !useLiveUtciOnMainRoute
			? comparisonAnalysis
			: comparisonHasVisibleLiveSurface
				? comparisonRendererDisplayAnalysis
				: comparisonHasBootstrapSceneSurface
					? comparisonBootstrapAnalysis
					: null;

	return {
		baseDisplayedAnalysis,
		comparisonRendererDisplayAnalysis,
		baseLiveReady,
		comparisonLiveReady,
		baseHasVisibleLiveSurface,
		comparisonHasVisibleLiveSurface,
		baseSceneAnalysis,
		comparisonSceneAnalysis,
		baseSceneRenderContext,
		comparisonSceneRenderContext,
		baseSceneSurfaceIdentity,
		comparisonSceneSurfaceIdentity,
		basePendingGpuResidentOutput,
		comparisonPendingGpuResidentOutput,
		basePendingRenderUpdateStartedAt: useLiveUtciOnMainRoute
			? liveRouteState.baseSceneSurfaceIdentity?.pendingRenderUpdateStartedAt
			: undefined,
		comparisonPendingRenderUpdateStartedAt: useLiveUtciOnMainRoute
			? liveRouteState.comparisonSceneSurfaceIdentity?.pendingRenderUpdateStartedAt
			: undefined
	};
}
