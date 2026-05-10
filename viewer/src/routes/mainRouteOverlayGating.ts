export type MainRouteOverlayGatingParams = {
	modelLoading: boolean;
	useLiveUtciOnMainRoute: boolean;
	baseLiveLoading: boolean;
	baseHasVisibleLiveSurface: boolean;
	isComparing: boolean;
	comparisonModelLoading: boolean;
	comparisonLiveLoading: boolean;
	comparisonHasVisibleLiveSurface: boolean;
};

export type MainRouteOverlayGatingState = {
	baseNeedsLiveComputeOverlay: boolean;
	comparisonNeedsLiveComputeOverlay: boolean;
	showOverlay: boolean;
	showComparisonModeOverlay: boolean;
};

export function getMainRouteOverlayGating(
	params: MainRouteOverlayGatingParams
): MainRouteOverlayGatingState {
	const baseNeedsLiveComputeOverlay =
		params.useLiveUtciOnMainRoute &&
		params.baseLiveLoading &&
		!params.baseHasVisibleLiveSurface;
	const comparisonNeedsLiveComputeOverlay =
		params.isComparing &&
		params.useLiveUtciOnMainRoute &&
		params.comparisonLiveLoading &&
		!params.comparisonHasVisibleLiveSurface;
	const showOverlay =
		params.modelLoading ||
		baseNeedsLiveComputeOverlay ||
		(params.isComparing &&
			(params.comparisonModelLoading || comparisonNeedsLiveComputeOverlay));
	const showComparisonModeOverlay =
		params.isComparing &&
		(params.comparisonModelLoading || comparisonNeedsLiveComputeOverlay) &&
		!params.modelLoading &&
		!baseNeedsLiveComputeOverlay;

	return {
		baseNeedsLiveComputeOverlay,
		comparisonNeedsLiveComputeOverlay,
		showOverlay,
		showComparisonModeOverlay
	};
}
