export function getInitialAnalysisId(
	search: string,
	defaultId: string
): string {
	if (!search) return defaultId;
	const params = new URLSearchParams(search);
	return params.get('analysis') || defaultId;
}

const MAIN_ROUTE_GRID_RESOLUTIONS = [10, 8, 6, 4, 2, 1, 0.5] as const;

export type MainRouteGridResolution = (typeof MAIN_ROUTE_GRID_RESOLUTIONS)[number];

export function parseMainRouteGridResolution(
	search: string | URLSearchParams | null | undefined,
	defaultResolution: MainRouteGridResolution = 2
): MainRouteGridResolution {
	if (!search) return defaultResolution;
	const params =
		typeof search === 'string'
			? new URLSearchParams(search)
			: search;
	const value = Number(params.get('gridResolution'));
	return MAIN_ROUTE_GRID_RESOLUTIONS.find((resolution) => resolution === value) ?? defaultResolution;
}
