import {
	parseDebugWebgpuUtciQuery,
	type DebugWebgpuUtciQueryState
} from '$lib/debug/debugWebgpuUtciQuery';

export type DebugRouteQueryState = DebugWebgpuUtciQueryState & {
	normalCollectMode: boolean;
};

export function parseDebugRouteQueryState(
	searchParams: URLSearchParams
): DebugRouteQueryState {
	const queryState = parseDebugWebgpuUtciQuery(searchParams);
	return {
		...queryState,
		normalCollectMode: queryState.collectMode === 'normal'
	};
}
