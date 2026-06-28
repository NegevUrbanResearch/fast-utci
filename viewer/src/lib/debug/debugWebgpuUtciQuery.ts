export type DebugWebgpuUtciCollectMode = 'off' | 'normal';
export type DebugWebgpuUtciOnDemandMode = 'off' | 'f32';

export type DebugWebgpuUtciQueryState = {
	parityMode: boolean;
	collectMode: DebugWebgpuUtciCollectMode;
	debugOnDemandMode: DebugWebgpuUtciOnDemandMode;
	binComparisonEnabled: boolean;
	binComparisonValid: boolean;
};

export function parseDebugWebgpuUtciQuery(
	searchParams: URLSearchParams
): DebugWebgpuUtciQueryState {
	const parityParam = searchParams.get('parity');
	const parityMode =
		parityParam === null ? searchParams.get('collect') !== 'normal' : parityParam === '1';
	const collectMode: DebugWebgpuUtciCollectMode =
		!parityMode && searchParams.get('collect') === 'normal' ? 'normal' : 'off';

	let debugOnDemandMode: DebugWebgpuUtciOnDemandMode;
	if (searchParams.has('utciOnDemand')) {
		debugOnDemandMode = searchParams.get('utciOnDemand') === 'f32' ? 'f32' : 'off';
	} else if (searchParams.get('onDemandPrototype') === '1') {
		debugOnDemandMode = 'f32';
	} else if (collectMode === 'normal') {
		debugOnDemandMode = 'off';
	} else {
		debugOnDemandMode = 'f32';
	}

	const monthIndexParam = searchParams.get('monthIndex');
	const monthIndexRaw = Number(monthIndexParam ?? '7');
	const monthIndex = Number.isInteger(monthIndexRaw)
		? Math.min(Math.max(monthIndexRaw, 0), 11)
		: null;

	return {
		parityMode,
		collectMode,
		debugOnDemandMode,
		binComparisonEnabled: parityMode,
		binComparisonValid: parityMode && monthIndex === 7
	};
}
