export type DebugSelectedHourModeInput = {
	onDemandPrototypeEnabled: boolean;
	debugOnDemandMode: 'off' | 'f32';
	parityMode: boolean;
	normalCollectMode: boolean;
	strictExposureOnlyEnabled: boolean;
	compareOneHourEnabled: boolean;
};

export type DebugSelectedHourDispatchCounters = {
	legacySelectedHourDispatchCount: number;
	legacyScrubScheduleCount: number;
};

export function shouldUseDebugSharedSelectedHourHost(
	input: DebugSelectedHourModeInput
): boolean {
	return (
		input.onDemandPrototypeEnabled &&
		input.debugOnDemandMode === 'f32' &&
		!input.parityMode &&
		!input.normalCollectMode &&
		!input.strictExposureOnlyEnabled &&
		!input.compareOneHourEnabled
	);
}

export function buildDebugSelectedHourDispatchCounters(
	counters: DebugSelectedHourDispatchCounters
): DebugSelectedHourDispatchCounters {
	return {
		legacySelectedHourDispatchCount: counters.legacySelectedHourDispatchCount,
		legacyScrubScheduleCount: counters.legacyScrubScheduleCount
	};
}
