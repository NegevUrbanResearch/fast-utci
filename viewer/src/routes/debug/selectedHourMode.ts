import {
	deriveDebugWebgpuUtciDiagnosticsState,
	type DebugSelectedHourEngine,
	type DebugWebgpuUtciDiagnosticsState
} from '$lib/debug/debugWebgpuUtciDiagnostics';
import {
	buildDebugSelectedHourDispatchCounters,
	shouldUseDebugSharedSelectedHourHost,
	type DebugSelectedHourDispatchCounters
} from '$lib/debug/debugSelectedHourMode';
import type { UtciRenderMode } from '$lib/utciRenderMode';

export type DebugRouteSelectedHourPolicyInputs = {
	browserEnabled: boolean;
	parityMode: boolean;
	normalCollectMode: boolean;
	debugOnDemandMode: 'off' | 'f32';
	utciRenderMode: UtciRenderMode;
	selectedMonthIndex: number;
	selectedHourIndex: number;
	selectedTimeIndex: number;
	legacySelectedHourDispatchCounters: DebugSelectedHourDispatchCounters;
	strictExposureOnlyRequested: boolean;
	compareOneHourRequested: boolean;
	selectedHourEngine?: DebugSelectedHourEngine;
};

export type DebugRouteSelectedHourPolicy = {
	debugDiagnosticsState: DebugWebgpuUtciDiagnosticsState;
	onDemandPrototypeEnabled: boolean;
	useDebugSharedSelectedHourHost: boolean;
	legacySelectedHourDispatchCounters: DebugSelectedHourDispatchCounters;
};

export function deriveDebugRouteSelectedHourPolicy(
	inputs: DebugRouteSelectedHourPolicyInputs
): DebugRouteSelectedHourPolicy {
	const legacySelectedHourDispatchCounters = buildDebugSelectedHourDispatchCounters(
		inputs.legacySelectedHourDispatchCounters
	);
	const debugDiagnosticsState = deriveDebugWebgpuUtciDiagnosticsState({
		parityMode: inputs.parityMode,
		collectMode: inputs.normalCollectMode ? 'normal' : 'off',
		debugOnDemandMode: inputs.debugOnDemandMode,
		utciRenderMode: inputs.utciRenderMode,
		selectedMonthIndex: inputs.selectedMonthIndex,
		selectedHourIndex: inputs.selectedHourIndex,
		selectedTimeIndex: inputs.selectedTimeIndex,
		selectedHourEngine: inputs.selectedHourEngine ?? 'legacy-debug',
		binComparisonValid: inputs.parityMode && inputs.selectedMonthIndex === 7,
		...legacySelectedHourDispatchCounters
	});
	const onDemandPrototypeEnabled = inputs.browserEnabled && debugDiagnosticsState.onDemandEnabled;
	const useDebugSharedSelectedHourHost = shouldUseDebugSharedSelectedHourHost({
		onDemandPrototypeEnabled,
		debugOnDemandMode: inputs.debugOnDemandMode,
		parityMode: inputs.parityMode,
		normalCollectMode: inputs.normalCollectMode,
		strictExposureOnlyEnabled: inputs.strictExposureOnlyRequested,
		compareOneHourEnabled: inputs.compareOneHourRequested
	});

	return {
		debugDiagnosticsState,
		onDemandPrototypeEnabled,
		useDebugSharedSelectedHourHost,
		legacySelectedHourDispatchCounters
	};
}
