import type { UtciRenderMode } from '$lib/utciRenderMode';

export type DebugSelectedHourEngine = 'legacy-debug' | 'shared-host';

export type DebugWebgpuUtciDiagnosticsInputs = {
	parityMode: boolean;
	collectMode: 'off' | 'normal';
	debugOnDemandMode: 'off' | 'f32';
	utciRenderMode: UtciRenderMode;
	selectedMonthIndex: number;
	selectedHourIndex: number;
	selectedTimeIndex: number;
	selectedHourEngine?: DebugSelectedHourEngine;
	binComparisonValid?: boolean;
	legacySelectedHourDispatchCount?: number;
	legacyScrubScheduleCount?: number;
};

export type DebugWebgpuUtciDiagnosticsState = {
	onDemandEnabled: boolean;
	binComparisonEnabled: boolean;
	binComparisonValid: boolean;
	collectNormalMode: boolean;
	windowDiagnosticsEnabled: boolean;
	renderMode: UtciRenderMode;
	selectedHourEngine: DebugSelectedHourEngine;
	legacySelectedHourDispatchCount: number;
	legacyScrubScheduleCount: number;
	selection: {
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
	};
};

export function deriveDebugWebgpuUtciDiagnosticsState(
	inputs: DebugWebgpuUtciDiagnosticsInputs
): DebugWebgpuUtciDiagnosticsState {
	return {
		onDemandEnabled: inputs.debugOnDemandMode === 'f32',
		binComparisonEnabled: inputs.parityMode,
		binComparisonValid: inputs.binComparisonValid ?? false,
		collectNormalMode: inputs.collectMode === 'normal',
		windowDiagnosticsEnabled: true,
		renderMode: inputs.utciRenderMode,
		selectedHourEngine: inputs.selectedHourEngine ?? 'legacy-debug',
		legacySelectedHourDispatchCount: inputs.legacySelectedHourDispatchCount ?? 0,
		legacyScrubScheduleCount: inputs.legacyScrubScheduleCount ?? 0,
		selection: {
			monthIndex: inputs.selectedMonthIndex,
			hourIndex: inputs.selectedHourIndex,
			timeIndex: inputs.selectedTimeIndex
		}
	};
}

export function shouldExposeDebugWindowDiagnostics(
	state: DebugWebgpuUtciDiagnosticsState | null | undefined
): boolean {
	return state?.windowDiagnosticsEnabled ?? true;
}
