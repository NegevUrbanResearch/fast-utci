import type { UtciRenderMode } from '$lib/utciRenderMode';

export type DebugWebgpuUtciDiagnosticsInputs = {
	parityMode: boolean;
	collectMode: 'off' | 'normal';
	debugOnDemandMode: 'off' | 'f32';
	utciRenderMode: UtciRenderMode;
	selectedMonthIndex: number;
	selectedHourIndex: number;
	selectedTimeIndex: number;
};

export type DebugWebgpuUtciDiagnosticsState = {
	onDemandEnabled: boolean;
	binComparisonEnabled: boolean;
	collectNormalMode: boolean;
	windowDiagnosticsEnabled: boolean;
	renderMode: UtciRenderMode;
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
		collectNormalMode: inputs.collectMode === 'normal',
		windowDiagnosticsEnabled: true,
		renderMode: inputs.utciRenderMode,
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
