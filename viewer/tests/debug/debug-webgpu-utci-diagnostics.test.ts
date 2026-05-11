import { describe, expect, it } from 'vitest';
import {
	deriveDebugWebgpuUtciDiagnosticsState,
	shouldExposeDebugWindowDiagnostics
} from '../../src/lib/debug/debugWebgpuUtciDiagnostics';

describe('debugWebgpuUtciDiagnostics', () => {
	it('keeps on-demand enabled only for f32 debug mode', () => {
		const state = deriveDebugWebgpuUtciDiagnosticsState({
			parityMode: false,
			collectMode: 'off',
			debugOnDemandMode: 'f32',
			utciRenderMode: 'auto',
			selectedMonthIndex: 7,
			selectedHourIndex: 12,
			selectedTimeIndex: 180
		});
		expect(state.onDemandEnabled).toBe(true);
		expect(state.renderMode).toBe('auto');
	});

	it('keeps normal collect mode distinct from on-demand diagnostics', () => {
		const state = deriveDebugWebgpuUtciDiagnosticsState({
			parityMode: false,
			collectMode: 'normal',
			debugOnDemandMode: 'off',
			utciRenderMode: 'data',
			selectedMonthIndex: 7,
			selectedHourIndex: 12,
			selectedTimeIndex: 180
		});
		expect(state).toMatchObject({
			onDemandEnabled: false,
			collectNormalMode: true
		});
	});

	it('allows bin comparison only as debug/parity behavior', () => {
		const state = deriveDebugWebgpuUtciDiagnosticsState({
			parityMode: true,
			collectMode: 'off',
			debugOnDemandMode: 'f32',
			utciRenderMode: 'gpu',
			selectedMonthIndex: 7,
			selectedHourIndex: 12,
			selectedTimeIndex: 180
		});
		expect(state.binComparisonEnabled).toBe(true);
	});

	it('carries bin comparison validity separately from parity enablement', () => {
		const state = deriveDebugWebgpuUtciDiagnosticsState({
			parityMode: true,
			collectMode: 'off',
			debugOnDemandMode: 'f32',
			utciRenderMode: 'auto',
			selectedMonthIndex: 3,
			selectedHourIndex: 9,
			selectedTimeIndex: 81,
			selectedHourEngine: 'legacy-debug',
			binComparisonValid: false
		});

		expect(state.binComparisonEnabled).toBe(true);
		expect(state.binComparisonValid).toBe(false);
	});

	it('reports legacy debug selected-hour execution until shared host migration is proven', () => {
		const state = deriveDebugWebgpuUtciDiagnosticsState({
			parityMode: false,
			collectMode: 'off',
			debugOnDemandMode: 'f32',
			utciRenderMode: 'auto',
			selectedMonthIndex: 7,
			selectedHourIndex: 12,
			selectedTimeIndex: 180,
			selectedHourEngine: 'legacy-debug'
		});

		expect(state.onDemandEnabled).toBe(true);
		expect(state.binComparisonEnabled).toBe(false);
		expect(state.selectedHourEngine).toBe('legacy-debug');
		expect(state.selectedHourRuntimeContract.selectedHourEngine).toBe('legacy-debug');
		expect(state.selectedHourRuntimeContract.strongVisibleGpuPath).toBe(false);
	});

	it('keeps parity comparison explicitly debug-only', () => {
		const state = deriveDebugWebgpuUtciDiagnosticsState({
			parityMode: true,
			collectMode: 'off',
			debugOnDemandMode: 'off',
			utciRenderMode: 'auto',
			selectedMonthIndex: 7,
			selectedHourIndex: 12,
			selectedTimeIndex: 180,
			selectedHourEngine: 'legacy-debug'
		});

		expect(state.onDemandEnabled).toBe(false);
		expect(state.binComparisonEnabled).toBe(true);
		expect(state.selectedHourEngine).toBe('legacy-debug');
	});

	it('exposes debug window diagnostics while the helper remains a debug-only gate', () => {
		const state = deriveDebugWebgpuUtciDiagnosticsState({
			parityMode: false,
			collectMode: 'off',
			debugOnDemandMode: 'off',
			utciRenderMode: 'data',
			selectedMonthIndex: 7,
			selectedHourIndex: 12,
			selectedTimeIndex: 180
		});
		expect(shouldExposeDebugWindowDiagnostics(state)).toBe(true);
	});

	it('keeps debug window diagnostics enabled before reactive state is initialized', () => {
		expect(shouldExposeDebugWindowDiagnostics(undefined)).toBe(true);
	});

	it('carries render and selection fields with the derived diagnostics state', () => {
		const state = deriveDebugWebgpuUtciDiagnosticsState({
			parityMode: false,
			collectMode: 'off',
			debugOnDemandMode: 'f32',
			utciRenderMode: 'gpu',
			selectedMonthIndex: 0,
			selectedHourIndex: 6,
			selectedTimeIndex: 6
		});

		expect(state).toMatchObject({
			renderMode: 'gpu',
			selection: {
				monthIndex: 0,
				hourIndex: 6,
				timeIndex: 6
			}
		});
	});

	it('keeps shared-host diagnostics honest before migration is proven', () => {
		const state = deriveDebugWebgpuUtciDiagnosticsState({
			parityMode: false,
			collectMode: 'off',
			debugOnDemandMode: 'f32',
			utciRenderMode: 'auto',
			selectedMonthIndex: 7,
			selectedHourIndex: 0,
			selectedTimeIndex: 168,
			selectedHourEngine: 'legacy-debug',
			legacySelectedHourDispatchCount: 1,
			legacyScrubScheduleCount: 1
		});

		expect(state.selectedHourEngine).toBe('legacy-debug');
		expect(state.legacySelectedHourDispatchCount).toBe(1);
		expect(state.legacyScrubScheduleCount).toBe(1);
	});

	it('allows shared-host diagnostics only with zero legacy dispatch counters', () => {
		const state = deriveDebugWebgpuUtciDiagnosticsState({
			parityMode: false,
			collectMode: 'off',
			debugOnDemandMode: 'f32',
			utciRenderMode: 'auto',
			selectedMonthIndex: 7,
			selectedHourIndex: 0,
			selectedTimeIndex: 168,
			selectedHourEngine: 'shared-host',
			legacySelectedHourDispatchCount: 0,
			legacyScrubScheduleCount: 0
		});

		expect(state.selectedHourEngine).toBe('shared-host');
		expect(state.legacySelectedHourDispatchCount).toBe(0);
		expect(state.legacyScrubScheduleCount).toBe(0);
		expect(state.selectedHourRuntimeContract.selectedHourEngine).toBe('shared-host');
		expect(state.selectedHourRuntimeContract.hasLegacyDebugOverlap).toBe(false);
	});

	it('preserves shared-host counter evidence when legacy counters are nonzero', () => {
		const state = deriveDebugWebgpuUtciDiagnosticsState({
			parityMode: false,
			collectMode: 'off',
			debugOnDemandMode: 'f32',
			utciRenderMode: 'auto',
			selectedMonthIndex: 7,
			selectedHourIndex: 0,
			selectedTimeIndex: 168,
			selectedHourEngine: 'shared-host',
			legacySelectedHourDispatchCount: 1,
			legacyScrubScheduleCount: 0
		});

		expect(state.selectedHourEngine).toBe('shared-host');
		expect(state.legacySelectedHourDispatchCount).toBe(1);
	});
});
