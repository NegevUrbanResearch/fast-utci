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
});
