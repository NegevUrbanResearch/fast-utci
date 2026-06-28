import { describe, expect, it } from 'vitest';
import {
	buildDebugSelectedHourDispatchCounters,
	shouldUseDebugSharedSelectedHourHost
} from '$lib/debug/debugSelectedHourMode';

describe('debug selected-hour mode', () => {
	it('uses the shared selected-hour host only for normal non-parity f32 mode', () => {
		expect(
			shouldUseDebugSharedSelectedHourHost({
				onDemandPrototypeEnabled: true,
				debugOnDemandMode: 'f32',
				parityMode: false,
				normalCollectMode: false,
				strictExposureOnlyEnabled: false,
				compareOneHourEnabled: false
			})
		).toBe(true);
	});

	it('keeps parity, collect, strict exposure, and one-hour comparison on legacy debug paths', () => {
		const base = {
			onDemandPrototypeEnabled: true,
			debugOnDemandMode: 'f32' as const,
			parityMode: false,
			normalCollectMode: false,
			strictExposureOnlyEnabled: false,
			compareOneHourEnabled: false
		};

		expect(shouldUseDebugSharedSelectedHourHost({ ...base, parityMode: true })).toBe(false);
		expect(shouldUseDebugSharedSelectedHourHost({ ...base, normalCollectMode: true })).toBe(false);
		expect(shouldUseDebugSharedSelectedHourHost({ ...base, strictExposureOnlyEnabled: true })).toBe(false);
		expect(shouldUseDebugSharedSelectedHourHost({ ...base, compareOneHourEnabled: true })).toBe(false);
		expect(shouldUseDebugSharedSelectedHourHost({ ...base, debugOnDemandMode: 'off' })).toBe(false);
		expect(shouldUseDebugSharedSelectedHourHost({ ...base, onDemandPrototypeEnabled: false })).toBe(false);
	});

	it('builds explicit legacy dispatch counters for diagnostics', () => {
		expect(
			buildDebugSelectedHourDispatchCounters({
				legacySelectedHourDispatchCount: 2,
				legacyScrubScheduleCount: 3
			})
		).toEqual({
			legacySelectedHourDispatchCount: 2,
			legacyScrubScheduleCount: 3
		});
	});
});
