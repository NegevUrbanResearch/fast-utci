import { describe, expect, it } from 'vitest';
import type { Analysis } from '$lib/types/analysis';
import {
	resolveAcceptedGpuResidentUtciRange,
	resolveLiveGpuResidentUtciRange,
	resolveLiveSelectedHourTimeIndex
} from '$lib/compute/selected-hour/liveUtciSelectedHour';

function createAugustAnalysis(): Analysis {
	return {
		metadata: {
			analysis_type: 'full_day',
			num_positions: 2,
			hours: Array.from({ length: 24 }, (_, hour) => `${hour}:00`),
			date: '20250815',
			utci_range: { min: 23, max: 38 },
			hour_statistics: Array.from({ length: 24 }, (_, hour) => ({
				hour,
				min: 20 + hour,
				max: 30 + hour,
				mean: 25 + hour
			})),
			grid_size: 2,
			coordinate_system: 'xy_ground',
			model_file: 'model.glb'
		},
		data: {
			numPositions: 2,
			numHours: 24,
			positions: new Float32Array([0, 0, 0, 1, 0, 0]),
			utciByHour: Array.from({ length: 24 }, () => new Float32Array([20, 30]))
		}
	};
}

describe('resolveLiveSelectedHourTimeIndex', () => {
	it('resolves live WebGPU month/hour selections to full-year time indices', () => {
		expect(resolveLiveSelectedHourTimeIndex({ monthIndex: 0, hourIndex: 0 })).toBe(0);
		expect(resolveLiveSelectedHourTimeIndex({ monthIndex: 7, hourIndex: 12 })).toBe(180);
		expect(resolveLiveSelectedHourTimeIndex({ monthIndex: 10, hourIndex: 18 })).toBe(258);
	});

	it('keeps the metadata-aware range resolver for debug-route callers', () => {
		const analysis = createAugustAnalysis();

		expect(
			resolveAcceptedGpuResidentUtciRange({
				base: analysis,
				monthIndex: 7,
				hourIndex: 12,
				colorMode: 'normalized'
			})
		).toEqual({ min: 23, max: 38 });

		expect(
			resolveAcceptedGpuResidentUtciRange({
				base: analysis,
				monthIndex: 0,
				hourIndex: 12,
				colorMode: 'normalized'
			})
		).toEqual({ min: 23, max: 38 });
	});

	it('uses selected-hour WebGPU values when available for live coloring', () => {
		expect(
			resolveLiveGpuResidentUtciRange({
				selectedHourUtci: new Float32Array([8, 14, 20])
			})
		).toEqual({ min: 8, max: 20 });
	});
});
