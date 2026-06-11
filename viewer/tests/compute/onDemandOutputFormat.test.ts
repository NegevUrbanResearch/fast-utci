import { describe, expect, it } from 'vitest';
import {
	ON_DEMAND_OUTPUT_FORMATS,
	F32_METRIC_OUTPUT_LAYOUT,
	F32_METRIC_OUTPUT_TYPES,
	getOnDemandOutputFormat
} from '$lib/compute/on-demand/onDemandOutputFormat';

describe('onDemandOutputFormat', () => {
	it('returns the baseline f32-utci format metadata', () => {
		expect(getOnDemandOutputFormat('f32-utci')).toMatchObject({
			id: 'f32-utci',
			bytesPerPoint: 4,
			includesMrt: false,
			requiresPacking: false
		});
	});

	it('returns the packed-mrt-utci format metadata', () => {
		expect(getOnDemandOutputFormat('packed-mrt-utci')).toMatchObject({
			id: 'packed-mrt-utci',
			bytesPerPoint: 4,
			includesMrt: true,
			requiresPacking: true
		});
	});

	it('exposes only the supported output format ids', () => {
		expect(Object.keys(ON_DEMAND_OUTPUT_FORMATS).sort()).toEqual(
			['f32-utci', 'packed-mrt-utci'].sort()
		);
	});

	it('keeps each format entry id aligned with its registry key', () => {
		expect(ON_DEMAND_OUTPUT_FORMATS['f32-utci'].id).toBe('f32-utci');
		expect(ON_DEMAND_OUTPUT_FORMATS['packed-mrt-utci'].id).toBe('packed-mrt-utci');
	});

	it('does not allow shared format metadata to be mutated through the getter', () => {
		const format = getOnDemandOutputFormat('f32-utci');

		expect(Object.isFrozen(ON_DEMAND_OUTPUT_FORMATS)).toBe(true);
		expect(Object.isFrozen(format)).toBe(true);
		expect(() => {
			(format as { description: string }).description = 'mutated';
		}).toThrow(TypeError);
		expect(getOnDemandOutputFormat('f32-utci').description).toBe(
			'Baseline bridge format with one f32 UTCI value per point.'
		);
	});

	it('exposes the metric-aware f32 output contract without adding a legacy format id', () => {
		expect(F32_METRIC_OUTPUT_LAYOUT).toBe('one-f32-per-point');
		expect(F32_METRIC_OUTPUT_TYPES).toEqual(['utci', 'shading_index']);
		expect(Object.keys(ON_DEMAND_OUTPUT_FORMATS).sort()).toEqual(
			['f32-utci', 'packed-mrt-utci'].sort()
		);
	});
});
