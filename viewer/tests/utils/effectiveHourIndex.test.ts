import { describe, it, expect } from 'vitest';
import { getEffectiveHourIndex } from '$lib/utils/effectiveHourIndex';
import type { Analysis } from '$lib/types/analysis';

describe('getEffectiveHourIndex', () => {
	it('returns hourIndex when analysis has no num_months', () => {
		const analysis = {
			metadata: { num_months: undefined },
			data: { numHours: 24 }
		} as unknown as Analysis;
		expect(getEffectiveHourIndex(analysis, 12, 7)).toBe(12);
	});

	it('returns monthIndex*24 + hourIndex when analysis has num_months=12', () => {
		const analysis = {
			metadata: { num_months: 12 },
			data: { numHours: 288 }
		} as unknown as Analysis;
		expect(getEffectiveHourIndex(analysis, 12, 0)).toBe(12);   // Jan, noon
		expect(getEffectiveHourIndex(analysis, 12, 7)).toBe(180);  // Aug, noon (7*24+12)
	});
});
