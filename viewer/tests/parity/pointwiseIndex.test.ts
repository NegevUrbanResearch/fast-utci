import { describe, expect, it } from 'vitest';
import { flatIndexFromPointwise, pointwiseIndexFromFlat } from '$lib/parity/pointwiseIndex';

describe('pointwiseIndex', () => {
	it('maps flat indices to point/hour for point-major layout', () => {
		expect(pointwiseIndexFromFlat(0, 24)).toEqual({ pointIndex: 0, hourIndex: 0 });
		expect(pointwiseIndexFromFlat(23, 24)).toEqual({ pointIndex: 0, hourIndex: 23 });
		expect(pointwiseIndexFromFlat(24, 24)).toEqual({ pointIndex: 1, hourIndex: 0 });
		expect(pointwiseIndexFromFlat(73, 24)).toEqual({ pointIndex: 3, hourIndex: 1 });
	});

	it('round-trips point/hour to flat index', () => {
		const flat = flatIndexFromPointwise(7, 5, 24);
		expect(flat).toBe(173);
		expect(pointwiseIndexFromFlat(flat, 24)).toEqual({ pointIndex: 7, hourIndex: 5 });
	});

	it('throws on invalid inputs', () => {
		expect(() => pointwiseIndexFromFlat(-1, 24)).toThrow(/flatIndex/i);
		expect(() => pointwiseIndexFromFlat(0, 0)).toThrow(/numHours/i);
		expect(() => flatIndexFromPointwise(-1, 0, 24)).toThrow(/pointIndex/i);
		expect(() => flatIndexFromPointwise(0, -1, 24)).toThrow(/hourIndex/i);
		expect(() => flatIndexFromPointwise(0, 24, 24)).toThrow(/hourIndex.*numHours/i);
	});
});
