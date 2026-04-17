import { describe, expect, it } from 'vitest';
import {
	extractTopMrtDeltas,
	flatIndexToHourPoint,
	type OptionalTermSeries
} from '$lib/parity/mrtWorstCellDiagnostics';

describe('mrtWorstCellDiagnostics helpers', () => {
	it('maps flat index to hour/point', () => {
		expect(flatIndexToHourPoint(9, 4)).toEqual({ hour: 1, pointIndex: 2 });
	});

	it('extracts top-N MRT rows with component deltas and optional terms', () => {
		const terms: OptionalTermSeries = {
			short_erf: {
				ref: new Float32Array([1, 2, 3, 4]),
				webgpu: [1.5, 2, 3.5, 3.75]
			},
			long_erf: {
				ref: new Float32Array([10, 20, 30, 40]),
				webgpu: [10, 19.5, 31, 38]
			}
		};
		const rows = extractTopMrtDeltas({
			refMrt: new Float32Array([10, 20, 30, 40]),
			webgpuMrt: [11.5, 18, 31, 35],
			numPositions: 2,
			topN: 2,
			indices: [0, 1, 3],
			terms
		});

		expect(rows).toHaveLength(2);
		expect(rows[0]).toMatchObject({
			index: 3,
			hour: 1,
			pointIndex: 1,
			diff: -5,
			absDiff: 5
		});
		expect(rows[0].termDeltas.short_erf).toBeCloseTo(-0.25);
		expect(rows[0].termDeltas.long_erf).toBeCloseTo(-2);
		expect(rows[1]).toMatchObject({
			index: 1,
			hour: 1,
			pointIndex: 0,
			diff: -2,
			absDiff: 2
		});
	});

	it('adds dominant-term attribution and contribution summary fields', () => {
		const terms: OptionalTermSeries = {
			short_erf: {
				ref: [10, 20],
				webgpu: [11, 21]
			},
			long_erf: {
				ref: [5, 5],
				webgpu: [4, 14]
			},
			short_dmrt: {
				ref: [2, 2],
				webgpu: [2.5, 2.5]
			}
		};

		const rows = extractTopMrtDeltas({
			refMrt: [20, 30],
			webgpuMrt: [22, 40],
			numPositions: 1,
			topN: 1,
			terms
		});

		expect(rows[0]).toMatchObject({
			index: 1,
			dominantTerm: 'long_erf',
			dominantTermDelta: 9,
			termAbsSum: 10.5
		});
	});
});
