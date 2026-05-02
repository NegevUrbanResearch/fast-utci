import { describe, it, expect } from 'vitest';
import {
	calculateUTCI,
	calculateBoundaryAveragedUtciSeries
} from '$lib/compute/utci';

describe('UTCI boundary-averaged series helper', () => {
	it('should return empty array for empty inputs', () => {
		const result = calculateBoundaryAveragedUtciSeries({
			airTemps: [],
			mrts: [],
			windSpeeds: [],
			relativeHumidities: []
		});
		expect(result).toEqual([]);
	});

	it('should throw when input lengths differ', () => {
		expect(() =>
			calculateBoundaryAveragedUtciSeries({
				airTemps: [25, 26],
				mrts: [25],
				windSpeeds: [1, 1],
				relativeHumidities: [50, 50]
			})
		).toThrow();
	});

	it('should fall back to per-hour UTCI when there is only one timestep', () => {
		const airTemps = [25];
		const mrts = [30];
		const windSpeeds = [1.0];
		const relativeHumidities = [50];

		const series = calculateBoundaryAveragedUtciSeries({
			airTemps,
			mrts,
			windSpeeds,
			relativeHumidities
		});

		expect(series).toHaveLength(1);
		const direct = calculateUTCI(airTemps[0], mrts[0], windSpeeds[0], relativeHumidities[0]);
		expect(series[0]).toBeCloseTo(direct, 6);
	});

	it('should average adjacent UTCI values for boundary samples', () => {
		const airTemps = [20, 25, 30];
		const mrts = [25, 30, 35];
		const windSpeeds = [1.0, 1.0, 1.0];
		const relativeHumidities = [50, 50, 50];

		const utci0 = calculateUTCI(airTemps[0], mrts[0], windSpeeds[0], relativeHumidities[0]);
		const utci1 = calculateUTCI(airTemps[1], mrts[1], windSpeeds[1], relativeHumidities[1]);
		const utci2 = calculateUTCI(airTemps[2], mrts[2], windSpeeds[2], relativeHumidities[2]);

		const series = calculateBoundaryAveragedUtciSeries({
			airTemps,
			mrts,
			windSpeeds,
			relativeHumidities
		});

		expect(series).toHaveLength(3);

		// First entry averages UTCI at t0 and t1
		expect(series[0]).toBeCloseTo((utci0 + utci1) / 2, 6);
		// Second entry averages UTCI at t1 and t2
		expect(series[1]).toBeCloseTo((utci1 + utci2) / 2, 6);
		// Last entry uses the duplicated final boundary, which equals UTCI at t2.
		expect(series[2]).toBeCloseTo(utci2, 6);
	});
});

