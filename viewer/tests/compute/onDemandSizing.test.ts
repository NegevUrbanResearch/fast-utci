import { describe, expect, it } from 'vitest';
import {
	calculateAllHoursBufferSizes,
	calculateOneHourOutputSizes,
	calculateSolarBitmaskBytes
} from '$lib/compute/on-demand/onDemandSizing';

describe('onDemandSizing', () => {
	it('calculates bit-packed solar exposure bytes', () => {
		expect(calculateSolarBitmaskBytes({ numPoints: 100, totalTimeSteps: 288 })).toBe(3600);
		expect(calculateSolarBitmaskBytes({ numPoints: 511840, totalTimeSteps: 288 })).toBe(
			18426240
		);
		expect(calculateSolarBitmaskBytes({ numPoints: 8200000, totalTimeSteps: 288 })).toBe(
			295200000
		);
		expect(calculateSolarBitmaskBytes({ numPoints: 33, totalTimeSteps: 1 })).toBe(8);
	});

	it('calculates all-hours buffer sizes', () => {
		expect(
			calculateAllHoursBufferSizes({
				numPoints: 511840,
				numHours: 24,
				numMonths: 12
			})
		).toEqual({
			totalTimeSteps: 288,
			solarExposureBytes: 18426240,
			skyExposureBytes: 2047360,
			utciAllHoursBytes: 589639680,
			mrtAllHoursBytes: 589639680,
			cpuInt16UtciBytes: 294819840
		});
	});

	it('calculates one-hour output sizes', () => {
		expect(calculateOneHourOutputSizes({ numPoints: 8200000 })).toEqual({
			utciF32Bytes: 32800000,
			mrtF32Bytes: 32800000,
			combinedF32Bytes: 65600000,
			packedMrtUtciBytes: 32800000
		});
	});
});
