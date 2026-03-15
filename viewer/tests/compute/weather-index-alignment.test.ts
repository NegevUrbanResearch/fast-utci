import { describe, it, expect, vi } from 'vitest';
import { ComputeManager } from '$lib/compute/compute-manager';
import type { UTCIComputePipeline } from '$lib/compute/gpu-pipeline';

function buildDeterministicEpwForJan1ToJan15(): string {
	const header = [
		'LOCATION,Beer Sheva,ISR,source,wmo,31.25,34.79,2,300',
		'DESIGN CONDITIONS,dummy',
		'TYPICAL/EXTREME,dummy',
		'GROUND TEMPERATURES,dummy',
		'HOLIDAYS/DAYLIGHT,dummy',
		'COMMENTS 1,dummy',
		'COMMENTS 2,dummy',
		'DATA PERIODS,dummy'
	];

	const lines: string[] = [];
	for (let day = 1; day <= 15; day++) {
		for (let hour = 1; hour <= 24; hour++) {
			const dryBulb = day * 100 + hour;
			const directNormal = day * 100 + hour;
			const diffuseHorizontal = day * 10 + hour;
			lines.push(
				`2020,1,${day},${hour},0,0,${dryBulb},20.0,50,99999,0,0,400,0,${directNormal},${diffuseHorizontal},0,0,0,0,0,3.0`
			);
		}
	}

	return `${header.join('\n')}\n${lines.join('\n')}`;
}

function createSerializedBvhFixture() {
	return {
		bvhNodeBuffer: new ArrayBuffer(0),
		bvhIndexBuffer: new ArrayBuffer(0),
		vertexBuffer: new Float32Array(0),
		indexBuffer: new Uint32Array(0)
	};
}

describe('weather index alignment', () => {
	it('packs weather to match Ladybug day-period semantics (hour0 uses previous day hour24)', async () => {
		const uploadStaticData = vi.fn().mockResolvedValue(undefined);
		const runAll = vi.fn().mockResolvedValue(undefined);
		const pipeline: UTCIComputePipeline = {
			uploadStaticData,
			runAll,
			readUtcisSlice: vi.fn()
		};
		const manager = new ComputeManager(pipeline, {
			numMonths: 1,
			numHoursPerDay: 24,
			startMonth: 1,
			representativeDay: 15
		});

		await manager.initFromModelAndWeather({
			serializedBvh: createSerializedBvhFixture(),
			epwContent: buildDeterministicEpwForJan1ToJan15(),
			gridResolution: 2,
			zHeight: 1.5,
			useRectangularGridFromBounds: true,
			analysisBounds: { x_min: -2, x_max: 2, y_min: -2, y_max: 2, z: 1.5 }
		});

		const weather = uploadStaticData.mock.calls[0][0].weather as Float32Array;
		const weatherStride = 7;

		const hour0AirTemp = weather[0 * weatherStride];
		const hour1AirTemp = weather[1 * weatherStride];
		const hour23AirTemp = weather[23 * weatherStride];
		const hour0DirectNormal = weather[0 * weatherStride + 4];
		const hour1DirectNormal = weather[1 * weatherStride + 4];
		expect(hour0AirTemp).toBe(1424); // previous day hour 24
		expect(hour1AirTemp).toBe(1501); // same day hour 1
		expect(hour23AirTemp).toBe(1523); // same day hour 23
		expect(hour0DirectNormal).toBe(1501); // shortwave remains same-day hour 1
		expect(hour1DirectNormal).toBe(1502); // shortwave remains same-day hour 2
	});
});
