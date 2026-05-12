import { describe, it, expect, vi } from 'vitest';
import { ComputeManager } from '$lib/compute/compute-manager';
import type { UTCIComputePipeline } from '$lib/compute/gpu/gpu-pipeline';
import * as sunpath from '$lib/compute/core/sunpath';

function buildMinimalEpw(): string {
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
	for (let hour = 1; hour <= 24; hour++) {
		lines.push(`2020,1,15,${hour},0,0,25.0,20.0,50,99999,0,0,400,0,800,200,0,0,0,0,0,3.0`);
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

describe('sun vector alignment', () => {
	it('uses fixture vectors unchanged in parity mode', async () => {
		const uploadStaticData = vi.fn().mockResolvedValue(undefined);
		const runAll = vi.fn().mockResolvedValue(undefined);
		const pipeline: UTCIComputePipeline = {
			uploadStaticData,
			runAll,
			readUtcisSlice: vi.fn()
		};
		const manager = new ComputeManager(pipeline, { numMonths: 1, numHoursPerDay: 24 });
		const fixtureVectors = new Float32Array(24 * 3);
		const fixtureAltitudes = new Float32Array(24);
		for (let i = 0; i < 24; i++) {
			fixtureVectors[i * 3] = 0.25;
			fixtureVectors[i * 3 + 1] = 0.75;
			fixtureVectors[i * 3 + 2] = -0.6;
			fixtureAltitudes[i] = 0.4;
		}

		const spy = vi.spyOn(sunpath, 'getSunVectors');
		await manager.initFromModelAndWeather({
			serializedBvh: createSerializedBvhFixture(),
			epwContent: buildMinimalEpw(),
			gridResolution: 2,
			zHeight: 1.5,
			useRectangularGridFromBounds: true,
			analysisBounds: { x_min: -2, x_max: 2, y_min: -2, y_max: 2, z: 1.5 },
			sunVectorsFixture: {
				sunVectors: fixtureVectors,
				sunAltitudes: fixtureAltitudes
			}
		});

		const uploadArgs = uploadStaticData.mock.calls[0][0];
		expect(uploadArgs.sunVectors).toBe(fixtureVectors);
		expect(uploadArgs.sunAltitudes).toBe(fixtureAltitudes);
		expect(spy).not.toHaveBeenCalled();
		spy.mockRestore();
	});
});
