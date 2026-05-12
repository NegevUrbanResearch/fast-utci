import { describe, it, expect } from 'vitest';
import { createPipelineConfig } from '$lib/compute/gpu/gpu-pipeline';
import exposureSolarWgsl from '$lib/compute/gpu/shaders/exposure_solar.wgsl?raw';

describe('Exposure pipeline', () => {
	it('should require bit-packed solar buffer size', () => {
		const config = createPipelineConfig({
			numPoints: 200,
			numHours: 24,
			numMonths: 12
		});
		expect(config.solarExposureBufferSize).toBe(Math.ceil((200 * 12 * 24) / 32) * 4);
	});

	it('should require sky buffer size numPoints * 4', () => {
		const config = createPipelineConfig({
			numPoints: 200,
			numHours: 24,
			numMonths: 12
		});
		expect(config.skyExposureBufferSize).toBe(200 * 4);
	});

	it('solar shader should short-circuit nighttime rays before BVH traversal', () => {
		expect(exposureSolarWgsl.includes('sun.y <= 0.0')).toBe(true);
		expect(exposureSolarWgsl.includes('sun_len2 < 1e-10')).toBe(true);
	});

	// Optional browser test: when WebGPU is available, run full pipeline with a
	// tiny mesh (e.g. box) and 2-point grid (one in sun, one in shade), read
	// back solar_exposure and assert at least one value is 0 and one is 1.
	// Skipped in Node/Vitest; enable in browser test runner with WebGPU.
	it.skip('should produce differing solar exposure for points in sun vs shade (WebGPU only)', async () => {
		// Would require: createWebgpuUtciPipeline(), uploadStaticData with mesh
		// (box) and 2 grid points, runAll(), readback of solar_exposure buffer,
		// then expect(solarExposure[0] !== solarExposure[1]).
	});
});
