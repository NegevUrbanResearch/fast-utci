import { describe, it, expect } from 'vitest';
import {
	createPipelineConfig,
	calculateDispatch,
	createPointDispatchChunks,
	getUtciFlatIndex,
	MAX_WEBGPU_WORKGROUPS_PER_DIMENSION
} from '$lib/compute/gpu/gpu-pipeline';

describe('GPU Compute Pipeline', () => {
	it('should create pipeline config with correct buffer sizes', () => {
		const config = createPipelineConfig({
			numPoints: 100,
			numHours: 24,
			numMonths: 12
		});

		expect(config.solarExposureBufferSize).toBe(Math.ceil((100 * 24 * 12) / 32) * 4);
		expect(config.utciResultBufferSize).toBe(100 * 24 * 12 * 4);

		// 1 sky view factor per point
		expect(config.skyExposureBufferSize).toBe(100 * 4);
	});

	it('should validate input when creating pipeline config', () => {
		expect(() =>
			createPipelineConfig({ numPoints: 0, numHours: 24, numMonths: 12 })
		).toThrowError();
		expect(() =>
			createPipelineConfig({ numPoints: 10, numHours: 0, numMonths: 12 })
		).toThrowError();
		expect(() =>
			createPipelineConfig({ numPoints: 10, numHours: 24, numMonths: 0 })
		).toThrowError();
	});

	it('should generate correct 2D dispatch dimensions', () => {
		const dispatch = calculateDispatch(10_000, 24, 1, 64);
		expect(dispatch.x).toBe(Math.ceil(10_000 / 64));
		expect(dispatch.y).toBe(24);
	});

	it('should include all months in dispatch Y dimension', () => {
		const dispatch = calculateDispatch(10_000, 24, 12, 64);
		expect(dispatch.y).toBe(24 * 12);
	});

	it('should compute flat UTCI index for point-major layout', () => {
		const totalTimeSteps = 24;
		expect(getUtciFlatIndex(0, 0, totalTimeSteps)).toBe(0);
		expect(getUtciFlatIndex(1, 0, totalTimeSteps)).toBe(24);
		expect(getUtciFlatIndex(2, 5, totalTimeSteps)).toBe(53);
	});

	it('should validate flat index inputs', () => {
		expect(() => getUtciFlatIndex(-1, 0, 24)).toThrowError();
		expect(() => getUtciFlatIndex(0, -1, 24)).toThrowError();
		expect(() => getUtciFlatIndex(0, 0, 0)).toThrowError();
	});

	it('should validate inputs for dispatch calculation', () => {
		expect(() => calculateDispatch(0, 24, 1, 64)).toThrowError();
		expect(() => calculateDispatch(10_000, 0, 1, 64)).toThrowError();
		expect(() => calculateDispatch(10_000, 24, 0, 64)).toThrowError();
		expect(() => calculateDispatch(10_000, 24, 1, 0)).toThrowError();
	});

	it('should match solar dispatch dimensions and the equivalent sky X workgroup count', () => {
		const numPoints = 500;
		const numHours = 24;
		const numMonths = 12;
		const workgroupSize = 64;

		// Solar dispatch uses the production helper directly.
		const solarDispatch = calculateDispatch(numPoints, numHours, numMonths, workgroupSize);
		expect(solarDispatch.x).toBe(Math.ceil(numPoints / workgroupSize));
		expect(solarDispatch.y).toBe(numMonths * numHours);

		// Sky exposure uses the same X workgroup count over points, but that path is not asserted here.
		const skyWorkgroupsX = Math.ceil(numPoints / workgroupSize);
		expect(skyWorkgroupsX).toBe(8);
	});

	it('chunks dense point dispatches under the WebGPU per-dimension workgroup limit', () => {
		const workgroupSize = 64;
		const maxPointsPerChunk = workgroupSize * MAX_WEBGPU_WORKGROUPS_PER_DIMENSION;
		const chunks = createPointDispatchChunks(maxPointsPerChunk + 100, workgroupSize);

		expect(chunks).toEqual([
			{
				pointOffset: 0,
				pointCount: maxPointsPerChunk,
				workgroupsX: MAX_WEBGPU_WORKGROUPS_PER_DIMENSION
			},
			{
				pointOffset: maxPointsPerChunk,
				pointCount: 100,
				workgroupsX: 2
			}
		]);
		expect(Math.max(...chunks.map((chunk) => chunk.workgroupsX))).toBeLessThanOrEqual(
			MAX_WEBGPU_WORKGROUPS_PER_DIMENSION
		);
	});
});
