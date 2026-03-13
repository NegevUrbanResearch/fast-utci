import { describe, it, expect } from 'vitest';
import { createPipelineConfig, calculateDispatch } from '$lib/compute/gpu-pipeline';

describe('GPU Compute Pipeline', () => {
	it('should create pipeline config with correct buffer sizes', () => {
		const config = createPipelineConfig({
			numPoints: 100,
			numHours: 24,
			numMonths: 12
		});

		// 100 points × 24 hours × 12 months × 4 bytes (f32)
		expect(config.solarExposureBufferSize).toBe(100 * 24 * 12 * 4);
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

	it('should validate inputs for dispatch calculation', () => {
		expect(() => calculateDispatch(0, 24, 1, 64)).toThrowError();
		expect(() => calculateDispatch(10_000, 0, 1, 64)).toThrowError();
		expect(() => calculateDispatch(10_000, 24, 0, 64)).toThrowError();
		expect(() => calculateDispatch(10_000, 24, 1, 0)).toThrowError();
	});
});

