import { describe, expect, it, vi } from 'vitest';
import { ComputeManager } from '$lib/compute/compute-manager';
import {
	createEmptyOnDemandDiagnostics,
	type OnDemandRuntimeDiagnostics
} from '$lib/compute/on-demand/onDemandDiagnostics';
import type {
	ExposurePrecomputeParams,
	F32MetricOutput,
	OnDemandUtciOutput,
	RunF32OutputRangeSummaryParams,
	RunShadingIndexParams,
	RunUtciForTimeIndexParams,
	UtciRangeSummary,
	UTCIComputePipeline
} from '$lib/compute/gpu/gpu-pipeline';

function basePipeline(): UTCIComputePipeline {
	return {
		uploadStaticData: vi.fn().mockResolvedValue(undefined),
		runAll: vi.fn().mockResolvedValue(undefined),
		readUtcisSlice: vi.fn().mockResolvedValue(new Float32Array(0))
	};
}

describe('ComputeManager on-demand wrappers', () => {
	it('runExposurePrecompute delegates when supported', async () => {
		const params: ExposurePrecomputeParams = {
			numPoints: 8,
			numHours: 24,
			numMonths: 1,
			signal: new AbortController().signal,
			exposureScheduling: {
				mode: 'chunked' as const,
				maxWorkgroupsPerSlice: 8192,
				yieldBetweenSlices: true
			}
		};
		const runExposurePrecompute = vi.fn().mockResolvedValue(undefined);
		const pipeline: UTCIComputePipeline = {
			...basePipeline(),
			runExposurePrecompute
		};
		const manager = new ComputeManager(pipeline, { numMonths: 1, numHoursPerDay: 24 });

		await manager.runExposurePrecompute(params);

		expect(runExposurePrecompute).toHaveBeenCalledTimes(1);
		expect(runExposurePrecompute).toHaveBeenCalledWith(params);
	});

	it('runExposurePrecompute throws clearly when unsupported', async () => {
		const manager = new ComputeManager(basePipeline(), { numMonths: 1, numHoursPerDay: 24 });

		await expect(
			manager.runExposurePrecompute({
				numPoints: 8,
				numHours: 24,
				numMonths: 1
			})
		).rejects.toThrow(/does not support exposure-only precompute/i);
	});

	it('runUtciForTimeIndex delegates when supported and returns the output', async () => {
		const params: RunUtciForTimeIndexParams = {
			timeIndex: 7,
			numPoints: 3,
			numHours: 24,
			numMonths: 1,
			format: 'f32-utci'
		};
		const output: OnDemandUtciOutput = {
			format: 'f32-utci',
			numPoints: 3,
			timeIndex: 7,
			debugLabel: 'unit-test-output'
		};
		const runUtciForTimeIndex = vi.fn().mockResolvedValue(output);
		const pipeline: UTCIComputePipeline = {
			...basePipeline(),
			runUtciForTimeIndex
		};
		const manager = new ComputeManager(pipeline, { numMonths: 1, numHoursPerDay: 24 });

		const result = await manager.runUtciForTimeIndex(params);

		expect(runUtciForTimeIndex).toHaveBeenCalledTimes(1);
		expect(runUtciForTimeIndex).toHaveBeenCalledWith(params);
		expect(result).toBe(output);
	});

	it('runUtciForTimeIndex throws clearly when unsupported', async () => {
		const manager = new ComputeManager(basePipeline(), { numMonths: 1, numHoursPerDay: 24 });

		await expect(
			manager.runUtciForTimeIndex({
				timeIndex: 7,
				numPoints: 3,
				numHours: 24,
				numMonths: 1,
				format: 'f32-utci'
			})
		).rejects.toThrow(/does not support one-hour UTCI compute/i);
	});

	it('runShadingIndex delegates when supported and returns the output', async () => {
		const params: RunShadingIndexParams = {
			numPoints: 3,
			numHours: 24,
			numMonths: 1,
			monthIndex: 0,
			startTimeIndex: 6,
			timeCount: 4
		};
		const output: F32MetricOutput = {
			source: 'webgpu-on-demand-snapshot',
			ownerId: 'webgpu-shading-index:test',
			metricType: 'shading_index',
			valueLayout: 'one-f32-per-point',
			period: { kind: 'month-index', index: 0, startTimeIndex: 6, timeCount: 4 },
			numPoints: 3,
			outputBytes: 12,
			debugLabel: 'webgpu-shading-index'
		};
		const runShadingIndex = vi.fn().mockResolvedValue(output);
		const pipeline: UTCIComputePipeline = {
			...basePipeline(),
			runShadingIndex
		};
		const manager = new ComputeManager(pipeline, { numMonths: 1, numHoursPerDay: 24 });

		const result = await manager.runShadingIndex(params);

		expect(runShadingIndex).toHaveBeenCalledTimes(1);
		expect(runShadingIndex).toHaveBeenCalledWith(params);
		expect(result).toBe(output);
	});

	it('runShadingIndex throws clearly when unsupported', async () => {
		const manager = new ComputeManager(basePipeline(), { numMonths: 1, numHoursPerDay: 24 });

		await expect(
			manager.runShadingIndex({
				numPoints: 3,
				numHours: 24,
				numMonths: 1,
				monthIndex: 0,
				startTimeIndex: 6,
				timeCount: 4
			})
		).rejects.toThrow(/does not support shading index compute/i);
	});

	it('runF32OutputRangeSummary delegates when supported and returns the summary', async () => {
		const params: RunF32OutputRangeSummaryParams = {
			metricType: 'shading_index',
			numPoints: 3,
			output: {
				source: 'webgpu-on-demand-snapshot',
				ownerId: 'webgpu-shading-index:test',
				metricType: 'shading_index',
				valueLayout: 'one-f32-per-point',
				period: { kind: 'month-index', index: 0, startTimeIndex: 6, timeCount: 4 },
				numPoints: 3,
				gpuOutputHandle: { disposed: false } as never,
				outputBytes: 12,
				debugLabel: 'webgpu-shading-index'
			}
		};
		const summary: UtciRangeSummary = {
			timeIndex: 6,
			range: { min: 0.2, max: 0.8 },
			validCount: 3,
			readbackBytes: 16,
			reductionPassCount: 1,
			debugLabel: 'webgpu-on-demand-f32-utci-range-summary'
		};
		const runF32OutputRangeSummary = vi.fn().mockResolvedValue(summary);
		const pipeline: UTCIComputePipeline = {
			...basePipeline(),
			runF32OutputRangeSummary
		};
		const manager = new ComputeManager(pipeline, { numMonths: 1, numHoursPerDay: 24 });

		const result = await manager.runF32OutputRangeSummary(params);

		expect(runF32OutputRangeSummary).toHaveBeenCalledTimes(1);
		expect(runF32OutputRangeSummary).toHaveBeenCalledWith(params);
		expect(result).toBe(summary);
	});

	it('runF32OutputRangeSummary rejects non one-f32-per-point outputs before delegating', async () => {
		const runF32OutputRangeSummary = vi.fn();
		const pipeline: UTCIComputePipeline = {
			...basePipeline(),
			runF32OutputRangeSummary
		};
		const manager = new ComputeManager(pipeline, { numMonths: 1, numHoursPerDay: 24 });

		await expect(
			manager.runF32OutputRangeSummary({
				metricType: 'shading_index',
				numPoints: 3,
				output: {
					source: 'webgpu-on-demand-snapshot',
					ownerId: 'webgpu-shading-index:test',
					metricType: 'shading_index',
					valueLayout: 'point-major-time-series' as never,
					period: { kind: 'month-index', index: 0, startTimeIndex: 6, timeCount: 4 },
					numPoints: 3,
					gpuOutputHandle: { disposed: false } as never,
					outputBytes: 12,
					debugLabel: 'webgpu-shading-index'
				}
			})
		).rejects.toThrow(/support only one-f32-per-point owned outputs/i);

		expect(runF32OutputRangeSummary).not.toHaveBeenCalled();
	});

	it('runF32OutputRangeSummary rejects outputs without owned GPU handles before delegating', async () => {
		const runF32OutputRangeSummary = vi.fn();
		const pipeline: UTCIComputePipeline = {
			...basePipeline(),
			runF32OutputRangeSummary
		};
		const manager = new ComputeManager(pipeline, { numMonths: 1, numHoursPerDay: 24 });

		await expect(
			manager.runF32OutputRangeSummary({
				metricType: 'shading_index',
				numPoints: 3,
				output: {
					source: 'webgpu-on-demand-snapshot',
					ownerId: 'webgpu-shading-index:test',
					metricType: 'shading_index',
					valueLayout: 'one-f32-per-point',
					period: { kind: 'month-index', index: 0, startTimeIndex: 6, timeCount: 4 },
					numPoints: 3,
					outputBytes: 12,
					debugLabel: 'webgpu-shading-index'
				}
			})
		).rejects.toThrow(/support only one-f32-per-point owned outputs/i);

		expect(runF32OutputRangeSummary).not.toHaveBeenCalled();
	});

	it('runF32OutputRangeSummary throws clearly when unsupported', async () => {
		const manager = new ComputeManager(basePipeline(), { numMonths: 1, numHoursPerDay: 24 });

		await expect(
			manager.runF32OutputRangeSummary({
				metricType: 'shading_index',
				numPoints: 3,
				output: {
					source: 'webgpu-on-demand-snapshot',
					ownerId: 'webgpu-shading-index:test',
					metricType: 'shading_index',
					valueLayout: 'one-f32-per-point',
					period: { kind: 'month-index', index: 0, startTimeIndex: 6, timeCount: 4 },
					numPoints: 3,
					gpuOutputHandle: { disposed: false } as never,
					outputBytes: 12,
					debugLabel: 'webgpu-shading-index'
				}
			})
		).rejects.toThrow(/does not support generic f32 output range summaries/i);
	});

	it('getOnDemandDiagnostics surfaces pipeline runtime diagnostics when supported', () => {
		const diagnostics: OnDemandRuntimeDiagnostics = {
			...createEmptyOnDemandDiagnostics(),
			navigatorGpu: true,
			rendererBackend: 'webgpu',
			path: 'exposure-only-f32',
			usedExposureOnlyPrecompute: true,
			oneHourOutputBytes: 400
		};
		const pipeline: UTCIComputePipeline = {
			...basePipeline(),
			getOnDemandDiagnostics: vi.fn().mockReturnValue(diagnostics)
		};
		const manager = new ComputeManager(pipeline, { numMonths: 1, numHoursPerDay: 24 });

		expect(manager.getOnDemandDiagnostics()).toMatchObject({
			path: 'exposure-only-f32',
			oneHourOutputBytes: 400
		});
	});
});
