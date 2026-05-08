import { describe, expect, it, vi } from 'vitest';
import { ComputeManager } from '$lib/compute/compute-manager';
import type { OnDemandRuntimeDiagnostics } from '$lib/compute/onDemandDiagnostics';
import type {
	ExposurePrecomputeParams,
	OnDemandUtciOutput,
	RunUtciForTimeIndexParams,
	UTCIComputePipeline
} from '$lib/compute/gpu-pipeline';

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
			numMonths: 1
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

	it('getOnDemandDiagnostics surfaces pipeline runtime diagnostics when supported', () => {
		const diagnostics: OnDemandRuntimeDiagnostics = {
			navigatorGpu: true,
			rendererBackend: 'webgpu',
			path: 'exposure-only-f32',
			timeIndices: [],
			usedRunAllForSelectedHour: false,
			usedExposureOnlyPrecompute: true,
			allHoursUtciBytesAllocated: 0,
			allHoursMrtBytesAllocated: 0,
			oneHourOutputBytes: 400,
			selectedHourTransferCount: 0,
			renderTransport: 'none',
			debugReadbackCount: 0,
			dataTextureBuildCount: 0,
			timings: {}
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
