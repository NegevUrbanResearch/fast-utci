import { describe, expect, it } from 'vitest';
import {
	createEmptyOnDemandDiagnostics,
	mergeTrackedGpuAllocationBytes,
	recordColdStartLifecycleTiming,
	recordOnDemandTiming,
	resetColdStartLifecycleTimings
} from '$lib/compute/onDemandDiagnostics';

describe('on-demand diagnostics helpers', () => {
	it('createEmptyOnDemandDiagnostics starts with conservative defaults', () => {
		const diagnostics = createEmptyOnDemandDiagnostics();

		expect(diagnostics.navigatorGpu).toBe(false);
		expect(diagnostics.rendererBackend).toBe('unknown');
		expect(diagnostics.path).toBe('idle');
		expect(diagnostics.timeIndices).toEqual([]);
		expect(diagnostics.usedRunAllForSelectedHour).toBe(false);
		expect(diagnostics.usedExposureOnlyPrecompute).toBe(false);
		expect(diagnostics.allHoursUtciBytesAllocated).toBe(0);
		expect(diagnostics.allHoursMrtBytesAllocated).toBe(0);
		expect(diagnostics.oneHourOutputBytes).toBe(0);
		expect(diagnostics.selectedHourTransferCount).toBe(0);
		expect(diagnostics.renderTransport).toBe('none');
		expect(diagnostics.debugReadbackCount).toBe(0);
		expect(diagnostics.dataTextureBuildCount).toBe(0);
		expect(diagnostics.timings).toEqual({});
		expect(diagnostics.selectedMonthIndex).toBeNull();
		expect(diagnostics.selectedTimeIndex).toBeNull();
		expect(diagnostics.completedMonthIndex).toBeNull();
		expect(diagnostics.completedTimeIndex).toBeNull();
		expect(diagnostics.activeRequestId).toBeNull();
		expect(diagnostics.completedRequestId).toBeNull();
		expect(diagnostics.staleResultDiscardCount).toBe(0);
		expect(diagnostics.inFlightCount).toBe(0);
		expect(diagnostics.scrubSampleCount).toBe(0);
		expect(diagnostics.trackedGpuAllocationBytes.persistentExposureBytes).toBe(0);
		expect(diagnostics.trackedGpuAllocationBytes.allHoursOutputBytes).toBe(0);
		expect(diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytes).toBe(0);
		expect(diagnostics.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark).toBe(0);
		expect(diagnostics.trackedGpuAllocationBytes.trackingScope).toBe(
			'utci-owned-webgpu-buffers'
		);
	});

	it('recordOnDemandTiming preserves metadata while updating one timing entry', () => {
		const diagnostics = {
			...createEmptyOnDemandDiagnostics(),
			adapterInfo: 'MockAdapter',
			maxStorageBufferBindingSize: 134217728,
			maxBufferSize: 268435456,
			maxStorageBuffersPerShaderStage: 8,
			modelId: 'model-1',
			scenarioId: 'scenario-1',
			gridResolution: 2,
			pointCount: 1234,
			selectedMonthIndex: 0,
			selectedTimeIndex: 11,
			completedMonthIndex: 0,
			completedTimeIndex: 7,
			activeRequestId: 5,
			completedRequestId: 4,
			staleResultDiscardCount: 2,
			inFlightCount: 1,
			scrubSampleCount: 3,
			timeIndices: [3, 7, 11],
			usedRunAllForSelectedHour: true,
			usedExposureOnlyPrecompute: false,
			allHoursUtciBytesAllocated: 100,
			allHoursMrtBytesAllocated: 200,
			oneHourOutputBytes: 12,
			selectedHourTransferCount: 1,
			trackedGpuAllocationBytes: {
				persistentExposureBytes: 16,
				allHoursOutputBytes: 32,
				selectedHourOutputBytes: 48,
				selectedHourOutputBytesHighWatermark: 64,
				trackingScope: 'utci-owned-webgpu-buffers' as const
			},
			renderTransport: 'compute-buffer-selected-hour' as const,
			debugReadbackCount: 2,
			dataTextureBuildCount: 3,
			timings: {
				exposurePrecomputeMs: 12
			}
		};

		const next = recordOnDemandTiming(diagnostics, 'oneHourDispatchMs', 4);

		expect(next).toEqual({
			navigatorGpu: false,
			rendererBackend: 'unknown',
			path: 'idle',
			gpuResidentRenderAvailable: false,
			sameDeviceForComputeAndRender: null,
			gpuResidentCopyStatus: 'idle',
			adapterInfo: 'MockAdapter',
			maxStorageBufferBindingSize: 134217728,
			maxBufferSize: 268435456,
			maxStorageBuffersPerShaderStage: 8,
			modelId: 'model-1',
			scenarioId: 'scenario-1',
			gridResolution: 2,
			pointCount: 1234,
			selectedMonthIndex: 0,
			selectedTimeIndex: 11,
			completedMonthIndex: 0,
			completedTimeIndex: 7,
			activeRequestId: 5,
			completedRequestId: 4,
			staleResultDiscardCount: 2,
			inFlightCount: 1,
			scrubSampleCount: 3,
			timeIndices: [3, 7, 11],
			usedRunAllForSelectedHour: true,
			usedExposureOnlyPrecompute: false,
			allHoursUtciBytesAllocated: 100,
			allHoursMrtBytesAllocated: 200,
			oneHourOutputBytes: 12,
			selectedHourTransferCount: 1,
			trackedGpuAllocationBytes: {
				persistentExposureBytes: 16,
				allHoursOutputBytes: 32,
				selectedHourOutputBytes: 48,
				selectedHourOutputBytesHighWatermark: 64,
				trackingScope: 'utci-owned-webgpu-buffers'
			},
			renderTransport: 'compute-buffer-selected-hour',
			debugReadbackCount: 2,
			dataTextureBuildCount: 3,
			timings: {
				exposurePrecomputeMs: 12,
				oneHourDispatchMs: 4
			}
		});
	});

	it('records selected-hour timing attribution fields without clearing existing timings', () => {
		const diagnostics = createEmptyOnDemandDiagnostics();

		const withReadback = recordOnDemandTiming(diagnostics, 'selectedHourReadbackMs', 11.5);
		const withSurface = recordOnDemandTiming(withReadback, 'gpuSurfaceUpdateMs', 7.25);

		expect(withSurface.timings.selectedHourReadbackMs).toBe(11.5);
		expect(withSurface.timings.gpuSurfaceUpdateMs).toBe(7.25);
	});

	it('records route-level cold-start timings without overwriting earlier phases', () => {
		const diagnostics = createEmptyOnDemandDiagnostics();

		const withPrepare = recordOnDemandTiming(diagnostics, 'payloadPrepareMs', 101.25);
		const withWorker = recordOnDemandTiming(withPrepare, 'workerBvhMs', 202.5);
		const withUpload = recordOnDemandTiming(withWorker, 'pipelineUploadMs', 303.75);
		const withReady = recordOnDemandTiming(withUpload, 'firstSelectedHourReadyMs', 404);
		const withVisible = recordOnDemandTiming(withReady, 'firstSelectedHourVisibleMs', 505.5);

		expect(withVisible.timings).toEqual({
			payloadPrepareMs: 101.25,
			workerBvhMs: 202.5,
			pipelineUploadMs: 303.75,
			firstSelectedHourReadyMs: 404,
			firstSelectedHourVisibleMs: 505.5
		});
	});

	it('resets only cold-start timing fields for a new prepare lifecycle', () => {
		const diagnostics = {
			...createEmptyOnDemandDiagnostics(),
			timings: {
				payloadPrepareMs: 10,
				workerBvhMs: 20,
				pipelineUploadMs: 30,
				firstSelectedHourReadyMs: 40,
				firstSelectedHourVisibleMs: 50,
				exposurePrecomputeMs: 60,
				oneHourDispatchMs: 70
			}
		};

		const next = resetColdStartLifecycleTimings(diagnostics);

		expect(next.timings).toEqual({
			exposurePrecomputeMs: 60,
			oneHourDispatchMs: 70
		});
	});

	it('records ready and visible once per cold-start lifecycle, then allows a fresh lifecycle to replace them', () => {
		const diagnostics = createEmptyOnDemandDiagnostics();

		const firstReady = recordColdStartLifecycleTiming(
			diagnostics,
			'firstSelectedHourReadyMs',
			125
		);
		const ignoredReadyOverwrite = recordColdStartLifecycleTiming(
			firstReady,
			'firstSelectedHourReadyMs',
			250
		);
		const firstVisible = recordColdStartLifecycleTiming(
			ignoredReadyOverwrite,
			'firstSelectedHourVisibleMs',
			300
		);
		const ignoredVisibleOverwrite = recordColdStartLifecycleTiming(
			firstVisible,
			'firstSelectedHourVisibleMs',
			450
		);
		const reset = resetColdStartLifecycleTimings(ignoredVisibleOverwrite);
		const secondReady = recordColdStartLifecycleTiming(
			reset,
			'firstSelectedHourReadyMs',
			25
		);
		const secondVisible = recordColdStartLifecycleTiming(
			secondReady,
			'firstSelectedHourVisibleMs',
			35
		);

		expect(ignoredVisibleOverwrite.timings.firstSelectedHourReadyMs).toBe(125);
		expect(ignoredVisibleOverwrite.timings.firstSelectedHourVisibleMs).toBe(300);
		expect(secondVisible.timings.firstSelectedHourReadyMs).toBe(25);
		expect(secondVisible.timings.firstSelectedHourVisibleMs).toBe(35);
	});

	it('tracks selected-hour output high-watermark without pretending to know total browser VRAM', () => {
		const diagnostics = createEmptyOnDemandDiagnostics();

		const first = mergeTrackedGpuAllocationBytes(diagnostics, {
			persistentExposureBytes: 128,
			selectedHourOutputBytes: 64
		});
		const second = mergeTrackedGpuAllocationBytes(first, {
			selectedHourOutputBytes: 32
		});

		expect(second.trackedGpuAllocationBytes.trackingScope).toBe('utci-owned-webgpu-buffers');
		expect(second.trackedGpuAllocationBytes.persistentExposureBytes).toBe(128);
		expect(second.trackedGpuAllocationBytes.allHoursOutputBytes).toBe(0);
		expect(second.trackedGpuAllocationBytes.selectedHourOutputBytes).toBe(32);
		expect(second.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark).toBe(64);
	});

	it('preserves persistent exposure bytes while switching between exposure-only and baseline output accounting', () => {
		const diagnostics = createEmptyOnDemandDiagnostics();

		const exposureOnly = mergeTrackedGpuAllocationBytes(diagnostics, {
			persistentExposureBytes: 128,
			allHoursOutputBytes: 0,
			selectedHourOutputBytes: 64
		});
		const baseline = mergeTrackedGpuAllocationBytes(exposureOnly, {
			allHoursOutputBytes: 512,
			selectedHourOutputBytes: 0
		});

		expect(baseline.trackedGpuAllocationBytes.persistentExposureBytes).toBe(128);
		expect(baseline.trackedGpuAllocationBytes.allHoursOutputBytes).toBe(512);
		expect(baseline.trackedGpuAllocationBytes.selectedHourOutputBytes).toBe(0);
		expect(baseline.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark).toBe(64);
	});
});
