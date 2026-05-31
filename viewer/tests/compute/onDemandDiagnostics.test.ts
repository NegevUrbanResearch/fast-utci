import { describe, expect, it, vi } from 'vitest';
import {
	buildGpuResidentSurfaceResetPatch,
	createEmptyOnDemandDiagnostics,
	invokeDiagnosticsCallbackSafely,
	mergeSelectedHourRenderTimings,
	mergeTrackedGpuAllocationBytes,
	prepareSelectedHourCycleTimings,
	recordColdStartLifecycleTiming,
	recordOnDemandTiming,
	recordSelectedHourReadbackReason,
	resetColdStartLifecycleTimings
} from '$lib/compute/on-demand/onDemandDiagnostics';

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
		const withSummary = recordOnDemandTiming(
			withReadback,
			'selectedHourRangeSummaryReadbackBytes',
			16
		);
		const withSurface = recordOnDemandTiming(withReadback, 'gpuSurfaceUpdateMs', 7.25);
		const withSceneDelay = recordOnDemandTiming(
			withSurface,
			'renderSceneSyncStartDelayMs',
			0.75
		);
		const withSceneTotal = recordOnDemandTiming(withSceneDelay, 'renderSceneSyncTotalMs', 6.5);
		const withLayout = recordOnDemandTiming(withSceneTotal, 'renderLayoutBuildMs', 1.5);
		const withDrain = recordOnDemandTiming(withLayout, 'renderQueueDrainMs', 2.25);

		expect(withDrain.timings.selectedHourReadbackMs).toBe(11.5);
		expect(withSummary.timings.selectedHourRangeSummaryReadbackBytes).toBe(16);
		expect(withDrain.timings.gpuSurfaceUpdateMs).toBe(7.25);
		expect(withDrain.timings.renderSceneSyncStartDelayMs).toBe(0.75);
		expect(withDrain.timings.renderSceneSyncTotalMs).toBe(6.5);
		expect(withDrain.timings.renderLayoutBuildMs).toBe(1.5);
		expect(withDrain.timings.renderQueueDrainMs).toBe(2.25);
	});

	it('records compact selected-hour range summary proof fields independently from CPU readbacks', () => {
		const diagnostics = createEmptyOnDemandDiagnostics();
		const next = {
			...diagnostics,
			timings: {
				...diagnostics.timings,
				selectedHourRangeSummaryMs: 2.5,
				selectedHourRangeSummaryDispatchMs: 1.5,
				selectedHourRangeSummaryReadbackMs: 0.75,
				selectedHourRangeSummaryReadbackBytes: 16,
				selectedHourRangeSummaryReadbackCount: 1,
				selectedHourRangeSummaryReductionPassCount: 1,
				selectedHourRangeFullReadbackAvoidedCount: 1
			}
		};

		expect(next.timings).toMatchObject({
			selectedHourRangeSummaryMs: 2.5,
			selectedHourRangeSummaryDispatchMs: 1.5,
			selectedHourRangeSummaryReadbackMs: 0.75,
			selectedHourRangeSummaryReadbackBytes: 16,
			selectedHourRangeSummaryReadbackCount: 1,
			selectedHourRangeSummaryReductionPassCount: 1,
			selectedHourRangeFullReadbackAvoidedCount: 1
		});
		expect(next.selectedHourReadbackReasons).toBeUndefined();
		expect(next.visibleSelectedHourReadbackCount).toBeUndefined();
	});

	it('merges GPU-resident render timing substeps into the selected-hour timing bucket', () => {
		const merged = mergeSelectedHourRenderTimings({
			existingTimings: {
				exposurePrecomputeMs: 12
			},
			renderUpdateMs: 22.5,
			gpuSurfaceUpdateMs: 22.5,
			firstSelectedHourVisibleMs: 44,
			renderSubsteps: {
				renderSceneSyncStartDelayMs: 1,
				renderSceneSyncTotalMs: 18.5,
				renderLayoutBuildMs: 1.25,
				renderSurfaceMeshMs: 2.5,
				renderStorageInitWaitMs: 3.75,
				renderBufferCopyMs: 4.5,
				renderQueueDrainMs: 5.25
			}
		});

		expect(merged).toEqual({
			exposurePrecomputeMs: 12,
			renderUpdateMs: 22.5,
			gpuSurfaceUpdateMs: 22.5,
			firstSelectedHourVisibleMs: 44,
			renderSceneSyncStartDelayMs: 1,
			renderSceneSyncTotalMs: 18.5,
			renderLayoutBuildMs: 1.25,
			renderSurfaceMeshMs: 2.5,
			renderStorageInitWaitMs: 3.75,
			renderBufferCopyMs: 4.5,
			renderQueueDrainMs: 5.25
		});
	});

	it('preserves render publication diagnostics when merging selected-hour render timings', () => {
		const merged = mergeSelectedHourRenderTimings({
			existingTimings: {
				exposurePrecomputeMs: 12
			},
			renderUpdateMs: 42,
			gpuSurfaceUpdateMs: 42,
			firstSelectedHourVisibleMs: 50,
			renderSubsteps: {
				renderSceneSyncStartDelayMs: 3,
				renderSceneSyncTotalMs: 39,
				renderLayoutBuildMs: 4,
				renderSurfaceMeshMs: 5,
				renderStorageInitWaitMs: 6,
				renderBufferCopyMs: 7,
				renderQueueDrainMs: 8,
				renderPublication: {
					renderPublicationVersion: 1,
					renderPublicationPath: 'compute-buffer-selected-hour',
					renderPublicationPhase: 'scrub',
					renderPublicationMeshAction: 'reused',
					renderPublicationPointCount: 8171761,
					renderPublicationVertexCount: 49030566,
					renderPublicationGridWidth: 2860,
					renderPublicationGridHeight: 2857,
					renderPublicationGridSize: 0.5,
					renderPublicationSourceByteLength: 32687044,
					renderPublicationTargetByteLength: 32687044,
					renderPublicationRenderOwnedBytes: 817177124
				}
			}
		});

		expect(merged.renderPublication).toMatchObject({
			renderPublicationVersion: 1,
			renderPublicationPath: 'compute-buffer-selected-hour',
			renderPublicationPhase: 'scrub',
			renderPublicationMeshAction: 'reused',
			renderPublicationPointCount: 8171761,
			renderPublicationTargetByteLength: 32687044
		});
	});

	it('keeps render timing substeps optional when the scene did not measure them', () => {
		const merged = mergeSelectedHourRenderTimings({
			existingTimings: {
				exposurePrecomputeMs: 12,
				renderPublication: {
					renderPublicationVersion: 1,
					renderPublicationPath: 'compute-buffer-selected-hour',
					renderPublicationPhase: 'scrub',
					renderPublicationMeshAction: 'reused',
					renderPublicationPointCount: 8171761
				}
			},
			renderUpdateMs: 22.5,
			gpuSurfaceUpdateMs: 22.5
		});

		expect(merged).toEqual({
			exposurePrecomputeMs: 12,
			renderUpdateMs: 22.5,
			gpuSurfaceUpdateMs: 22.5
		});
		expect(merged.renderSceneSyncStartDelayMs).toBeUndefined();
		expect(merged.renderSceneSyncTotalMs).toBeUndefined();
		expect(merged.renderLayoutBuildMs).toBeUndefined();
		expect(merged.renderSurfaceMeshMs).toBeUndefined();
		expect(merged.renderStorageInitWaitMs).toBeUndefined();
		expect(merged.renderBufferCopyMs).toBeUndefined();
		expect(merged.renderQueueDrainMs).toBeUndefined();
		expect(merged.renderPublication).toBeUndefined();
	});

	it('clears stale render timing substeps from earlier GPU-resident renders', () => {
		const merged = mergeSelectedHourRenderTimings({
			existingTimings: {
				exposurePrecomputeMs: 12,
				renderSceneSyncStartDelayMs: 1,
				renderSceneSyncTotalMs: 18.5,
				renderLayoutBuildMs: 1.25,
				renderSurfaceMeshMs: 2.5,
				renderStorageInitWaitMs: 3.75,
				renderBufferCopyMs: 4.5,
				renderQueueDrainMs: 5.25
			},
			renderUpdateMs: 22.5,
			gpuSurfaceUpdateMs: 22.5,
			renderSubsteps: {
				renderSceneSyncStartDelayMs: undefined,
				renderSceneSyncTotalMs: 14.25,
				renderLayoutBuildMs: undefined,
				renderSurfaceMeshMs: 6.5,
				renderStorageInitWaitMs: undefined,
				renderBufferCopyMs: undefined,
				renderQueueDrainMs: 7.25
			}
		});

		expect(merged).toEqual({
			exposurePrecomputeMs: 12,
			renderUpdateMs: 22.5,
			gpuSurfaceUpdateMs: 22.5,
			renderSceneSyncTotalMs: 14.25,
			renderSurfaceMeshMs: 6.5,
			renderQueueDrainMs: 7.25
		});
		expect(merged.renderSceneSyncStartDelayMs).toBeUndefined();
		expect(merged.renderLayoutBuildMs).toBeUndefined();
		expect(merged.renderStorageInitWaitMs).toBeUndefined();
		expect(merged.renderBufferCopyMs).toBeUndefined();
	});

	it('clears stale GPU render timings when the route starts a new selected-hour cycle', () => {
		const nextTimings = prepareSelectedHourCycleTimings({
			existingTimings: {
				exposurePrecomputeMs: 12,
				renderUpdateMs: 22.5,
				gpuSurfaceUpdateMs: 22.5,
				renderSceneSyncStartDelayMs: 1,
				renderSceneSyncTotalMs: 18.5,
				renderLayoutBuildMs: 1.25,
				renderSurfaceMeshMs: 2.5,
				renderStorageInitWaitMs: 3.75,
				renderBufferCopyMs: 4.5,
				renderQueueDrainMs: 5.25
			},
			pipelineTimings: {
				oneHourDispatchMs: 9.5
			},
			firstSelectedHourReadyMs: 44,
			selectedHourReadbackMs: 5.5
		});

		expect(nextTimings).toEqual({
			exposurePrecomputeMs: 12,
			oneHourDispatchMs: 9.5,
			firstSelectedHourReadyMs: 44,
			selectedHourReadbackMs: 5.5
		});
		expect(nextTimings.renderUpdateMs).toBeUndefined();
		expect(nextTimings.gpuSurfaceUpdateMs).toBeUndefined();
		expect(nextTimings.renderSceneSyncStartDelayMs).toBeUndefined();
		expect(nextTimings.renderSceneSyncTotalMs).toBeUndefined();
		expect(nextTimings.renderLayoutBuildMs).toBeUndefined();
		expect(nextTimings.renderSurfaceMeshMs).toBeUndefined();
		expect(nextTimings.renderStorageInitWaitMs).toBeUndefined();
		expect(nextTimings.renderBufferCopyMs).toBeUndefined();
		expect(nextTimings.renderQueueDrainMs).toBeUndefined();
	});

	it('clears stale GPU copy completion and render timings when the route resets surface diagnostics', () => {
		const resetPatch = buildGpuResidentSurfaceResetPatch({
			existingTimings: {
				exposurePrecomputeMs: 12,
				renderUpdateMs: 22.5,
				gpuSurfaceUpdateMs: 22.5,
				renderSceneSyncStartDelayMs: 1,
				renderSceneSyncTotalMs: 18.5,
				renderLayoutBuildMs: 1.25,
				renderSurfaceMeshMs: 2.5,
				renderStorageInitWaitMs: 3.75,
				renderBufferCopyMs: 4.5,
				renderQueueDrainMs: 5.25
			}
		});

		expect(resetPatch).toEqual({
			utciSurfaceSource: undefined,
			selectedHourTransferCount: 0,
			dataTextureBuildCount: 0,
			gpuResidentCopyStatus: 'idle',
			gpuResidentCopyError: undefined,
			gpuResidentCopyRequestId: undefined,
			timings: {
				exposurePrecomputeMs: 12
			}
		});
	});

	it('catches rejected async diagnostics callbacks so they do not leak unhandled promises', async () => {
		const error = new Error('surface diagnostics failed');
		const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

		invokeDiagnosticsCallbackSafely(
			async () => {
				throw error;
			},
			{ phase: 'surface-reset' },
			'UTCIPointCloud onUtciSurfaceDiagnostics'
		);

		await Promise.resolve();
		await Promise.resolve();

		expect(consoleError).toHaveBeenCalledWith(
			'[UTCIPointCloud onUtciSurfaceDiagnostics] diagnostics callback failed.',
			error
		);
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

	it('records selected-hour CPU readback reasons separately from visible readback count', () => {
		const diagnostics = createEmptyOnDemandDiagnostics();
		const next = recordSelectedHourReadbackReason(
			recordSelectedHourReadbackReason(diagnostics, 'range'),
			'tooltip'
		);

		expect(next.selectedHourReadbackReasons).toEqual(['range', 'tooltip']);
		expect(next.selectedHourReadbackReasonCounts).toEqual({
			range: 1,
			tooltip: 1
		});
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
