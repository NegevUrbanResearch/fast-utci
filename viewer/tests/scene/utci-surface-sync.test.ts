import { describe, expect, it } from 'vitest';
import { Mesh } from 'three';
import {
	buildUtciSurfaceDiagnostics,
	buildCpuPublicationDiagnostics,
	getAcceptedGpuResidentKey,
	isComputeBufferUtciSurface,
	shouldRecreateComputeBufferUtciSurface
} from '../../src/lib/components/scene/utciSurfaceSync';
import { createAcceptedGpuResidentSurfaceSync as createSurfaceSync } from '../../src/lib/components/scene/acceptedGpuResidentSurfaceSync';

describe('utciSurfaceSync', () => {
	it('builds a stable accepted GPU resident key from request and range', () => {
		const key = getAcceptedGpuResidentKey({
			requestId: 5,
			monthIndex: 7,
			hourIndex: 12,
			timeIndex: 180,
			utciRange: { min: 18.5, max: 41.25 },
			output: {
				format: 'f32-utci',
				numPoints: 2,
				timeIndex: 180,
				gpuBuffer: {} as GPUBuffer
			}
		});

		expect(key).toBe('5:7:180:18.5:41.25');
	});

	it('includes metric type and metric range in accepted GPU resident keys', () => {
		const key = getAcceptedGpuResidentKey({
			requestId: 5,
			metricType: 'shading_index',
			monthIndex: 7,
			hourIndex: 0,
			timeIndex: 168,
			utciRange: { min: 18.5, max: 41.25 },
			shadingIndexRange: { min: 0, max: 1 },
			output: {
				source: 'webgpu-on-demand-snapshot',
				ownerId: 'shading:7',
				metricType: 'shading_index',
				valueLayout: 'one-f32-per-point',
				period: { kind: 'month-index', index: 7, startTimeIndex: 168, timeCount: 24 },
				numPoints: 2,
				gpuBuffer: {} as GPUBuffer,
				debugLabel: 'webgpu-shading-index'
			}
		});

		expect(key).toBe('5:shading_index:7:168:0:1');
	});

	it('does not treat userData alone as authoritative compute-buffer proof', () => {
		const mesh = new Mesh();
		mesh.userData.utciSurfaceSource = 'compute-buffer-selected-hour';
		expect(isComputeBufferUtciSurface(mesh)).toBe(false);
	});

	it('keeps analysis identity change reusable when the compute-buffer surface layout still matches', () => {
		expect(
			shouldRecreateComputeBufferUtciSurface({
				missingSurface: false,
				notComputeBufferSurface: false,
				analysisIdentityChanged: true,
				layoutCompatible: true
			})
		).toEqual({
			shouldRecreate: false,
			recreateDecision: {
				missingSurface: false,
				notComputeBufferSurface: false,
				analysisIdentityChanged: true,
				layoutCompatible: true
			}
		});
	});

	it('still recreates compute-buffer surfaces for missing, wrong-type, or layout-incompatible cases', () => {
		expect(
			shouldRecreateComputeBufferUtciSurface({
				missingSurface: true,
				notComputeBufferSurface: false,
				analysisIdentityChanged: false,
				layoutCompatible: false
			}).shouldRecreate
		).toBe(true);
		expect(
			shouldRecreateComputeBufferUtciSurface({
				missingSurface: false,
				notComputeBufferSurface: true,
				analysisIdentityChanged: false,
				layoutCompatible: true
			}).shouldRecreate
		).toBe(true);
		expect(
			shouldRecreateComputeBufferUtciSurface({
				missingSurface: false,
				notComputeBufferSurface: false,
				analysisIdentityChanged: true,
				layoutCompatible: false
			}).shouldRecreate
		).toBe(true);
	});

	it('builds request-scoped CPU publication diagnostics for non-compute surfaces', () => {
		const mesh = new Mesh();
		const diagnostics = buildCpuPublicationDiagnostics({
			mesh,
			liveSelectedHourSurfaceIdentity: {
				controllerIdentity: 'test-controller',
				controllerInstanceId: 0,
				requestId: 9,
				monthIndex: 1,
				hourIndex: 8,
				timeIndex: 32,
				selectionKey: 'selection',
				pendingRenderUpdateStartedAt: undefined,
				acceptedGpuResidentOutput: null
			}
		});

		expect(diagnostics).toMatchObject({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			cpuPublishRequestId: 9,
			cpuPublishSelectionKey: 'selection'
		});
	});

	it('assembles common surface diagnostics from mesh and copy state', () => {
		const mesh = new Mesh();
		mesh.userData.utciSurfaceSource = 'data-texture';
		mesh.userData.selectedHourTransferCount = 3;
		mesh.userData.dataTextureBuildCount = 2;
		mesh.userData.renderOwnedSelectedHourBytes = 1024;

		const diagnostics = buildUtciSurfaceDiagnostics({
			mesh,
			cpuPublicationDiagnostics: {
				utciSurfaceSource: 'cpu-uploaded-selected-hour',
				cpuPublishRequestId: 9
			},
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 9,
			gpuResidentRenderTimings: {
				renderBufferCopyMs: 1.5
			}
		});

		expect(diagnostics).toMatchObject({
			utciSurfaceSource: 'cpu-uploaded-selected-hour',
			selectedHourTransferCount: 3,
			dataTextureBuildCount: 2,
			renderOwnedSelectedHourBytes: 1024,
			cpuPublishRequestId: 9,
			gpuResidentCopyStatus: 'complete',
			gpuResidentCopyRequestId: 9,
			renderBufferCopyMs: 1.5
		});
	});

	it('includes render publication diagnostics in UTCI surface diagnostics', () => {
		const renderPublication = {
			renderPublicationVersion: 1 as const,
			renderPublicationPath: 'compute-buffer-selected-hour' as const,
			renderPublicationPhase: 'scrub' as const,
			renderPublicationMeshAction: 'reused' as const,
			renderPublicationPointCount: 100,
			renderPublicationTargetByteLength: 400,
			renderPublicationTimeline: {
				routeProjectedAtMs: 101,
				scenePendingSurfaceObservedAtMs: 102,
				sceneReactiveToSyncQueuedMs: 12,
				sceneSyncQueuedToStartMs: 34,
				sceneSyncAttemptStartedAtMs: 102.5,
				sceneSyncAttemptToken: 8,
				sceneSurfaceReceivedAtMs: 103,
				sceneLayoutKeyStartedAtMs: 102.75,
				sceneLayoutKeyCompletedAtMs: 103.25,
				scenePublicationPlanReadyAtMs: 104,
				renderLayoutReuseSourceSignatureMs: 0.2,
				renderLayoutReusePositionsSignatureMs: 0.1,
				renderLayoutReusePositionsSignatureCacheHit: true,
				renderLayoutReuseFrameCacheLookupMs: 0.05,
				renderStorageWaitStartedAtMs: 105,
				renderStoragePreWaitMs: 2.5,
				sceneSyncCompletedAtMs: 107,
				sceneSyncActiveWindowResetHistory: [
					{
						resetAtMs: 102.25,
						resetReason: 'fallback-cpu-surface',
						invalidateActiveRun: false,
						previousCopyRunToken: 7,
						nextCopyRunToken: 7
					}
				]
			}
		};
		const diagnostics = buildUtciSurfaceDiagnostics({
			mesh: {
				userData: {
					utciSurfaceSource: 'compute-buffer-selected-hour',
					renderOwnedSelectedHourBytes: 2048
				}
			} as any,
			gpuResidentCopyStatus: 'complete',
			gpuResidentRenderTimings: {
				renderSceneSyncTotalMs: 10,
				renderPublication
			}
		});

		renderPublication.renderPublicationTimeline.sceneSyncCompletedAtMs = 999;
		renderPublication.renderPublicationTimeline.sceneSyncActiveWindowResetHistory[0].resetReason =
			'mutated';

		expect(diagnostics.renderPublication).toMatchObject({
			renderPublicationPath: 'compute-buffer-selected-hour',
			renderPublicationPhase: 'scrub',
			renderPublicationMeshAction: 'reused',
			renderPublicationTimeline: {
				routeProjectedAtMs: 101,
				scenePendingSurfaceObservedAtMs: 102,
				sceneReactiveToSyncQueuedMs: 12,
				sceneSyncQueuedToStartMs: 34,
				sceneSyncAttemptStartedAtMs: 102.5,
				sceneSyncAttemptToken: 8,
				sceneSurfaceReceivedAtMs: 103,
				sceneLayoutKeyStartedAtMs: 102.75,
				sceneLayoutKeyCompletedAtMs: 103.25,
				scenePublicationPlanReadyAtMs: 104,
				renderLayoutReuseSourceSignatureMs: 0.2,
				renderLayoutReusePositionsSignatureMs: 0.1,
				renderLayoutReusePositionsSignatureCacheHit: true,
				renderLayoutReuseFrameCacheLookupMs: 0.05,
				renderStorageWaitStartedAtMs: 105,
				renderStoragePreWaitMs: 2.5,
				sceneSyncCompletedAtMs: 107,
				sceneSyncActiveWindowResetHistory: [
					{
						resetAtMs: 102.25,
						resetReason: 'fallback-cpu-surface',
						invalidateActiveRun: false,
						previousCopyRunToken: 7,
						nextCopyRunToken: 7
					}
				]
			}
		});
	});
});

describe('acceptedGpuResidentSurfaceSync', () => {
	const acceptedOutput = {
		requestId: 5,
		monthIndex: 7,
		hourIndex: 1,
		timeIndex: 169,
		utciRange: { min: 18, max: 42 },
		output: {
			format: 'f32-utci',
			numPoints: 1,
			timeIndex: 169,
			gpuBuffer: {} as GPUBuffer
		}
	} as const;
	const liveSelectedHourSurfaceIdentity = {
		controllerIdentity: 'test-controller',
		controllerInstanceId: 3,
		requestId: 5,
		monthIndex: 7,
		hourIndex: 1,
		timeIndex: 169,
		selectionKey: 'selection',
		pendingRenderUpdateStartedAt: undefined,
		acceptedGpuResidentOutput: null
	};

	it('records bounded reset history with copy-safe snapshots', () => {
		const statuses: string[] = [];
		const sync = createSurfaceSync({
			componentName: 'test',
			getOnAcceptedGpuResidentOutputRelease: () => undefined,
			setCopyDiagnostics: (status) => statuses.push(status)
		});

		sync.reset({
			invalidateActiveRun: true,
			reason: 'first-reset',
			now: () => 10
		});

		const firstSnapshot = sync.getResetHistory();
		firstSnapshot[0].resetReason = 'mutated';

		expect(sync.getResetHistory()).toEqual([
			{
				resetAtMs: 10,
				resetReason: 'first-reset',
				invalidateActiveRun: true,
				previousCopyRunToken: 0,
				nextCopyRunToken: 1,
				previousSyncRunKey: undefined
			}
		]);

		for (let index = 0; index < 12; index += 1) {
			sync.reset({
				reason: `reset-${index}`,
				now: () => 20 + index
			});
		}

		const history = sync.getResetHistory();
		expect(history).toHaveLength(10);
		expect(history[0]).toMatchObject({
			resetAtMs: 22,
			resetReason: 'reset-2',
			previousCopyRunToken: 1,
			nextCopyRunToken: 1
		});
		expect(history.at(-1)).toMatchObject({
			resetAtMs: 31,
			resetReason: 'reset-11'
		});
		expect(statuses).toContain('idle');
	});

	it('keeps invalidating resets token-bumping and active-sync-clearing', () => {
		const statuses: string[] = [];
		const sync = createSurfaceSync({
			componentName: 'test',
			getOnAcceptedGpuResidentOutputRelease: () => undefined,
			setCopyDiagnostics: (status) => statuses.push(status)
		});

		const run = sync.startSync({
			acceptedOutput,
			liveSelectedHourSurfaceIdentity
		});
		expect(run).not.toBeNull();
		expect(sync.getActiveSyncRunKey()).toBe(run!.syncRunKey);
		expect(sync.getActiveCopyRunToken()).toBe(1);

		sync.reset({
			invalidateActiveRun: true,
			reason: 'dispose-utci-surface',
			now: () => 20
		});

		expect(sync.getActiveSyncRunKey()).toBeNull();
		expect(sync.getActiveCopyRunToken()).toBe(2);
		expect(sync.getResetHistory()).toEqual([
			{
				resetAtMs: 20,
				resetReason: 'dispose-utci-surface',
				invalidateActiveRun: true,
				previousCopyRunToken: 1,
				nextCopyRunToken: 2,
				previousSyncRunKey: run!.syncRunKey
			}
		]);
		expect(statuses).toEqual(['pending', 'idle']);
	});

	it('preserves active sync state for matching non-invalidating compute-surface recreation', () => {
		const statuses: string[] = [];
		const sync = createSurfaceSync({
			componentName: 'test',
			getOnAcceptedGpuResidentOutputRelease: () => undefined,
			setCopyDiagnostics: (status) => statuses.push(status)
		});

		const run = sync.startSync({
			acceptedOutput,
			liveSelectedHourSurfaceIdentity
		});
		expect(run).not.toBeNull();

		const didReset = sync.resetUnlessActiveSyncRunKeyMatches({
			expectedActiveSyncRunKey: run!.syncRunKey,
			invalidateActiveRun: false,
			reason: 'compute-surface-recreation',
			now: () => 30
		});

		expect(didReset).toBe(false);
		expect(sync.getActiveSyncRunKey()).toBe(run!.syncRunKey);
		expect(sync.getActiveCopyRunToken()).toBe(1);
		expect(sync.getResetHistory()).toEqual([]);
		expect(statuses).toEqual(['pending']);
	});

	it('resets non-invalidating compute-surface recreation when the sync key is missing or stale', () => {
		const statuses: string[] = [];
		const sync = createSurfaceSync({
			componentName: 'test',
			getOnAcceptedGpuResidentOutputRelease: () => undefined,
			setCopyDiagnostics: (status) => statuses.push(status)
		});

		const run = sync.startSync({
			acceptedOutput,
			liveSelectedHourSurfaceIdentity
		});
		expect(run).not.toBeNull();

		expect(
			sync.resetUnlessActiveSyncRunKeyMatches({
				expectedActiveSyncRunKey: null,
				invalidateActiveRun: false,
				reason: 'compute-surface-recreation',
				now: () => 40
			})
		).toBe(true);
		expect(sync.getActiveSyncRunKey()).toBeNull();
		expect(sync.getActiveCopyRunToken()).toBe(1);

		const nextRun = sync.startSync({
			acceptedOutput,
			liveSelectedHourSurfaceIdentity
		});
		expect(nextRun).not.toBeNull();
		expect(
			sync.resetUnlessActiveSyncRunKeyMatches({
				expectedActiveSyncRunKey: `${nextRun!.syncRunKey}:stale`,
				invalidateActiveRun: false,
				reason: 'compute-surface-recreation',
				now: () => 41
			})
		).toBe(true);
		expect(sync.getActiveSyncRunKey()).toBeNull();
		expect(sync.getActiveCopyRunToken()).toBe(2);
		expect(sync.getResetHistory()).toEqual([
			expect.objectContaining({
				resetAtMs: 40,
				resetReason: 'compute-surface-recreation',
				invalidateActiveRun: false,
				previousCopyRunToken: 1,
				nextCopyRunToken: 1,
				previousSyncRunKey: run!.syncRunKey
			}),
			expect.objectContaining({
				resetAtMs: 41,
				resetReason: 'compute-surface-recreation',
				invalidateActiveRun: false,
				previousCopyRunToken: 2,
				nextCopyRunToken: 2,
				previousSyncRunKey: nextRun!.syncRunKey
			})
		]);
		expect(statuses).toEqual(['pending', 'idle', 'pending', 'idle']);
	});

	it('explains missing controller identity resets before starting a sync', () => {
		const sync = createSurfaceSync({
			componentName: 'test',
			getOnAcceptedGpuResidentOutputRelease: () => undefined,
			setCopyDiagnostics: () => undefined
		});

		const run = sync.startSync({
			acceptedOutput,
			liveSelectedHourSurfaceIdentity: null
		});

		expect(run).toBeNull();
		expect(sync.getResetHistory()).toEqual([
			expect.objectContaining({
				resetReason: 'missing-controller-identity',
				invalidateActiveRun: true,
				previousCopyRunToken: 0,
				nextCopyRunToken: 1
			})
		]);
	});
});
