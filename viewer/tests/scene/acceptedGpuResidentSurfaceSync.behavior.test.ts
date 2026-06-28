import { describe, expect, it, vi } from 'vitest';

import {
	createAcceptedGpuResidentSurfaceSync,
	type AcceptedGpuResidentSurfaceSyncRun
} from '$lib/components/scene/acceptedGpuResidentSurfaceSync';
import type { LiveSelectedHourSurfaceIdentity } from '$lib/compute/selected-hour/liveSelectedHourSurfaceIdentity';
import type { SelectedHourGpuResidentOutput } from '$lib/compute/selected-hour/liveUtciSelectedHourSession';
import type { SelectedHourRenderTimingSubsteps } from '$lib/compute/on-demand/onDemandDiagnostics';
import type { AcceptedGpuResidentOutputReleaseCallback } from '$lib/components/scene/acceptedGpuResidentOutputRelease';
import type { GpuResidentCopyStatus } from '$lib/components/scene/utciSurfaceSync';

function createAcceptedOutput(): SelectedHourGpuResidentOutput {
	return {
		requestId: 11,
		monthIndex: 7,
		hourIndex: 12,
		timeIndex: 180,
		utciRange: { min: 18.5, max: 41.25 },
		output: {
			format: 'f32-utci',
			numPoints: 4,
			timeIndex: 180,
			gpuBuffer: { size: 16 } as GPUBuffer
		} as SelectedHourGpuResidentOutput['output']
	};
}

function createSurfaceIdentity(
	controllerInstanceId: number
): LiveSelectedHourSurfaceIdentity {
	return {
		controllerIdentity: 'controller-a',
		controllerInstanceId,
		requestId: 11,
		monthIndex: 7,
		hourIndex: 12,
		timeIndex: 180,
		selectionKey: 'selection',
		pendingRenderUpdateStartedAt: undefined,
		acceptedGpuResidentOutput: null
	};
}

function startSyncOrThrow(
	startedRun: AcceptedGpuResidentSurfaceSyncRun | null
): AcceptedGpuResidentSurfaceSyncRun {
	expect(startedRun).not.toBeNull();
	return startedRun as AcceptedGpuResidentSurfaceSyncRun;
}

describe('acceptedGpuResidentSurfaceSync behavior', () => {
	it('maps late old-instance completion to superseded and releases the current instance exactly once', () => {
		const onAcceptedGpuResidentOutputRelease =
			vi.fn<AcceptedGpuResidentOutputReleaseCallback>();
		const setCopyDiagnostics = vi.fn<
			(status: GpuResidentCopyStatus, options?: {
				error?: string;
				requestId?: number;
				renderTimings?: SelectedHourRenderTimingSubsteps;
			}) => void
		>();
		const sync = createAcceptedGpuResidentSurfaceSync({
			componentName: 'behavior-test',
			getOnAcceptedGpuResidentOutputRelease: () =>
				onAcceptedGpuResidentOutputRelease,
			setCopyDiagnostics
		});
		const acceptedOutput = createAcceptedOutput();
		let liveSelectedHourSurfaceIdentity = createSurfaceIdentity(1);

		const instance1Run = startSyncOrThrow(
			sync.startSync({
				acceptedOutput,
				liveSelectedHourSurfaceIdentity
			})
		);

		liveSelectedHourSurfaceIdentity = createSurfaceIdentity(2);
		const instance2Run = startSyncOrThrow(
			sync.startSync({
				acceptedOutput,
				liveSelectedHourSurfaceIdentity
			})
		);

		expect(
			sync.completeSync(instance1Run, {
				acceptedGpuResidentOutput: acceptedOutput,
				liveSelectedHourSurfaceIdentity,
				renderTimings: { renderBufferCopyMs: 1.25 }
			})
		).toBe('superseded');

		expect(onAcceptedGpuResidentOutputRelease).toHaveBeenCalledTimes(1);
		expect(onAcceptedGpuResidentOutputRelease).toHaveBeenNthCalledWith(1, {
			controllerIdentity: 'controller-a',
			controllerInstanceId: 1,
			requestId: 11,
			monthIndex: 7,
			timeIndex: 180,
			reason: 'superseded'
		});
		expect(setCopyDiagnostics).toHaveBeenNthCalledWith(1, 'pending', {
			requestId: 11
		});
		expect(setCopyDiagnostics).toHaveBeenNthCalledWith(2, 'pending', {
			requestId: 11
		});

		expect(
			sync.completeSync(instance2Run, {
				acceptedGpuResidentOutput: acceptedOutput,
				liveSelectedHourSurfaceIdentity,
				renderTimings: { renderBufferCopyMs: 2.5 }
			})
		).toBe('complete');
		expect(
			sync.completeSync(instance2Run, {
				acceptedGpuResidentOutput: acceptedOutput,
				liveSelectedHourSurfaceIdentity,
				renderTimings: { renderBufferCopyMs: 3.5 }
			})
		).toBe('already-released');

		expect(onAcceptedGpuResidentOutputRelease).toHaveBeenCalledTimes(2);
		expect(onAcceptedGpuResidentOutputRelease).toHaveBeenNthCalledWith(2, {
			controllerIdentity: 'controller-a',
			controllerInstanceId: 2,
			requestId: 11,
			monthIndex: 7,
			timeIndex: 180,
			reason: 'copy-complete'
		});
		expect(setCopyDiagnostics).toHaveBeenNthCalledWith(3, 'complete', {
			requestId: 11,
			renderTimings: { renderBufferCopyMs: 2.5 }
		});
		expect(setCopyDiagnostics).toHaveBeenCalledTimes(3);
	});

	it('maps late old-instance rejection to superseded without failing the current instance', () => {
		const onAcceptedGpuResidentOutputRelease =
			vi.fn<AcceptedGpuResidentOutputReleaseCallback>();
		const setCopyDiagnostics = vi.fn<
			(status: GpuResidentCopyStatus, options?: {
				error?: string;
				requestId?: number;
				renderTimings?: SelectedHourRenderTimingSubsteps;
			}) => void
		>();
		const sync = createAcceptedGpuResidentSurfaceSync({
			componentName: 'behavior-test',
			getOnAcceptedGpuResidentOutputRelease: () =>
				onAcceptedGpuResidentOutputRelease,
			setCopyDiagnostics
		});
		const acceptedOutput = createAcceptedOutput();
		let liveSelectedHourSurfaceIdentity = createSurfaceIdentity(1);

		const instance1Run = startSyncOrThrow(
			sync.startSync({
				acceptedOutput,
				liveSelectedHourSurfaceIdentity
			})
		);

		liveSelectedHourSurfaceIdentity = createSurfaceIdentity(2);
		const instance2Run = startSyncOrThrow(
			sync.startSync({
				acceptedOutput,
				liveSelectedHourSurfaceIdentity
			})
		);

		expect(
			sync.failSync(instance1Run, {
				acceptedGpuResidentOutput: acceptedOutput,
				liveSelectedHourSurfaceIdentity,
				errorMessage: 'late copy failure'
			})
		).toBe('superseded');

		expect(onAcceptedGpuResidentOutputRelease).toHaveBeenCalledTimes(1);
		expect(onAcceptedGpuResidentOutputRelease).toHaveBeenNthCalledWith(1, {
			controllerIdentity: 'controller-a',
			controllerInstanceId: 1,
			requestId: 11,
			monthIndex: 7,
			timeIndex: 180,
			reason: 'superseded'
		});
		expect(setCopyDiagnostics).toHaveBeenNthCalledWith(1, 'pending', {
			requestId: 11
		});
		expect(setCopyDiagnostics).toHaveBeenNthCalledWith(2, 'pending', {
			requestId: 11
		});

		expect(
			sync.completeSync(instance2Run, {
				acceptedGpuResidentOutput: acceptedOutput,
				liveSelectedHourSurfaceIdentity,
				renderTimings: { renderQueueDrainMs: 4.5 }
			})
		).toBe('complete');

		expect(onAcceptedGpuResidentOutputRelease).toHaveBeenCalledTimes(2);
		expect(onAcceptedGpuResidentOutputRelease).toHaveBeenNthCalledWith(2, {
			controllerIdentity: 'controller-a',
			controllerInstanceId: 2,
			requestId: 11,
			monthIndex: 7,
			timeIndex: 180,
			reason: 'copy-complete'
		});
		expect(setCopyDiagnostics).toHaveBeenNthCalledWith(3, 'complete', {
			requestId: 11,
			renderTimings: { renderQueueDrainMs: 4.5 }
		});
		expect(setCopyDiagnostics).toHaveBeenCalledTimes(3);
	});

	it('includes controller instance id in the active sync run key', () => {
		const sync = createAcceptedGpuResidentSurfaceSync({
			componentName: 'behavior-test',
			getOnAcceptedGpuResidentOutputRelease: () => undefined,
			setCopyDiagnostics: vi.fn()
		});
		const acceptedOutput = createAcceptedOutput();

		const instance1Key = sync.getSyncRunKey({
			acceptedGpuResidentOutput: acceptedOutput,
			liveSelectedHourSurfaceIdentity: createSurfaceIdentity(1)
		});
		const instance2Key = sync.getSyncRunKey({
			acceptedGpuResidentOutput: acceptedOutput,
			liveSelectedHourSurfaceIdentity: createSurfaceIdentity(2)
		});

		expect(instance1Key).not.toBeNull();
		expect(instance2Key).not.toBeNull();
		expect(instance1Key).not.toBe(instance2Key);

		const instance1Run = startSyncOrThrow(
			sync.startSync({
				acceptedOutput,
				liveSelectedHourSurfaceIdentity: createSurfaceIdentity(1)
			})
		);
		expect(sync.getActiveSyncKey()).toBe(instance1Run.syncKey);
		expect(sync.getActiveSyncRunKey()).toBe(instance1Key);
	});
});
