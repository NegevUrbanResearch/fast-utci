import { describe, expect, it, vi } from 'vitest';

import {
	buildMainRouteLiveSelectedHourDiagnostics,
	releaseBaseAcceptedGpuResidentOutput,
	releaseComparisonAcceptedGpuResidentOutput,
	type MainRouteAcceptedGpuResidentOutputReleaseParams,
	type MainRouteLiveSelectedHourDiagnosticsParams
} from '../../src/routes/main/liveSelectedHour';
import type { LiveSelectedHourRouteHost } from '$lib/compute/liveSelectedHourRouteHost';

function createReleaseParams(): MainRouteAcceptedGpuResidentOutputReleaseParams {
	return {
		controllerIdentity: 'controller-a',
		controllerInstanceId: 7,
		requestId: 11,
		monthIndex: 2,
		timeIndex: 51,
		reason: 'copy-complete'
	};
}

function createDiagnosticsParams(): MainRouteLiveSelectedHourDiagnosticsParams {
	return {
		enabled: true,
		utciOnDemand: 'f32',
		utciRenderRequested: 'gpu',
		utciRenderResolved: 'gpuNative',
		rendererBackend: 'webgpu',
		rendererRequiredLimits: undefined,
		rendererDeviceLimits: undefined,
		liveRouteState: {
			base: {
				renderSurfaceDiagnostics: {
					utciSurfaceSource: 'compute-buffer-selected-hour',
					selectedHourTransferCount: 0,
					dataTextureBuildCount: 0,
					gpuResidentCopyStatus: 'complete',
					gpuResidentCopyRequestId: 11
				},
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true,
				visibleSelectedHourReadbackCount: 0,
				readbackInstrumentation: 'not-instrumented',
				selectedHourReadbackReasons: [],
				selectedHourReadbackReasonCounts: {}
			},
			comparison: {
				renderSurfaceDiagnostics: {
					utciSurfaceSource: 'compute-buffer-selected-hour',
					selectedHourTransferCount: 0,
					dataTextureBuildCount: 0,
					gpuResidentCopyStatus: 'complete',
					gpuResidentCopyRequestId: 13
				},
				renderTransport: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true,
				selectedHourReadbackReasons: [],
				selectedHourReadbackReasonCounts: {}
			},
			baseSurfaceIdentity: {
				requestId: 11,
				selectionKey: 'analysis|7|12'
			},
			baseSceneSurfaceIdentity: {
				requestId: 11,
				selectionKey: 'analysis|7|12'
			},
			comparisonSurfaceIdentity: {
				requestId: 13,
				selectionKey: 'comparison|7|12'
			}
		} as unknown as MainRouteLiveSelectedHourDiagnosticsParams['liveRouteState'],
		lastBaseGpuResidentCopyFailure: undefined,
		baseLiveReady: true,
		comparisonLiveReady: true,
		selectedMonthIndex: 7,
		selectedHourIndex: 12,
		selectedTimeIndex: 180,
		baseSceneRenderContextTimeIndex: 180,
		baseAcceptedUtciRange: { min: -10, max: 42 },
		tooltipHoverSampleCount: 3,
		cameraWheelEventCount: 1
	};
}

describe('main route live selected-hour helper', () => {
	it('forwards base accepted GPU resident output releases with controller instance ids intact', () => {
		const host = {
			releaseBaseAcceptedGpuResidentOutput: vi.fn(),
			releaseComparisonAcceptedGpuResidentOutput: vi.fn()
		} as unknown as LiveSelectedHourRouteHost;
		const params = createReleaseParams();

		releaseBaseAcceptedGpuResidentOutput(host, params);

		expect(host.releaseBaseAcceptedGpuResidentOutput).toHaveBeenCalledWith(params);
	});

	it('forwards comparison accepted GPU resident output releases with controller instance ids intact', () => {
		const host = {
			releaseBaseAcceptedGpuResidentOutput: vi.fn(),
			releaseComparisonAcceptedGpuResidentOutput: vi.fn()
		} as unknown as LiveSelectedHourRouteHost;
		const params = createReleaseParams();

		releaseComparisonAcceptedGpuResidentOutput(host, params);

		expect(host.releaseComparisonAcceptedGpuResidentOutput).toHaveBeenCalledWith(params);
	});

	it('builds diagnostics without debug-only fields', () => {
		const diagnostics = buildMainRouteLiveSelectedHourDiagnostics(createDiagnosticsParams());

		expect(diagnostics).toMatchObject({
			baseRenderTransport: 'compute-buffer-selected-hour',
			comparisonRenderTransport: 'compute-buffer-selected-hour',
			baseSelectionKey: 'analysis|7|12'
		});
		expect(diagnostics?.selectedHourRuntimeContract.acceptedRequestId).toBe(11);
		expect(JSON.stringify(diagnostics)).not.toMatch(
			/\.bin|parity|Python|loadReferenceFromFs|__onDemandPrototypeDiagnostics__|LEGACY_DEBUG_SELECTED_HOUR_CONTROLLER_ID/i
		);
	});
});
