import { describe, expect, it } from 'vitest';
import { buildMainRouteUtciDiagnostics } from '$lib/diagnostics/mainRouteUtciDiagnostics';

describe('buildMainRouteUtciDiagnostics', () => {
	it('returns undefined when diagnostics are disabled', () => {
		expect(
			buildMainRouteUtciDiagnostics({
				enabled: false,
				utciOnDemand: 'f32',
				utciRenderRequested: 'auto',
				utciRenderResolved: 'gpuNative',
				rendererBackend: 'webgpu',
				baseSurfaceDiagnostics: {},
				comparisonSurfaceDiagnostics: {},
				baseRenderTransport: 'idle',
				comparisonRenderTransport: 'idle',
				baseLiveReady: false,
				comparisonLiveReady: true,
				baseSameDeviceForComputeAndRender: null,
				comparisonSameDeviceForComputeAndRender: null,
				baseSelectedMonthIndex: 7,
				baseSelectedHourIndex: 12,
				baseSelectedTimeIndex: 180
			})
		).toBeUndefined();
	});

	it('builds a gpu-native selected-hour payload without debug parity fields', () => {
		const diagnostics = buildMainRouteUtciDiagnostics({
			enabled: true,
			utciOnDemand: 'f32',
			utciRenderRequested: 'auto',
			utciRenderResolved: 'gpuNative',
			rendererBackend: 'webgpu',
			rendererRequiredLimits: { maxStorageBufferBindingSize: 1, maxBufferSize: 1 },
			rendererDeviceLimits: { maxStorageBufferBindingSize: 1, maxBufferSize: 1 },
			baseSurfaceDiagnostics: {
				utciSurfaceSource: 'compute-buffer-selected-hour',
				selectedHourTransferCount: 0,
				dataTextureBuildCount: 0,
				gpuResidentCopyStatus: 'complete',
				gpuResidentCopyRequestId: 3
			},
			comparisonSurfaceDiagnostics: {},
			baseRenderTransport: 'compute-buffer-selected-hour',
			comparisonRenderTransport: 'idle',
			baseLiveReady: true,
			comparisonLiveReady: true,
			baseSurfaceRequestId: 3,
			baseSelectionKey: 'analysis|7|12',
			baseSceneSurfaceRequestId: 3,
			baseSceneSelectionKey: 'analysis|7|12',
			baseSameDeviceForComputeAndRender: true,
			baseSelectedMonthIndex: 7,
			baseSelectedHourIndex: 12,
			baseSelectedTimeIndex: 180,
			baseRenderContextTimeIndex: 180,
			baseAcceptedUtciRange: { min: 20, max: 41 },
			comparisonSameDeviceForComputeAndRender: null
		});

		expect(diagnostics).toMatchObject({
			utciOnDemand: 'f32',
			utciRenderRequested: 'auto',
			utciRenderResolved: 'gpuNative',
			rendererBackend: 'webgpu',
			utciSurfaceSource: 'compute-buffer-selected-hour',
			baseRenderTransport: 'compute-buffer-selected-hour',
			baseLiveReady: true,
			baseSurfaceRequestId: 3,
			baseSelectionKey: 'analysis|7|12',
			baseSelectedMonthIndex: 7,
			baseSelectedHourIndex: 12,
			baseSelectedTimeIndex: 180,
			baseAcceptedUtciRange: { min: 20, max: 41 }
		});
		expect(JSON.stringify(diagnostics)).not.toMatch(/\.bin|parity|Python|loadReferenceFromFs/i);
	});
});
