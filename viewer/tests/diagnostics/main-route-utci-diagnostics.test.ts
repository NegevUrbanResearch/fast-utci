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
			comparisonSameDeviceForComputeAndRender: null,
			selectedHourReadbackReasons: ['range', 'tooltip'],
			selectedHourReadbackReasonCounts: {
				range: 1,
				tooltip: 1
			}
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
		expect(diagnostics?.selectedHourRuntimeContract.readbackInstrumentation).toBe('not-instrumented');
		expect(diagnostics?.selectedHourRuntimeContract.acceptedRequestId).toBe(3);
		expect(diagnostics?.selectedHourRuntimeContract.sceneRequestId).toBe(3);
		expect(diagnostics?.selectedHourRuntimeContract.requestMatchesScene).toBe(true);
		expect(diagnostics?.selectedHourRuntimeContract.strongVisibleGpuPath).toBe(false);
		expect(diagnostics?.selectedHourRuntimeContract.visibleRenderPathAvoidsCpuReadback).toBe(false);
		expect(diagnostics?.selectedHourRuntimeContract.readbackReasons).toEqual([
			'range',
			'tooltip'
		]);
		expect(diagnostics?.selectedHourRuntimeContract.readbackReasonCounts).toEqual({
			range: 1,
			tooltip: 1
		});
		expect(diagnostics?.selectedHourRuntimeContract.totalSelectedHourReadbackReasonCount).toBe(2);
		expect(JSON.stringify(diagnostics)).not.toMatch(/\.bin|parity|Python|loadReferenceFromFs/i);
	});

	it('includes comparison readback reasons and counts in the main route contract', () => {
		const diagnostics = buildMainRouteUtciDiagnostics({
			enabled: true,
			utciOnDemand: 'f32',
			utciRenderRequested: 'auto',
			utciRenderResolved: 'gpuNative',
			rendererBackend: 'webgpu',
			baseSurfaceDiagnostics: {
				utciSurfaceSource: 'compute-buffer-selected-hour',
				selectedHourTransferCount: 0,
				dataTextureBuildCount: 0
			},
			comparisonSurfaceDiagnostics: {
				utciSurfaceSource: 'compute-buffer-selected-hour',
				selectedHourTransferCount: 0,
				dataTextureBuildCount: 0
			},
			baseRenderTransport: 'compute-buffer-selected-hour',
			comparisonRenderTransport: 'compute-buffer-selected-hour',
			baseLiveReady: true,
			comparisonLiveReady: true,
			baseSurfaceRequestId: 3,
			baseSelectionKey: 'base|7|12',
			baseSceneSurfaceRequestId: 3,
			baseSceneSelectionKey: 'base|7|12',
			baseSameDeviceForComputeAndRender: true,
			baseSelectedMonthIndex: 7,
			baseSelectedHourIndex: 12,
			baseSelectedTimeIndex: 180,
			comparisonSurfaceRequestId: 4,
			comparisonSelectionKey: 'comparison|7|12',
			comparisonSameDeviceForComputeAndRender: true,
			selectedHourReadbackReasons: ['range'],
			selectedHourReadbackReasonCounts: {
				range: 1
			},
			comparisonSelectedHourReadbackReasons: ['comparison', 'tooltip'],
			comparisonSelectedHourReadbackReasonCounts: {
				comparison: 1,
				tooltip: 2
			}
		});

		expect(diagnostics?.selectedHourReadbackReasons).toEqual([
			'range',
			'comparison',
			'tooltip'
		]);
		expect(diagnostics?.selectedHourReadbackReasonCounts).toEqual({
			range: 1,
			comparison: 1,
			tooltip: 2
		});
		expect(diagnostics?.selectedHourRuntimeContract.readbackInstrumentation).toBe('not-instrumented');
		expect(diagnostics?.selectedHourRuntimeContract.strongVisibleGpuPath).toBe(false);
		expect(diagnostics?.selectedHourRuntimeContract.visibleRenderPathAvoidsCpuReadback).toBe(false);
		expect(diagnostics?.selectedHourRuntimeContract.readbackReasons).toEqual([
			'range',
			'comparison',
			'tooltip'
		]);
		expect(diagnostics?.selectedHourRuntimeContract.readbackReasonCounts).toEqual({
			range: 1,
			comparison: 1,
			tooltip: 2
		});
		expect(diagnostics?.selectedHourRuntimeContract.totalSelectedHourReadbackReasonCount).toBe(4);
		expect(JSON.stringify(diagnostics)).not.toMatch(/\.bin|parity|Python|loadReferenceFromFs/i);
	});

	it('publishes a strong visible GPU path when visible readbacks are explicitly instrumented', () => {
		const diagnostics = buildMainRouteUtciDiagnostics({
			enabled: true,
			utciOnDemand: 'f32',
			utciRenderRequested: 'auto',
			utciRenderResolved: 'gpuNative',
			rendererBackend: 'webgpu',
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
			comparisonSameDeviceForComputeAndRender: null,
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented'
		});

		expect(diagnostics?.selectedHourRuntimeContract).toMatchObject({
			readbackInstrumentation: 'instrumented',
			visibleSelectedHourReadbackCount: 0,
			visibleSelectedHourReadbackCountInstrumented: true,
			strongVisibleGpuPath: true
		});
	});
});
