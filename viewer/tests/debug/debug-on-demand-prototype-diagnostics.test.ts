import { describe, expect, it } from 'vitest';
import { createEmptyOnDemandDiagnostics } from '$lib/compute/on-demand/onDemandDiagnostics';
import {
	buildDebugOnDemandPrototypeDiagnostics,
	type DebugOnDemandPrototypeDiagnosticsDefaults
} from '$lib/debug/debugOnDemandPrototypeDiagnostics';
import type { SelectedHourReadbackReason } from '$lib/diagnostics/selectedHourRuntimeContract';
import { createEmptyTooltipInteractionDiagnostics } from '$lib/services/tooltipService';
import { createEmptyCameraInteractionTelemetry } from '$lib/services/cameraInteractionTelemetry';

function createTooltipDiagnostics(options?: { disabledByQuery?: boolean; sampleCount?: number }) {
	return {
		...createEmptyTooltipInteractionDiagnostics(options?.disabledByQuery ?? false),
		sampleCount: options?.sampleCount ?? 0
	};
}

function createCameraDiagnostics(wheelEventCount = 0) {
	return {
		...createEmptyCameraInteractionTelemetry().diagnostics,
		wheelEventCount
	};
}

function createDefaults(
	overrides: Partial<DebugOnDemandPrototypeDiagnosticsDefaults> = {}
): DebugOnDemandPrototypeDiagnosticsDefaults {
	return {
		navigatorGpu: true,
		rendererBackend: 'webgpu',
		utciRenderRequested: 'auto',
		utciRenderResolved: 'gpuNative',
		selectedHourEngine: 'shared-host',
		binComparisonEnabled: false,
		binComparisonValid: false,
		legacySelectedHourDispatchCount: 0,
		legacyScrubScheduleCount: 0,
		tooltipInteraction: createTooltipDiagnostics(),
		cameraInteraction: createCameraDiagnostics(),
		readbackInstrumentation: 'not-instrumented',
		...overrides
	};
}

function getRuntimeContract(result: ReturnType<typeof buildDebugOnDemandPrototypeDiagnostics>) {
	expect(result.selectedHourRuntimeContract).toBeDefined();
	if (!result.selectedHourRuntimeContract) {
		throw new Error('Expected selectedHourRuntimeContract to be defined');
	}
	return result.selectedHourRuntimeContract;
}

describe('buildDebugOnDemandPrototypeDiagnostics', () => {
	it('merges existing diagnostics and preserves debug-only parity validity', () => {
		const existing = {
			...createEmptyOnDemandDiagnostics(),
			navigatorGpu: true,
			selectedHourEngine: 'legacy-debug' as const,
			binComparisonEnabled: true,
			binComparisonValid: true
		};

		const result = buildDebugOnDemandPrototypeDiagnostics({
			existing,
			patch: { renderTransport: 'compute-buffer-selected-hour' },
			defaults: createDefaults({
				selectedHourEngine: 'legacy-debug',
				binComparisonEnabled: true,
				binComparisonValid: true,
				legacySelectedHourDispatchCount: 1,
				legacyScrubScheduleCount: 1
			})
		});

		const contract = getRuntimeContract(result);
		expect(result.binComparisonEnabled).toBe(true);
		expect(result.binComparisonValid).toBe(true);
		expect(contract.selectedHourEngine).toBe('legacy-debug');
		expect(contract.strongVisibleGpuPath).toBe(false);
	});

	it('keeps shared-host diagnostics conservative before visible-readback instrumentation exists', () => {
		const existingReadbackReasons: SelectedHourReadbackReason[] = ['debug'];
		const existing = {
			...createEmptyOnDemandDiagnostics(),
			selectedHourEngine: 'legacy-debug' as const,
			binComparisonEnabled: true,
			binComparisonValid: true,
			error: 'stale prior error',
			tooltipInteraction: createTooltipDiagnostics({
				disabledByQuery: true,
				sampleCount: 99
			}),
			cameraInteraction: createCameraDiagnostics(42),
			legacySelectedHourDispatchCount: 7,
			legacyScrubScheduleCount: 8,
			selectedHourReadbackReasons: existingReadbackReasons,
			selectedHourReadbackReasonCounts: { debug: 1 },
			selectionKey: 'stale|selection',
			sceneSelectionKey: 'stale|selection'
		};

		const result = buildDebugOnDemandPrototypeDiagnostics({
			existing,
			replace: true,
			patch: {
				selectedHourEngine: 'shared-host',
				renderTransport: 'compute-buffer-selected-hour',
				utciSurfaceSource: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true,
				dataTextureBuildCount: 0,
				surfaceRequestId: 4,
				sceneSurfaceRequestId: 4,
				selectionKey: 'analysis|7|12',
				sceneSelectionKey: 'analysis|7|12',
				selectedHourReadbackReasons: ['range'],
				selectedHourReadbackReasonCounts: { range: 1 }
			},
			defaults: createDefaults({
				selectedHourEngine: 'shared-host',
				tooltipInteraction: createTooltipDiagnostics({ sampleCount: 1 }),
				cameraInteraction: createCameraDiagnostics(1)
			})
		});

		expect(getRuntimeContract(result)).toMatchObject({
			selectedHourEngine: 'shared-host',
			readbackInstrumentation: 'not-instrumented',
			strongVisibleGpuPath: false,
			readbackReasons: ['range']
		});
		expect(result.binComparisonEnabled).toBe(false);
		expect(result.binComparisonValid).toBe(false);
		expect(result.error).toBeUndefined();
		expect(result.selectedHourReadbackReasons).toEqual(['range']);
		expect(result.selectedHourReadbackReasonCounts).toEqual({ range: 1 });
		expect(result.selectionKey).toBe('analysis|7|12');
		expect(result.sceneSelectionKey).toBe('analysis|7|12');
		expect(result.tooltipInteraction).toMatchObject({ sampleCount: 1, disabledByQuery: false });
		expect(result.cameraInteraction).toMatchObject({ wheelEventCount: 1 });
		expect(result.legacySelectedHourDispatchCount).toBe(0);
		expect(result.legacyScrubScheduleCount).toBe(0);
	});

	it('does not make a strong claim when instrumentation is missing', () => {
		const result = buildDebugOnDemandPrototypeDiagnostics({
			replace: true,
			patch: {
				selectedHourEngine: 'shared-host',
				renderTransport: 'compute-buffer-selected-hour',
				utciSurfaceSource: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true,
				dataTextureBuildCount: 0,
				visibleSelectedHourReadbackCount: 0,
				surfaceRequestId: 4,
				sceneSurfaceRequestId: 4,
				selectionKey: 'analysis|7|12',
				sceneSelectionKey: 'analysis|7|12'
			},
			defaults: createDefaults({
				selectedHourEngine: 'shared-host'
			})
		});

		expect(getRuntimeContract(result).strongVisibleGpuPath).toBe(false);
	});

	it('does not infer visible readback proof from selectedHourReadbackCount alone', () => {
		const result = buildDebugOnDemandPrototypeDiagnostics({
			replace: true,
			patch: {
				selectedHourEngine: 'shared-host',
				renderTransport: 'compute-buffer-selected-hour',
				utciSurfaceSource: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true,
				dataTextureBuildCount: 0,
				selectedHourReadbackCount: 0,
				surfaceRequestId: 4,
				sceneSurfaceRequestId: 4,
				selectionKey: 'analysis|7|12',
				sceneSelectionKey: 'analysis|7|12'
			},
			defaults: createDefaults({
				selectedHourEngine: 'shared-host',
				readbackInstrumentation: 'instrumented'
			})
		});

		const contract = getRuntimeContract(result);
		expect(result.selectedHourReadbackCount).toBe(0);
		expect(result.visibleSelectedHourReadbackCount).toBeUndefined();
		expect(contract.visibleSelectedHourReadbackCount).toBeUndefined();
		expect(contract.visibleSelectedHourReadbackCountInstrumented).toBe(false);
		expect(contract.visibleRenderPathAvoidsCpuReadback).toBe(false);
		expect(contract.strongVisibleGpuPath).toBe(false);
	});

	it('makes a strong visible GPU claim only with explicit visible readback proof', () => {
		const result = buildDebugOnDemandPrototypeDiagnostics({
			replace: true,
			patch: {
				selectedHourEngine: 'shared-host',
				renderTransport: 'compute-buffer-selected-hour',
				utciSurfaceSource: 'compute-buffer-selected-hour',
				sameDeviceForComputeAndRender: true,
				dataTextureBuildCount: 0,
				visibleSelectedHourReadbackCount: 0,
				surfaceRequestId: 4,
				sceneSurfaceRequestId: 4,
				selectionKey: 'analysis|7|12',
				sceneSelectionKey: 'analysis|7|12'
			},
			defaults: createDefaults({
				selectedHourEngine: 'shared-host',
				readbackInstrumentation: 'instrumented'
			})
		});

		const contract = getRuntimeContract(result);
		expect(contract.visibleSelectedHourReadbackCount).toBe(0);
		expect(contract.visibleRenderPathAvoidsCpuReadback).toBe(true);
		expect(contract.strongVisibleGpuPath).toBe(true);
	});
});
