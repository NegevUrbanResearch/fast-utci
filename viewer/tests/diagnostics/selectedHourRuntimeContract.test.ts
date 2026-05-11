import { describe, expect, it } from 'vitest';
import {
	buildSelectedHourRuntimeContract,
	type SelectedHourReadbackReason
} from '$lib/diagnostics/selectedHourRuntimeContract';

describe('selectedHourRuntimeContract', () => {
	it('classifies a strong compute-buffer visible path', () => {
		const contract = buildSelectedHourRuntimeContract({
			route: 'main',
			selectedHourEngine: 'shared-host',
			renderTransport: 'compute-buffer-selected-hour',
			utciSurfaceSource: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: true,
			dataTextureBuildCount: 0,
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented',
			legacySelectedHourDispatchCount: 0,
			legacyScrubScheduleCount: 0,
			requestId: 42,
			sceneRequestId: 42,
			selectionKey: 'Ben-Gurion|7|12',
			sceneSelectionKey: 'Ben-Gurion|7|12',
			readbackReasons: []
		});

		expect(contract.strongVisibleGpuPath).toBe(true);
		expect(contract.visibleRenderPathAvoidsCpuReadback).toBe(true);
		expect(contract.hasLegacyDebugOverlap).toBe(false);
		expect(contract.acceptedRequestId).toBe(42);
	});

	it('keeps non-visible CPU readbacks separate from visible transport proof', () => {
		const reasons: SelectedHourReadbackReason[] = ['range', 'tooltip'];
		const contract = buildSelectedHourRuntimeContract({
			route: 'debug',
			selectedHourEngine: 'shared-host',
			renderTransport: 'compute-buffer-selected-hour',
			utciSurfaceSource: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: true,
			dataTextureBuildCount: 0,
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented',
			legacySelectedHourDispatchCount: 0,
			legacyScrubScheduleCount: 0,
			requestId: 7,
			sceneRequestId: 7,
			selectionKey: 'Ben-Gurion|7|12',
			sceneSelectionKey: 'Ben-Gurion|7|12',
			readbackReasons: reasons,
			readbackReasonCounts: {
				range: 1,
				tooltip: 1
			}
		});

		expect(contract.strongVisibleGpuPath).toBe(true);
		expect(contract.visibleRenderPathAvoidsCpuReadback).toBe(true);
		expect(contract.readbackReasons).toEqual(['range', 'tooltip']);
		expect(contract.readbackReasonCounts).toEqual({ range: 1, tooltip: 1 });
		expect(contract.totalSelectedHourReadbackReasonCount).toBe(2);
	});

	it('does not allow strong GPU proof when readback instrumentation is missing', () => {
		const contract = buildSelectedHourRuntimeContract({
			route: 'main',
			selectedHourEngine: 'shared-host',
			renderTransport: 'compute-buffer-selected-hour',
			utciSurfaceSource: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: true,
			dataTextureBuildCount: 0,
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'not-instrumented',
			legacySelectedHourDispatchCount: 0,
			legacyScrubScheduleCount: 0,
			requestId: 8,
			selectionKey: 'Ben-Gurion|7|12',
			sceneSelectionKey: 'Ben-Gurion|7|12',
			readbackReasons: []
		});

		expect(contract.visibleRenderPathAvoidsCpuReadback).toBe(false);
		expect(contract.strongVisibleGpuPath).toBe(false);
	});

	it('does not allow strong GPU proof when instrumented callers omit proof counts', () => {
		const contract = buildSelectedHourRuntimeContract({
			route: 'main',
			selectedHourEngine: 'shared-host',
			renderTransport: 'compute-buffer-selected-hour',
			utciSurfaceSource: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: true,
			readbackInstrumentation: 'instrumented',
			legacySelectedHourDispatchCount: 0,
			legacyScrubScheduleCount: 0,
			requestId: 8,
			sceneRequestId: 8,
			selectionKey: 'Ben-Gurion|7|12',
			sceneSelectionKey: 'Ben-Gurion|7|12',
			readbackReasons: []
		});

		expect(contract.visibleRenderPathAvoidsCpuReadback).toBe(false);
		expect(contract.strongVisibleGpuPath).toBe(false);
	});

	it('does not allow strong GPU proof when scene request id is stale', () => {
		const contract = buildSelectedHourRuntimeContract({
			route: 'main',
			selectedHourEngine: 'shared-host',
			renderTransport: 'compute-buffer-selected-hour',
			utciSurfaceSource: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: true,
			dataTextureBuildCount: 0,
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented',
			legacySelectedHourDispatchCount: 0,
			legacyScrubScheduleCount: 0,
			requestId: 8,
			sceneRequestId: 7,
			selectionKey: 'Ben-Gurion|7|12',
			sceneSelectionKey: 'Ben-Gurion|7|12',
			readbackReasons: []
		});

		expect(contract.selectionMatchesScene).toBe(true);
		expect(contract.requestMatchesScene).toBe(false);
		expect(contract.strongVisibleGpuPath).toBe(false);
	});

	it('does not allow strong GPU proof when scene request id is missing', () => {
		const contract = buildSelectedHourRuntimeContract({
			route: 'main',
			selectedHourEngine: 'shared-host',
			renderTransport: 'compute-buffer-selected-hour',
			utciSurfaceSource: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: true,
			dataTextureBuildCount: 0,
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented',
			legacySelectedHourDispatchCount: 0,
			legacyScrubScheduleCount: 0,
			requestId: 8,
			selectionKey: 'Ben-Gurion|7|12',
			sceneSelectionKey: 'Ben-Gurion|7|12',
			readbackReasons: []
		});

		expect(contract.requestMatchesScene).toBe(false);
		expect(contract.strongVisibleGpuPath).toBe(false);
	});

	it('flags legacy debug overlap when shared-host claims coexist with legacy counters', () => {
		const contract = buildSelectedHourRuntimeContract({
			route: 'debug',
			selectedHourEngine: 'shared-host',
			renderTransport: 'compute-buffer-selected-hour',
			utciSurfaceSource: 'compute-buffer-selected-hour',
			sameDeviceForComputeAndRender: true,
			dataTextureBuildCount: 0,
			visibleSelectedHourReadbackCount: 0,
			readbackInstrumentation: 'instrumented',
			legacySelectedHourDispatchCount: 1,
			legacyScrubScheduleCount: 0,
			requestId: 9,
			selectionKey: 'Ben-Gurion|7|12',
			sceneSelectionKey: 'Ben-Gurion|7|12',
			readbackReasons: []
		});

		expect(contract.strongVisibleGpuPath).toBe(false);
		expect(contract.hasLegacyDebugOverlap).toBe(true);
	});
});
