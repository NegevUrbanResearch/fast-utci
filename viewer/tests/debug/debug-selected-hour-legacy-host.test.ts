import { describe, expect, it, vi } from 'vitest';
import {
	createDebugSelectedHourLegacyHost,
	type DebugLegacyAcceptedOutput,
	type DebugLegacyDeferredFallback
} from '$lib/debug/debugSelectedHourLegacyHost';

function createAcceptedOutput(id: number): DebugLegacyAcceptedOutput {
	return {
		requestId: id,
		monthIndex: 7,
		timeIndex: 12,
		output: { gpuBuffer: { destroy: vi.fn() } },
		payload: { kind: 'accepted', id }
	};
}

describe('createDebugSelectedHourLegacyHost', () => {
	it('retires the previous accepted GPU buffer on supersession until the scene releases it', () => {
		const host = createDebugSelectedHourLegacyHost();
		const first = createAcceptedOutput(1);
		const second = createAcceptedOutput(2);
		const firstDestroy = first.output.gpuBuffer?.destroy;
		const secondDestroy = second.output.gpuBuffer?.destroy;

		host.setAcceptedOutput(first);
		host.setAcceptedOutput(second);

		expect(firstDestroy).not.toHaveBeenCalled();
		expect(first.output.gpuBuffer).toBeDefined();
		expect(secondDestroy).not.toHaveBeenCalled();

		host.releaseAcceptedOutput({
			requestId: 1,
			monthIndex: 7,
			timeIndex: 12
		});

		expect(firstDestroy).toHaveBeenCalledTimes(1);
		expect(first.output.gpuBuffer).toBeUndefined();
		expect(second.output.gpuBuffer).toBeDefined();
	});

	it('marks the current accepted output releasable and destroys it once it is later cleared', () => {
		const host = createDebugSelectedHourLegacyHost();
		const output = createAcceptedOutput(1);
		const destroy = output.output.gpuBuffer?.destroy;

		host.setAcceptedOutput(output);
		host.releaseAcceptedOutput({
			requestId: 1,
			monthIndex: 7,
			timeIndex: 12
		});
		host.clearAcceptedOutput();
		host.clearAcceptedOutput();

		expect(destroy).toHaveBeenCalledTimes(1);
		expect(output.output.gpuBuffer).toBeUndefined();
	});

	it('keeps the GPU buffer alive when a new wrapper reuses the same output object', () => {
		const host = createDebugSelectedHourLegacyHost<{ kind: string; range?: [number, number] }>();
		const output = { gpuBuffer: { destroy: vi.fn() } };
		const destroy = output.gpuBuffer.destroy;
		const first: DebugLegacyAcceptedOutput<{ kind: string; range?: [number, number] }> = {
			requestId: 1,
			monthIndex: 7,
			timeIndex: 12,
			output,
			payload: { kind: 'accepted', range: [0, 1] }
		};
		const second: DebugLegacyAcceptedOutput<{ kind: string; range?: [number, number] }> = {
			requestId: 1,
			monthIndex: 7,
			timeIndex: 12,
			output,
			payload: { kind: 'accepted', range: [2, 3] }
		};

		host.setAcceptedOutput(first);
		host.setAcceptedOutput(second);

		expect(destroy).not.toHaveBeenCalled();
		expect(host.getAcceptedOutput()).toBe(second);
		expect(second.output.gpuBuffer).toBe(output.gpuBuffer);
	});

	it('does not destroy a retired buffer when a different request is released', () => {
		const host = createDebugSelectedHourLegacyHost();
		const first = createAcceptedOutput(1);
		const second = createAcceptedOutput(2);
		const firstDestroy = first.output.gpuBuffer?.destroy;

		host.setAcceptedOutput(first);
		host.setAcceptedOutput(second);
		host.releaseAcceptedOutput({
			requestId: 99,
			monthIndex: 7,
			timeIndex: 12
		});

		expect(firstDestroy).not.toHaveBeenCalled();
		expect(first.output.gpuBuffer).toBeDefined();
	});

	it('activates only the matching deferred CPU fallback', () => {
		const host = createDebugSelectedHourLegacyHost();
		const fallback: DebugLegacyDeferredFallback = {
			requestId: 3,
			monthIndex: 7,
			timeIndex: 12,
			payload: { kind: 'fallback', id: 3 }
		};

		host.setDeferredCpuFallback(fallback);

		expect(host.takeDeferredCpuFallback({ requestId: 4, monthIndex: 7, timeIndex: 12 })).toBeNull();
		expect(host.takeDeferredCpuFallback({ requestId: 3, monthIndex: 6, timeIndex: 12 })).toBeNull();
		expect(host.takeDeferredCpuFallback({ requestId: 3, monthIndex: 7, timeIndex: 13 })).toBeNull();
		expect(host.takeDeferredCpuFallback({ requestId: 3, monthIndex: 7, timeIndex: 12 })).toBe(fallback);
		expect(host.takeDeferredCpuFallback({ requestId: 3, monthIndex: 7, timeIndex: 12 })).toBeNull();
	});

	it('tracks legacy dispatch and scrub scheduling counters', () => {
		const host = createDebugSelectedHourLegacyHost();

		expect(host.getCounters()).toEqual({
			legacySelectedHourDispatchCount: 0,
			legacyScrubScheduleCount: 0
		});

		host.recordDispatch();
		const runId = host.recordScrubSchedule();

		expect(runId).toBe(1);
		expect(host.getCounters()).toEqual({
			legacySelectedHourDispatchCount: 1,
			legacyScrubScheduleCount: 1
		});
	});

	it('invalidates stale scrub work without incrementing scrub schedule counters', () => {
		const host = createDebugSelectedHourLegacyHost();

		const invalidationRunId = host.invalidateScrubSchedule();

		expect(invalidationRunId).toBe(1);
		expect(host.getScrubScheduleRunId()).toBe(1);
		expect(host.getCounters()).toEqual({
			legacySelectedHourDispatchCount: 0,
			legacyScrubScheduleCount: 0
		});
	});
});
