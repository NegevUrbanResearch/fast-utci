import { describe, expect, it, vi } from 'vitest';
import type { F32MetricOutput } from '$lib/compute/gpu/gpu-pipeline';
import {
	resolveOwnedF32MetricOutputHandle,
	createSelectedHourOutputHandle,
	disposeSelectedHourOutputHandle
} from '$lib/compute/gpu/selectedHourOutputHandle';

describe('selectedHourOutputHandle', () => {
	const createBuffer = () => ({ destroy: vi.fn() } as unknown as GPUBuffer);

	const createOwnedHandle = (overrides: Partial<Parameters<typeof createSelectedHourOutputHandle>[0]> = {}) =>
		createSelectedHourOutputHandle({
			buffer: createBuffer(),
			byteLength: 40,
			ownerId: 'summary-owner',
			source: 'webgpu-on-demand-snapshot',
			metricType: 'shading_index',
			valueLayout: 'one-f32-per-point',
			period: { kind: 'month-index', index: 1, startTimeIndex: 24, timeCount: 24 },
			...overrides
		});

	const createOutput = (
		overrides: Partial<F32MetricOutput & { gpuOutputHandle?: ReturnType<typeof createOwnedHandle> }> = {}
	) => {
		const handle = overrides.gpuOutputHandle ?? createOwnedHandle();
		const output: F32MetricOutput = {
			source: 'webgpu-on-demand-snapshot',
			ownerId: 'summary-owner',
			metricType: 'shading_index',
			valueLayout: 'one-f32-per-point',
			period: { kind: 'month-index', index: 1, startTimeIndex: 24, timeCount: 24 },
			numPoints: 10,
			gpuOutputHandle: handle,
			gpuBuffer: handle.buffer,
			outputBytes: 40,
			debugLabel: 'selected-hour-output:test',
			...overrides
		};
		return { handle, output };
	};

	it('disposes the owned GPU buffer once', () => {
		const destroy = vi.fn();
		const handle = createSelectedHourOutputHandle({
			buffer: { destroy } as unknown as GPUBuffer,
			byteLength: 16,
			requestId: 5,
			timeIndex: 12,
			source: 'webgpu-on-demand-snapshot'
		});

		disposeSelectedHourOutputHandle(handle);
		disposeSelectedHourOutputHandle(handle);

		expect(destroy).toHaveBeenCalledTimes(1);
		expect(handle.disposed).toBe(true);
	});

	it('keeps request identity with the buffer handle', () => {
		const handle = createSelectedHourOutputHandle({
			buffer: { destroy: vi.fn() } as unknown as GPUBuffer,
			byteLength: 32,
			requestId: 17,
			timeIndex: 90,
			source: 'webgpu-on-demand-snapshot'
		});

		expect(handle.requestId).toBe(17);
		expect(handle.timeIndex).toBe(90);
		expect(handle.byteLength).toBe(32);
		expect(handle.source).toBe('webgpu-on-demand-snapshot');
	});

	it('keeps metric ownership metadata with the buffer handle', () => {
		const handle = createSelectedHourOutputHandle({
			buffer: { destroy: vi.fn() } as unknown as GPUBuffer,
			byteLength: 96,
			requestId: 21,
			source: 'webgpu-on-demand-snapshot',
			ownerId: 'selected-hour-output:21',
			metricType: 'shading_index',
			valueLayout: 'one-f32-per-point',
			period: {
				kind: 'month-index',
				index: 4,
				startTimeIndex: 96,
				timeCount: 24
			}
		});

		expect(handle.ownerId).toBe('selected-hour-output:21');
		expect(handle.metricType).toBe('shading_index');
		expect(handle.valueLayout).toBe('one-f32-per-point');
		expect(handle.period).toEqual({
			kind: 'month-index',
			index: 4,
			startTimeIndex: 96,
			timeCount: 24
		});
	});

	it('defaults legacy selected-hour UTCI handles to the stable UTCI shape', () => {
		const handle = createSelectedHourOutputHandle({
			buffer: { destroy: vi.fn() } as unknown as GPUBuffer,
			byteLength: 32,
			requestId: 17,
			timeIndex: 90,
			source: 'webgpu-on-demand-snapshot'
		});

		expect(handle.ownerId).toBe('webgpu-on-demand-snapshot');
		expect(handle.metricType).toBe('utci');
		expect(handle.valueLayout).toBe('one-f32-per-point');
		expect(handle.period).toEqual({ kind: 'time-index', index: 90 });
	});

	it('accepts an owned one-f32-per-point output handle for matching summaries', () => {
		const handle = createOwnedHandle();
		const { output } = createOutput({ gpuOutputHandle: handle, gpuBuffer: handle.buffer });

		const resolved = resolveOwnedF32MetricOutputHandle({
			output,
			metricType: 'shading_index',
			numPoints: 10,
			source: 'webgpu-on-demand-snapshot',
			ownerId: 'summary-owner'
		});

		expect(resolved).toBe(handle);
	});

	it('rejects mismatched ownership metadata before summary reuse', () => {
		const handle = createOwnedHandle({
			metricType: 'utci',
			period: { kind: 'time-index', index: 24 }
		});
		const { output } = createOutput({
			gpuOutputHandle: handle,
			metricType: 'shading_index',
			period: { kind: 'month-index', index: 1, startTimeIndex: 24, timeCount: 24 }
		});

		expect(() =>
			resolveOwnedF32MetricOutputHandle({
				output,
				metricType: 'shading_index',
				numPoints: 10,
				source: 'webgpu-on-demand-snapshot',
				ownerId: 'summary-owner'
			})
		).toThrow(/metric type mismatch/);
	});

	it('rejects output owner mismatches before summary reuse', () => {
		const handle = createOwnedHandle();
		const { output } = createOutput({
			gpuOutputHandle: handle,
			ownerId: 'different-owner'
		});

		expect(() =>
			resolveOwnedF32MetricOutputHandle({
				output,
				metricType: 'shading_index',
				numPoints: 10,
				source: 'webgpu-on-demand-snapshot',
				ownerId: 'summary-owner'
			})
		).toThrow(/owner mismatch between output and handle/);
	});

	it('rejects the remaining summary-handle guard failures', () => {
		const bufferMismatch = createOutput({ gpuBuffer: createBuffer() });
		const disposed = createOutput();
		disposed.handle.dispose();

		const cases: Array<{
			name: string;
			output: F32MetricOutput;
			numPoints?: number;
			expected: RegExp;
		}> = [
			{
				name: 'missing handle',
				output: createOutput({ gpuOutputHandle: undefined }).output,
				expected: /requires a GPU output handle/
			},
			{
				name: 'wrong layout',
				output: createOutput({
					valueLayout: 'point-major-time-series' as unknown as F32MetricOutput['valueLayout']
				}).output,
				expected: /requires one-f32-per-point value layout/
			},
			{
				name: 'period mismatch',
				output: createOutput({
					period: { kind: 'month-index', index: 2, startTimeIndex: 48, timeCount: 24 }
				}).output,
				expected: /period mismatch/
			},
			{
				name: 'numPoints mismatch',
				output: createOutput({ numPoints: 9 }).output,
				numPoints: 10,
				expected: /numPoints mismatch/
			},
			{
				name: 'disposed handle',
				output: disposed.output,
				expected: /cannot use a disposed GPU output handle/
			},
			{
				name: 'handle too small',
				output: createOutput({ gpuOutputHandle: createOwnedHandle({ byteLength: 36 }) }).output,
				expected: /GPU output handle is too small/
			},
			{
				name: 'outputBytes too small',
				output: createOutput({ outputBytes: 36 }).output,
				expected: /outputBytes is too small/
			},
			{
				name: 'raw buffer mismatch',
				output: bufferMismatch.output,
				expected: /raw GPU buffer does not match/
			}
		];

		for (const testCase of cases) {
			expect(() =>
				resolveOwnedF32MetricOutputHandle({
					output: testCase.output,
					metricType: 'shading_index',
					numPoints: testCase.numPoints ?? 10,
					source: 'webgpu-on-demand-snapshot',
					ownerId: 'summary-owner'
				})
			).toThrow(testCase.expected);
		}
	});
});
