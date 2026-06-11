import { describe, expect, it, vi } from 'vitest';
import {
	MetricPointReadbackCache,
	clearMetricPointReadbackCache,
	readF32MetricPointValue
} from '$lib/compute/gpu/metricPointReadback';

const TEST_GPU_BUFFER_USAGE = {
	MAP_READ: 1,
	COPY_DST: 8
};

describe('metricPointReadback', () => {
	it('reads one f32 metric value from the requested point offset', async () => {
		const copyBufferToBuffer = vi.fn();
		const submit = vi.fn();
		const onSubmittedWorkDone = vi.fn(async () => undefined);
		const readbackBuffer = createMappedReadbackBuffer(0.625);
		const device = {
			createBuffer: vi.fn(() => readbackBuffer),
			createCommandEncoder: vi.fn(() => ({
				copyBufferToBuffer,
				finish: () => 'commands' as unknown as GPUCommandBuffer
			})),
			queue: { submit, onSubmittedWorkDone }
		} as unknown as GPUDevice;
		const sourceBuffer = { size: 16 } as GPUBuffer;

		const value = await readF32MetricPointValue({
			device,
			sourceBuffer,
			pointIndex: 2,
			numPoints: 4
		});

		expect(value).toBeCloseTo(0.625);
		expect(device.createBuffer).toHaveBeenCalledWith({
			size: 4,
			usage: TEST_GPU_BUFFER_USAGE.COPY_DST | TEST_GPU_BUFFER_USAGE.MAP_READ
		});
		expect(copyBufferToBuffer).toHaveBeenCalledWith(sourceBuffer, 8, readbackBuffer, 0, 4);
		expect(submit).toHaveBeenCalledWith(['commands']);
		expect(onSubmittedWorkDone).toHaveBeenCalled();
		expect(readbackBuffer.unmap).toHaveBeenCalled();
		expect(readbackBuffer.destroy).toHaveBeenCalled();
	});

	it('caches repeated reads for the same metric/month/point/request key', async () => {
		const cache = new MetricPointReadbackCache();
		const read = vi.fn(async () => 0.75);
		const key = {
			metricType: 'shading_index' as const,
			monthIndex: 7,
			positionIndex: 12,
			requestId: 4,
			ownerId: 'shading-output'
		};

		await expect(cache.getOrRead(key, read)).resolves.toBe(0.75);
		await expect(cache.getOrRead(key, read)).resolves.toBe(0.75);

		expect(read).toHaveBeenCalledTimes(1);
	});

	it('does not cache failed reads and retries until one succeeds', async () => {
		const cache = new MetricPointReadbackCache();
		const key = {
			metricType: 'shading_index' as const,
			monthIndex: 7,
			positionIndex: 12,
			requestId: 4,
			ownerId: 'shading-output'
		};
		const transientError = new Error('transient gpu readback failure');
		const read = vi
			.fn<() => Promise<number>>()
			.mockRejectedValueOnce(transientError)
			.mockResolvedValueOnce(0.5);

		await expect(cache.getOrRead(key, read)).rejects.toThrow(transientError);
		await expect(cache.getOrRead(key, read)).resolves.toBe(0.5);
		await expect(cache.getOrRead(key, read)).resolves.toBe(0.5);

		expect(read).toHaveBeenCalledTimes(2);
	});

	it('evicts the least-recently-used hover read when capacity is exceeded', async () => {
		const cache = new MetricPointReadbackCache(2);
		const reads = new Map<string, ReturnType<typeof vi.fn>>();
		const readFor = (value: number) => {
			const read = vi.fn(async () => value);
			reads.set(String(value), read);
			return read;
		};
		const key = (positionIndex: number) => ({
			metricType: 'shading_index' as const,
			monthIndex: 7,
			positionIndex,
			requestId: 4,
			ownerId: 'shading-output'
		});

		await expect(cache.getOrRead(key(1), readFor(1))).resolves.toBe(1);
		await expect(cache.getOrRead(key(2), readFor(2))).resolves.toBe(2);
		await expect(cache.getOrRead(key(1), readFor(10))).resolves.toBe(1);
		await expect(cache.getOrRead(key(3), readFor(3))).resolves.toBe(3);
		await expect(cache.getOrRead(key(2), readFor(20))).resolves.toBe(20);

		expect(reads.get('1')).toHaveBeenCalledTimes(1);
		expect(reads.get('2')).toHaveBeenCalledTimes(1);
		expect(reads.get('10')).not.toHaveBeenCalled();
		expect(reads.get('3')).toHaveBeenCalledTimes(1);
		expect(reads.get('20')).toHaveBeenCalledTimes(1);
		expect(cache.size).toBe(2);
	});

	it('clears the shared hover readback cache for teardown', async () => {
		const read = vi.fn(async () => 0.25);
		const key = {
			metricType: 'shading_index' as const,
			monthIndex: 7,
			positionIndex: 1,
			requestId: 4,
			ownerId: 'shading-output'
		};

		expect(() => clearMetricPointReadbackCache()).not.toThrow();
		const cache = new MetricPointReadbackCache();
		await expect(cache.getOrRead(key, read)).resolves.toBe(0.25);
		cache.clear();
		await expect(cache.getOrRead(key, read)).resolves.toBe(0.25);

		expect(read).toHaveBeenCalledTimes(2);
	});
});

function createMappedReadbackBuffer(value: number) {
	const bytes = new ArrayBuffer(4);
	new DataView(bytes).setFloat32(0, value, true);
	return {
		size: 4,
		mapAsync: vi.fn(async () => undefined),
		getMappedRange: vi.fn(() => bytes),
		unmap: vi.fn(),
		destroy: vi.fn()
	} as unknown as GPUBuffer & {
		unmap: ReturnType<typeof vi.fn>;
		destroy: ReturnType<typeof vi.fn>;
	};
}
