import type { F32MetricType } from '$lib/compute/on-demand/onDemandOutputFormat';

export type MetricPointReadbackKey = {
	metricType: F32MetricType;
	monthIndex: number;
	positionIndex: number;
	requestId?: number;
	ownerId?: string;
};

export type ReadF32MetricPointValueParams = {
	device: GPUDevice;
	sourceBuffer: GPUBuffer & { size?: number };
	pointIndex: number;
	numPoints: number;
};

const F32_BYTES = 4;
const DEFAULT_METRIC_POINT_READBACK_CACHE_MAX_ENTRIES = 128;
const FALLBACK_GPU_BUFFER_USAGE = {
	MAP_READ: 1,
	COPY_DST: 8
};
const FALLBACK_GPU_MAP_MODE = {
	READ: 1
};

function getGpuBufferUsage(): { MAP_READ: number; COPY_DST: number } {
	return (globalThis as { GPUBufferUsage?: { MAP_READ: number; COPY_DST: number } })
		.GPUBufferUsage ?? FALLBACK_GPU_BUFFER_USAGE;
}

function getGpuMapMode(): { READ: number } {
	return (globalThis as { GPUMapMode?: { READ: number } }).GPUMapMode ??
		FALLBACK_GPU_MAP_MODE;
}

export async function readF32MetricPointValue(
	params: ReadF32MetricPointValueParams
): Promise<number> {
	const { device, sourceBuffer, pointIndex, numPoints } = params;
	if (!Number.isInteger(pointIndex) || pointIndex < 0 || pointIndex >= numPoints) {
		throw new Error(`Metric point readback index ${pointIndex} is outside 0..${numPoints - 1}.`);
	}

	const sourceOffset = pointIndex * F32_BYTES;
	if (sourceBuffer.size !== undefined && sourceBuffer.size < sourceOffset + F32_BYTES) {
		throw new Error('Metric point readback source buffer is smaller than the requested point.');
	}

	const usage = getGpuBufferUsage();
	const readbackBuffer = device.createBuffer({
		size: F32_BYTES,
		usage: usage.COPY_DST | usage.MAP_READ
	});
	try {
		const encoder = device.createCommandEncoder();
		encoder.copyBufferToBuffer(sourceBuffer, sourceOffset, readbackBuffer, 0, F32_BYTES);
		device.queue.submit([encoder.finish()]);
		await device.queue.onSubmittedWorkDone();
		await readbackBuffer.mapAsync(getGpuMapMode().READ);
		const mappedRange = readbackBuffer.getMappedRange(0, F32_BYTES);
		const value = new DataView(mappedRange).getFloat32(0, true);
		readbackBuffer.unmap();
		return value;
	} finally {
		readbackBuffer.destroy();
	}
}

export class MetricPointReadbackCache {
	private readonly cache = new Map<string, Promise<number>>();

	constructor(
		private readonly maxEntries = DEFAULT_METRIC_POINT_READBACK_CACHE_MAX_ENTRIES
	) {}

	get size(): number {
		return this.cache.size;
	}

	getOrRead(
		key: MetricPointReadbackKey,
		read: () => Promise<number>
	): Promise<number> {
		return this.getOrReadWithStats(key, read).then((result) => result.value);
	}

	getOrReadWithStats(
		key: MetricPointReadbackKey,
		read: () => Promise<number>
	): Promise<{ value: number; cacheHit: boolean }> {
		const cacheKey = serializeMetricPointReadbackKey(key);
		const cached = this.cache.get(cacheKey);
		if (cached) {
			this.cache.delete(cacheKey);
			this.cache.set(cacheKey, cached);
			return cached.then((value) => ({ value, cacheHit: true }));
		}

		const pending = read().catch((error) => {
			this.cache.delete(cacheKey);
			throw error;
		});
		this.cache.set(cacheKey, pending);
		this.evictOldestEntries();
		return pending.then((value) => ({ value, cacheHit: false }));
	}

	clear(): void {
		this.cache.clear();
	}

	private evictOldestEntries(): void {
		const capacity = Math.max(1, this.maxEntries);
		while (this.cache.size > capacity) {
			const oldestKey = this.cache.keys().next().value as string | undefined;
			if (oldestKey === undefined) return;
			this.cache.delete(oldestKey);
		}
	}
}

export const sharedMetricPointReadbackCache = new MetricPointReadbackCache();

export function clearMetricPointReadbackCache(): void {
	sharedMetricPointReadbackCache.clear();
}

function serializeMetricPointReadbackKey(key: MetricPointReadbackKey): string {
	return [
		key.metricType,
		key.monthIndex,
		key.positionIndex,
		key.requestId ?? 'none',
		key.ownerId ?? 'none'
	].join(':');
}
