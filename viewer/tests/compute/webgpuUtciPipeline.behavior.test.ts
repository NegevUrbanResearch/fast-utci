import { beforeAll, describe, expect, it, vi } from 'vitest';
import {
	__TEST_ONLY_WebgpuUtciComputePipeline,
	createWebgpuUtciPipeline
} from '$lib/compute/gpu/webgpuUtciPipeline';

function createFakeBuffer(size: number) {
	return {
		size,
		destroy: vi.fn()
	};
}

function createFakeDevice() {
	return {
		limits: {
			maxStorageBuffersPerShaderStage: 8
		},
		queue: {
			writeBuffer: vi.fn(),
			submit: vi.fn(),
			onSubmittedWorkDone: vi.fn().mockResolvedValue(undefined)
		},
		createBuffer: vi.fn(({ size }: { size: number }) => createFakeBuffer(size))
	};
}

function createFakeDeviceWithStorageLimit(maxStorageBuffersPerShaderStage: number) {
	return {
		...createFakeDevice(),
		limits: {
			maxStorageBuffersPerShaderStage
		}
	};
}

function baseUploadParams() {
	return {
		gridPoints: new Float32Array([0, 0, 0]),
		sunVectors: new Float32Array([1, 0, 0]),
		weather: new Float32Array([1, 2, 3, 4, 5, 6, 7])
	};
}

describe('WebgpuUtciComputePipeline behavioral guards', () => {
	beforeAll(() => {
		Object.assign(globalThis, {
			GPUBufferUsage: {
				STORAGE: 1,
				COPY_DST: 2,
				COPY_SRC: 4,
				UNIFORM: 8,
				MAP_READ: 16
			}
		});
	});

	it('snapshots weather on upload so later caller mutation does not affect samples', async () => {
		const device = createFakeDevice();
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false);
		const weather = new Float32Array([1, 2, 3, 4, 5, 6, 7]);

		await pipeline.uploadStaticData({
			...baseUploadParams(),
			weather
		});
		weather[0] = 99;
		weather[4] = 123;

		expect(pipeline.getWeatherSample(1)).toEqual([
			{
				air_temp: 1,
				direct_normal: 5,
				diffuse_horizontal: 6,
				horiz_infrared: 7,
				wind_speed: 3,
				rel_humidity: 4
			}
		]);
	});

	it('can wrap a provided renderer-owned GPUDevice instead of requesting a standalone device', async () => {
		const device = createFakeDeviceWithStorageLimit(8);
		const pipeline = await createWebgpuUtciPipeline({ device: device as unknown as GPUDevice });

		expect(pipeline.getDeviceForDebug?.()).toBe(device);
		expect(pipeline.supportsMrtComponentDiagnostics()).toBe(false);
	});

	it('enables MRT component diagnostics for a provided device only when its limits support them', async () => {
		const limitedDevice = createFakeDeviceWithStorageLimit(8);
		const capableDevice = createFakeDeviceWithStorageLimit(10);

		const limitedPipeline = await createWebgpuUtciPipeline({
			device: limitedDevice as unknown as GPUDevice,
			enableDiagnostics: true
		});
		const capablePipeline = await createWebgpuUtciPipeline({
			device: capableDevice as unknown as GPUDevice,
			enableDiagnostics: true
		});

		expect(limitedPipeline.supportsMrtComponentDiagnostics()).toBe(false);
		expect(capablePipeline.supportsMrtComponentDiagnostics()).toBe(true);
	});

	it('clears stale BVH and optional upload buffers when a later upload omits them', async () => {
		const device = createFakeDevice();
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false);
		const serializedBvh = {
			bvhNodeBuffer: new ArrayBuffer(32),
			bvhIndexBuffer: new ArrayBuffer(4),
			vertexBuffer: new Float32Array([0, 0, 0]),
			indexBuffer: new Uint32Array([0])
		};

		await pipeline.uploadStaticData({
			...baseUploadParams(),
			sunAltitudes: new Float32Array([0.5]),
			domeVectors: new Float32Array([0, 1, 0]),
			domeWeights: new Float32Array([1]),
			serializedBvh
		});
		await pipeline.uploadStaticData(baseUploadParams());

		expect((pipeline as any).bvhNodeBuffer).toBeNull();
		expect((pipeline as any).bvhIndexBuffer).toBeNull();
		expect((pipeline as any).bvhVertexBuffer).toBeNull();
		expect((pipeline as any).bvhParamsBuffer).toBeNull();
		expect((pipeline as any).sunAltitudesBuffer).toBeNull();
		expect((pipeline as any).domeVectorsBuffer).toBeNull();
		expect((pipeline as any).domeWeightsBuffer).toBeNull();
	});

	it('rejects UTCI readback dimensions that do not match the producing run config', async () => {
		const device = createFakeDevice();
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false) as any;

		pipeline.utciBuffer = createFakeBuffer(16);
		pipeline.lastConfig = { numPoints: 1, numHours: 24, numMonths: 1 };

		await expect(
			pipeline.readUtcisSlice({
				monthIndex: 0,
				hourIndex: 0,
				numPoints: 2,
				numHours: 24,
				numMonths: 1
			})
		).rejects.toThrow(/readUtcisSlice request does not match the last run config/i);

		await expect(
			pipeline.readUtciBulk({
				numPoints: 1,
				numHours: 12,
				numMonths: 1
			})
		).rejects.toThrow(/readUtciBulk request does not match the last run config/i);
	});
});
