import { beforeAll, describe, expect, it, vi } from 'vitest';
import {
	__TEST_ONLY_WebgpuUtciComputePipeline,
	createWebgpuUtciPipeline
} from '$lib/compute/gpu/webgpuUtciPipeline';
import { DEFAULT_EXPOSURE_MAX_WORKGROUPS_PER_SLICE } from '$lib/compute/gpu/exposureScheduling';

function createFakeBuffer(size: number) {
	return {
		size,
		destroy: vi.fn()
	};
}

function createFakePipeline() {
	return {
		getBindGroupLayout: vi.fn(() => ({}))
	};
}

function createFakeComputePass() {
	return {
		setPipeline: vi.fn(),
		setBindGroup: vi.fn(),
		dispatchWorkgroups: vi.fn(),
		end: vi.fn()
	};
}

function createFakeDevice(options?: {
	onSubmittedWorkDone?: () => Promise<void>;
	throwOnCreateBindGroupCall?: number;
}) {
	const computePasses: ReturnType<typeof createFakeComputePass>[] = [];
	const buffers: ReturnType<typeof createFakeBuffer>[] = [];
	const createBindGroup = vi.fn(() => {
		if (
			options?.throwOnCreateBindGroupCall !== undefined &&
			createBindGroup.mock.calls.length === options.throwOnCreateBindGroupCall
		) {
			throw new Error(`createBindGroup failed on call ${options.throwOnCreateBindGroupCall}`);
		}
		return {};
	});
	return {
		limits: {
			maxStorageBuffersPerShaderStage: 8
		},
		__computePasses: computePasses,
		__buffers: buffers,
		queue: {
			writeBuffer: vi.fn(),
			submit: vi.fn(),
			onSubmittedWorkDone: vi.fn(options?.onSubmittedWorkDone ?? (() => Promise.resolve()))
		},
		createBuffer: vi.fn(({ size }: { size: number }) => {
			const buffer = createFakeBuffer(size);
			buffers.push(buffer);
			return buffer;
		}),
		createShaderModule: vi.fn(() => ({})),
		createComputePipelineAsync: vi.fn(() => Promise.resolve(createFakePipeline())),
		createBindGroup,
		createCommandEncoder: vi.fn(() => ({
			beginComputePass: vi.fn(() => {
				const pass = createFakeComputePass();
				computePasses.push(pass);
				return pass;
			}),
			copyBufferToBuffer: vi.fn(),
			finish: vi.fn(() => ({}))
		}))
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

function exposureUploadParams(numPoints: number) {
	const gridPoints = new Float32Array(numPoints * 3);
	const sunVectors = new Float32Array([1, 0, 0]);
	const domeVectors = new Float32Array(145 * 3);
	const domeWeights = new Float32Array(145);
	return {
		gridPoints,
		sunVectors,
		sunAltitudes: new Float32Array([0.5]),
		weather: new Float32Array([1, 2, 3, 4, 5, 6, 7]),
		domeVectors,
		domeWeights,
		serializedBvh: {
			bvhNodeBuffer: new ArrayBuffer(32),
			bvhIndexBuffer: new ArrayBuffer(4),
			vertexBuffer: new Float32Array([0, 0, 0]),
			indexBuffer: new Uint32Array([0])
		}
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
		expect(pipeline.supportsMrtComponentDiagnostics).toBeDefined();
		expect(pipeline.supportsMrtComponentDiagnostics?.call(pipeline)).toBe(false);
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

		expect(limitedPipeline.supportsMrtComponentDiagnostics).toBeDefined();
		expect(capablePipeline.supportsMrtComponentDiagnostics).toBeDefined();
		expect(limitedPipeline.supportsMrtComponentDiagnostics?.call(limitedPipeline)).toBe(false);
		expect(capablePipeline.supportsMrtComponentDiagnostics?.call(capablePipeline)).toBe(true);
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

	it('uses single-submit exposure scheduling by default', async () => {
		const device = createFakeDevice();
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false);

		await pipeline.uploadStaticData(exposureUploadParams(1024));
		await pipeline.runExposurePrecompute({
			numPoints: 1024,
			numHours: 1,
			numMonths: 1
		});

		const timings = pipeline.getOnDemandDiagnostics().timings;
		expect(device.queue.submit).toHaveBeenCalledTimes(1);
		expect(device.queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);
		expect(timings.exposureSchedulerMode).toBe('single-submit');
		expect(timings.exposureSchedulerSliceCount).toBe(1);
		expect(timings.exposureSchedulerSubmitCount).toBe(1);
		expect(timings.exposureSchedulerYieldCount).toBe(0);
		expect(timings.exposurePointDispatchChunkCount).toBe(1);
		expect(timings.exposureSchedulerMaxWorkgroupsPerSlice).toBe(
			DEFAULT_EXPOSURE_MAX_WORKGROUPS_PER_SLICE
		);
		expect(timings.exposureSchedulerQueueWaitTotalMs).toEqual(expect.any(Number));
		expect(timings.exposureSchedulerQueueWaitMaxMs).toEqual(expect.any(Number));
		expect(timings.exposureSchedulerQueueWaitMinMs).toEqual(expect.any(Number));
		expect(timings.exposurePrecomputeMs).toBeGreaterThanOrEqual(
			timings.exposureCommandEncodeTotalMs ?? Number.POSITIVE_INFINITY
		);
	});

	it('aborts default single-submit exposure during queue wait and blocks later UTCI dispatch', async () => {
		const controller = new AbortController();
		const device = createFakeDevice({
			onSubmittedWorkDone: async () => {
				controller.abort();
			}
		});
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false);

		await pipeline.uploadStaticData(exposureUploadParams(1024));
		await expect(
			pipeline.runExposurePrecompute({
				numPoints: 1024,
				numHours: 1,
				numMonths: 1,
				signal: controller.signal
			})
		).rejects.toMatchObject({ name: 'AbortError' });

		await expect(
			pipeline.runUtciForTimeIndex({
				format: 'f32-utci',
				numPoints: 1024,
				numHours: 1,
				numMonths: 1,
				timeIndex: 0
			})
		).rejects.toThrow(/solar\/sky exposure passes did not run/i);
	});

	it('rejects already-aborted exposure precompute before GPU setup work', async () => {
		const controller = new AbortController();
		controller.abort();
		const device = createFakeDevice();
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false);

		await pipeline.uploadStaticData(exposureUploadParams(1024));
		device.createComputePipelineAsync.mockClear();
		device.createCommandEncoder.mockClear();

		await expect(
			pipeline.runExposurePrecompute({
				numPoints: 1024,
				numHours: 1,
				numMonths: 1,
				signal: controller.signal
			})
		).rejects.toMatchObject({ name: 'AbortError' });

		expect(device.createComputePipelineAsync).not.toHaveBeenCalled();
		expect(device.createCommandEncoder).not.toHaveBeenCalled();
		expect(device.queue.submit).not.toHaveBeenCalled();
	});

	it('rejects exposure precompute after dispose with an abort error', async () => {
		const device = createFakeDevice();
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false);

		await pipeline.uploadStaticData(exposureUploadParams(1024));
		pipeline.dispose();

		await expect(
			pipeline.runExposurePrecompute({
				numPoints: 1024,
				numHours: 1,
				numMonths: 1
			})
		).rejects.toMatchObject({
			name: 'AbortError',
			message: expect.stringContaining('disposed')
		});
		expect(device.createComputePipelineAsync).not.toHaveBeenCalled();
		expect(device.queue.submit).not.toHaveBeenCalled();
	});

	it('destroys single-submit exposure transient uniform buffers when queue wait rejects', async () => {
		const queueError = new Error('queue wait failed');
		const device = createFakeDevice({
			onSubmittedWorkDone: async () => {
				throw queueError;
			}
		});
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false) as any;
		pipeline.paramsBuffer = createFakeBuffer(16);

		await pipeline.uploadStaticData(exposureUploadParams(1024));
		const firstExposureBufferIndex = device.__buffers.length;

		await expect(
			pipeline.runExposurePrecompute({
				numPoints: 1024,
				numHours: 1,
				numMonths: 1
			})
		).rejects.toBe(queueError);

		const transientUniformBuffers = device.__buffers
			.slice(firstExposureBufferIndex)
			.filter((buffer) => buffer.size === 16);
		expect(transientUniformBuffers).toHaveLength(2);
		for (const buffer of transientUniformBuffers) {
			expect(buffer.destroy).toHaveBeenCalledTimes(1);
		}

		await expect(
			pipeline.runUtciForTimeIndex({
				format: 'f32-utci',
				numPoints: 1024,
				numHours: 1,
				numMonths: 1,
				timeIndex: 0
			})
		).rejects.toThrow(/solar\/sky exposure passes did not run/i);
	});

	it('uses one queue submit and wait per chunked exposure scheduler slice', async () => {
		const device = createFakeDevice();
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false);

		await pipeline.uploadStaticData(exposureUploadParams(1024));
		await pipeline.runExposurePrecompute({
			numPoints: 1024,
			numHours: 1,
			numMonths: 1,
			exposureScheduling: {
				mode: 'chunked',
				maxWorkgroupsPerSlice: 4,
				yieldBetweenSlices: false
			}
		});

		const timings = pipeline.getOnDemandDiagnostics().timings;
		expect(device.queue.submit).toHaveBeenCalledTimes(4);
		expect(device.queue.onSubmittedWorkDone).toHaveBeenCalledTimes(4);
		expect(timings.exposureSchedulerMode).toBe('chunked');
		expect(timings.exposureSchedulerSliceCount).toBe(4);
		expect(timings.exposureSchedulerSubmitCount).toBe(4);
		expect(timings.exposureSchedulerYieldCount).toBe(0);
		expect(timings.exposureSchedulerMaxWorkgroupsPerSlice).toBe(4);
		expect(timings.exposurePointDispatchChunkCount).toBe(4);
		expect(timings.exposureSchedulerQueueWaitTotalMs).toEqual(expect.any(Number));
		expect(timings.exposureSchedulerQueueWaitMaxMs).toEqual(expect.any(Number));
		expect(timings.exposureSchedulerQueueWaitMinMs).toEqual(expect.any(Number));
		expect(timings.exposureQueueWaitMs).toEqual(expect.any(Number));
		expect(timings.exposurePointChunks).toBe(4);
		expect(timings.exposureSolarRayBudget).toBe(1024);
		expect(timings.exposureSkyRayBudget).toBe(1024 * 145);
		expect(timings.exposurePrecomputeMs).toEqual(expect.any(Number));
		expect(timings.exposurePrecomputeMs).toBeGreaterThanOrEqual(
			timings.exposureCommandEncodeTotalMs ?? Number.POSITIVE_INFINITY
		);
	});

	it('clears stale selected-hour diagnostics when exposure precompute reruns', async () => {
		const device = createFakeDevice();
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false);

		await pipeline.uploadStaticData(exposureUploadParams(1024));
		await pipeline.runExposurePrecompute({
			numPoints: 1024,
			numHours: 1,
			numMonths: 1
		});
		await pipeline.runUtciForTimeIndex({
			format: 'f32-utci',
			numPoints: 1024,
			numHours: 1,
			numMonths: 1,
			timeIndex: 0
		});

		const afterSelectedHour = pipeline.getOnDemandDiagnostics();
		expect(afterSelectedHour.timeIndices).toEqual([0]);
		expect(afterSelectedHour.oneHourOutputBytes).toBeGreaterThan(0);
		expect(afterSelectedHour.trackedGpuAllocationBytes.selectedHourOutputBytes).toBeGreaterThan(0);
		expect(afterSelectedHour.timings.oneHourDispatchMs).toEqual(expect.any(Number));

		await pipeline.runExposurePrecompute({
			numPoints: 1024,
			numHours: 1,
			numMonths: 1
		});

		const afterRerun = pipeline.getOnDemandDiagnostics();
		expect(afterRerun.timeIndices).toEqual([]);
		expect(afterRerun.oneHourOutputBytes).toBe(0);
		expect(afterRerun.trackedGpuAllocationBytes.selectedHourOutputBytes).toBe(0);
		expect(
			afterRerun.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark
		).toBeGreaterThan(0);
		expect(afterRerun.timings.oneHourDispatchMs).toBeUndefined();
		expect(afterRerun.timings.selectedHourReadbackMs).toBeUndefined();
		expect(afterRerun.timings.renderUpdateMs).toBeUndefined();
	});

	it('destroys helper-owned exposure transient buffers when encoding throws', async () => {
		const device = createFakeDevice({ throwOnCreateBindGroupCall: 2 });
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false) as any;
		pipeline.paramsBuffer = createFakeBuffer(16);

		await pipeline.uploadStaticData(exposureUploadParams(1024));
		const firstExposureBufferIndex = device.__buffers.length;

		await expect(
			pipeline.runExposurePrecompute({
				numPoints: 1024,
				numHours: 1,
				numMonths: 1
			})
		).rejects.toThrow(/createBindGroup failed on call 2/);

		const transientUniformBuffers = device.__buffers
			.slice(firstExposureBufferIndex)
			.filter((buffer) => buffer.size === 16);
		expect(transientUniformBuffers).toHaveLength(1);
		expect(transientUniformBuffers[0]?.destroy).toHaveBeenCalledTimes(1);
	});

	it('destroys chunked exposure transient uniform buffers after each slice wait', async () => {
		const secondQueueWaitError = new Error('second queue wait failed');
		let device!: ReturnType<typeof createFakeDevice>;
		let firstExposureBufferIndex = 0;
		device = createFakeDevice({
			onSubmittedWorkDone: async () => {
				const waitCount = device.queue.onSubmittedWorkDone.mock.calls.length;
				if (waitCount === 2) {
					const firstSliceUniformBuffers = device.__buffers
						.slice(firstExposureBufferIndex)
						.filter((buffer) => buffer.size === 16)
						.slice(0, 2);
					expect(firstSliceUniformBuffers).toHaveLength(2);
					for (const buffer of firstSliceUniformBuffers) {
						expect(buffer.destroy).toHaveBeenCalledTimes(1);
					}
					throw secondQueueWaitError;
				}
			}
		});
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false) as any;
		pipeline.paramsBuffer = createFakeBuffer(16);

		await pipeline.uploadStaticData(exposureUploadParams(1024));
		firstExposureBufferIndex = device.__buffers.length;

		await expect(
			pipeline.runExposurePrecompute({
				numPoints: 1024,
				numHours: 1,
				numMonths: 1,
				exposureScheduling: {
					mode: 'chunked',
					maxWorkgroupsPerSlice: 4,
					yieldBetweenSlices: false
				}
			})
		).rejects.toBe(secondQueueWaitError);

		expect(device.queue.submit).toHaveBeenCalledTimes(2);
	});

	it('does not allow UTCI dispatch after chunked exposure aborts before completion', async () => {
		const controller = new AbortController();
		const device = createFakeDevice({
			onSubmittedWorkDone: async () => {
				controller.abort();
			}
		});
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false);

		await pipeline.uploadStaticData(exposureUploadParams(1024));
		await expect(
			pipeline.runExposurePrecompute({
				numPoints: 1024,
				numHours: 1,
				numMonths: 1,
				exposureScheduling: {
					mode: 'chunked',
					maxWorkgroupsPerSlice: 4,
					yieldBetweenSlices: false
				},
				signal: controller.signal
			})
		).rejects.toMatchObject({ name: 'AbortError' });

		await expect(
			pipeline.runUtciForTimeIndex({
				format: 'f32-utci',
				numPoints: 1024,
				numHours: 1,
				numMonths: 1,
				timeIndex: 0
			})
		).rejects.toThrow(/solar\/sky exposure passes did not run/i);
	});

	it('destroys runAll exposure transient uniform buffers when queue wait rejects', async () => {
		const queueError = new Error('runAll queue wait failed');
		const device = createFakeDevice({
			onSubmittedWorkDone: async () => {
				throw queueError;
			}
		});
		const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false) as any;

		try {
			await pipeline.uploadStaticData(exposureUploadParams(1024));
			const firstExposureBufferIndex = device.__buffers.length;
			await pipeline.runAll({
				numPoints: 1024,
				numHours: 1,
				numMonths: 1
			});
			await Promise.resolve();

			const transientUniformBuffers = device.__buffers
				.slice(firstExposureBufferIndex)
				.filter((buffer) => buffer.size === 16)
				.slice(1);
			expect(transientUniformBuffers).toHaveLength(2);
			for (const buffer of transientUniformBuffers) {
				expect(buffer.destroy).toHaveBeenCalledTimes(1);
			}
			expect(consoleErrorSpy).toHaveBeenCalledWith(
				'WebGPU UTCI pipeline: runAll queue completion failed',
				queueError
			);
		} finally {
			consoleErrorSpy.mockRestore();
		}
	});

	it('blocks later UTCI dispatch when runAll exposure encoding throws before submit', async () => {
		const device = createFakeDevice({ throwOnCreateBindGroupCall: 2 });
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false) as any;

		await pipeline.uploadStaticData(exposureUploadParams(1024));
		pipeline.lastConfig = { numPoints: 1024, numHours: 1, numMonths: 1 };
		const firstRunAllBufferIndex = device.__buffers.length;

		await expect(
			pipeline.runAll({
				numPoints: 1024,
				numHours: 1,
				numMonths: 1
			})
		).rejects.toThrow(/createBindGroup failed on call 2/);

		const transientUniformBuffers = device.__buffers
			.slice(firstRunAllBufferIndex)
			.filter((buffer) => buffer.size === 16)
			.slice(1);
		expect(transientUniformBuffers).toHaveLength(1);
		expect(transientUniformBuffers[0]?.destroy).toHaveBeenCalledTimes(1);
		await expect(
			pipeline.runUtciForTimeIndex({
				format: 'f32-utci',
				numPoints: 1024,
				numHours: 1,
				numMonths: 1,
				timeIndex: 0
			})
		).rejects.toThrow(/solar\/sky exposure passes did not run/i);
		expect(pipeline.getOnDemandDiagnostics().path).not.toBe('run-all-baseline');
		expect(pipeline.getOnDemandDiagnostics().usedRunAllForSelectedHour).toBe(false);
	});

	it('destroys runAll exposure transient uniform buffers when MRT bind setup throws before submit', async () => {
		const device = createFakeDevice({ throwOnCreateBindGroupCall: 5 });
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false) as any;

		await pipeline.uploadStaticData(exposureUploadParams(1024));
		pipeline.lastConfig = { numPoints: 1024, numHours: 1, numMonths: 1 };
		const firstRunAllBufferIndex = device.__buffers.length;

		await expect(
			pipeline.runAll({
				numPoints: 1024,
				numHours: 1,
				numMonths: 1
			})
		).rejects.toThrow(/createBindGroup failed on call 5/);

		expect(device.queue.submit).not.toHaveBeenCalled();
		const transientUniformBuffers = device.__buffers
			.slice(firstRunAllBufferIndex)
			.filter((buffer) => buffer.size === 16)
			.slice(1);
		expect(transientUniformBuffers).toHaveLength(2);
		for (const buffer of transientUniformBuffers) {
			expect(buffer.destroy).toHaveBeenCalledTimes(1);
		}
		expect(pipeline.getOnDemandDiagnostics().path).not.toBe('run-all-baseline');
		expect(pipeline.getOnDemandDiagnostics().usedRunAllForSelectedHour).toBe(false);
		await expect(
			pipeline.runUtciForTimeIndex({
				format: 'f32-utci',
				numPoints: 1024,
				numHours: 1,
				numMonths: 1,
				timeIndex: 0
			})
		).rejects.toThrow(/solar\/sky exposure passes did not run/i);
	});

	it('stops chunked exposure before later slices when aborted after a queue wait', async () => {
		const controller = new AbortController();
		const device = createFakeDevice({
			onSubmittedWorkDone: async () => {
				controller.abort();
			}
		});
		const pipeline = new __TEST_ONLY_WebgpuUtciComputePipeline(device as any, false);

		await pipeline.uploadStaticData(exposureUploadParams(1024));

		await expect(
			pipeline.runExposurePrecompute({
				numPoints: 1024,
				numHours: 1,
				numMonths: 1,
				exposureScheduling: {
					mode: 'chunked',
					maxWorkgroupsPerSlice: 4,
					yieldBetweenSlices: false
				},
				signal: controller.signal
			})
		).rejects.toMatchObject({ name: 'AbortError' });

		expect(device.queue.submit).toHaveBeenCalledTimes(1);
		expect(device.queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);
	});
});
