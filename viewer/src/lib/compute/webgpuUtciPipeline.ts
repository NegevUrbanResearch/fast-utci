import type { UTCIComputePipeline } from '$lib/compute/gpu-pipeline';
import mrtUtciShader from '$lib/compute/shaders/mrt_utci.wgsl?raw';

/**
 * Browser-only WebGPU implementation of the UTCI compute pipeline.
 *
 * Phase 1 scope:
 * - Uses the MRT+UTCI compute shader to calculate UTCI from per-hour weather.
 * - Ignores geometry-dependent fields (solar/sky exposure) for now; MRT comes
 *   directly from the longwave term in the weather buffer, matching Phase 1
 *   of the migration plan.
 */

interface RunConfig {
	numPoints: number;
	numHours: number;
	numMonths: number;
}

class WebgpuUtciComputePipeline implements UTCIComputePipeline {
	private device: GPUDevice;
	private queue: GPUQueue;

	private weatherData: Float32Array | null = null;
	private utciBuffer: GPUBuffer | null = null;
	private stagingBuffer: GPUBuffer | null = null;
	private weatherBuffer: GPUBuffer | null = null;
	private paramsBuffer: GPUBuffer | null = null;
	private pipeline: GPUComputePipeline | null = null;
	private lastConfig: RunConfig | null = null;

	constructor(device: GPUDevice) {
		this.device = device;
		this.queue = device.queue;
	}

	private ensurePipeline() {
		if (this.pipeline) return;
		const module = this.device.createShaderModule({
			code: mrtUtciShader
		});

		this.pipeline = this.device.createComputePipeline({
			layout: 'auto',
			compute: {
				module,
				entryPoint: 'main'
			}
		});
	}

	async uploadStaticData(params: {
		gridPoints: Float32Array;
		sunVectors: Float32Array;
		weather: Float32Array;
	}): Promise<void> {
		// Phase 1: only weather data influences UTCI via MRT; geometry-dependent
		// terms are reserved for later phases.
		this.weatherData = params.weather;
	}

	async runAll(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
		workgroupSize?: number;
	}): Promise<void> {
		if (!this.weatherData) {
			throw new Error('WebGPU UTCI pipeline: weather data not uploaded');
		}

		const { numPoints, numHours, numMonths } = params;
		const totalTimeSteps = numHours * numMonths;

		this.ensurePipeline();
		if (!this.pipeline) {
			throw new Error('WebGPU UTCI pipeline: failed to create compute pipeline');
		}

		// (Re)create weather buffer when size changes.
		const requiredWeatherBytes = this.weatherData.byteLength;
		if (!this.weatherBuffer || this.weatherBuffer.size !== requiredWeatherBytes) {
			this.weatherBuffer?.destroy();
			this.weatherBuffer = this.device.createBuffer({
				size: requiredWeatherBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			});
		}

		// UTCI results buffer: one f32 per (point × timestep). Storage-only; we
		// use a separate staging buffer for MAP_READ to satisfy WebGPU rules.
		const utciBytes = numPoints * totalTimeSteps * 4;
		if (!this.utciBuffer || this.utciBuffer.size !== utciBytes) {
			this.utciBuffer?.destroy();
			this.utciBuffer = this.device.createBuffer({
				size: utciBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
			});
		}

		// Staging buffer for readback.
		if (!this.stagingBuffer || this.stagingBuffer.size !== utciBytes) {
			this.stagingBuffer?.destroy();
			this.stagingBuffer = this.device.createBuffer({
				size: utciBytes,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
		}

		// MRTParams uniform: { num_points: u32; num_time_steps: u32; }
		const paramsBytes = 16; // Align to 16 bytes for uniform buffer.
		if (!this.paramsBuffer || this.paramsBuffer.size !== paramsBytes) {
			this.paramsBuffer?.destroy();
			this.paramsBuffer = this.device.createBuffer({
				size: paramsBytes,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
		}

		// Upload static data.
		this.queue.writeBuffer(this.weatherBuffer!, 0, this.weatherData.buffer, this.weatherData.byteOffset, this.weatherData.byteLength);

		const paramArray = new Uint32Array([numPoints, totalTimeSteps]);
		this.queue.writeBuffer(this.paramsBuffer!, 0, paramArray.buffer, paramArray.byteOffset, paramArray.byteLength);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				// Shader currently reads only bindings 2, 3, 4 (weather_data, utci_results, params).
				{ binding: 2, resource: { buffer: this.weatherBuffer! } },
				{ binding: 3, resource: { buffer: this.utciBuffer! } },
				{ binding: 4, resource: { buffer: this.paramsBuffer! } }
			]
		});

		const workgroupSize = params.workgroupSize ?? 64;
		const workgroupsX = Math.ceil(numPoints / workgroupSize);

		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(workgroupsX, totalTimeSteps, 1);
		pass.end();

		this.queue.submit([encoder.finish()]);

		this.lastConfig = { numPoints, numHours, numMonths };
	}

	async readUtcisSlice(params: {
		monthIndex: number;
		hourIndex: number;
		numPoints: number;
		numHours: number;
		numMonths: number;
	}): Promise<Float32Array> {
		if (!this.utciBuffer || !this.lastConfig) {
			throw new Error('WebGPU UTCI pipeline: results buffer not available');
		}

		const { monthIndex, hourIndex, numPoints, numHours, numMonths } = params;
		const totalTimeSteps = numHours * numMonths;
		const timeIndex = monthIndex * numHours + hourIndex;

		if (timeIndex < 0 || timeIndex >= totalTimeSteps) {
			throw new Error(`Invalid time index ${timeIndex} for totalTimeSteps=${totalTimeSteps}`);
		}

		// Copy GPU results into the staging buffer, then map that for reading.
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(
			this.utciBuffer,
			0,
			this.stagingBuffer!,
			0,
			this.utciBuffer.size
		);
		this.queue.submit([encoder.finish()]);

		await this.stagingBuffer!.mapAsync(GPUMapMode.READ);
		const data = new Float32Array(this.stagingBuffer!.getMappedRange());

		const slice = new Float32Array(numPoints);
		for (let i = 0; i < numPoints; i++) {
			const flatIndex = i * totalTimeSteps + timeIndex;
			slice[i] = data[flatIndex];
		}

		this.stagingBuffer!.unmap();
		return slice;
	}
}

async function getWebgpuDevice(): Promise<GPUDevice> {
	if (typeof navigator === 'undefined' || !(navigator as any).gpu) {
		throw new Error('WebGPU not available in this environment');
	}

	const adapter = await (navigator as any).gpu.requestAdapter();
	if (!adapter) {
		throw new Error('Failed to acquire WebGPU adapter');
	}

	const device: GPUDevice = await adapter.requestDevice();
	return device;
}

let cachedDevicePromise: Promise<GPUDevice> | null = null;

export async function createWebgpuUtciPipeline(): Promise<UTCIComputePipeline> {
	if (!cachedDevicePromise) {
		cachedDevicePromise = getWebgpuDevice();
	}
	const device = await cachedDevicePromise;
	return new WebgpuUtciComputePipeline(device);
}

