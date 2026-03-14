import { getUtciFlatIndex, type UTCIComputePipeline } from '$lib/compute/gpu-pipeline';
import { serializeBvhForGpu } from '$lib/compute/bvhGpuUpload';
import * as THREE from 'three';
import mrtUtciShader from '$lib/compute/shaders/mrt_utci.wgsl?raw';
import bvhRaycastWgsl from '$lib/compute/shaders/bvh_raycast.wgsl?raw';
import exposureSolarWgsl from '$lib/compute/shaders/exposure_solar.wgsl?raw';
import exposureSkyWgsl from '$lib/compute/shaders/exposure_sky.wgsl?raw';

interface RunConfig {
	numPoints: number;
	numHours: number;
	numMonths: number;
}

const SOLAR_SHADER_CODE = bvhRaycastWgsl + '\n' + exposureSolarWgsl;
const SKY_SHADER_CODE = bvhRaycastWgsl + '\n' + exposureSkyWgsl;
const GATHER_SLICE_SHADER = `
struct Params {
	num_points: u32,
	total_time_steps: u32,
	time_index: u32,
	_pad: u32,
}

@group(0) @binding(0)
var<storage, read> source_utci: array<f32>;

@group(0) @binding(1)
var<storage, read_write> out_slice: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i >= params.num_points) { return; }
	let src_idx = i * params.total_time_steps + params.time_index;
	out_slice[i] = source_utci[src_idx];
}
`;

class WebgpuUtciComputePipeline implements UTCIComputePipeline {
	private device: GPUDevice;
	private queue: GPUQueue;

	private weatherData: Float32Array | null = null;
	private utciBuffer: GPUBuffer | null = null;
	private stagingBuffer: GPUBuffer | null = null;
	private solarExposureBuffer: GPUBuffer | null = null;
	private skyExposureBuffer: GPUBuffer | null = null;
	private weatherBuffer: GPUBuffer | null = null;
	private paramsBuffer: GPUBuffer | null = null;
	private lastConfig: RunConfig | null = null;

	private gridPointsBuffer: GPUBuffer | null = null;
	private sunVectorsBuffer: GPUBuffer | null = null;
	private sunAltitudesBuffer: GPUBuffer | null = null;
	private domeVectorsBuffer: GPUBuffer | null = null;
	private domeWeightsBuffer: GPUBuffer | null = null;
	private skyParamsBuffer: GPUBuffer | null = null;
	private bvhNodeBuffer: GPUBuffer | null = null;
	private bvhIndexBuffer: GPUBuffer | null = null;
	private bvhVertexBuffer: GPUBuffer | null = null;
	private bvhParamsBuffer: GPUBuffer | null = null;

	private sliceParamsBuffer: GPUBuffer | null = null;
	private sliceBuffer: GPUBuffer | null = null;
	private sliceStagingBuffer: GPUBuffer | null = null;

	private solarStagingBuffer: GPUBuffer | null = null;
	private skyStagingBuffer: GPUBuffer | null = null;

	private mrtBuffer: GPUBuffer | null = null;
	private mrtStagingBuffer: GPUBuffer | null = null;

	private pipeline: GPUComputePipeline | null = null;
	private solarPipeline: GPUComputePipeline | null = null;
	private skyPipeline: GPUComputePipeline | null = null;
	private gatherSlicePipeline: GPUComputePipeline | null = null;

	private pipelinePromise: Promise<GPUComputePipeline> | null = null;
	private solarPipelinePromise: Promise<GPUComputePipeline> | null = null;
	private skyPipelinePromise: Promise<GPUComputePipeline> | null = null;
	private gatherSlicePipelinePromise: Promise<GPUComputePipeline> | null = null;

	/** Set when solar/sky passes are dispatched in runAll; cleared at start of runAll. Used to fail readback clearly if exposure was skipped. */
	private ranExposurePassesThisRun = false;

	/** Last-uploaded sun vector samples (hour 0, 12, 23) for debugging; [x0,y0,z0, x12,y12,z12, x23,y23,z23]. */
	private lastSunVectorSamples: number[] = [];

	constructor(device: GPUDevice) {
		this.device = device;
		this.queue = device.queue;
	}

	private async ensurePipeline(): Promise<GPUComputePipeline> {
		if (this.pipeline) return this.pipeline;
		if (!this.pipelinePromise) {
			const module = this.device.createShaderModule({ code: mrtUtciShader });
			this.pipelinePromise = this.device
				.createComputePipelineAsync({
					layout: 'auto',
					compute: { module, entryPoint: 'main' }
				})
				.then((p) => {
					this.pipeline = p;
					return p;
				});
		}
		return this.pipelinePromise;
	}

	private async ensureSolarPipeline(): Promise<GPUComputePipeline> {
		if (this.solarPipeline) return this.solarPipeline;
		if (!this.solarPipelinePromise) {
			const module = this.device.createShaderModule({ code: SOLAR_SHADER_CODE });
			this.solarPipelinePromise = this.device
				.createComputePipelineAsync({
					layout: 'auto',
					compute: { module, entryPoint: 'main' }
				})
				.then((p) => {
					this.solarPipeline = p;
					return p;
				});
		}
		return this.solarPipelinePromise;
	}

	private async ensureSkyPipeline(): Promise<GPUComputePipeline> {
		if (this.skyPipeline) return this.skyPipeline;
		if (!this.skyPipelinePromise) {
			const module = this.device.createShaderModule({ code: SKY_SHADER_CODE });
			this.skyPipelinePromise = this.device
				.createComputePipelineAsync({
					layout: 'auto',
					compute: { module, entryPoint: 'main' }
				})
				.then((p) => {
					this.skyPipeline = p;
					return p;
				});
		}
		return this.skyPipelinePromise;
	}

	private async ensureGatherSlicePipeline(): Promise<GPUComputePipeline> {
		if (this.gatherSlicePipeline) return this.gatherSlicePipeline;
		if (!this.gatherSlicePipelinePromise) {
			const module = this.device.createShaderModule({ code: GATHER_SLICE_SHADER });
			this.gatherSlicePipelinePromise = this.device
				.createComputePipelineAsync({
					layout: 'auto',
					compute: { module, entryPoint: 'main' }
				})
				.then((p) => {
					this.gatherSlicePipeline = p;
					return p;
				});
		}
		return this.gatherSlicePipelinePromise;
	}

	async uploadStaticData(params: {
		gridPoints: Float32Array;
		sunVectors: Float32Array;
		sunAltitudes?: Float32Array;
		weather: Float32Array;
		domeVectors?: Float32Array;
		domeWeights?: Float32Array;
		mesh?: { geometry: import('three').BufferGeometry };
		serializedBvh?: import('$lib/compute/gpu-pipeline').SerializedBvhForGpu;
	}): Promise<void> {
		this.weatherData = params.weather;

		const numPoints = params.gridPoints.length / 3;
		if (!Number.isFinite(numPoints) || numPoints <= 0) {
			throw new Error(`WebGPU UTCI pipeline: invalid gridPoints length=${params.gridPoints.length}`);
		}

		const weatherStride = 7;
		if (this.weatherData.length % weatherStride !== 0) {
			throw new Error(
				`WebGPU UTCI pipeline: weather array length (${this.weatherData.length}) is not a multiple of ${weatherStride}`
			);
		}
		const totalTimeSteps = this.weatherData.length / weatherStride;

		const solarBytes = numPoints * totalTimeSteps * 4;
		if (!this.solarExposureBuffer || this.solarExposureBuffer.size !== solarBytes) {
			this.solarExposureBuffer?.destroy();
			this.solarExposureBuffer = this.device.createBuffer({
				size: solarBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
			});
		}

		const skyBytes = numPoints * 4;
		if (!this.skyExposureBuffer || this.skyExposureBuffer.size !== skyBytes) {
			this.skyExposureBuffer?.destroy();
			this.skyExposureBuffer = this.device.createBuffer({
				size: skyBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
			});
		}

		// Keep CPU allocations small by reusing tiny zero chunk writes.
		const zeroChunk = new Float32Array(4096);
		for (let offset = 0; offset < solarBytes; offset += zeroChunk.byteLength) {
			this.queue.writeBuffer(this.solarExposureBuffer, offset, zeroChunk.buffer, 0, Math.min(zeroChunk.byteLength, solarBytes - offset));
		}
		for (let offset = 0; offset < skyBytes; offset += zeroChunk.byteLength) {
			this.queue.writeBuffer(this.skyExposureBuffer, offset, zeroChunk.buffer, 0, Math.min(zeroChunk.byteLength, skyBytes - offset));
		}

		const gridBytes = params.gridPoints.byteLength;
		if (!this.gridPointsBuffer || this.gridPointsBuffer.size !== gridBytes) {
			this.gridPointsBuffer?.destroy();
			this.gridPointsBuffer = this.device.createBuffer({
				size: gridBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			});
		}
		this.queue.writeBuffer(this.gridPointsBuffer, 0, params.gridPoints.buffer, params.gridPoints.byteOffset, params.gridPoints.byteLength);

		const sunBytes = params.sunVectors.byteLength;
		if (!this.sunVectorsBuffer || this.sunVectorsBuffer.size !== sunBytes) {
			this.sunVectorsBuffer?.destroy();
			this.sunVectorsBuffer = this.device.createBuffer({
				size: sunBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			});
		}
		this.queue.writeBuffer(this.sunVectorsBuffer, 0, params.sunVectors.buffer, params.sunVectors.byteOffset, params.sunVectors.byteLength);

		// Store samples for debugging (hours 0, 12, 23) so we can verify sun vectors are non-zero and Y-up.
		this.lastSunVectorSamples = [];
		for (const hour of [0, 12, 23]) {
			if (hour * 3 + 2 < params.sunVectors.length) {
				this.lastSunVectorSamples.push(
					params.sunVectors[hour * 3] ?? 0,
					params.sunVectors[hour * 3 + 1] ?? 0,
					params.sunVectors[hour * 3 + 2] ?? 0
				);
			}
		}

		const altBytes = totalTimeSteps * 4;
		if (!this.sunAltitudesBuffer || this.sunAltitudesBuffer.size !== altBytes) {
			this.sunAltitudesBuffer?.destroy();
			this.sunAltitudesBuffer = this.device.createBuffer({
				size: altBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			});
		}
		if (params.sunAltitudes) {
			this.queue.writeBuffer(
				this.sunAltitudesBuffer,
				0,
				params.sunAltitudes.buffer,
				params.sunAltitudes.byteOffset,
				params.sunAltitudes.byteLength
			);
		}

		if (params.domeVectors && params.domeWeights) {
			const domeVecBytes = params.domeVectors.byteLength;
			if (!this.domeVectorsBuffer || this.domeVectorsBuffer.size !== domeVecBytes) {
				this.domeVectorsBuffer?.destroy();
				this.domeVectorsBuffer = this.device.createBuffer({
					size: domeVecBytes,
					usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
				});
			}
			this.queue.writeBuffer(this.domeVectorsBuffer, 0, params.domeVectors.buffer, params.domeVectors.byteOffset, params.domeVectors.byteLength);

			const domeWeightBytes = params.domeWeights.byteLength;
			if (!this.domeWeightsBuffer || this.domeWeightsBuffer.size !== domeWeightBytes) {
				this.domeWeightsBuffer?.destroy();
				this.domeWeightsBuffer = this.device.createBuffer({
					size: domeWeightBytes,
					usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
				});
			}
			this.queue.writeBuffer(this.domeWeightsBuffer, 0, params.domeWeights.buffer, params.domeWeights.byteOffset, params.domeWeights.byteLength);
		}

		const serialized =
			params.serializedBvh ??
			(params.mesh?.geometry
				? (() => {
						const mesh = params.mesh as THREE.Mesh;
						if (typeof mesh.updateMatrixWorld === 'function') mesh.updateMatrixWorld(true);
						return serializeBvhForGpu(mesh.geometry as THREE.BufferGeometry);
					})()
				: null);

		if (serialized) {
			const numNodes = serialized.bvhNodeBuffer.byteLength / 32;
			const numVertices = serialized.vertexBuffer.length / 3;
			const numIndices = serialized.indexBuffer.length;

			this.bvhNodeBuffer?.destroy();
			this.bvhNodeBuffer = this.device.createBuffer({
				size: serialized.bvhNodeBuffer.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			});
			this.queue.writeBuffer(this.bvhNodeBuffer, 0, serialized.bvhNodeBuffer);

			this.bvhIndexBuffer?.destroy();
			this.bvhIndexBuffer = this.device.createBuffer({
				size: serialized.bvhIndexBuffer.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			});
			this.queue.writeBuffer(this.bvhIndexBuffer, 0, serialized.bvhIndexBuffer);

			this.bvhVertexBuffer?.destroy();
			this.bvhVertexBuffer = this.device.createBuffer({
				size: serialized.vertexBuffer.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			});
			this.queue.writeBuffer(
				this.bvhVertexBuffer,
				0,
				serialized.vertexBuffer.buffer,
				serialized.vertexBuffer.byteOffset,
				serialized.vertexBuffer.byteLength
			);

			this.bvhParamsBuffer?.destroy();
			this.bvhParamsBuffer = this.device.createBuffer({
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
			this.queue.writeBuffer(this.bvhParamsBuffer, 0, new Uint32Array([numNodes, numVertices, numIndices, 0]));
		}
	}

	private createBvhBindGroup(pipeline: GPUComputePipeline): GPUBindGroup {
		if (!this.bvhNodeBuffer || !this.bvhIndexBuffer || !this.bvhVertexBuffer || !this.bvhParamsBuffer) {
			throw new Error('WebGPU UTCI pipeline: BVH buffers not initialized');
		}
		return this.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(1),
			entries: [
				{ binding: 0, resource: { buffer: this.bvhNodeBuffer } },
				{ binding: 1, resource: { buffer: this.bvhIndexBuffer } },
				{ binding: 2, resource: { buffer: this.bvhVertexBuffer } },
				{ binding: 3, resource: { buffer: this.bvhParamsBuffer } }
			]
		});
	}

	async runAll(params: { numPoints: number; numHours: number; numMonths: number; workgroupSize?: number }): Promise<void> {
		if (!this.weatherData) {
			throw new Error('WebGPU UTCI pipeline: weather data not uploaded');
		}

		const { numPoints, numHours, numMonths } = params;
		const totalTimeSteps = numHours * numMonths;
		const [mrtPipeline, solarPipeline, skyPipeline] = await Promise.all([
			this.ensurePipeline(),
			this.ensureSolarPipeline(),
			this.ensureSkyPipeline()
		]);

		const requiredWeatherBytes = this.weatherData.byteLength;
		if (!this.weatherBuffer || this.weatherBuffer.size !== requiredWeatherBytes) {
			this.weatherBuffer?.destroy();
			this.weatherBuffer = this.device.createBuffer({
				size: requiredWeatherBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			});
		}

		const utciBytes = numPoints * totalTimeSteps * 4;
		if (!this.utciBuffer || this.utciBuffer.size !== utciBytes) {
			this.utciBuffer?.destroy();
			this.utciBuffer = this.device.createBuffer({
				size: utciBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
			});
		}

		if (!this.mrtBuffer || this.mrtBuffer.size !== utciBytes) {
			this.mrtBuffer?.destroy();
			this.mrtBuffer = this.device.createBuffer({
				size: utciBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
			});
		}

		if (!this.paramsBuffer || this.paramsBuffer.size !== 16) {
			this.paramsBuffer?.destroy();
			this.paramsBuffer = this.device.createBuffer({
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
		}

		this.queue.writeBuffer(this.weatherBuffer, 0, this.weatherData.buffer, this.weatherData.byteOffset, this.weatherData.byteLength);
		this.queue.writeBuffer(this.paramsBuffer, 0, new Uint32Array([numPoints, totalTimeSteps]));

		const workgroupSize = params.workgroupSize ?? 64;
		const workgroupsX = Math.ceil(numPoints / workgroupSize);
		this.ranExposurePassesThisRun = false;
		const encoder = this.device.createCommandEncoder();

		const hasBvh = this.bvhNodeBuffer && this.bvhIndexBuffer && this.bvhVertexBuffer && this.bvhParamsBuffer;
		if (hasBvh && this.gridPointsBuffer && this.sunVectorsBuffer && this.solarExposureBuffer) {
			this.ranExposurePassesThisRun = true;
			const solarBindGroup0 = this.device.createBindGroup({
				layout: solarPipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: this.gridPointsBuffer } },
					{ binding: 1, resource: { buffer: this.sunVectorsBuffer } },
					{ binding: 2, resource: { buffer: this.solarExposureBuffer } },
					{ binding: 3, resource: { buffer: this.paramsBuffer } }
				]
			});
			const solarPass = encoder.beginComputePass();
			solarPass.setPipeline(solarPipeline);
			solarPass.setBindGroup(0, solarBindGroup0);
			solarPass.setBindGroup(1, this.createBvhBindGroup(solarPipeline));
			solarPass.dispatchWorkgroups(workgroupsX, totalTimeSteps, 1);
			solarPass.end();
		}

		const numPatches = 145;
		if (hasBvh && this.gridPointsBuffer && this.domeVectorsBuffer && this.domeWeightsBuffer && this.skyExposureBuffer) {
			this.ranExposurePassesThisRun = true;
			if (!this.skyParamsBuffer || this.skyParamsBuffer.size !== 16) {
				this.skyParamsBuffer?.destroy();
				this.skyParamsBuffer = this.device.createBuffer({
					size: 16,
					usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
				});
			}
			this.queue.writeBuffer(this.skyParamsBuffer, 0, new Uint32Array([numPoints, numPatches]));
			const skyBindGroup0 = this.device.createBindGroup({
				layout: skyPipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: this.gridPointsBuffer } },
					{ binding: 1, resource: { buffer: this.domeVectorsBuffer } },
					{ binding: 2, resource: { buffer: this.domeWeightsBuffer } },
					{ binding: 3, resource: { buffer: this.skyExposureBuffer } },
					{ binding: 4, resource: { buffer: this.skyParamsBuffer } }
				]
			});
			const skyPass = encoder.beginComputePass();
			skyPass.setPipeline(skyPipeline);
			skyPass.setBindGroup(0, skyBindGroup0);
			skyPass.setBindGroup(1, this.createBvhBindGroup(skyPipeline));
			skyPass.dispatchWorkgroups(workgroupsX, 1, 1);
			skyPass.end();
		}

		if (!this.solarExposureBuffer || !this.skyExposureBuffer || !this.weatherBuffer || !this.utciBuffer || !this.mrtBuffer || !this.paramsBuffer || !this.sunAltitudesBuffer) {
			throw new Error('WebGPU UTCI pipeline: MRT bindings are not initialized');
		}
		const bindGroup = this.device.createBindGroup({
			layout: mrtPipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: this.solarExposureBuffer } },
				{ binding: 1, resource: { buffer: this.skyExposureBuffer } },
				{ binding: 2, resource: { buffer: this.weatherBuffer } },
				{ binding: 3, resource: { buffer: this.utciBuffer } },
				{ binding: 4, resource: { buffer: this.paramsBuffer } },
				{ binding: 5, resource: { buffer: this.sunAltitudesBuffer } },
				{ binding: 6, resource: { buffer: this.mrtBuffer } }
			]
		});

		const pass = encoder.beginComputePass();
		pass.setPipeline(mrtPipeline);
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

		const gatherPipeline = await this.ensureGatherSlicePipeline();
		const sliceBytes = numPoints * 4;

		if (!this.sliceBuffer || this.sliceBuffer.size !== sliceBytes) {
			this.sliceBuffer?.destroy();
			this.sliceBuffer = this.device.createBuffer({
				size: sliceBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
			});
		}
		if (!this.sliceStagingBuffer || this.sliceStagingBuffer.size !== sliceBytes) {
			this.sliceStagingBuffer?.destroy();
			this.sliceStagingBuffer = this.device.createBuffer({
				size: sliceBytes,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
		}
		if (!this.sliceParamsBuffer || this.sliceParamsBuffer.size !== 16) {
			this.sliceParamsBuffer?.destroy();
			this.sliceParamsBuffer = this.device.createBuffer({
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
		}
		this.queue.writeBuffer(this.sliceParamsBuffer, 0, new Uint32Array([numPoints, totalTimeSteps, timeIndex, 0]));

		const bindGroup = this.device.createBindGroup({
			layout: gatherPipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: this.utciBuffer } },
				{ binding: 1, resource: { buffer: this.sliceBuffer } },
				{ binding: 2, resource: { buffer: this.sliceParamsBuffer } }
			]
		});

		const workgroups = Math.ceil(numPoints / 256);
		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(gatherPipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(workgroups, 1, 1);
		pass.end();

		encoder.copyBufferToBuffer(this.sliceBuffer, 0, this.sliceStagingBuffer, 0, sliceBytes);
		this.queue.submit([encoder.finish()]);

		await this.sliceStagingBuffer.mapAsync(GPUMapMode.READ);
		const mapped = new Float32Array(this.sliceStagingBuffer.getMappedRange());
		const out = new Float32Array(mapped.length);
		out.set(mapped);
		this.sliceStagingBuffer.unmap();
		return out;
	}

	async readSolarExposureFull(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
	}): Promise<Float32Array> {
		if (!this.solarExposureBuffer || !this.lastConfig) {
			throw new Error('WebGPU UTCI pipeline: solar exposure buffer not available (run runAll first)');
		}
		if (!this.ranExposurePassesThisRun) {
			throw new Error(
				'WebGPU UTCI pipeline: solar/sky exposure passes did not run (no BVH?). readSolarExposureFull would return zeros.'
			);
		}
		await this.queue.onSubmittedWorkDone();
		const { numPoints, numHours, numMonths } = params;
		const totalTimeSteps = numHours * numMonths;
		const bytes = numPoints * totalTimeSteps * 4;
		if (!this.solarStagingBuffer || this.solarStagingBuffer.size !== bytes) {
			this.solarStagingBuffer?.destroy();
			this.solarStagingBuffer = this.device.createBuffer({
				size: bytes,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
		}
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.solarExposureBuffer, 0, this.solarStagingBuffer, 0, bytes);
		this.queue.submit([encoder.finish()]);
		await this.solarStagingBuffer.mapAsync(GPUMapMode.READ);
		const mapped = new Float32Array(this.solarStagingBuffer.getMappedRange());
		const out = new Float32Array(mapped.length);
		out.set(mapped);
		this.solarStagingBuffer.unmap();
		return out;
	}

	async readSkyExposure(params: { numPoints: number }): Promise<Float32Array> {
		if (!this.skyExposureBuffer || !this.lastConfig) {
			throw new Error('WebGPU UTCI pipeline: sky exposure buffer not available (run runAll first)');
		}
		if (!this.ranExposurePassesThisRun) {
			throw new Error(
				'WebGPU UTCI pipeline: solar/sky exposure passes did not run (no BVH?). readSkyExposure would return zeros.'
			);
		}
		await this.queue.onSubmittedWorkDone();
		const { numPoints } = params;
		const bytes = numPoints * 4;
		if (!this.skyStagingBuffer || this.skyStagingBuffer.size !== bytes) {
			this.skyStagingBuffer?.destroy();
			this.skyStagingBuffer = this.device.createBuffer({
				size: bytes,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
		}
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.skyExposureBuffer, 0, this.skyStagingBuffer, 0, bytes);
		this.queue.submit([encoder.finish()]);
		await this.skyStagingBuffer.mapAsync(GPUMapMode.READ);
		const mapped = new Float32Array(this.skyStagingBuffer.getMappedRange());
		const out = new Float32Array(mapped.length);
		out.set(mapped);
		this.skyStagingBuffer.unmap();
		return out;
	}

	async readMrtFull(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
	}): Promise<Float32Array> {
		if (!this.mrtBuffer || !this.lastConfig) {
			throw new Error('WebGPU UTCI pipeline: MRT buffer not available (run runAll first)');
		}
		await this.queue.onSubmittedWorkDone();
		const { numPoints, numHours, numMonths } = params;
		const totalTimeSteps = numHours * numMonths;
		const bytes = numPoints * totalTimeSteps * 4;
		if (!this.mrtStagingBuffer || this.mrtStagingBuffer.size !== bytes) {
			this.mrtStagingBuffer?.destroy();
			this.mrtStagingBuffer = this.device.createBuffer({
				size: bytes,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
		}
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.mrtBuffer, 0, this.mrtStagingBuffer, 0, bytes);
		this.queue.submit([encoder.finish()]);
		await this.mrtStagingBuffer.mapAsync(GPUMapMode.READ);
		const mapped = new Float32Array(this.mrtStagingBuffer.getMappedRange());
		const out = new Float32Array(mapped.length);
		out.set(mapped);
		this.mrtStagingBuffer.unmap();
		return out;
	}

	getSunVectorSamples(): number[] | null {
		return this.lastSunVectorSamples.length > 0 ? [...this.lastSunVectorSamples] : null;
	}

	/**
	 * Return first N hours of uploaded weather as objects for parity comparison with Python export.
	 * Layout matches WeatherSample: [air_temp, mrt_longwave, wind_speed, rel_humidity, direct_normal, diffuse_horizontal, horiz_infrared].
	 */
	getWeatherSample(numHours = 3): Array<{
		air_temp: number;
		direct_normal: number;
		diffuse_horizontal: number;
		horiz_infrared: number;
		wind_speed: number;
		rel_humidity: number;
	}> {
		if (!this.weatherData) return [];
		const stride = 7;
		const out: Array<{
			air_temp: number;
			direct_normal: number;
			diffuse_horizontal: number;
			horiz_infrared: number;
			wind_speed: number;
			rel_humidity: number;
		}> = [];
		for (let h = 0; h < numHours && h * stride + 6 < this.weatherData.length; h++) {
			const base = h * stride;
			out.push({
				air_temp: this.weatherData[base] ?? 0,
				direct_normal: this.weatherData[base + 4] ?? 0,
				diffuse_horizontal: this.weatherData[base + 5] ?? 0,
				horiz_infrared: this.weatherData[base + 6] ?? 0,
				wind_speed: this.weatherData[base + 2] ?? 0,
				rel_humidity: this.weatherData[base + 3] ?? 0
			});
		}
		return out;
	}

	dispose(): void {
		this.solarExposureBuffer?.destroy();
		this.solarExposureBuffer = null;
		this.skyExposureBuffer?.destroy();
		this.skyExposureBuffer = null;
		this.gridPointsBuffer?.destroy();
		this.gridPointsBuffer = null;
		this.sunVectorsBuffer?.destroy();
		this.sunVectorsBuffer = null;
		this.sunAltitudesBuffer?.destroy();
		this.sunAltitudesBuffer = null;
		this.domeVectorsBuffer?.destroy();
		this.domeVectorsBuffer = null;
		this.domeWeightsBuffer?.destroy();
		this.domeWeightsBuffer = null;
		this.bvhNodeBuffer?.destroy();
		this.bvhNodeBuffer = null;
		this.bvhIndexBuffer?.destroy();
		this.bvhIndexBuffer = null;
		this.bvhVertexBuffer?.destroy();
		this.bvhVertexBuffer = null;
		this.bvhParamsBuffer?.destroy();
		this.bvhParamsBuffer = null;
		this.weatherBuffer?.destroy();
		this.weatherBuffer = null;
		this.utciBuffer?.destroy();
		this.utciBuffer = null;
		this.stagingBuffer?.destroy();
		this.stagingBuffer = null;
		this.paramsBuffer?.destroy();
		this.paramsBuffer = null;
		this.skyParamsBuffer?.destroy();
		this.skyParamsBuffer = null;
		this.sliceParamsBuffer?.destroy();
		this.sliceParamsBuffer = null;
		this.sliceBuffer?.destroy();
		this.sliceBuffer = null;
		this.sliceStagingBuffer?.destroy();
		this.sliceStagingBuffer = null;
		this.solarStagingBuffer?.destroy();
		this.solarStagingBuffer = null;
		this.skyStagingBuffer?.destroy();
		this.skyStagingBuffer = null;
		this.mrtBuffer?.destroy();
		this.mrtBuffer = null;
		this.mrtStagingBuffer?.destroy();
		this.mrtStagingBuffer = null;

		this.pipeline = null;
		this.solarPipeline = null;
		this.skyPipeline = null;
		this.gatherSlicePipeline = null;
		this.pipelinePromise = null;
		this.solarPipelinePromise = null;
		this.skyPipelinePromise = null;
		this.gatherSlicePipelinePromise = null;
		this.lastConfig = null;
		this.weatherData = null;
		this.ranExposurePassesThisRun = false;
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
	device.lost.then(() => {
		if (cachedDevicePromise && cachedDevice === device) {
			cachedDevicePromise = null;
			cachedDevice = null;
		}
	});
	return device;
}

let cachedDevicePromise: Promise<GPUDevice> | null = null;
let cachedDevice: GPUDevice | null = null;

export async function createWebgpuUtciPipeline(): Promise<UTCIComputePipeline> {
	if (!cachedDevicePromise) {
		cachedDevicePromise = getWebgpuDevice().then((device) => {
			cachedDevice = device;
			return device;
		});
	}
	const device = await cachedDevicePromise;
	return new WebgpuUtciComputePipeline(device);
}

export function __resetWebgpuDeviceCacheForTests(): void {
	cachedDevicePromise = null;
	cachedDevice = null;
}
