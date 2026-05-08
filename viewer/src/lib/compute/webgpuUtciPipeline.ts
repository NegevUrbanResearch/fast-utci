import {
	getUtciFlatIndex,
	type OnDemandUtciOutput,
	type RunUtciForTimeIndexParams,
	type UTCIComputePipeline
} from '$lib/compute/gpu-pipeline';
import {
	createEmptyOnDemandDiagnostics,
	mergeTrackedGpuAllocationBytes,
	recordOnDemandTiming,
	type OnDemandRuntimeDiagnostics
} from '$lib/compute/onDemandDiagnostics';
import { serializeBvhForGpu } from '$lib/compute/bvhGpuUpload';
import * as THREE from 'three';
import mrtUtciShaderRaw from '$lib/compute/shaders/mrt_utci.wgsl?raw';
import mrtUtciOnDemandShaderRaw from '$lib/compute/shaders/mrt_utci_on_demand.wgsl?raw';
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
const MRT_DECLS_PATTERN = /\/\/ MRT_COMPONENT_DECLS_START[\s\S]*?\/\/ MRT_COMPONENT_DECLS_END\n?/;
const MRT_WRITES_PATTERN = /[ \t]*\/\/ MRT_COMPONENT_WRITES_START[\s\S]*?[ \t]*\/\/ MRT_COMPONENT_WRITES_END\n?/;

function getMrtShaderCode(enableMrtComponents: boolean): string {
	if (enableMrtComponents) return mrtUtciShaderRaw;
	let code = mrtUtciShaderRaw.replace(MRT_DECLS_PATTERN, '').replace(MRT_WRITES_PATTERN, '');
	const stillHasComponentBindings = /@group\(0\)\s*@binding\((7|8|9|10)\)/.test(code);
	const stillHasComponentWrites =
		code.includes('short_erf_results[') ||
		code.includes('long_erf_results[') ||
		code.includes('short_dmrt_results[') ||
		code.includes('long_dmrt_results[');
	if (stillHasComponentBindings || stillHasComponentWrites) {
		throw new Error(
			'Failed to build MRT shader without component diagnostics; marker replacement did not fully remove component bindings/writes.'
		);
	}
	return code;
}
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
	private supportsMrtComponents: boolean;
	private onDemandDiagnostics = createEmptyOnDemandDiagnostics();

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
	private onDemandOutputBuffer: GPUBuffer | null = null;
	private onDemandParamsBuffer: GPUBuffer | null = null;
	private onDemandReadbackBuffer: GPUBuffer | null = null;
	private onDemandReadbackSerial: Promise<void> = Promise.resolve();

	private solarStagingBuffer: GPUBuffer | null = null;
	private skyStagingBuffer: GPUBuffer | null = null;

	private mrtBuffer: GPUBuffer | null = null;
	private mrtStagingBuffer: GPUBuffer | null = null;
	private shortErfBuffer: GPUBuffer | null = null;
	private longErfBuffer: GPUBuffer | null = null;
	private shortDmrtBuffer: GPUBuffer | null = null;
	private longDmrtBuffer: GPUBuffer | null = null;
	private shortErfStagingBuffer: GPUBuffer | null = null;
	private longErfStagingBuffer: GPUBuffer | null = null;
	private shortDmrtStagingBuffer: GPUBuffer | null = null;
	private longDmrtStagingBuffer: GPUBuffer | null = null;

	private pipeline: GPUComputePipeline | null = null;
	private solarPipeline: GPUComputePipeline | null = null;
	private skyPipeline: GPUComputePipeline | null = null;
	private gatherSlicePipeline: GPUComputePipeline | null = null;
	private onDemandPipeline: GPUComputePipeline | null = null;

	private pipelinePromise: Promise<GPUComputePipeline> | null = null;
	private solarPipelinePromise: Promise<GPUComputePipeline> | null = null;
	private skyPipelinePromise: Promise<GPUComputePipeline> | null = null;
	private gatherSlicePipelinePromise: Promise<GPUComputePipeline> | null = null;
	private onDemandPipelinePromise: Promise<GPUComputePipeline> | null = null;

	/** Set when solar/sky passes are dispatched in runAll/runExposurePrecompute; cleared at start of each run. Used to fail readback clearly if exposure was skipped. */
	private ranExposurePassesThisRun = false;

	/** Last-uploaded sun vector samples (hour 0, 12, 23) for debugging; [x0,y0,z0, x12,y12,z12, x23,y23,z23]. */
	private lastSunVectorSamples: number[] = [];

	constructor(device: GPUDevice, supportsMrtComponents: boolean) {
		this.device = device;
		this.queue = device.queue;
		this.supportsMrtComponents = supportsMrtComponents;
	}

	getOnDemandDiagnostics(): OnDemandRuntimeDiagnostics {
		return {
			...this.onDemandDiagnostics,
			trackedGpuAllocationBytes: { ...this.onDemandDiagnostics.trackedGpuAllocationBytes },
			timeIndices: [...this.onDemandDiagnostics.timeIndices],
			timings: { ...this.onDemandDiagnostics.timings }
		};
	}

	private async ensurePipeline(): Promise<GPUComputePipeline> {
		if (this.pipeline) return this.pipeline;
		if (!this.pipelinePromise) {
			const module = this.device.createShaderModule({
				code: getMrtShaderCode(this.supportsMrtComponents)
			});
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

	private async ensureWeatherBuffer(): Promise<GPUBuffer> {
		if (!this.weatherData) {
			throw new Error('WebGPU UTCI pipeline: weather data not uploaded');
		}

		const requiredWeatherBytes = this.weatherData.byteLength;
		if (!this.weatherBuffer || this.weatherBuffer.size !== requiredWeatherBytes) {
			this.weatherBuffer?.destroy();
			this.weatherBuffer = this.device.createBuffer({
				size: requiredWeatherBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			});
		}
		this.queue.writeBuffer(
			this.weatherBuffer,
			0,
			this.weatherData.buffer,
			this.weatherData.byteOffset,
			this.weatherData.byteLength
		);
		return this.weatherBuffer;
	}

	private async ensureOnDemandPipeline(): Promise<GPUComputePipeline> {
		if (this.onDemandPipeline) return this.onDemandPipeline;
		if (!this.onDemandPipelinePromise) {
			const module = this.device.createShaderModule({ code: mrtUtciOnDemandShaderRaw });
			this.onDemandPipelinePromise = this.device
				.createComputePipelineAsync({
					layout: 'auto',
					compute: { module, entryPoint: 'main' }
				})
				.then((p) => {
					this.onDemandPipeline = p;
					return p;
				});
		}
		return this.onDemandPipelinePromise;
	}

	private clearBvhState(): void {
		this.bvhNodeBuffer?.destroy();
		this.bvhNodeBuffer = null;
		this.bvhIndexBuffer?.destroy();
		this.bvhIndexBuffer = null;
		this.bvhVertexBuffer?.destroy();
		this.bvhVertexBuffer = null;
		this.bvhParamsBuffer?.destroy();
		this.bvhParamsBuffer = null;
	}

	private clearSunAltitudeState(): void {
		this.sunAltitudesBuffer?.destroy();
		this.sunAltitudesBuffer = null;
	}

	private clearSkyDomeState(): void {
		this.domeVectorsBuffer?.destroy();
		this.domeVectorsBuffer = null;
		this.domeWeightsBuffer?.destroy();
		this.domeWeightsBuffer = null;
	}

	private assertMatchesLastConfig(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
		context: string;
	}): void {
		if (!this.lastConfig) {
			return;
		}

		const { numPoints, numHours, numMonths, context } = params;
		if (
			this.lastConfig.numPoints !== numPoints ||
			this.lastConfig.numHours !== numHours ||
			this.lastConfig.numMonths !== numMonths
		) {
			throw new Error(
				`WebGPU UTCI pipeline: ${context} request does not match the last run config ` +
					`(expected numPoints=${this.lastConfig.numPoints}, numHours=${this.lastConfig.numHours}, numMonths=${this.lastConfig.numMonths}; ` +
					`got numPoints=${numPoints}, numHours=${numHours}, numMonths=${numMonths}).`
			);
		}
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
		this.weatherData = new Float32Array(params.weather);
		this.ranExposurePassesThisRun = false;

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

		// Bit-packed: 1 bit per (point, time_step), packed into u32 words.
		const totalSolarBits = numPoints * totalTimeSteps;
		const solarWords = Math.ceil(totalSolarBits / 32);
		const solarBytes = solarWords * 4;
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

		if (params.sunAltitudes) {
			const altBytes = totalTimeSteps * 4;
			if (!this.sunAltitudesBuffer || this.sunAltitudesBuffer.size !== altBytes) {
				this.sunAltitudesBuffer?.destroy();
				this.sunAltitudesBuffer = this.device.createBuffer({
					size: altBytes,
					usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
				});
			}
			this.queue.writeBuffer(
				this.sunAltitudesBuffer,
				0,
				params.sunAltitudes.buffer,
				params.sunAltitudes.byteOffset,
				params.sunAltitudes.byteLength
			);
		} else {
			this.clearSunAltitudeState();
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
		} else {
			this.clearSkyDomeState();
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
		} else {
			this.clearBvhState();
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

	private async encodeExposurePasses(params: {
		encoder: GPUCommandEncoder;
		numPoints: number;
		totalTimeSteps: number;
		workgroupSize: number;
		solarPipeline: GPUComputePipeline;
		skyPipeline: GPUComputePipeline;
	}): Promise<void> {
		const { encoder, numPoints, totalTimeSteps, workgroupSize, solarPipeline, skyPipeline } = params;
		if (!this.paramsBuffer) {
			throw new Error('WebGPU UTCI pipeline: params buffer not initialized');
		}
		const paramsBuffer = this.paramsBuffer;
		const workgroupsX = Math.ceil(numPoints / workgroupSize);
		const hasBvh = this.bvhNodeBuffer && this.bvhIndexBuffer && this.bvhVertexBuffer && this.bvhParamsBuffer;

		if (hasBvh && this.gridPointsBuffer && this.sunVectorsBuffer && this.solarExposureBuffer) {
			this.ranExposurePassesThisRun = true;
			const solarBindGroup0 = this.device.createBindGroup({
				layout: solarPipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: this.gridPointsBuffer } },
					{ binding: 1, resource: { buffer: this.sunVectorsBuffer } },
					{ binding: 2, resource: { buffer: this.solarExposureBuffer } },
					{ binding: 3, resource: { buffer: paramsBuffer } }
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
		await this.ensureWeatherBuffer();

		const utciBytes = numPoints * totalTimeSteps * 4;
		if (!this.utciBuffer || this.utciBuffer.size !== utciBytes) {
			this.utciBuffer?.destroy();
			this.utciBuffer = this.device.createBuffer({
				size: utciBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
			});
		}

		const mrtBytes = numPoints * totalTimeSteps * 4;
		if (!this.mrtBuffer || this.mrtBuffer.size !== mrtBytes) {
			this.mrtBuffer?.destroy();
			this.mrtBuffer = this.device.createBuffer({
				size: mrtBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
			});
		}
		if (this.supportsMrtComponents) {
			if (!this.shortErfBuffer || this.shortErfBuffer.size !== utciBytes) {
				this.shortErfBuffer?.destroy();
				this.shortErfBuffer = this.device.createBuffer({
					size: utciBytes,
					usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
				});
			}
			if (!this.longErfBuffer || this.longErfBuffer.size !== utciBytes) {
				this.longErfBuffer?.destroy();
				this.longErfBuffer = this.device.createBuffer({
					size: utciBytes,
					usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
				});
			}
			if (!this.shortDmrtBuffer || this.shortDmrtBuffer.size !== utciBytes) {
				this.shortDmrtBuffer?.destroy();
				this.shortDmrtBuffer = this.device.createBuffer({
					size: utciBytes,
					usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
				});
			}
			if (!this.longDmrtBuffer || this.longDmrtBuffer.size !== utciBytes) {
				this.longDmrtBuffer?.destroy();
				this.longDmrtBuffer = this.device.createBuffer({
					size: utciBytes,
					usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
				});
			}
		}

		if (!this.paramsBuffer || this.paramsBuffer.size !== 16) {
			this.paramsBuffer?.destroy();
			this.paramsBuffer = this.device.createBuffer({
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
		}

		this.queue.writeBuffer(this.paramsBuffer, 0, new Uint32Array([numPoints, totalTimeSteps, numHours, 0]));

		const workgroupSize = params.workgroupSize ?? 64;
		const workgroupsX = Math.ceil(numPoints / workgroupSize);
		this.ranExposurePassesThisRun = false;
		const encoder = this.device.createCommandEncoder();
		await this.encodeExposurePasses({
			encoder,
			numPoints,
			totalTimeSteps,
			workgroupSize,
			solarPipeline,
			skyPipeline
		});

		if (
			!this.solarExposureBuffer ||
			!this.skyExposureBuffer ||
			!this.weatherBuffer ||
			!this.utciBuffer ||
			!this.mrtBuffer ||
			!this.paramsBuffer ||
			!this.sunAltitudesBuffer
		) {
			throw new Error('WebGPU UTCI pipeline: MRT bindings are not initialized');
		}
		const solarExposureBytes = this.solarExposureBuffer.size;
		const skyExposureBytes = this.skyExposureBuffer.size;
		this.onDemandDiagnostics = mergeTrackedGpuAllocationBytes(
			{
				...this.onDemandDiagnostics,
				path: 'run-all-baseline',
				timeIndices: [],
				usedRunAllForSelectedHour: true,
				usedExposureOnlyPrecompute: false,
				allHoursUtciBytesAllocated: utciBytes,
				allHoursMrtBytesAllocated: mrtBytes,
				oneHourOutputBytes: 0
			},
			{
				persistentExposureBytes: solarExposureBytes + skyExposureBytes,
				allHoursOutputBytes: utciBytes + mrtBytes,
				selectedHourOutputBytes: 0
			}
		);
		const bindEntries: GPUBindGroupEntry[] = [
			{ binding: 0, resource: { buffer: this.solarExposureBuffer } },
			{ binding: 1, resource: { buffer: this.skyExposureBuffer } },
			{ binding: 2, resource: { buffer: this.weatherBuffer } },
			{ binding: 3, resource: { buffer: this.utciBuffer } },
			{ binding: 4, resource: { buffer: this.paramsBuffer } },
			{ binding: 5, resource: { buffer: this.sunAltitudesBuffer } },
			{ binding: 6, resource: { buffer: this.mrtBuffer } }
		];
		if (this.supportsMrtComponents) {
			if (!this.shortErfBuffer || !this.longErfBuffer || !this.shortDmrtBuffer || !this.longDmrtBuffer) {
				throw new Error('WebGPU UTCI pipeline: MRT component buffers are not initialized');
			}
			bindEntries.push(
				{ binding: 7, resource: { buffer: this.shortErfBuffer } },
				{ binding: 8, resource: { buffer: this.longErfBuffer } },
				{ binding: 9, resource: { buffer: this.shortDmrtBuffer } },
				{ binding: 10, resource: { buffer: this.longDmrtBuffer } }
			);
		}
		const bindGroup = this.device.createBindGroup({
			layout: mrtPipeline.getBindGroupLayout(0),
			entries: bindEntries
		});

		const pass = encoder.beginComputePass();
		pass.setPipeline(mrtPipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(workgroupsX, totalTimeSteps, 1);
		pass.end();

		this.queue.submit([encoder.finish()]);
		this.lastConfig = { numPoints, numHours, numMonths };
	}

	async runExposurePrecompute(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
	}): Promise<void> {
		if (!this.weatherData) {
			throw new Error('WebGPU UTCI pipeline: weather data not uploaded');
		}

		const { numPoints, numHours, numMonths } = params;
		const totalTimeSteps = numHours * numMonths;
		const [solarPipeline, skyPipeline] = await Promise.all([
			this.ensureSolarPipeline(),
			this.ensureSkyPipeline()
		]);
		await this.ensureWeatherBuffer();

		if (!this.paramsBuffer || this.paramsBuffer.size !== 16) {
			this.paramsBuffer?.destroy();
			this.paramsBuffer = this.device.createBuffer({
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
		}
		this.queue.writeBuffer(this.paramsBuffer, 0, new Uint32Array([numPoints, totalTimeSteps, numHours, 0]));

		this.ranExposurePassesThisRun = false;
		const encoder = this.device.createCommandEncoder();
		await this.encodeExposurePasses({
			encoder,
			numPoints,
			totalTimeSteps,
			workgroupSize: 64,
			solarPipeline,
			skyPipeline
		});
		const exposurePrecomputeStart = performance.now();
		this.queue.submit([encoder.finish()]);
		await this.queue.onSubmittedWorkDone();
		const solarExposureBytes = Math.ceil((numPoints * numHours * numMonths) / 32) * 4;
		const skyExposureBytes = numPoints * 4;
		this.onDemandDiagnostics = recordOnDemandTiming(
			mergeTrackedGpuAllocationBytes(
				{
					...this.onDemandDiagnostics,
					path: 'exposure-only-f32',
					timeIndices: [],
					usedRunAllForSelectedHour: false,
					usedExposureOnlyPrecompute: true,
					allHoursUtciBytesAllocated: 0,
					allHoursMrtBytesAllocated: 0,
					oneHourOutputBytes: 0
				},
				{
					persistentExposureBytes: solarExposureBytes + skyExposureBytes,
					allHoursOutputBytes: 0
				}
			),
			'exposurePrecomputeMs',
			performance.now() - exposurePrecomputeStart
		);
		this.lastConfig = { numPoints, numHours, numMonths };
	}

	async runUtciForTimeIndex(params: RunUtciForTimeIndexParams): Promise<OnDemandUtciOutput> {
		if (!this.solarExposureBuffer) {
			throw new Error(
				'WebGPU UTCI pipeline: solar exposure buffer not available (run runAll or runExposurePrecompute first)'
			);
		}
		if (!this.skyExposureBuffer) {
			throw new Error(
				'WebGPU UTCI pipeline: sky exposure buffer not available (run runAll or runExposurePrecompute first)'
			);
		}
		if (!this.sunAltitudesBuffer) {
			throw new Error(
				'WebGPU UTCI pipeline: sun altitude buffer not available (upload static data before on-demand UTCI runs)'
			);
		}
		if (!this.ranExposurePassesThisRun) {
			throw new Error(
				'WebGPU UTCI pipeline: solar/sky exposure passes did not run (no BVH?). runUtciForTimeIndex would use zero-filled exposure buffers.'
			);
		}

		const { format, numHours, numMonths, numPoints, timeIndex } = params;
		this.assertMatchesLastConfig({
			numPoints,
			numHours,
			numMonths,
			context: 'runUtciForTimeIndex'
		});
		if (format !== 'f32-utci') {
			throw new Error(`WebGPU UTCI pipeline: unsupported on-demand output format "${format}"`);
		}

		const totalTimeSteps = numHours * numMonths;
		if (timeIndex < 0 || timeIndex >= totalTimeSteps) {
			throw new Error(`Invalid time index ${timeIndex} for totalTimeSteps=${totalTimeSteps}`);
		}

		const solarExposureBuffer = this.solarExposureBuffer;
		const skyExposureBuffer = this.skyExposureBuffer;
		const sunAltitudesBuffer = this.sunAltitudesBuffer;
		const weatherBuffer = await this.ensureWeatherBuffer();
		const onDemandPipeline = await this.ensureOnDemandPipeline();
		const outputBytes = numPoints * 4;
		if (!this.onDemandOutputBuffer || this.onDemandOutputBuffer.size !== outputBytes) {
			this.onDemandOutputBuffer?.destroy();
			this.onDemandOutputBuffer = this.device.createBuffer({
				size: outputBytes,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
			});
		}
		if (!this.onDemandParamsBuffer || this.onDemandParamsBuffer.size !== 32) {
			this.onDemandParamsBuffer?.destroy();
			this.onDemandParamsBuffer = this.device.createBuffer({
				size: 32,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
		}
		const onDemandOutputBuffer = this.onDemandOutputBuffer;
		const onDemandParamsBuffer = this.onDemandParamsBuffer;

		this.queue.writeBuffer(
			onDemandParamsBuffer,
			0,
			new Uint32Array([numPoints, totalTimeSteps, numHours, timeIndex, 0, 0, 0, 0])
		);

		const bindGroup = this.device.createBindGroup({
			layout: onDemandPipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: solarExposureBuffer } },
				{ binding: 1, resource: { buffer: skyExposureBuffer } },
				{ binding: 2, resource: { buffer: weatherBuffer } },
				{ binding: 3, resource: { buffer: onDemandOutputBuffer } },
				{ binding: 4, resource: { buffer: onDemandParamsBuffer } },
				{ binding: 5, resource: { buffer: sunAltitudesBuffer } }
			]
		});

		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(onDemandPipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(Math.ceil(numPoints / 64), 1, 1);
		pass.end();

		const oneHourDispatchStart = performance.now();
		this.queue.submit([encoder.finish()]);
		await this.queue.onSubmittedWorkDone();
		const nextTimeIndices = this.onDemandDiagnostics.timeIndices.includes(timeIndex)
			? this.onDemandDiagnostics.timeIndices
			: [...this.onDemandDiagnostics.timeIndices, timeIndex];
		this.onDemandDiagnostics = recordOnDemandTiming(
			mergeTrackedGpuAllocationBytes(
				{
					...this.onDemandDiagnostics,
					path: 'exposure-only-f32',
					timeIndices: nextTimeIndices,
					usedRunAllForSelectedHour: false,
					usedExposureOnlyPrecompute: true,
					allHoursUtciBytesAllocated: 0,
					allHoursMrtBytesAllocated: 0,
					oneHourOutputBytes: outputBytes
				},
				{
					selectedHourOutputBytes: outputBytes,
					allHoursOutputBytes: 0
				}
			),
			'oneHourDispatchMs',
			performance.now() - oneHourDispatchStart
		);

		return {
			format,
			numPoints,
			timeIndex,
			gpuBuffer: onDemandOutputBuffer,
			debugLabel: 'webgpu-on-demand-f32-utci'
		};
	}

	async readOnDemandUtciForDebug(params: { numPoints: number }): Promise<Float32Array> {
		if (!this.onDemandOutputBuffer) {
			throw new Error('WebGPU UTCI pipeline: on-demand output buffer not available');
		}

		const previousReadback = this.onDemandReadbackSerial.catch(() => undefined);
		let finishReadback!: () => void;
		this.onDemandReadbackSerial = new Promise<void>((resolve) => {
			finishReadback = resolve;
		});
		await previousReadback;

		const { numPoints } = params;
		const bytes = numPoints * 4;
		try {
			if (!this.onDemandReadbackBuffer || this.onDemandReadbackBuffer.size !== bytes) {
				this.onDemandReadbackBuffer?.destroy();
				this.onDemandReadbackBuffer = this.device.createBuffer({
					size: bytes,
					usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
				});
			}

			const debugReadbackStart = performance.now();
			const encoder = this.device.createCommandEncoder();
			encoder.copyBufferToBuffer(this.onDemandOutputBuffer, 0, this.onDemandReadbackBuffer, 0, bytes);
			this.queue.submit([encoder.finish()]);

			await this.onDemandReadbackBuffer.mapAsync(GPUMapMode.READ);
			const mapped = new Float32Array(this.onDemandReadbackBuffer.getMappedRange());
			const out = new Float32Array(mapped.length);
			out.set(mapped);
			this.onDemandReadbackBuffer.unmap();
			this.onDemandDiagnostics = recordOnDemandTiming(
				{
					...this.onDemandDiagnostics,
					debugReadbackCount: this.onDemandDiagnostics.debugReadbackCount + 1
				},
				'debugReadbackMs',
				performance.now() - debugReadbackStart
			);
			return out;
		} finally {
			finishReadback();
		}
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
		this.assertMatchesLastConfig({
			numPoints,
			numHours,
			numMonths,
			context: 'readUtcisSlice'
		});
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

	async readUtciBulk(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
	}): Promise<Float32Array> {
		if (!this.utciBuffer || !this.lastConfig) {
			throw new Error('WebGPU UTCI pipeline: results buffer not available');
		}

		const { numPoints, numHours, numMonths } = params;
		this.assertMatchesLastConfig({
			numPoints,
			numHours,
			numMonths,
			context: 'readUtciBulk'
		});
		const totalElements = numPoints * numHours * numMonths;
		const totalBytes = totalElements * 4;

		// Reuse the main staging buffer if it's the right size
		if (!this.stagingBuffer || this.stagingBuffer.size !== totalBytes) {
			this.stagingBuffer?.destroy();
			this.stagingBuffer = this.device.createBuffer({
				size: totalBytes,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
		}

		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.utciBuffer, 0, this.stagingBuffer, 0, totalBytes);
		this.queue.submit([encoder.finish()]);

		await this.stagingBuffer.mapAsync(GPUMapMode.READ);
		const mapped = new Float32Array(this.stagingBuffer.getMappedRange());
		const out = new Float32Array(mapped.length);
		out.set(mapped);
		this.stagingBuffer.unmap();
		return out;
	}

	async readSolarExposureFull(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
	}): Promise<Float32Array> {
		if (!this.solarExposureBuffer || !this.lastConfig) {
			throw new Error(
				'WebGPU UTCI pipeline: solar exposure buffer not available (run runAll or runExposurePrecompute first)'
			);
		}
		if (!this.ranExposurePassesThisRun) {
			throw new Error(
				'WebGPU UTCI pipeline: solar/sky exposure passes did not run (no BVH?). readSolarExposureFull would return zeros.'
			);
		}
		await this.queue.onSubmittedWorkDone();
		const { numPoints, numHours, numMonths } = params;
		const totalTimeSteps = numHours * numMonths;
		const packedBytes = this.solarExposureBuffer.size;
		if (!this.solarStagingBuffer || this.solarStagingBuffer.size !== packedBytes) {
			this.solarStagingBuffer?.destroy();
			this.solarStagingBuffer = this.device.createBuffer({
				size: packedBytes,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
		}
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.solarExposureBuffer, 0, this.solarStagingBuffer, 0, packedBytes);
		this.queue.submit([encoder.finish()]);
		await this.solarStagingBuffer.mapAsync(GPUMapMode.READ);
		const packed = new Uint32Array(this.solarStagingBuffer.getMappedRange());

		// Unpack bits back to f32 (0.0 or 1.0) for parity comparison
		const totalElements = numPoints * totalTimeSteps;
		const out = new Float32Array(totalElements);
		for (let i = 0; i < totalElements; i++) {
			const wordIdx = Math.floor(i / 32);
			const bitIdx = i % 32;
			out[i] = (packed[wordIdx] >> bitIdx) & 1 ? 1.0 : 0.0;
		}

		this.solarStagingBuffer.unmap();
		return out;
	}

	async readSkyExposure(params: { numPoints: number }): Promise<Float32Array> {
		if (!this.skyExposureBuffer || !this.lastConfig) {
			throw new Error(
				'WebGPU UTCI pipeline: sky exposure buffer not available (run runAll or runExposurePrecompute first)'
			);
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

	async readMrtComponentsFull(params: {
		numPoints: number;
		numHours: number;
		numMonths: number;
	}): Promise<{
		shortErf: Float32Array;
		longErf: Float32Array;
		shortDmrt: Float32Array;
		longDmrt: Float32Array;
	}> {
		if (!this.supportsMrtComponents) {
			throw new Error(
				'WebGPU adapter/device does not support MRT component diagnostics (requires maxStorageBuffersPerShaderStage >= 10).'
			);
		}
		if (
			!this.shortErfBuffer ||
			!this.longErfBuffer ||
			!this.shortDmrtBuffer ||
			!this.longDmrtBuffer ||
			!this.lastConfig
		) {
			throw new Error('WebGPU UTCI pipeline: MRT component buffers not available (run runAll first)');
		}
		await this.queue.onSubmittedWorkDone();
		const { numPoints, numHours, numMonths } = params;
		const totalTimeSteps = numHours * numMonths;
		const bytes = numPoints * totalTimeSteps * 4;

		if (!this.shortErfStagingBuffer || this.shortErfStagingBuffer.size !== bytes) {
			this.shortErfStagingBuffer?.destroy();
			this.shortErfStagingBuffer = this.device.createBuffer({
				size: bytes,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
		}
		if (!this.longErfStagingBuffer || this.longErfStagingBuffer.size !== bytes) {
			this.longErfStagingBuffer?.destroy();
			this.longErfStagingBuffer = this.device.createBuffer({
				size: bytes,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
		}
		if (!this.shortDmrtStagingBuffer || this.shortDmrtStagingBuffer.size !== bytes) {
			this.shortDmrtStagingBuffer?.destroy();
			this.shortDmrtStagingBuffer = this.device.createBuffer({
				size: bytes,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
		}
		if (!this.longDmrtStagingBuffer || this.longDmrtStagingBuffer.size !== bytes) {
			this.longDmrtStagingBuffer?.destroy();
			this.longDmrtStagingBuffer = this.device.createBuffer({
				size: bytes,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
		}

		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.shortErfBuffer, 0, this.shortErfStagingBuffer, 0, bytes);
		encoder.copyBufferToBuffer(this.longErfBuffer, 0, this.longErfStagingBuffer, 0, bytes);
		encoder.copyBufferToBuffer(this.shortDmrtBuffer, 0, this.shortDmrtStagingBuffer, 0, bytes);
		encoder.copyBufferToBuffer(this.longDmrtBuffer, 0, this.longDmrtStagingBuffer, 0, bytes);
		this.queue.submit([encoder.finish()]);

		await this.shortErfStagingBuffer.mapAsync(GPUMapMode.READ);
		await this.longErfStagingBuffer.mapAsync(GPUMapMode.READ);
		await this.shortDmrtStagingBuffer.mapAsync(GPUMapMode.READ);
		await this.longDmrtStagingBuffer.mapAsync(GPUMapMode.READ);

		const shortErfMapped = new Float32Array(this.shortErfStagingBuffer.getMappedRange());
		const longErfMapped = new Float32Array(this.longErfStagingBuffer.getMappedRange());
		const shortDmrtMapped = new Float32Array(this.shortDmrtStagingBuffer.getMappedRange());
		const longDmrtMapped = new Float32Array(this.longDmrtStagingBuffer.getMappedRange());

		const shortErf = new Float32Array(shortErfMapped.length);
		shortErf.set(shortErfMapped);
		const longErf = new Float32Array(longErfMapped.length);
		longErf.set(longErfMapped);
		const shortDmrt = new Float32Array(shortDmrtMapped.length);
		shortDmrt.set(shortDmrtMapped);
		const longDmrt = new Float32Array(longDmrtMapped.length);
		longDmrt.set(longDmrtMapped);

		this.shortErfStagingBuffer.unmap();
		this.longErfStagingBuffer.unmap();
		this.shortDmrtStagingBuffer.unmap();
		this.longDmrtStagingBuffer.unmap();

		return { shortErf, longErf, shortDmrt, longDmrt };
	}

	supportsMrtComponentDiagnostics(): boolean {
		return this.supportsMrtComponents;
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
		this.onDemandOutputBuffer?.destroy();
		this.onDemandOutputBuffer = null;
		this.onDemandParamsBuffer?.destroy();
		this.onDemandParamsBuffer = null;
		this.onDemandReadbackBuffer?.destroy();
		this.onDemandReadbackBuffer = null;
		this.solarStagingBuffer?.destroy();
		this.solarStagingBuffer = null;
		this.skyStagingBuffer?.destroy();
		this.skyStagingBuffer = null;
		this.mrtBuffer?.destroy();
		this.mrtBuffer = null;
		this.mrtStagingBuffer?.destroy();
		this.mrtStagingBuffer = null;
		this.shortErfBuffer?.destroy();
		this.shortErfBuffer = null;
		this.longErfBuffer?.destroy();
		this.longErfBuffer = null;
		this.shortDmrtBuffer?.destroy();
		this.shortDmrtBuffer = null;
		this.longDmrtBuffer?.destroy();
		this.longDmrtBuffer = null;
		this.shortErfStagingBuffer?.destroy();
		this.shortErfStagingBuffer = null;
		this.longErfStagingBuffer?.destroy();
		this.longErfStagingBuffer = null;
		this.shortDmrtStagingBuffer?.destroy();
		this.shortDmrtStagingBuffer = null;
		this.longDmrtStagingBuffer?.destroy();
		this.longDmrtStagingBuffer = null;

		this.pipeline = null;
		this.solarPipeline = null;
		this.skyPipeline = null;
		this.gatherSlicePipeline = null;
		this.onDemandPipeline = null;
		this.pipelinePromise = null;
		this.solarPipelinePromise = null;
		this.skyPipelinePromise = null;
		this.gatherSlicePipelinePromise = null;
		this.onDemandPipelinePromise = null;
		this.lastConfig = null;
		this.weatherData = null;
		this.ranExposurePassesThisRun = false;
	}
}

async function getWebgpuDevice(enableDiagnostics: boolean): Promise<{ device: GPUDevice; supportsMrtComponents: boolean }> {
	if (typeof navigator === 'undefined' || !(navigator as any).gpu) {
		throw new Error('WebGPU not available in this environment');
	}

	const gpu = (navigator as any).gpu;
	let adapter = await gpu.requestAdapter();
	if (!adapter) {
		adapter = await gpu.requestAdapter({ forceFallbackAdapter: true });
	}
	if (!adapter) {
		throw new Error(
			`Failed to acquire WebGPU adapter (secureContext=${String(window.isSecureContext)}, userAgent=${navigator.userAgent})`
		);
	}
	const requiredStorageBuffersPerStage = 10;
	const supportedStorageBuffersPerStage = adapter.limits.maxStorageBuffersPerShaderStage;
	const supportsMrtComponents = enableDiagnostics && (supportedStorageBuffersPerStage >= requiredStorageBuffersPerStage);

	// Default limits are often 128MB binding / 256MB buffer; large models (e.g. Ness Tziona)
	// need solar/utci buffers >256MB. Request adapter max for both so CreateBuffer and
	// bindings succeed without grid coarsening.
	const maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
	const maxBufferSize = adapter.limits.maxBufferSize;

	const device: GPUDevice = supportsMrtComponents
		? await adapter.requestDevice({
				requiredLimits: {
					maxStorageBuffersPerShaderStage: requiredStorageBuffersPerStage,
					maxStorageBufferBindingSize,
					maxBufferSize
				}
			})
		: await adapter.requestDevice({
				requiredLimits: {
					maxStorageBufferBindingSize,
					maxBufferSize
				}
			});
	device.lost.then(() => {
		if (cachedDevicePromise && cachedDevice === device) {
			cachedDevicePromise = null;
			cachedDevice = null;
		}
	});
	return { device, supportsMrtComponents };
}

let cachedDevicePromise: Promise<GPUDevice> | null = null;
let cachedDevice: GPUDevice | null = null;
let cachedSupportsMrtComponents = false;

export async function createWebgpuUtciPipeline(options: { enableDiagnostics?: boolean } = {}): Promise<UTCIComputePipeline> {
	const enableDiagnostics = options.enableDiagnostics ?? false;

	if (!cachedDevicePromise || (enableDiagnostics && !cachedSupportsMrtComponents)) {
		cachedDevicePromise = getWebgpuDevice(enableDiagnostics).then(({ device, supportsMrtComponents }) => {
			cachedDevice = device;
			cachedSupportsMrtComponents = supportsMrtComponents;
			if (enableDiagnostics && !supportsMrtComponents) {
				console.warn("MRT diagnostics were requested but hardware does not support them.");
			}
			return device;
		});
	}
	const device = await cachedDevicePromise;
	const useDiagnostics = enableDiagnostics && cachedSupportsMrtComponents;
	return new WebgpuUtciComputePipeline(device, useDiagnostics);
}

export function __resetWebgpuDeviceCacheForTests(): void {
	cachedDevicePromise = null;
	cachedDevice = null;
	cachedSupportsMrtComponents = false;
}

export { WebgpuUtciComputePipeline as __TEST_ONLY_WebgpuUtciComputePipeline };
