/**
 * Client for running merge + BVH + grid in a Web Worker.
 * Use this to avoid main-thread freezes and crashes on large models.
 */

import type * as THREE from 'three';
import { emitComputeTelemetry } from '$lib/compute/telemetry';
import type { AnalysisBounds } from '$lib/compute/core/analysisGridFromBounds';
import { analysisBoundsToViewerRectangularBounds } from '$lib/compute/core/analysisGridFromBounds';

export interface MeshPayload {
	position: Float32Array;
	index: Uint32Array | null;
	matrixWorld: number[];
}

export interface MergeAndBvhResult {
	gridPoints: Float32Array;
	serializedBvh: {
		bvhNodeBuffer: ArrayBuffer;
		bvhIndexBuffer: ArrayBuffer;
		vertexBuffer: Float32Array;
		indexBuffer: Uint32Array;
	};
}

export interface PayloadPreflightEstimate {
	totalTriangles: number;
	meshCount: number;
	estimatedGridPoints: number;
	estimatedBytes: number;
	bounds: { min: [number, number, number]; max: [number, number, number] };
}

/** Above this triangle count we refuse to run on main thread and require the worker. */
export const MAX_TRIANGLES_FOR_MAIN_THREAD = 250_000;

/** Above this we refuse to run at all to prevent freeze/crash. */
export const MAX_TRIANGLES_HARD_CAP = 2_000_000;

/** Above this grid point count we reject to prevent runaway memory/time. */
export const MAX_GRID_POINTS_GUARD = 600_000;

/** Soft byte budget; reject before payload copy if estimate exceeds this.
 * 1.2 GB allows 2m grid for large models (e.g. Ness Tziona); device limits handle buffer size. */
export const DEFAULT_MAX_ESTIMATED_BYTES = 1280 * 1024 * 1024;

/** Copy typed arrays in chunks so large meshes never block one long task. */
const COPY_CHUNK_FLOATS = 100_000;
const COPY_CHUNK_UINTS = 100_000;

interface PreparePayloadOptions {
	signal?: AbortSignal;
	gridResolution?: number;
	numHours?: number;
	numMonths?: number;
	maxGridPoints?: number;
	maxEstimatedBytes?: number;
	hasWorkerSupport?: boolean;
	analysisBounds?: AnalysisBounds;
	coordinateSystem?: 'xy_ground' | 'xz_ground';
}

interface WorkerProgressMessage {
	type: 'progress';
	stage: 'prepare' | 'merge' | 'grid' | 'bvh' | 'transfer';
	ms?: number;
	numPoints?: number;
}

interface WorkerErrorMessage {
	type: 'error';
	error: string;
}

/**
 * Count triangles and meshes without copying buffers.
 */
export function countTrianglesInGroup(
	group: THREE.Group | THREE.Object3D
): { totalTriangles: number; meshCount: number } {
	let totalTriangles = 0;
	let meshCount = 0;
	group.traverse((child: THREE.Object3D) => {
		if (!(child as THREE.Mesh).isMesh || !(child as THREE.Mesh).geometry) return;
		const mesh = child as THREE.Mesh;
		const geom = mesh.geometry;
		const posAttr = geom.getAttribute('position');
		if (!posAttr) return;
		meshCount += 1;
		const idxAttr = geom.getIndex();
		totalTriangles += idxAttr ? idxAttr.count / 3 : posAttr.count / 3;
	});
	return { totalTriangles, meshCount };
}

function yieldToMain(): Promise<void> {
	return new Promise((resolve) => {
		if (typeof requestAnimationFrame !== 'undefined') {
			requestAnimationFrame(() => resolve());
		} else {
			setTimeout(() => resolve(), 0);
		}
	});
}

function accumulateBounds(target: { min: [number, number, number]; max: [number, number, number] }, box: THREE.Box3) {
	target.min[0] = Math.min(target.min[0], box.min.x);
	target.min[1] = Math.min(target.min[1], box.min.y);
	target.min[2] = Math.min(target.min[2], box.min.z);
	target.max[0] = Math.max(target.max[0], box.max.x);
	target.max[1] = Math.max(target.max[1], box.max.y);
	target.max[2] = Math.max(target.max[2], box.max.z);
}

function shouldIncludeMeshInComputeBvh(mesh: THREE.Mesh): boolean {
	return mesh.userData.includeInComputeBvh !== false;
}

function getTriangleCount(geom: THREE.BufferGeometry): number {
	const posAttr = geom.getAttribute('position');
	if (!posAttr) return 0;
	const idxAttr = geom.getIndex();
	return idxAttr ? idxAttr.count / 3 : posAttr.count / 3;
}

function createBoundsFromAnalysisMetadata(params: {
	analysisBounds: AnalysisBounds;
	coordinateSystem: 'xy_ground' | 'xz_ground';
}): { min: [number, number, number]; max: [number, number, number] } {
	const rectangularBounds = analysisBoundsToViewerRectangularBounds({
		bounds: params.analysisBounds,
		coordinateSystem: params.coordinateSystem
	});
	const y = params.analysisBounds.z ?? 0;
	return {
		min: [rectangularBounds.minX, y, rectangularBounds.minZ],
		max: [rectangularBounds.maxX, y, rectangularBounds.maxZ]
	};
}

function estimateBudget(params: {
	totalTriangles: number;
	estimatedGridPoints: number;
	numHours: number;
	numMonths: number;
}): number {
	const { totalTriangles, estimatedGridPoints, numHours, numMonths } = params;
	const totalHours = Math.max(1, numHours * numMonths);

	const meshBytes = totalTriangles * (3 * 4 * 3 + 3 * 4);
	const workerMergeAndBvhBytes = meshBytes * 1.5;
	const gridBytes = estimatedGridPoints * 3 * 4;
	const solarBytes = estimatedGridPoints * totalHours * 4;
	const skyBytes = estimatedGridPoints * 4;
	// UTCI stored as Int16 (scale 100), not Float32 - halves this portion
	const utciBytes = estimatedGridPoints * totalHours * 2;
	const readbackAndStatsBytes = estimatedGridPoints * (4 + 12);

	return Math.ceil(
		meshBytes +
			workerMergeAndBvhBytes +
			gridBytes +
			solarBytes +
			skyBytes +
			utciBytes +
			readbackAndStatsBytes
	);
}

async function copyWithYields<T extends Float32Array | Uint32Array>(
	source: T,
	ctor: new (length: number) => T,
	chunkSize: number,
	signal?: AbortSignal
): Promise<T> {
	const result = new ctor(source.length) as T;
	for (let offset = 0; offset < source.length; offset += chunkSize) {
		if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
		const end = Math.min(offset + chunkSize, source.length);
		result.set(source.subarray(offset, end), offset);
		if (end < source.length) await yieldToMain();
	}
	return result;
}

/**
 * Single-pass scan + preflight + payload copy.
 * We traverse once to gather mesh metadata, bounds, and triangle count, reject
 * unsafe runs before any large typed-array copy, then copy payload incrementally.
 */
export async function prepareMeshPayloadForWorkerAsync(
	group: THREE.Group | THREE.Object3D,
	options: PreparePayloadOptions = {}
): Promise<{ meshes: MeshPayload[]; totalTriangles: number; preflight: PayloadPreflightEstimate }> {
	const {
		signal,
		gridResolution = 2,
		numHours = 24,
		numMonths = 1,
		maxGridPoints = MAX_GRID_POINTS_GUARD,
		maxEstimatedBytes = DEFAULT_MAX_ESTIMATED_BYTES,
		hasWorkerSupport = typeof Worker !== 'undefined',
		analysisBounds,
		coordinateSystem = 'xy_ground'
	} = options;
	const t0 = performance.now();
	await yieldToMain();
	if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

	group.updateMatrixWorld(true);

	const bounds = {
		min: [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY] as [number, number, number],
		max: [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY] as [number, number, number]
	};

	let totalTriangles = 0;
	const meshList: Array<{ mesh: THREE.Mesh; geom: THREE.BufferGeometry }> = [];

	group.traverse((child: THREE.Object3D) => {
		if (!(child as THREE.Mesh).isMesh || !(child as THREE.Mesh).geometry) return;
		const mesh = child as THREE.Mesh;
		if (!shouldIncludeMeshInComputeBvh(mesh)) return;
		const geom = mesh.geometry;
		const posAttr = geom.getAttribute('position');
		if (!posAttr) return;

		meshList.push({ mesh, geom });
		totalTriangles += getTriangleCount(geom);

		if (!analysisBounds) {
			if (!geom.boundingBox) {
				geom.computeBoundingBox();
			}
			if (geom.boundingBox) {
				const worldBox = geom.boundingBox.clone().applyMatrix4(mesh.matrixWorld);
				accumulateBounds(bounds, worldBox);
			}
		}
	});

	if (meshList.length === 0) {
		throw new Error('No mesh geometry found for worker payload');
	}

	if (totalTriangles > MAX_TRIANGLES_HARD_CAP) {
		throw new Error(
			`Model too large (${(totalTriangles / 1e6).toFixed(1)}M triangles). Maximum ${(MAX_TRIANGLES_HARD_CAP / 1e6).toFixed(0)}M triangles supported.`
		);
	}
	if (!hasWorkerSupport && totalTriangles > MAX_TRIANGLES_FOR_MAIN_THREAD) {
		throw new Error(
			`Model too large (${(totalTriangles / 1e6).toFixed(1)}M triangles). Web Workers are required.`
		);
	}

	const preflightBounds = analysisBounds
		? createBoundsFromAnalysisMetadata({ analysisBounds, coordinateSystem })
		: bounds;

	const width = Math.max(0, preflightBounds.max[0] - preflightBounds.min[0]);
	const depth = Math.max(0, preflightBounds.max[2] - preflightBounds.min[2]);
	const estimatedGridPoints = Math.max(1, Math.ceil((width * depth) / Math.max(0.25, gridResolution * gridResolution)));
	if (estimatedGridPoints > maxGridPoints) {
		throw new Error(
			`Estimated grid too dense (${estimatedGridPoints.toLocaleString()} points) exceeds safety cap (${maxGridPoints.toLocaleString()}). Increase grid size.`
		);
	}

	const estimatedBytes = estimateBudget({
		totalTriangles,
		estimatedGridPoints,
		numHours,
		numMonths
	});
	if (estimatedBytes > maxEstimatedBytes) {
		throw new Error(
			`Estimated memory ${(estimatedBytes / (1024 * 1024)).toFixed(1)} MB exceeds budget ${(maxEstimatedBytes / (1024 * 1024)).toFixed(1)} MB.`
		);
	}

	const meshes: MeshPayload[] = [];
	for (let i = 0; i < meshList.length; i++) {
		if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
		const { mesh, geom } = meshList[i];
		const posAttr = geom.getAttribute('position');
		if (!posAttr) continue;

		const srcPos = posAttr.array as Float32Array;
		const position =
			srcPos.length <= COPY_CHUNK_FLOATS
				? (() => {
						const arr = new Float32Array(srcPos.length);
						arr.set(srcPos);
						return arr;
					})()
				: await copyWithYields(srcPos, Float32Array, COPY_CHUNK_FLOATS, signal);

		const idxAttr = geom.getIndex();
		let index: Uint32Array | null = null;
		if (idxAttr) {
			const srcIdx = idxAttr.array as Uint32Array;
			const len = idxAttr.count;
			index =
				len <= COPY_CHUNK_UINTS
					? (() => {
							const arr = new Uint32Array(len);
							arr.set(srcIdx, 0);
							return arr;
						})()
					: await copyWithYields(srcIdx, Uint32Array, COPY_CHUNK_UINTS, signal);
		}

		meshes.push({
			position,
			index,
			matrixWorld: mesh.matrixWorld.toArray()
		});

		if (i < meshList.length - 1) await yieldToMain();
	}

	const preflight: PayloadPreflightEstimate = {
		totalTriangles,
		meshCount: meshList.length,
		estimatedGridPoints,
		estimatedBytes,
		bounds: preflightBounds
	};

	emitComputeTelemetry('payload.prepare.done', {
		ms: performance.now() - t0,
		data: {
			meshCount: meshList.length,
			totalTriangles,
			estimatedGridPoints,
			estimatedBytes
		}
	});

	return { meshes, totalTriangles, preflight };
}

/**
 * Synchronous payload preparation. Avoid for large models.
 */
export function prepareMeshPayloadForWorker(
	group: THREE.Group | THREE.Object3D
): { meshes: MeshPayload[]; totalTriangles: number } {
	group.updateMatrixWorld(true);

	const meshes: MeshPayload[] = [];
	let totalTriangles = 0;

	group.traverse((child: THREE.Object3D) => {
		if (!(child as THREE.Mesh).isMesh || !(child as THREE.Mesh).geometry) return;
		const mesh = child as THREE.Mesh;
		if (!shouldIncludeMeshInComputeBvh(mesh)) return;
		const geom = mesh.geometry;
		const posAttr = geom.getAttribute('position');
		if (!posAttr) return;

		const position = new Float32Array(posAttr.array.length);
		position.set(posAttr.array as Float32Array);

		const idxAttr = geom.getIndex();
		let index: Uint32Array | null = null;
		if (idxAttr) {
			const arr = idxAttr.array;
			const len = idxAttr.count;
			index = new Uint32Array(len);
			index.set(arr as Uint32Array, 0);
			totalTriangles += len / 3;
		} else {
			totalTriangles += posAttr.count / 3;
		}

		meshes.push({ position, index, matrixWorld: mesh.matrixWorld.toArray() });
	});

	return { meshes, totalTriangles };
}

const WORKER_TIMEOUT_MS = 10 * 60 * 1000;

/**
 * Run merge + BVH + grid in a Web Worker.
 */
export function runMergeAndBvhInWorker(params: {
	meshes: MeshPayload[];
	gridResolution: number;
	zHeight: number;
	maxSlopeDegrees?: number;
	signal?: AbortSignal;
	maxGridPoints?: number;
	/** When true, worker returns only serializedBvh and empty gridPoints (for parity mode). */
	bvhOnly?: boolean;
}): Promise<MergeAndBvhResult> {
	const {
		meshes,
		gridResolution,
		zHeight,
		maxSlopeDegrees = 45,
		signal,
		maxGridPoints = MAX_GRID_POINTS_GUARD,
		bvhOnly = false
	} = params;

	return new Promise((resolve, reject) => {
		const startedAt = performance.now();
		if (typeof Worker === 'undefined') {
			reject(new Error('Web Workers not supported'));
			return;
		}

		const worker = new Worker(new URL('./mergeAndBvh.worker.ts', import.meta.url), {
			type: 'module'
		});

		const timeoutId = setTimeout(() => {
			worker.terminate();
			reject(new Error('Merge + BVH worker timed out (model may be too large)'));
		}, WORKER_TIMEOUT_MS);

		const cleanup = () => {
			clearTimeout(timeoutId);
			worker.terminate();
		};

		signal?.addEventListener(
			'abort',
			() => {
				try {
					worker.postMessage({ type: 'cancel' });
				} finally {
					cleanup();
					reject(new DOMException('Aborted', 'AbortError'));
				}
			},
			{ once: true }
		);

		worker.onmessage = (
			e: MessageEvent<MergeAndBvhResult | WorkerErrorMessage | WorkerProgressMessage>
		) => {
			const data = e.data;
			if (data && (data as WorkerProgressMessage).type === 'progress') {
				const p = data as WorkerProgressMessage;
				emitComputeTelemetry(`worker.${p.stage}`, {
					...(typeof p.ms === 'number' ? { ms: p.ms } : {}),
					data: { ...(typeof p.numPoints === 'number' ? { numPoints: p.numPoints } : {}) }
				});
				return;
			}
			cleanup();
			if (data && (data as WorkerErrorMessage).type === 'error') {
				reject(new Error((data as WorkerErrorMessage).error));
				return;
			}
			emitComputeTelemetry('worker.complete', { ms: performance.now() - startedAt });
			resolve(data as MergeAndBvhResult);
		};

		worker.onerror = (err) => {
			cleanup();
			reject(new Error(err.message || 'Merge + BVH worker failed'));
		};

		// Clone payload so each buffer is unique: postMessage rejects if the same ArrayBuffer
		// appears in the message (or transfer list) more than once.
		const seen = new Set<ArrayBufferLike>();
		const meshesToSend: MeshPayload[] = [];
		const transferList: Transferable[] = [];
		for (const m of meshes) {
			const position = new Float32Array(m.position.length);
			position.set(m.position);
			const index = m.index ? new Uint32Array(m.index) : null;
			meshesToSend.push({
				position,
				index,
				matrixWorld: m.matrixWorld
			});
			if (!seen.has(position.buffer)) {
				seen.add(position.buffer);
				transferList.push(position.buffer);
			}
			if (index && !seen.has(index.buffer)) {
				seen.add(index.buffer);
				transferList.push(index.buffer);
			}
		}

		worker.postMessage(
			{ type: 'start', meshes: meshesToSend, gridResolution, zHeight, maxSlopeDegrees, maxGridPoints, bvhOnly },
			transferList
		);
	});
}
