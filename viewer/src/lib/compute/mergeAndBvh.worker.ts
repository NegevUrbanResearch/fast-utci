/**
 * Web Worker: merge scene meshes, build BVH, generate grid.
 * Runs off the main thread to prevent UI freezes on large models.
 */

import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import { computeBoundsTree, acceleratedRaycast } from 'three-mesh-bvh';
import { serializeBvhForGpu } from './bvhGpuUpload';
import { generateGridFromMesh } from './grid-generator';
import { MAX_GRID_POINTS_GUARD } from './mergeAndBvhWorkerClient';

THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

export interface MeshPayload {
	position: Float32Array;
	index: Uint32Array | null;
	matrixWorld: number[];
}

interface StartRequest {
	type: 'start';
	meshes: MeshPayload[];
	gridResolution: number;
	zHeight: number;
	maxSlopeDegrees?: number;
	maxGridPoints?: number;
}

interface CancelRequest {
	type: 'cancel';
}

type MergeAndBvhRequest = StartRequest | CancelRequest;

export interface MergeAndBvhResult {
	gridPoints: Float32Array;
	serializedBvh: {
		bvhNodeBuffer: ArrayBuffer;
		bvhIndexBuffer: ArrayBuffer;
		vertexBuffer: Float32Array;
		indexBuffer: Uint32Array;
	};
}

const workerScope = self as unknown as {
	postMessage: (message: unknown, transfer?: Transferable[]) => void;
};

let cancelled = false;

function postProgress(stage: 'prepare' | 'merge' | 'grid' | 'bvh' | 'transfer', ms?: number, numPoints?: number) {
	workerScope.postMessage({ type: 'progress', stage, ...(typeof ms === 'number' ? { ms } : {}), ...(typeof numPoints === 'number' ? { numPoints } : {}) });
}

function ensureNotCancelled() {
	if (cancelled) throw new DOMException('Aborted', 'AbortError');
}

function yieldWorker(): Promise<void> {
	return new Promise((r) => setTimeout(r, 0));
}

function buildGeometryFromPayload(p: MeshPayload): THREE.BufferGeometry {
	const geom = new THREE.BufferGeometry();
	geom.setAttribute('position', new THREE.BufferAttribute(p.position, 3));
	if (p.index && p.index.length > 0) {
		geom.setIndex(new THREE.BufferAttribute(p.index, 1));
	} else {
		const count = p.position.length / 3;
		const indices = new Uint32Array(count);
		for (let i = 0; i < count; i++) indices[i] = i;
		geom.setIndex(new THREE.BufferAttribute(indices, 1));
	}
	const matrix = new THREE.Matrix4().fromArray(p.matrixWorld);
	geom.applyMatrix4(matrix);
	return geom;
}

async function handleStart(req: StartRequest) {
	const t0 = performance.now();
	const { meshes, gridResolution, zHeight, maxSlopeDegrees = 45, maxGridPoints = MAX_GRID_POINTS_GUARD } = req;
	if (!meshes?.length) {
		workerScope.postMessage({ type: 'error', error: 'No meshes' });
		return;
	}

	cancelled = false;
	postProgress('prepare');

	const geometries: THREE.BufferGeometry[] = [];
	for (let i = 0; i < meshes.length; i++) {
		ensureNotCancelled();
		geometries.push(buildGeometryFromPayload(meshes[i]));
		if (i > 0 && i % 16 === 0) {
			await yieldWorker();
		}
	}

	const mergeStart = performance.now();
	const merged = BufferGeometryUtils.mergeGeometries(geometries, false);
	if (!merged) {
		workerScope.postMessage({ type: 'error', error: 'Merge failed' });
		return;
	}
	postProgress('merge', performance.now() - mergeStart);

	await yieldWorker();
	ensureNotCancelled();

	const gridStart = performance.now();
	const mesh = new THREE.Mesh(merged);
	mesh.matrixWorld.identity();
	const grid = generateGridFromMesh(mesh, gridResolution, zHeight, maxSlopeDegrees);
	if (grid.points.length > maxGridPoints) {
		workerScope.postMessage({
			type: 'error',
			error: `Grid too dense (${grid.points.length.toLocaleString()} points) exceeds safety cap (${maxGridPoints.toLocaleString()})`
		});
		return;
	}
	postProgress('grid', performance.now() - gridStart, grid.points.length);

	await yieldWorker();
	ensureNotCancelled();

	const numPoints = grid.points.length;
	const gridPoints = new Float32Array(numPoints * 3);
	for (let i = 0; i < numPoints; i++) {
		ensureNotCancelled();
		gridPoints[i * 3] = grid.points[i].x;
		gridPoints[i * 3 + 1] = grid.points[i].y;
		gridPoints[i * 3 + 2] = grid.points[i].z;
		if (i > 0 && i % 50_000 === 0) {
			await yieldWorker();
		}
	}

	const bvhStart = performance.now();
	const serialized = serializeBvhForGpu(merged, { zeroCopy: true });
	postProgress('bvh', performance.now() - bvhStart);

	await yieldWorker();
	ensureNotCancelled();

	postProgress('transfer', performance.now() - t0, numPoints);
	workerScope.postMessage(
		{
			gridPoints,
			serializedBvh: {
				bvhNodeBuffer: serialized.bvhNodeBuffer,
				bvhIndexBuffer: serialized.bvhIndexBuffer,
				vertexBuffer: serialized.vertexBuffer,
				indexBuffer: serialized.indexBuffer
			}
		},
		[
			gridPoints.buffer,
			serialized.bvhNodeBuffer,
			serialized.bvhIndexBuffer,
			serialized.vertexBuffer.buffer,
			serialized.indexBuffer.buffer
		]
	);
}

self.onmessage = async (e: MessageEvent<MergeAndBvhRequest>) => {
	try {
		const msg = e.data;
		if (!msg || typeof msg !== 'object') return;
		if (msg.type === 'cancel') {
			cancelled = true;
			return;
		}
		await handleStart(msg);
	} catch (err) {
		if (err instanceof DOMException && err.name === 'AbortError') {
			workerScope.postMessage({ type: 'error', error: 'Aborted' });
			return;
		}
		workerScope.postMessage({
			type: 'error',
			error: err instanceof Error ? err.message : String(err)
		});
	}
};
