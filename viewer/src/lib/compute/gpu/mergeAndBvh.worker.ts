/**
 * Web Worker: merge scene meshes and build BVH for exposure.
 * Grid is built from analysis bounds on the main thread; this worker only produces the BVH.
 */

import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import { computeBoundsTree, acceleratedRaycast } from 'three-mesh-bvh';
import { serializeBvhForGpu } from './bvhGpuUpload';

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
	gridResolution?: number;
	zHeight?: number;
	maxSlopeDegrees?: number;
	maxGridPoints?: number;
	/** @deprecated Grid is always built from bounds; worker only returns BVH. */
	bvhOnly?: boolean;
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

function uniqueTransferList(buffers: ArrayBuffer[]): Transferable[] {
	const seen = new Set<ArrayBuffer>();
	const transfer: Transferable[] = [];
	for (const buffer of buffers) {
		if (seen.has(buffer)) continue;
		seen.add(buffer);
		transfer.push(buffer);
	}
	return transfer;
}

function ensureOwnedFloat32Array(view: Float32Array): Float32Array {
	const { buffer, byteOffset, byteLength } = view;
	if (
		buffer instanceof ArrayBuffer &&
		byteOffset === 0 &&
		byteLength === buffer.byteLength
	) {
		return view;
	}
	return new Float32Array(new Uint8Array(buffer, byteOffset, byteLength).slice().buffer);
}

function ensureOwnedUint32Array(view: Uint32Array): Uint32Array {
	const { buffer, byteOffset, byteLength } = view;
	if (
		buffer instanceof ArrayBuffer &&
		byteOffset === 0 &&
		byteLength === buffer.byteLength
	) {
		return view;
	}
	return new Uint32Array(new Uint8Array(buffer, byteOffset, byteLength).slice().buffer);
}

function ownedTransferBuffer(view: ArrayBufferView): ArrayBuffer {
	const { buffer } = view;
	if (buffer instanceof ArrayBuffer) {
		return buffer;
	}
	return new Uint8Array(buffer).slice().buffer;
}

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
	const { meshes } = req;
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

	postProgress('grid', 0, 0);

	const bvhStart = performance.now();
	const serialized = serializeBvhForGpu(merged, { zeroCopy: true });
	postProgress('bvh', performance.now() - bvhStart);

	await yieldWorker();
	ensureNotCancelled();

	const gridPoints = new Float32Array(0);
	postProgress('transfer', performance.now() - t0, 0);
	const vertexBuffer = ensureOwnedFloat32Array(serialized.vertexBuffer);
	const indexBuffer = ensureOwnedUint32Array(serialized.indexBuffer);
	const transferList = uniqueTransferList([
		serialized.bvhNodeBuffer,
		serialized.bvhIndexBuffer,
		ownedTransferBuffer(vertexBuffer),
		ownedTransferBuffer(indexBuffer)
	]);
	workerScope.postMessage(
		{
			gridPoints,
			serializedBvh: {
				bvhNodeBuffer: serialized.bvhNodeBuffer,
				bvhIndexBuffer: serialized.bvhIndexBuffer,
				vertexBuffer,
				indexBuffer
			}
		},
		transferList
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
