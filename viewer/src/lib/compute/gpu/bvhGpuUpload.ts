import * as THREE from 'three';
import { MeshBVH } from 'three-mesh-bvh';

/**
 * Node stride in bytes for the serialized BVH layout (three-mesh-bvh Constants.BYTES_PER_NODE).
 * WGSL traversal must use the same layout:
 * - Offset 0-11:  bounds min (3 × f32)
 * - Offset 12-23: bounds max (3 × f32)
 * - Offset 24-27: uint32 at node+6 — internal: relative offset to right child (in nodes); leaf: triangle offset into index buffer
 * - Offset 28-29: uint16 at node+14 — triangle count (leaf)
 * - Offset 30-31: uint16 at node+15 — 0xFFFF = leaf, else internal
 */
export const BVH_NODE_STRIDE_BYTES = 32;

/** Byte offsets into each 32-byte node for WGSL (node index × 32 + offset). */
export const BVH_LAYOUT = {
	/** Stride in bytes. */
	BYTES_PER_NODE: 32,
	/** Bounds min: 3 × f32 at byte 0. */
	OFFSET_BOUNDS_MIN: 0,
	/** Bounds max: 3 × f32 at byte 12. */
	OFFSET_BOUNDS_MAX: 12,
	/** Metadata: uint32 at byte 24 = right child relative offset (internal) or triangle offset (leaf). */
	OFFSET_METADATA_U32: 24,
	/** uint16 at byte 28 = triangle count (leaf). */
	OFFSET_COUNT_U16: 28,
	/** uint16 at byte 30 = 0xFFFF if leaf. */
	OFFSET_LEAF_FLAG_U16: 30,
	LEAF_FLAG: 0xffff
} as const;

export interface SerializedBvhForGpu {
	/** Raw node data (32 bytes per node). Upload to GPU as storage buffer. */
	bvhNodeBuffer: ArrayBuffer;
	/** Triangle indices (3 per triangle). Same as geometry index; upload to GPU. */
	bvhIndexBuffer: ArrayBuffer;
	/** Vertex positions (3 floats per vertex). From geometry position attribute. */
	vertexBuffer: Float32Array;
	/** Same as bvhIndexBuffer, typed for convenience. */
	indexBuffer: Uint32Array;
}

export interface SerializeBvhOptions {
	/**
	 * When true, try to reuse existing typed-array buffers to reduce peak memory.
	 * Intended for worker pipelines that immediately transfer ownership to main.
	 */
	zeroCopy?: boolean;
}

/**
 * Build a MeshBVH from the given mesh or geometry, serialize it to buffers
 * suitable for GPU upload, and return node buffer, index buffer, and
 * vertex buffer for WGSL BVH traversal and ray–triangle intersection.
 *
 * If the geometry already has a boundsTree (e.g. from computeBoundsTree() in
 * grid-generator), that tree is serialized so the BVH is built only once.
 *
 * When used in a worker: the returned vertexBuffer/indexBuffer buffers may be
 * transferred to the main thread via postMessage transfer list; do not use
 * them in the worker after transfer (they become detached).
 */
export function serializeBvhForGpu(
	meshOrGeometry: THREE.Mesh | THREE.BufferGeometry,
	options: SerializeBvhOptions = {}
): SerializedBvhForGpu {
	const { zeroCopy = false } = options;
	const geometry = meshOrGeometry instanceof THREE.Mesh ? meshOrGeometry.geometry : meshOrGeometry;
	if (!geometry.isBufferGeometry) {
		throw new Error('bvhGpuUpload: only BufferGeometry is supported');
	}

	const bvh = (geometry as THREE.BufferGeometry & { boundsTree?: InstanceType<typeof MeshBVH> }).boundsTree ?? new MeshBVH(geometry);
	const serialized = MeshBVH.serialize(bvh, { cloneBuffers: !zeroCopy });

	if (!serialized.roots || serialized.roots.length === 0) {
		throw new Error('bvhGpuUpload: serialized BVH has no roots');
	}

	const rootBuffer = serialized.roots[0];
	const nodeArray = new Uint8Array(rootBuffer as ArrayBuffer);
	const bvhNodeBuffer =
		zeroCopy && nodeArray.byteOffset === 0 && nodeArray.byteLength === nodeArray.buffer.byteLength
			? (nodeArray.buffer as ArrayBuffer)
			: (nodeArray.buffer.slice(nodeArray.byteOffset, nodeArray.byteOffset + nodeArray.byteLength) as ArrayBuffer);

	const indexAttr = geometry.getIndex();
	const indexArray = serialized.index ?? (indexAttr ? indexAttr.array : null);
	if (!indexArray) {
		throw new Error('bvhGpuUpload: geometry has no index (non-indexed geometry not supported for GPU BVH)');
	}
	const indexBuffer =
		indexArray instanceof Uint32Array
			? zeroCopy
				? indexArray
				: indexArray.slice()
			: new Uint32Array(Array.from(indexArray as unknown as ArrayLike<number>));
	const bvhIndexBuffer =
		zeroCopy && indexBuffer.byteOffset === 0 && indexBuffer.byteLength === indexBuffer.buffer.byteLength
			? (indexBuffer.buffer as ArrayBuffer)
			: (indexBuffer.buffer.slice(indexBuffer.byteOffset, indexBuffer.byteOffset + indexBuffer.byteLength) as ArrayBuffer);

	const posAttr = geometry.getAttribute('position');
	if (!posAttr) {
		throw new Error('bvhGpuUpload: geometry has no position attribute');
	}
	const vertexBuffer =
		posAttr.array instanceof Float32Array
			? zeroCopy
				? posAttr.array
				: posAttr.array.slice()
			: new Float32Array(posAttr.array);

	return {
		bvhNodeBuffer,
		bvhIndexBuffer,
		vertexBuffer,
		indexBuffer
	};
}
