import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { serializeBvhForGpu } from '$lib/compute/gpu/bvhGpuUpload';
import { createWebgpuUtciPipeline } from '$lib/compute/gpu/webgpuUtciPipeline';
import { getTregenzaDome } from '$lib/compute/core/tregenza';
import { MeshBVH } from 'three-mesh-bvh';

const HAS_WEBGPU = typeof navigator !== 'undefined' && (navigator as any).gpu;

describe('bvhGpuUpload', () => {
	it('should serialize a BVH from a tiny geometry and expose nodes and index', () => {
		// Single triangle
		const geometry = new THREE.BufferGeometry();
		const pos = new Float32Array([0, 0, 0, 1, 0, 0, 0.5, 1, 0]);
		geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
		geometry.setIndex([0, 1, 2]);
		geometry.computeBoundingSphere();

		const serialized = serializeBvhForGpu(geometry);

		expect(serialized).toBeDefined();
		expect(serialized.bvhNodeBuffer).toBeDefined();
		expect(serialized.bvhIndexBuffer).toBeDefined();
		expect(serialized.vertexBuffer).toBeDefined();
		expect(serialized.indexBuffer).toBeDefined();

		// three-mesh-bvh uses 32 bytes per node (BYTES_PER_NODE = 32)
		expect(serialized.bvhNodeBuffer.byteLength % 32).toBe(0);
		expect(serialized.bvhNodeBuffer.byteLength).toBeGreaterThanOrEqual(32);
	});

	it('should produce index buffer matching geometry index type', () => {
		const geometry = new THREE.BoxGeometry(1, 1, 1);
		const serialized = serializeBvhForGpu(geometry);

		expect(serialized.indexBuffer).toBeInstanceOf(Uint32Array);
		expect(serialized.indexBuffer.length).toBeGreaterThan(0);
	});

	it('should produce vertex buffer as Float32Array with 3 components per vertex', () => {
		const geometry = new THREE.BoxGeometry(1, 1, 1);
		const serialized = serializeBvhForGpu(geometry);

		expect(serialized.vertexBuffer).toBeInstanceOf(Float32Array);
		expect(serialized.vertexBuffer.length % 3).toBe(0);
	});

	it('should reuse existing boundsTree when present', () => {
		const geometry = new THREE.BoxGeometry(1, 1, 1);
		const prebuilt = new MeshBVH(geometry);
		(geometry as THREE.BufferGeometry & { boundsTree?: MeshBVH }).boundsTree = prebuilt;
		const serializeSpy = vi.spyOn(MeshBVH, 'serialize');

		serializeBvhForGpu(geometry);

		expect(serializeSpy).toHaveBeenCalled();
		expect(serializeSpy.mock.calls[0][0]).toBe(prebuilt);
		serializeSpy.mockRestore();
	});

	it('should return detached-safe copies of geometry attributes', () => {
		const geometry = new THREE.BoxGeometry(1, 1, 1);
		const serialized = serializeBvhForGpu(geometry);
		const pos = geometry.getAttribute('position').array as Float32Array;
		const idx = geometry.getIndex()!.array as Uint16Array | Uint32Array;

		// Mutate source arrays; serialized copies should not change.
		const originalVertex0 = serialized.vertexBuffer[0];
		const originalIndex0 = serialized.indexBuffer[0];
		pos[0] = pos[0] + 1234;
		idx[0] = idx[0] + 7;

		expect(serialized.vertexBuffer[0]).toBe(originalVertex0);
		expect(serialized.indexBuffer[0]).toBe(originalIndex0);
	});

	it('should allow zero-copy mode for worker transfer paths', () => {
		const geometry = new THREE.BoxGeometry(1, 1, 1);
		const serialized = serializeBvhForGpu(geometry, { zeroCopy: true });
		const pos = geometry.getAttribute('position').array as Float32Array;

		// In zero-copy mode, vertex buffer can alias source geometry memory.
		expect(serialized.vertexBuffer.buffer).toBe(pos.buffer);
		expect(serialized.vertexBuffer.length).toBe(pos.length);
	});
});

describe('BVH GPU raycast (WebGPU)', () => {
	(HAS_WEBGPU ? it : it.skip)(
		'should produce different UTCI for point in sun vs in shade via GPU BVH raycast',
		async () => {
			// Box 2Ã—2Ã—2 centered at origin (Y-up). Grid: point 0 above (0,2,0), point 1 below (0,-2,0).
			// Sun direction (0,1,0) = up. Ray from above goes up â†’ no hit â†’ exposed â†’ higher UTCI.
			// Ray from below goes up â†’ hits box â†’ shaded â†’ lower UTCI.
			const box = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2));
			box.position.set(0, 0, 0);
			box.updateMatrixWorld(true);

			const numPoints = 2;
			const numMonths = 1;
			const numHours = 1;
			const gridPoints = new Float32Array([
				0, 2, 0, // above box
				0, -2, 0 // below box
			]);
			const sunVectors = new Float32Array([0, 1, 0]); // up
			const weather = new Float32Array(7);
			weather[0] = 30; // air_temp
			weather[1] = 30; // mrt_longwave approx
			weather[2] = 1.0; // wind_speed
			weather[3] = 50; // rel_humidity
			weather[4] = 800; // direct_normal
			weather[5] = 200; // diffuse_horizontal
			weather[6] = 400; // horiz_infrared

			const dome = getTregenzaDome();
			const domeVectors = new Float32Array(dome.vectors.length * 3);
			for (let i = 0; i < dome.vectors.length; i++) {
				domeVectors[i * 3] = dome.vectors[i][0];
				domeVectors[i * 3 + 1] = dome.vectors[i][2]; // Z-up â†’ Y-up
				domeVectors[i * 3 + 2] = dome.vectors[i][1];
			}
			const domeWeights = new Float32Array(dome.weights);

			const pipeline = await createWebgpuUtciPipeline();
			await pipeline.uploadStaticData({
				gridPoints,
				sunVectors,
				weather,
				mesh: box,
				domeVectors,
				domeWeights
			});
			await pipeline.runAll({ numPoints, numHours, numMonths });

			const utciSlice = await pipeline.readUtcisSlice({
				monthIndex: 0,
				hourIndex: 0,
				numPoints,
				numHours,
				numMonths
			});

			expect(utciSlice.length).toBe(2);
			// Point 0 (above, in sun) should have higher UTCI than point 1 (below, in shade)
			expect(utciSlice[0]).toBeGreaterThan(utciSlice[1]);
		}
	);
});
