import { describe, it, expect, vi, afterEach } from 'vitest';
import * as THREE from 'three';
import {
	prepareMeshPayloadForWorker,
	prepareMeshPayloadForWorkerAsync,
	runMergeAndBvhInWorker,
	MAX_GRID_POINTS_GUARD
} from '$lib/compute/gpu/mergeAndBvhWorkerClient';

function createPlane(size = 10): THREE.Group {
	const group = new THREE.Group();
	const geometry = new THREE.PlaneGeometry(size, size);
	geometry.rotateX(-Math.PI / 2);
	const mesh = new THREE.Mesh(geometry);
	group.add(mesh);
	group.updateMatrixWorld(true);
	return group;
}

function createMesh(geometry: THREE.BufferGeometry, includeInComputeBvh?: boolean): THREE.Mesh {
	const mesh = new THREE.Mesh(geometry);
	if (includeInComputeBvh !== undefined) {
		mesh.userData.includeInComputeBvh = includeInComputeBvh;
	}
	return mesh;
}

describe('mergeAndBvhWorkerClient preflight', () => {
	it('uses the Nes Tziona-tuned grid guard value', () => {
		expect(MAX_GRID_POINTS_GUARD).toBe(600_000);
	});

	it('returns preflight estimate before worker handoff', async () => {
		const group = createPlane(10);
		const result = await prepareMeshPayloadForWorkerAsync(group, {
			gridResolution: 2,
			numHours: 24,
			numMonths: 1
		});
		expect(result.meshes.length).toBeGreaterThan(0);
		expect(result.totalTriangles).toBeGreaterThan(0);
		expect(result.preflight.estimatedBytes).toBeGreaterThan(0);
	});

	it('rejects when estimated bytes exceed budget', async () => {
		const group = createPlane(20);
		await expect(
			prepareMeshPayloadForWorkerAsync(group, {
				gridResolution: 1,
				numHours: 24,
				numMonths: 1,
				maxEstimatedBytes: 1024
			})
		).rejects.toThrow(/exceeds budget/i);
	});

	it('rejects when estimated grid points exceed guard', async () => {
		const group = createPlane(2000);
		await expect(
			prepareMeshPayloadForWorkerAsync(group, {
				gridResolution: 1,
				numHours: 24,
				numMonths: 1
			})
		).rejects.toThrow(`${MAX_GRID_POINTS_GUARD.toLocaleString()}`);
	});

	it('excludes explicit non-compute meshes from async payload triangles while preserving missing metadata behavior', async () => {
		const group = new THREE.Group();
		const building = createMesh(new THREE.BoxGeometry(1, 1, 1), true);
		const ground = createMesh(new THREE.BoxGeometry(10, 0.1, 10), false);
		const legacy = createMesh(new THREE.BoxGeometry(1, 1, 1));
		group.add(building, ground, legacy);

		const result = await prepareMeshPayloadForWorkerAsync(group, {
			gridResolution: 2,
			numHours: 24,
			numMonths: 1
		});

		expect(result.meshes).toHaveLength(2);
		expect(result.totalTriangles).toBe(24);
		expect(result.preflight.meshCount).toBe(2);
		expect(result.preflight.totalTriangles).toBe(24);
	});

	it('excludes explicit non-compute meshes from sync payload triangles', () => {
		const group = new THREE.Group();
		group.add(createMesh(new THREE.BoxGeometry(1, 1, 1), true));
		group.add(createMesh(new THREE.BoxGeometry(10, 0.1, 10), false));

		const result = prepareMeshPayloadForWorker(group);

		expect(result.meshes).toHaveLength(1);
		expect(result.totalTriangles).toBe(12);
	});

	it('keeps preflight grid estimates tied to metadata bounds when filtered ground has the larger extent', async () => {
		const group = new THREE.Group();
		const building = createMesh(new THREE.BoxGeometry(1, 1, 1), true);
		const ground = createMesh(new THREE.BoxGeometry(100, 0.1, 100), false);
		group.add(building, ground);

		const result = await prepareMeshPayloadForWorkerAsync(group, {
			gridResolution: 10,
			numHours: 24,
			numMonths: 1,
			analysisBounds: { x_min: -50, x_max: 50, y_min: -50, y_max: 50, z: 0.9 },
			coordinateSystem: 'xz_ground'
		});

		expect(result.meshes).toHaveLength(1);
		expect(result.totalTriangles).toBe(12);
		expect(result.preflight.estimatedGridPoints).toBe(100);
		expect(result.preflight.bounds).toEqual({
			min: [-50, 0.9, -50],
			max: [50, 0.9, 50]
		});
	});

	it.skipIf(typeof Worker === 'undefined')(
		'when bvhOnly is true, returns serializedBvh and empty gridPoints',
		async () => {
			const group = createPlane(10);
			const { meshes } = await prepareMeshPayloadForWorkerAsync(group, {
				gridResolution: 2,
				numHours: 24,
				numMonths: 1
			});
			const result = await runMergeAndBvhInWorker({
				meshes,
				gridResolution: 2,
				zHeight: 0.9,
				bvhOnly: true
			});
			expect(result.serializedBvh).toBeDefined();
			expect(result.serializedBvh.bvhNodeBuffer).toBeDefined();
			expect(result.serializedBvh.vertexBuffer).toBeDefined();
			expect(result.gridPoints.length).toBe(0);
		}
	);
});


describe('worker lifecycle', () => {
	class FakeWorker {
		static instances: FakeWorker[] = [];
		onmessage: ((event: any) => void) | null = null;
		onerror: ((event: any) => void) | null = null;
		terminate = vi.fn();
		postMessage = vi.fn();
		constructor() { FakeWorker.instances.push(this); }
	}
	const params = { meshes: [], gridResolution: 2, zHeight: 0.9 };
	it('rejects constructor failure without leaving a timer or signal callback', async () => {
		vi.useFakeTimers();
		vi.stubGlobal('Worker', class { constructor() { throw new Error('constructor failure'); } });
		const controller = new AbortController();
		const add = vi.spyOn(controller.signal, 'addEventListener');
		await expect(runMergeAndBvhInWorker({ ...params, signal: controller.signal })).rejects.toThrow('constructor failure');
		expect(add).not.toHaveBeenCalled();
		expect(vi.getTimerCount()).toBe(0);
	});
	it('clones repeated payload buffers into a unique transfer list', async () => {
		vi.stubGlobal('Worker', FakeWorker);
		const mesh = { position: new Float32Array([1, 2, 3]), index: new Uint32Array([0]), matrixWorld: [] };
		const pending = runMergeAndBvhInWorker({ ...params, meshes: [mesh, mesh] });
		const worker = FakeWorker.instances[0];
		const [message, transfers] = worker.postMessage.mock.calls[0] as unknown as [any, ArrayBuffer[]];
		expect(new Set(transfers).size).toBe(4);
		for (const sent of message.meshes) {
			expect(sent.position).toEqual(mesh.position);
			expect(sent.position.buffer).not.toBe(mesh.position.buffer);
			expect(sent.index.buffer).not.toBe(mesh.index.buffer);
		}
		worker.onmessage?.({ data: {} });
		await pending;
	});
	afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); vi.useRealTimers(); FakeWorker.instances = []; });
	it('rejects an already aborted signal without creating a worker', async () => {
		vi.stubGlobal('Worker', FakeWorker);
		const controller = new AbortController(); controller.abort();
		const pending = runMergeAndBvhInWorker({ ...params, signal: controller.signal });
		FakeWorker.instances[0]?.onmessage?.({ data: {} });
		await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
		expect(FakeWorker.instances).toHaveLength(0);
	});
	it.each(['success', 'error', 'workerError', 'abort', 'timeout', 'postFailure', 'payloadFailure'])('cleans callbacks and timers on %s', async (path) => {
		vi.useFakeTimers(); vi.stubGlobal('Worker', FakeWorker);
		const controller = new AbortController();
		const remove = vi.spyOn(controller.signal, 'removeEventListener');
		const payload = path === 'payloadFailure' ? [{ get position() { throw new Error('payload failure'); } }] as any : [];
		if (path === 'postFailure') {
			class ThrowingWorker extends FakeWorker { postMessage = vi.fn(() => { throw new Error('post failure'); }); }
			vi.stubGlobal('Worker', ThrowingWorker);
		}
		const pending = runMergeAndBvhInWorker({ ...params, meshes: payload, signal: controller.signal });
		const settled = pending.catch((error) => error);
		const worker = FakeWorker.instances[0];
		const lateMessage = worker.onmessage;
		const lateError = worker.onerror;
		if (path === 'success') worker.onmessage?.({ data: {} });
		if (path === 'error') worker.onmessage?.({ data: { type: 'error', error: 'failed' } });
		if (path === 'workerError') worker.onerror?.({ message: 'failed' });
		if (path === 'abort') controller.abort();
		if (path === 'timeout') vi.advanceTimersByTime(600_000);
		const outcome = await settled;
		if (path === 'abort') expect(outcome).toMatchObject({ name: 'AbortError' });
		else if (path !== 'success') expect(outcome).toBeInstanceOf(Error);
		expect(remove).toHaveBeenCalledWith('abort', expect.any(Function));
		expect(worker.onmessage).toBeNull(); expect(worker.onerror).toBeNull();
		expect(worker.terminate).toHaveBeenCalledTimes(1);
		expect(vi.getTimerCount()).toBe(0);
		controller.abort(); expect(worker.terminate).toHaveBeenCalledTimes(1);
		lateMessage?.({ data: { type: 'progress', stage: 'merge' } });
		lateError?.({ message: 'late error' });
		expect(worker.terminate).toHaveBeenCalledTimes(1);
	});
});
