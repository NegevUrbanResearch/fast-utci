import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
	prepareMeshPayloadForWorkerAsync,
	runMergeAndBvhInWorker,
	MAX_GRID_POINTS_GUARD
} from '$lib/compute/mergeAndBvhWorkerClient';

function createPlane(size = 10): THREE.Group {
	const group = new THREE.Group();
	const geometry = new THREE.PlaneGeometry(size, size);
	geometry.rotateX(-Math.PI / 2);
	const mesh = new THREE.Mesh(geometry);
	group.add(mesh);
	group.updateMatrixWorld(true);
	return group;
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
