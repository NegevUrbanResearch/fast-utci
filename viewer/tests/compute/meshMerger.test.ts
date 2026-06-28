import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { mergeSceneMeshes } from '$lib/compute/gpu/meshMerger';
import { MAX_TRIANGLES_FOR_MAIN_THREAD } from '$lib/compute/gpu/mergeAndBvhWorkerClient';

describe('mergeSceneMeshes', () => {
	it('should merge multiple mesh children into a single mesh', () => {
		const group = new THREE.Group();
		// Two boxes side by side
		const box1 = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
		box1.position.set(-2, 0, 0);
		box1.updateMatrixWorld(true);
		const box2 = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
		box2.position.set(2, 0, 0);
		box2.updateMatrixWorld(true);
		group.add(box1, box2);
		group.updateMatrixWorld(true);

		const merged = mergeSceneMeshes(group);
		expect(merged).not.toBeNull();

		// Merged mesh should cover both boxes
		merged!.geometry.computeBoundingBox();
		const bbox = merged!.geometry.boundingBox!;
		expect(bbox.min.x).toBeLessThanOrEqual(-1.5);
		expect(bbox.max.x).toBeGreaterThanOrEqual(1.5);
	});

	it('should handle nested groups', () => {
		const root = new THREE.Group();
		const child = new THREE.Group();
		child.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1)));
		root.add(child);
		root.updateMatrixWorld(true);

		const merged = mergeSceneMeshes(root);
		expect(merged).not.toBeNull();
		expect(merged!.geometry.getAttribute('position')).toBeDefined();
	});

	it('should return null for empty group', () => {
		const group = new THREE.Group();
		const merged = mergeSceneMeshes(group);
		expect(merged).toBeNull();
	});

	it('should return null when scene exceeds main-thread triangle safety cap', () => {
		const group = new THREE.Group();
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute(
			'position',
			new THREE.Float32BufferAttribute([0, 0, 0, 1, 0, 0, 0, 1, 0], 3)
		);
		const triCount = MAX_TRIANGLES_FOR_MAIN_THREAD + 1;
		const index = new Uint32Array(triCount * 3);
		for (let i = 0; i < index.length; i += 3) {
			index[i] = 0;
			index[i + 1] = 1;
			index[i + 2] = 2;
		}
		geometry.setIndex(new THREE.BufferAttribute(index, 1));
		group.add(new THREE.Mesh(geometry));

		const merged = mergeSceneMeshes(group);
		expect(merged).toBeNull();
	});
});
