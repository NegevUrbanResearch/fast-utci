import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { calculateModelBounds, calculateModelCenter, calculateModelSize } from '$lib/utils/bounds';

describe('bounds utilities', () => {
	it('should calculate model bounds from group', () => {
		const group = new THREE.Group();
		const geometry = new THREE.BoxGeometry(10, 10, 10);
		const material = new THREE.MeshBasicMaterial();
		const mesh = new THREE.Mesh(geometry, material);
		mesh.position.set(5, 5, 5);
		group.add(mesh);
		
		const bounds = calculateModelBounds(group);
		
		expect(bounds.min.x).toBeLessThan(bounds.max.x);
		expect(bounds.min.y).toBeLessThan(bounds.max.y);
		expect(bounds.min.z).toBeLessThan(bounds.max.z);
	});

	it('should calculate model center', () => {
		const group = new THREE.Group();
		const geometry = new THREE.BoxGeometry(10, 10, 10);
		const material = new THREE.MeshBasicMaterial();
		const mesh = new THREE.Mesh(geometry, material);
		mesh.position.set(5, 5, 5);
		group.add(mesh);
		
		const center = calculateModelCenter(group);
		
		expect(center.x).toBeCloseTo(5, 1);
		expect(center.y).toBeCloseTo(5, 1);
		expect(center.z).toBeCloseTo(5, 1);
	});

	it('should calculate model size', () => {
		const group = new THREE.Group();
		const geometry = new THREE.BoxGeometry(10, 20, 30);
		const material = new THREE.MeshBasicMaterial();
		const mesh = new THREE.Mesh(geometry, material);
		group.add(mesh);
		
		const size = calculateModelSize(group);
		
		expect(size.x).toBeCloseTo(10, 1);
		expect(size.y).toBeCloseTo(20, 1);
		expect(size.z).toBeCloseTo(30, 1);
	});
});


