import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { applyCoordinateTransform } from '$lib/utils/coordinates';

describe('coordinate transformation utilities', () => {
	it('should apply xy_ground transformation (Z-up to Y-up)', () => {
		const object = new THREE.Group();
		applyCoordinateTransform(object, 'xy_ground');
		
		expect(object.rotation.x).toBeCloseTo(-Math.PI / 2, 2);
		expect(object.rotation.y).toBe(0);
		expect(object.rotation.z).toBe(0);
	});

	it('should not apply transformation for xz_ground', () => {
		const object = new THREE.Group();
		applyCoordinateTransform(object, 'xz_ground');
		
		expect(object.rotation.x).toBe(0);
		expect(object.rotation.y).toBe(0);
		expect(object.rotation.z).toBe(0);
	});
});


