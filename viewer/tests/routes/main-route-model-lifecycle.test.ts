import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import { getMainRouteModelLoadedEffects } from '../../src/routes/main/modelLifecycle';

function createBounds() {
	return new THREE.Box3(
		new THREE.Vector3(-1, 0, -2),
		new THREE.Vector3(3, 4, 2)
	);
}

describe('main route model lifecycle policy', () => {
	it('computes first camera fit and marks fit as complete', () => {
		const center = new THREE.Vector3(1, 2, 0);
		const size = new THREE.Vector3(4, 4, 4);
		const bounds = createBounds();
		const boundsBefore = bounds.clone();
		const centerBefore = center.clone();
		const sizeBefore = size.clone();
		const expectedPosition = center
			.clone()
			.add(new THREE.Vector3(0, 4.2, 0.01));

		const result = getMainRouteModelLoadedEffects({
			bounds,
			center,
			size,
			hasFitOnce: false
		});

		expect(result.nextHasFitOnce).toBe(true);
		expect(result.sceneBounds).toBeInstanceOf(THREE.Box3);
		expect(result.sceneBounds.equals(bounds)).toBe(true);
		expect(result.sceneBounds).not.toBe(bounds);
		expect(result.cameraFit).toBeDefined();
		expect(result.cameraFit?.target.equals(center)).toBe(true);
		expect(result.cameraFit?.target).not.toBe(center);
		expect(result.cameraFit?.position.equals(expectedPosition)).toBe(true);
		expect(bounds.equals(boundsBefore)).toBe(true);
		expect(center.equals(centerBefore)).toBe(true);
		expect(size.equals(sizeBefore)).toBe(true);
	});

	it('returns cloned bounds and no camera fit after one has already happened', () => {
		const bounds = createBounds();
		const boundsBefore = bounds.clone();
		const center = new THREE.Vector3(1, 2, 0);
		const size = new THREE.Vector3(4, 4, 4);
		const centerBefore = center.clone();
		const sizeBefore = size.clone();

		const result = getMainRouteModelLoadedEffects({
			bounds,
			center,
			size,
			hasFitOnce: true
		});

		expect(result.nextHasFitOnce).toBe(true);
		expect(result.sceneBounds.equals(bounds)).toBe(true);
		expect(result.sceneBounds).not.toBe(bounds);
		expect(result.cameraFit).toBeUndefined();
		expect(bounds.equals(boundsBefore)).toBe(true);
		expect(center.equals(centerBefore)).toBe(true);
		expect(size.equals(sizeBefore)).toBe(true);
	});
});
